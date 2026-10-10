# SPDX-License-Identifier: Apache-2.0
"""Opt-in post-load FP8 dense linears for the pinned GLM-5.3 v8 image.

Based on the per-channel weight / per-token activation CUTLASS path in
kindlingai/glm-5.3-flash-gx10, experimental/snapshot/dense_fp8.py,
commit c748079d45e6e070b2acb108a91edfe52f4a7747 (Apache-2.0).
Restricted here to TP=1 GLM target projections. Experts, router, indexer,
kv_b_proj, embedding, lm_head, vision and the MTP draft stay unchanged.
This is lossy W8A8 quantization, not the upstream "lossless8" checkpoint.
"""
import os
import re

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase, LinearMethodBase, UnquantizedLinearMethod,
)

logger = init_logger(__name__)
_TARGET = re.compile(
    r"(?:^|\.)layers\.(\d+)\."
    r"(self_attn\.(?:in_proj_qkvbfg_a|fused_qkv_a_proj|q_b_proj|o_proj)"
    r"|mlp\.(?:shared_experts\.)?(?:gate_up_proj|down_proj))$"
)
_ROWS = 2048


class DenseFP8Method(LinearMethodBase):
    def create_weights(self, *args, **kwargs):
        raise RuntimeError("Dense FP8 is installed only after BF16 loading completes")

    def apply(self, layer, x, bias=None):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        n = layer.weight.shape[0]
        if not x2.shape[0]:
            return x.new_empty((*shape[:-1], n))
        xq, xs = ops.scaled_fp8_quant(x2, use_per_token_if_dynamic=True)
        if x2.shape[0] <= _ROWS:
            out = ops.cutlass_scaled_mm(
                xq, layer.weight.t(), xs, layer.weight_scale, x.dtype, bias,
            )
        else:
            # Bound large-prefill GEMMs; no extra copy of the dense weight.
            out = torch.empty((x2.shape[0], n), dtype=x.dtype, device=x.device)
            for start in range(0, x2.shape[0], _ROWS):
                torch.ops._C.cutlass_scaled_mm(
                    out[start:start + _ROWS], xq[start:start + _ROWS],
                    layer.weight.t(), xs[start:start + _ROWS], layer.weight_scale, bias,
                )
        return out.reshape(*shape[:-1], n)


@torch.no_grad()
def quantize_layer(layer):
    """Replace one BF16 weight after loading; return the net saved bytes."""
    w = layer.weight
    q, scale = ops.scaled_fp8_quant(w.contiguous(), use_per_token_if_dynamic=True)
    # Weight rows are output channels; scaled_mm expects scales along N.
    scale = scale.reshape(1, -1).float().contiguous()
    layer.weight = Parameter(q, requires_grad=False)
    layer.weight_scale = Parameter(scale, requires_grad=False)
    layer.quant_method = DenseFP8Method()
    if hasattr(layer, "_use_min_latency_gemm"):
        layer._use_min_latency_gemm = False
    return w.numel() * w.element_size() - q.numel() - scale.numel() * 4


@torch.no_grad()
def kernel_self_test():
    """Check actual v8 CUDA ops, zero rows, prefill, and graph replay."""
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("GLM53_DENSE_FP8 currently supports GB10 / SM121 only")
    if not ops.cutlass_scaled_mm_supports_fp8(121):
        raise RuntimeError("This image does not provide CUTLASS FP8 on SM121")
    gen = torch.Generator(device="cuda").manual_seed(53)
    w = torch.randn(128, 256, generator=gen, device="cuda", dtype=torch.bfloat16)
    w[0].zero_()
    layer = torch.nn.Module()
    layer.weight = Parameter(w, requires_grad=False)
    quantize_layer(layer)
    worst = 0.0
    for rows in (1, 2, 4, 17, 2051):
        x = torch.randn(rows, 256, generator=gen, device="cuda", dtype=torch.bfloat16)
        if rows > 1:
            x[0].zero_()
        bias = torch.linspace(-1, 1, 128, device="cuda", dtype=torch.bfloat16)
        y = layer.quant_method.apply(layer, x, bias)
        reference = x.float() @ w.float().t() + bias.float()
        error = (y.float() - reference).norm() / reference.norm().clamp_min(1e-9)
        if not torch.isfinite(y).all() or error.item() > 0.08:
            raise RuntimeError(f"Dense FP8 kernel check failed: rows={rows}, error={error.item()}")
        worst = max(worst, error.item())
    # Decode uses FULL_DECODE_ONLY CUDA graphs. Exercise the same apply path.
    x = torch.ones(2, 256, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer.quant_method.apply(layer, x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer.quant_method.apply(layer, x)
    x.fill_(0.5)
    graph.replay()
    expected = layer.quant_method.apply(layer, x)
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    logger.info("GLM53 dense FP8 kernel/graph check passed (max relative L2 %.4f)", worst)


def expected_targets(config):
    """Every eligible target projection must exist; no silent partial conversion."""
    if config.num_hidden_layers != 45 or len(config.layer_types) != 45 or len(config.mlp_layer_types) != 45:
        raise RuntimeError("Dense FP8 requires the 45-layer GLM-5.3-Flash config")
    targets = set()
    for i, (attention, mlp) in enumerate(zip(config.layer_types, config.mlp_layer_types)):
        if attention == "linear_attention":
            projections = ("in_proj_qkvbfg_a", "o_proj")
        elif attention == "deepseek_sparse_attention":
            projections = ("fused_qkv_a_proj", "q_b_proj", "o_proj")
        else:
            raise RuntimeError(f"Unsupported attention: {attention}")
        targets.update((i, "self_attn." + name) for name in projections)
        if mlp not in ("dense", "sparse"):
            raise RuntimeError(f"Unsupported MLP: {mlp}")
        prefix = "mlp.shared_experts." if mlp == "sparse" else "mlp."
        targets.update((i, prefix + name) for name in ("gate_up_proj", "down_proj"))
    return targets


@torch.no_grad()
def convert(model, vllm_config):
    if os.environ.get("GLM53_DENSE_FP8", "0") != "1":
        return
    if getattr(model, "_glm53_dense_fp8_done", False):
        return
    cls = type(model)
    if cls.__module__ == "vllm.models.glm5next.nvidia.mtp" and cls.__name__ == "Glm5NextMTP":
        logger.info("GLM53 dense FP8: preserving the MTP draft in its original precision")
        return
    if cls.__module__ != "vllm.models.glm5next.nvidia.model" or cls.__name__ not in (
        "Glm5NextForCausalLM", "Glm5NextForConditionalGeneration",
    ):
        raise RuntimeError(f"Dense FP8 is not supported for {cls.__module__}.{cls.__name__}")
    if vllm_config.parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("Dense FP8 overlay requires TP=1")
    expected = expected_targets(vllm_config.model_config.hf_text_config)
    found = set()
    layers = []
    for name, layer in model.named_modules():
        match = _TARGET.search(name)
        if match is None:
            continue
        key = (int(match[1]), match[2])
        if key not in expected or key in found:
            raise RuntimeError(f"Unexpected/duplicate dense projection: {name}")
        w = getattr(layer, "weight", None)
        if not (isinstance(layer, LinearBase)
                and type(layer.quant_method) is UnquantizedLinearMethod
                and isinstance(w, torch.Tensor) and w.dtype == torch.bfloat16
                and w.is_cuda and w.ndim == 2 and all(d % 16 == 0 for d in w.shape)):
            raise RuntimeError(f"Unsupported BF16 dense layer: {name}")
        layers.append(layer)
        found.add(key)
    if found != expected:
        raise RuntimeError(f"Missing dense FP8 projections: {sorted(expected - found)}")
    kernel_self_test()
    torch.cuda.empty_cache()
    saved = 0
    for layer in layers:
        saved += quantize_layer(layer)
        # Free replaced weights promptly on GB10's shared CPU/GPU memory.
        torch.cuda.empty_cache()
    model._glm53_dense_fp8_done = True
    logger.info("GLM53 dense FP8: converted %d target linears, saved %.3f GiB/rank; "
                "experts/router/indexer/kv_b/embedding/lm_head/MTP unchanged",
                len(layers), saved / 2**30)


if __name__ == "__main__":
    kernel_self_test()
