#!/usr/bin/env python3
"""CPU tests for the uncensored-specific MTP hook and BF16 dense layers.

Usage: python test-padding.py UPSTREAM_CHECKOUT PINNED_VLLM_MTP_SOURCE
Requires CPU torch, but does not load checkpoint weights or import vLLM.
"""
import ast
import importlib.util
import logging
from pathlib import Path
import sys
import tempfile
import types

import torch

upstream, mtp_source = map(Path, sys.argv[1:])
stub = types.ModuleType("vllm.logger")
stub.init_logger = logging.getLogger
sys.modules["vllm"] = types.ModuleType("vllm")
sys.modules["vllm.logger"] = stub


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pad = load("pad", upstream / "experimental/tp3/tp3pad.py")
patch = load("patch", Path(__file__).resolve().parents[1] / "patches/kindling/patch-mtp.py")
# Reproduce the already-applied upstream build patch using its literal strings.
constants = {}
tree = ast.parse((upstream / "image/patches/glm53_mtp_bf16.py").read_text())
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
        for target in node.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = node.value.value
source = mtp_source.read_text().replace(constants["OLD"], constants["NEW"])
source = source.replace(constants["HELPER_ANCHOR"], constants["HELPER"] + constants["HELPER_ANCHOR"])
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "mtp.py"
    path.write_text(source)
    patch.patch(path)
    rewritten = path.read_text()
    assert rewritten.index("weights = pad_weights(weights, self.config)") < rewritten.index("for name, loaded_weight in weights:")
    try:
        patch.patch(path)
    except SystemExit:
        pass
    else:
        raise AssertionError("Modified source was not rejected")

config = types.SimpleNamespace(
    tp_pad_orig=dict(num_attention_heads=4, linear_num_heads=4, moe_intermediate_size=32),
    num_attention_heads=6, linear_num_heads=6, moe_intermediate_size=48,
    qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8, linear_head_dim=8, n_shared_experts=1)
prefix = "model.language_model.layers.45."
weights = {
    prefix + "self_attn.q_b_proj.weight": torch.ones(32, 16, dtype=torch.bfloat16),
    prefix + "mlp.experts.0.down_proj.weight": torch.ones(64, 16, dtype=torch.uint8),
    prefix + "mlp.experts.0.down_proj.weight_scale": torch.ones(64, 2).to(torch.float8_e4m3fn),
    "model.language_model.layers.0.mlp.gate_proj.weight": torch.ones(96, 64, dtype=torch.bfloat16),
}
result = dict(pad.pad_weights(weights.items(), config))
for name, expected in ((prefix + "self_attn.q_b_proj.weight", (48, 16)),
                       (prefix + "mlp.experts.0.down_proj.weight", (64, 24)),
                       (prefix + "mlp.experts.0.down_proj.weight_scale", (64, 3))):
    original, padded = weights[name], result[name]
    assert tuple(padded.shape) == expected
    if original.element_size() == 1:
        original, padded = original.view(torch.uint8), padded.view(torch.uint8)
    slices = tuple(slice(0, n) for n in original.shape)
    assert torch.equal(padded[slices], original)
    tail = padded.clone()
    tail[slices] = 0
    assert not tail.any()
name = "model.language_model.layers.0.mlp.gate_proj.weight"
assert result[name] is weights[name], "BF16 dense intermediate is already divisible by three"
again = dict(pad.pad_weights(result.items(), config))
assert all(again[k] is v for k, v in result.items()), "Padding must not run twice"
print("PASS: MTP patch source guard/order, packed FP4 and FP8 padding, BF16 dense passthrough, idempotence")
