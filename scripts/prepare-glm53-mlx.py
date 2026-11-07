#!/usr/bin/env python3
"""Solstice oQ4e (mixed-bit MLX) を mlx-vlm の glm5_next ローダーで読める形にする。

    venv/bin/python -I scripts/prepare-glm53-mlx.py <SOURCE> <SERVE>

通常は scripts/prepare-model.sh 経由で呼ぶ。sudo 不要、Metal も使わない (CPU)。

なぜ必要か (2026-11-07 実機で確認):
  1. config の per-module 量子化キーが checkpoint 名 (model.language_model.layers.N.*)
     のままで、mlx-vlm が引くモジュールパス (language_model.model.layers.N.*) と
     一致しない。全モジュールが既定の 4bit 扱いになり、6/8bit の tensor と形が合わない。
  2. mlx-vlm の sanitize は q/k/v -> qkv_proj、f_a/b/g_a -> fbg_a_proj、
     q_a/kv_a -> qkv_a_proj、gate/up -> gate_up_proj を連結する。oQ4e は同じ層の
     中でも q=6bit / k=8bit のように混在しており、連結できない
     ("[concatenate] ... (8192,768), (8192,1024), (8192,768)")。
     連結後の名前や kv_b_proj から作られる embed_q / unembed_out にも指定が無い。
  3. 元の dealignai 版で float32 の hc_*_base / hc_*_scale / A_log / dt_bias /
     e_score_correction_bias が BF16 で保存されている。mlx-vlm の HyperConnection の
     Metal カーネル (hc_sinkhorn_collapse) は base を float4 として読むので、BF16 だと
     Sinkhorn の混合行列が壊れ、GPU だけ出力が文字化けする (CPU は正しい)。
  4. chat template が Solstice 独自のもので、独自の system prompt を常に注入する。

やること:
  - SOURCE の全ファイルを SERVE へ hardlink (巨大な複製はしない)
  - 連結グループ内でビット数が低いものを高い方へ再量子化 (6bit -> 8bit。精度は落ちない方向)
  - 3. の tensor を float32 に戻す (BF16 からの upcast なので値は BF16 精度のまま)
  - 上の 2 つを追加 shard (model-zz-mlxvlm-fixup.safetensors) に書き、index の同じキーを
    この shard に向ける。mlx-vlm は index の shard を名前順に読み、後勝ちで上書きする。
  - config.json / index / tokenizer_config.json / chat_template.jinja は新しいファイルとして
    書く (hardlink 先を書き換えない)。chat template は SOURCE/upstream-dealignai/ の元版。
  - 連結後の名前・モジュールパス名の量子化指定を config に足す
SOURCE の shard は一切変更しない。同じ SERVE への再実行は検証だけ行う。
"""
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

REQUANT_SHARD = "model-zz-mlxvlm-fixup.safetensors"
MANIFEST = "mlxvlm-preparation.json"
TEMPLATE = Path("upstream-dealignai") / "chat_template.jinja"
REWRITTEN = ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "chat_template.jinja")
FP32_SUFFIXES = (
    "hc_attn_base", "hc_attn_scale", "hc_ffn_base", "hc_ffn_scale",
    "self_attn.A_log", "self_attn.dt_bias", "mlp.gate.e_score_correction_bias",
)
CK = "model.language_model."   # checkpoint / config のキー
MP = "language_model.model."   # mlx-vlm のモジュールパス


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups_for_layer(i: int, layer_type: str, has):
    """(連結後の名前, [元の名前...]) を返す。名前は 'layers.N.' 以降。"""
    out = []
    if layer_type == "linear_attention":
        out.append(("self_attn.qkv_proj", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]))
        out.append(("self_attn.fbg_a_proj", ["self_attn.f_a_proj", "self_attn.b_proj", "self_attn.g_a_proj"]))
    else:
        out.append(("self_attn.qkv_a_proj", ["self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"]))
    for mlp in ("mlp", "mlp.shared_experts"):
        if has(i, f"{mlp}.gate_proj"):
            out.append((f"{mlp}.gate_up_proj", [f"{mlp}.gate_proj", f"{mlp}.up_proj"]))
    return out


def main(src: Path, dst: Path) -> None:
    config = json.loads((src / "config.json").read_text())
    index = json.loads((src / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    quant = config["quantization"]
    default = {k: quant[k] for k in ("group_size", "bits", "mode")}
    text = config["text_config"]
    n_layers = text["num_hidden_layers"]
    layer_types = text["layer_types"]
    assert config["model_type"] == "glm5_next" and default["mode"] == "affine"

    def has(i, name):
        return f"{CK}layers.{i}.{name}.weight" in wmap

    def quantized(i, name):
        return f"{CK}layers.{i}.{name}.scales" in wmap

    def qcfg(i, name):
        return {**default, **quant.get(f"{CK}layers.{i}.{name}", {})}

    new_quant = dict(quant)
    # 1. checkpoint 名の指定を、mlx-vlm のモジュールパス名にも複製する
    for key, value in quant.items():
        if isinstance(value, dict) and key.startswith(CK):
            new_quant[MP + key[len(CK):]] = value

    # 2. 連結グループのビット数をそろえる
    plan = []  # (layer, member, from_bits, to_bits)
    for i in range(n_layers):
        for fused, members in groups_for_layer(i, layer_types[i], has):
            members = [m for m in members if has(i, m)]
            if not members:
                continue
            flags = {quantized(i, m) for m in members}
            assert len(flags) == 1, f"layer {i} {fused}: 量子化あり/なしが混在"
            if not flags.pop():
                continue
            cfgs = [qcfg(i, m) for m in members]
            assert {c["group_size"] for c in cfgs} == {default["group_size"]}
            target = max(c["bits"] for c in cfgs)
            tcfg = {**default, "bits": target}
            for m, c in zip(members, cfgs):
                if c["bits"] != target:
                    plan.append((i, m, c["bits"], target))
                new_quant[f"{CK}layers.{i}.{m}"] = tcfg
                new_quant[f"{MP}layers.{i}.{m}"] = tcfg
            new_quant[f"{MP}layers.{i}.{fused}"] = tcfg
        # kv_b_proj は sanitize で embed_q / unembed_out に分割・同じビットで再量子化される
        if layer_types[i] != "linear_attention" and quantized(i, "self_attn.kv_b_proj"):
            kv_b = qcfg(i, "self_attn.kv_b_proj")
            for name in ("self_attn.embed_q", "self_attn.unembed_out"):
                new_quant[f"{MP}layers.{i}.{name}"] = kv_b
    # lm_head / embed_tokens / vision はモジュールパスの別名で既に一致する

    new_config = dict(config, quantization=new_quant, quantization_config=new_quant)
    requant_keys = sorted(
        f"{CK}layers.{i}.{m}.{s}" for i, m, _, _ in plan for s in ("weight", "scales", "biases")
    )
    fp32_keys = sorted(k for k in wmap if k.startswith(CK) and k.endswith(FP32_SUFFIXES))
    new_wmap = dict(wmap)
    for k in requant_keys + fp32_keys:
        new_wmap[k] = REQUANT_SHARD
    new_index = dict(index, weight_map=new_wmap)

    # chat template は元の dealignai 版。tokenizer_config の埋め込み版は消す
    template = (src / TEMPLATE).read_text()
    assert "Protium" not in template and "[gMASK]" in template, f"{src / TEMPLATE} が元版ではない"
    tok_cfg = json.loads((src / "tokenizer_config.json").read_text())
    tok_cfg.pop("chat_template", None)

    manifest = {
        "source": str(src),
        "source_config_sha256": sha256(src / "config.json"),
        "source_index_sha256": sha256(src / "model.safetensors.index.json"),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "mlx_vlm": version("mlx-vlm"),
        "rapid_mlx": version("rapid-mlx"),
        "requantized": [f"layers.{i}.{m}: {a}->{b}bit" for i, m, a, b in plan],
        "fp32_restored": len(fp32_keys),
    }

    # 再実行: 同じ内容なら検証だけ
    if (dst / MANIFEST).exists():
        old = json.loads((dst / MANIFEST).read_text())
        same = {k: old.get(k) for k in manifest} == manifest
        assert same, f"{dst} は別の条件で作られている。消してから再実行すること"
        assert json.loads((dst / "config.json").read_text()) == new_config
        assert json.loads((dst / "model.safetensors.index.json").read_text()) == new_index
        assert (dst / "chat_template.jinja").read_text() == template
        assert (dst / REQUANT_SHARD).is_file()
        print(f"{dst}: 準備済み (再量子化 {len(plan)} 個 / float32 復元 {len(fp32_keys)} 個)")
        return
    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f"{dst} が空ではなく manifest も無い。手で作ったものなら退避すること")

    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.name not in REWRITTEN:
            os.link(f, dst / f.name)

    # 再量子化と float32 復元 (shard ごとにまとめて読む)
    by_shard = {}
    for i, m, a, b in plan:
        by_shard.setdefault(wmap[f"{CK}layers.{i}.{m}.weight"], []).append(("requant", (i, m, a, b)))
    for k in fp32_keys:
        by_shard.setdefault(wmap[k], []).append(("fp32", k))
    out = {}
    for shard, items in sorted(by_shard.items()):
        tensors = mx.load(str(src / shard))
        for kind, item in items:
            if kind == "fp32":
                out[item] = tensors[item].astype(mx.float32)
                continue
            i, m, a, b = item
            p = f"{CK}layers.{i}.{m}"
            w, s, z = tensors[f"{p}.weight"], tensors[f"{p}.scales"], tensors[f"{p}.biases"]
            gs = default["group_size"]
            full = mx.dequantize(w, s, z, group_size=gs, bits=a, mode="affine")
            w2, s2, z2 = mx.quantize(full.astype(mx.float32), group_size=gs, bits=b, mode="affine")
            assert w2.shape[-1] * 32 // b == w.shape[-1] * 32 // a, f"{p}: 入力次元が変わった"
            out[f"{p}.weight"], out[f"{p}.scales"], out[f"{p}.biases"] = w2, s2.astype(s.dtype), z2.astype(z.dtype)
        mx.eval(list(out.values()))
        del tensors
        print(f"  {shard}: {len(items)} 個", flush=True)
    mx.save_safetensors(str(dst / REQUANT_SHARD), out, metadata={"format": "mlx"})

    (dst / "config.json").write_text(json.dumps(new_config, indent=2, ensure_ascii=False) + "\n")
    (dst / "model.safetensors.index.json").write_text(json.dumps(new_index, indent=2) + "\n")
    (dst / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")
    (dst / "chat_template.jinja").write_text(template)
    (dst / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"{dst}: 作成 (再量子化 {len(plan)} 個 / float32 復元 {len(fp32_keys)} 個、"
          f"{REQUANT_SHARD} {sum(v.nbytes for v in out.values()) / 2**20:.0f} MiB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
