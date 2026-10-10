#!/usr/bin/env python3
"""CPU/stdlib-only preparation; never starts Docker or changes the active .env."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

COMMIT = "c748079d45e6e070b2acb108a91edfe52f4a7747"
URL = "https://github.com/kindlingai/glm-5.3-flash-gx10.git"
MODEL = "GLM-5.3-Flash-UNCENSORED-NVFP4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_model(src, dst):
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["text_config"]
    expected = dict(hidden_size=4096, num_hidden_layers=45, num_attention_heads=64,
                    moe_intermediate_size=2048, intermediate_size=12288,
                    num_nextn_predict_layers=1, vocab_size=154880)
    if any(tc.get(k) != v for k, v in expected.items()) or "tp_pad_orig" in tc:
        raise ValueError("Unexpected source architecture or already padded config")
    if tc["linear_attn_config"]["num_heads"] != 64:
        raise ValueError("Unexpected KDA heads")
    if cfg["quantization_config"].get("quant_algo") != "NVFP4":
        raise ValueError("Expected ModelOpt NVFP4")
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    # Read only small safetensors headers. No Torch, GPU allocation or full tensor reads.
    headers = {}
    for shard in sorted(set(index.values())):
        path = (src / shard).resolve()
        if path.parent != src.resolve():
            raise ValueError("Shard outside source model directory")
        with path.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if not 0 < size < 64 * 1024 * 1024:
                raise ValueError("Invalid safetensors header")
            header = json.loads(f.read(size))
        headers.update({k: v for k, v in header.items() if k != "__metadata__"})
    if not set(index).issubset(headers):
        raise ValueError("Checkpoint index references missing tensors")
    prefix = "model.language_model.layers."
    for layer in range(3):
        for proj, shape in (("gate_proj", [12288, 4096]), ("up_proj", [12288, 4096]),
                            ("down_proj", [4096, 12288])):
            t = headers[f"{prefix}{layer}.mlp.{proj}.weight"]
            if t["dtype"] != "BF16" or t["shape"] != shape:
                raise ValueError("Unexpected uncensored dense MLP format")
    for layer in (3, 45):
        for proj, shape in (("gate_proj", [2048, 2048]), ("down_proj", [4096, 1024])):
            t = headers[f"{prefix}{layer}.mlp.experts.0.{proj}.weight"]
            if t["dtype"] != "U8" or t["shape"] != shape:
                raise ValueError("Expected packed NVFP4 target and MTP experts")
    manifest = {"source": str(src.resolve()), "config_sha256": sha(src / "config.json"),
                "index_sha256": sha(src / "model.safetensors.index.json"),
                "kindling_commit": COMMIT, "shards": len(set(index.values())),
                "tensor_count": len(index), "format": "uncensored-tp3-v1"}
    tc["tp_pad_orig"] = {"num_attention_heads": 64, "linear_num_heads": 64,
                         "moe_intermediate_size": 2048}
    tc.update(num_attention_heads=66, num_key_value_heads=66, linear_num_heads=66,
              moe_intermediate_size=2304)
    tc["linear_attn_config"]["num_heads"] = 66
    # Vocab stays 154880. Upstream's embedding overlay pads storage to 154944.
    encoded = json.dumps(cfg, indent=2) + "\n"
    if dst.exists():
        if json.loads((dst / "tp3-preparation.json").read_text()) != manifest:
            raise ValueError("Existing destination has different provenance")
        if (dst / "config.json").read_text() != encoded:
            raise ValueError("Existing padded config was modified")
        for path in src.iterdir():
            if path.is_file() and path.name != "config.json":
                if not os.path.samefile(path, dst / path.name):
                    raise ValueError(f"Hardlink changed: {path.name}")
        return manifest
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tp3-prepare-", dir=dst.parent))
    try:
        for path in src.iterdir():
            if path.is_file() and path.name != "config.json":
                os.link(path, tmp / path.name)
        (tmp / "config.json").write_text(encoded)
        (tmp / "tp3-preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.rename(dst)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    source = root / ".cache/kindling-tp3/upstream"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", COMMIT], check=True)
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise SystemExit("Unexpected kindling checkout; refusing to modify it")
    subprocess.run(["git", "-C", str(source), "submodule", "update", "--init", "--recursive"], check=True)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Modified kindling source; refusing to continue")
    env = {}
    if (root / ".env").exists():
        for line in (root / ".env").read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip().strip("\"'")
    src = root / env.get("SOURCE_MODEL_PATH", "models/" + MODEL)
    dest = root / env.get("MODEL_PATH", "models/" + MODEL + "-kindling-tp3")
    if src.resolve() == dest.resolve():
        raise SystemExit("MODEL_PATH must be the TP=3 serving directory, separate from SOURCE_MODEL_PATH")
    manifest = prepare_model(src, dest)
    for subdir in ("cache", "logs"):
        (root / ".cache/kindling-tp3/runtime" / subdir).mkdir(parents=True, exist_ok=True)
    print(json.dumps({**manifest, "destination": str(dest)}, indent=2))


if __name__ == "__main__":
    main()
