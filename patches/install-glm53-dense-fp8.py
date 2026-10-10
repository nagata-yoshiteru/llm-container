#!/usr/bin/env python3
"""Install a guarded post-load hook into the pinned v8 container's writable layer.

The mounted runtime module is imported by every loader process; no sitecustomize
replacement or import-order monkey patch is needed. Does not touch host weights.
"""
import argparse
import hashlib
from pathlib import Path
import sysconfig

ORIGINAL_SHA256 = "a7e925f232ad3eebbee7ab37d3aba724c24465c3078da29489da0438664c6b08"
ANCHOR = "            process_weights_after_loading(model, model_config, target_device)\n"
HOOK = (
    "\n            # Local GLM-5.3 opt-in; run after all native weight finalizers.\n"
    "            from vllm.model_executor.model_loader.glm53_dense_fp8 import convert\n"
    "            convert(model, vllm_config)\n"
)


def install(path):
    raw = path.read_bytes()
    source = raw.decode()
    # Validate both an unmodified file and our exact idempotent installed form.
    original = source.replace(ANCHOR + HOOK, ANCHOR)
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError(f"Unexpected vLLM loader: {path}; requires the pinned sm121-v8 image")
    if original.count(ANCHOR) != 1:
        raise RuntimeError("Expected one post-load hook anchor")
    patched = original.replace(ANCHOR, ANCHOR + HOOK)
    if source not in (original, patched):
        raise RuntimeError("Unexpected existing dense FP8 hook")
    compile(patched, str(path), "exec")
    if source != patched:
        path.write_text(patched)
    print("[dense-fp8] verified v8 post-load hook installed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", type=Path, default=Path(sysconfig.get_path("purelib")) /
                        "vllm/model_executor/model_loader/base_loader.py")
    install(parser.parse_args().loader)
