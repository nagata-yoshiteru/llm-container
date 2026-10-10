#!/usr/bin/env python3
"""Validate profiles/build inputs; --check-images also checks public registries.

Neither mode requires a Docker daemon. Network mode verifies ARM64 and the
vLLM build commit, so an expired nightly tag cannot pass just a syntax check.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".cache/kindling-tp3/upstream"


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True)


for role in ("head", "worker1", "worker2"):
    config = json.loads(run("docker", "compose", "--profile", role, "config", "--format", "json"))
    services = config["services"]
    assert set(services) == {role, "mentat-" + role}, set(services)
    model, daemon = services[role], services["mentat-" + role]
    env, denv = model["environment"], daemon["environment"]
    assert env["VLLM_HOST_IP"] == denv["MENTAT_NODE_IP"]
    assert env["ROLE"] == ("head" if role == "head" else "worker")
    assert env["MENTAT_UNIVERSE"] == denv["MENTAT_UNIVERSE"]
    assert set(model["depends_on"]) == {"mentat-" + role}
    for key, value in dict(TP="3", SPEC_METHOD="mtp", SPEC_TOKENS="3", MOE_BACKEND="marlin",
                           VLLM_DENSE_W4="", VLLM_GLM_SP_MOE_FUSED="0").items():
        assert env[key] == value, (key, env[key])
    sizes = list(map(int, env["CUDAGRAPH_CAPTURE_SIZES"].split()))
    width = int(env["SPEC_TOKENS"]) + 1
    assert all(n % width == 0 for n in sizes)
    assert max(sizes) == int(env["MAX_NUM_SEQS"]) * width
    assert model["image"] == daemon["image"]
    assert model["build"] == daemon["build"]
    assert model["pull_policy"] == daemon["pull_policy"] == "build"
    build = model["build"]
    assert Path(build["context"]).resolve() == SOURCE.resolve()
    assert (Path(build["context"]) / build["dockerfile"]).resolve() == ROOT / "Dockerfile.kindling"
    assert Path(build["additional_contexts"]["local_patches"]).is_dir()
    mounts = {v["target"]: v for v in model["volumes"]}
    assert "/entrypoint.sh" not in mounts, "Do not mask kindling's entrypoint"
    assert mounts[env["MODEL_DIR"]]["read_only"]
    assert not mounts[env["MODEL_DIR"]]["bind"]["create_host_path"]
    assert all(Path(v["source"]).exists() for v in mounts.values())
    # Compose v5 JSON omits zero-valued soft/hard fields, rendering core=0 as {}.
    core = model["ulimits"]["core"]
    assert core == 0 or (isinstance(core, dict) and core.get("soft", 0) == core.get("hard", 0) == 0)
    print(f"PASS: {role} + local daemon, build context, model mounts, TP/MTP settings")

dockerfile = (ROOT / "Dockerfile.kindling").read_text()
original = (SOURCE / "image/Dockerfile").read_text()
original_base = "vllm/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4"
base = "public.ecr.aws/q9t5s3a7/vllm-release-repo@sha256:2352b4a6a8f290967eed33fad5946c94c668479f6fa5816c13d26ef7aa13f889"
# Same official build, retained in public ECR after Docker Hub nightly cleanup.
assert original.count("ARG BASE=" + original_base) == 1
original = original.replace("ARG BASE=" + original_base, "ARG BASE=" + base)
assert dockerfile.split("# --- llm-container:")[0].split("\n", 1)[1] == original + "\n"
for line in dockerfile.splitlines():
    if line.startswith("COPY ") and not line.startswith("COPY --from="):
        for src in line.split()[1:-1]:
            assert (SOURCE / src).exists(), src
for required in ("tp3pad.py", "vocab_parallel_embedding.py", "dense_fp8.py", "modelopt.py",
                 "recoverssm.py", "gb10_sparse_mla.py", "patch-mtp.py", "patch-entrypoint.py"):
    assert required in dockerfile
assert "megamoe_vllm.py" not in dockerfile
print("PASS: pinned build recipe and baked overlay inputs")

# Execute the actual patched final vllm command against a tiny argv recorder.
spec = importlib.util.spec_from_file_location("patch_entrypoint", ROOT / "patches/kindling/patch-entrypoint.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    entry = tmp / "entrypoint.sh"
    entry.write_bytes((SOURCE / "image/entrypoint.sh").read_bytes())
    module.patch(entry)
    subprocess.run(["bash", "-n", str(entry)], check=True)
    recorder = tmp / "vllm"
    recorder.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    recorder.chmod(0o755)
    shell = entry.read_text().split('read -r -a _served_names <<< "$SERVED"', 1)[1]
    shell = 'read -r -a _served_names <<< "$SERVED"' + shell
    env = dict(os.environ, PATH=str(tmp) + os.pathsep + os.environ["PATH"],
               SERVED="glm-5.3-flash-uncensored glm-5.3-flash", MODEL="/models/test", TP="3")
    argv = json.loads(subprocess.check_output(["bash", "-c", shell], env=env, text=True))
    start = argv.index("--served-model-name") + 1
    assert argv[start:start + 3] == ["glm-5.3-flash-uncensored", "glm-5.3-flash", "--tensor-parallel-size"]
    try:
        module.patch(entry)
    except SystemExit:
        pass
    else:
        raise AssertionError("Unexpected source must be rejected")
print("PASS: both existing model aliases reach vLLM as separate arguments; source guard")

if "--check-images" in sys.argv:
    def image_config(ref):
        return json.loads(run("docker", "buildx", "imagetools", "inspect", ref,
                              "--format", "{{json .Image}}"))

    cfg = image_config(base)
    assert (cfg["os"], cfg["architecture"]) == ("linux", "arm64")
    assert cfg["config"]["Labels"]["ai.vllm.build.commit"] == "ddd6fbca148a867aad1fcab7ec72f582b9977db4"
    print("PASS: public ECR base exists, linux/arm64, exact vLLM commit")
    version = next(x.removeprefix("ARG MENTAT_VERSION=") for x in dockerfile.splitlines()
                   if x.startswith("ARG MENTAT_VERSION="))
    mentat = "mmastrac/mentat-artifacts:"
    manifest = json.loads(run("docker", "buildx", "imagetools", "inspect", mentat + version, "--raw"))
    arm = next(x for x in manifest["manifests"] if x["platform"].get("architecture") == "arm64"
               and x["platform"].get("os") == "linux")
    cfg = image_config("mmastrac/mentat-artifacts@" + arm["digest"])
    assert (cfg["os"], cfg["architecture"]) == ("linux", "arm64")
    print(f"PASS: mentat-artifacts:{version} exists, linux/arm64")
