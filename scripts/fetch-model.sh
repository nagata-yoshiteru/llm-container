#!/usr/bin/env bash
# Downloads pinned native checkpoint, including Engram and DSpark weights.
# Run separately on each host; no sudo, Docker, conversion or remote writes.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v hf >/dev/null || { echo 'Install huggingface_hub (hf CLI) first.' >&2; exit 1; }
python3 - "$@" <<'PY'
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from deepseek41_config import LOCK, ROOT, read_env, check_model
env = read_env()
pin = LOCK['model']
if env.get('MODEL_REPO') != pin['repository'] or env.get('MODEL_REVISION') != pin['revision']:
    raise SystemExit('MODEL_REPO / MODEL_REVISION must match source-lock.json')
if len(sys.argv) > 2:
    raise SystemExit('Usage: fetch-model.sh [destination]')
destination = Path(sys.argv[1]) if len(sys.argv) == 2 else ROOT / env['MODEL_PATH']
destination.mkdir(parents=True, exist_ok=True)
if (destination / 'tp3-preparation.json').exists():
    raise SystemExit('Refusing a GLM padded serving directory')
if (destination / 'config.json').exists():
    if hashlib.sha256((destination / 'config.json').read_bytes()).hexdigest() != pin['config_sha256']:
        raise SystemExit('Destination contains a different model/config; choose an empty directory')
free = shutil.disk_usage(destination).free / 2**30
print(f'{pin["repository"]}@{pin["revision"]} -> {destination}', flush=True)
print(f'Checkpoint ~475 GiB, local Engram pack ~63 GiB extra; free={free:.1f} GiB', flush=True)
if free < 550:
    print('Note: fresh download + Engram preparation needs about 550 GiB; existing downloads can resume.', flush=True)
subprocess.run(['hf', 'download', pin['repository'], '--revision', pin['revision'],
                '--local-dir', str(destination), '--max-workers', '4'], check=True)
check_model(destination)
print('48 shards and Engram headers validated. Next: prepare-deepseek41-tp3.py --pack <role>')
PY
