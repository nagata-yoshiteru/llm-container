#!/usr/bin/env bash
# Local validation; the image's /entrypoint.sh owns kindling launch flags.
set -euo pipefail
[[ ${TP:-} == 3 ]] || { echo 'This configuration requires TP_SIZE=3' >&2; exit 1; }
[[ ${MOE_BACKEND:-} == marlin ]] || { echo 'The uncensored W4A16 checkpoint requires Marlin' >&2; exit 1; }
python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['MODEL_DIR'])
c = json.loads((p / 'config.json').read_text())
t = c['text_config']
assert t.get('tp_pad_orig') == dict(num_attention_heads=64, linear_num_heads=64, moe_intermediate_size=2048), 'Run scripts/prepare-glm53-tp3.py'
assert (t['num_attention_heads'], t['linear_attn_config']['num_heads'], t['moe_intermediate_size']) == (66, 66, 2304)
assert t['vocab_size'] == 154880, 'Do not expand logical vocabulary'
assert (p / 'tp3-preparation.json').is_file(), 'Missing TP=3 preparation manifest'
groups = c['quantization_config']['config_groups']
assert groups and all(g.get('input_activations') is None for g in groups.values()), 'Expected W4A16 checkpoint'
print('GLM53 uncensored: TP=3 padded config / W4A16 validated', flush=True)
PY
exec /entrypoint.sh "$@"
