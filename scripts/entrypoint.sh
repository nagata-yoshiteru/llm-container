#!/usr/bin/env bash
# =============================================================================
# モデルを検証してから vmlx serve を exec する (フォアグラウンド)。
# 通常は scripts/serve.sh up から呼ばれる。デバッグ時は直接叩いてもよい。
#
# 起動フラグはすべてここで組み立てる。値は .env (無ければ env.sh の既定値)。
# DSV4.1 では vmlx 自身が「同時実行 1 / continuous batching on / KV 量子化なし /
# DSpark はバンドル同梱の draft」に揃える (cli._apply_dsv41_runtime_policy)。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

[ -x "${VMLX}" ] || { echo "${VMLX} がありません — ./scripts/setup.sh" >&2; exit 1; }
[ -d "${MODEL_PATH}" ] || { echo "${MODEL_PATH} がありません — ./scripts/fetch-model.sh" >&2; exit 1; }

"${VENV_DIR}/bin/python" -I - "${MODEL_PATH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
c = json.loads((p / 'config.json').read_text())
assert c.get('model_type') == 'deepseek_v41', f"model_type={c.get('model_type')} (deepseek_v41 のはず)"
j = json.loads((p / 'jang_config.json').read_text())
assert j.get('profile', '').startswith('JANG_DSV41'), f"JANG profile={j.get('profile')}"
w = json.loads((p / 'model.safetensors.index.json').read_text())['weight_map']
missing = sorted(s for s in set(w.values()) if not (p / s).is_file())
assert not missing, f'shard 不足 {len(missing)} 件 (例: {missing[:3]}) — ./scripts/fetch-model.sh を再実行'
rt = c.get('runtime', {})
engram = sorted((p / rt.get('engram_dir', 'engram')).glob('*.safetensors'))
assert len(engram) == 2, f'Engram テーブルが {len(engram)} 個 (2 個のはず)'
if rt.get('bundle_has_mtp'):
    assert (p / rt.get('dspark_dir', 'dspark') / 'dspark.safetensors').is_file(), 'DSpark draft がない'
print(f'DSV4.1 uncensored JANG_2K: {len(set(w.values()))} shards + engram {len(engram)} + '
      f'dspark={bool(rt.get("bundle_has_mtp"))} validated', flush=True)
PY

# MLX が wired にできる上限は iogpu.wired_limit_mb。既定 (約 75%) では 197GiB の重みが入らない。
CUR_WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if [ "${CUR_WIRED}" -lt "${WIRED_LIMIT_MB}" ]; then
    echo "iogpu.wired_limit_mb=${CUR_WIRED} (0 は OS 既定)。先に以下を実行してください:" >&2
    echo "  sudo sysctl iogpu.wired_limit_mb=${WIRED_LIMIT_MB}" >&2
    exit 1
fi

ARGS=(
    serve "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${API_HOST}"
    --port "${HOST_PORT}"
    --max-tokens "${MAX_TOKENS}"
    --timeout "${REQUEST_TIMEOUT}"
    --dspark "${DSPARK}"
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --reasoning-parser "${REASONING_PARSER}"
    --default-temperature "${DEFAULT_TEMPERATURE}"
    --default-top-p "${DEFAULT_TOP_P}"
)
[ -n "${API_KEY}" ]                 && ARGS+=(--api-key "${API_KEY}")
[ -n "${DEFAULT_ENABLE_THINKING}" ] && ARGS+=(--default-enable-thinking "${DEFAULT_ENABLE_THINKING}")

# 空白で分割して足す (JSON を渡すなら空白を入れないこと)
if [ -n "${VMLX_EXTRA_ARGS}" ]; then
    read -r -a EXTRA <<< "${VMLX_EXTRA_ARGS}"
    ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})
fi

# 重みはローカルにあるので HF には取りに行かせない。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1

echo "+ vmlx ${ARGS[*]}" | sed -E 's/(--api-key )[^ ]+/\1***/'
exec "${VMLX}" "${ARGS[@]}"
