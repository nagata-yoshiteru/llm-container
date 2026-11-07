#!/usr/bin/env bash
# =============================================================================
# モデルを検証してから rapid-mlx serve を exec する (フォアグラウンド)。
# 通常は scripts/serve.sh up から呼ばれる。デバッグ時は直接叩いてもよい。
#
# 起動フラグはすべてここで組み立てる。値は .env (無ければ env.sh の既定値)。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

[ -x "${RAPID_MLX}" ] || { echo "${RAPID_MLX} がありません — ./scripts/setup.sh" >&2; exit 1; }
[ -d "${SERVE_MODEL_PATH}" ] || { echo "${SERVE_MODEL_PATH} がありません — ./scripts/prepare-model.sh" >&2; exit 1; }

"${VENV_DIR}/bin/python" -I - "${SERVE_MODEL_PATH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
assert (p / 'mlxvlm-preparation.json').is_file(), 'mlx-vlm 用の準備が未完了 — ./scripts/prepare-model.sh'
c = json.loads((p / 'config.json').read_text())
assert c.get('model_type') == 'glm5_next', f"model_type={c.get('model_type')} (glm5_next のはず)"
q = c.get('quantization') or c.get('quantization_config') or {}
assert (q.get('bits'), q.get('group_size')) == (4, 64), f'MLX 4bit/g64 のはず: {q.get("bits")}/{q.get("group_size")}'
w = json.loads((p / 'model.safetensors.index.json').read_text())['weight_map']
missing = sorted(s for s in set(w.values()) if not (p / s).is_file())
assert not missing, f'shard 不足 {len(missing)} 件 (例: {missing[:3]}) — ./scripts/fetch-model.sh を再実行'
print(f'GLM53 uncensored: MLX oQ4e / {len(set(w.values()))} shards validated', flush=True)
PY

# rapid-mlx は max_recommended_working_set_size (= iogpu.wired_limit_mb) までしか
# wired にしない。既定 (約 75%) のままだと 186GB の重みが入りきらない。
CUR_WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if [ "${CUR_WIRED}" -lt "${WIRED_LIMIT_MB}" ]; then
    echo "iogpu.wired_limit_mb=${CUR_WIRED} (0 は OS 既定)。先に以下を実行してください:" >&2
    echo "  sudo sysctl iogpu.wired_limit_mb=${WIRED_LIMIT_MB}" >&2
    exit 1
fi

ARGS=(
    serve "${SERVE_MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${API_HOST}"
    --port "${HOST_PORT}"
    --context-length "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --prefill-step-size "${PREFILL_STEP_SIZE}"
    # ローカルパスは名前から parser を推定できないことがあるので明示する
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --reasoning-parser "${REASONING_PARSER}"
    --disable-model-downloads
)
[ -n "${API_KEY}" ]                  && ARGS+=(--api-key "${API_KEY}")
[ -n "${KV_CACHE_DTYPE}" ]           && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[ -n "${CACHE_MEMORY_MB}" ]          && ARGS+=(--cache-memory-mb "${CACHE_MEMORY_MB}")
[ -n "${DEFAULT_REASONING_EFFORT}" ] && ARGS+=(--default-reasoning-effort "${DEFAULT_REASONING_EFFORT}")
[ "${SPEC_DECODE}" = "1" ]           || ARGS+=(--no-spec-decode)

# 空白で分割して足す (JSON を渡すなら空白を入れないこと)
if [ -n "${RAPID_MLX_EXTRA_ARGS}" ]; then
    read -r -a EXTRA <<< "${RAPID_MLX_EXTRA_ARGS}"
    ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})
fi

# 重みはローカルにあるので HF には一切取りに行かせない。
# 外部送信 (利用統計・更新チェック) も止める。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1 RAPID_MLX_DISABLE_VERSION_CHECK=1

echo "+ rapid-mlx ${ARGS[*]}" | sed -E 's/(--api-key )[^ ]+/\1***/'
exec "${RAPID_MLX}" "${ARGS[@]}"
