# shellcheck shell=bash
# =============================================================================
# 全スクリプト共通の設定読み込み。source して使う (単体では実行しない)。
#
# .env は docker compose 時代と同じ「KEY=VALUE、クォートなし」の書式。
# source はせず、必要なキーだけリテラルに読む。.env に無いキーは下の既定値。
# macOS の /bin/bash は 3.2 なので、連想配列や ${x,,} は使わない。
# =============================================================================
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# env_get KEY [既定値]   .env にあればその値 (空文字も「指定あり」として扱う)
env_get() {
    if [ -f "${REPO_ROOT}/.env" ] && grep -q "^$1=" "${REPO_ROOT}/.env"; then
        sed -n "s/^$1=//p" "${REPO_ROOT}/.env" | tail -1
    else
        printf '%s' "${2:-}"
    fi
}

# 相対パスは repo ルート基準にする (どこから叩いても同じ場所を指すように)
abs_path() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s/%s' "${REPO_ROOT}" "${1#./}" ;;
    esac
}

load_env() {
    # --- 重み (MLX / Solstice-AI が dealignai の FP8 版から変換した oQ4e) ---
    MODEL_REPO=$(env_get MODEL_REPO Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2)
    MODEL_REVISION=$(env_get MODEL_REVISION bff40d24efb689d33e094083ce4a5d6835601730)
    MODEL_PATH=$(abs_path "$(env_get MODEL_PATH ./models/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e)")
    # chat template は Solstice 独自のもの (独自 system prompt を常に注入) ではなく
    # 元の dealignai 版を使う。fetch-model.sh が MODEL_PATH/upstream-dealignai/ に取得する。
    TEMPLATE_REPO=$(env_get TEMPLATE_REPO dealignai/GLM-5.3-Flash-UNCENSORED-FP8)
    TEMPLATE_REVISION=$(env_get TEMPLATE_REVISION d5822b068da8ddb301665b48c58718e6e18a2c3a)
    # 実際に serve するのは prepare-model.sh が作るこちら (shard は MODEL_PATH への hardlink)
    SERVE_MODEL_PATH=$(abs_path "$(env_get SERVE_MODEL_PATH "${MODEL_PATH}-mlxvlm")")

    # --- ランタイム ---
    VENV_DIR=$(abs_path "$(env_get VENV_DIR ./venv)")
    RAPID_MLX_VERSION=$(env_get RAPID_MLX_VERSION 1.0.2)
    RUNTIME_DIR=$(abs_path "$(env_get RUNTIME_DIR ./.cache/rapid-mlx)")

    # --- API ---
    SERVED_MODEL_NAME=$(env_get SERVED_MODEL_NAME glm-5.3-flash-uncensored)
    API_HOST=$(env_get API_HOST 127.0.0.1)
    HOST_PORT=$(env_get HOST_PORT 8910)
    API_KEY=$(env_get API_KEY "")

    # --- サービング ---
    MAX_MODEL_LEN=$(env_get MAX_MODEL_LEN 262144)
    MAX_NUM_SEQS=$(env_get MAX_NUM_SEQS 4)
    PREFILL_STEP_SIZE=$(env_get PREFILL_STEP_SIZE 1024)
    KV_CACHE_DTYPE=$(env_get KV_CACHE_DTYPE "")
    CACHE_MEMORY_MB=$(env_get CACHE_MEMORY_MB "")
    SPEC_DECODE=$(env_get SPEC_DECODE 0)
    DEFAULT_REASONING_EFFORT=$(env_get DEFAULT_REASONING_EFFORT "")
    TOOL_CALL_PARSER=$(env_get TOOL_CALL_PARSER glm47)
    REASONING_PARSER=$(env_get REASONING_PARSER glm5)
    RAPID_MLX_EXTRA_ARGS=$(env_get RAPID_MLX_EXTRA_ARGS "")

    # --- macOS の GPU wired メモリ上限 (MiB)。sudo sysctl で設定する ---
    WIRED_LIMIT_MB=$(env_get WIRED_LIMIT_MB 245760)

    RAPID_MLX="${VENV_DIR}/bin/rapid-mlx"
    HF="${VENV_DIR}/bin/hf"
    LOG_DIR="${RUNTIME_DIR}/logs"
    LOG_FILE="${LOG_DIR}/serve.log"
    PID_FILE="${RUNTIME_DIR}/serve.pid"
}

load_env
