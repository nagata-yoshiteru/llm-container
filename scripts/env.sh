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
    # --- 重み (dealignai の uncensored 版、JANG_2K = JANG 形式の MLX 混合 2〜8bit) ---
    MODEL_REPO=$(env_get MODEL_REPO dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-JANG_2K)
    MODEL_REVISION=$(env_get MODEL_REVISION f914c4b17429864d9cfeeebabdfaf15aa400d3d6)
    MODEL_PATH=$(abs_path "$(env_get MODEL_PATH ./models/DeepSeek-V4.1-Flash-1017-UNCENSORED-JANG_2K)")

    # --- ランタイム (vmlx + jang_tools。Rapid-MLX とは依存がぶつかるので venv を分ける) ---
    VENV_DIR=$(abs_path "$(env_get VENV_DIR ./venv-vmlx)")
    VMLX_VERSION=$(env_get VMLX_VERSION 1.6.84)
    JANG_VERSION=$(env_get JANG_VERSION 2.5.53)
    # vmlx 1.6.84 は mlx-lm 0.32 系で ImportError になる (SequenceStateMachine が無い)
    MLX_LM_VERSION=$(env_get MLX_LM_VERSION 0.31.3)
    RUNTIME_DIR=$(abs_path "$(env_get RUNTIME_DIR ./.cache/vmlx)")

    # --- API ---
    SERVED_MODEL_NAME=$(env_get SERVED_MODEL_NAME deepseek-v4.1-flash-uncensored)
    API_HOST=$(env_get API_HOST 127.0.0.1)
    HOST_PORT=$(env_get HOST_PORT 8910)
    API_KEY=$(env_get API_KEY "")

    # --- サービング ---
    # 同時実行は vmlx が DSV4.1 では常に 1 に固定する (cli._apply_dsv41_runtime_policy)。
    MAX_TOKENS=$(env_get MAX_TOKENS 32768)
    REQUEST_TIMEOUT=$(env_get REQUEST_TIMEOUT 3600)
    DSPARK=$(env_get DSPARK auto)
    TOOL_CALL_PARSER=$(env_get TOOL_CALL_PARSER deepseek_v41)
    REASONING_PARSER=$(env_get REASONING_PARSER deepseek_r1)
    DEFAULT_TEMPERATURE=$(env_get DEFAULT_TEMPERATURE 1.0)
    DEFAULT_TOP_P=$(env_get DEFAULT_TOP_P 1.0)
    DEFAULT_ENABLE_THINKING=$(env_get DEFAULT_ENABLE_THINKING "")
    VMLX_EXTRA_ARGS=$(env_get VMLX_EXTRA_ARGS "")

    # --- macOS の GPU wired メモリ上限 (MiB)。sudo sysctl で設定する ---
    WIRED_LIMIT_MB=$(env_get WIRED_LIMIT_MB 245760)

    VMLX="${VENV_DIR}/bin/vmlx"
    HF="${VENV_DIR}/bin/hf"
    LOG_DIR="${RUNTIME_DIR}/logs"
    LOG_FILE="${LOG_DIR}/serve.log"
    PID_FILE="${RUNTIME_DIR}/serve.pid"
}

load_env
