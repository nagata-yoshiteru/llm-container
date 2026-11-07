#!/usr/bin/env bash
# =============================================================================
# docker compose の代わり。vmlx をバックグラウンドで起動・停止する。
#
#   ./scripts/serve.sh up       # 起動 (ready まで待つ)
#   ./scripts/serve.sh logs     # ログを追う (Ctrl-C で抜けてもサーバは止まらない)
#   ./scripts/serve.sh status   # 稼働確認 (/health と /v1/models)
#   ./scripts/serve.sh down     # 停止
#   ./scripts/serve.sh fg       # フォアグラウンドで起動 (デバッグ用)
#
# ログは .cache/vmlx/logs/serve.log、PID は .cache/vmlx/serve.pid。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

URL="http://127.0.0.1:${HOST_PORT}"
AUTH=()
[ -n "${API_KEY}" ] && AUTH=(-H "Authorization: Bearer ${API_KEY}")

running_pid() {
    [ -f "${PID_FILE}" ] || return 1
    local pid; pid=$(cat "${PID_FILE}")
    kill -0 "${pid}" 2>/dev/null && { echo "${pid}"; return 0; }
    rm -f "${PID_FILE}"; return 1
}

case "${1:-}" in
up)
    if pid=$(running_pid); then echo "起動済み (pid ${pid})"; exit 0; fi
    mkdir -p "${LOG_DIR}"
    echo "==> $(date '+%F %T') up" >> "${LOG_FILE}"
    nohup "${REPO_ROOT}/scripts/entrypoint.sh" >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "pid $(cat "${PID_FILE}") / log ${LOG_FILE}"
    echo "ready を待つ (197GiB のロードで数分かかる。Ctrl-C で待つのをやめても起動は続く)"
    # /health はロード中でも返ることがあるので、/v1/models にモデル名が出るまで待つ
    while ! curl -fsS -m 5 ${AUTH[@]+"${AUTH[@]}"} "${URL}/v1/models" 2>/dev/null | grep -q "\"${SERVED_MODEL_NAME}\""; do
        if ! running_pid >/dev/null; then
            echo "起動に失敗。ログの末尾:" >&2; tail -n 40 "${LOG_FILE}" >&2; exit 1
        fi
        sleep 5
    done
    echo "ready: ${URL}/v1  model=${SERVED_MODEL_NAME}"
    ;;
down)
    pid=$(running_pid) || { echo "起動していません"; exit 0; }
    kill -TERM "${pid}"
    for _ in $(seq 60); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
    if kill -0 "${pid}" 2>/dev/null; then
        echo "60 秒で止まらないので KILL" >&2; kill -KILL "${pid}"
    fi
    rm -f "${PID_FILE}"
    echo "停止しました"
    ;;
logs)
    exec tail -n 200 -F "${LOG_FILE}"
    ;;
status)
    if pid=$(running_pid); then echo "pid ${pid}"; else echo "停止中"; exit 1; fi
    curl -fsS ${AUTH[@]+"${AUTH[@]}"} "${URL}/health" && echo
    curl -fsS ${AUTH[@]+"${AUTH[@]}"} "${URL}/v1/models" && echo
    ;;
fg)
    exec "${REPO_ROOT}/scripts/entrypoint.sh"
    ;;
*)
    sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
