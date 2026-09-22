#!/usr/bin/env bash
# =============================================================================
# page-cache flusher。**コンテナを起動したら 2 台とも回しておく。**
#
#   ./scripts/flusher.sh &            # 既定 10 時間
#   DURATION=3600 ./scripts/flusher.sh &
#   pkill -f scripts/flusher.sh       # 停止
#
# GB10 の NVRM は MemAvailable ではなく **MemFree** で割当を判定し、page cache を
# 強制回収しない。重み 198GB のロードで page cache が MemFree を取り潰すため、
# 回収可能なキャッシュが数 GB 残っているだけで KV slab の割当が
# NV_ERR_NO_MEMORY で落ちる。
#
# 閾値式のフラッシャは使わないこと。上流では閾値フラッシャが起動 25 分後に
# 期限切れし、その 1 分後に 9GiB の回収可能キャッシュを抱えたままノードが死んだ。
# **無条件に 20 秒ごと**が正解。
# =============================================================================
set -uo pipefail

DURATION="${DURATION:-36000}"
INTERVAL="${INTERVAL:-20}"
LOG="${LOG:-${HOME}/glm53-flusher.log}"

if ! sudo -n true 2>/dev/null; then
    echo "sudo を非対話で叩けません。以下を先に通しておくこと:" >&2
    echo "  sudo -v" >&2
    exit 2
fi

echo "$(date +%T) flusher start (duration=${DURATION}s interval=${INTERVAL}s)" | tee -a "${LOG}"
t0=$(date +%s)
n=0
while [ $(( $(date +%s) - t0 )) -lt "${DURATION}" ]; do
    sync
    echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
    free_gb=$(awk '/^MemFree:/ {print int($2 / 1048576)}' /proc/meminfo)
    # MemFree が薄いときは断片化も解す (大きな KV slab は連続領域を要求する)
    if [ "${free_gb}" -lt 6 ]; then
        echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null 2>&1 || true
    fi
    n=$((n + 1))
    if [ $((n % 30)) -eq 0 ]; then
        echo "$(date +%T) tick ${n} MemFree ${free_gb}G" >> "${LOG}"
    fi
    sleep "${INTERVAL}"
done
echo "$(date +%T) flusher done after ${n} ticks" | tee -a "${LOG}"
