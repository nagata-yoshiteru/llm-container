#!/usr/bin/env bash
# Validate before importing SGLang's sitecustomize or allocating GPU memory.
set -euo pipefail
NCCL_IB_GID_INDEX=$(python3 -S /opt/llm/deepseek41_config.py)
export NCCL_IB_GID_INDEX
echo "DeepSeek V4.1-1017: rank=${NODE_RANK}/3, NVMe Engram, RoCEv2 GID=${NCCL_IB_GID_INDEX}"
# boot.py pipes engine stdout. NCCL's C stdout buffer can be lost when SGLang
# kills the scheduler after a failure. NCCL_DEBUG_FILE uses line buffering.
# Keep each launch separate: container PIDs can repeat after recreation.
nccl_log_dir="${STATE_PATH:-/state}/nccl"
mkdir -p -m 755 "$nccl_log_dir"
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-${nccl_log_dir}/rank${NODE_RANK}-$(date -u +%Y%m%dT%H%M%S%N)-%h-%p.log}"
echo "NCCL diagnostics: ${NCCL_DEBUG_FILE} (host: .cache/deepseek41-tp3/runtime/state/nccl/)"
exec python3 -u /opt/dsv41/boot.py run
