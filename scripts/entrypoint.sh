#!/usr/bin/env bash
# Validate before importing SGLang's sitecustomize or allocating GPU memory.
set -euo pipefail
NCCL_IB_GID_INDEX=$(python3 -S /opt/llm/deepseek41_config.py)
export NCCL_IB_GID_INDEX
echo "DeepSeek V4.1-1017: rank=${NODE_RANK}/3, NVMe Engram, RoCEv2 GID=${NCCL_IB_GID_INDEX}"
exec python3 -u /opt/dsv41/boot.py run
