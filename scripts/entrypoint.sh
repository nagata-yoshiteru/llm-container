#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash-0731 / dual DGX Spark (GB10, SM121) entrypoint
#
# ROLE=head   -> vllm serve (rank 0, API サーバを持つ)
# ROLE=worker -> vllm serve --headless (rank 1, API なし)
#
# 分散バックエンドは Ray ではなく mp (torch.distributed SPMD)。
# head/worker が同じ `vllm serve` を --nnodes/--node-rank/--master-addr 付きで
# 起動し、MASTER_ADDR:MASTER_PORT で rendezvous する。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# compose は未設定の変数を `${VAR:-}` で「空文字がセットされた状態」で渡してくる。
# vLLM / FlashInfer の一部パーサは「空文字」と「未設定」を区別して前者で落ちる
# (例: FLASHINFER_CUDA_ARCH_LIST="" -> arch.split(".") で ValueError)。
# なので空のものは明示的に unset する。
# ---------------------------------------------------------------------------
unset_if_empty() {
    local name
    for name in "$@"; do
        if [ -n "${!name+x}" ] && [ -z "${!name}" ]; then
            unset "$name"
        fi
    done
}

unset_if_empty \
    VLLM_ATTENTION_BACKEND \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN \
    VLLM_NCCL_SO_PATH \
    VLLM_CACHE_ROOT \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS \
    VLLM_USE_BREAKABLE_CUDAGRAPH \
    VLLM_USE_DEEP_GEMM_E8M0 \
    VLLM_MOE_USE_DEEP_GEMM \
    VLLM_USE_B12X_MOE \
    VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM \
    VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M \
    VLLM_USE_FLASHINFER_SAMPLER \
    FLASHINFER_CUDA_ARCH_LIST \
    FLASHINFER_DISABLE_VERSION_CHECK \
    FLASHINFER_WORKSPACE_BASE \
    TILELANG_CLEANUP_TEMP_FILES \
    DG_JIT_USE_NVRTC \
    DG_JIT_NVCC_COMPILER \
    CUTE_DSL_ARCH \
    DSPARK_ENCODING_FILE \
    TORCH_CUDA_ARCH_LIST \
    NCCL_IGNORE_CPU_AFFINITY \
    NCCL_IB_GID_INDEX \
    NCCL_NET \
    NCCL_CROSS_NIC \
    NCCL_CUMEM_ENABLE \
    NCCL_IB_ADDR_FAMILY \
    NCCL_IB_ROCE_VERSION_NUM \
    MAX_JOBS

# ---------------------------------------------------------------------------
# LD_LIBRARY_PATH はイメージ側の値を潰さないよう「前に足す」。
# (bjk110 イメージでは HPC-X の NCCL RDMA plugin を pip 版 NCCL より先に見せる。
#  anemll イメージでは /usr/local/cuda/lib64 だけを足す)
# ---------------------------------------------------------------------------
if [ -n "${VLLM_LD_LIBRARY_PATH_EXTRA:-}" ]; then
    export LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH_EXTRA}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

: "${ROLE:?ROLE must be 'head' or 'worker'}"
: "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH must be set}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set}"
: "${TP_SIZE:=2}"
: "${MASTER_PORT:=29501}"
: "${NNODES:=${TP_SIZE}}"

for v in HEAD_ROCE_IP WORKER_ROCE_IP ROCE_IF_NAME IB_HCA_NAME; do
    if [ -z "${!v:-}" ]; then
        echo "[entrypoint] ERROR: ${v} is required (see .env)" >&2
        exit 1
    fi
done

: "${MASTER_ADDR:=${HEAD_ROCE_IP}}"
if [ "${ROLE}" = "head" ]; then
    : "${NODE_RANK:=0}"
else
    : "${NODE_RANK:=1}"
fi
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK

echo "[entrypoint] role=${ROLE} rank=${NODE_RANK}/${NNODES} tp=${TP_SIZE}"
echo "[entrypoint] rendezvous=${MASTER_ADDR}:${MASTER_PORT} iface=${ROCE_IF_NAME} hca=${IB_HCA_NAME} gid=${NCCL_IB_GID_INDEX:-<unset>}"
echo "[entrypoint] model=${MODEL_CONTAINER_PATH}"

# ---------------------------------------------------------------------------
# RDMA プリフライト: HCA が見えていない状態で起動すると NCCL が
# "unhandled system error" で数分後に死ぬので、先に落とす。
# ---------------------------------------------------------------------------
if command -v ibv_devinfo >/dev/null 2>&1; then
    if ! ibv_devinfo -d "${IB_HCA_NAME}" 2>/dev/null | grep -q "PORT_ACTIVE"; then
        echo "[entrypoint] ERROR: HCA ${IB_HCA_NAME} is not PORT_ACTIVE inside the container." >&2
        echo "[entrypoint]   --device /dev/infiniband と /sys/class/infiniband のマウントを確認。" >&2
        exit 1
    fi
    echo "[entrypoint] RDMA OK: ${IB_HCA_NAME} PORT_ACTIVE"
fi

# ---------------------------------------------------------------------------
# 0731 同梱の encoding をランタイムに入れる (DSPARK_ENCODING_INSTALL=1 のときだけ)
#
# 0731 は chat template を同梱せず、`encoding/encoding_dsv4.py` が
# メッセージ整形と reasoning_effort (low/high/max) の解釈を持っている。
# チェックポイントより古いランタイムだと low が high に潰されるので、
# encoder を上書きしたうえで wrapper の分岐も直す (どちらも best-effort)。
# `--tokenizer-mode deepseek_v4` とセットで使うこと。
# ---------------------------------------------------------------------------
if [ "${DSPARK_ENCODING_INSTALL:-0}" = "1" ]; then
    ENCODING_SRC="${DSPARK_ENCODING_FILE:-${MODEL_CONTAINER_PATH}/encoding/encoding_dsv4.py}"
    VLLM_DIR="$(python3 -c 'import os,vllm;print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -f "${ENCODING_SRC}" ] && [ -n "${VLLM_DIR}" ] && [ -d "${VLLM_DIR}/tokenizers" ]; then
        cp "${ENCODING_SRC}" "${VLLM_DIR}/tokenizers/deepseek_v4_encoding.py"
        echo "[entrypoint] encoding installed: ${ENCODING_SRC} -> ${VLLM_DIR}/tokenizers/deepseek_v4_encoding.py"
        python3 - "${VLLM_DIR}" <<'PY' || echo "[entrypoint] WARN: reasoning_effort パッチはスキップ (パターン不一致)" >&2
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "tokenizers" / "deepseek_v4.py"
old = ('elif reasoning_effort in ("max", "xhigh"):\n'
       '                reasoning_effort = "max"\n'
       '            else:\n'
       '                reasoning_effort = "high"')
new = ('elif reasoning_effort in ("max", "xhigh"):\n'
       '                reasoning_effort = "max"\n'
       '            elif reasoning_effort == "high":\n'
       '                reasoning_effort = "high"\n'
       '            else:\n'
       '                reasoning_effort = "low"')
s = p.read_text()
if new in s:
    print("[entrypoint] reasoning_effort パッチは適用済み")
    raise SystemExit(0)
if old not in s:
    raise SystemExit(1)
p.write_text(s.replace(old, new))
print("[entrypoint] reasoning_effort パッチを適用した")
PY
    else
        echo "[entrypoint] WARN: DSPARK_ENCODING_INSTALL=1 だが ${ENCODING_SRC} が無い。スキップする" >&2
    fi
fi

# ---------------------------------------------------------------------------
# vllm serve コマンド組み立て
# ---------------------------------------------------------------------------
# set -f: SERVED_MODEL_NAME に複数エイリアスを空白区切りで書けるようにしつつ、
# cudagraph_capture_sizes の [8] などが glob 展開されるのを防ぐ。
set -f

VLLM_CMD=(
    vllm serve "${MODEL_CONTAINER_PATH}"
    --served-model-name ${SERVED_MODEL_NAME}
    --trust-remote-code
    --host 0.0.0.0
    --port "${HOST_PORT:-8910}"
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --max-num-seqs "${MAX_NUM_SEQS:-1}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.87}"
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
)

[ "${ROLE}" = "worker" ] && VLLM_CMD+=(--headless)

# VLLM_*_ARGS は空白区切りで展開する。
# JSON を渡す場合は値の内側に空白を入れないこと (例: {"method":"dspark"})。
#
#   VLLM_PARSER_ARGS : reasoning / tool-call パーサ。全構成で共通
#   VLLM_KV_ARGS     : KV キャッシュ関連。コンテキスト長ごとに変わる (presets/)
#   VLLM_EXTRA_ARGS  : その他。全構成で共通
for _args in "${VLLM_PARSER_ARGS:-}" "${VLLM_KV_ARGS:-}" "${VLLM_EXTRA_ARGS:-}"; do
    [ -n "${_args}" ] || continue
    # shellcheck disable=SC2206
    VLLM_CMD+=(${_args})
done
set +f

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
