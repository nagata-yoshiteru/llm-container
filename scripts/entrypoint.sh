#!/usr/bin/env bash
# =============================================================================
# nvidia/Qwen3.8-Flash-Next-NVFP4 / dual DGX Spark (GB10, SM121) entrypoint
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
# vLLM / NCCL の一部パーサは「空文字」と「未設定」を区別して前者で落ちる。
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
    VLLM_ALLOW_LONG_MAX_MODEL_LEN \
    VLLM_NCCL_SO_PATH \
    VLLM_USE_DEEP_GEMM \
    NCCL_IB_GID_INDEX \
    NCCL_IB_MERGE_NICS \
    NCCL_CROSS_NIC \
    NCCL_IGNORE_CPU_AFFINITY \
    MAX_JOBS

# イメージ側の値を潰さないよう「前に足す」
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
        echo "[entrypoint] ERROR: ${v} is required (see .env.example)" >&2
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
# モデルのプリフライト。HF_HUB_OFFLINE=1 なので、マウントが空だと
# 「repo id として解決できない」系の分かりにくいエラーで死ぬ。先に落とす。
# ---------------------------------------------------------------------------
for f in config.json model.safetensors.index.json tokenizer.json; do
    if [ ! -f "${MODEL_CONTAINER_PATH}/${f}" ]; then
        echo "[entrypoint] ERROR: ${MODEL_CONTAINER_PATH}/${f} が無い。" >&2
        echo "[entrypoint]   scripts/fetch-model.sh を **両ノードで** 実行すること。" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# RDMA プリフライト: HCA が見えていない状態で起動すると NCCL が
# "unhandled system error" で数分後に死ぬので、先に落とす。
# IB_HCA_NAME はカンマ区切りで複数書ける (dual-HCA / NCCL_IB_MERGE_NICS=1)。
# ---------------------------------------------------------------------------
if command -v ibv_devinfo >/dev/null 2>&1; then
    _hca_ok=1
    IFS=',' read -ra _HCA_LIST <<< "${IB_HCA_NAME}"
    for _hca in "${_HCA_LIST[@]}"; do
        [ -n "${_hca}" ] || continue
        if ibv_devinfo -d "${_hca}" 2>/dev/null | grep -q "PORT_ACTIVE"; then
            echo "[entrypoint] RDMA OK: ${_hca} PORT_ACTIVE"
        else
            echo "[entrypoint] ERROR: HCA ${_hca} is not PORT_ACTIVE inside the container." >&2
            _hca_ok=0
        fi
    done
    if [ "${_hca_ok}" != "1" ]; then
        echo "[entrypoint]   --device /dev/infiniband と /sys/class/infiniband のマウントを確認。" >&2
        echo "[entrypoint]   dual-HCA なら 2 本目に IP が振られているかも確認 (.env.example 参照)。" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# vllm serve コマンド組み立て
# ---------------------------------------------------------------------------
# set -f: SERVED_MODEL_NAME に複数エイリアスを空白区切りで書けるようにしつつ、
# JSON 中の [11,11,10] などが glob 展開されるのを防ぐ。
set -f

VLLM_CMD=(
    vllm serve "${MODEL_CONTAINER_PATH}"
    --served-model-name ${SERVED_MODEL_NAME}
    --trust-remote-code
    --host 0.0.0.0
    --port "${HOST_PORT:-8910}"
    --max-model-len "${MAX_MODEL_LEN:-1048576}"
    --max-num-seqs "${MAX_NUM_SEQS:-4}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
)

[ "${ROLE}" = "worker" ] && VLLM_CMD+=(--headless)

# VLLM_*_ARGS は空白区切りで展開する。
# JSON を渡す場合は値の内側に空白を入れないこと (例: {"method":"mtp"})。
#
#   VLLM_PARSER_ARGS : reasoning / tool-call パーサ
#   VLLM_KV_ARGS     : KV キャッシュ / context 長まわり
#   VLLM_EXTRA_ARGS  : 量子化 / 投機デコード / 速度レバー
for _args in "${VLLM_PARSER_ARGS:-}" "${VLLM_KV_ARGS:-}" "${VLLM_EXTRA_ARGS:-}"; do
    [ -n "${_args}" ] || continue
    # shellcheck disable=SC2206
    VLLM_CMD+=(${_args})
done
set +f

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
