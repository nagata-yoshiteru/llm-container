#!/usr/bin/env bash
# =============================================================================
# GLM-5.3-Flash-NVFP4 / dual DGX Spark (GB10, SM121) entrypoint
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
# compose は未設定の変数を `${VAR-}` で「空文字がセットされた状態」で渡してくる。
# vLLM / NCCL / FlashInfer の一部パーサは「空文字」と「未設定」を区別して前者で
# 落ちる (例: FLASHINFER_CUDA_ARCH_LIST="" -> arch.split(".") で ValueError)
# また、VLLM_* の未知変数は vLLM が warning に出す。
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
    NCCL_IB_ADDR_RANGE \
    NCCL_IGNORE_CPU_AFFINITY \
    NCCL_IB_GID_INDEX \
    MAX_JOBS \
    TORCH_CUDA_ARCH_LIST \
    FLASHINFER_CUDA_ARCH_LIST \
    MAX_NUM_BATCHED_TOKENS

# ---------------------------------------------------------------------------
# 起動必須変数のチェック
# ---------------------------------------------------------------------------
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
# kpool top-k SM121 修正の bind-mount チェック (必須)
#
# イメージ標準の sparse_attn_indexer_kpool.py は persistent_topk カーネルを
# SM 数 78 以上で使うが、GB10 (48 SM / 99KB smem) では ~24K トークン超の
# decode で CTA が超過し RuntimeError -> EngineDeadError になる。
# compose が patches/sparse_attn_indexer_kpool_sm121.py を上書き mount する
# はずなので、ゲートが入っているか確認してない場合は落とす。
# ---------------------------------------------------------------------------
KPOOL_PY="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
if [ -f "${KPOOL_PY}" ] && ! grep -q 'multi_processor_count >= 78' "${KPOOL_PY}"; then
    echo "[entrypoint] ERROR: ${KPOOL_PY} に SM121 ゲートがありません。" >&2
    echo "[entrypoint]   patches/sparse_attn_indexer_kpool_sm121.py の bind-mount を確認 (24K ctx 超の decode で engine が死ぬ)" >&2
    exit 1
fi

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
# vllm serve コマンド組み立て
# ---------------------------------------------------------------------------
# set -f: SERVED_MODEL_NAME に複数エイリアスを空白区切りで書けるようにしつつ、
# speculative-config の JSON などが glob 展開されるのを防ぐ。
set -f

VLLM_CMD=(
    vllm serve "${MODEL_CONTAINER_PATH}"
    --served-model-name ${SERVED_MODEL_NAME}
    --trust-remote-code
    --host 0.0.0.0
    --port "${HOST_PORT:-8910}"
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --max-num-seqs "${MAX_NUM_SEQS:-6}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}"
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
)

# 検証済みレシピは max-num-batched-tokens を渡さない (ハイブリッド
# mamba/attention モデルで vLLM の既定が正しい)。設定されていれば渡す。
if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    VLLM_CMD+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

[ "${ROLE}" = "worker" ] && VLLM_CMD+=(--headless)

# VLLM_*_ARGS は空白区切りで展開する。
# JSON を渡す場合は値の内側に空白を入れないこと (例: {"method":"mtp"})。
#
#   VLLM_PARSER_ARGS : reasoning / tool-call パーサ
#   VLLM_KV_ARGS     : KV キャッシュ関連
#   VLLM_EXTRA_ARGS  : その他
for _args in "${VLLM_PARSER_ARGS:-}" "${VLLM_KV_ARGS:-}" "${VLLM_EXTRA_ARGS:-}"; do
    [ -n "${_args}" ] || continue
    # shellcheck disable=SC2206
    VLLM_CMD+=(${_args})
done
set +f

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
