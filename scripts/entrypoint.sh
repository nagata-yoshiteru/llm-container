#!/usr/bin/env bash
# =============================================================================
# dealignai/GLM-5.3-Flash-UNCENSORED-FP8 (native block-FP8 128x128) /
# 4x NVIDIA DGX Spark (GB10 / SM121) / 光スイッチ (フルメッシュ・同一 L2)
#
# -----------------------------------------------------------------------------
# 並列化 = TP=4 (モデルカードの推奨構成と同じ)
# -----------------------------------------------------------------------------
# glm5_next の形状は 4 ですべて割り切れる (3 台のときは割れずに詰んでいた):
#   num_attention_heads 64 / 4 = 16    (attention.py:452 の assert を通る)
#   KDA num_heads       64 / 4 = 16    (kda.py:219 の assert を通る)
#   vocab_size      154880 / 4 = 38720
#   moe_intermediate  2048 / 4 = 512   (FP8 block 128 の倍数)
#   intermediate     12288 / 4 = 3072
# PP は glm5_next では vllm 側でゲートされている (make_empty_intermediate_tensors
# 未実装) ので使わない。DP/EP も不要 (TP なら複製が無く、重みが一番軽い)。
#
# 重み = (306GiB - vision 1GiB) / 4 = 約 76GiB/rank。3 台 DP3/EP3 のときは
# 111GiB/rank で、起動時の空き 109.3GiB に収まらず落ちた (2026-10-06)。
#
# -----------------------------------------------------------------------------
# 起動のしかた (分散バックエンドは Ray ではなく mp = torch.distributed SPMD)
# -----------------------------------------------------------------------------
#   4 台が同じ `vllm serve` を --nnodes 4 --node-rank N --master-addr <head>
#   付きで起動し、MASTER_ADDR:MASTER_PORT で rendezvous する。
#   ROLE=head   -> rank 0。API サーバあり
#   ROLE=worker -> rank 1..3。--headless
#   TP の all-reduce は毎層 4 ノードをまたぐので NCCL (RoCEv2) が効いている前提。
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
    VLLM_USE_BREAKABLE_CUDAGRAPH \
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB \
    VLLM_ENGINE_READY_TIMEOUT_S \
    FLASHINFER_CUDA_ARCH_LIST \
    TORCH_CUDA_ARCH_LIST \
    GLM53_SUPPRESS_STOPS_IN_REASONING \
    GLM53_MIXED_PREFILL_CHUNK \
    GLM53_SPINWAIT_MS \
    NCCL_IGNORE_CPU_AFFINITY \
    NCCL_IB_GID_INDEX \
    NCCL_IB_ADDR_RANGE \
    NCCL_IB_ADDR_FAMILY \
    NCCL_IB_ROCE_VERSION_NUM \
    NCCL_NET \
    NCCL_CROSS_NIC \
    NCCL_IB_MERGE_NICS \
    NCCL_CUMEM_ENABLE \
    NCCL_BUFFSIZE \
    NCCL_LL128_BUFFSIZE \
    NCCL_PROTO \
    NCCL_MAX_NCHANNELS \
    LIMIT_MM \
    MAX_JOBS

# ---------------------------------------------------------------------------
# LD_LIBRARY_PATH はイメージ側の値を潰さないよう「前に足す」。
# ---------------------------------------------------------------------------
if [ -n "${VLLM_LD_LIBRARY_PATH_EXTRA:-}" ]; then
    export LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH_EXTRA}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

: "${ROLE:?ROLE must be 'head' or 'worker'}"
: "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH must be set}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set}"
: "${TP_SIZE:=4}"
: "${NNODES:=${TP_SIZE}}"
: "${MASTER_PORT:=29501}"
: "${NODE_RANK:?NODE_RANK must be set (head=0 / worker1=1 / worker2=2 / worker3=3)}"
: "${HEAD_ROCE_IP:?HEAD_ROCE_IP must be set (rendezvous address)}"

for v in WORKER1_ROCE_IP WORKER2_ROCE_IP WORKER3_ROCE_IP ROCE_IF_NAME IB_HCA_NAME; do
    if [ -z "${!v:-}" ]; then
        echo "[entrypoint] ERROR: ${v} is required (see .env)" >&2
        exit 1
    fi
done

# 前提が崩れていたら起動前に落とす (重みのロードに 10 分以上待たされた挙句
# assert で死ぬのは辛い)。
if [ "${TP_SIZE}" != "4" ] || [ "${NNODES}" != "4" ]; then
    echo "[entrypoint] ERROR: TP_SIZE=${TP_SIZE} NNODES=${NNODES}。この構成は TP=4 / 4 ノード固定。" >&2
    echo "[entrypoint]   glm5_next は heads=64 / vocab=154880 なので TP=3 は assert で落ち、" >&2
    echo "[entrypoint]   FP8 の重み (306GiB) は 2〜3 台には載らない。" >&2
    exit 1
fi
case "${NODE_RANK}" in
    0|1|2|3) ;;
    *) echo "[entrypoint] ERROR: NODE_RANK=${NODE_RANK} は 0..3 のはず" >&2; exit 1 ;;
esac

: "${MASTER_ADDR:=${HEAD_ROCE_IP}}"
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK

echo "[entrypoint] role=${ROLE} rank=${NODE_RANK}/${NNODES} tp=${TP_SIZE}"
echo "[entrypoint] rendezvous=${MASTER_ADDR}:${MASTER_PORT} iface=${ROCE_IF_NAME} hca=${IB_HCA_NAME} gid=${NCCL_IB_GID_INDEX:-<unset>}"
echo "[entrypoint] model=${MODEL_CONTAINER_PATH}"

# ---------------------------------------------------------------------------
# RDMA プリフライト: HCA が見えていない状態で起動すると NCCL が
# "unhandled system error" で数分後に死ぬので、先に落とす。
# ---------------------------------------------------------------------------
if command -v ibv_devinfo >/dev/null 2>&1; then
    # IB_HCA_NAME はカンマ区切りで複数書ける (dual-HCA / NCCL_IB_MERGE_NICS=1)。
    # 1 本ずつ PORT_ACTIVE を確認する。
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
        echo "[entrypoint]   4 ノード構成では、このノードのケーブルが光スイッチのどのポートに" >&2
        echo "[entrypoint]   刺さっているかも確認すること (2026-10 の flat L2 構成)。" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# vllm serve コマンド組み立て
# ---------------------------------------------------------------------------
# set -f: SERVED_MODEL_NAME に複数エイリアスを空白区切りで書けるようにしつつ、
# JSON 内の [ ... ] などが glob 展開されるのを防ぐ。
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
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
)

# --max-num-batched-tokens は GLM-5.3 (hybrid mamba/attention) では
# 「渡さない」のが検証済みレシピ。.env で空にしておくとここで落ちる。
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && VLLM_CMD+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")

[ "${ROLE}" = "worker" ] && VLLM_CMD+=(--headless)

# 条件付きフラグ (Makefile 的に .env のスイッチで切り替える)
#   LANGUAGE_MODEL_ONLY=1 : text-only サーバ (vision tower を積まない)。
#                           SM12x の sparse-MLA prefill には画像幅のカーネルが
#                           無く、メモリも ~1.2GiB/rank 余分に食うので既定 1。
#   ENFORCE_EAGER=1       : CUDA graph capture を止める。GLM-5.3 のこのパスは
#                           graph capture 不可 (上流実測)。VRAM 節約にもなる。
#   SKIP_MM_PROFILING     : vision ON のとき mm profiling を走らせない。
#   LIMIT_MM              : {"image":2,"video":1} 等。空白を入れないこと。
[ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]  && VLLM_CMD+=(--language-model-only)
[ "${ENFORCE_EAGER:-0}" = "1" ]         && VLLM_CMD+=(--enforce-eager)
[ "${SKIP_MM_PROFILING:-0}" = "1" ]     && VLLM_CMD+=(--skip-mm-profiling)
[ -n "${LIMIT_MM:-}" ]                  && VLLM_CMD+=(--limit-mm-per-prompt "${LIMIT_MM}")

# VLLM_*_ARGS は空白区切りで展開する。
# JSON を渡す場合は値の内側に空白を入れないこと
# (例: {"method":"mtp","num_speculative_tokens":1} / {"image":2,"video":1})。
#
#   VLLM_PARSER_ARGS : tokenizer / reasoning / tool-call パーサ
#   VLLM_KV_ARGS     : KV キャッシュ関連 (--kv-cache-dtype / --block-size)
#   VLLM_EXTRA_ARGS  : quantization / moe-backend / speculative / その他
for _args in "${VLLM_PARSER_ARGS:-}" "${VLLM_KV_ARGS:-}" "${VLLM_EXTRA_ARGS:-}"; do
    [ -n "${_args}" ] || continue
    # shellcheck disable=SC2206
    VLLM_CMD+=(${_args})
done
set +f

# 詰まったと感じたら手で:
#   sudo docker exec <container> py-spy dump --native <EngineCore pid>
#   sudo docker logs <container> | tail
# worker が rendezvous で止まっていたら、head の MASTER_PORT が開いているかから疑う:
#   ss -ltnp | grep "${MASTER_PORT}"

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
