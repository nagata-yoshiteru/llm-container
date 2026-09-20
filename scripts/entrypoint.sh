#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1-Flash (EXL3) / dual DGX Spark (GB10, SM121) entrypoint
#   (出典: MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks の起動引数を読替)
#
# ROLE=head   -> vllm serve (rank 0, API サーバを持つ)
# ROLE=worker -> vllm serve --headless (rank 1, API なし)
#
# 分散バックエンドは Ray ではなく mp (torch.distributed SPMD)。
# head/worker が同じ `vllm serve` を --nnodes/--node-rank/--master-addr 付きで
# 起動し、MASTER_ADDR:MASTER_PORT で rendezvous する。
#
# SM121 向けパッチ (E3 v2 grouped kernels / block64 envelope / exllamav3 lock
# buffer / pinned H2D staging) はイメージ側に焼かれているので、Vision-Exp まで
# あった patches/ のランタイム hotfix 適用は無い。
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
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB \
    FLASHINFER_CUDA_ARCH_LIST \
    TORCH_CUDA_ARCH_LIST \
    DSV41_EXL3_SERIAL_STREAMS \
    DSV41_IO_THREADS \
    DSV41_INDEXER_PREFILL_FACTOR \
    DSV41_PREFILL_EMPTY_CACHE_TOKENS \
    DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB \
    DSV41_PREFILL_END_EMPTY_CACHE \
    DSV41_DROP_PAGE_CACHE \
    DSV41_SKIP_MIXED_WARMUP \
    DSV41_BOOT_SHAPE_WARMUP \
    DSV41_CACHE_GIB \
    DSV41_RESIDENT_SCALES \
    VLLM_DISABLE_SHARED_EXPERTS_STREAM \
    EXL3_FUSED_MOE \
    EXL3_FAT_KERNEL \
    EXL3_FAT_GROUPED \
    EXL3_TEMP_ROWS_FUSED \
    GLM53_SUPPRESS_STOPS_IN_REASONING \
    GLM53_MIXED_PREFILL_CHUNK \
    GLM53_SPINWAIT_MS \
    NCCL_IGNORE_CPU_AFFINITY \
    NCCL_IB_GID_INDEX \
    NCCL_NET \
    NCCL_CROSS_NIC \
    NCCL_CUMEM_ENABLE \
    NCCL_IB_ADDR_FAMILY \
    NCCL_IB_ROCE_VERSION_NUM \
    NCCL_IB_MERGE_NICS \
    NCCL_BUFFSIZE \
    NCCL_LL128_BUFFSIZE \
    NCCL_PROTO \
    NCCL_MAX_NCHANNELS \
    LIMIT_MM \
    SKIP_MM_PROFILING \
    MAX_JOBS

# ---------------------------------------------------------------------------
# LD_LIBRARY_PATH はイメージ側の値を潰さないよう「前に足す」。
# ---------------------------------------------------------------------------
if [ -n "${VLLM_LD_LIBRARY_PATH_EXTRA:-}" ]; then
    export LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH_EXTRA}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

: "${ROLE:?ROLE must be 'head' or 'worker'}"
: "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH must be set}"
: "${ENGRAM_CONTAINER_PATH:?ENGRAM_CONTAINER_PATH must be set (Engram は量子化されない。--hf-overrides の engram_table_dir と一致させること)}"
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

# Engram がマウントされていないと --hf-overrides が空振りし、起動して最初の
# n-gram 参照で意味の通らない出力になる (fail-closed)。
if [ ! -d "${ENGRAM_CONTAINER_PATH}" ]; then
    echo "[entrypoint] ERROR: ${ENGRAM_CONTAINER_PATH} が無い。compose の ENGRAM_PATH マウントと" >&2
    echo "[entrypoint]   ./scripts/fetch-model.sh engram を確認すること。" >&2
    exit 1
fi
if [ ! -f "${ENGRAM_CONTAINER_PATH}/model-00047-of-00048.safetensors" ] || \
   [ ! -f "${ENGRAM_CONTAINER_PATH}/model-00048-of-00048.safetensors" ]; then
    echo "[entrypoint] ERROR: ${ENGRAM_CONTAINER_PATH} に shard 47/48 が無い。" >&2
    echo "[entrypoint]   ネイティブ deepseek-ai/DeepSeek-V4.1-Flash の model-00047/00048 が必要。" >&2
    exit 1
fi

: "${MASTER_ADDR:=${HEAD_ROCE_IP}}"
if [ "${ROLE}" = "head" ]; then
    : "${NODE_RANK:=0}"
else
    : "${NODE_RANK:=1}"
fi
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK

echo "[entrypoint] role=${ROLE} rank=${NODE_RANK}/${NNODES} tp=${TP_SIZE}"
echo "[entrypoint] rendezvous=${MASTER_ADDR}:${MASTER_PORT} iface=${ROCE_IF_NAME} hca=${IB_HCA_NAME} gid=${NCCL_IB_GID_INDEX:-<unset>}"
echo "[entrypoint] model=${MODEL_CONTAINER_PATH} engram=${ENGRAM_CONTAINER_PATH}"

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
        echo "[entrypoint]   dual-HCA にしたなら、2 本目に IP が振られているかも確認 (README 参照)。" >&2
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
    --max-model-len "${MAX_MODEL_LEN:-600000}"
    --max-num-seqs "${MAX_NUM_SEQS:-2}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1536}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.88}"
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
)

[ "${ROLE}" = "worker" ] && VLLM_CMD+=(--headless)

# 条件付きフラグ (出典レシピの .env ノブに対応)
#   LANGUAGE_MODEL_ONLY=1 : text-only サーバ (vision tower を積まない)
#   ENFORCE_EAGER=1       : CUDA graph _CAPTURE を止める (VRAM 節約/切り分け用)
#   SKIP_MM_PROFILING     : recipe 既定 =1。vision tower ON でも mm profiling を
#                           走らせない (overlay が max_image_tokens を 0 に pin して
#                           いるため見積もり対象が無い)。LANGUAGE_MODEL_ONLY=1 なら
#                           どちらにせよ不要。
#   LIMIT_MM              : {"image":100} 等。空白を入れないこと。
[ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]  && VLLM_CMD+=(--language-model-only)
[ "${ENFORCE_EAGER:-0}" = "1" ]         && VLLM_CMD+=(--enforce-eager)
[ "${SKIP_MM_PROFILING:-0}" = "1" ]     && VLLM_CMD+=(--skip-mm-profiling)
[ -n "${LIMIT_MM:-}" ]                  && VLLM_CMD+=(--limit-mm-per-prompt "${LIMIT_MM}")

# VLLM_*_ARGS は空白区切りで展開する。
# JSON を渡す場合は値の内側に空白を入れないこと (例: {"method":"dspark"} /
# engram_table_dir のコンテナ内パスに空白を入れないこと)。
#
#   VLLM_PARSER_ARGS : tokenizer / reasoning / tool-call パーサ
#   VLLM_KV_ARGS     : KV キャッシュ関連 (--kv-cache-dtype は渡さない)
#   VLLM_EXTRA_ARGS  : quantization / hf-overrides / speculative / retention
for _args in "${VLLM_PARSER_ARGS:-}" "${VLLM_KV_ARGS:-}" "${VLLM_EXTRA_ARGS:-}"; do
    [ -n "${_args}" ] || continue
    # shellcheck disable=SC2206
    VLLM_CMD+=(${_args})
done
set +f

# 出典レシピには「ブート中にログが 420s 黙ったら py-spy で両 rank のスタックを
# 拾う」hang detector があるが、ここでは未移植。詰まったと感じたら手で:
#   sudo docker exec dsv41-head py-spy dump --native <EngineCore pid>
#   sudo docker logs dsv41-worker | tail

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
