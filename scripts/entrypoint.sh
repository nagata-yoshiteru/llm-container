#!/usr/bin/env bash
# =============================================================================
# dealignai/GLM-5.3-Flash-UNCENSORED-FP8 (native block-FP8 128x128) /
# 3x NVIDIA DGX Spark (GB10 / SM121) / 光スイッチ (フルメッシュ・同一 L2)
#
# -----------------------------------------------------------------------------
# なぜ TP=1 + DP=3 + EP=3 なのか (ここが一番むずかしい)
# -----------------------------------------------------------------------------
# TP=3 は素の vllm では起動しない。glm5_next の形状が 3 で割れないため:
#   attention.py:452-453   assert num_heads % tp_size == 0      (num_heads=64)
#   kda.py:219             assert self.num_heads % self.tp_size == 0 (64)
#   vocab_parallel_embedding  vocab_size=154880 も 3 で割れない
# TP=3 を通すには「head をゼロパディングしたチェックポイント + ロード時の
# pad overlay + カーネル側の power-of-2 対応」が要る (kindling の tp3 一式)。
# この repo は設定のみで完結させたいので、その路線は採らない。
#
# PP=3 も塞がっている。glm5_next は make_empty_intermediate_tensors を
# 実装していないため、vllm 側で PP が明示的にゲートされている:
#   common/model.py:788    "PP is gated off for GLM-5.3-Flash"
#   common/model.py:1169   "does not implement make_empty_intermediate_tensors"
#
# 残るのが DP=3 + EP=3。fused_moe/config.py:1189-1233 のとおり
# --enable-expert-parallel を付けると ep_size = DP x TP (=3) になり、
# MoE 側の tp_size は **1 に落ちる**。したがって
#   - routed experts: 288 / 3 = 96 ずつ。EP 要件 (num_experts % ep_size == 0) を満たす
#   - moe_intermediate_size=2048 は分割されない → パディング不要
#   - attention / embedding は TP=1 なので割り切れなくても assert に当たらない
# つまり形状の問題が全部消える。代償は「attention/dense/embed/vision が 3 ノードに
# 複製される」こと (~+9GiB/rank)。
#
# -----------------------------------------------------------------------------
# 起動のしかた (vllm 公式の multi-node internal DP。--nnodes は使わない)
# -----------------------------------------------------------------------------
#   各ノードが自分の vllm serve を持ち、DP coordinator (= head の rpc port) に
#   ぶら下がる。HTTP は head の 1 本だけ。
#   ROLE=head   -> DP rank 0。API サーバあり
#   ROLE=worker -> DP rank 1 / 2。--headless
#
#   ★ EP の all-to-all は 3 ノードをまたぐので NCCL (RoCEv2) が効いている前提。
#   ★ メモリは極端に厳しい。重み ~107GiB/rank + CUDA ctx/NCCL ~5GiB で
#     121GiB の unified memory をほぼ食い切る。GMU は 0.92〜0.95 で調整し、
#     起動前に preflight.sh で MemAvailable を確認すること。
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
: "${TP_SIZE:=1}"
: "${DP_SIZE:=3}"
: "${DP_START_RANK:?DP_START_RANK must be set (head=0 / worker=1 / worker2=2)}"
: "${DP_RPC_PORT:=13345}"
: "${HEAD_ROCE_IP:?HEAD_ROCE_IP must be set (DP coordinator address)}"

for v in WORKER1_ROCE_IP WORKER2_ROCE_IP ROCE_IF_NAME IB_HCA_NAME; do
    if [ -z "${!v:-}" ]; then
        echo "[entrypoint] ERROR: ${v} is required (see .env)" >&2
        exit 1
    fi
done

# DP の前提が崩れていたら起動前に落とす (10 分待たされた挙句 assert で死ぬのは辛い)。
if [ "${TP_SIZE}" != "1" ]; then
    echo "[entrypoint] ERROR: TP_SIZE=${TP_SIZE}。このリポジトリの GLM-5.3 構成は TP=1 前提。" >&2
    echo "[entrypoint]   TP>1 は glm5_next の num_heads=64 / vocab=154880 が割り切れず assert で落ちる。" >&2
    echo "[entrypoint]   (TP=3 を通すにはゼロパディング済みチェックポイントが必要。README 参照)" >&2
    exit 1
fi
if [ "${DP_SIZE}" != "3" ]; then
    echo "[entrypoint] ERROR: DP_SIZE=${DP_SIZE}。3 ノード 1 rank ずつなので 3 固定。" >&2
    exit 1
fi
case "${DP_START_RANK}" in
    0|1|2) ;;
    *) echo "[entrypoint] ERROR: DP_START_RANK=${DP_START_RANK} は 0..2 のはず" >&2; exit 1 ;;
esac

echo "[entrypoint] role=${ROLE} dp_rank=${DP_START_RANK}/${DP_SIZE} tp=${TP_SIZE}"
echo "[entrypoint] dp_coordinator=${HEAD_ROCE_IP}:${DP_RPC_PORT} iface=${ROCE_IF_NAME} hca=${IB_HCA_NAME} gid=${NCCL_IB_GID_INDEX:-<unset>}"
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
        echo "[entrypoint]   3 ノード構成では、このノードのケーブルが光スイッチのどのポートに" >&2
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
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
    # --- attention: TP=1 (どのランクも attention を丸ごと持つ) ---
    --tensor-parallel-size "${TP_SIZE}"
    # --- MoE: DP=3 + EP=3 (expert 96 ずつ、中間次元は分割されない) ---
    --data-parallel-size "${DP_SIZE}"
    --data-parallel-size-local 1
    --data-parallel-start-rank "${DP_START_RANK}"
    --data-parallel-address "${HEAD_ROCE_IP}"
    --data-parallel-rpc-port "${DP_RPC_PORT}"
    --data-parallel-backend mp
    --enable-expert-parallel
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
# DP coordinator は head にいるので、worker が rendezvous で止まっていたら
# head の `--data-parallel-rpc-port` が開いているかから疑う:
#   ss -ltnp | grep "${DP_RPC_PORT}"

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
