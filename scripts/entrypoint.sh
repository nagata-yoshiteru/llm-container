#!/usr/bin/env bash
# =============================================================================
# nvidia/Qwen3.8-Flash-Next-NVFP4 / 単一ホスト + SM12x GPU 複数枚 entrypoint
#
# 単一ノードなので `vllm serve` を 1 つ立てるだけ。TP は TP_SIZE で決まる。
# 分散バックエンドは Ray ではなく mp (torch.distributed SPMD) で、rank 間は
# 同一ホスト内のプロセス間通信 + NCCL。
#
# DGX Spark x2 ブランチにあった ROLE / NODE_RANK / MASTER_ADDR / --headless /
# RDMA プリフライトは全部要らなくなったので落としてある。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# compose は未設定の変数を `${VAR-}` で「空文字がセットされた状態」で渡してくる。
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
    VLLM_USE_DEEP_GEMM \
    VLLM_MOE_FORCE_MARLIN \
    VLLM_WORKER_MULTIPROC_METHOD \
    NCCL_P2P_DISABLE \
    NCCL_CUMEM_ENABLE \
    MAX_JOBS

: "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH must be set}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set}"
: "${TP_SIZE:=2}"

echo "[entrypoint] single-node tp=${TP_SIZE}"
echo "[entrypoint] model=${MODEL_CONTAINER_PATH}"

# ---------------------------------------------------------------------------
# GPU プリフライト
#
# device cgroup を通し損ねていると (no-cgroups=true + devices: 書き忘れ)、
# vLLM は "Failed to infer device type" という分かりにくいエラーで死ぬ。
# TP_SIZE より GPU が少ない場合も、モデルロードを全部終えたあとに
# 割り当てで落ちるので時間を無駄にする。どちらもここで落とす。
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! GPU_LIST=$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap \
                    --format=csv,noheader 2>&1); then
        echo "[entrypoint] ERROR: コンテナ内で nvidia-smi が動きません:" >&2
        echo "${GPU_LIST}" | sed 's/^/[entrypoint]   /' >&2
        echo "[entrypoint]   compose の devices: に /dev/nvidia0 /dev/nvidia1 が" >&2
        echo "[entrypoint]   両方あるか、nvidia-container-toolkit が入っているか確認。" >&2
        exit 1
    fi
    echo "${GPU_LIST}" | sed 's/^/[entrypoint] GPU: /'

    N_GPU=$(echo "${GPU_LIST}" | grep -c .)
    if [ "${N_GPU}" -lt "${TP_SIZE}" ]; then
        echo "[entrypoint] ERROR: GPU が ${N_GPU} 個しか見えていません (TP_SIZE=${TP_SIZE})。" >&2
        echo "[entrypoint]   NVIDIA_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES と" >&2
        echo "[entrypoint]   compose の devices: を確認すること。" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# モデルのプリフライト。HF_HUB_OFFLINE=1 なので、マウントが空だと
# 「repo id として解決できない」系の分かりにくいエラーで死ぬ。先に落とす。
# ---------------------------------------------------------------------------
for f in config.json model.safetensors.index.json tokenizer.json; do
    if [ ! -f "${MODEL_CONTAINER_PATH}/${f}" ]; then
        echo "[entrypoint] ERROR: ${MODEL_CONTAINER_PATH}/${f} が無い。" >&2
        echo "[entrypoint]   scripts/fetch-model.sh を実行すること。" >&2
        exit 1
    fi
done

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
    --max-num-seqs "${MAX_NUM_SEQS:-8}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend mp
)

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
