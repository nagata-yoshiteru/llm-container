#!/usr/bin/env bash
# =============================================================================
# MiniMax-M3 (NVFP4) / 3x DGX Spark (GB10, SM121) entrypoint
#
# ROLE=head   -> Ray head を起動 -> 3 GPU 揃うのを待つ -> vllm serve (API を持つ)
# ROLE=worker -> Ray head に join して --block で居座るだけ
#
# DIST_BACKEND=mp にすると Ray を使わず torch.distributed SPMD で起動する
# (head/worker がそれぞれ vllm serve を --nnodes/--node-rank 付きで起動)。
# ただし実測されているのは ray 側。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# compose は未設定の変数を `${VAR:-}` で「空文字がセットされた状態」で渡してくる。
# vLLM / FlashInfer の一部パーサは「空文字」と「未設定」を区別して前者で落ちる
# (例: FLASHINFER_CUDA_ARCH_LIST="" -> arch.split(".") で ValueError)。
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
    VLLM_CACHE_ROOT \
    FLASHINFER_CUDA_ARCH_LIST \
    TORCH_CUDA_ARCH_LIST \
    CUTE_DSL_ARCH \
    NCCL_IGNORE_CPU_AFFINITY \
    NCCL_IB_ADDR_RANGE \
    NCCL_IB_HCA \
    MAX_JOBS

# ---------------------------------------------------------------------------
# 3 ノードメッシュでは NCCL_IB_GID_INDEX を絶対に固定しない。
# 固定すると subnet-aware routing が無効になり、NCCL が隣人ごとに正しい
# RoCEv2 GID を選べなくなる。結果として「隣人 A を向いている HCA で
# 隣人 B の /30 にダイヤルする」クロスペアリングが起き、
# ibv_modify_qp err 110 (Connection timed out) で rendezvous ごと死ぬ。
# ベースイメージや外から紛れ込んでいても、ここで確実に消す。
# ---------------------------------------------------------------------------
unset NCCL_IB_GID_INDEX || true

: "${ROLE:?ROLE must be 'head' or 'worker'}"
: "${MODEL_CONTAINER_PATH:?MODEL_CONTAINER_PATH must be set}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set}"
: "${TP_SIZE:=1}"
: "${PP_SIZE:=3}"
: "${NNODES:=3}"
: "${DIST_BACKEND:=ray}"
: "${RAY_PORT:=6379}"
: "${MASTER_PORT:=29501}"
: "${NODE_RANK:=0}"

for v in OOB_IF_NAME NODE0_MGMT_IP NODE_MGMT_IP; do
    if [ -z "${!v:-}" ]; then
        echo "[entrypoint] ERROR: ${v} is required (see .env)" >&2
        exit 1
    fi
done

HEAD_IP="${NODE0_MGMT_IP}"
: "${MASTER_ADDR:=${HEAD_IP}}"
export MASTER_ADDR MASTER_PORT

WORLD_SIZE=$((TP_SIZE * PP_SIZE))
if [ "${WORLD_SIZE}" -ne "${NNODES}" ]; then
    echo "[entrypoint] ERROR: TP_SIZE(${TP_SIZE}) * PP_SIZE(${PP_SIZE}) = ${WORLD_SIZE} != NNODES(${NNODES})" >&2
    echo "[entrypoint]   Spark は 1 node 1 GPU なので、並列度の積がノード数と一致する必要がある。" >&2
    exit 1
fi

echo "[entrypoint] role=${ROLE} rank=${NODE_RANK}/${NNODES} tp=${TP_SIZE} pp=${PP_SIZE} backend=${DIST_BACKEND}"
echo "[entrypoint] head=${HEAD_IP} self=${NODE_MGMT_IP} oob=${OOB_IF_NAME}"
echo "[entrypoint] ib_hca=${NCCL_IB_HCA:-<unset>} addr_range=${NCCL_IB_ADDR_RANGE:-<unset>} gid_index=<unset by design>"
echo "[entrypoint] model=${MODEL_CONTAINER_PATH}"

# ---------------------------------------------------------------------------
# NCCL のバージョンを確認しておく。
#
# 3 ノードメッシュには subnet-aware routing が要るので 2.30.7 以上であること。
# それ未満だと NCCL が隣人ごとに正しい HCA を選べず、繋がっていないサブネットに
# ダイヤルして ibv_modify_qp err 110 (Connection timed out) で死ぬ。
#
# Dockerfile で pip の nvidia-nccl-cu13 を 2.30.7 に固定し、system 側が
# 2.30.7 以上ならそちらに張り替えてある。ここでは実際にロードされる版を出すだけ。
#
# 判定には /usr/local/bin/nccl-version (Dockerfile が COPY する
# scripts/nccl-version.py) を使う。pip 版は site-packages に置かれるだけで
# ld キャッシュには載らないので、soname だけでは開けない。
#
# /opt/nccl230 があるのは chthonic イメージ。その場合はそちらを明示的に優先する。
# ---------------------------------------------------------------------------
if [ -e /opt/nccl230/build/lib/libnccl.so.2 ]; then
    unset VLLM_NCCL_SO_PATH NCCL_LOCAL_INFERENCE_PATH NCCL_PR2127_PATH || true
    export LD_PRELOAD=/opt/nccl230/build/lib/libnccl.so.2
    export LD_LIBRARY_PATH=/opt/nccl230/build/lib:${LD_LIBRARY_PATH:-}
fi
if NCCL_VER="$(nccl-version 2>/dev/null)"; then
    echo "[entrypoint] NCCL version: ${NCCL_VER} ($((NCCL_VER / 10000)).$((NCCL_VER / 100 % 100)).$((NCCL_VER % 100)))"
    if [ "${NCCL_VER}" -lt 23007 ]; then
        echo "[entrypoint] WARN: NCCL < 2.30.7。3 ノードメッシュでは subnet-aware routing が無く" >&2
        echo "[entrypoint]       ibv_modify_qp err 110 になる可能性が高い" >&2
    fi
else
    echo "[entrypoint] WARN: NCCL のバージョンを取得できなかった" >&2
fi

# ---------------------------------------------------------------------------
# RDMA プリフライト: HCA が見えていない状態で起動すると NCCL が
# "unhandled system error" で数分後に死ぬので、先に落とす。
# ---------------------------------------------------------------------------
if command -v ibv_devinfo >/dev/null 2>&1 && [ -n "${NCCL_IB_HCA:-}" ]; then
    IFS=',' read -ra _hcas <<< "${NCCL_IB_HCA}"
    for _hca in "${_hcas[@]}"; do
        if ! ibv_devinfo -d "${_hca}" 2>/dev/null | grep -q "PORT_ACTIVE"; then
            echo "[entrypoint] ERROR: HCA ${_hca} is not PORT_ACTIVE inside the container." >&2
            echo "[entrypoint]   --device /dev/infiniband と /sys/class/infiniband のマウントを確認。" >&2
            exit 1
        fi
    done
    echo "[entrypoint] RDMA OK: ${NCCL_IB_HCA} all PORT_ACTIVE"
fi

# ---------------------------------------------------------------------------
# Ray
# ---------------------------------------------------------------------------
: "${RAY_OBJECT_STORE_MEMORY:=1073741824}"

if [ "${DIST_BACKEND}" = "ray" ]; then
    ray stop --force >/dev/null 2>&1 || true
    rm -rf /tmp/ray || true

    if [ "${ROLE}" = "worker" ]; then
        echo "[entrypoint] waiting for Ray head ${HEAD_IP}:${RAY_PORT} ..."
        for _ in $(seq 1 120); do
            if python3 -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
try: s.connect(('${HEAD_IP}', ${RAY_PORT})); sys.exit(0)
except Exception: sys.exit(1)
finally: s.close()
"; then
                break
            fi
            sleep 5
        done
        exec ray start \
            --address="${HEAD_IP}:${RAY_PORT}" \
            --node-ip-address="${NODE_MGMT_IP}" \
            --num-gpus=1 \
            --object-store-memory="${RAY_OBJECT_STORE_MEMORY}" \
            --disable-usage-stats \
            --block
    fi

    # head は API サーバと Ray head を抱える分、worker より free メモリが
    # 5GiB ほど少ない。PP の全体 block 数は各ノードの min で決まるので、
    # head が KV の律速になる。長いコンテキストが要るときは dashboard を切る。
    RAY_HEAD_OPTS=()
    if [ "${RAY_INCLUDE_DASHBOARD:-1}" = "1" ]; then
        RAY_HEAD_OPTS+=(--dashboard-host=0.0.0.0)
    else
        RAY_HEAD_OPTS+=(--include-dashboard=false)
    fi

    ray start --head \
        --port="${RAY_PORT}" \
        --node-ip-address="${NODE_MGMT_IP}" \
        --num-gpus=1 \
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY}" \
        "${RAY_HEAD_OPTS[@]}" \
        --disable-usage-stats

    echo "[entrypoint] waiting for ${NNODES} GPUs to join the Ray cluster ..."
    for _ in $(seq 1 120); do
        if ray status 2>/dev/null | grep -qE "/${NNODES}\.0 GPU"; then
            echo "[entrypoint] Ray cluster full: ${NNODES} GPU"
            break
        fi
        sleep 5
    done
    ray status 2>&1 | tail -20
fi

# ---------------------------------------------------------------------------
# vllm serve コマンド組み立て
# ---------------------------------------------------------------------------
# set -f: SERVED_MODEL_NAME に複数エイリアスを空白区切りで書けるようにしつつ、
# JSON 内の [8] などが glob 展開されるのを防ぐ。
set -f

VLLM_CMD=(
    vllm serve "${MODEL_CONTAINER_PATH}"
    --served-model-name ${SERVED_MODEL_NAME}
    --trust-remote-code
    --host 0.0.0.0
    --port "${HOST_PORT:-8910}"
    --max-model-len "${MAX_MODEL_LEN:-262144}"
    --max-num-seqs "${MAX_NUM_SEQS:-2}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.82}"
    --tensor-parallel-size "${TP_SIZE}"
    --pipeline-parallel-size "${PP_SIZE}"
    --distributed-executor-backend "${DIST_BACKEND}"
)

if [ "${DIST_BACKEND}" = "mp" ]; then
    # Ray を使わない SPMD 起動。head/worker が同じコマンドを rank 違いで叩く。
    VLLM_CMD+=(
        --nnodes "${NNODES}"
        --node-rank "${NODE_RANK}"
        --master-addr "${MASTER_ADDR}"
        --master-port "${MASTER_PORT}"
    )
    [ "${ROLE}" = "worker" ] && VLLM_CMD+=(--headless)
fi

# VLLM_*_ARGS は空白区切りで展開する。
# JSON を渡す場合は値の内側に空白を入れないこと (例: {"cudagraph_mode":"NONE"})。
for _args in "${VLLM_QUANT_ARGS:-}" "${VLLM_PARSER_ARGS:-}" "${VLLM_KV_ARGS:-}" \
             "${VLLM_EXTRA_ARGS:-}"; do
    [ -n "${_args}" ] || continue
    # shellcheck disable=SC2206
    VLLM_CMD+=(${_args})
done
set +f

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
