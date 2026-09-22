#!/usr/bin/env bash
# =============================================================================
# GLM-5.3-Flash-NVFP4 + DFlash2 / dual DGX Spark (GB10, SM121) entrypoint
#
# ROLE=head   -> vllm serve (rank 0, API サーバを持つ)
# ROLE=worker -> vllm serve --headless (rank 1, API なし)
#
# 分散バックエンドは Ray ではなく mp (torch.distributed SPMD)。
# head/worker が同じ `vllm serve` を --nnodes/--node-rank/--master-addr 付きで
# 起動し、MASTER_ADDR:MASTER_PORT で rendezvous する。
#
# 任意パッチ (PREFIX_FIX / ROCE) は compose が /patches:ro で渡してくるものを
# ここで site-packages にコピーする。compose では「ファイルがあればマウント」が
# 書けないため。必須の kpool 修正だけは従来どおり直接 bind-mount。
# =============================================================================
set -euo pipefail

SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages
PATCH_DIR=/patches

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
    MAX_NUM_BATCHED_TOKENS \
    KV_CACHE_MEMORY \
    VLLM_MM_ARGS \
    VLLM_CHAT_TEMPLATE_ARGS \
    ROCE_EXPECT_VLLM

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
# 永続 JIT キャッシュのディレクトリを先に作る。
# 無いと TileLang / Triton が既定の場所に落ちてキャッシュが効かず、
# fp4 CUTLASS GEMM の variant を boot ごとに再コンパイルする (初回 41 分)。
# ---------------------------------------------------------------------------
mkdir -p "${TILELANG_CACHE_DIR:-/root/.cache/tilelang}" \
         "${TRITON_CACHE_DIR:-/root/.triton/cache}" \
         /root/.cache/flashinfer

# ---------------------------------------------------------------------------
# kpool top-k SM121 修正の bind-mount チェック (必須)
#
# イメージ標準の sparse_attn_indexer_kpool.py は persistent_topk カーネルを
# SM 数 78 以上で使うが、GB10 (48 SM / 99KB smem) では ~24K トークン超の
# decode で CTA が超過し RuntimeError -> EngineDeadError になる。
# compose が patches/sparse_attn_indexer_kpool_sm121.py を上書き mount する
# はずなので、ゲートが入っているか確認してない場合は落とす。
# ---------------------------------------------------------------------------
KPOOL_PY="${SITE_PACKAGES}/vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
if [ -f "${KPOOL_PY}" ] && ! grep -q 'multi_processor_count >= 78' "${KPOOL_PY}"; then
    echo "[entrypoint] ERROR: ${KPOOL_PY} に SM121 ゲートがありません。" >&2
    echo "[entrypoint]   patches/sparse_attn_indexer_kpool_sm121.py の bind-mount を確認 (24K ctx 超の decode で engine が死ぬ)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# PREFIX_FIX: #18 prefix-cache 修正 (任意 / 既定 ON)
#
# 公開イメージ sm121-v11-dflash2 は 2026-08-28 ビルドでこの修正を持たない。
# 無いと DFlash2 の draft group が target group の hit 長を潰し、ブロック境界に
# 揃ったプロンプトの再送でも prefix_cache_hits_total が 0 のままになる。
# エージェントのセッションが毎ターン会話全体を再 prefill する
# (13K トークン反復の TTFT 21.1s -> 6.3s / -70%)。
#
# ファイルは ./scripts/build-prefix-fix.sh が生成する (self-check が
# site-packages を要求するのでホストでは当てられない)。
# ---------------------------------------------------------------------------
PREFIX_FIX_SRC="${PATCH_DIR}/kv_cache_coordinator_prefix_fix.py"
PREFIX_FIX_DST="${SITE_PACKAGES}/vllm/v1/core/kv_cache_coordinator.py"
if [ "${PREFIX_FIX:-1}" = "1" ]; then
    if [ ! -f "${PREFIX_FIX_SRC}" ]; then
        echo "[entrypoint] note: prefix-cache 修正 (#18) OFF — ${PREFIX_FIX_SRC} がありません"
        echo "[entrypoint]   生成: ./scripts/build-prefix-fix.sh (2 台とも / prefix cache が 0 hit のままになります)"
    elif ! grep -q '_glm53_is_draft_swa_spec' "${PREFIX_FIX_SRC}"; then
        echo "[entrypoint] ERROR: ${PREFIX_FIX_SRC} にパッチのマーカーがありません (生成失敗の可能性)" >&2
        echo "[entrypoint]   ./scripts/build-prefix-fix.sh を再実行するか PREFIX_FIX=0 で起動" >&2
        exit 1
    else
        cp "${PREFIX_FIX_SRC}" "${PREFIX_FIX_DST}"
        echo "[entrypoint] prefix-cache 修正 (#18) ON"
    fi
else
    echo "[entrypoint] note: prefix-cache 修正 (#18) は PREFIX_FIX=0 で無効"
fi

# ---------------------------------------------------------------------------
# ROCE: b12x RoCEnante one-shot RoCE all-reduce (任意 / 既定 OFF)
#
# aggregate +5〜18% (C1-C6)、単発 decode はほぼ変化なし。NCCL の all-reduce を
# RDMA の one-shot に置き換える。vLLM のファイルを 5 本まるごと差し替えるので、
# イメージの vLLM ツリーが違うと黙って壊れる -> バージョンを照合してから入れる。
#
# patches/roce/ の作り方は patches/roce/README.md / scripts/build-roce-bundle.sh。
# ---------------------------------------------------------------------------
ROCE_SRC="${PATCH_DIR}/roce"
if [ "${ROCE:-0}" = "1" ]; then
    if [ ! -d "${ROCE_SRC}/b12x" ]; then
        echo "[entrypoint] note: RoCE all-reduce OFF (NCCL のまま) — ${ROCE_SRC}/b12x がありません"
        echo "[entrypoint]   作り方: ./scripts/build-roce-bundle.sh (patches/roce/README.md)"
    else
        # イメージの vLLM バージョン照合。import せずに dist-info / _version.py を読む
        # (import vllm は CUDA を掴むので起動前にやりたくない)。
        ROCE_WANT="${ROCE_EXPECT_VLLM:-0.1.dev20051+g487ecf187}"
        ROCE_HAVE=$(sed -n 's/^Version: //p' "${SITE_PACKAGES}"/vllm-*.dist-info/METADATA 2>/dev/null | head -1)
        if [ -z "${ROCE_HAVE}" ]; then
            ROCE_HAVE=$(sed -n "s/^version *= *['\"]\\(.*\\)['\"].*/\\1/p" "${SITE_PACKAGES}/vllm/_version.py" 2>/dev/null | head -1)
        fi
        if [ -z "${ROCE_HAVE}" ]; then
            echo "[entrypoint] ERROR: vLLM のバージョンが読めないので RoCE パッチを当てられません" >&2
            echo "[entrypoint]   ROCE=0 で起動するか、ROCE_EXPECT_VLLM を空でなく設定して原因を調べる" >&2
            exit 1
        fi
        if [ "${ROCE_HAVE}" != "${ROCE_WANT}" ]; then
            echo "[entrypoint] ERROR: RoCE パッチの対象ツリーと一致しません" >&2
            echo "[entrypoint]   image の vLLM = ${ROCE_HAVE} / パッチの想定 = ${ROCE_WANT}" >&2
            echo "[entrypoint]   vllm のファイル 5 本を丸ごと置換するので、不一致は黙って壊れます。" >&2
            echo "[entrypoint]   ROCE=0 にするか、patches/roce/ をこのツリー向けに作り直すこと。" >&2
            exit 1
        fi

        cp -r "${ROCE_SRC}/b12x" "${SITE_PACKAGES}/b12x"
        cp -r "${ROCE_SRC}/b12x-1.3.0.dist-info" "${SITE_PACKAGES}/b12x-1.3.0.dist-info"
        cp -r "${ROCE_SRC}/b12x-roce" /opt/b12x-roce
        cp "${ROCE_SRC}/b12x_roce_all_reduce.py" \
           "${SITE_PACKAGES}/vllm/distributed/device_communicators/b12x_roce_all_reduce.py"
        cp "${ROCE_SRC}/cuda_communicator.py" \
           "${SITE_PACKAGES}/vllm/distributed/device_communicators/cuda_communicator.py"
        cp "${ROCE_SRC}/parallel_state.py" "${SITE_PACKAGES}/vllm/distributed/parallel_state.py"
        cp "${ROCE_SRC}/envs.py"           "${SITE_PACKAGES}/vllm/envs.py"
        cp "${ROCE_SRC}/gpu_worker.py"     "${SITE_PACKAGES}/vllm/v1/worker/gpu_worker.py"
        export VLLM_ENABLE_ROCE_ALLREDUCE=1
        echo "[entrypoint] RoCE all-reduce ON (vllm=${ROCE_HAVE} hca=${B12X_ROCE_HCA:-?} gid=${B12X_ROCE_GID_INDEX:-?})"
        echo "[entrypoint]   起動後に 'RoCEnante all-reduce is live' がログに出ることを確認"
    fi
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
# KV プールを固定する場合のガード
#
# --kv-cache-memory を渡すと vLLM はメモリプロファイリングを丸ごとスキップする。
# つまり --max-num-batched-tokens の活性化ピークを誰も検証しない。上流の実測で
# mnbt 16384 は両ノードが NVRM NV_ERR_NO_MEMORY で死んだ (KV 8 GiB 固定時)。
# 8192 が検証済みの上限なので、それを超える組み合わせは起動前に落とす。
# ---------------------------------------------------------------------------
if [ -n "${KV_CACHE_MEMORY:-}" ] && [ "${KV_CACHE_MEMORY}" != "0" ]; then
    if [ "${MAX_NUM_BATCHED_TOKENS:-8192}" -gt 8192 ]; then
        echo "[entrypoint] ERROR: KV_CACHE_MEMORY 固定時に MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS} は危険です。" >&2
        echo "[entrypoint]   KV を固定すると vLLM はメモリプロファイリングをスキップするので、" >&2
        echo "[entrypoint]   活性化ピークが検証されません (上流実測: 16384 は両ノードで NVRM OOM)。" >&2
        echo "[entrypoint]   8192 以下にするか、KV_CACHE_MEMORY=0 でプロファイラ任せに戻すこと。" >&2
        exit 1
    fi
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
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --tensor-parallel-size "${TP_SIZE}"
    --distributed-executor-backend mp
    --nnodes "${NNODES}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
)

if [ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]; then
    VLLM_CMD+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
fi

# KV プールのバイト数を固定する (既定 8 GiB / 上流 2026-09-18)。
# 0 か空ならフラグを渡さず vLLM のプロファイラに任せる。
if [ -n "${KV_CACHE_MEMORY:-}" ] && [ "${KV_CACHE_MEMORY}" != "0" ]; then
    VLLM_CMD+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
fi

[ "${ROLE}" = "worker" ] && VLLM_CMD+=(--headless)

# VLLM_*_ARGS は空白区切りで展開する。
# JSON を渡す場合は値の内側に空白を入れないこと (例: {"method":"dflash"})。
#
#   VLLM_PARSER_ARGS        : reasoning / tool-call パーサ
#   VLLM_KV_ARGS            : KV キャッシュ関連
#   VLLM_MM_ARGS            : マルチモーダル (--limit-mm-per-prompt)
#   VLLM_CHAT_TEMPLATE_ARGS : chat template / 既定 kwargs
#   VLLM_EXTRA_ARGS         : その他 (speculative-config など)
for _args in "${VLLM_PARSER_ARGS:-}" "${VLLM_KV_ARGS:-}" "${VLLM_MM_ARGS:-}" \
             "${VLLM_CHAT_TEMPLATE_ARGS:-}" "${VLLM_EXTRA_ARGS:-}"; do
    [ -n "${_args}" ] || continue
    # shellcheck disable=SC2206
    VLLM_CMD+=(${_args})
done
set +f

echo "[entrypoint] exec: ${VLLM_CMD[*]}"
exec "${VLLM_CMD[@]}"
