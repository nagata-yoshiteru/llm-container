#!/usr/bin/env bash
# =============================================================================
# 起動前チェック。head / worker 両方で実行する。
#
#   ./scripts/preflight.sh
#
# ここで赤が出た状態で起動すると、だいたい 5〜10 分待たされた挙句
# NCCL の "unhandled system error" か OOM-kill で死ぬ。
#
# ベンチを取るなら ./scripts/gputest.sh も 2 台で回すこと (クロッククランプ検出)。
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

RC=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
ng()   { printf '  \033[31mNG\033[0m   %s\n' "$*"; RC=1; }

if [ ! -f .env ]; then
    ng ".env がありません (cp .env.example .env)"
    exit 1
fi

# .env は docker compose の書式 (クォートなし・空白を含む値あり) なので
# source せず、必要なキーだけリテラルに読む。
env_get() {
    sed -n "s/^$1=//p" .env | tail -1
}
for k in VLLM_IMAGE MODEL_PATH DRAFT_PATH HEAD_ROCE_IP WORKER_ROCE_IP \
         ROCE_IF_NAME IB_HCA_NAME NCCL_IB_GID_INDEX \
         KV_CACHE_MEMORY MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION \
         VLLM_EXTRA_ARGS PREFIX_FIX ROCE; do
    printf -v "$k" '%s' "$(env_get "$k")"
done
: "${NCCL_IB_GID_INDEX:=3}"
: "${PREFIX_FIX:=1}"
: "${ROCE:=0}"

echo "== ホスト =="
echo "  hostname: $(hostname)  arch: $(uname -m)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  GPU: /'

echo
echo "== RoCE リンク =="
LOCAL_IPS=$(ip -4 -o addr show dev "${ROCE_IF_NAME}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
if [ -z "${LOCAL_IPS}" ]; then
    ng "${ROCE_IF_NAME} に IPv4 が付いていません (netplan / ケーブルを確認)"
else
    ok "${ROCE_IF_NAME} = ${LOCAL_IPS}"
    if ! echo "${LOCAL_IPS}" | grep -qx -e "${HEAD_ROCE_IP}" -e "${WORKER_ROCE_IP}"; then
        ng "実 IP が .env の HEAD_ROCE_IP(${HEAD_ROCE_IP}) / WORKER_ROCE_IP(${WORKER_ROCE_IP}) と一致しません"
    fi
fi

for PEER in "${HEAD_ROCE_IP}" "${WORKER_ROCE_IP}"; do
    if echo "${LOCAL_IPS}" | grep -qx "${PEER}"; then continue; fi
    if ping -c 2 -W 2 "${PEER}" >/dev/null 2>&1; then
        ok "peer ${PEER} に疎通"
    else
        ng "peer ${PEER} に ping が通りません"
    fi
done

echo
echo "== RDMA / GID =="
if ibv_devinfo -d "${IB_HCA_NAME}" 2>/dev/null | grep -q PORT_ACTIVE; then
    ok "${IB_HCA_NAME} PORT_ACTIVE"
else
    ng "${IB_HCA_NAME} が PORT_ACTIVE ではありません (ケーブルが刺さっている方の HCA か確認)"
fi
GID_LINE=$(show_gids 2>/dev/null | awk -v d="${IB_HCA_NAME}" -v i="${NCCL_IB_GID_INDEX}" '$1==d && $3==i')
if [ -n "${GID_LINE}" ] && echo "${GID_LINE}" | grep -q 'v2'; then
    ok "GID index ${NCCL_IB_GID_INDEX} = RoCEv2 ($(echo "${GID_LINE}" | awk '{print $5}'))"
else
    ng "${IB_HCA_NAME} の GID index ${NCCL_IB_GID_INDEX} が RoCEv2/IPv4 ではありません: show_gids で確認"
fi
[ -e /dev/infiniband/uverbs0 ] && ok "/dev/infiniband あり" || ng "/dev/infiniband がありません"

MTU=$(cat "/sys/class/net/${ROCE_IF_NAME}/mtu" 2>/dev/null)
ACTIVE_MTU=$(ibv_devinfo -d "${IB_HCA_NAME}" 2>/dev/null | awk '/active_mtu/ {print $2; exit}')
if [ "${MTU:-0}" -ge 9000 ]; then
    ok "MTU ${MTU} (RoCE path MTU ${ACTIVE_MTU:-?})"
else
    warn "MTU ${MTU:-?} / RoCE path MTU ${ACTIVE_MTU:-?} — 9000 に上げると path MTU が 4096 になる"
    warn "  手順は README の「セットアップ 4. MTU を 9000 に上げる」。両ノードで揃えること"
fi

echo
echo "== モデル =="
if [ -d "${MODEL_PATH}" ]; then
    N=$(ls "${MODEL_PATH}"/model-*.safetensors 2>/dev/null | wc -l)
    if [ "${N}" -eq 10 ]; then
        ok "${MODEL_PATH} (10 shard, $(du -sh "${MODEL_PATH}" | cut -f1))"
    else
        ng "${MODEL_PATH} の shard 数が ${N} です (10 のはず) — scripts/fetch-model.sh を再実行"
    fi
    [ -f "${MODEL_PATH}/model_mtp.safetensors" ] \
        && ok "model_mtp.safetensors あり (MTP head)" \
        || ng "model_mtp.safetensors がありません (MTP 用 draft head / 約 7.6GB)"

    # ModelOpt 量子化ガード。
    # ModelOpt NVFP4 は intermittent に token ID を壊す (vLLM #54150)。
    # 英文ではほぼ見えないが、tool-call ブロックの中で壊れると parser が
    # desync して生成が反復ロックに入る。Hangul プローブで U+FFFD が
    # ModelOpt 4/9/8 個 vs RedHatAI (compressed-tensors) 0/0/0。
    if [ -f "${MODEL_PATH}/config.json" ]; then
        QUANT=$(python3 -c "import json,sys;print(json.load(open('${MODEL_PATH}/config.json')).get('quantization_config',{}).get('quant_method',''))" 2>/dev/null)
        if [ "${QUANT}" = "modelopt" ]; then
            if [ "${ALLOW_MODELOPT:-0}" = "1" ]; then
                warn "quant_method=modelopt (ALLOW_MODELOPT=1 で許可) — token 破損のリスクあり"
            else
                ng "${MODEL_PATH} が ModelOpt ビルドです (quant_method=modelopt)"
                echo "       ModelOpt NVFP4 は token ID を壊す (vLLM #54150)。tool-call の中で壊れると parser が desync する。"
                echo "       RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors) を使うこと。"
                echo "       承知の上で進めるなら ALLOW_MODELOPT=1 ./scripts/preflight.sh"
            fi
        elif [ -n "${QUANT}" ]; then
            ok "quant_method=${QUANT}"
        fi
    fi

    # vision 用の chat template。checkpoint 同梱のものを使う (--chat-template は
    # 渡さない)。上流 repo の chat_template_mm.jinja は古い版なので入れない。
    [ -f "${MODEL_PATH}/chat_template.jinja" ] \
        && ok "chat_template.jinja あり (vision の placeholder 発行)" \
        || warn "chat_template.jinja がありません — image リクエストが 500 になります"
else
    ng "${MODEL_PATH} がありません — ./scripts/fetch-model.sh"
fi

# DFlash2 が既定レシピなので drafter は必須。MTP 構成 (presets/mtp4.env) なら不要。
if echo "${VLLM_EXTRA_ARGS}" | grep -q 'dflash'; then
    if [ -n "${DRAFT_PATH}" ] && [ -f "${DRAFT_PATH}/config.json" ]; then
        ok "${DRAFT_PATH} (DFlash2 drafter)"
    else
        ng "${DRAFT_PATH:-DRAFT_PATH} に DFlash2 drafter がありません — ./scripts/fetch-model.sh draft"
        echo "       .env が DFlash2 を要求しています (VLLM_EXTRA_ARGS の method=dflash)。"
        echo "       MTP-4 に戻すなら presets/mtp4.env を重ねて起動すること。"
    fi
else
    warn ".env が DFlash2 を使っていません (MTP 構成)。既定は DFlash2 (約 2.15x)"
fi

echo
echo "== vLLM 引数の整合性 =="
if [ -n "${KV_CACHE_MEMORY}" ] && [ "${KV_CACHE_MEMORY}" != "0" ]; then
    ok "KV_CACHE_MEMORY=${KV_CACHE_MEMORY} ($((KV_CACHE_MEMORY / 1024 / 1024 / 1024)) GiB 固定)"
    # 固定すると vLLM はメモリプロファイリングをスキップするので、mnbt の
    # 活性化ピークを誰も検証しない。上流実測で 16384 は両ノードで NVRM OOM。
    if [ "${MAX_NUM_BATCHED_TOKENS:-8192}" -gt 8192 ] 2>/dev/null; then
        ng "KV 固定 + MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS} は危険 (8192 以下にすること)"
        echo "       KV を固定すると vLLM はメモリプロファイリングをスキップするので、"
        echo "       活性化ピークが検証されません (上流実測: 16384 は両ノードで NVRM OOM)。"
    else
        ok "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-<既定 8192>}"
    fi
else
    warn "KV_CACHE_MEMORY 未固定 (プロファイラ任せ)。既定は 8589934592 = 8 GiB で プール +33%"
fi
case "${GPU_MEMORY_UTILIZATION:-0.85}" in
    0.85) ok "GPU_MEMORY_UTILIZATION=0.85" ;;
    *)    warn "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} — 検証値は 0.85 (0.78〜0.80 は KV が枯れ、0.87 は到達できないノードがある)" ;;
esac

echo
echo "== SM121 kpool パッチ =="
if [ -f patches/sparse_attn_indexer_kpool_sm121.py ] \
   && grep -q 'multi_processor_count >= 78' patches/sparse_attn_indexer_kpool_sm121.py; then
    ok "patches/sparse_attn_indexer_kpool_sm121.py (SM121 ゲート入り)"
else
    ng "patches/sparse_attn_indexer_kpool_sm121.py が無い、または SM121 ゲートがない"
    echo "       ~24K トークン超の decode で engine が死ぬ (persistent_topk)"
fi

echo
echo "== 任意パッチ =="
# #18 prefix-cache 修正。無いとエージェントのセッションが毎ターン会話全体を
# 再 prefill する (13K トークン反復の TTFT 21.1s -> 6.3s)。
if [ "${PREFIX_FIX}" = "1" ]; then
    if [ -f patches/kv_cache_coordinator_prefix_fix.py ] \
       && grep -q '_glm53_is_draft_swa_spec' patches/kv_cache_coordinator_prefix_fix.py; then
        ok "prefix-cache 修正 (#18) 生成済み"
    else
        warn "PREFIX_FIX=1 だが patches/kv_cache_coordinator_prefix_fix.py が未生成 — 自動で OFF になります"
        echo "       生成: ./scripts/build-prefix-fix.sh   (2 台とも / prefix cache が 0 hit のままになる)"
    fi
else
    warn "PREFIX_FIX=0 — prefix cache が効きません (エージェント用途では毎ターン全再 prefill)"
fi
# b12x RoCE all-reduce
if [ "${ROCE}" = "1" ]; then
    if [ -d patches/roce/b12x ]; then
        ok "RoCE all-reduce の bundle あり"
    else
        warn "ROCE=1 だが patches/roce/b12x がありません — 自動で NCCL にフォールバックします"
        echo "       作り方: SRC_IMAGE=... ./scripts/build-roce-bundle.sh (patches/roce/README.md)"
    fi
fi

echo
echo "== メモリ =="
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
SWAPPINESS=$(sysctl -n vm.swappiness 2>/dev/null || echo '?')

# メモリ儀式 (README「起動前にやるメモリ儀式」): GB10 の NVRM は MemFree で
# 割当を判定し page cache は強制 reclamation されない。swappiness が既定の
# ままだと shard ロード中に UVM livelock / worker 死 (上流実測)。
# 未設定なら自動で直す (sudo -n できない場合は手動コマンドを出す)。
if [ "${SWAPPINESS}" != "0" ] || [ "${AVAIL_GB}" -lt 110 ]; then
    if sudo -n true 2>/dev/null; then
        warn "swappiness=${SWAPPINESS} / MemAvailable ${AVAIL_GB} GB — リセットを自動実行"
        sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
        [ "${SWAPPINESS}" != "0" ] && sudo -n sysctl -w vm.swappiness=0
    else
        warn "swappiness=${SWAPPINESS} / MemAvailable ${AVAIL_GB} GB — リセットが要ります (sudo 非対話不可):"
        echo "       sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && sudo sysctl -w vm.swappiness=0"
    fi
fi

# リセット後に実状態を再読み (自動実行できなくても嘘の PASS を出さない)
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
SWAPPINESS=$(sysctl -n vm.swappiness 2>/dev/null || echo '?')
if [ "${SWAPPINESS}" != "0" ]; then
    ng "vm.swappiness=${SWAPPINESS} — swap livelock の危険。0 にしてから起動すること"
fi

# 重み 約 198GB / TP2 = 約 99GB + drafter + KV 8GiB + ランタイム
if [ "${AVAIL_GB}" -ge 105 ]; then
    ok "MemAvailable ${AVAIL_GB} GB"
elif [ "${AVAIL_GB}" -ge 95 ]; then
    warn "MemAvailable ${AVAIL_GB} GB — ギリギリ。他のコンテナを止めて再実行"
else
    ng "MemAvailable ${AVAIL_GB} GB — 足りません。他のコンテナを止めるか再起動してください"
fi
RUNNING=$(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')
[ -n "${RUNNING}" ] && warn "rootless docker で起動中: ${RUNNING}"

# page-cache flusher。起動中ずっと回しておく必要がある (閾値式は期限切れ後に
# ノードを殺した事例あり / 上流 2026-09-18)。
if pgrep -f 'scripts/flusher.sh' >/dev/null 2>&1; then
    ok "scripts/flusher.sh 稼働中"
else
    warn "scripts/flusher.sh が動いていません — コンテナ起動後に 2 台とも回すこと"
    echo "       ./scripts/flusher.sh &   (MemFree を空け続けないと KV slab の割当が落ちる)"
fi

echo
echo "== rootful docker =="
[ -e /dev/nvidia0 ] && ok "/dev/nvidia0 あり" || ng "/dev/nvidia0 がありません (ドライバを確認)"

# rootful daemon に nvidia ランタイムが登録されているか。
# rootless 側 (~/.config/docker/daemon.json) に登録されていても rootful には効かない。
# 未登録だとコンテナ内で NVML が初期化できず、vLLM が
# "Failed to infer device type" で即死する。
if grep -qs nvidia /etc/docker/daemon.json; then
    ok "rootful daemon に nvidia ランタイム登録済み"
else
    ng "/etc/docker/daemon.json に nvidia ランタイムがありません"
    echo "       sudo nvidia-ctk runtime configure --runtime=docker"
    echo "       sudo systemctl restart docker"
fi

# no-cgroups=true は rootless docker で GPU を使うのに必須だが、rootful では
# device cgroup の許可リストが更新されなくなる。compose 側で /dev/nvidia* を
# 明示的に渡してあるのでそれで通るが、通らない場合は privileged: true が要る。
if grep -qsE '^\s*no-cgroups\s*=\s*true' /etc/nvidia-container-runtime/config.toml; then
    warn "no-cgroups=true (rootless 用の設定)。rootful では /dev/nvidia* の明示渡しが必要"
    warn "  compose の devices: で対応済み。それでも NVML が死ぬなら privileged: true を足す"
fi

if sudo -n docker info >/dev/null 2>&1; then
    ok "sudo docker 利用可"
    sudo -n docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia' \
        && ok "docker info に nvidia ランタイムあり" \
        || ng "docker info に nvidia ランタイムがありません"
    sudo -n docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 \
        && ok "イメージ取得済み" \
        || warn "イメージ未取得 — sudo docker pull ${VLLM_IMAGE}"
else
    warn "sudo docker が非対話で叩けません — 以下を手で確認すること:"
    echo "       sudo docker info | grep -i runtimes        # nvidia が出ること"
    echo "       sudo docker run --rm --gpus all ${VLLM_IMAGE} nvidia-smi"
fi

echo
[ "${RC}" -eq 0 ] && echo "==> preflight PASS" || echo "==> preflight FAIL"
echo "    ベンチを取るなら ./scripts/gputest.sh も (クランプしたノードは 2.5 倍遅い)"
exit "${RC}"
