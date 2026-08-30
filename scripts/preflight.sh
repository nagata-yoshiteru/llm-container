#!/usr/bin/env bash
# =============================================================================
# 起動前チェック。head / worker 両方で実行する。
#
#   ./scripts/preflight.sh
#
# ここで赤が出た状態で起動すると、だいたい 5〜10 分待たされた挙句
# NCCL の "unhandled system error" か OOM-kill で死ぬ。
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
for k in VLLM_IMAGE MODEL_PATH HEAD_ROCE_IP WORKER_ROCE_IP ROCE_IF_NAME IB_HCA_NAME NCCL_IB_GID_INDEX; do
    printf -v "$k" '%s' "$(env_get "$k")"
done
: "${NCCL_IB_GID_INDEX:=3}"

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
else
    ng "${MODEL_PATH} がありません — ./scripts/fetch-model.sh"
fi
DRAFT_DIR=$(sed -n 's/^DRAFT_PATH=//p' .env | tail -1)
if [ -n "${DRAFT_DIR}" ] && [ ! -f "${DRAFT_DIR}/config.json" ]; then
    warn "${DRAFT_DIR} (DFlash2 drafter) が未取得 — presets/dflash2.env を使う場合だけ ./scripts/fetch-model.sh draft"
fi

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
echo "== メモリ =="
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
# 重み 約 198GB / TP2 = 約 99GB + MTP head 約 4GB + KV + ランタイム
if [ "${AVAIL_GB}" -ge 105 ]; then
    ok "MemAvailable ${AVAIL_GB} GB"
elif [ "${AVAIL_GB}" -ge 95 ]; then
    warn "MemAvailable ${AVAIL_GB} GB — ギリギリ。他のコンテナを止めて sync && drop_caches 推奨"
else
    ng "MemAvailable ${AVAIL_GB} GB — 足りません。他のコンテナを止めるか再起動してください"
fi
RUNNING=$(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')
[ -n "${RUNNING}" ] && warn "rootless docker で起動中: ${RUNNING}"

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
exit "${RC}"
