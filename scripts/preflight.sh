#!/usr/bin/env bash
# =============================================================================
# 起動前チェック。3 台すべてで実行する。
#
#   ./scripts/preflight.sh
#
# ここで赤が出た状態で起動すると、だいたい 5〜10 分待たされた挙句
# NCCL の err 110 か OOM-kill で死ぬ。
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
env_get() { sed -n "s/^$1=//p" .env | tail -1; }
for k in VLLM_IMAGE VLLM_BASE_IMAGE MODEL_PATH OOB_IF_NAME \
         NODE0_MGMT_IP NODE1_MGMT_IP NODE2_MGMT_IP \
         NCCL_IB_HCA NCCL_IB_ADDR_RANGE TP_SIZE PP_SIZE NNODES; do
    printf -v "$k" '%s' "$(env_get "$k")"
done
: "${TP_SIZE:=3}"; : "${PP_SIZE:=1}"; : "${NNODES:=3}"

echo "== ホスト =="
echo "  hostname: $(hostname)  arch: $(uname -m)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  GPU: /'
echo "  topology: TP=${TP_SIZE} PP=${PP_SIZE} / ${NNODES} nodes"
if [ "$((TP_SIZE * PP_SIZE))" -ne "${NNODES}" ]; then
    ng "TP_SIZE * PP_SIZE (${TP_SIZE}x${PP_SIZE}) が NNODES (${NNODES}) と一致しません"
fi

echo
echo "== 管理 NIC (Ray / NCCL bootstrap) =="
# 3 ノードメッシュでは RoCE が全ノード間で直結していないので、制御面は
# 全ノードが到達できる 10GbE を使う必要がある。
OOB_IPS=$(ip -4 -o addr show dev "${OOB_IF_NAME}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
if [ -z "${OOB_IPS}" ]; then
    ng "${OOB_IF_NAME} に IPv4 が付いていません"
else
    ok "${OOB_IF_NAME} = ${OOB_IPS}"
    if ! echo "${OOB_IPS}" | grep -qx -e "${NODE0_MGMT_IP}" -e "${NODE1_MGMT_IP}" -e "${NODE2_MGMT_IP}"; then
        ng "実 IP が .env の NODE{0,1,2}_MGMT_IP のどれとも一致しません"
    fi
fi
for PEER in "${NODE0_MGMT_IP}" "${NODE1_MGMT_IP}" "${NODE2_MGMT_IP}"; do
    echo "${OOB_IPS}" | grep -qx "${PEER}" && continue
    if ping -c 2 -W 2 "${PEER}" >/dev/null 2>&1; then
        ok "peer ${PEER} に疎通"
    else
        ng "peer ${PEER} に ping が通りません"
    fi
done

echo
echo "== RoCE メッシュ (データ面) =="
# Spark は QSFP 1 ポートあたり 2 本の twin IF を見せる。3 ノードの三角メッシュでは
# 両ポートを使うので、4 本すべてが up かつ IPv4 付きになっているのが正常。
MESH_IFS=$(ls /sys/class/net 2>/dev/null | grep -E '^en(p1|P2p1)s0f[01]np[01]$' | sort)
if [ -z "${MESH_IFS}" ]; then
    ng "CX7 のインターフェースが見つかりません"
fi
ROCE_IPS=""
for IF in ${MESH_IFS}; do
    STATE=$(cat "/sys/class/net/${IF}/operstate" 2>/dev/null)
    MTU=$(cat "/sys/class/net/${IF}/mtu" 2>/dev/null)
    IP4=$(ip -4 -o addr show dev "${IF}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | tr '\n' ' ')
    ROCE_IPS="${ROCE_IPS} ${IP4}"
    if [ "${STATE}" != "up" ]; then
        ng "${IF} が ${STATE} (ケーブルを確認 / 3 ノードは両ポート使用)"
    elif [ -z "${IP4}" ]; then
        ng "${IF} は up だが IPv4 が付いていません (netplan を確認)"
    elif [ "${MTU:-0}" -lt 9000 ]; then
        warn "${IF} = ${IP4}(MTU ${MTU}) — 9000 に上げると RoCE path MTU が 4096 になる"
    else
        ok "${IF} = ${IP4}(MTU ${MTU})"
    fi
done
# 電源断のたびに CX7 のランタイム MTU は 1500 に戻る。netplan が効いていても油断しない。
if ip -d link show 2>/dev/null | grep -q .; then :; fi

echo
echo "== RDMA / GID =="
if [ -z "${NCCL_IB_HCA}" ]; then
    ng ".env の NCCL_IB_HCA が空です"
else
    IFS=',' read -ra HCAS <<< "${NCCL_IB_HCA}"
    for HCA in "${HCAS[@]}"; do
        if ibv_devinfo -d "${HCA}" 2>/dev/null | grep -q PORT_ACTIVE; then
            AMTU=$(ibv_devinfo -d "${HCA}" 2>/dev/null | awk '/active_mtu/ {print $2; exit}')
            ok "${HCA} PORT_ACTIVE (path MTU ${AMTU:-?})"
        else
            ng "${HCA} が PORT_ACTIVE ではありません"
        fi
    done
fi
[ -e /dev/infiniband/uverbs0 ] && ok "/dev/infiniband あり" || ng "/dev/infiniband がありません"

# GID index を固定すると subnet-aware routing が無効になり、隣人ごとに
# 正しい HCA を選べずクロスペアリング -> ibv_modify_qp err 110 で死ぬ。
# 2 ノード直結の設定 (GID_INDEX=3) を持ち込んでいないかを見る。
if grep -qsE '^\s*NCCL_IB_GID_INDEX\s*=' .env; then
    ng ".env に NCCL_IB_GID_INDEX があります — 3 ノードメッシュでは【消すこと】"
    echo "       固定すると subnet-aware routing が効かず err 110 になります"
else
    ok "NCCL_IB_GID_INDEX は未設定 (メッシュではこれが正しい)"
fi

# NCCL は NCCL_IB_ADDR_RANGE の CIDR を見て隣人ごとの GID を選ぶ。
# RoCE 側アドレスがこの範囲から外れていると選択に失敗する。
if [ -z "${NCCL_IB_ADDR_RANGE}" ]; then
    ng ".env の NCCL_IB_ADDR_RANGE が空です"
else
    OUT=$(python3 - "${NCCL_IB_ADDR_RANGE}" ${ROCE_IPS} <<'PY'
import ipaddress, sys
net = ipaddress.ip_network(sys.argv[1], strict=False)
bad = [ip for ip in sys.argv[2:] if ipaddress.ip_address(ip) not in net]
print(" ".join(bad))
PY
)
    if [ -n "${OUT}" ]; then
        ng "NCCL_IB_ADDR_RANGE=${NCCL_IB_ADDR_RANGE} に含まれない RoCE IP: ${OUT}"
    else
        ok "NCCL_IB_ADDR_RANGE=${NCCL_IB_ADDR_RANGE} が RoCE IP を全部カバー"
    fi
fi

echo
echo "== モデル =="
if [ -d "${MODEL_PATH}" ]; then
    N=$(ls "${MODEL_PATH}"/*.safetensors 2>/dev/null | wc -l)
    if [ "${N}" -ge 40 ]; then
        ok "${MODEL_PATH} (${N} shard, $(du -sh "${MODEL_PATH}" | cut -f1))"
    else
        ng "${MODEL_PATH} の shard 数が ${N} です — scripts/fetch-model.sh を再実行"
    fi
else
    ng "${MODEL_PATH} がありません — ./scripts/fetch-model.sh"
fi

echo
echo "== メモリ =="
# 重み 250GB / 3 = 77.6GiB + ランタイム約 12GiB + KV
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
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
        && ok "イメージビルド済み (${VLLM_IMAGE})" \
        || warn "イメージ未ビルド — sudo docker compose --profile head build"
else
    warn "sudo docker が非対話で叩けません — 以下を手で確認すること:"
    echo "       sudo docker info | grep -i runtimes        # nvidia が出ること"
    echo "       sudo docker image inspect ${VLLM_IMAGE}"
fi

echo
[ "${RC}" -eq 0 ] && echo "==> preflight PASS" || echo "==> preflight FAIL"
exit "${RC}"
