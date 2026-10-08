#!/usr/bin/env bash
# =============================================================================
# 起動前チェック。head / worker1 / worker2 / worker3 の 4 台すべてで実行する。
#
#   ./scripts/preflight.sh
#
# ここで赤が出た状態で起動すると、だいたい 5〜10 分待たされた挙句
# NCCL の "unhandled system error" か OOM-kill で死ぬ。
# GPU_MEMORY_UTILIZATION は起動時の空きを超えると即落ちるので、== メモリ == は必ず見ること。
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
for k in VLLM_IMAGE MODEL_PATH MODEL_CONTAINER_PATH TP_SIZE NNODES \
         HEAD_ROCE_IP WORKER1_ROCE_IP WORKER2_ROCE_IP WORKER3_ROCE_IP \
         ROCE_IF_NAME IB_HCA_NAME NCCL_IB_GID_INDEX GPU_MEMORY_UTILIZATION; do
    printf -v "$k" '%s' "$(env_get "$k")"
done
# ROCE_IF_NAME / IB_HCA_NAME は dual-HCA だとカンマ区切りになる
# (例: enp1s0f0np0,enP2p1s0f0np0)。配列に割っておく。
IFS=',' read -ra ROCE_IFS <<< "${ROCE_IF_NAME}"
IFS=',' read -ra IB_HCAS  <<< "${IB_HCA_NAME}"

# NCCL_IB_GID_INDEX は dual-HCA では「意図的に未設定」が正解なので、
# 空を既定値で埋めない (埋めると固定してあるかのように見えてしまう)。

# 3 ノード分の peer (自分以外) を作っておく。
ALL_PEERS=("${HEAD_ROCE_IP}" "${WORKER1_ROCE_IP}" "${WORKER2_ROCE_IP}" "${WORKER3_ROCE_IP}")
PEER_OCTS=()
for _p in "${ALL_PEERS[@]}"; do PEER_OCTS+=("${_p##*.}"); done

echo "== トポロジ =="
echo "  TP=${TP_SIZE} NNODES=${NNODES} — glm5_next は heads=64 / vocab=154880 なので TP は 2 か 4"
if [ "${TP_SIZE}" != "4" ] || [ "${NNODES}" != "4" ]; then
    ng "TP_SIZE=${TP_SIZE} / NNODES=${NNODES}。この構成は TP=4 / 4 ノード固定 (entrypoint が落とす)"
else
    ok "TP=4 / 4 ノード (--nnodes 4, mp backend)"
fi

echo
echo "== ホスト =="
echo "  hostname: $(hostname)  arch: $(uname -m)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  GPU: /'

echo
echo "== RoCE リンク =="
[ "${#ROCE_IFS[@]}" -gt 1 ] && echo "  (dual-HCA: ${#ROCE_IFS[@]} 本構成)"
LOCAL_IPS=""
for IFN in "${ROCE_IFS[@]}"; do
    IPS=$(ip -4 -o addr show dev "${IFN}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
    if [ -z "${IPS}" ]; then
        ng "${IFN} に IPv4 が付いていません (netplan / スイッチのポートを確認)"
    else
        ok "${IFN} = ${IPS}"
        LOCAL_IPS="${LOCAL_IPS}${LOCAL_IPS:+$'\n'}${IPS}"
    fi
done
if [ -n "${LOCAL_IPS}" ] && \
   ! echo "${LOCAL_IPS}" | grep -qx -e "${HEAD_ROCE_IP}" -e "${WORKER1_ROCE_IP}" -e "${WORKER2_ROCE_IP}" -e "${WORKER3_ROCE_IP}"; then
    warn "どの IF の IP も .env の HEAD/WORKER1..3_ROCE_IP と一致しません。"
    warn "  この機体が .env のどの役割か確認すること (compose の profile と IP が対応する)。"
fi

for PEER in "${ALL_PEERS[@]}"; do
    if echo "${LOCAL_IPS}" | grep -qx "${PEER}"; then continue; fi
    if ping -c 2 -W 2 "${PEER}" >/dev/null 2>&1; then
        ok "peer ${PEER} に疎通"
    else
        ng "peer ${PEER} に ping が通りません"
    fi
done

echo
echo "== RDMA / GID =="
for HCA in "${IB_HCAS[@]}"; do
    if ibv_devinfo -d "${HCA}" 2>/dev/null | grep -q PORT_ACTIVE; then
        ok "${HCA} PORT_ACTIVE"
    else
        ng "${HCA} が PORT_ACTIVE ではありません (ケーブルが刺さっている方の HCA か確認)"
    fi
done

if [ -z "${NCCL_IB_GID_INDEX}" ]; then
    ok "GID index は未固定 (NCCL に選ばせる) — dual-HCA ではこれが正解"
    # 参考情報として、各 HCA の RoCEv2/IPv4 の行だけ出しておく
    for HCA in "${IB_HCAS[@]}"; do
        show_gids 2>/dev/null | awk -v d="${HCA}" '$1==d && /v2/ && $5 ~ /^[0-9]+\./ {
            printf "       %s index %s -> %s (v2)\n", $1, $3, $5 }'
    done
else
    warn "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} を固定している。リンクイベントで index が"
    warn "  ずれると片方の HCA だけ無言でハングする。dual-HCA なら空にするのが正解。"
    for HCA in "${IB_HCAS[@]}"; do
        GID_LINE=$(show_gids 2>/dev/null | awk -v d="${HCA}" -v i="${NCCL_IB_GID_INDEX}" '$1==d && $3==i')
        if [ -n "${GID_LINE}" ] && echo "${GID_LINE}" | grep -q 'v2'; then
            ok "${HCA} GID index ${NCCL_IB_GID_INDEX} = RoCEv2 ($(echo "${GID_LINE}" | awk '{print $5}'))"
        else
            ng "${HCA} の GID index ${NCCL_IB_GID_INDEX} が RoCEv2/IPv4 ではありません: show_gids で確認"
        fi
    done
fi
[ -e /dev/infiniband/uverbs0 ] && ok "/dev/infiniband あり" || ng "/dev/infiniband がありません"

# 光スイッチはジャンボを通さない。自 IF の MTU が 1500 か、経路で DF ping が
# 通るかを 3 peer 分すべて見る (小さい ping は通っても NCCL の TCP bootstrap が
# Send-Q 詰まりで無言ハングする事故が 2026-10-05 に実測されている)。
for i in "${!ROCE_IFS[@]}"; do
    IFN="${ROCE_IFS[$i]}"
    HCA="${IB_HCAS[$i]:-${IB_HCAS[0]}}"
    MTU=$(cat "/sys/class/net/${IFN}/mtu" 2>/dev/null)
    ACTIVE_MTU=$(ibv_devinfo -d "${HCA}" 2>/dev/null | awk '/active_mtu/ {print $2; exit}')
    if [ "${MTU:-0}" -ge 9000 ]; then
        ok "${IFN} MTU ${MTU} (RoCE path MTU ${ACTIVE_MTU:-?})"
    else
        warn "${IFN} MTU ${MTU:-?} / RoCE path MTU ${ACTIVE_MTU:-?} — 光スイッチはジャンボを"
        warn "  通さないので 1500 で正解。全ノードで揃えること (9000 のままだと bootstrap が詰まる)。"
    fi
    LIP=$(ip -4 -o addr show dev "${IFN}" 2>/dev/null | awk '{print $4; exit}' | cut -d/ -f1)
    if [ -n "${LIP}" ] && [ -n "${MTU}" ]; then
        for POCT in "${PEER_OCTS[@]}"; do
            PEER_IP="${LIP%.*}.${POCT}"
            [ "${PEER_IP}" = "${LIP}" ] && continue
            if ping -M do -c 2 -W 2 -s $((MTU - 28)) -I "${IFN}" "${PEER_IP}" >/dev/null 2>&1; then
                ok "${IFN} -> ${PEER_IP} に MTU ${MTU} の DF ping が通る"
            else
                ng "${IFN} -> ${PEER_IP} に MTU ${MTU} の DF ping が通りません"
                echo "       スイッチがジャンボを落としている / 相手が down / arp が別 NIC に答えている"
                echo "       (arp_ignore=1 / arp_announce=2 を 4 台とも確認)"
            fi
        done
    fi
done

echo
echo "== モデル =="
if [ -d "${MODEL_PATH}" ]; then
    N=$(ls "${MODEL_PATH}"/model-*.safetensors 2>/dev/null | wc -l)
    if [ "${N}" -eq 62 ]; then
        ok "${MODEL_PATH} (${N} shard, $(du -sh "${MODEL_PATH}" 2>/dev/null | cut -f1))"
    else
        ng "${MODEL_PATH} の shard 数が ${N} です (GLM-5.3-Flash-UNCENSORED-FP8 は 62) — ./scripts/fetch-model.sh を再実行"
    fi
    if [ -f "${MODEL_PATH}/model.safetensors.index.json" ]; then
        ok "${MODEL_PATH}/model.safetensors.index.json あり (MTP head はこの中)"
    else
        ng "${MODEL_PATH}/model.safetensors.index.json がありません"
    fi
else
    ng "${MODEL_PATH} がありません — ./scripts/fetch-model.sh"
fi

echo
echo "== メモリ =="
AVAIL_GIB=$(awk '/MemAvailable/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
# vLLM は起動時に「空き >= GPU_MEMORY_UTILIZATION x 119.63GiB」を要求し、満たさないと
#   ValueError: Free memory on device cuda:0 (109.32/119.63 GiB) on startup is less
#   than desired GPU memory utilization (0.92, 110.06 GiB)
# で即落ちる (2026-10-06 head 実測)。GB10 の cuda 空き ~= MemAvailable なので、ここで先に見る。
# 重みは FP8 / TP=4 で ~76.2GiB/rank なので、要求量を満たせば残りは KV に回る。
NEED_GIB=$(awk -v g="${GPU_MEMORY_UTILIZATION:-0.88}" 'BEGIN {printf "%.1f", g*119.63}')
if awk -v a="${AVAIL_GIB}" -v n="${NEED_GIB}" 'BEGIN {exit !(a >= n + 1.0)}'; then
    ok "MemAvailable ${AVAIL_GIB} GiB >= GMU ${GPU_MEMORY_UTILIZATION} x 119.63 = ${NEED_GIB} GiB"
elif awk -v a="${AVAIL_GIB}" -v n="${NEED_GIB}" 'BEGIN {exit !(a >= n)}'; then
    warn "MemAvailable ${AVAIL_GIB} GiB は要求 ${NEED_GIB} GiB をかろうじて超える程度。"
    warn "  他コンテナを止めて sync && echo 3 | sudo tee /proc/sys/vm/drop_caches を推奨"
else
    ng "MemAvailable ${AVAIL_GIB} GiB < 要求 ${NEED_GIB} GiB (GMU ${GPU_MEMORY_UTILIZATION})。"
    ng "  vLLM が起動直後に落ちる。GPU_MEMORY_UTILIZATION を下げるか、他プロセスを止める"
fi
RUNNING=$(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')
[ -n "${RUNNING}" ] && warn "起動中のコンテナ: ${RUNNING}"

echo
echo "== rootful docker =="
[ -e /dev/nvidia0 ] && ok "/dev/nvidia0 あり" || ng "/dev/nvidia0 がありません (ドライバを確認)"

# rootful daemon に nvidia ランタイムが登録されているか。
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
