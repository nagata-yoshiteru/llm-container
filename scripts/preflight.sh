#!/usr/bin/env bash
# =============================================================================
# 起動前チェック。head / worker 両方で実行する。
#
#   ./scripts/preflight.sh
#
# ここで赤が出た状態で起動すると、だいたい 5〜10 分待たされた挙句
# NCCL の "unhandled system error" か OOM-kill で死ぬ。
#
# このブランチ (Qwen3.8-Flash-Next-NVFP4 / TP=2) 固有の検査を後半に足してある。
# このモデルは「設定ミスでも起動はするが黙って品質だけ壊れる」経路が複数ある
# ので、そこを重点的に見る。
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
for k in VLLM_IMAGE MODEL_PATH HEAD_ROCE_IP WORKER_ROCE_IP ROCE_IF_NAME IB_HCA_NAME \
         NCCL_IB_GID_INDEX TP_SIZE MAX_MODEL_LEN VLLM_KV_ARGS VLLM_EXTRA_ARGS \
         VLLM_PARSER_ARGS VLLM_USE_DEEP_GEMM VLLM_ALLOW_LONG_MAX_MODEL_LEN; do
    printf -v "$k" '%s' "$(env_get "$k")"
done
# ROCE_IF_NAME / IB_HCA_NAME は dual-HCA だとカンマ区切りになる
# (例: enp1s0f0np0,enP2p1s0f0np0)。配列に割っておく。
IFS=',' read -ra ROCE_IFS <<< "${ROCE_IF_NAME}"
IFS=',' read -ra IB_HCAS  <<< "${IB_HCA_NAME}"

# NCCL_IB_GID_INDEX は dual-HCA では「意図的に未設定」が正解なので、
# 空を既定値で埋めない (埋めると固定してあるかのように見えてしまう)。

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
        ng "${IFN} に IPv4 が付いていません (netplan / ケーブルを確認)"
    else
        ok "${IFN} = ${IPS}"
        LOCAL_IPS="${LOCAL_IPS}${LOCAL_IPS:+$'\n'}${IPS}"
    fi
done
if [ -n "${LOCAL_IPS}" ] && ! echo "${LOCAL_IPS}" | grep -qx -e "${HEAD_ROCE_IP}" -e "${WORKER_ROCE_IP}"; then
    ng "どの IF の IP も .env の HEAD_ROCE_IP(${HEAD_ROCE_IP}) / WORKER_ROCE_IP(${WORKER_ROCE_IP}) と一致しません"
fi
# dual-HCA の 2 本目は rendezvous には使わない (NCCL が勝手に束ねる) ので、
# .env の HEAD/WORKER_ROCE_IP と一致しなくてよい。IP が付いてさえいればよい。

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
for HCA in "${IB_HCAS[@]}"; do
    if ibv_devinfo -d "${HCA}" 2>/dev/null | grep -q PORT_ACTIVE; then
        ok "${HCA} PORT_ACTIVE"
    else
        ng "${HCA} が PORT_ACTIVE ではありません (ケーブルが刺さっている方の HCA か確認)"
    fi
done

if [ -z "${NCCL_IB_GID_INDEX}" ]; then
    ok "GID index は未固定 (NCCL に選ばせる) — dual-HCA ではこれが正解"
    for HCA in "${IB_HCAS[@]}"; do
        show_gids 2>/dev/null | awk -v d="${HCA}" '$1==d && /v2/ && $5 ~ /^[0-9]+\./ {
            printf "       %s index %s -> %s (v2)\n", $1, $3, $5 }'
    done
else
    warn "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} を固定している。リンクイベントで index が"
    warn "  ずれると片方の HCA だけ無言でハングする"
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

for i in "${!ROCE_IFS[@]}"; do
    IFN="${ROCE_IFS[$i]}"
    HCA="${IB_HCAS[$i]:-${IB_HCAS[0]}}"
    MTU=$(cat "/sys/class/net/${IFN}/mtu" 2>/dev/null)
    ACTIVE_MTU=$(ibv_devinfo -d "${HCA}" 2>/dev/null | awk '/active_mtu/ {print $2; exit}')
    if [ "${MTU:-0}" -ge 9000 ]; then
        ok "${IFN} MTU ${MTU} (RoCE path MTU ${ACTIVE_MTU:-?})"
    else
        warn "${IFN} MTU ${MTU:-?} / RoCE path MTU ${ACTIVE_MTU:-?} — 9000 に上げると path MTU が 4096 になる"
        warn "  両ノードで揃えること"
    fi
done

echo
echo "== モデル =="
# nvidia/Qwen3.8-Flash-Next-NVFP4 の構成:
#   model-00001-of-00010.safetensors .. model-00010-of-00010.safetensors  (本体)
#   model-fp8-mtp-ple.safetensors                                          (MTP + PLE)
# 合計 123.5 GiB。PLE (47.7 GiB) が別シャードに切られているのが特徴。
if [ -d "${MODEL_PATH}" ]; then
    N=$(ls "${MODEL_PATH}"/model-*-of-*.safetensors 2>/dev/null | wc -l)
    if [ "${N}" -eq 10 ]; then
        ok "本体 shard 10 個"
    else
        ng "本体 shard が ${N} 個です (10 のはず) — scripts/fetch-model.sh を再実行"
    fi
    if [ -f "${MODEL_PATH}/model-fp8-mtp-ple.safetensors" ]; then
        ok "model-fp8-mtp-ple.safetensors あり (MTP + PLE)"
    else
        ng "model-fp8-mtp-ple.safetensors がありません — これが無いと MTP も PLE も死にます"
    fi
    for f in config.json model.safetensors.index.json tokenizer.json hf_quant_config.json; do
        [ -f "${MODEL_PATH}/${f}" ] || ng "${MODEL_PATH}/${f} がありません"
    done
    SZ_GIB=$(du -sB1 "${MODEL_PATH}" 2>/dev/null | awk '{printf "%d", $1/1073741824}')
    if [ "${SZ_GIB:-0}" -ge 120 ]; then
        ok "${MODEL_PATH} = ${SZ_GIB} GiB"
    else
        ng "${MODEL_PATH} が ${SZ_GIB} GiB しかありません (123 GiB 前後のはず) — 途中で切れています"
    fi
else
    ng "${MODEL_PATH} がありません — ./scripts/fetch-model.sh"
fi

echo
echo "== メモリ =="
# TP=2 なので 1 台あたり: 重み 123.5/2 = 約 62 GiB + KV 24 GiB + ランタイム。
# GPU_MEMORY_UTILIZATION 0.85 x 121 GiB = 約 103 GiB を vLLM が確保しにいく。
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
if [ "${AVAIL_GB}" -ge 105 ]; then
    ok "MemAvailable ${AVAIL_GB} GB"
elif [ "${AVAIL_GB}" -ge 98 ]; then
    warn "MemAvailable ${AVAIL_GB} GB — ギリギリ。他のコンテナを止めて sync && drop_caches 推奨"
else
    ng "MemAvailable ${AVAIL_GB} GB — 足りません。他のコンテナを止めるか再起動してください"
fi
RUNNING=$(docker ps --format '{{.Names}}' 2>/dev/null | tr '\n' ' ')
[ -n "${RUNNING}" ] && warn "rootless docker で起動中: ${RUNNING}"

echo
echo "== rootful docker =="
[ -e /dev/nvidia0 ] && ok "/dev/nvidia0 あり" || ng "/dev/nvidia0 がありません (ドライバを確認)"

if grep -qs nvidia /etc/docker/daemon.json; then
    ok "rootful daemon に nvidia ランタイム登録済み"
else
    ng "/etc/docker/daemon.json に nvidia ランタイムがありません"
    echo "       sudo nvidia-ctk runtime configure --runtime=docker"
    echo "       sudo systemctl restart docker"
fi

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

# =============================================================================
# ここから下は Qwen3.8-Flash-Next-NVFP4 / TP=2 固有。
# どれも「起動はするのに黙って壊れる」類なので、必ず緑にしてから起動すること。
# =============================================================================
echo
echo "== このモデル固有の設定 =="

# --- 1台構成の検出 -----------------------------------------------------------
# 重みだけで 123.5 GiB あり、DGX Spark 1 台 (121 GiB) には物理的に載らない。
# VLLM_PLE_CPU_OFFLOAD は pinned host memory を確保するので UMA では逃げ場に
# ならない (swap にも落ちない)。詳細は docker-compose.yml 冒頭。
if [ "${TP_SIZE:-2}" = "1" ]; then
    ng "TP_SIZE=1 — このチェックポイントは重み 123.5 GiB で 1 台 (121 GiB) には載りません"
    echo "       PLE (47.7 GiB) を NVFP4 化した派生版を使うか、TP_SIZE=2 に戻すこと"
else
    ok "TP_SIZE=${TP_SIZE} (2 台に分散)"
fi

# --- イメージのバージョン ----------------------------------------------------
# 必要な修正:
#   d4d703c (2026-09-03, PR #54882) FP8 PLE ローダ
#     -> 無いと PLE を黙って誤った値でロードする。落ちないので気付けない。
#   PR #55513 (2026-09-08)          block-FP8 MTP
#     -> 無いと MTP 投機デコードが使えない。
# v0.29.0 のリリースブランチは main から behind_by 388 でどちらも未取込。
case "${VLLM_IMAGE}" in
    *nightly*|*@sha256:*)
        ok "イメージは nightly / digest 固定 (${VLLM_IMAGE##*:})" ;;
    *:v0.2[0-9].*|*:v0.1[0-9].*|*:latest|*:v0.29.0*)
        ng "VLLM_IMAGE がリリースタグです: ${VLLM_IMAGE}"
        echo "       v0.29.0 以前には FP8 PLE ローダ修正 (d4d703c) と block-FP8 MTP 修正"
        echo "       (PR #55513) がどちらも入っていません。前者が無いと PLE を黙って"
        echo "       誤ロードします。.env.example の nightly タグを使ってください" ;;
    *)
        warn "VLLM_IMAGE のバージョンを判定できません: ${VLLM_IMAGE}"
        warn "  d4d703c (2026-09-03) 以降の main から焼かれたものであること" ;;
esac

# --- 量子化の指定 ------------------------------------------------------------
# hf_quant_config.json の quant_algo は MIXED_PRECISION。
# 現行 vLLM では modelopt(FP8経路) / modelopt_fp4 / modelopt_mixed に分かれて
# おり、モデルカードにある `--quantization modelopt` は FP8 経路の名指しになる。
if echo "${VLLM_EXTRA_ARGS}" | grep -qE '(^| )--quantization +modelopt( |$)'; then
    ng "--quantization modelopt が指定されています (モデルカードの記載をそのまま写した状態)"
    echo "       このチェックポイントの quant_algo は MIXED_PRECISION なので"
    echo "       正解は modelopt_mixed。指定ごと消せば自動解決されます"
elif echo "${VLLM_EXTRA_ARGS}" | grep -qE '(^| )--quantization'; then
    ok "--quantization は modelopt 以外を明示"
else
    ok "--quantization は未指定 (MIXED_PRECISION -> modelopt_mixed に自動解決)"
fi

# --- expert parallel ---------------------------------------------------------
# MTP の routed experts は 128x128 ブロック FP8。moe_intermediate_size=640 を
# TP=2 で割ると 320 になり 128 の倍数でなくなる。EP なら expert 数 (512) 側を
# 割るので各 expert の幅 640 が保たれる。
if echo "${VLLM_EXTRA_ARGS}" | grep -q -- '--enable-expert-parallel'; then
    ok "--enable-expert-parallel あり (TP=2 の MTP に必須)"
else
    ng "--enable-expert-parallel がありません"
    echo "       TP=2 では MTP の 128x128 FP8 ブロックが割り切れず壊れます"
    echo "       (moe_intermediate_size 640 / 2 = 320 は 128 の倍数ではない)"
fi

# --- 投機デコード ------------------------------------------------------------
if echo "${VLLM_EXTRA_ARGS}" | grep -q '"method":"mtp"'; then
    K=$(echo "${VLLM_EXTRA_ARGS}" | sed -n 's/.*"num_speculative_tokens":\([0-9]*\).*/\1/p')
    ok "MTP 投機デコード k=${K:-?}"
else
    warn "MTP 投機デコードが無効です。decode 速度が数割落ちます"
fi

# --- 1M context の整合 -------------------------------------------------------
# config の max_position_embeddings は 262144。超える場合は
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 と --hf-overrides の YaRN がセットで要る。
MML=${MAX_MODEL_LEN:-262144}
if [ "${MML}" -gt 262144 ]; then
    if [ "${VLLM_ALLOW_LONG_MAX_MODEL_LEN}" = "1" ]; then
        ok "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 (262144 超に必須)"
    else
        ng "MAX_MODEL_LEN=${MML} だが VLLM_ALLOW_LONG_MAX_MODEL_LEN が 1 ではありません"
    fi
    if echo "${VLLM_KV_ARGS}" | grep -q '"rope_type":"yarn"'; then
        FACTOR=$(echo "${VLLM_KV_ARGS}" | sed -n 's/.*"factor":\([0-9.]*\).*/\1/p')
        ORIG=$(echo "${VLLM_KV_ARGS}" | sed -n 's/.*"original_max_position_embeddings":\([0-9]*\).*/\1/p')
        EFF=$(awk -v o="${ORIG:-262144}" -v f="${FACTOR:-1}" 'BEGIN{printf "%d", o*f}')
        if [ "${EFF}" = "${MML}" ]; then
            ok "YaRN factor ${FACTOR} x ${ORIG} = ${EFF} (MAX_MODEL_LEN と一致)"
        else
            ng "YaRN の実効長 ${EFF} と MAX_MODEL_LEN ${MML} が一致しません"
        fi
    else
        ng "MAX_MODEL_LEN=${MML} だが --hf-overrides の YaRN がありません"
    fi
else
    ok "MAX_MODEL_LEN=${MML} (native 262144 以内)"
fi

# --- KV 容量の妥当性 ---------------------------------------------------------
# full_attention_interval=4 なので 48 層中 12 層だけが full attention。
#   12 層 x 2 kv-head x 256 head_dim x 2 (K+V) x 2 B = 24 KiB/token (全体)
#   TP=2 で 12 KiB/token/node
KVB=$(echo "${VLLM_KV_ARGS}" | sed -n 's/.*--kv-cache-memory-bytes \([0-9]*\).*/\1/p')
if [ -n "${KVB}" ]; then
    KV_GIB=$((KVB / 1073741824))
    SEQS=$(awk -v kv="${KVB}" -v ml="${MML}" 'BEGIN{printf "%.1f", kv/(ml*12288)}')
    ok "KV ${KV_GIB} GiB/node = ${MML} token を ${SEQS} 本分 (12 KiB/token/node)"
    awk -v s="${SEQS}" 'BEGIN{ if (s < 1.0) exit 1 }' || \
        ng "  1 本も張れません。--kv-cache-memory-bytes を上げるか MAX_MODEL_LEN を下げること"
else
    warn "--kv-cache-memory-bytes が未指定。UMA では自動プロファイルが当てにならないので固定推奨"
fi
# KV の fp8 化は未検証 (チェックポイントに KV 量子化メタデータが無い)
echo "${VLLM_KV_ARGS}" | grep -q -- '--kv-cache-dtype' && \
    warn "--kv-cache-dtype が指定されています。この QSA 実装での fp8 KV は未検証です"

# --- GB10 の数値契約 ---------------------------------------------------------
if [ "${VLLM_USE_DEEP_GEMM}" = "0" ]; then
    ok "VLLM_USE_DEEP_GEMM=0 (GB10 では必須)"
else
    ng "VLLM_USE_DEEP_GEMM が 0 ではありません"
    echo "       GB10 では DeepGEMM の block-FP8 経路が正しく動きません。"
    echo "       このチェックポイントは MTP が 128x128 block FP8 なので必ず踏みます"
fi

# --- 廃止された環境変数 ------------------------------------------------------
# 他ブランチ (Nemotron / DeepSeek) からコピーしてきたときに残りがち。
for dead in VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FLASHINFER_MOE_FP4 \
            VLLM_USE_DEEP_GEMM_E8M0 VLLM_MOE_USE_DEEP_GEMM; do
    if grep -qE "^${dead}=" .env; then
        warn "${dead} が .env にあります — 現行 vLLM には存在しないか、この構成には無関係です"
    fi
done

# --- JSON 引数に空白が無いか -------------------------------------------------
# entrypoint が VLLM_*_ARGS を空白で分割するので、JSON の中に空白があると
# 途中で千切れて意味不明なエラーになる。
if command -v python3 >/dev/null 2>&1; then
    BAD=$(
        PF_KV="${VLLM_KV_ARGS}" PF_EX="${VLLM_EXTRA_ARGS}" PF_PA="${VLLM_PARSER_ARGS}" \
        python3 -c '
import json, os
bad = []
for name in ("PF_PA", "PF_KV", "PF_EX"):
    for tok in os.environ.get(name, "").split():
        if tok.startswith("{"):
            try:
                json.loads(tok)
            except Exception:
                bad.append(tok[:60])
print("\n".join(bad))
'
    )
    if [ -z "${BAD}" ]; then
        ok "VLLM_*_ARGS の JSON は空白なしで健全"
    else
        ng "VLLM_*_ARGS の JSON が壊れています (値の中に空白を入れないこと):"
        echo "${BAD}" | sed 's/^/       /'
    fi
fi

# --- パーサ ------------------------------------------------------------------
echo "${VLLM_PARSER_ARGS}" | grep -q -- '--reasoning-parser qwen3' \
    && ok "--reasoning-parser qwen3" \
    || warn "--reasoning-parser qwen3 が無い — thinking が content に混ざります"
echo "${VLLM_PARSER_ARGS}" | grep -q -- '--tool-call-parser qwen3_xml' \
    && ok "--tool-call-parser qwen3_xml" \
    || warn "--tool-call-parser qwen3_xml が無い — tool 呼び出しが解釈されません"

echo
[ "${RC}" -eq 0 ] && echo "==> preflight PASS" || echo "==> preflight FAIL"
exit "${RC}"
