#!/usr/bin/env bash
# =============================================================================
# 起動前チェック (単一ホスト / SM12x GPU を複数枚)
#
#   実機の GPU 枚数・VRAM・PCIe 配線・NVLink の有無を読んで、.env の設定と
#   噛み合っているかを判定する。値をハードコードしていないので、枚数や VRAM が
#   違う機体でもそのまま使える。
#
#   ./scripts/preflight.sh          … 通常のチェック
#   PREFLIGHT_DEEP=1 ./scripts/preflight.sh
#                                   … 上に加えてイメージを 1 回起動し、
#                                     CUDA arch list に sm_120 が居るかまで見る
#
# ここで赤が出た状態で起動すると、だいたい 10〜20 分待たされた挙句
# CUDA OOM か「起動はするが出力が壊れている」で終わる。
#
# 前半 = ハードウェア / Docker 環境 (機体ごとに違う部分)。
# 後半 = Qwen3.8-Flash-Next-NVFP4 固有。こちらは「設定ミスでも起動はするが
#        黙って品質だけ壊れる」経路が複数あるので重点的に見る。
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
    local v
    v=$(sed -n "s/^$1=//p" .env | tail -1)
    # 空白を含む値は 'シングルクォート' で括ってある。docker compose は展開時に
    # 剥がすので、ここでも剥がしておく。剥がさないと下の JSON 検査が末尾の
    # クォートを JSON の一部とみなして誤検知する。
    case "${v}" in
        \'*\') v=${v#\'}; v=${v%\'} ;;
        \"*\") v=${v#\"}; v=${v%\"} ;;
    esac
    printf '%s' "${v}"
}
for k in VLLM_IMAGE MODEL_PATH TP_SIZE MAX_MODEL_LEN MAX_NUM_SEQS \
         MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION \
         VLLM_KV_ARGS VLLM_EXTRA_ARGS VLLM_PARSER_ARGS \
         VLLM_USE_DEEP_GEMM VLLM_ALLOW_LONG_MAX_MODEL_LEN \
         NCCL_P2P_DISABLE VLLM_MOE_FORCE_MARLIN; do
    printf -v "$k" '%s' "$(env_get "$k")"
done
TP_SIZE=${TP_SIZE:-2}

echo "== ホスト =="
echo "  hostname: $(hostname)  arch: $(uname -m)  cpus: $(nproc)"
if [ "$(uname -m)" = "x86_64" ]; then
    ok "x86_64 (DGX Spark ブランチの aarch64 前提はこのブランチでは全部外してある)"
else
    ng "$(uname -m) — このブランチは x86_64 前提です"
fi

echo
echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    ng "nvidia-smi がありません (ドライバ未導入)"
    exit 1
fi
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version \
    --format=csv,noheader | sed 's/^/  /'

N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c .)
if [ "${N_GPU}" -ge "${TP_SIZE}" ]; then
    ok "GPU ${N_GPU} 枚 (TP_SIZE=${TP_SIZE})"
else
    ng "GPU が ${N_GPU} 枚しかありません (TP_SIZE=${TP_SIZE})"
fi

# compute capability。sm_120 (RTX PRO 6000) / sm_121 (GB10) はどちらも
# vLLM の is_device_capability_family(120) にマッチする同じ SM12x ファミリ。
while IFS=, read -r IDX CC; do
    CC=$(echo "${CC}" | tr -d ' ')
    case "${CC}" in
        12.0) ok "GPU${IDX} compute_cap ${CC} (sm_120 / SM12x ファミリ)" ;;
        12.*) ok "GPU${IDX} compute_cap ${CC} (SM12x ファミリ)" ;;
        *)    warn "GPU${IDX} compute_cap ${CC} — この .env は SM12x 向けに書かれています" ;;
    esac
done < <(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader)

# 混在構成だと TP のシャードが揃わない
N_DISTINCT=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort -u | grep -c .)
[ "${N_DISTINCT}" -eq 1 ] || ng "GPU の型番/VRAM が揃っていません。TP には同一構成が必要です"

# 既に何かが VRAM を掴んでいないか。
# ただし「このスタック自身が起動中」は正常なので区別する。
SELF_UP=0
docker compose ps --status running --quiet 2>/dev/null | grep -q . && SELF_UP=1
while IFS=, read -r IDX USED; do
    USED=$(echo "${USED}" | tr -dc '0-9')
    if [ "${USED:-0}" -lt 1024 ]; then
        ok "GPU${IDX} 使用中 ${USED} MiB"
    elif [ "${USED:-0}" -lt 4096 ]; then
        warn "GPU${IDX} が既に ${USED} MiB 使用中 (デスクトップ / ブラウザ?)"
        warn "  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-?} の余白を食います"
    elif [ "${SELF_UP}" = "1" ]; then
        # 自分自身が起動中なだけ。起動後に様子を見るために回すこともあるので
        # これは異常ではない。
        warn "GPU${IDX} が ${USED} MiB 使用中 — このスタック自身が起動中です"
    else
        ng "GPU${IDX} が既に ${USED} MiB 使用中 — 他のプロセスを止めてください"
    fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)

echo
echo "== NCCL P2P =="
# ★★ ここは nvidia-smi の言うことを信じてはいけない ★★
#
# NVLink の無い Blackwell ワークステーション機 (RTX PRO 6000 等を PCIe だけで
# 複数枚) では NCCL の P2P 経路がドライバ/NCCL レベルで壊れており、最初の
# collective で 100% ハングする (NVIDIA/nccl #1999, vllm #33041, sglang #15181)。
# それでも
#     nvidia-smi topo -p2p r   -> OK
#     can_device_access_peer() -> true
# と出るので、これらを根拠に NCCL_P2P_DISABLE を外すと刺さる。
#
# 一方 NVLink / NVSwitch で繋がっている機体では P2P は正常に効くので、
# 切ると素直に遅くなるだけ。よって「topo が OK か」ではなく
# 「NVLink があるか」で必要な設定が変わる。
if [ "${N_GPU}" -le 1 ] || [ "${TP_SIZE}" = "1" ]; then
    ok "GPU 1 枚構成なので collective が走らない — NCCL_P2P_DISABLE は不問"
elif nvidia-smi topo -m 2>/dev/null | grep -qE '(^|[[:space:]])NV[0-9]+([[:space:]]|$)'; then
    HAS_NVLINK=1
    if [ "${NCCL_P2P_DISABLE}" = "1" ]; then
        warn "NVLink があるのに NCCL_P2P_DISABLE=1 です。all-reduce が"
        warn "  ホストメモリ経由に落ちて遅くなります。外すことを検討してください"
    else
        ok "NVLink 接続 — P2P はそのまま使える (NCCL_P2P_DISABLE は未設定が正解)"
    fi
elif [ "${NCCL_P2P_DISABLE}" = "1" ]; then
    ok "PCIe のみ + NCCL_P2P_DISABLE=1 (この構成では必須。topo の OK 表示は当てにならない)"
else
    ng "NVLink が無い複数 GPU 構成なのに NCCL_P2P_DISABLE が 1 ではありません"
    echo "       この構成では起動しても NCCL init でハングする可能性が高いです。"
    echo "       症状: ログが \`vLLM is using nccl==...\` で止まり、VRAM は 1 GiB のまま"
    echo "             GPU util だけ 100% (スピン待ち)。何時間待っても進みません。"
    echo "       Blackwell ワークステーション機の既知の不具合:"
    echo "         NVIDIA/nccl #1999 / vllm-project/vllm #33041 / sgl-project/sglang #15181"
    echo "       .env に NCCL_P2P_DISABLE=1 を設定してください"
fi
# vLLM 独自 all-reduce も CUDA IPC/P2P 前提なので、P2P を切るなら一緒に切る。
if echo "${VLLM_EXTRA_ARGS}" | grep -q -- '--disable-custom-all-reduce'; then
    ok "--disable-custom-all-reduce あり (P2P を切った環境では実質必須)"
else
    warn "--disable-custom-all-reduce がありません。vLLM 独自 all-reduce は CUDA IPC/P2P"
    warn "  前提なので、worker 初期化中にハングする報告があります。足しておくのが無難"
fi
# 参考情報として topo も出す (判定には使わない)
nvidia-smi topo -p2p r 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -E '^[[:space:]]*GPU[0-9]' | sed 's/^/       (参考) /'

# --- GPU が刺さっていないか --------------------------------------------------
# NCCL ハングを踏むと、コンテナを落としてもドライバ内にカーネルが残り、
# 「プロセスは無いのに util 100%」という状態になる。この状態で再起動しても
# また刺さるので、先にリセットさせる。
NPROC_GPU=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
while IFS=, read -r IDX UTIL; do
    UTIL=$(echo "${UTIL}" | tr -dc '0-9')
    if [ "${UTIL:-0}" -ge 50 ] && [ "${NPROC_GPU}" -eq 0 ]; then
        ng "GPU${IDX} が util ${UTIL}% なのに compute プロセスがありません (wedged)"
        echo "       前回の NCCL ハングの残骸です。このまま起動しても また刺さります。"
        echo "         sudo nvidia-smi -r -i ${IDX}      # 画面出力が乗っていると失敗します"
        echo "       ダメならリブート。"
    fi
done < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader)

# --- PCIe リンク幅 -----------------------------------------------------------
# ★ 見落としやすい性能要因。物理的に x16 の形をしたスロットでも、チップセット
#   配下の x4 / x8 にしか配線されていないことがある。
#   アイドル時は速度 (GT/s) が Gen1 まで落ちるのが正常なので、**幅だけ**見る。
#   幅は上流ブリッジの LnkCap を読む (GPU 側の max_link_width は
#   「GPU が対応する最大」でスロット配線を反映しない)。
echo
echo "== PCIe リンク幅 (上流ブリッジの LnkCap) =="
while IFS=, read -r IDX BUSID; do
    BUSID=$(echo "${BUSID}" | tr -d ' ' | tr 'A-F' 'a-f')
    # nvidia-smi は 8 桁ドメインで返すが sysfs は 4 桁 (例 00000000:NN:00.0 ->
    # 0000:NN:00.0) なので先頭を削る
    SYSID="${BUSID#0000}"
    DEVDIR="/sys/bus/pci/devices/${SYSID}"
    if [ ! -d "${DEVDIR}" ]; then
        warn "GPU${IDX} (${SYSID}) を sysfs で見つけられません"
        continue
    fi
    BRIDGE=$(basename "$(dirname "$(readlink -f "${DEVDIR}")")")
    BW=$(cat "/sys/bus/pci/devices/${BRIDGE}/max_link_width" 2>/dev/null)
    BS=$(cat "/sys/bus/pci/devices/${BRIDGE}/max_link_speed" 2>/dev/null)
    CW=$(cat "${DEVDIR}/current_link_width" 2>/dev/null)
    LABEL="GPU${IDX} ${SYSID} <- ${BRIDGE}: cap ${BS:-?} x${BW:-?} / 現在 x${CW:-?}"
    if [ "${BW:-0}" -ge 16 ]; then
        ok "${LABEL}"
    elif [ "${BW:-0}" -ge 8 ]; then
        warn "${LABEL} — x8。TP/EP の通信が半分の帯域になります"
    else
        warn "${LABEL}"
        echo "       ★ このスロットは x${BW:-?} しか配線されていません。"
        echo "         TP=2 の all-reduce と EP の all-to-all がここに律速されます。"
        echo "         CPU 直結の x16 スロットが空いているなら挿し替えるのが、"
        echo "         .env のどの値をいじるより効きます。BIOS の PCIe bifurcation も確認。"
        echo "         (挿し替えられないなら .env の「MTP を外す場合」を検討)"
    fi
done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)

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
echo "== VRAM 収支 =="
# 重み 123.5 GiB を TP_SIZE で割ったもの + 活性化 + CUDA graph + KV。
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
UTIL=${GPU_MEMORY_UTILIZATION:-0.92}
read -r VRAM_GIB BUDGET_GIB WEIGHT_GIB REST_GIB <<< "$(
    awk -v m="${VRAM_MIB}" -v u="${UTIL}" -v tp="${TP_SIZE}" 'BEGIN{
        v=m/1024; b=v*u; w=123.5/tp; printf "%.1f %.1f %.1f %.1f", v, b, w, b-w }'
)"
echo "  VRAM ${VRAM_GIB} GiB/GPU x ${UTIL} = ${BUDGET_GIB} GiB を vLLM が確保"
echo "  うち重み ${WEIGHT_GIB} GiB -> 活性化 + CUDA graph + KV に ${REST_GIB} GiB"
if awk -v r="${REST_GIB}" 'BEGIN{exit !(r < 6)}'; then
    ng "残り ${REST_GIB} GiB では KV が取れません"
    echo "       GPU_MEMORY_UTILIZATION を上げるか、TP_SIZE / GPU を増やすこと"
elif awk -v r="${REST_GIB}" 'BEGIN{exit !(r < 12)}'; then
    warn "残り ${REST_GIB} GiB — KV がかなり細くなります"
else
    # KV は残りから活性化 + CUDA graph の概算 (5 GiB) を引いたもの。
    # KV は 24 KiB/token (全体) なので TP で割って KiB/token/GPU。
    ok "$(awk -v r="${REST_GIB}" -v tp="${TP_SIZE}" -v ml="${MAX_MODEL_LEN:-1048576}" 'BEGIN{
        kv = r - 5;
        tok = kv * 1048576 / (24.0 / tp);
        printf "KV 概算 %.0f GiB/GPU = 約 %.2fM token (%s token を %.1f 本)",
               kv, tok/1e6, ml, tok/ml }')"
fi
if [ "${TP_SIZE}" = "1" ]; then
    ng "TP_SIZE=1 — 重み 123.5 GiB は 1 枚 (${VRAM_GIB} GiB) には載りません"
fi

echo
echo "== ホスト RAM / ディスク =="
AVAIL_GB=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
# 専有 VRAM なのでホスト RAM は主にページキャッシュ用。UMA だった DGX Spark と
# 違い、ここが埋まっていても致命傷にはならない (ロードが遅くなるだけ)。
if [ "${AVAIL_GB}" -ge 32 ]; then
    ok "MemAvailable ${AVAIL_GB} GB"
else
    warn "MemAvailable ${AVAIL_GB} GB — チェックポイントのページキャッシュが効かず初回ロードが遅くなります"
fi
DISK_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${DISK_GB:-0}" -ge 20 ] && ok "空きディスク ${DISK_GB} GB" \
                           || warn "空きディスク ${DISK_GB} GB — キャッシュの置き場に注意"

echo
echo "== Docker / NVIDIA Container Toolkit =="
if ! command -v docker >/dev/null 2>&1; then
    ng "docker がありません"
else
    ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
fi

# GPU をコンテナに渡すには NVIDIA Container Toolkit が要る。
# 素の docker の --gpus だけでは動かない。
if command -v nvidia-ctk >/dev/null 2>&1 || command -v nvidia-container-cli >/dev/null 2>&1; then
    ok "nvidia-container-toolkit 導入済み"
else
    ng "nvidia-container-toolkit がありません (これが無いと --gpus / gpus: all が効きません)"
    echo "       curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\"
    echo "         | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "       curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\"
    echo "         | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\"
    echo "         | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "       sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    echo "       sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

CTX=$(docker context show 2>/dev/null)
ok "docker context = ${CTX:-default}"
if [ "${CTX}" = "rootless" ]; then
    # rootless では nvidia-container-cli が device cgroup を触れないので、
    # no-cgroups=true にしたうえで compose 側が /dev/nvidia* を明示的に渡す。
    if grep -qsE '^\s*no-cgroups\s*=\s*true' \
         "${HOME}/.config/nvidia-container-runtime/config.toml" \
         /etc/nvidia-container-runtime/config.toml; then
        ok "no-cgroups=true (rootless に必要な設定)。/dev/nvidia* は compose の devices: で明示渡し"
    else
        warn "rootless なのに no-cgroups=true が見つかりません。GPU が渡らない場合は:"
        echo "       nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place"
    fi
fi

# --- rlimit ------------------------------------------------------------------
# ★ rootless で一番踏みやすい罠。ハード rlimit の引き上げには初期 user
#   namespace の CAP_SYS_RESOURCE が要るので、rootless では**ホストのハード上限を
#   1 バイトでも超える ulimits を compose に書いた時点で起動できない**:
#     error setting rlimit type 8: operation not permitted   (type 8 = MEMLOCK)
#   このブランチは RDMA を使わないので memlock は compose から外してある。
HARD_MEMLOCK=$(ulimit -Hl)
if grep -qE '^\s*memlock:' docker-compose.yml; then
    if [ "${CTX}" = "rootless" ] && [ "${HARD_MEMLOCK}" != "unlimited" ]; then
        ng "compose に memlock の ulimit があり、rootless のハード上限は ${HARD_MEMLOCK} KB です"
        echo "       このまま起動すると runc が type 8 (RLIMIT_MEMLOCK) で EPERM になります。"
        echo "       RDMA を使わないこのブランチでは memlock は不要なので、"
        echo "       docker-compose.yml の ulimits から丸ごと消すのが正解です"
    else
        ok "compose の memlock ulimit は現在の権限で設定可能"
    fi
else
    ok "compose に memlock の ulimit なし (RDMA を使わないので不要 / rootless でも起動できる)"
fi
if [ "$(ulimit -Hs)" = "unlimited" ]; then
    ok "stack のハード上限は unlimited (compose の stack: 64 MiB は通る)"
else
    warn "stack のハード上限が $(ulimit -Hs) KB です。compose の stack (65536 KB) を超えるなら下げること"
fi

# compose の `gpus: all` は daemon.json への runtime 登録を**必要としない**。
# moby 組み込みの nvidia device driver が OCI prestart hook を直接差し込むので、
# nvidia-container-runtime-hook のバイナリさえあれば素の runc でも GPU が通る。
# (rootless で daemon.json を置いていなくても動くのはこのため)
if command -v nvidia-container-runtime-hook >/dev/null 2>&1; then
    ok "nvidia-container-runtime-hook あり — \`gpus: all\` は runtime 登録なしで通る"
else
    ng "nvidia-container-runtime-hook がありません — \`gpus: all\` が GPU を渡せません"
fi
if docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
    ok "daemon に nvidia ランタイムも登録済み (--runtime=nvidia も使える)"
fi

for DK in "docker" "sudo -n docker"; do
    if ${DK} info >/dev/null 2>&1; then
        ${DK} image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 \
            && ok "\`${DK}\` でイメージ取得済み" \
            || warn "\`${DK}\` でイメージ未取得 — ${DK} pull ${VLLM_IMAGE}"
    fi
done
for D in /dev/nvidia0 /dev/nvidia1 /dev/nvidiactl /dev/nvidia-uvm; do
    [ -e "${D}" ] && ok "${D} あり" || ng "${D} がありません"
done

# =============================================================================
# ここから下は Qwen3.8-Flash-Next-NVFP4 固有。
# どれも「起動はするのに黙って壊れる」類なので、必ず緑にしてから起動すること。
# =============================================================================
echo
echo "== このモデル固有の設定 =="

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

# --- expert parallel と MTP の整合 -------------------------------------------
# MTP の routed experts は 128x128 ブロック FP8。moe_intermediate_size=640 を
# TP=2 で割ると 320 になり 128 の倍数でなくなる。EP なら expert 数 (512) 側を
# 割るので各 expert の幅 640 が保たれる。
# 本体側の routed experts は NVFP4 (16 要素ブロック) なので 320 でも割り切れる。
# つまり「MTP を使うなら EP 必須 / MTP を使わないなら EP 不要」。
HAS_EP=0; HAS_MTP=0
echo "${VLLM_EXTRA_ARGS}" | grep -q -- '--enable-expert-parallel' && HAS_EP=1
echo "${VLLM_EXTRA_ARGS}" | grep -q '"method":"mtp"' && HAS_MTP=1

if [ "${HAS_MTP}" = "1" ] && [ "${HAS_EP}" = "1" ]; then
    K=$(echo "${VLLM_EXTRA_ARGS}" | sed -n 's/.*"num_speculative_tokens":\([0-9]*\).*/\1/p')
    ok "MTP 投機デコード k=${K:-?} + --enable-expert-parallel (TP=${TP_SIZE} では必須の組)"
    warn "  EP の all-to-all は PCIe を往復します。上の「PCIe リンク幅」が x16 でない側が"
    warn "  あるなら、.env の「MTP を外す場合」(MTP と EP をセットで外す) と比較すること"
elif [ "${HAS_MTP}" = "1" ] && [ "${HAS_EP}" = "0" ]; then
    ng "MTP が有効なのに --enable-expert-parallel がありません"
    echo "       TP=${TP_SIZE} では MTP の 128x128 FP8 ブロックが割り切れず壊れます"
    echo "       (moe_intermediate_size 640 / ${TP_SIZE} は 128 の倍数ではない)"
elif [ "${HAS_MTP}" = "0" ] && [ "${HAS_EP}" = "1" ]; then
    warn "MTP 無しで EP だけ有効です。本体の NVFP4 experts には EP は不要なので、"
    warn "  --enable-expert-parallel を外すと all-to-all が消えて PCIe が楽になります"
else
    warn "MTP 投機デコードが無効です (EP も無し)。decode 速度が数割落ちる代わりに"
    warn "  GPU 間通信は all-reduce だけになります。x4 リンク環境では妥当な選択"
fi

# Marlin 強制と MTP の併用は受理率を落とすという報告がある
if [ -n "${VLLM_MOE_FORCE_MARLIN}" ] && [ "${HAS_MTP}" = "1" ]; then
    warn "VLLM_MOE_FORCE_MARLIN と MTP を併用しています。Marlin は dequant 経路なので"
    warn "  drafter の受理率が落ちて逆に遅くなるという報告があります。外して比較すること"
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

# --- KV の指定方法 -----------------------------------------------------------
# 専有 VRAM では vLLM の起動時プロファイルが正しく効くので、DGX Spark (UMA)
# と違って --kv-cache-memory-bytes の固定は不要。
KVB=$(echo "${VLLM_KV_ARGS}" | sed -n 's/.*--kv-cache-memory-bytes \([0-9]*\).*/\1/p')
if [ -n "${KVB}" ]; then
    KV_GIB=$((KVB / 1073741824))
    SEQS=$(awk -v kv="${KVB}" -v ml="${MML}" -v tp="${TP_SIZE}" \
           'BEGIN{printf "%.1f", kv/(ml*24576/tp)}')
    ok "KV を ${KV_GIB} GiB/GPU に固定 = ${MML} token を ${SEQS} 本分"
    warn "  専有 VRAM では自動プロファイルが効くので、固定しない方が安全です"
    awk -v s="${SEQS}" 'BEGIN{ if (s < 1.0) exit 1 }' || \
        ng "  1 本も張れません。値を上げるか MAX_MODEL_LEN を下げること"
else
    ok "--kv-cache-memory-bytes 未指定 (起動時プロファイルに任せる = 専有 VRAM での正解)"
fi
# KV の fp8 化は未検証 (チェックポイントに KV 量子化メタデータが無い)
echo "${VLLM_KV_ARGS}" | grep -q -- '--kv-cache-dtype' && \
    warn "--kv-cache-dtype が指定されています。この QSA 実装での fp8 KV は未検証です"

# --- SM12x の数値契約 --------------------------------------------------------
if [ "${VLLM_USE_DEEP_GEMM}" = "0" ]; then
    ok "VLLM_USE_DEEP_GEMM=0 (DeepGEMM は sm_90a/sm_100a 向けで SM12x 用カーネルが無い)"
else
    ng "VLLM_USE_DEEP_GEMM が 0 ではありません"
    echo "       このチェックポイントは MTP が 128x128 block FP8 なので block-FP8 経路を"
    echo "       必ず踏みます。SM12x で選ばれると黙って壊れます"
fi

# --- 廃止された / このブランチでは有害な環境変数 -----------------------------
for dead in VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FLASHINFER_MOE_FP4 \
            VLLM_USE_DEEP_GEMM_E8M0 VLLM_MOE_USE_DEEP_GEMM; do
    if grep -qE "^${dead}=" .env; then
        warn "${dead} が .env にあります — 現行 vLLM には存在しないか、この構成には無関係です"
    fi
done
for gone in VLLM_NCCL_SO_PATH HEAD_ROCE_IP WORKER_ROCE_IP ROCE_IF_NAME IB_HCA_NAME \
            NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_MERGE_NICS NNODES MASTER_PORT; do
    if grep -qE "^${gone}=" .env; then
        ng "${gone} が .env に残っています — DGX Spark x2 ブランチの遺物です"
        echo "       単一ノードのこのブランチでは不要 (VLLM_NCCL_SO_PATH は"
        echo "       aarch64 のパスなので x86_64 では NCCL のロードに失敗します)"
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

# --- イメージの中身まで見る (PREFLIGHT_DEEP=1) -------------------------------
# amd64 ビルドの CUDA arch list に sm_120 が無いと PTX JIT 経由になり、
# 起動が極端に遅くなる (数十分) か、カーネルによっては動かない。
if [ "${PREFLIGHT_DEEP:-0}" = "1" ]; then
    echo
    echo "== イメージの CUDA arch list (PREFLIGHT_DEEP=1) =="
    DK=docker
    docker image inspect "${VLLM_IMAGE}" >/dev/null 2>&1 || DK="sudo docker"
    ARCHES=$(${DK} run --rm --entrypoint python3 "${VLLM_IMAGE}" \
             -c 'import torch; print(" ".join(torch.cuda.get_arch_list()))' 2>&1 | tail -1)
    echo "  ${ARCHES}"
    case "${ARCHES}" in
        *sm_120*) ok "sm_120 のバイナリを同梱" ;;
        *ERROR*|*Traceback*) warn "arch list を取得できませんでした" ;;
        *) ng "arch list に sm_120 がありません — PTX JIT に落ちて起動が極端に遅くなります" ;;
    esac
fi

echo
[ "${RC}" -eq 0 ] && echo "==> preflight PASS" || echo "==> preflight FAIL"
exit "${RC}"
