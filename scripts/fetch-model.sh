#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1-Flash (EXL3 2.9bpw) の重みを取得する。
#   (出典: MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks の download.sh 読替)
#
#   ./scripts/fetch-model.sh          # 両方 (EXL3 -> Engram の順)
#   ./scripts/fetch-model.sh exl3     # EXL3 ツリーだけ (39 shard / 約 197GiB)
#   ./scripts/fetch-model.sh engram   # Engram テーブルだけ (約 190GiB)
#
# Engram (layers 1 と 14 の n-gram 埋込) だけは量子化されず EXL3 ツリーに
# 入っていない。ネイティブ deepseek-ai/DeepSeek-V4.1-Flash (48 shard / 476GiB)
# の **shard 47+48 + index + config.json だけ** を落とし、embed 専用の slim
# index に置き換えて ENGRAM_PATH に置く (hardlink なので追加容量は実質 0)。
#
# ★ネイティブのフルチェックポイント (476GiB) は要らない。47+48 だけ。
# ★既に DeepSeek-V4.1-Flash の checkout を持っている人は環境変数 NATIVE_SRC
#   でそこを指すと 190GiB の DL をスキップできる:
#     NATIVE_SRC=/data/models/DeepSeek-V4.1-Flash ./scripts/fetch-model.sh engram
#
# 2 台とも「同じ絶対パス」に落とす必要がある。head 側で実行したあと、
# worker 側でも同じコマンドを実行すること (rsync でコピーしてもよい)。
# resumable: 失敗したら同じコマンドを再実行すれば続きから。
#
# rootless / rootful どちらの docker とも無関係。sudo は不要。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

EXL3_REPO="${EXL3_REPO:-Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
NATIVE_REPO="${NATIVE_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
EXPECTED_SHARDS=39

env_get() { [ -f .env ] && sed -n "s/^$1=//p" .env | tail -1 || true; }

# 引数: [exl3|engram] [EXL3 保存先パス override]
MODE="all"
ARG_PATH=""
case "${1:-}" in
    exl3|engram) MODE="$1"; ARG_PATH="${2:-}" ;;
    *)           ARG_PATH="${1:-}" ;;
esac

MODEL_PATH_DEST="${ARG_PATH:-$(env_get MODEL_PATH)}"
MODEL_PATH_DEST="${MODEL_PATH_DEST:-./models/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
ENGRAM_PATH_DEST="$(env_get ENGRAM_PATH)"
ENGRAM_PATH_DEST="${ENGRAM_PATH_DEST:-./models/DeepSeek-V4.1-Flash-engram}"
# ネイティブ 4 ファイルの一時置き場 (slim 化元)。同じ FS 上でないと hardlink 不可。
ENGRAM_SRC_DIR="${ENGRAM_SRC_DIR:-${ENGRAM_PATH_DEST%/*}/DeepSeek-V4.1-Flash-native-partial}"

if ! command -v hf >/dev/null 2>&1; then
    echo "huggingface_hub CLI (hf) が見つかりません。以下でインストールしてください:" >&2
    echo "  pip install -U 'huggingface_hub[cli,hf_transfer]'" >&2
    exit 1
fi

check_disk() {  # $1=宛先dir $2=必要GiB
    mkdir -p "$1"
    local avail_kb; avail_kb=$(df -Pk "$(dirname "$1")" | awk 'NR==2 {print $4}')
    if [ "${avail_kb}" -lt $(( $2 * 1024 * 1024 )) ]; then
        echo "WARN: $(dirname "$1") の空きが ${2}GB 未満です ($((avail_kb / 1024 / 1024)) GB)" >&2
    fi
}

# hf_transfer があれば有効化 (200Gbps を活かすなら効く)
python3 -c 'import hf_transfer' 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1

fetch_exl3() {
    echo "==> EXL3: ${EXL3_REPO} -> ${MODEL_PATH_DEST} (39 shard / 約 197GiB)"
    check_disk "${MODEL_PATH_DEST}" 210
    hf download "${EXL3_REPO}" --local-dir "${MODEL_PATH_DEST}" --max-workers 8
    local n; n=$(find "${MODEL_PATH_DEST}" -maxdepth 1 -name 'model-*.safetensors' | wc -l)
    echo "==> safetensors shards: ${n} (${EXPECTED_SHARDS} なら OK)"
    [ "${n}" -eq "${EXPECTED_SHARDS}" ] || echo "WARN: shard 数が ${EXPECTED_SHARDS} と違います" >&2
}

fetch_engram() {
    # slim index が出来上がっていれば完了済みとみなす
    if grep -qs dsv41_engram_src "${ENGRAM_PATH_DEST}/model.safetensors.index.json"; then
        echo "==> Engram: ${ENGRAM_PATH_DEST} は準備済み (embed-only index)。スキップ"
        return 0
    fi

    local src="${NATIVE_SRC:-${ENGRAM_SRC_DIR}}"
    if [ ! -f "${src}/model-00047-of-00048.safetensors" ]; then
        echo "==> Engram 元: ${src} にネイティブ shard 47+48 + index + config.json を取得 (約 190GiB)"
        check_disk "${src}" 200
        # --include で 4 ファイルだけ。476GiB の全体は落かない。
        hf download "${NATIVE_REPO}" --local-dir "${src}" --max-workers 4 \
            --include "model-00047-of-00048.safetensors" \
                      "model-00048-of-00048.safetensors" \
                      "model.safetensors.index.json" \
                      "config.json"
    else
        echo "==> Engram 元: ${src} を再利用 (shard 47/48 既存)"
    fi

    echo "==> slim 化: ${src} -> ${ENGRAM_PATH_DEST} (hardlink + embed-only index)"
    python3 scripts/prepare-engram-src.py --src "${src}" --dst "${ENGRAM_PATH_DEST}"

    echo "==> 完了。置き場所を固定できたので一時置き場の index だけ片付けてもよい"
    echo "    (shard 自体は hardlink なので消しても ENGRAM_PATH 側の容量は減らない):"
    echo "    ls ${ENGRAM_PATH_DEST}"
}

case "${MODE}" in
    exl3)   fetch_exl3 ;;
    engram) fetch_engram ;;
    all)    fetch_exl3; echo; fetch_engram ;;
esac

echo
echo "==> 取得結果"
du -sh "${MODEL_PATH_DEST}" "${ENGRAM_PATH_DEST}" 2>/dev/null || true
