#!/usr/bin/env bash
# =============================================================================
# GLM-5.3-Flash-NVFP4 の重みを取得する (約 198GB / 10 shard + MTP head)。
#
# 2 台とも「同じ絶対パス」に落とす必要がある。head 側で実行したあと、
# worker 側でも同じコマンドを実行すること (rsync でコピーしてもよい)。
#
#   ./scripts/fetch-model.sh            # メインモデル -> .env の MODEL_PATH
#   ./scripts/fetch-model.sh draft      # DFlash2 drafter (2.2GB, 任意)
#
# DFlash2 drafter は presets/dflash2.env (高速化) で使う場合だけ必要。
# ライセンス: CC-BY-NC-ND-4.0 (非商用・改変禁止) — 商用利用は確認すること。
#
# rootless / rootful どちらの docker とも無関係。sudo は不要。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
case "${MODE}" in
    draft)
        REPO_ID="incoai/GLM-5.3-Flash-DFlash2"
        DEFAULT_DEST="./models/GLM-5.3-Flash-DFlash2"
        EXPECT_NOTE="model_mtp.safetensors が無く、drafter の config が揃っていれば OK"
        ;;
    "")
        REPO_ID="${REPO_ID:-RedHatAI/GLM-5.3-Flash-NVFP4}"
        DEFAULT_DEST="./models/GLM-5.3-Flash-NVFP4"
        EXPECT_NOTE="10 shard + model_mtp.safetensors (約 198GB) になっていれば OK"
        ;;
    *)
        echo "unknown mode: ${MODE} (draft のみ)" >&2
        exit 2
        ;;
esac

# 保存先は .env の MODEL_PATH / DRAFT_PATH に合わせる (引数で上書き可)。
DEST="${2:-}"
if [ -z "${DEST}" ] && [ -f .env ]; then
    KEY="MODEL_PATH"
    [ "${MODE}" = "draft" ] && KEY="DRAFT_PATH"
    DEST=$(sed -n "s/^${KEY}=//p" .env | tail -1)
fi
DEST="${DEST:-${DEFAULT_DEST}}"

if ! command -v hf >/dev/null 2>&1; then
    echo "huggingface_hub CLI (hf) が見つかりません。以下でインストールしてください:" >&2
    echo "  pip install -U 'huggingface_hub[cli,hf_transfer]'" >&2
    exit 1
fi

AVAIL_KB=$(df -Pk "$(dirname "${DEST}")" | awk 'NR==2 {print $4}')
if [ "${AVAIL_KB}" -lt $((200 * 1024 * 1024)) ]; then
    echo "WARN: $(dirname "${DEST}") の空きが 200GB 未満です ($((AVAIL_KB / 1024 / 1024)) GB)" >&2
fi

mkdir -p "${DEST}"

# hf_transfer があれば有効化 (200Gbps を活かすなら効く)
python3 -c 'import hf_transfer' 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> ${REPO_ID} -> ${DEST}"
hf download "${REPO_ID}" --local-dir "${DEST}" --max-workers 8

echo
echo "==> 取得結果"
du -sh "${DEST}"
ls "${DEST}"/model-*.safetensors 2>/dev/null | wc -l | xargs echo "safetensors shards:"
echo "(${EXPECT_NOTE})"
