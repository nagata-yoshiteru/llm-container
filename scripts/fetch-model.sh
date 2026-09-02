#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4-Flash-Vision-Exp の重みを取得する (約 168GB / safetensors 48 shard)。
#
# 0731 と同じ 48 shard 構成で、増えているのは vision tower (BF16) の分だけ
# (166.9GB -> 167.8GB)。
#
# 2 台とも「同じ絶対パス」に落とす必要がある。head 側で実行したあと、
# worker 側でも同じコマンドを実行すること (rsync でコピーしてもよい)。
#
#   ./scripts/fetch-model.sh
#
# rootless / rootful どちらの docker とも無関係。sudo は不要。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ID="${REPO_ID:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"

# 保存先は .env の MODEL_PATH に合わせる (第 1 引数で上書き可)。
DEST="${1:-${MODEL_PATH:-}}"
if [ -z "${DEST}" ] && [ -f .env ]; then
    DEST=$(sed -n 's/^MODEL_PATH=//p' .env | tail -1)
fi
DEST="${DEST:-./models/DeepSeek-V4-Flash-Vision-Exp}"

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
echo "(48 shard / 約 168GB になっていれば OK)"
