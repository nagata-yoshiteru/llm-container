#!/usr/bin/env bash
# =============================================================================
# MiniMax-M3 NVFP4 の重みを取得する (約 250GB / safetensors 88 shard)。
#
# 3 台とも「同じ絶対パス」に落とす必要がある。head 側で実行したあと、
# 残り 2 台でも同じコマンドを実行すること (rsync でコピーしてもよい)。
#
#   ./scripts/fetch-model.sh
#
# 既定は NVIDIA 公式の nvidia/MiniMax-M3-NVFP4。NGC の vLLM (upstream) は
# indexer を qkv に畳む fused 実装なので、`self_attn.index_k_proj` 命名の
# こちらでないと `Shard id for QKVParallelLinear ... got shard id index_k` で落ちる。
#
# tonyd2wild の chthonic fork イメージに載せ替えるなら luke 版 (非 fused):
#   REPO_ID=lukealonso/MiniMax-M3-NVFP4 ./scripts/fetch-model.sh ./models/MiniMax-M3-NVFP4-luke
#
# rootless / rootful どちらの docker とも無関係。sudo は不要。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ID="${REPO_ID:-nvidia/MiniMax-M3-NVFP4}"

# 保存先は .env の MODEL_PATH に合わせる (第 1 引数で上書き可)。
DEST="${1:-${MODEL_PATH:-}}"
if [ -z "${DEST}" ] && [ -f .env ]; then
    DEST=$(sed -n 's/^MODEL_PATH=//p' .env | tail -1)
fi
DEST="${DEST:-./models/MiniMax-M3-NVFP4}"

if ! command -v hf >/dev/null 2>&1; then
    echo "huggingface_hub CLI (hf) が見つかりません。以下でインストールしてください:" >&2
    echo "  pip install -U 'huggingface_hub[cli,hf_transfer]'" >&2
    exit 1
fi

mkdir -p "$(dirname "${DEST}")"
AVAIL_KB=$(df -Pk "$(dirname "${DEST}")" | awk 'NR==2 {print $4}')
if [ "${AVAIL_KB}" -lt $((300 * 1024 * 1024)) ]; then
    echo "WARN: $(dirname "${DEST}") の空きが 300GB 未満です ($((AVAIL_KB / 1024 / 1024)) GB)" >&2
fi

mkdir -p "${DEST}"

# hf_transfer があれば有効化 (200Gbps を活かすなら効く)
python3 -c 'import hf_transfer' 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> ${REPO_ID} -> ${DEST}"
hf download "${REPO_ID}" --local-dir "${DEST}" --max-workers 8

echo
echo "==> 取得結果"
du -sh "${DEST}"
N=$(ls "${DEST}"/*.safetensors 2>/dev/null | wc -l)
echo "safetensors shards: ${N}"
if [ "${REPO_ID}" = "nvidia/MiniMax-M3-NVFP4" ]; then
    echo "(88 shard / 約 250GB になっていれば OK)"
fi
