#!/usr/bin/env bash
# =============================================================================
# nvidia/Qwen3.8-Flash-Next-NVFP4 を .env の MODEL_PATH に revision 固定で落とす。
#
#   単一ノード (2x RTX PRO 6000) 構成なので、実行するのは 1 回だけ。
#   TP=2 の 2 プロセスは同じコンテナから同じマウントを読む。
#
#   使い方:  ./scripts/fetch-model.sh
#
#   sudo は不要。docker も使わない。約 124 GiB。
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "ERROR: .env が無い。cp .env.example .env してから実行すること。" >&2
    exit 1
fi

# .env から必要な値だけ読む (コメント / 空行は無視)
set -a
# shellcheck disable=SC1091
. ./.env
set +a

: "${MODEL_REPO:?MODEL_REPO must be set in .env}"
: "${MODEL_PATH:?MODEL_PATH must be set in .env}"
: "${MODEL_REVISION:?MODEL_REVISION must be set in .env}"

# hf CLI は同梱の venv を優先する
if [ -x ./venv/bin/hf ]; then
    HF_BIN=./venv/bin/hf
elif command -v hf >/dev/null 2>&1; then
    HF_BIN=hf
else
    echo "ERROR: hf CLI が無い。python3 -m venv venv && ./venv/bin/pip install -U huggingface_hub" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 空き容量の確認。124 GiB 必要。途中で切れると再開はできるが時間を無駄にする。
# ---------------------------------------------------------------------------
NEED_GIB=130
AVAIL_GIB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GIB}" -lt "${NEED_GIB}" ]; then
    echo "ERROR: 空き容量が足りない (${AVAIL_GIB} GiB / 必要 ${NEED_GIB} GiB)" >&2
    exit 1
fi
echo "[fetch] 空き ${AVAIL_GIB} GiB / 必要 ${NEED_GIB} GiB"

mkdir -p "${MODEL_PATH}"

echo "[fetch] repo=${MODEL_REPO}"
echo "[fetch] rev =${MODEL_REVISION}"
echo "[fetch] dest=${MODEL_PATH}"

# --max-workers は NVMe とページキャッシュを食い潰さない程度に。
# HF_TOKEN は gated ではないので空でよい (レート制限回避には効く)。
HF_TOKEN="${HF_TOKEN:-}" "${HF_BIN}" download \
    "${MODEL_REPO}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${MODEL_PATH}" \
    --max-workers 8

# ---------------------------------------------------------------------------
# entrypoint が起動時に見るファイルをここでも確認しておく
# ---------------------------------------------------------------------------
for f in config.json model.safetensors.index.json tokenizer.json; do
    if [ ! -f "${MODEL_PATH}/${f}" ]; then
        echo "ERROR: ${MODEL_PATH}/${f} が落ちていない。ダウンロードをやり直すこと。" >&2
        exit 1
    fi
done

echo "[fetch] 完了: $(du -sh "${MODEL_PATH}" | cut -f1)"
echo "[fetch] 次: ./scripts/preflight.sh"
