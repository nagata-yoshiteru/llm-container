#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1-Flash-1017-UNCENSORED-JANG_2K (dealignai) を取得する。
#
#   ./scripts/fetch-model.sh          # .env の MODEL_PATH に取得 (約 424GB)
#   ./scripts/fetch-model.sh <path>   # 保存先を上書き
#   VERIFY=1 ./scripts/fetch-model.sh # 取得後に SHA256SUMS で全ファイルを検証 (数十分)
#
# 中身 (revision f914c4b1):
#   - 本体 45 shard (共通 5 + routed expert 40) / 211.6GB (197GiB)。起動時にメモリへ載る (wired)
#   - engram/ 2 shard / 203GB。Engram n-gram テーブル (FP8)。実行中にディスクから
#     行単位で読む (mmap)。メモリには載らないが、置き場は内蔵 SSD が望ましい
#   - dspark/ 8.9GB。DSpark (MTP) の draft。先読みを使うときにメモリへ載る
#   - encoding/ は DeepSeek 純正の chat encoder (Python)。*.pyc は取得しない
#
# revision は commit SHA で固定する。失敗したら同じコマンドを再実行すれば続きから。
# sudo は不要。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

EXPECTED_SHARDS=45
NEED_GIB=400
DEST=$(abs_path "${1:-${MODEL_PATH}}")

if [ ! -x "${HF}" ]; then
    echo "${HF} がありません。先に ./scripts/setup.sh を実行してください。" >&2
    exit 1
fi

mkdir -p "${DEST}"
AVAIL_GIB=$(df -Pk "${DEST}" | awk 'NR==2 {printf "%d", $4/1024/1024}')
if [ "${AVAIL_GIB}" -lt "${NEED_GIB}" ]; then
    echo "WARN: ${DEST} の空きが ${NEED_GIB}GiB 未満です (${AVAIL_GIB} GiB)" >&2
fi

echo "==> ${MODEL_REPO}@${MODEL_REVISION:0:8} -> ${DEST} (約 424GB)"
"${HF}" download "${MODEL_REPO}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${DEST}" \
    --exclude '*.pyc' \
    --max-workers 8

# 本体は model-0000N-of-00045 (5 本) + model-routed-000NN (routed expert、40 本) の 45 本
n=$(find "${DEST}" -maxdepth 1 -name 'model-*.safetensors' | wc -l | tr -d ' ')
echo "==> 本体 shards: ${n} (${EXPECTED_SHARDS} なら OK)"
[ "${n}" -eq "${EXPECTED_SHARDS}" ] || echo "WARN: shard 数が ${EXPECTED_SHARDS} と違います" >&2
for f in config.json jang_config.json model.safetensors.index.json tokenizer.json \
         encoding/encoding.py dspark/dspark.safetensors \
         engram/model-00047-of-00048.safetensors engram/model-00048-of-00048.safetensors; do
    [ -f "${DEST}/${f}" ] || echo "WARN: ${f} がありません" >&2
done

if [ "${VERIFY:-0}" = "1" ]; then
    echo "==> SHA256SUMS で検証 (約 424GB を読む)"
    # SHA256SUMS は "<hash>  <path>" 形式。*.pyc は取得していないので除く。
    (cd "${DEST}" && grep -v '\.pyc$' SHA256SUMS | shasum -a 256 -c --quiet) && echo "    OK"
fi

echo
du -sh "${DEST}" 2>/dev/null || true
echo "==> ./scripts/preflight.sh で wired メモリ上限とモデルを確認してから起動する。"
