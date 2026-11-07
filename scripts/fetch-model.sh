#!/usr/bin/env bash
# =============================================================================
# GLM-5.3-Flash-UNCENSORED の MLX 4bit (oQ4e) を取得する。
#
#   ./scripts/fetch-model.sh          # .env の MODEL_PATH に取得 (約 186GB)
#   ./scripts/fetch-model.sh <path>   # 保存先を上書き
#
# DGX 版で使っていた dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 は modelopt NVFP4
# (CUDA 専用) なので Mac では読めない。代わりに、同じ dealignai の uncensored
# 重み (FP8 版) を Solstice-AI が MLX に変換したものを使う:
#   - 4-bit affine / group_size=64 (~4.6 bpw)
#   - lm_head / router / gate は 8-bit、vision tower などは BF16 のまま
#   - safetensors 38 shard、合計 185.7GB
#
# 同じ repo にある GGUF 2 本 (mmproj-BF16.gguf 1.2GB と
# speculative/GLM-5.3-Flash-DFlash2-bf16.gguf 2.4GB) は llama.cpp 用。
# Rapid-MLX では使わないので取得しない。
#
# revision は commit SHA で固定する。上流が重みを差し替えても勝手に変わらない。
# 失敗したら同じコマンドを再実行すれば続きから取得する。sudo は不要。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

EXPECTED_SHARDS=38
NEED_GIB=195
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

echo "==> ${MODEL_REPO}@${MODEL_REVISION:0:8} -> ${DEST} (約 186GB)"
"${HF}" download "${MODEL_REPO}" \
    --revision "${MODEL_REVISION}" \
    --local-dir "${DEST}" \
    --exclude '*.gguf' \
    --max-workers 8

n=$(find "${DEST}" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l | tr -d ' ')
echo "==> safetensors shards: ${n} (${EXPECTED_SHARDS} なら OK)"
[ "${n}" -eq "${EXPECTED_SHARDS}" ] || echo "WARN: shard 数が ${EXPECTED_SHARDS} と違います" >&2

for f in config.json model.safetensors.index.json tokenizer.json chat_template.jinja; do
    [ -f "${DEST}/${f}" ] || echo "WARN: ${f} がありません" >&2
done

# 元の dealignai 版の chat template (Solstice 版は独自 system prompt を常に注入する)
echo "==> chat template: ${TEMPLATE_REPO}@${TEMPLATE_REVISION:0:8} -> ${DEST}/upstream-dealignai"
"${HF}" download "${TEMPLATE_REPO}" chat_template.jinja \
    --revision "${TEMPLATE_REVISION}" \
    --local-dir "${DEST}/upstream-dealignai"

echo
du -sh "${DEST}" 2>/dev/null || true
echo "==> 次は ./scripts/prepare-model.sh (mlx-vlm 用の serving ディレクトリを作る)"
