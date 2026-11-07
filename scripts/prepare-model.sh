#!/usr/bin/env bash
# =============================================================================
# 取得した oQ4e (MODEL_PATH) から、mlx-vlm で読める serving ディレクトリ
# (SERVE_MODEL_PATH) を作る。fetch-model.sh のあとに 1 回だけ実行する。
#
#   ./scripts/prepare-model.sh
#
# shard は hardlink なので追加のディスク消費は数 GB 程度。数分かかる。
# 理由と中身は scripts/prepare-glm53-mlx.py の先頭コメントを参照。
# 同じディレクトリへの再実行は検証だけ行う。sudo は不要。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

[ -x "${VENV_DIR}/bin/python" ] || { echo "先に ./scripts/setup.sh" >&2; exit 1; }
[ -f "${MODEL_PATH}/config.json" ] || { echo "先に ./scripts/fetch-model.sh" >&2; exit 1; }

exec "${VENV_DIR}/bin/python" -I "${REPO_ROOT}/scripts/prepare-glm53-mlx.py" \
    "${MODEL_PATH}" "${SERVE_MODEL_PATH}"
