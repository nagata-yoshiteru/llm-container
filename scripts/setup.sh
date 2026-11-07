#!/usr/bin/env bash
# =============================================================================
# Rapid-MLX を repo 直下の venv に入れる。Mac Studio で最初に 1 回だけ実行する。
#
#   ./scripts/setup.sh
#
# Rapid-MLX は Python 3.12 以上が必要。macOS 同梱の python3 は 3.9 なので使えない。
# venv の作成には uv を使う (Python 3.12 本体も uv が取ってくる)。
# Docker は使わない。macOS の Docker から Metal GPU には触れないため。
# sudo は不要。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv が見つかりません。先に入れてください (どちらか):" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    echo "  brew install uv" >&2
    exit 1
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "==> venv を作成: ${VENV_DIR} (Python 3.12)"
    uv venv --python 3.12 "${VENV_DIR}"
fi

# rapid-mlx は mlx / mlx-lm / mlx-vlm の上限を自分で固定しているので、
# ここでは rapid-mlx 本体だけをバージョン固定する。
echo "==> rapid-mlx==${RAPID_MLX_VERSION} と hf CLI をインストール"
uv pip install --python "${VENV_DIR}/bin/python" \
    "rapid-mlx==${RAPID_MLX_VERSION}" \
    huggingface_hub

echo
"${RAPID_MLX}" --version
"${VENV_DIR}/bin/python" -c 'import mlx.core as mx; print("mlx", mx.__version__, "/ metal:", mx.metal.is_available())'
echo
echo "==> 次は ./scripts/fetch-model.sh で重みを取得する"
