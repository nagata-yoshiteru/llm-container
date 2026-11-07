#!/usr/bin/env bash
# =============================================================================
# vmlx (+ jang_tools) を repo 直下の venv-vmlx に入れる。最初に 1 回だけ実行する。
#
#   ./scripts/setup.sh
#
# DeepSeek-V4.1 の JANG 形式は Rapid-MLX では読めない。モデルカードが指定する
# vmlx (jjang-ai) と、その DSV4.1 runtime を持つ jang (import 名は jang_tools) を使う。
# vmlx は mlx-vlm 0.5 系 / transformers 5.18+ を要求し、GLM 用の Rapid-MLX
# (mlx-vlm 0.7.2) とぶつかるので、venv は ./venv とは別にする。
#
# Python 3.12 は uv が取ってくる (macOS 同梱の python3 は 3.9)。sudo は不要。
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

# vmlx が mlx / mlx-vlm / huggingface_hub (hf CLI) を引いてくる。
# mlx-lm は固定する: vmlx 1.6.84 は mlx_lm.generate.SequenceStateMachine を使うが、
# 依存指定は mlx-lm>=0.31.3 だけで、0.32.0 以降ではこのクラスが消えて起動時に ImportError になる。
echo "==> vmlx==${VMLX_VERSION} / jang==${JANG_VERSION} / mlx-lm==${MLX_LM_VERSION} をインストール"
uv pip install --python "${VENV_DIR}/bin/python" \
    "vmlx==${VMLX_VERSION}" \
    "jang==${JANG_VERSION}" \
    "mlx-lm==${MLX_LM_VERSION}"

echo
"${VENV_DIR}/bin/python" -I - <<'PY'
from importlib.metadata import version
import mlx.core as mx
import jang_tools.dsv4_1.mlx_model  # noqa: F401  (DSV4.1 runtime があること)
print(", ".join(f"{p} {version(p)}" for p in ("vmlx", "jang", "mlx", "mlx-lm", "mlx-vlm")),
      "/ metal:", mx.metal.is_available())
PY
echo
echo "==> 次は ./scripts/fetch-model.sh で重みを取得する"
