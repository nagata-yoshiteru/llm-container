#!/usr/bin/env bash
# =============================================================================
# patches/roce/ を完成させる (b12x ランタイム本体を別イメージから取り出す)。
#
#   SRC_IMAGE=vllm-dsv41:exl3b-roce ./scripts/build-roce-bundle.sh
#
# **2 台とも**で実行すること。詳細と前提は patches/roce/README.md。
#
# repo には b12x RoCEnante の「vLLM 側アダプタ 5 本」だけが入っている。
# ランタイム本体 (b12x パッケージ + /opt/b12x-roce) は Apache-2.0 だが
# PyPI に無いので、それを既に含むイメージ (上流の DS4 レーンのもの) から抜く。
#
# 揃ったら presets/roce.env を重ねて起動する:
#   sudo docker compose --env-file .env --env-file presets/roce.env --profile head up -d
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

SRC_IMAGE="${SRC_IMAGE:-vllm-dsv41:exl3b-roce}"
OUT=patches/roce
SP=/usr/local/lib/python3.12/dist-packages

for f in b12x_roce_all_reduce cuda_communicator parallel_state envs gpu_worker; do
    test -f "${OUT}/${f}.py" || { echo "${OUT}/${f}.py がありません (repo が壊れています)" >&2; exit 2; }
done

if ! sudo docker image inspect "${SRC_IMAGE}" >/dev/null 2>&1; then
    echo "SRC_IMAGE=${SRC_IMAGE} がローカルにありません。" >&2
    echo "b12x ランタイムを含むイメージを指定してください (patches/roce/README.md)。" >&2
    exit 2
fi

echo "==> ${SRC_IMAGE} から b12x ランタイムを取り出します"
cid=$(sudo docker create "${SRC_IMAGE}")
trap 'sudo docker rm "${cid}" >/dev/null 2>&1 || true' EXIT
sudo docker cp "${cid}:${SP}/b12x"                "${OUT}/b12x"
sudo docker cp "${cid}:${SP}/b12x-1.3.0.dist-info" "${OUT}/b12x-1.3.0.dist-info"
sudo docker cp "${cid}:/opt/b12x-roce"             "${OUT}/b12x-roce"

sudo find "${OUT}" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
sudo chown -R "$(id -u):$(id -g)" "${OUT}"
chmod -R a+rX "${OUT}"

echo
echo "==> ${OUT}: $(ls "${OUT}" | tr '\n' ' ')"
echo "==> b12x roce sha: $(cat "${OUT}/b12x-roce/B12X_ROCE_SHA" 2>/dev/null || echo '?')"
echo
echo "有効化 (2 台とも同じ組み合わせで / worker -> head):"
echo "  sudo docker compose --env-file .env --env-file presets/roce.env --profile worker up -d"
echo "  sudo docker compose --env-file .env --env-file presets/roce.env --profile head up -d"
