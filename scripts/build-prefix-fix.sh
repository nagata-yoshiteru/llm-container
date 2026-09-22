#!/usr/bin/env bash
# =============================================================================
# #18 prefix-cache 修正を「使い捨てコンテナの中で」当てて、パッチ済みの
# kv_cache_coordinator.py を patches/ に書き出す。
#
#   ./scripts/build-prefix-fix.sh            # .env の VLLM_IMAGE を使う
#   VLLM_IMAGE=... ./scripts/build-prefix-fix.sh
#
# **2 台とも**で実行すること (片方だけだと rank 間で挙動が変わる)。
#
# なぜコンテナの中で当てるのか:
#   patch_prefix_cache_draft_group.py は当てたあとに
#   `from vllm.v1.core.kv_cache_coordinator import _glm53_is_draft_swa_spec` で
#   self-check する。site-packages に置かれた状態でないと通らないので、
#   ホスト上のファイルに対しては当てられない。
#
# 何が直るのか:
#   公開イメージ sm121-v11-dflash2 (2026-08-28 ビルド) は、DFlash2 の draft
#   group が target group の hit 長を潰すバグを持つ。ブロック境界に揃った
#   プロンプトを再送しても prefix_cache_hits_total が 0 のまま = エージェントの
#   セッションが毎ターン会話全体を再 prefill する。
#   実測: 13K トークン反復の TTFT 21.1s -> 6.3s (-70%)。
#
# GPU は要求しない (self-check は import だけ)。もし import で落ちるようなら、
# 他のワークロードを止めてから `GPUS=1` を付けて再実行する。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_NAME=kv_cache_coordinator_prefix_fix.py
PATCH_NAME=patch_prefix_cache_draft_group.py
SP=/usr/local/lib/python3.12/dist-packages

if [ -z "${VLLM_IMAGE:-}" ] && [ -f .env ]; then
    VLLM_IMAGE=$(sed -n 's/^VLLM_IMAGE=//p' .env | tail -1)
fi
if [ -z "${VLLM_IMAGE:-}" ]; then
    echo "VLLM_IMAGE が決まりません (.env に無ければ環境変数で渡す)" >&2
    exit 2
fi
test -f "patches/${PATCH_NAME}" || { echo "patches/${PATCH_NAME} がありません" >&2; exit 2; }

GPU_ARGS=()
[ "${GPUS:-0}" = "1" ] && GPU_ARGS=(--gpus all)

echo "==> image: ${VLLM_IMAGE}"
echo "==> patches/ を rw でマウントして、コンテナ内で当てて書き出します"

sudo docker run --rm "${GPU_ARGS[@]}" \
    --entrypoint bash \
    -v "${PWD}/patches:/out" \
    "${VLLM_IMAGE}" -c "
set -e
python3 /out/${PATCH_NAME}
cp ${SP}/vllm/v1/core/kv_cache_coordinator.py /out/${OUT_NAME}
"

# コンテナが root で書くので、呼び出したユーザーに戻す
sudo chown "$(id -u):$(id -g)" "patches/${OUT_NAME}"

echo
if grep -q '_glm53_is_draft_swa_spec' "patches/${OUT_NAME}"; then
    echo "==> OK: patches/${OUT_NAME} ($(wc -l < "patches/${OUT_NAME}") 行)"
    echo "    entrypoint が PREFIX_FIX=1 のときにこれを site-packages へコピーします。"
    echo "    反映には両ノードの再起動が必要 (worker -> head)。"
else
    echo "==> FAIL: マーカーが見つかりません。出力を破棄します。" >&2
    rm -f "patches/${OUT_NAME}"
    exit 1
fi
