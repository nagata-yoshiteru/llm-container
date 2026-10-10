#!/usr/bin/env bash
# =============================================================================
# dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4 (modelopt NVFP4) の重みを取得する。
#
#   ./scripts/fetch-model.sh          # MODEL_PATH に 121 shard (約 181GiB)
#   ./scripts/fetch-model.sh <path>   # 保存先を上書き
#
# DeepSeek-V4.1 と違い **追加テーブルは無い**。MTP の draft head
# (num_nextn_predict_layers=1) も同じ shard 群に入っているので、
# model_mtp.safetensors のような別ファイルは落とさない。
#
# 3 台とも「同じ絶対パス」に落とす必要がある。head で実行したあと、
# worker1 / worker2 でも同じコマンドを実行すること (rsync でコピーしてもよい)。
# resumable: 失敗したら同じコマンドを再実行すれば続きから。
#
# 181GiB は 1 台のディスクに置くので、事前に空きを確認すること
# (この repo の生成物は置き場を問わないが、3 台すべてで同じパスにする)。
#
# rootless / rootful どちらの docker とも無関係。sudo は不要。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_REPO="${MODEL_REPO:-dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4}"
EXPECTED_SHARDS=121
NEED_GIB=195

env_get() { [ -f .env ] && sed -n "s/^$1=//p" .env | tail -1 || true; }

MODEL_PATH_DEST="${1:-$(env_get MODEL_PATH)}"
MODEL_PATH_DEST="${MODEL_PATH_DEST:-./models/GLM-5.3-Flash-UNCENSORED-NVFP4}"

if ! command -v hf >/dev/null 2>&1; then
    echo "huggingface_hub CLI (hf) が見つかりません。以下でインストールしてください:" >&2
    echo "  pip install -U 'huggingface_hub[cli,hf_transfer]'" >&2
    exit 1
fi

# hf_transfer があれば有効化 (光スイッチ経由でも回線が太ければ効く)
python3 -c 'import hf_transfer' 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1

check_disk() {  # $1=宛先dir $2=必要GiB
    mkdir -p "$1"
    local avail_kb; avail_kb=$(df -Pk "$(dirname "$1")" | awk 'NR==2 {print $4}')
    if [ "${avail_kb}" -lt $(( $2 * 1024 * 1024 )) ]; then
        echo "WARN: $(dirname "$1") の空きが ${2}GiB 未満です ($((avail_kb / 1024 / 1024)) GiB)" >&2
    fi
}

echo "==> GLM-5.3-Flash-UNCENSORED-NVFP4: ${MODEL_REPO} -> ${MODEL_PATH_DEST} (約 181GiB)"
check_disk "${MODEL_PATH_DEST}" "${NEED_GIB}"

# ★ shard を positional で並べない。素の引数なしでリポジトリ全体を落とす
#   (旧 huggingface CLI でも新 hf でも同じ挙動。--include だと新版が無視して
#    一部だけ落ちる事故がある)。
hf download "${MODEL_REPO}" --local-dir "${MODEL_PATH_DEST}" --max-workers 8

n=$(find "${MODEL_PATH_DEST}" -maxdepth 1 -name 'model-*.safetensors' | wc -l)
echo "==> safetensors shards: ${n} (${EXPECTED_SHARDS} なら OK)"
[ "${n}" -eq "${EXPECTED_SHARDS}" ] || echo "WARN: shard 数が ${EXPECTED_SHARDS} と違います" >&2

for f in config.json generation_config.json model.safetensors.index.json; do
    [ -f "${MODEL_PATH_DEST}/${f}" ] || echo "WARN: ${f} がありません" >&2
done

echo
echo "==> 取得結果"
du -sh "${MODEL_PATH_DEST}" 2>/dev/null || true
echo
echo "==> これを 3 台とも同じパスに置くこと (head / worker1 / worker2)。"
echo "    ./scripts/preflight.sh で shard 数と空きメモリを確認してから起動する。"
