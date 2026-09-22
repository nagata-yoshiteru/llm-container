#!/usr/bin/env bash
# =============================================================================
# GPU のクロッククランプ検出。**ベンチを取る前に 2 台とも必ず実行する。**
#
#   ./scripts/gputest.sh
#
# GB10 は不正なリセット (watchdog リブート / `nvidia-smi -r` 後の CUDA 実行) の
# あと、プラットフォームの電力バジェットが約 14W のフォールバック値に張り付く
# ことがある。負荷時 611〜890MHz / bf16 で 26〜33 TFLOPS しか出なくなり、
# 測定値が全部 2.5 倍遅くなる。上流はこれで一晩のベンチを丸ごと捨てている。
#
# 復帰方法は **AC 電源を抜く** (完全な電源サイクル) だけ。
# `nvidia-smi -r` / `-lgc` / `-ac` / `-pl` と通常の再起動では戻らない。
# しかも `nvidia-smi -r` 後に再起動せず CUDA を回すと SMMU timeout で GPU が
# fault する。やらないこと。
#
# 判定: 健全なら 65〜82 TFLOPS。**50 TFLOPS 未満はクランプ。**
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VLLM_IMAGE:-}" ] && [ -f .env ]; then
    VLLM_IMAGE=$(sed -n 's/^VLLM_IMAGE=//p' .env | tail -1)
fi
if [ -z "${VLLM_IMAGE:-}" ]; then
    echo "VLLM_IMAGE が決まりません (.env に無ければ環境変数で渡す)" >&2
    exit 2
fi

echo "== $(hostname) =="
nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader | sed 's/^/  now: /'

OUT=$(sudo docker run --rm --gpus all --entrypoint python3 "${VLLM_IMAGE}" -c "
import torch, time
a = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize(); t = time.time(); n = 0
while time.time() - t < 4:
    c = a @ b; n += 1
torch.cuda.synchronize(); dt = time.time() - t
x = torch.randn(64 * 1024 * 1024, device='cuda', dtype=torch.bfloat16)
torch.cuda.synchronize(); t2 = time.time()
for _ in range(20):
    y = x * 1.0001
torch.cuda.synchronize()
tflops = n * 2 * 4096 ** 3 / dt / 1e12
membw = 20 * 2 * x.numel() * 2 / (time.time() - t2) / 1e9
print(f'{tflops:.1f} TFLOPS, membw {membw:.0f} GB/s')
" 2>&1 | grep TFLOPS)

echo "  ${OUT}"
TF=${OUT%% *}
if awk "BEGIN{exit !(${TF} < 50)}"; then
    echo
    echo "  *** クランプ検出 (${TF} TFLOPS < 50) ***"
    echo "  このノードの測定値は信用できません。AC 電源を抜いて入れ直すこと。"
    echo "  nvidia-smi -r は使わないこと (再起動せず CUDA を回すと GPU が fault する)。"
    exit 1
fi
echo "  OK (健全な範囲は 65〜82 TFLOPS)"
