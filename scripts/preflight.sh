#!/usr/bin/env bash
# =============================================================================
# 起動前チェック (Mac Studio 1 台)。
#
#   ./scripts/preflight.sh
#
# ここで赤が出た状態で起動すると、186GB を数分かけて読んだ挙句に
# Metal の OOM (kIOGPUCommandBufferCallbackErrorOutOfMemory) か swap 地獄になる。
# 特に == メモリ == の wired 上限は必ず見ること。
# =============================================================================
set -uo pipefail
source "$(dirname "$0")/env.sh"

RC=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
ng()   { printf '  \033[31mNG\033[0m   %s\n' "$*"; RC=1; }

[ -f "${REPO_ROOT}/.env" ] || warn ".env がありません。すべて既定値で動かす (cp .env.example .env)"

echo "== ホスト =="
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
MEM_GIB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
echo "  $(hostname -s) / ${CHIP} / ${MEM_GIB} GiB / macOS $(sw_vers -productVersion)"
[ "$(uname -m)" = "arm64" ] && ok "Apple Silicon" || ng "arm64 ではありません (MLX は Apple Silicon 専用)"
if [ "${MEM_GIB}" -ge 256 ]; then
    ok "unified memory ${MEM_GIB} GiB"
else
    ng "unified memory ${MEM_GIB} GiB — 重み 186GB の oQ4e は 256GB 機が前提"
fi

echo
echo "== ランタイム =="
if [ -x "${RAPID_MLX}" ]; then
    V=$("${VENV_DIR}/bin/python" -c 'from importlib.metadata import version as v; print(v("rapid-mlx"))' 2>/dev/null)
    if [ "${V}" = "${RAPID_MLX_VERSION}" ]; then
        ok "rapid-mlx ${V}"
    else
        warn "rapid-mlx ${V:-?} (.env の RAPID_MLX_VERSION は ${RAPID_MLX_VERSION}) — ./scripts/setup.sh"
    fi
    "${VENV_DIR}/bin/python" -c 'import mlx.core as mx; assert mx.metal.is_available()' 2>/dev/null \
        && ok "MLX から Metal が見える" || ng "MLX から Metal が見えません"
else
    ng "${RAPID_MLX} がありません — ./scripts/setup.sh"
fi

echo
echo "== モデル =="
if [ -d "${SERVE_MODEL_PATH}" ] && [ -x "${VENV_DIR}/bin/python" ]; then
    if "${VENV_DIR}/bin/python" -I - "${SERVE_MODEL_PATH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / 'mlxvlm-preparation.json').read_text())
print(f'  mlx-vlm 用に準備済み (再量子化 {len(m["requantized"])} 個 / mlx-vlm {m["mlx_vlm"]})')
c = json.loads((p / 'config.json').read_text())
assert c['model_type'] == 'glm5_next'
assert (c['quantization']['bits'], c['quantization']['group_size']) == (4, 64)
w = json.loads((p / 'model.safetensors.index.json').read_text())['weight_map']
shards = set(w.values())
assert all((p / s).is_file() for s in shards)
gb = sum((p / s).stat().st_size for s in shards) / 1e9
print(f'  index が参照する {len(shards)} shards が存在 ({gb:.1f} GB)')
PY
    then
        ok "glm5_next / MLX 4bit g64 / checkpoint index"
    else
        ng "モデルが不完全 — ./scripts/fetch-model.sh → ./scripts/prepare-model.sh"
    fi
else
    ng "${SERVE_MODEL_PATH} がありません — ./scripts/fetch-model.sh → ./scripts/prepare-model.sh"
fi

echo
echo "== メモリ =="
# MLX が wired にできる上限 (max_recommended_working_set_size) は
# iogpu.wired_limit_mb で決まる。0 は OS 既定 (256GB 機で約 75% = 192GiB 前後)。
# 重み 173GiB + KV + 活性化 では足りないので引き上げる。再起動で 0 に戻る。
CUR_WIRED=$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)
if [ "${CUR_WIRED}" -ge "${WIRED_LIMIT_MB}" ]; then
    ok "iogpu.wired_limit_mb=${CUR_WIRED} (>= ${WIRED_LIMIT_MB})"
else
    ng "iogpu.wired_limit_mb=${CUR_WIRED} — sudo sysctl iogpu.wired_limit_mb=${WIRED_LIMIT_MB}"
fi
if [ "${WIRED_LIMIT_MB}" -gt $(( MEM_GIB * 1024 - 12288 )) ]; then
    warn "WIRED_LIMIT_MB=${WIRED_LIMIT_MB} は OS の取り分が 12GiB 未満。固まったら下げる"
fi

# 空き = free + inactive + speculative + purgeable (vm_stat のページ数)
FREE_GIB=$(vm_stat | awk -v ps="$(pagesize)" '
    /Pages free/ || /Pages inactive/ || /Pages speculative/ || /Pages purgeable/ {
        gsub(/\./, "", $NF); s += $NF }
    END { printf "%d", s * ps / 1024 / 1024 / 1024 }')
if [ "${FREE_GIB}" -ge 200 ]; then
    ok "空きメモリ 約 ${FREE_GIB} GiB"
else
    ng "空きメモリ 約 ${FREE_GIB} GiB (< 200 GiB)。他のアプリ / モデルサーバを止める"
fi
for P in ollama 'LM Studio' llama-server mlx_lm.server mlx_vlm.server; do
    pgrep -fq "${P}" && warn "${P} が動いている (メモリを取り合う)"
done
SWAP=$(sysctl -n vm.swapusage | awk '{print $6}')
echo "  swap used: ${SWAP}"

echo
echo "== API =="
if lsof -nP -iTCP:"${HOST_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    warn ":${HOST_PORT} は使用中 (起動済みなら ./scripts/serve.sh status)"
else
    ok ":${HOST_PORT} は空いている"
fi
if [ "${API_HOST}" != "127.0.0.1" ] && [ "${API_HOST}" != "localhost" ] && [ -z "${API_KEY}" ]; then
    ng "API_HOST=${API_HOST} で API_KEY が空。rapid-mlx は無認証の公開バインドを拒否する"
fi

echo
[ "${RC}" -eq 0 ] && echo "==> preflight PASS" || echo "==> preflight FAIL"
exit "${RC}"
