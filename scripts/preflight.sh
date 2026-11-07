#!/usr/bin/env bash
# =============================================================================
# 起動前チェック (Mac Studio 1 台)。
#
#   ./scripts/preflight.sh
#
# ここで赤が出た状態で起動すると、197GiB を数分かけて読んだ挙句に
# Metal の OOM か swap 地獄になる。特に == メモリ == の wired 上限は必ず見ること。
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
    ng "unified memory ${MEM_GIB} GiB — JANG_2K (常駐 197GiB) は 256GB 機が前提"
fi

echo
echo "== ランタイム =="
if [ -x "${VMLX}" ]; then
    V=$("${VENV_DIR}/bin/python" -c 'from importlib.metadata import version as v; print(v("vmlx"), v("jang"))' 2>/dev/null)
    if [ "${V}" = "${VMLX_VERSION} ${JANG_VERSION}" ]; then
        ok "vmlx ${VMLX_VERSION} / jang ${JANG_VERSION}"
    else
        warn "vmlx/jang = ${V:-?} (.env は ${VMLX_VERSION} / ${JANG_VERSION}) — ./scripts/setup.sh"
    fi
    "${VENV_DIR}/bin/python" -I -c 'import mlx.core as mx, jang_tools.dsv4_1.mlx_model; assert mx.metal.is_available()' 2>/dev/null \
        && ok "MLX から Metal が見え、jang_tools.dsv4_1 を import できる" \
        || ng "MLX の Metal か jang_tools.dsv4_1 が使えません"
else
    ng "${VMLX} がありません — ./scripts/setup.sh"
fi

echo
echo "== モデル =="
if [ -d "${MODEL_PATH}" ] && [ -x "${VENV_DIR}/bin/python" ]; then
    if "${VENV_DIR}/bin/python" -I - "${MODEL_PATH}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
c = json.loads((p / 'config.json').read_text())
assert c['model_type'] == 'deepseek_v41'
j = json.loads((p / 'jang_config.json').read_text())
print(f'  JANG profile {j["profile"]} (routed experts 平均 {j["affine_bits"]["routed_expert_mean"]} bit)')
w = json.loads((p / 'model.safetensors.index.json').read_text())['weight_map']
shards = set(w.values())
assert all((p / s).is_file() for s in shards)
gib = sum((p / s).stat().st_size for s in shards) / 2**30
rt = c.get('runtime', {})
engram = sorted((p / rt.get('engram_dir', 'engram')).glob('*.safetensors'))
assert len(engram) == 2
eg = sum(f.stat().st_size for f in engram) / 2**30
ds = p / rt.get('dspark_dir', 'dspark') / 'dspark.safetensors'
assert ds.is_file()
print(f'  本体 {len(shards)} shards {gib:.1f} GiB (常駐) / Engram {eg:.1f} GiB (ディスク) / '
      f'DSpark {ds.stat().st_size / 2**30:.1f} GiB')
PY
    then
        ok "deepseek_v41 / JANG_2K / 本体・Engram・DSpark"
    else
        ng "モデルが不完全 — ./scripts/fetch-model.sh を再実行"
    fi
    # Engram は実行中に行単位で読む。外付け/ネットワークドライブだと prefill が遅くなる。
    DEV=$(df "${MODEL_PATH}" | awk 'NR==2 {print $1}')
    diskutil info "${DEV}" 2>/dev/null | grep -q -E 'Device Location: +Internal|Internal: +Yes' \
        && ok "モデルは内蔵ストレージ上 (Engram をディスクから読む)" \
        || warn "モデルが内蔵ストレージ上か確認できない (Engram はディスクから読む)"
else
    ng "${MODEL_PATH} がありません — ./scripts/fetch-model.sh"
fi

echo
echo "== メモリ =="
# MLX が wired にできる上限 (max_recommended_working_set_size) は
# iogpu.wired_limit_mb で決まる。0 は OS 既定 (256GB 機で約 75% = 192GiB 前後)。
# 本体 197GiB + DSpark 8.3GiB + KV では足りないので引き上げる。再起動で 0 に戻る。
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
if [ "${FREE_GIB}" -ge 215 ]; then
    ok "空きメモリ 約 ${FREE_GIB} GiB"
else
    ng "空きメモリ 約 ${FREE_GIB} GiB (< 215 GiB)。他のアプリ / モデルサーバを止める"
fi
# GLM 用ブランチの Rapid-MLX サーバと同時には載らない (173 + 197 GiB)
for P in rapid-mlx rapid_mlx_launch ollama 'LM Studio' llama-server mlx_lm.server mlx_vlm.server; do
    pgrep -fq "${P}" && ng "${P} が動いている (同時には載らない。先に止める)"
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
    ng "API_HOST=${API_HOST} で API_KEY が空。LAN に出すなら API_KEY を設定する"
fi

echo
[ "${RC}" -eq 0 ] && echo "==> preflight PASS" || echo "==> preflight FAIL"
exit "${RC}"
