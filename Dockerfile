# =============================================================================
# MiniMax-M3 / 3x DGX Spark 用の薄いラッパイメージ
#
# ベースは vLLM 公式の nightly (linux-arm64) を digest で固定したもの。
#   vllm/vllm-openai:nightly-700d39b5 / 2026-08-08 / CUDA 13.0.2
#
# なぜ NGC (nvcr.io/nvidia/vllm) ではないか:
#   NGC の最新は 26.07-py3 で、中身は upstream commit 092c4842 (2026-06-17)。
#   MiniMax-M3 の pipeline parallelism 対応は
#     d7c1821b5a [Model][MiniMax-M3] Add pipeline parallelism support (#45810)
#   で 2026-06-24 に入っており、7 日ぶん足りない。
#   M3 は num_key_value_heads=4 なので TP=3 も使えず、3 台構成では
#   PP=3 以外に選択肢が無い。NGC 26.08 が出たら戻ることを検討する。
#
# 重みは NVIDIA 公式の nvidia/MiniMax-M3-NVFP4 のまま (upstream の fused
# indexer 実装と組み合わせが一致する)。
#
#   sudo docker compose build
# =============================================================================
ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:5ee06541ff9fbd220c0a8f51608e72ea0da8d1a71ebf812af7d7c6b5a1aebbb7
FROM ${VLLM_BASE_IMAGE}

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ロードできる libnccl の版番号を出すヘルパ。ビルド時と起動時の両方で使う。
# 一行 python に押し込むと読めないうえ壊しやすいので実ファイルに切り出してある。
COPY scripts/nccl-version.py /usr/local/bin/nccl-version
RUN chmod +x /usr/local/bin/nccl-version

# 1) Ray — multi-node 用。入っていれば何もしない。
RUN set -eux; \
    if python3 -c "import ray" 2>/dev/null; then \
        echo "ray already present: $(python3 -c 'import ray; print(ray.__version__)')"; \
    else \
        python3 -m pip install --no-cache-dir "ray[default]==2.55.1"; \
    fi

# 2) NCCL を 2.30.7 に固定する。
#
# 3 ノードの switchless メッシュでは subnet-aware routing が要る。これが無いと
# NCCL が隣人ごとに正しい HCA を選べず、繋がっていないサブネットにダイヤルして
# ibv_modify_qp err 110 (Connection timed out) で死ぬ。対応が入ったのが 2.30.7。
#
# 公式 vLLM イメージが同梱する nvidia-nccl-cu13 の版は build 次第なので、
# ここで明示的に 2.30.7 を入れて決め打ちする。
RUN python3 -m pip install --no-cache-dir "nvidia-nccl-cu13==2.30.7"

# 3) system 側に 2.30.7 以上の libnccl があればそちらを優先する。
#
# DGX Spark では公式 vLLM イメージが pip 版 NCCL を先に読んで multi-node で
# ハングする報告がある (eugr/spark-vllm-docker の use-official-vllm mod が
# 同じ対処をしている)。system 側が無いか古ければ 2 の pip 版のまま使う。
RUN set -eux; \
    SYS=""; \
    for c in /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2; do \
        if [ -e "$c" ]; then SYS="$c"; break; fi; \
    done; \
    if [ -z "$SYS" ]; then \
        echo "system libnccl.so.2 なし。pip 版 2.30.7 をそのまま使う"; \
    else \
        SYSVER="$(nccl-version "$SYS")"; \
        echo "system NCCL: $SYS = $SYSVER"; \
        if [ "$SYSVER" -lt 23007 ]; then \
            echo "system NCCL が 2.30.7 未満。pip 版 2.30.7 を使う"; \
        else \
            for d in $(find /opt /usr -type d -path '*/nvidia/nccl/lib' 2>/dev/null); do \
                if [ -e "${d}/libnccl.so.2" ] && [ ! -e "${d}/libnccl.so.2.orig" ]; then \
                    mv "${d}/libnccl.so.2" "${d}/libnccl.so.2.orig"; \
                fi; \
                rm -f "${d}/libnccl.so.2"; \
                ln -sf "$SYS" "${d}/libnccl.so.2"; \
                echo "relinked ${d}/libnccl.so.2 -> $SYS"; \
            done; \
        fi; \
    fi

# 実際にロードされる版を焼き込み時に確認する。2.30.7 未満ならここで止める。
#
# pip 版は site-packages に置かれるだけで ld キャッシュには載らないので、
# soname (`libnccl.so.2`) では開けない。ヘルパが site-packages 側も含めて探す。
RUN set -eux; \
    V="$(nccl-version)"; \
    echo "BAKED_NCCL_VERSION ${V} ($((V / 10000)).$((V / 100 % 100)).$((V % 100)))"; \
    if [ "$V" -lt 23007 ]; then \
        echo "ERROR: NCCL < 2.30.7 — 3 ノードメッシュで ibv_modify_qp err 110 になる" >&2; \
        exit 1; \
    fi
