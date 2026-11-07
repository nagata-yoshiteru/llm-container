#!/usr/bin/env python3
"""rapid-mlx を hotfix 付きで起動する (scripts/entrypoint.sh から呼ぶ)。

    venv/bin/python -I scripts/rapid_mlx_launch.py serve <model> ...

引数は rapid-mlx の CLI にそのまま渡す。サーバは同じプロセスで動くので、
ここで当てたパッチがそのまま効く。fail-closed: 対象が見つからなければ起動しない。

hotfix: MLLM (vision) lane の decode に再帰キャッシュの eval barrier を足す
-------------------------------------------------------------------------
rapid-mlx 1.0.2 の text lane は、hybrid (再帰) キャッシュの関数的な更新が遅延グラフを
保持し続けるのを、数ステップごとの ``mx.eval([c.state for c in cache])`` で抑えている
(scheduler.py ``_RECURRENT_CACHE_MATERIALIZE_INTERVAL``、上流 #1834 / #2834)。
MLLM lane の decode (``MLLMBatchGenerator._step``) にはこの barrier が無い
(prefill の chunk 間にはある)。GLM-5.3-Flash は vision 込みなので MLLM lane で動き、
KDA 34 層 + DSA 11 層のキャッシュ更新が 1 トークンあたり約 45 個の Metal handle を
残し続ける。2026-11-07 の実機では、約 11,200 トークンを生成したところで
``[metal::malloc] Resource limit (499000) exceeded`` (insufficient_memory) で落ちた。
素の mlx-vlm で同じモデルを 12,500 トークン decode しても落ちないことも確認した。
上流 issue #3270 (vision lane が handle 上限まで走る) と同じ系統。

text lane と同じ予算 (同時実行数 × 間隔 = 64) で、_step の後に state を eval する。
"""
import os
import sys
from importlib.metadata import version

EXPECTED_RAPID_MLX = "1.0.2"
HANDLE_BUDGET = 64   # scheduler._RECURRENT_MATERIALIZE_HANDLE_BUDGET と同じ
MIN_INTERVAL = 8     # scheduler._RECURRENT_CACHE_MATERIALIZE_INTERVAL と同じ
MAX_INTERVAL = 64    # scheduler._RECURRENT_MATERIALIZE_MAX_INTERVAL と同じ


def install_mllm_decode_barrier() -> None:
    have = version("rapid-mlx")
    if have != EXPECTED_RAPID_MLX and os.environ.get("RAPID_MLX_HOTFIX_ALLOW_VERSION") != have:
        raise SystemExit(
            f"rapid-mlx {have} は hotfix の検証対象 ({EXPECTED_RAPID_MLX}) と違う。"
            f"上流で直っていないか確認し、問題なければ RAPID_MLX_HOTFIX_ALLOW_VERSION={have} を付けて起動する")

    import mlx.core as mx
    from rapid_mlx import mllm_batch_generator as mbg

    gen = getattr(mbg, "MLLMBatchGenerator", None)
    original = getattr(gen, "_step", None)
    if original is None or getattr(original, "_llm_container_barrier", False):
        raise SystemExit("MLLMBatchGenerator._step が見つからない (または二重適用)。hotfix を見直すこと")

    def _step(self, input_tokens, cache, requests=None):
        result = original(self, input_tokens, cache, requests)
        active = max(1, int(input_tokens.shape[0]))
        interval = max(MIN_INTERVAL, min(MAX_INTERVAL, HANDLE_BUDGET // active))
        depth = getattr(self, "_llm_container_barrier_depth", 0) + 1
        if depth >= interval:
            # 今ステップの出力 (result) は呼び出し側が async_eval する。ここでは
            # キャッシュの state だけを確定させ、遅延グラフを切る。
            mx.eval([c.state for c in cache])
            depth = 0
        self._llm_container_barrier_depth = depth
        return result

    _step._llm_container_barrier = True
    _step.__doc__ = original.__doc__
    gen._step = _step
    print(f"[llm-container] hotfix: MLLM decode recurrent-cache barrier "
          f"(rapid-mlx {have}, interval {MIN_INTERVAL}-{MAX_INTERVAL}, budget {HANDLE_BUDGET})",
          flush=True)


if __name__ == "__main__":
    install_mllm_decode_barrier()
    from rapid_mlx.dependency_repair import cli_entrypoint
    sys.argv[0] = "rapid-mlx"
    sys.exit(cli_entrypoint())
