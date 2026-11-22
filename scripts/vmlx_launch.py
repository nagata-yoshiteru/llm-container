#!/usr/bin/env python3
"""vmlx を hotfix 付きで起動する (scripts/entrypoint.sh から呼ぶ)。

    venv-vmlx/bin/python -I scripts/vmlx_launch.py serve <model> ...

引数は vmlx の CLI にそのまま渡す。サーバ (uvicorn) は同じプロセスで動くので、
ここで当てたパッチがそのまま効く。fail-closed: 対象が見つからなければ起動しない。

hotfix: Anthropic 形式の thinking.type=adaptive を DSV4.1 で受ける
------------------------------------------------------------------
Claude Code などの Anthropic 形式のクライアントは、``thinking: {"type": "adaptive"}``
(考える深さをモデルに任せる) を送ることがある。vmlx 1.6.84 の /v1/messages は、これを
``chat_template_kwargs.thinking_mode="adaptive"`` に変換する
(api/anthropic_adapter.py ``to_chat_completion``)。その先の server.py
``_resolve_enable_thinking`` は、モデル登録の ``native_thinking_modes`` に adaptive を
持つ家系 (1.6.84 では MiniMax M3 だけ) 以外に HTTP 400 を返す。
DSV4.1 には adaptive にあたるモードが無いので、2026-11-09 の実機では
``deepseek_v4_1 does not support native thinking_mode='adaptive'`` で必ず失敗した。

DSV4.1 を読み込んでいるときだけ、adaptive を ``{"type": "enabled"}`` (予算なし) に
読み替えてから、元の変換に渡す。結果は「思考 ON」で、effort は次のとおり。
output_config.effort などで指定があればそれを使い、無ければエンコーダの既定 (high = 75)。
enabled / disabled や、ほかのモデルの経路には手を入れない。
"""
import importlib.util
import os
import sys
from importlib.metadata import version
from pathlib import Path

EXPECTED_VMLX = "1.6.84"
MARK = "_llm_container_dsv41_adaptive"


def install_dsv41_adaptive_thinking() -> None:
    have = version("vmlx")
    if have != EXPECTED_VMLX and os.environ.get("VMLX_HOTFIX_ALLOW_VERSION") != have:
        raise SystemExit(
            f"vmlx {have} は hotfix の検証対象 ({EXPECTED_VMLX}) と違う。"
            f"上流で直っていないか確認し、問題なければ VMLX_HOTFIX_ALLOW_VERSION={have} を付けて起動する")

    from vmlx_engine.api import anthropic_adapter as aa

    original = getattr(aa, "to_chat_completion", None)
    thinking_cls = getattr(aa, "AnthropicThinking", None)
    if original is None or thinking_cls is None or getattr(original, MARK, False):
        raise SystemExit("anthropic_adapter.to_chat_completion / AnthropicThinking が見つからない"
                         " (または二重適用)。hotfix を見直すこと")
    # 置き換え先の型が今も作れること (Literal から enabled が消えていればここで落ちる)
    thinking_cls(type="enabled")
    # server は CLI の serve が最初に import する。ここで先に import して初期化の順番を
    # 変えないよう、判定に使う関数があることだけをソースで確かめる。
    spec = importlib.util.find_spec("vmlx_engine.server")
    if spec is None or "def _is_loaded_dsv41_model(" not in Path(spec.origin).read_text():
        raise SystemExit("vmlx_engine.server._is_loaded_dsv41_model が見つからない。hotfix を見直すこと")

    state = {"logged": False}

    def to_chat_completion(req):
        thinking = getattr(req, "thinking", None)
        kind = thinking.get("type") if isinstance(thinking, dict) else getattr(thinking, "type", None)
        if kind == "adaptive":
            from vmlx_engine import server  # リクエスト時には読み込み済み

            if server._is_loaded_dsv41_model(getattr(req, "model", "") or ""):
                req = req.model_copy(update={"thinking": thinking_cls(type="enabled")})
                if not state["logged"]:
                    state["logged"] = True
                    print("[llm-container] DSV4.1: thinking.type=adaptive -> enabled "
                          "(初回だけ記録する)", flush=True)
        return original(req)

    setattr(to_chat_completion, MARK, True)
    to_chat_completion.__doc__ = original.__doc__
    aa.to_chat_completion = to_chat_completion
    print(f"[llm-container] hotfix: DSV4.1 accepts Anthropic thinking.type=adaptive as enabled "
          f"(vmlx {have})", flush=True)


if __name__ == "__main__":
    install_dsv41_adaptive_thinking()
    from vmlx_engine.cli import main

    sys.argv[0] = "vmlx"
    sys.exit(main())
