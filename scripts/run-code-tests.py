#!/usr/bin/env python3
"""bench-glm53-eval.py が保存した生成コード (<run>-generated.py) を、glm53_eval_tasks.py のテストで実行する。

    venv-vmlx/bin/python -I scripts/run-code-tests.py --reviewed results/<dir>

**生成コードは、先に人が読んでから実行すること** (--reviewed はその確認)。
このスクリプトは中身を検査しない。ファイル・ネットワーク・プロセスに触れるコードが
混ざっていたら、その <run>-generated.py を退避してから実行し、手で FAIL にする。
各テストは結果ディレクトリとは別の一時ディレクトリで、python -I・タイムアウト付きで実行する。
結果は <dir>/code-grades.json に追記する (summarize-glm53-eval.py が読む)。
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_eval_tasks import CODE, HARD_CODE  # noqa: E402

ALL_CODE = {**CODE, **HARD_CODE}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reviewed", action="store_true", help="生成コードを読んで確認した")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("生成コードを読んで確認してから --reviewed を付けて実行すること")
    grades_path = args.directory / "code-grades.json"
    grades = json.loads(grades_path.read_text()) if grades_path.exists() else {}
    with tempfile.TemporaryDirectory(prefix="code-tests-") as work:
        for src in sorted(args.directory.glob("*-generated.py")):
            run = src.name.removesuffix("-generated.py")
            task = re.sub(r"-r\d+$", "", run)
            if task not in ALL_CODE:
                continue
            prog = Path(work) / f"{run}_test.py"
            prog.write_text(src.read_text() + "\n\n# ---- test from scripts/glm53_eval_tasks.py ----\n"
                            + ALL_CODE[task][1])
            t = time.time()
            try:
                p = subprocess.run([sys.executable, "-I", str(prog)], cwd=work, capture_output=True,
                                   text=True, timeout=args.timeout)
                ok, err = p.returncode == 0, p.stderr[-600:]
            except subprocess.TimeoutExpired:
                ok, err = False, f"timeout {args.timeout}s"
            grades[run] = {"pass": ok, "seconds": round(time.time() - t, 2), "stderr_tail": err,
                           "method": "generated code reviewed manually, then run with repo test "
                                     f"(python -I, {args.timeout}s timeout)"}
            print(f"{run}: {'PASS' if ok else 'FAIL'} ({grades[run]['seconds']}s) {err[-160:]}", flush=True)
    grades_path.write_text(json.dumps(grades, indent=2) + "\n")


if __name__ == "__main__":
    main()
