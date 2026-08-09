#!/usr/bin/env python3
"""ロードできる libnccl の版番号 (例: 23007 = 2.30.7) を stdout に出す。

引数にパスを渡せばそれを、省略すれば以下の順で試す:

  1. pip の nvidia-nccl-* パッケージ (site-packages/nvidia/nccl/lib)
  2. ディストリの system ライブラリ
  3. soname のまま (ld のサーチパス)

1 を先に見るのは、pip 版が site-packages に置かれるだけで ld キャッシュには
載らず、`ctypes.CDLL("libnccl.so.2")` では開けないため。torch は自前で
このパスを解決して読むので、soname が引けない = NCCL が無い、ではない。

なお nvidia.nccl は namespace package なので __file__ が None になる。
パスは __path__ から取ること。

見つからなければ終了コード 1。
"""

import ctypes
import os
import sys


def candidates(argv: list[str]) -> list[str]:
    if len(argv) > 1:
        return [argv[1]]

    found: list[str] = []
    try:
        import nvidia.nccl  # noqa: PLC0415

        found += [os.path.join(p, "lib", "libnccl.so.2") for p in nvidia.nccl.__path__]
    except Exception:
        pass

    found += [
        "/usr/lib/aarch64-linux-gnu/libnccl.so.2",
        "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        "libnccl.so.2",
    ]
    return found


def main() -> int:
    tried = []
    for path in candidates(sys.argv):
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            tried.append(f"{path}: {exc}")
            continue
        version = ctypes.c_int()
        lib.ncclGetVersion(ctypes.byref(version))
        print(version.value)
        return 0

    print("libnccl.so.2 が見つかりません:", file=sys.stderr)
    for line in tried:
        print(f"  {line}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
