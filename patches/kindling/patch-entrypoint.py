#!/usr/bin/env python3
"""Keep the existing space-separated served model aliases on pinned kindling."""
import hashlib
from pathlib import Path
import sys


def patch(path):
    text = path.read_text()
    expected = "fa0d017911fdb22a7cce7ce9aba0e2109b31804b753919b0a523be8e96db7a10"
    if hashlib.sha256(text.encode()).hexdigest() != expected:
        raise SystemExit("Unexpected kindling entrypoint; refusing alias patch")
    anchor = 'exec vllm serve "$MODEL"'
    flag = '--served-model-name "$SERVED"'
    if text.count(anchor) != 1 or text.count(flag) != 1:
        raise SystemExit("Served model alias anchors are not unique")
    text = text.replace(anchor, 'read -r -a _served_names <<< "$SERVED"\n' + anchor)
    path.write_text(text.replace(flag, '--served-model-name "${_served_names[@]}"'))


if __name__ == "__main__":
    patch(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/entrypoint.sh"))
