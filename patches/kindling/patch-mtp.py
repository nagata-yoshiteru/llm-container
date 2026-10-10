#!/usr/bin/env python3
"""Apply after kindling's MTP BF16 fix; reject any other source version."""
import hashlib
from pathlib import Path
import sys

EXPECTED = "7af065bf3d7a6a70ef0ba702ae43767997344136f1fa1a713f8bbcf24a58afde"
ANCHOR = "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
INSERT = '''        # UNCENSORED-TP3-MTP: pad BEFORE TP slicing and name rewriting.
        if getattr(self.config, "tp_pad_orig", None):
            from .tp3pad import pad_weights
            weights = pad_weights(weights, self.config)
'''


def patch(path):
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED:
        raise SystemExit("Unexpected kindling MTP source; refusing patch")
    if source.count(ANCHOR) != 1:
        raise SystemExit("MTP load_weights anchor is not unique")
    updated = source.replace(ANCHOR, ANCHOR + INSERT)
    compile(updated, str(path), "exec")
    path.write_text(updated)


if __name__ == "__main__":
    patch(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/mtp.py"))
