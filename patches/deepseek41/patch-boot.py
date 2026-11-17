"""Bind the upstream downloader and launch receipt to this pinned checkpoint."""
import json
from pathlib import Path

lock = json.loads(Path('/opt/llm/source-lock.json').read_text())['model']
path = Path('/opt/dsv41/boot.py')
text = path.read_text()
for old, new in [
    ("REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'", f"REPO = {lock['repository']!r}"),
    ("REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'", f"REVISION = {lock['revision']!r}"),
]:
    if text.count(old) != 1:
        raise RuntimeError('Upstream boot.py changed; review before rebuilding')
    text = text.replace(old, new)
path.write_text(text)
