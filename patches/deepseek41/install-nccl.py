"""Replace the single pip NCCL library without adding a second loader path."""
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig

source = Path('/tmp/libnccl.so.2.30.7')
data = source.read_bytes()
if data[:6] != b'\x7fELF\x02\x01' or int.from_bytes(data[18:20], 'little') != 183:
    raise RuntimeError('Expected AArch64 ELF64 NCCL')
for marker in (b'IB_SUBNET_AWARE_ROUTING', b'IB_EXTENDED_IPV4_GIDS', b'IB_PRESERVE_PCI_DOMAIN'):
    if marker not in data:
        raise RuntimeError(f'Missing NCCL patch marker {marker!r}')
destination = Path(sysconfig.get_path('purelib')) / 'nvidia/nccl/lib/libnccl.so.2'
if not destination.is_file():
    raise RuntimeError(f'Pinned SGLang image has no pip NCCL at {destination}')
shutil.copyfile(source, destination.resolve())
receipt = {'path': str(destination.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
           'gpu_validated': False}
Path('/opt/llm/nccl-build.json').write_text(json.dumps(receipt, indent=2) + '\n')
source.unlink()
