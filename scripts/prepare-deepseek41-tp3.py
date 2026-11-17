#!/usr/bin/env python3
"""Prepare pinned source or local Engram packs. Never invokes Docker/sudo/SSH."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen

from deepseek41_config import ROOT, LOCK, ROLES, read_env, check_model, check_packed

CACHE = ROOT / '.cache/deepseek41-tp3'


def tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sources():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, key in [('upstream', 'recipe'), ('nccl', 'nccl')]:
        pin = LOCK[key]
        destination, receipt = CACHE / name, CACHE / f'{name}.json'
        if destination.exists():
            if not receipt.exists() or json.loads(receipt.read_text()) != dict(pin, tree_sha256=tree_hash(destination)):
                raise ValueError(f'{destination}: missing/different receipt or modified source; move it aside and rerun')
            print(f'Verified {name}: {pin["revision"]}')
            continue
        url = f'https://codeload.github.com/{pin["repository"]}/tar.gz/{pin["revision"]}'
        data = urlopen(url, timeout=120).read()
        if hashlib.sha256(data).hexdigest() != pin['sha256']:
            raise ValueError(f'{name}: source archive checksum mismatch')
        temp = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=CACHE))
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
                for member in archive.getmembers():
                    _, _, relative = member.name.partition('/')
                    if not relative:
                        continue
                    member.name = relative
                    archive.extract(member, temp, filter='data')
            checksum = tree_hash(temp)
            temp.rename(destination)
            receipt.write_text(json.dumps(dict(pin, tree_sha256=checksum), indent=2) + '\n')
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    patch = ROOT / 'patches/deepseek41/nccl-dual-pci-domain.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != LOCK['nccl_patch']['sha256']:
        raise ValueError('NCCL patch checksum mismatch')
    subprocess.run(['patch', '--dry-run', '-d', str(CACHE / 'nccl'), '-p1', '-i', str(patch)], check=True)
    print('Pinned sources ready; no weights downloaded and no containers started.')


def pack(role):
    env = read_env()
    model, destination = ROOT / env['MODEL_PATH'], ROOT / env['ENGRAM_PATH']
    config = check_model(model)
    rank = ROLES.index(role)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / f'provenance-r{rank}.json'
    targets = [destination / f'engram-l{layer}-r{rank}of3.bin' for layer in (1, 14)]
    if any(path.exists() for path in targets):
        if not receipt.exists() or json.loads(receipt.read_text()) != LOCK['model']:
            raise ValueError('Existing Engram pack has unknown provenance; use a fresh ENGRAM_PATH')
    else:
        receipt.write_text(json.dumps(LOCK['model'], indent=2) + '\n')
    if all(path.exists() for path in targets):
        check_packed(destination, config, rank)
        print(f'{role}: Engram pack already complete')
        return
    sources()
    subprocess.run([sys.executable, str(CACHE / 'upstream/scripts/pack_engram.py'),
                    '--model', str(model), '--out', str(destination), '--rank', str(rank), '--tp', '3'], check=True)
    check_packed(destination, config, rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', choices=ROLES, help='pack only this local node (requires numpy and the downloaded model)')
    args = parser.parse_args()
    if args.pack:
        pack(args.pack)
    else:
        sources()


if __name__ == '__main__':
    main()
