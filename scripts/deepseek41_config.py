"""Shared, stdlib-only checks. Safe to run with python3 -S before engine imports."""
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / 'patches/deepseek41/source-lock.json'
if not LOCK_PATH.exists():
    LOCK_PATH = Path('/opt/llm/source-lock.json')
LOCK = json.loads(LOCK_PATH.read_text())
ROLES = ('head', 'worker1', 'worker2')


def read_env(path=ROOT / '.env'):
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        if value[:1] in ('"', "'") and value[-1:] == value[:1]:
            value = value[1:-1]
        values[key] = value
    return values


def check_settings(env):
    for key, expected in {'TP_SIZE': '3', 'NNODES': '3', 'EP_SIZE': '1',
                          'OFFLOAD_MODE': 'nvme', 'DSV41_CACHE_GIB': '0',
                          'DSV41_RESIDENT_SCALES': '0', 'DSV41_TP_PAD': '1'}.items():
        if env.get(key) != expected:
            raise ValueError(f'{key} must be {expected} for this profile')
    for key in ('MODEL_REPO', 'MODEL_REVISION'):
        expected = LOCK['model']['repository' if key == 'MODEL_REPO' else 'revision']
        if env.get(key) != expected:
            raise ValueError(f'{key} does not match source-lock.json')
    addresses = [ipaddress.IPv4Address(env[f'{role.upper()}_NODE_IP']) for role in ROLES]
    if len(set(addresses)) != 3:
        raise ValueError('Three distinct management addresses are required')
    context, pool = int(env['CONTEXT_LENGTH']), int(env['MAX_TOTAL_TOKENS'])
    if not 4096 <= context <= 262144 or pool < context or pool > 750000:
        raise ValueError('Require 4096 <= context <= 262144 and context <= KV pool <= 750000')
    if not 1 <= int(env['MAX_RUNNING_REQUESTS']) <= 4:
        raise ValueError('MAX_RUNNING_REQUESTS must be 1..4')
    chunk = int(env['CHUNKED_PREFILL_SIZE'])
    if chunk not in (256, 512, 768):
        raise ValueError('CHUNKED_PREFILL_SIZE must be 256, 512 or 768')


def header(path):
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f'Truncated safetensors: {path}')
        length = struct.unpack('<Q', prefix)[0]
        if not 0 < length <= 64 * 1024 * 1024:
            raise ValueError(f'Invalid safetensors header: {path}')
        tensors = json.loads(handle.read(length))
    end = max(t['data_offsets'][1] for k, t in tensors.items() if k != '__metadata__')
    if path.stat().st_size != 8 + length + end:
        raise ValueError(f'Incomplete safetensors: {path}')
    return tensors


def check_model(root):
    root = Path(root).resolve()
    for filename, key in [('config.json', 'config_sha256'),
                          ('model.safetensors.index.json', 'index_sha256')]:
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != LOCK['model'][key]:
            raise ValueError(f'{filename}: wrong model/revision or modified config; fetch the pinned checkpoint')
    config = json.loads((root / 'config.json').read_text())
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    shards = set(index.values())
    if len(shards) != 48:
        raise ValueError('Expected 48 native FP8/FP4 shards')
    for shard in sorted(shards):
        path = (root / shard).resolve()
        if path.parent != root:
            raise ValueError('Shard outside checkpoint directory')
        tensors = header(path)
        if not {key for key, value in index.items() if value == shard} <= tensors.keys():
            raise ValueError(f'Index references missing tensors in {shard}')
    for layer, rows in zip((1, 14), config['text_config']['engram_num_embeddings']):
        prefix = f'layers.{layer}.engram.embed.'
        if index[prefix + 'weight'] != index[prefix + 'scale']:
            raise ValueError('Engram weights/scales must share a shard')
        tensors = header(root / index[prefix + 'weight'])
        for suffix, dtype, width in [('weight', 'F8_E4M3', 256), ('scale', 'F8_E8M0', 8)]:
            t = tensors[prefix + suffix]
            if t['dtype'] != dtype or t['shape'] != [rows, width]:
                raise ValueError(f'Unexpected Engram format: {prefix}{suffix}')
    return config


def check_packed(directory, config, rank):
    for layer, rows in zip((1, 14), config['text_config']['engram_num_embeddings']):
        path = Path(directory) / f'engram-l{layer}-r{rank}of3.bin'
        lo, hi = rows * rank // 3, rows * (rank + 1) // 3
        with path.open('rb') as handle:
            actual = struct.unpack('<6Q', handle.read(48))
        if actual != (0x31344E4531565344, layer, lo, hi, rows, 264):
            raise ValueError(f'Wrong Engram rank/header: {path}')
        if path.stat().st_size != 4096 + (hi - lo) * 264:
            raise ValueError(f'Incomplete Engram pack: {path}')
    receipt = json.loads((Path(directory) / f'provenance-r{rank}.json').read_text())
    if receipt != LOCK['model']:
        raise ValueError('Engram provenance differs; prepare from the selected checkpoint')


def gid_index(hcas, override='', sysfs=Path('/sys/class/infiniband')):
    """Find a common RoCEv2 IPv4 GID; never assume that index 3 is correct."""
    ports = []
    for hca in hcas.removeprefix('=').split(','):
        if not hca or '/' in hca or ':' in hca:
            raise ValueError('Use exact HCA names without port suffixes')
        port = sysfs / hca / 'ports/1'
        if (port / 'state').read_text().strip() != '4: ACTIVE':
            raise ValueError(f'{hca} is not ACTIVE')
        ports.append(port)
    candidates = [override] if override else sorted(p.name for p in (ports[0] / 'gids').iterdir())
    for idx in candidates:
        if not idx.isdigit():
            continue
        try:
            if all((port / 'gid_attrs/types' / idx).read_text().strip() == 'RoCE v2'
                   and ipaddress.IPv6Address((port / 'gids' / idx).read_text().strip()).ipv4_mapped
                   not in (None, ipaddress.IPv4Address('0.0.0.0')) for port in ports):
                return idx
        except (OSError, ValueError):
            continue
    raise ValueError('No common IPv4 RoCEv2 GID on the selected HCAs')


if __name__ == '__main__':
    check_settings(os.environ)
    config = check_model(os.environ['MODEL_PATH'])
    rank = int(os.environ['NODE_RANK'])
    if rank not in range(3):
        raise ValueError('NODE_RANK must be 0, 1 or 2')
    check_packed(os.environ['DSV41_PACKED_DIR'], config, rank)
    # stdout is consumed by the shell wrapper, so print only the GID index.
    print(gid_index(os.environ['NCCL_IB_HCA'], os.environ.get('NCCL_IB_GID_INDEX', '')))
