#!/usr/bin/env python3
"""Read-only preflight. No Docker, sudo, SSH, GPU allocation or network changes."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

from deepseek41_config import ROOT, ROLES, read_env, check_settings, check_model, check_packed, gid_index


def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=ROLES)
    parser.add_argument('--network-only', action='store_true', help='check only settings and links; not startup readiness')
    args = parser.parse_args()
    env = read_env()
    errors = []

    def check(label, action):
        try:
            result = action()
            print(f'OK: {label}' + (f': {result}' if result is not None else ''))
            return result
        except Exception as exc:
            errors.append(label)
            print(f'FAIL: {label}: {exc}')
            return None

    check('settings', lambda: check_settings(env))
    print(f'host: {socket.gethostname()}')
    check('GPU', lambda: run('nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader').strip())
    addresses = json.loads(run('ip', '-j', '-4', 'addr', 'show'))
    local = {item['local'] for interface in addresses for item in interface.get('addr_info', [])}
    roles = [role for role in ROLES if env[f'{role.upper()}_NODE_IP'] in local]
    if len(roles) != 1 or (args.role and roles != [args.role]):
        print('FAIL: local management IP does not match the chosen role; check DHCP and .env')
        return 1
    role, rank = roles[0], ROLES.index(roles[0])
    print(f'role: {role}, rank: {rank}')
    for peer in ROLES:
        if peer != role:
            check(f'management {peer}', lambda p=peer: run('ping', '-c', '1', '-W', '2', env[f'{p.upper()}_NODE_IP']).splitlines()[-1])
    interfaces = env['ROCE_IF_NAME'].split(',')
    # Ordered f0 root0/root1, f1 root0/root1; verify both subnet and peer identity.
    expected = ((4, 5, 0, 1), (0, 1, 2, 3), (2, 3, 4, 5))[rank]
    peers = ((3, 3, 2, 2), (1, 1, 3, 3), (2, 2, 1, 1))[rank]
    if len(interfaces) != 4:
        print('FAIL: expected four RoCE interfaces')
        return 1
    for interface, subnet, peer in zip(interfaces, expected, peers):
        def link(interface=interface, subnet=subnet, peer=peer):
            info = next(x for x in addresses if x['ifname'] == interface)
            ipv4 = [(x['local'], x['prefixlen']) for x in info['addr_info']]
            if ipv4 != [(f'192.168.{subnet}.{rank+1}', 24)] or info['mtu'] != 9000:
                raise ValueError(f'Expected one /24 IP and MTU 9000; got {ipv4}, MTU={info["mtu"]}')
            run('ping', '-M', 'do', '-c', '1', '-W', '2', '-s', '8972', '-I', interface, f'192.168.{subnet}.{peer}')
            return f'192.168.{subnet}.{peer}, 9000-byte DF ping'
        check(interface, link)
    check('RoCEv2 GID', lambda: gid_index(env['IB_HCA_NAME'], env.get('NCCL_IB_GID_INDEX', '')))
    if not args.network_only:
        config = check('checkpoint (48 shards)', lambda: check_model(ROOT / env['MODEL_PATH']) and 'complete')
        if config:
            cfg = json.loads((ROOT / env['MODEL_PATH'] / 'config.json').read_text())
            check('local Engram pack', lambda: check_packed(ROOT / env['ENGRAM_PATH'], cfg, rank))
        def memory():
            mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            available = int(mem['MemAvailable'].split()[0]) / 1024**2
            if available < 110:
                raise ValueError(f'MemAvailable={available:.1f} GiB; stop old model workloads before startup (need ~110 GiB free)')
            return f'{available:.1f} GiB available; runtime margin still needs GPU validation'
        check('unified memory', memory)
    suffix = ' (network only; model readiness not checked)' if args.network_only else ''
    print(('FAIL' if errors else 'PASS') + suffix)
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
