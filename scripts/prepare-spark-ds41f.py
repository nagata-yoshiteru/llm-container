"""Stage the pinned vLLM checkout and TP3 profiles on this Spark (no Docker)."""
import getpass
import ipaddress
import json
from pathlib import Path
import socket
import subprocess
import tempfile

from deepseek41_config import LOCK, ROOT, ROLES, check_model, gid_index, read_env


TAG = 'spark-ds41f-1120'
REVISION = 'd3d40d941f2752fea95aa770eb10d41b5ad5aa1a'
CHECKOUT = ROOT / '.cache' / TAG
HOSTS = ('spark-a814', 'spark-aa30', 'spark-d544')
# f1 connects each rank to the next rank, over both PCI domains.
PORTS = (('rocep1s0f0', 'roceP2p1s0f0'), ('rocep1s0f1', 'roceP2p1s0f1'))


def node_map(env):
    nodes = []
    for rank, (name, role) in enumerate(zip(HOSTS, ROLES)):
        previous, following = (rank - 1) % 3, (rank + 1) % 3
        subnets = {
            hca: f'192.168.{2 * edge + domain}.0/24'
            for port, edge in zip(PORTS, (previous, rank))
            for domain, hca in enumerate(port)
        }
        nodes.append({
            'name': name, 'rank': rank, 'head': rank == 0,
            'management_ip': str(ipaddress.IPv4Address(env[f'{role.upper()}_NODE_IP'])),
            'roce_peer_hcas': {str(previous): list(PORTS[0]), str(following): list(PORTS[1])},
            'roce_subnets': subnets, 'roce_gid_index': 3,
        })
    return {'schema_version': 1, 'ssh_user': getpass.getuser(), 'nodes': nodes}


def verify_node(node, interface):
    links = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr', 'show', 'dev', interface]))
    if node['management_ip'] not in [a['local'] for a in links[0]['addr_info']]:
        raise SystemExit(f"Management IP on {interface} differs from .env")
    gid_index(','.join(node['roce_subnets']), '3')
    for hca, subnet in node['roce_subnets'].items():
        gid = Path('/sys/class/infiniband') / hca / 'ports/1/gids/3'
        address = ipaddress.IPv6Address(gid.read_text().strip()).ipv4_mapped
        if address not in ipaddress.IPv4Network(subnet):
            raise SystemExit(f'{hca} has GID {address}, expected {subnet}')


def write_config(name, data):
    target = CHECKOUT / 'config' / name
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    Path(f.name).replace(target)


def main():
    env = read_env()
    nodes = node_map(env)
    node = next((n for n in nodes['nodes'] if n['name'] == socket.gethostname()), None)
    if not node:
        raise SystemExit(f'Run on one of {HOSTS}')
    if len({n['management_ip'] for n in nodes['nodes']}) != 3:
        raise SystemExit('Expected three distinct management IPs')
    verify_node(node, env['GLOO_SOCKET_IFNAME'])
    model = (ROOT / env['MODEL_PATH']).resolve(strict=True)
    if (env['MODEL_REPO'], env['MODEL_REVISION']) != (LOCK['model']['repository'], LOCK['model']['revision']):
        raise SystemExit('Expected the pinned dealignai FP8 checkpoint')
    check_model(model)
    if not CHECKOUT.exists():
        CHECKOUT.parent.mkdir(exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', TAG,
                        'https://github.com/christopherowen/spark-ds41f.git', str(CHECKOUT)], check=True)
    revision = subprocess.check_output(['git', '-C', str(CHECKOUT), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f'{CHECKOUT}: expected {REVISION}, found {revision}')
    (CHECKOUT / 'cache').mkdir(exist_ok=True)
    write_config('nodes.json', nodes)
    # Let the pinned recipe derive graph capture sizes and the prefill threshold
    # from the two-sequence limit, including the speculative decode rows.
    base = 'config/cluster.json'
    local_recipe = 'config/llm-shapes.local.json'
    recipe = json.loads((CHECKOUT / base).read_text())
    shape_1m_4k = 'tp3-triangle-4k-1m-x2'
    recipe['shapes'][shape_1m_4k] = {
        'extends': 'tp3-triangle-4k',
        'sizing': {'max_model_len': 1048576, 'max_num_seqs': 2},
    }
    write_config(Path(local_recipe).name, recipe)
    for name, source, shape in [('4k-1m-x2', local_recipe, shape_1m_4k),
                                ('1m', base, 'tp3-triangle'), ('256k', base, 'tp3-triangle-4k')]:
        profile = f'config/llm-{name}.local.json'
        data = {
            'extends': {'config': source, 'shape': shape},
            'set': {
                'deployment': {'path': str(CHECKOUT),
                               'ready_url': f"http://127.0.0.1:{env['HOST_PORT']}/v1/models"},
                'host': {'home': str(Path.home())},
                'distributed': {'master_addr': env['HEAD_NODE_IP']},
                'environment': {'NCCL_SOCKET_IFNAME': env['NCCL_SOCKET_IFNAME'],
                                'GLOO_SOCKET_IFNAME': env['GLOO_SOCKET_IFNAME'],
                                'TP_SOCKET_IFNAME': env['GLOO_SOCKET_IFNAME']},
                'model': env['MODEL_REPO'],
                'models': {env['MODEL_REPO']: {'hub': str(model), 'snapshot': '.'}},
            },
            'serve_options': {'--port': env['HOST_PORT'], '--served-model-name': env['SERVED_MODEL_NAME']},
            'unset': ['glp'],
        }
        write_config(Path(profile).name, data)
        subprocess.run([str(CHECKOUT / 'bin/spark'), '--cluster-config', profile, 'doctor'],
                       cwd=CHECKOUT, check=True)
    print(f"{node['name']}: staged 4 KiB 1M x 2, 4 KiB 256K and 64 KiB 1M profiles in {CHECKOUT / 'config'}")


if __name__ == '__main__':
    main()
