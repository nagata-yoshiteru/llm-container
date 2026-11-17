#!/usr/bin/env python3
"""CPU checks and daemon-free Compose expansion; no Docker build/run or sudo."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
from deepseek41_config import (ROOT, LOCK, ROLES, read_env, check_settings,
                               header, check_model, check_packed, gid_index)


class Configuration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = read_env()
        cls.composes = {}
        for role in ROLES:
            # `config` only parses local files; does not connect to a Docker daemon.
            cls.composes[role] = json.loads(subprocess.check_output(
                ['docker', 'compose', '--profile', role, 'config', '--format', 'json'], cwd=ROOT, text=True))

    def test_profiles_and_runtime_contract(self):
        for rank, role in enumerate(ROLES):
            services = self.composes[role]['services']
            self.assertEqual(set(services), {role})
            service = services[role]
            env = service['environment']
            check_settings(env)
            self.assertEqual(env['NODE_RANK'], str(rank))
            self.assertEqual(env['HOST_IP'], self.env[f'{role.upper()}_NODE_IP'])
            self.assertEqual(env['MODEL_PATH'], env['DSV41_SOURCE'])
            self.assertEqual(env['DIST_INIT_ADDR'], self.env['HEAD_NODE_IP'] + ':' + self.env['DIST_PORT'])
            for key, value in {'NCCL_NET': 'IB', 'NCCL_ALGO': 'Ring',
                               'NCCL_IB_SUBNET_AWARE_ROUTING': '1', 'NCCL_IB_EXTENDED_IPV4_GIDS': '1',
                               'NCCL_IB_PRESERVE_PCI_DOMAIN': '1', 'SKIP_PREPARE': '1',
                               'SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE': '0',
                               'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:False'}.items():
                self.assertEqual(env[key], value)
            self.assertNotIn('LD_LIBRARY_PATH', env)
            self.assertEqual(service['network_mode'], 'host')
            self.assertEqual(service['pull_policy'], 'build')
            self.assertEqual(service['entrypoint'], ['bash', '/opt/llm/entrypoint.sh'])
            mounts = {m['target']: m for m in service['volumes']}
            for target in (env['MODEL_PATH'], '/engram'):
                self.assertTrue(mounts[target]['read_only'])
                self.assertFalse(mounts[target]['bind']['create_host_path'])
            if rank:
                self.assertTrue(service['healthcheck']['disable'])
            else:
                self.assertNotIn('healthcheck', service)  # inherits image healthcheck
            build = service['build']
            self.assertEqual((Path(build['context']) / build['dockerfile']).resolve(), ROOT / 'Dockerfile.deepseek41')
            for path in build['additional_contexts'].values():
                self.assertTrue(Path(path).is_dir())

    def test_reject_unsafe_memory_and_wrong_model_settings(self):
        check_settings(self.env)
        for key, value in [('OFFLOAD_MODE', 'ram'), ('TP_SIZE', '2'), ('DSV41_CACHE_GIB', '16'),
                           ('MODEL_REVISION', 'main'), ('CONTEXT_LENGTH', '1048576'),
                           ('MAX_TOTAL_TOKENS', '1024'), ('MAX_RUNNING_REQUESTS', '8')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_settings(dict(self.env, **{key: value}))

    def test_build_pins_and_copy_inputs(self):
        root = ROOT / '.cache/deepseek41-tp3/upstream'
        dockerfile = (ROOT / 'Dockerfile.deepseek41').read_text()
        self.assertIn((root / 'Dockerfile').read_text(), dockerfile)
        self.assertEqual(dockerfile.count('sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349'), 2)
        for line in dockerfile.splitlines():
            if line.startswith('COPY ') and not line.startswith('COPY --from='):
                for item in line.split()[1:-1]:
                    self.assertTrue((root / item).exists(), item)
        p = ROOT / 'patches/deepseek41/nccl-dual-pci-domain.patch'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), LOCK['nccl_patch']['sha256'])

    def test_patched_boot_generates_three_node_sglang_argv(self):
        source = ROOT / '.cache/deepseek41-tp3/upstream/boot.py'
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            boot = temp / 'boot.py'
            boot.write_bytes(source.read_bytes())
            script = (ROOT / 'patches/deepseek41/patch-boot.py').read_text()
            script = script.replace('/opt/llm/source-lock.json', str(ROOT / 'patches/deepseek41/source-lock.json'))
            script = script.replace('/opt/dsv41/boot.py', str(boot))
            exec(compile(script, 'patch-boot.py', 'exec'), {})
            for role in ROLES:
                env = self.composes[role]['services'][role]['environment']
                with patch.dict(os.environ, env):
                    spec = importlib.util.spec_from_file_location('test_boot', boot)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertEqual(module.REPO, LOCK['model']['repository'])
                    self.assertEqual(module.REVISION, LOCK['model']['revision'])
                    module._gpu_lines = lambda: ['NVIDIA GB10']
                    module.save = lambda *args: None
                    class Captured(Exception):
                        pass
                    def capture(argv, **kwargs):
                        self.assertEqual(argv[:3], [sys.executable, '-m', 'sglang.launch_server'])
                        for flag, value in {'--tp': '3', '--ep-size': '1', '--nnodes': '3',
                                            '--node-rank': env['NODE_RANK'], '--model-path': env['MODEL_PATH'],
                                            '--context-length': '32768', '--max-total-tokens': '131072',
                                            '--speculative-algorithm': 'DSPARK', '--speculative-dspark-block-size': '3',
                                            '--tool-call-parser': 'deepseekv41', '--reasoning-parser': 'deepseek-v41'}.items():
                            self.assertEqual(argv[argv.index(flag) + 1], value)
                        self.assertEqual(kwargs['env']['DSV41_SOURCE'], env['MODEL_PATH'])
                        raise Captured()
                    with patch.object(module.subprocess, 'Popen', capture), self.assertRaises(Captured):
                        module.serve()

    def test_reject_truncated_safetensors_and_foreign_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            t = root / 'tiny.safetensors'
            meta = json.dumps({'a': {'dtype': 'U8', 'shape': [4], 'data_offsets': [0, 4]}}).encode()
            data = struct.pack('<Q', len(meta)) + meta + b'abcd'
            t.write_bytes(data)
            self.assertIn('a', header(t))
            t.write_bytes(data[:-1])
            with self.assertRaises(ValueError):
                header(t)
            (root / 'config.json').write_text('{}')
            with self.assertRaises(ValueError):
                check_model(root)

    def test_packed_rank_and_provenance(self):
        config = {'text_config': {'engram_num_embeddings': [12, 15]}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for layer, rows in zip((1, 14), (12, 15)):
                lo, hi = rows // 3, rows * 2 // 3
                data = bytearray(4096 + (hi-lo) * 264)
                struct.pack_into('<6Q', data, 0, 0x31344E4531565344, layer, lo, hi, rows, 264)
                (root / f'engram-l{layer}-r1of3.bin').write_bytes(data)
            receipt = root / 'provenance-r1.json'
            receipt.write_text(json.dumps(LOCK['model']))
            check_packed(root, config, 1)
            with self.assertRaises(FileNotFoundError):
                check_packed(root, config, 0)
            receipt.write_text('{}')
            with self.assertRaises(ValueError):
                check_packed(root, config, 1)

    def test_gid_discovery_and_bad_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for hca in ('roce0', 'roce1'):
                port = root / hca / 'ports/1'
                for sub in ('gids', 'gid_attrs/types'):
                    (port / sub).mkdir(parents=True)
                (port / 'state').write_text('4: ACTIVE\n')
                for idx, gid, kind in [('0', 'fe80::1', 'RoCE v2'),
                                        ('3', '::ffff:192.168.0.1', 'RoCE v1'),
                                        ('5', '::ffff:192.168.0.1', 'RoCE v2')]:
                    (port / 'gids' / idx).write_text(gid)
                    (port / 'gid_attrs/types' / idx).write_text(kind)
            self.assertEqual(gid_index('=roce0,roce1', sysfs=root), '5')
            with self.assertRaises(ValueError):
                gid_index('roce0,roce1', '3', sysfs=root)
            (root / 'roce1/ports/1/state').write_text('2: INIT')
            with self.assertRaises(ValueError):
                gid_index('roce0,roce1', sysfs=root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
