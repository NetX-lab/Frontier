"""Coordinate an original-model native P/D run using the existing replay harness.

Optional config keys: `mode` (default `light`) selects the serving mode, `serve_observers`
(absent for clean native runs) holds the per-role recorder inputs, `replay_script` (default
`pd_replay.py` in `harness_dir`) is the client driver started on the prefill head with the
pd_replay arguments, and `evidence_dir` names a task-local directory that receives a copy of
each node's small records and the replay outputs after the run.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback


def node_plan(config, peers, host, devices):
    """Map observed hosts to internal-DP heads and headless peers in each role."""
    peers = sorted(peers, key=lambda peer: peer['host'])
    engine = dict(config['engine'])
    local = config['data_parallel_size_local']
    dp = engine['data_parallel_size']
    if dp % local or len(peers) != 2 * (dp // local):
        raise ValueError('P/D node count must match the declared local DP partition')
    if (len({peer['host'] for peer in peers}) != len(peers)
            or len({peer['ip'] for peer in peers}) != len(peers)):
        raise ValueError('P/D requires distinct observed hosts and node addresses')
    count = engine['tensor_parallel_size'] * local
    if len(devices) != count or len(set(devices)) != count:
        raise ValueError('Visible device count must match TP times local DP size')
    position = [peer['host'] for peer in peers].index(host)
    nodes_per_role = dp // local
    role = ('prefill', 'decode')[position // nodes_per_role]
    block_counts = config.get('role_num_gpu_blocks_override', {})
    if (type(block_counts) is not dict or set(block_counts) - {'prefill', 'decode'}
            or any(type(value) is not int or value <= 0 for value in block_counts.values())):
        raise ValueError('Role GPU block overrides require positive integer counts')
    if role in block_counts:
        engine['num_gpu_blocks_override'] = block_counts[role]
    lane = position % nodes_per_role * local
    heads = dict(zip(('prefill', 'decode'), (peers[0], peers[nodes_per_role])))
    engine.update(host=peers[position]['ip'], data_parallel_size_local=local,
                  data_parallel_address=heads[role]['ip'],
                  data_parallel_rpc_port=config['dp_rpc_ports'][role],
                  role_devices={role: devices}, headless=lane != 0)
    if lane:
        engine['data_parallel_start_rank'] = lane
    elif 'data_parallel_start_rank' in engine:
        raise ValueError('An internal-DP API head cannot specify a start rank')
    engine['kv_transfer'] = engine['kv_transfer'] | {'side_channel_host': engine['host']}
    return dict(position=position, role=role, lane=lane, engine=engine,
                heads=heads, peers=peers,
                urls={name: f"http://{peer['ip']}:{config['http_ports'][name]}"
                      for name, peer in heads.items()})


def run(config):
    stage = Path(config['script']).parent
    sys.path[:0] = [str(stage), str(Path(config['harness_dir']))]
    from probe_nixl_read import SYSFS_INFINIBAND, gid_table, roce_v2_ipv4_ports
    from probe_runtime_inventory import primary_ipv4, read_text, sha256, weight_candidates, write_json
    from pd_replay import instance_devices, instance_env, kv_transfer_config
    from vllm_replay import server_command, stop_server

    root = Path(config['output_dir'])
    if not root.resolve().is_relative_to('/mnt/codesign-exp/ycfeng'):
        raise ValueError('P/D output must remain inside the personal cloud root')
    mode = config.get('mode', 'light')
    observers = config.get('serve_observers')
    host = socket.gethostname()
    node = root / 'nodes' / host
    node.mkdir(parents=True, exist_ok=False)
    with (node / 'worker.log').open('w', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    result = dict(host=host, qualification_verified=False, observation_errors=[])
    server = replay = None
    plan = None
    exit_code = 1

    def record(name):
        result['stage'] = name
        try:
            write_json(node / 'result.json', result)
        except Exception:
            error = traceback.format_exc()
            result['observation_errors'].append(error)
            print('NATIVE_PD_OBSERVATION_ERROR', error, flush=True)
        print('NATIVE_PD_STAGE', name, flush=True)

    def await_condition(condition, timeout, *, native_required=False):
        deadline = time.monotonic() + timeout
        while True:
            failed = list(root.glob('nodes/*/failed.json'))
            if failed:
                raise RuntimeError(f'Native P/D peer failed: {failed}')
            if native_required and server.poll() is not None:
                raise RuntimeError(f'Native server exited early: {server.returncode}')
            if condition():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('Native P/D rendezvous or workload deadline exceeded')
            time.sleep(1)

    try:
        scratch = Path(tempfile.mkdtemp(prefix='native_pd_'))
        for name, child in config['cache_environment'].items():
            os.environ[name] = str(scratch / child)
        model = Path(config['model_dir'])
        if {path.name: sha256(path) for path in model.iterdir()} != config['model_hashes']:
            raise ValueError('Original metadata allowlist or hash mismatch')
        package = Path(config['installed_package'])
        for name, expected in config['expected_source_hashes'].items():
            if sha256(package.parent / name) != expected:
                raise ValueError(f'Locked native source mismatch: {name}')
        record('rendezvous')
        write_json(node / 'ready.json', dict(host=host, ip=primary_ipv4()))
        await_condition(lambda: len(list(root.glob('nodes/*/ready.json'))) == config['replica'],
                        config['peer_wait_seconds'])
        peers = [json.loads(path.read_text()) for path in root.glob('nodes/*/ready.json')]
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = visible.split(',') if visible else [str(index) for index in range(config['gpu'])]
        plan = node_plan(config, peers, host, devices)
        result['plan'] = plan
        role = plan['role']
        rdma = {device.name: {port.name: dict(state=read_text(port / 'state'),
                 link_layer=read_text(port / 'link_layer'), gids=gid_table(port))
                 for port in (device / 'ports').glob('*')}
                 for device in SYSFS_INFINIBAND.glob('*')}
        result['ucx_devices'] = roce_v2_ipv4_ports(rdma)
        if not result['ucx_devices']:
            raise RuntimeError('No ACTIVE RoCE v2 IPv4 port')
        os.environ['UCX_NET_DEVICES'] = ','.join(result['ucx_devices'])
        try:
            result['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '-L'], text=True)
            result['versions'] = {name: importlib.metadata.version(name)
                                  for name in ('torch', 'vllm', 'aiohttp')}
        except Exception:
            result['observation_errors'].append(traceback.format_exc())
        selected = instance_devices(dict(os.environ), engine=plan['engine'], role=role)
        observer = None if observers is None else observers[role]
        env, changed = instance_env(mode, plan['engine'], node, dict(os.environ),
                                    selected[role], observer=observer, role=role)
        port = config['http_ports'][role]
        command = server_command(plan['engine'], model, port) + [
            '--kv-transfer-config', json.dumps(kv_transfer_config(plan['engine'], role, port))]
        result.update(server_command=command, server_env_set=changed)
        record('native_start')
        with (node / 'server.log').open('w') as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if plan['lane'] == 0:
            import urllib.request

            def healthy():
                try:
                    with urllib.request.urlopen(plan['urls'][role] + '/health', timeout=2) as response:
                        return response.status == 200
                except OSError:
                    return False

            await_condition(healthy, config['startup_timeout_s'], native_required=True)
            write_json(node / 'server_ready.json', dict(role=role, url=plan['urls'][role]))
        record('await_replay')
        if plan['position'] == 0:
            await_condition(lambda: len(list(root.glob('nodes/*/server_ready.json'))) == 2,
                            config['startup_timeout_s'], native_required=True)
            replay_script = config.get('replay_script', str(Path(config['harness_dir']) / 'pd_replay.py'))
            replay_command = [sys.executable, replay_script,
                '--engine-config', str(stage / 'engine.json'), '--trace-dir', str(stage / 'trace'),
                '--output-dir', str(root / 'replay'), '--mode', mode, '--model-dir', str(model),
                '--prefill-url', plan['urls']['prefill'], '--decode-url', plan['urls']['decode'],
                '--proxy-port', str(config['proxy_port']),
                '--startup-timeout-s', str(config['startup_timeout_s']),
                '--drain-s', str(config['drain_seconds'])]
            if observers is not None:
                replay_command += ['--observer-config', str(stage / 'serve_observers.json')]
            result['replay_command'] = replay_command
            record('replay')
            with (node / 'replay.log').open('w') as log:
                replay = subprocess.Popen(replay_command, stdout=log, stderr=subprocess.STDOUT)
            await_condition(lambda: replay.poll() is not None, config['replay_timeout_s'],
                            native_required=True)
            write_json(root / 'replay_finished.json', dict(returncode=replay.returncode))
        else:
            await_condition(lambda: (root / 'replay_finished.json').is_file(),
                            config['startup_timeout_s'] + config['replay_timeout_s'],
                            native_required=True)
        exit_code = json.loads((root / 'replay_finished.json').read_text())['returncode']
        record('drain_returned')
        write_json(node / 'drained.json', dict(role=role, lane=plan['lane']))
        if plan['lane'] == 0:
            await_condition(lambda: len(list(root.glob('nodes/*/drained.json'))) == config['replica'],
                            config['peer_wait_seconds'], native_required=True)
    except BaseException:
        exit_code = 1
        result['exception'] = traceback.format_exc()
        write_json(node / 'failed.json', dict(host=host, error=result['exception']))
        print(result['exception'], flush=True)
    finally:
        if replay is not None and replay.poll() is None:
            result['replay_stop'] = stop_server(replay, config['stop_timeout_s'])
        if server is not None:
            if plan['lane'] and not (node / 'failed.json').exists():
                head_stopped = root / 'nodes' / plan['heads'][plan['role']]['host'] / 'native_stopped.json'
                deadline = time.monotonic() + config['stop_timeout_s']
                while server.poll() is None and not head_stopped.exists() and time.monotonic() < deadline:
                    time.sleep(1)
            result['server_stop'] = stop_server(server, config['stop_timeout_s'])
            if result['server_stop']['returncode'] != 0:
                exit_code = 1
            write_json(node / 'native_stopped.json', result['server_stop'])
        try:
            result['weight_audit'] = weight_candidates([scratch / 'hf', model],
                                                       tuple(config['weight_suffixes']))
        except Exception:
            result['observation_errors'].append(traceback.format_exc())
        result['workload_returncode'] = exit_code
        record('finished')
        if 'evidence_dir' in config:
            try:
                evidence = Path(config['evidence_dir'])
                shutil.copytree(node, evidence / 'nodes' / host, dirs_exist_ok=True)
                if plan is not None and plan['position'] == 0 and (root / 'replay').is_dir():
                    shutil.copytree(root / 'replay', evidence / 'replay', dirs_exist_ok=True)
            except Exception:
                error = traceback.format_exc()
                result['observation_errors'].append(error)
                print('NATIVE_PD_OBSERVATION_ERROR', error, flush=True)
                record('finished')
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    raise SystemExit(run(json.loads(parser.parse_args().config.read_text())))
