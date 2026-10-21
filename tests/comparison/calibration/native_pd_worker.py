"""Coordinate an original-model native P/D run using the existing replay harness."""

import argparse
from datetime import datetime, timezone
import hashlib
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


def write_run_manifest(config, authorization):
    """Bind actual clean sources and staged inputs before the native job launches."""
    import yaml

    def git(path, *args):
        return subprocess.run(['git', '--no-optional-locks', '-C', str(path), *args],
                              check=True, capture_output=True, text=True).stdout.strip()

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    if not authorization.strip():
        raise ValueError('The native run requires the actual root/user authorization')
    case_dir = Path(config['case_dir']).resolve()
    case = yaml.safe_load((case_dir / 'manifest.yaml').read_text())
    engine = json.loads((case_dir / 'inputs/engine.json').read_text())
    execution = json.loads((case_dir / 'inputs/native_config.json').read_text())
    if (engine != config['engine'] or execution != config
            or engine['vllm_version'] != '0.28.0' or config['mode'] != 'clean'):
        raise ValueError('Canonical native clean engine/config mismatch')
    source = Path(case['groundtruth_checkout_path']).resolve()
    source_commit = git(source, 'rev-parse', 'HEAD')
    source_dirty = git(source, 'status', '--short')
    if (source_commit != case['groundtruth_commit']
            or source_commit != engine['native_reference']['source_sha'] or source_dirty):
        raise ValueError(f'Native source must match the pinned clean checkout: {source_commit}, {source_dirty}')
    harness = Path(config['harness_worktree']).resolve()
    harness_commit = git(harness, 'rev-parse', 'HEAD')
    dirty = git(harness, 'status', '--short').splitlines()
    if dirty:
        raise ValueError(f'Commit the actual native harness before launch: {dirty}')
    stage = Path(config['script']).resolve().parent
    inputs = {'manifest.yaml': digest(case_dir / 'manifest.yaml')}
    for name in ('engine.json', 'native_config.json'):
        relative = f'inputs/{name}'
        inputs[relative] = digest(case_dir / relative)
    for name, expected in case['workload_sha256'].items():
        relative = f'inputs/trace/{name}'
        if digest(case_dir / relative) != expected or digest(stage / 'trace' / name) != expected:
            raise ValueError(f'Frozen canonical workload changed: {name}')
        inputs[relative] = expected
        inputs[str(stage / 'trace' / name)] = expected
    modules = {}
    for name, target in config['harness_files'].items():
        committed = subprocess.run(['git', '-C', str(harness), 'show', f'{harness_commit}:{name}'],
                                   check=True, capture_output=True).stdout
        expected = hashlib.sha256(committed).hexdigest()
        path = stage / target
        if digest(path) != expected:
            raise ValueError(f'Staged harness differs from its actual commit: {name}')
        modules[name] = expected
        inputs[str(path)] = expected
    for name, expected in config['model_hashes'].items():
        path = Path(config['model_dir']) / name
        if digest(path) != expected:
            raise ValueError(f'Original model metadata changed: {name}')
        inputs[str(path)] = expected
    for name, expected in config.get('retained_files', {}).items():
        if digest(stage / name) != expected:
            raise ValueError(f'Retained native launch dependency changed: {name}')
        inputs[str(stage / name)] = expected
    run_dir = case_dir / 'runs' / 'groundtruth_clean' / config['run_tag']
    manifest = dict(case_id=case['case_id'], run_tag=config['run_tag'],
        run_generation=case['run_generation'], mode='clean',
        written_before_launch_utc=datetime.now(timezone.utc).isoformat(),
        groundtruth_checkout_path=str(source), groundtruth_commit=source_commit,
        groundtruth_branch=git(source, 'branch', '--show-current'), groundtruth_tree_dirty=False,
        harness_worktree=str(harness), harness_commit=harness_commit,
        harness_branch=git(harness, 'branch', '--show-current'), harness_tree_dirty_paths=dirty,
        engine_config='inputs/engine.json', trace_dir='inputs/trace',
        execution_config='inputs/native_config.json', inputs_sha256=inputs, harness_sha256=modules,
        resources={key: config[key] for key in ('gpu_type', 'replica', 'gpu', 'charged_group',
                   'image', 'cpu', 'mem_gb', 'timeout_seconds', 'replay_timeout_s')} |
                  {'gpus': config['replica'] * config['gpu']}, authorization=authorization)
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / 'run_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


def stage_run_manifest(config, root):
    """Retain the already-written pre-launch evidence alongside native outputs."""
    case_dir = Path(config['case_dir'])
    source = case_dir / 'runs' / 'groundtruth_clean' / config['run_tag'] / 'run_manifest.json'
    manifest = json.loads(source.read_text())
    cloud_case = root.parents[2]
    for name in (n for n in manifest['inputs_sha256'] if not Path(n).is_absolute()):
        destination = cloud_case / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(case_dir / name, destination)
    shutil.copy2(source, root / 'run_manifest.json')


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
        scratch = Path(tempfile.mkdtemp(prefix='glm_full_pd_'))
        for name, child in config['cache_environment'].items():
            os.environ[name] = str(scratch / child)
        model = Path(config['model_dir'])
        result['model_hashes'] = {path.name: sha256(path) for path in model.iterdir()}
        if result['model_hashes'] != config['model_hashes']:
            raise ValueError('Original metadata allowlist or hash mismatch')
        package = Path(config['installed_package'])
        result['native_source_hashes'] = {}
        for name, expected in config['expected_source_hashes'].items():
            result['native_source_hashes'][name] = sha256(package.parent / name)
            if result['native_source_hashes'][name] != expected:
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
        if plan['position'] == 0 and 'case_dir' in config:
            try:
                stage_run_manifest(config, root)
            except Exception:
                result['observation_errors'].append(traceback.format_exc())
                print('NATIVE_PD_OBSERVATION_ERROR', result['observation_errors'][-1], flush=True)
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
        observer = config['serve_observers'][role]
        mode = config.get('mode', 'light')
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
            replay_command = [sys.executable, str(Path(config['harness_dir']) / 'pd_replay.py'),
                '--engine-config', str(stage / 'engine.json'), '--trace-dir', str(stage / 'trace'),
                '--output-dir', str(root / ('run' if 'case_dir' in config else 'replay')), '--mode', mode,
                '--observer-config', str(stage / 'serve_observers.json'), '--model-dir', str(model),
                '--prefill-url', plan['urls']['prefill'], '--decode-url', plan['urls']['decode'],
                '--proxy-port', str(config['proxy_port']),
                '--startup-timeout-s', str(config['startup_timeout_s']),
                '--drain-s', str(config['drain_seconds'])]
            for field in ('idle_timeout_s', 'idle_poll_s'):
                if field in config:
                    replay_command += ['--' + field.replace('_', '-'), str(config[field])]
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
            result['shutdown_order'] = dict(drained_before_stop=(node / 'drained.json').is_file(),
                all_nodes_drained_before_head_stop=sorted(p.parent.name for p in root.glob('nodes/*/drained.json'))
                    if plan is not None and plan['lane'] == 0 else [],
                native_returned_before_requested_stop=server.poll() is not None)
            if plan['lane'] and not (node / 'failed.json').exists():
                head_stopped = root / 'nodes' / plan['heads'][plan['role']]['host'] / 'native_stopped.json'
                deadline = time.monotonic() + config['stop_timeout_s']
                while not head_stopped.exists() and time.monotonic() < deadline:
                    if server.poll() not in (None, 0):
                        break
                    time.sleep(1)
                result['shutdown_order']['head_stopped_before_headless_stop'] = head_stopped.exists()
                if not head_stopped.exists():
                    result['observation_errors'].append('Head shutdown marker missing; native stop order unverified')
            result['shutdown_order']['native_returned_before_requested_stop'] = server.poll() is not None
            result['server_stop'] = stop_server(server, config['stop_timeout_s'])
            if result['server_stop']['returncode'] != 0:
                exit_code = 1
            write_json(node / 'native_stopped.json', result['server_stop'] | {'shutdown_order': result['shutdown_order']})
        try:
            result['weight_audit'] = weight_candidates([scratch / 'hf', model],
                                                       tuple(config['weight_suffixes']))
        except Exception:
            result['observation_errors'].append(traceback.format_exc())
        result['workload_returncode'] = exit_code
        record('finished')
    return exit_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    raise SystemExit(run(json.loads(parser.parse_args().config.read_text())))
