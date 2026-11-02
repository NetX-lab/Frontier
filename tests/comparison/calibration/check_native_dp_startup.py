"""Exercise native DP startup methods on CPU without importing the GPU stack.

The selected source functions execute unchanged. Configuration, EngineCore's
model body, NUMA setup and the post-init world wrapper are CPU test fixtures.
TCPStore, sockets, Gloo, process spawning and all-reduce are real.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import ipaddress
import json
import logging
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import textwrap
import threading
import time
import types
import weakref
from datetime import timedelta


def extract(root, relative, name, namespace, *, cls=None):
    path = root / relative
    text = path.read_text()
    tree = ast.parse(text)
    nodes = tree.body if cls is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name and not any(isinstance(d, ast.Name) and d.id == 'overload' for d in n.decorator_list))
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    source = textwrap.dedent('\n'.join(text.splitlines()[start - 1:node.end_lineno]))
    exec(compile('from __future__ import annotations\n' + '\n' * (start - 2) + source, str(path), 'exec'), namespace)
    return namespace[name]


def source_namespace(root, cfg=None):
    import functools
    import torch
    from torch.distributed import ProcessGroup, Store, TCPStore
    from torch.distributed.distributed_c10d import Backend, PrefixStore, _get_default_timeout, _unregister_process_group
    from torch.distributed.rendezvous import rendezvous
    ns = dict(globals(), functools=functools, torch=torch, ProcessGroup=ProcessGroup, Store=Store, TCPStore=TCPStore,
              Backend=Backend, PrefixStore=PrefixStore, _get_default_timeout=_get_default_timeout,
              _unregister_process_group=_unregister_process_group, rendezvous=rendezvous,
              logger=logging.getLogger('native_startup'), envs=types.SimpleNamespace(VLLM_LOGGING_LEVEL='INFO', VLLM_DISTRIBUTED_USE_SPLIT_GROUP=False),
              _WORLD=None, _NODE_COUNT=None, _INNER_DP_WORLD=None)
    sources = {
        'vllm/utils/network_utils.py': ['is_valid_ipv6_address', 'get_tcp_uri'],
        'vllm/utils/system_utils.py': ['suppress_stdout'],
        'vllm/distributed/utils.py': ['create_tcp_store', 'get_cached_tcp_store_client', 'get_cpu_distributed_timeout_or_none',
                                   'get_distributed_timeout_or_none', 'init_gloo_process_group', 'stateless_init_torch_distributed_process_group'],
        'vllm/distributed/parallel_state.py': ['init_distributed_environment'],
    }
    for relative, names in sources.items():
        for name in names:
            extract(root, relative, name, ns)
    for name in ['get_next_dp_init_port', '_pick_stateless_dp_port', 'stateless_init_dp_group']:
        extract(root, 'vllm/config/parallel.py', name, ns, cls='ParallelConfig')
        if cfg is not None:
            setattr(cfg, name, types.MethodType(ns[name], cfg))
    ns['get_distributed_init_method'] = ns['get_tcp_uri']
    ns['init_world_group'] = lambda ranks, local_rank, backend: types.SimpleNamespace(cpu_group=torch.distributed.group.WORLD)
    ns['_node_count'] = lambda group: 1
    for name in ['vllm', 'vllm.distributed', 'vllm.distributed.utils', 'vllm.config', 'vllm.v1', 'vllm.v1.engine', 'vllm.v1.engine.core']:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules['vllm.distributed.utils'].__dict__.update({k: ns[k] for k in sources['vllm/distributed/utils.py']})
    sys.modules['vllm.config'].get_current_vllm_config_or_none = lambda: types.SimpleNamespace(parallel_config=cfg) if cfg else None
    sys.modules['vllm.v1.engine.core'].EngineCoreProc = types.SimpleNamespace(run_engine_core=engine_rank)
    return ns


def worker_rank(root, attrs, local_rank, output):
    import torch
    cfg = types.SimpleNamespace(**attrs)
    ns = source_namespace(Path(root), cfg)
    ns['init_distributed_environment'](world_size=2, rank=local_rank, local_rank=local_rank, backend='gloo', timeout=timedelta(seconds=20))
    rank = cfg.data_parallel_rank * 2 + local_rank
    value = torch.tensor([float(rank + 1)])
    torch.distributed.all_reduce(value)
    expected = cfg.world_size_across_dp * (cfg.world_size_across_dp + 1) / 2
    assert value.item() == expected, (value, expected)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    Path(output, f'worker-{rank}.json').write_text(json.dumps(dict(rank=rank, sum=value.item(), expected=expected)))


def engine_rank(vllm_config, dp_rank, **kwargs):
    import torch
    cfg = vllm_config.parallel_config
    cfg.data_parallel_rank = dp_rank
    ns = source_namespace(Path(vllm_config.root), cfg)
    original_ports = list(cfg._data_parallel_master_port_list)
    groups = []
    observed = []
    for sequence in range(2):
        group, store = cfg.stateless_init_dp_group(return_store=True)
        value = torch.tensor([float(dp_rank + 1)])
        group.allreduce([value]).wait()
        expected = cfg.data_parallel_size * (cfg.data_parallel_size + 1) / 2
        assert value.item() == expected
        observed.append(dict(sequence=sequence, port=store.port, sum=value.item(), expected=expected))
        groups.append((group, store))
    assert observed[0]['port'] != observed[1]['port']
    assert cfg._data_parallel_master_port_list == original_ports
    # A worker receives the post-engine config through the real spawn boundary.
    attrs = {k: v for k, v in vars(cfg).items() if not isinstance(v, types.MethodType)}
    ctx = multiprocessing.get_context('spawn')
    workers = [ctx.Process(target=worker_rank, args=(vllm_config.root, attrs, rank, vllm_config.output)) for rank in range(2)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(40)
        assert process.exitcode == 0, ('worker failed', process.pid, process.exitcode)
    for group, store in groups:
        group.barrier().wait()
    del group, store, groups
    gc.collect()
    Path(vllm_config.output, f'engine-{dp_rank}.json').write_text(json.dumps(dict(rank=dp_rank, groups=observed, coord_port=cfg._coord_store_port)))


def stop_processes(processes, timeout=None):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        if process.pid is not None:
            process.join(timeout or 10)


def manager_type(root):
    ns = source_namespace(root)
    ns.update(get_mp_context=lambda: multiprocessing.get_context('spawn'),
              current_platform=types.SimpleNamespace(is_cuda_alike=lambda: True, is_xpu=lambda: False),
              numa_utils=types.SimpleNamespace(configure_subprocess=lambda *args, **kwargs: contextlib.nullcontext()),
              shutdown=stop_processes)
    return extract(root, 'vllm/v1/engine/utils.py', 'CoreEngineProcManager', ns)


def start_role(root, output, ranks, occupied):
    output.mkdir(parents=True, exist_ok=False)
    squatter = socket.socket()
    squatter.bind(('127.0.0.1', 0))
    port = squatter.getsockname()[1]
    if occupied:
        squatter.listen()
    else:
        squatter.close()
    cfg = types.SimpleNamespace(data_parallel_size=ranks, data_parallel_size_local=ranks,
        data_parallel_rank=0, data_parallel_rank_local=None, data_parallel_master_ip='127.0.0.1',
        data_parallel_master_port=port, _data_parallel_master_port_list=[port], _coord_store_port=0,
        _stateless_dp_group_sequence=0, enable_elastic_ep=False, assigned_physical_gpu_ids=None,
        use_ray=False, cpu_distributed_timeout_seconds=20, distributed_timeout_seconds=20,
        distributed_executor_backend='mp', nnodes=1, nnodes_within_dp=1, world_size_across_dp=ranks * 2)
    config = types.SimpleNamespace(parallel_config=cfg, model_config=types.SimpleNamespace(is_moe=True),
                                   root=str(root), output=str(output))
    manager = manager_type(root)(local_engine_count=ranks, start_index=0, local_start_index=0,
        vllm_config=config, local_client=True, handshake_address='cpu-fixture', executor_class=None, log_stats=False)
    assert manager._coord_store is not None and cfg._coord_store_port == manager._coord_store.port
    return manager, squatter, cfg._coord_store_port


def collect_role(output, role, ranks):
    manager, squatter, coord_port = role
    try:
        for process in manager.processes:
            process.join(55)
            assert process.exitcode == 0, (process.name, process.exitcode)
        engines = [json.loads((output / f'engine-{rank}.json').read_text()) for rank in range(ranks)]
        workers = [json.loads((output / f'worker-{rank}.json').read_text()) for rank in range(ranks * 2)]
        ports = [{record['groups'][sequence]['port'] for record in engines} for sequence in range(2)]
        assert all(len(values) == 1 for values in ports)
        assert ports[0] != ports[1]
    finally:
        manager.shutdown()
        squatter.close()
    assert manager._coord_store is None
    # Once the manager and children stop, no listener may survive on the store port.
    probe = socket.socket()
    assert probe.connect_ex(('127.0.0.1', coord_port)) != 0
    probe.close()
    return dict(engines=engines, workers=workers, coord_port=coord_port, listener_released=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    os.environ['OMP_NUM_THREADS'] = '1'
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for ranks in (2, 4):
        for occupied in (False, True):
            name = f'dp{ranks}-occupied{int(occupied)}'
            output = args.output / name
            role = start_role(args.native_root, output, ranks, occupied)
            results[name] = collect_role(output, role, ranks)
            print(name, 'PASS', flush=True)
    prefill = start_role(args.native_root, args.output / 'prefill', 2, True)
    decode = start_role(args.native_root, args.output / 'decode', 2, True)
    results['prefill'] = collect_role(args.output / 'prefill', prefill, 2)
    results['decode'] = collect_role(args.output / 'decode', decode, 2)
    assert results['prefill']['coord_port'] != results['decode']['coord_port']
    import torch
    result = dict(status='PASS', python=sys.version, torch=torch.__version__, cases=results,
                  source_hashes={p: hashlib.sha256((args.native_root / p).read_bytes()).hexdigest() for p in
                    ['vllm/config/parallel.py', 'vllm/distributed/utils.py', 'vllm/distributed/parallel_state.py', 'vllm/v1/engine/utils.py']})
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ALL PASS', flush=True)


if __name__ == '__main__':
    main()
