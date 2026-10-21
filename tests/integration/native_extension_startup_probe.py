"""Verify staged native extension startup in CPU children without constructing a Worker."""

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


def worker_probe(queue, expected_module, native_startup):
    """Use the native wrapper's extension-resolution path, then stop before construction."""
    try:
        if native_startup:
            import torch
            from vllm.v1.worker import worker_base
            from vllm.v1.worker.gpu_worker import Worker

            assert not torch.cuda.is_initialized()

            class ConstructorNotRun(Exception):
                pass

            config = SimpleNamespace(
                enable_trace_function_call_for_thread=lambda: None,
                parallel_config=SimpleNamespace(
                    worker_cls='vllm.v1.worker.gpu_worker.Worker',
                    worker_extension_cls='native_full_structure.FullModelInventory'),
                model_config=SimpleNamespace(multimodal_config=None))
            wrapper = worker_base.WorkerWrapperBase(rpc_rank=0)
            with patch.object(Worker, '__init__', side_effect=ConstructorNotRun), \
                    patch.object(worker_base, 'set_current_vllm_config',
                                 side_effect=lambda value: contextlib.nullcontext()):
                try:
                    wrapper.init_worker([dict(vllm_config=config)])
                except ConstructorNotRun:
                    pass
                else:
                    raise AssertionError('The real Worker constructor must not run')
            import native_full_structure as module

            observer = module._NATIVE_SERVING_INVENTORY
            assert observer is not None and observer.errors == []
            assert len(observer.hooks) == 6
            assert observer.forwards == observer.replays == 0
            assert module.FullModelInventory in Worker.__bases__
            targets = [(Worker, name) for name in
                       ('execute_model', 'shutdown', 'compile_or_warm_up_model')]
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
            targets += [(GPUModelRunner, name) for name in ('_model_forward', '_dummy_run')]
            targets += [(torch.cuda.CUDAGraph, 'replay')]
            for cls, name in targets:
                assert getattr(getattr(cls, name), '_frontier_native_serving_observer') is observer
            assert not torch.cuda.is_initialized()
            native = dict(native_wrapper_extension_injected=True,
                          actual_native_hooks_installed=sorted(observer.hooks),
                          observation_errors=observer.errors, successful_forwards=0,
                          successful_replays=0, gpu_initialized=False,
                          real_worker_constructor_called=False,
                          config_and_constructor_are_cpu_standins=True)
        else:
            import native_full_structure as module
            native = dict(native_startup_checked=False)
        assert Path(module.__file__).resolve() == expected_module.resolve()
        queue.put(dict(ok=True, module=str(module.__file__),
                       module_sha256=hashlib.sha256(expected_module.read_bytes()).hexdigest(),
                       extension=module.FullModelInventory.__name__, **native))
    except BaseException:
        import traceback
        queue.put(dict(ok=False, error=traceback.format_exc()))
        raise


def child(expected_module, native_startup):
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=worker_probe,
                              args=(queue, expected_module, native_startup))
    process.start()
    result = queue.get(timeout=120)
    process.join(timeout=30)
    assert process.exitcode == 0 and result['ok'], result
    print('STAGED_EXTENSION_PROBE ' + json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--captured-root', type=Path)
    parser.add_argument('--native-startup', action='store_true')
    parser.add_argument('--child-module', type=Path)
    args = parser.parse_args()
    if args.child_module:
        child(args.child_module, args.native_startup)
        return
    config = json.loads((args.package / 'config.json').read_text())
    stage, harness = Path(config['script']).parent, Path(config['harness_dir'])
    # Match native_pd_worker.run's parent-only imports, not a child import shortcut.
    sys.path[:0] = [str(stage), str(harness)]
    from pd_replay import instance_env, kv_transfer_config
    from vllm_replay import server_command

    inherited = dict(os.environ)
    inherited['PYTHONPATH'] = os.pathsep.join(filter(None, (
        str(stage), str(args.package), inherited.get('PYTHONPATH'))))
    inherited['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (
        config['cuda_compat_path'], inherited.get('LD_LIBRARY_PATH'))))
    results = []
    for path in sorted((args.captured_root / 'nodes').glob('*/result.json')):
        captured = json.loads(path.read_text())
        plan = captured['plan']
        role, engine = plan['role'], plan['engine']
        env, changed = instance_env(config['mode'], engine, Path('/tmp/cpu-probe'),
            inherited, '0,1,2,3,4,5,6,7', observer=None, role=role)
        if args.native_startup:
            env.update(FRONTIER_NATIVE_SERVING_EVIDENCE_DIR='/tmp/cpu-probe',
                       FRONTIER_NATIVE_RUN_GENERATION='CPU_STARTUP_ONLY_NOT_A_RUN',
                       FRONTIER_NATIVE_ROLE=role)
        command = server_command(engine, Path(config['model_dir']), config['http_ports'][role])
        command += ['--kv-transfer-config', json.dumps(kv_transfer_config(
            engine, role, config['http_ports'][role]))]
        probe = [command[0], '-P', str(Path(__file__).resolve()),
                 '--child-module', str(harness / 'native_full_structure.py')]
        if args.native_startup:
            probe.append('--native-startup')
        completed = subprocess.run(probe, env=env, cwd=stage,
            capture_output=True, text=True, timeout=180)
        print(completed.stdout, end='', flush=True)
        print(completed.stderr, end='', file=sys.stderr, flush=True)
        assert completed.returncode == 0, (path.name, role, plan['lane'], completed.returncode)
        results.append(dict(host=path.parent.name, role=role, lane=plan['lane'],
                            server_command=command, cwd=str(stage),
                            exported_pythonpath=env['PYTHONPATH'], changed=changed))
    assert len(results) == 4
    print(json.dumps(dict(verified_nodes=results, native_admission='UNVERIFIED',
                          production_gpus_used=0), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
