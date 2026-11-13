#!/usr/bin/env python3
"""Read installed owners needed to qualify Step5 runtime and precision bindings."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import time


SOURCES = {
    'vllm/config/__init__.py': 'Native configuration exports and current-config context.',
    'vllm/config/vllm.py': 'VllmConfig resolution and validation without model allocation.',
    'vllm/config/cache.py': 'Runtime page size and KV dtype resolution.',
    'vllm/config/model.py': 'Complete pinned HF metadata and quantization resolution.',
    'vllm/engine/arg_utils.py': 'Native EngineArgs configuration construction.',
    'vllm/model_executor/layers/attention/attention.py': 'Actual attention backend/cache/spec binding.',
    'vllm/v1/attention/backends/flash_attn.py': 'H800 ordinary SWA tuple and precision dispatch.',
    'vllm/model_executor/layers/quantization/fp8.py': 'Actual excluded/FP8 linear and MoE methods/scales.',
    'vllm/model_executor/layers/quantization/utils/quant_utils.py': 'Packed-prefix exclusion matching.',
    'vllm/model_executor/layers/linear.py': 'Native rank-local linear construction and processing.',
    'vllm/model_executor/layers/fused_moe/layer.py': 'Native full expert module construction and execution.',
    'vllm/model_executor/layers/fused_moe/router/gate_linear.py': 'Native router dtype and dispatch.',
    'vllm/model_executor/model_loader/dummy_loader.py': 'Unchanged native dummy initialization lifecycle.',
    'vllm/model_executor/model_loader/utils.py': 'Native quantization and model initialization owners.',
    'vllm/model_executor/model_loader/weight_utils.py': 'Native dummy weight/buffer initialization.',
    'vllm/model_executor/parameter.py': 'Native parameter and scale identities.',
    'vllm/v1/kv_cache_interface.py': 'Physical cache spec, dtype and page allocation contract.',
    'vllm/platforms/cuda.py': 'H800 platform configuration and supported backend resolution.',
}


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'run'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    launcher = load(args.task_root/'native_source_export/submit_export.py')
    launcher.inspection.locked_inputs()
    assert args.task_root.is_absolute() and args.output_dir.is_absolute()
    assert not args.output_dir.exists()
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND'] == 'rjob'
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(launcher.inspection.WORKTREE)
        os.environ['EXP_ID'] = 'step5-g1-runtime-source'
        command = 'export VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; exec python3 -B '+shlex.join([str(Path(__file__).resolve()), 'run', '--task-root', str(args.task_root), '--output-dir', str(args.output_dir)])
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()),
                              **launcher.spawn_arguments('timeout 600 bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    started = time.perf_counter()
    package = importlib.metadata.distribution('vllm')
    assert package.version == launcher.EXPECTED_VLLM_VERSION
    root = Path(package.locate_file('vllm')).resolve()
    prior = json.loads(launcher.LEDGER.read_text())['process_and_package_observations']['vllm_module_hashes']['files']
    ledger = {row['relative_path']: row for row in prior}
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for relative, question in SOURCES.items():
        path = root.joinpath(*Path(relative).parts[1:]).resolve()
        assert path.is_relative_to(root)
        row = dict(relative_path=relative, question=question)
        if not path.is_file():
            row['status'] = 'missing_requested_source'
        else:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            previous = ledger.get(relative)
            if previous is not None:
                assert previous['status'] == 'hashed' and previous['sha256'] == digest
                assert previous['size_bytes'] == len(payload)
            destination = args.output_dir/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(payload)
            row.update(status='copied', sha256=digest, size_bytes=len(payload),
                       prior_ledger_match=previous is not None)
        rows.append(row)
    collector = load(args.task_root/'native_g1/run_g1_cases.py')
    environment = collector.record_environment(args.output_dir)
    assert environment['device_name'] == 'NVIDIA H800'
    collector.dump_json(args.output_dir/'source_manifest.json', dict(
        scope='read_only_source_collection_not_runtime_binding', files=rows,
        environment=environment, worker_sha256=collector.sha256_file(Path(__file__)),
        collection_wall_seconds=time.perf_counter()-started))
    print('RUNTIME_SOURCE_COLLECTION', sum(row['status'] == 'copied' for row in rows), '/', len(rows), flush=True)
    raise SystemExit(any(row['status'] != 'copied' for row in rows))


if __name__ == '__main__':
    main()
