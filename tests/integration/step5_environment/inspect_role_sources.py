#!/usr/bin/env python3
"""Collect installed full-role owners without constructing or changing a model."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys

WORKTREE = Path(__file__).resolve().parents[3]
TASK = WORKTREE/'task_memory/task_2026-10-15_step5_preview_h800'
SOURCES = {
    'vllm/entrypoints/llm.py': 'Native LLM entry and tokenizer-free request construction.',
    'vllm/sampling_params.py': 'Native fixed output length, EOS and detokenization controls.',
    'vllm/config/parallel.py': 'Actual TP/DP/EP defaults and executor resolution.',
    'vllm/config/compilation.py': 'Actual graph capture policy and supported runtime modes.',
    'vllm/v1/engine/llm_engine.py': 'Native synchronous engine execution and completion.',
    'vllm/v1/engine/core.py': 'Native EngineCore initialization and request execution.',
    'vllm/v1/engine/core_client.py': 'Actual EngineCore process ownership and client dispatch.',
    'vllm/v1/executor/abstract.py': 'Native executor selection and worker RPC contract.',
    'vllm/v1/executor/multiproc_executor.py': 'Full TP group worker creation and shutdown.',
    'vllm/v1/worker/gpu_worker.py': 'Actual GPU worker initialization and runner choice.',
    'vllm/v1/worker/gpu_model_runner.py': 'Installed legacy runner and MRv2 selection boundary.',
    'vllm/v1/worker/utils.py': 'Actual attention groups and memory/capture utilities.',
    'vllm/v1/worker/gpu/model_states/default.py': 'Step5 inherited MRv2 metadata and cache preparation.',
    'vllm/v1/worker/gpu/model_states/interface.py': 'Model state and metadata contract.',
    'vllm/v1/worker/gpu/input_batch.py': 'Actual persistent request state and live/padded token ownership.',
    'vllm/v1/core/kv_cache_utils.py': 'Native memory budget, cache grouping and allocation.',
    'vllm/utils/mem_utils.py': 'Native memory profiling and non-KV accounting.',
    'vllm/model_executor/models/step3p7.py': 'Unchanged outer wrapper resident modules and nested text construction.',
    'vllm/model_executor/models/registry.py': 'Actual model class resolution, not guessed text unwrapping.',
    'vllm/model_executor/models/utils.py': 'Native model construction and nested config helpers.',
}
MRV2_SOURCES = {
    'vllm/v1/worker/gpu/model_runner.py': 'Actual MRv2 native model loading, capacity and execution.',
    'vllm/v1/worker/gpu/cudagraph_utils.py': 'Actual MRv2 graph capture and replay ownership.',
    'vllm/v1/worker/gpu/attn_utils.py': 'Actual metadata groups, cache binding and preparation.',
    'vllm/v1/worker/gpu/warmup.py': 'Actual native warmup and dummy-run lifecycle.',
    'vllm/v1/worker/gpu/buffer_utils.py': 'Persistent host/device buffer allocation.',
    'vllm/v1/worker/gpu/states.py': 'Persistent request state and input ownership.',
    'vllm/v1/worker/worker_base.py': 'Supported worker extension and collective RPC boundary.',
    'vllm/model_executor/models/interfaces.py': 'Native wrapper residency markers and model-state dispatch.',
    'vllm/model_executor/models/step_vl.py': 'Actual resident outer tower parameter construction only.',
    'vllm/multimodal/registry.py': 'Native text-only request versus resident wrapper policy.',
}
FRONTEND_SOURCES = {
    'vllm/renderers/registry.py': 'Actual native renderer selection before EngineCore initialization.',
    'vllm/renderers/base.py': 'Native processor construction and tokenizer requirement.',
    'vllm/renderers/hf.py': 'Native renderer initialization and original tokenizer handling.',
    'vllm/multimodal/processing/processor.py': 'Actual native rejection and processor input prerequisite.',
    'vllm/model_executor/models/step3_vl.py': 'Inherited native frontend assets required for text serving, not multimodal implementation.',
}
NATIVE_LAUNCH_SOURCES = {
    'vllm/v1/engine/utils.py': 'Native EngineCore launch, headless peer and rendezvous lifecycle before any TP16 capacity proposal.',
    'vllm/entrypoints/cli/serve.py': 'Actual native multi-node headless entrypoint and shutdown ownership.',
    'vllm/entrypoints/openai/api_server.py': 'Native frontend/engine startup configuration propagation for multi-node execution.',
}
SOURCE_SETS = {'role': SOURCES, 'mrv2': MRV2_SOURCES, 'frontend': FRONTEND_SOURCES,
               'native_launch': NATIVE_LAUNCH_SOURCES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'run'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-set', choices=SOURCE_SETS, default='role')
    args = parser.parse_args()
    task, output = args.task_root.resolve(), args.output_dir.resolve()
    assert task == TASK
    assert output.is_relative_to(TASK) and output != TASK and not output.exists()
    sys.argv = [str(Path(__file__).resolve()), args.mode, '--task-root', str(task), '--output-dir', str(output)]
    path = Path(__file__).with_name('inspect_runtime_sources.py')
    spec = importlib.util.spec_from_file_location('runtime_sources', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    if args.mode == 'run':
        # Distribution membership is availability evidence, not an import or launch trace.
        package = importlib.metadata.distribution('vllm')
        launcher = collector.load(task/'native_source_export/submit_export.py')
        launcher.inspection.locked_inputs()
        assert package.version == launcher.EXPECTED_VLLM_VERSION
        members = sorted(str(member) for member in package.files
                         if str(member).endswith('.py') and str(member).startswith((
                             'vllm/v1/worker/gpu/', 'vllm/v1/engine/', 'vllm/v1/executor/')))
        with (output.parent/(output.name+'_available_members.json')).open('x') as stream:
            json.dump(dict(scope='Installed distribution members, not loaded objects', members=members), stream, indent=2)
            stream.write('\n')
    collector.SOURCES = SOURCE_SETS[args.source_set]
    collector.__file__ = __file__
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND'] == 'rjob'
        launcher = collector.load(task/'native_source_export/submit_export.py')
        launcher.inspection.locked_inputs()
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(WORKTREE)
        os.environ['EXP_ID'] = 'step5-g1-runtime-source'
        argv = [str(Path(__file__).resolve()), 'run', '--source-set', args.source_set,
                '--task-root', str(task), '--output-dir', str(output)]
        command = ('export VLLM_ENABLE_CUDA_COMPATIBILITY=1 '
                   'LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; '
                   'exec python3 -B '+shlex.join(argv))
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()),
                              **launcher.spawn_arguments('timeout 600 bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
    else:
        collector.main()


if __name__ == '__main__':
    main()
