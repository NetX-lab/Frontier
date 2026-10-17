#!/usr/bin/env python3
"""Observe the native block-FP8 quantizer's resolved mode in a new process."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import sys
import time

CLOUD = Path('/mnt/codesign-exp/ycfeng/step5-preview/quantization_binding_20261017_01')
MANIFESTS = {
    'native_numerical_sources/observed': 'ed759479073708213e7c307023e411c52b4c3869a43e830b66ea350694cb26ea',
    'native_precision_dispatch_sources/observed_generation2': '8c51e63de178192bb941ef7ab60408d1105c86901b9ac03460280a877e14fea4',
    'native_precision_dispatch_sources/observed_generation3': 'bd1282efee6dd57aefcaaaaac332ef36a2c58f5ba8102101ee7bbd9207790fd5',
}
ENV_NAMES = ('VLLM_USE_DEEP_GEMM', 'VLLM_USE_DEEP_GEMM_E8M0', 'VLLM_MOE_USE_DEEP_GEMM')


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def require_cloud_mount():
    assert os.path.ismount('/mnt/codesign-exp'), 'Required cloud volume is not mounted'


def run(task, output, collector, precision):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.step5.nvidia.model import Step5ForCausalLM
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
    from vllm.utils import deep_gemm
    import vllm.envs as envs

    started = time.perf_counter()
    raw_env = {name: os.environ.get(name) for name in ENV_NAMES}
    native = EngineArgs(model=str(task/'reference_inputs'), load_format='dummy', skip_tokenizer_init=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, max_num_seqs=4,
        max_num_batched_tokens=512, max_model_len=8192).create_engine_config()
    native = native.with_hf_config(native.model_config.hf_text_config)
    configure_quant_config(native.quant_config, Step5ForCausalLM)
    text, quant = native.model_config.hf_config, native.quant_config
    assert text.num_hidden_layers == 92 and text.moe_num_experts == 352 and len(text.layer_types_with_mtp) == 95
    assert len(quant.ignored_layers) == 1406
    require_cloud_mount()
    assert not CLOUD.exists()
    CLOUD.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    frames = []
    with set_current_vllm_config(native), torch.no_grad():
        for name, value in precision.inputs(torch):
            # The native entry point resolves use_ue8m0 itself; do not pass a mode.
            q, scales = moe_kernel_quantize_input(value, None, torch.float8_e4m3fn,
                per_act_token_quant=False, block_shape=quant.weight_block_size)
            torch.cuda.synchronize()
            records = precision.retain(torch, [('input', value), ('quantized', q), ('scales', scales)], CLOUD/name, collector)
            frames.append(dict(input=name, tensors=records, finite=bool(torch.isfinite(q.float()).all() and torch.isfinite(scales).all())))
            collector.dump_json(output/'frames.json', dict(frames=frames))
            assert frames[-1]['finite']
    # These reads follow the quantizer's natural first-use initialization; no cache reset.
    supported = deep_gemm.is_deep_gemm_supported()
    use_ue8m0 = deep_gemm.is_deep_gemm_e8m0_used()
    modules = []
    for name in ('vllm._C_stable_libtorch', 'vllm.third_party.deep_gemm',
                 'vllm.third_party.deep_gemm._C', 'vllm.third_party.deep_gemm._C_extension'):
        module = sys.modules.get(name)
        if module is not None and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            modules.append(dict(name=name, path=str(path), size_bytes=path.stat().st_size,
                                sha256=collector.sha256_file(path)))
    raw_bytes = sum(record['size_bytes'] for frame in frames for record in frame['tensors'])
    assert raw_bytes <= 4*1024**2
    collector.dump_json(output/'binding.json', dict(scope='new_process_quantizer_binding_not_historical_04e_factory_or_compiled_bound',
        layers=92, declarations=95, experts=352, exclusion_count=1406, dtype=str(native.model_config.dtype),
        block_shape=quant.weight_block_size, raw_environment=raw_env,
        resolved_environment={name:getattr(envs,name) for name in ENV_NAMES},
        deep_gemm_supported=supported, use_ue8m0=use_ue8m0,
        fp8_gemm_symbol_available=deep_gemm._fp8_gemm_nt_impl is not None,
        imported_modules=modules, frames=frames, raw_bytes=raw_bytes,
        numerical_guarantee='UNVERIFIED: runtime mode and bytes do not establish a compiled-kernel bound',
        host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        wall_seconds=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit','run'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); task, output = args.task_root.resolve(), args.output_dir.resolve()
    expected_task = Path(__file__).resolve().parents[3]/'task_memory/task_2026-10-15_step5_preview_h800'
    assert task == expected_task and output.is_relative_to(task) and output != task and not output.exists()
    precision = load(Path(__file__).with_name('check_precision_ffn.py'))
    launcher = load(task/'native_source_export/submit_export.py')
    collector = load(task/'native_g1/run_g1_cases.py')
    launcher.inspection.locked_inputs()
    assert task == launcher.inspection.TASK_ROOT and output.is_relative_to(task) and output != task and not output.exists()
    assert collector.sha256_file(Path(precision.__file__)) == '7cb9acf7b030fd61dd545abe9dc989c1934b148684c5f8775f7625c6c9cabeb7'
    _, manifest, _ = collector.verify_inputs(task/'g1_preparation.json')
    files = list(manifest['files'])
    batches = {batch+'/observed':digest for batch,digest in precision.SOURCE_BATCHES.items()} | MANIFESTS
    for batch,digest in batches.items():
        path = task/batch/'source_manifest.json'
        assert collector.sha256_file(path) == digest
        files += [dict(relative_path=row['relative_path'], source_sha256=row['sha256'], source_size_bytes=row['size_bytes'])
                  for row in json.loads(path.read_text())['files'] if row['status']=='copied']
    collector.IMPORTED_FILES = [row['relative_path'] for row in files]
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND']=='rjob'
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(launcher.inspection.WORKTREE); os.environ['EXP_ID']='step5-quantization-binding'
        command = 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; exec python3 -B '+shlex.join([str(Path(__file__).resolve()),'run','--task-root',str(task),'--output-dir',str(output)])
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()),
                              **launcher.spawn_arguments('timeout 600 bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    environment = collector.record_environment(output)
    assert environment['device_name']=='NVIDIA H800'
    collector.verify_installed({'files': files})
    collector.dump_json(output/'worker.json',dict(sha256=collector.sha256_file(Path(__file__))))
    run(task,output,collector,precision)


if __name__ == '__main__':
    main()
