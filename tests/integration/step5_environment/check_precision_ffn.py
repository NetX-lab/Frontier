#!/usr/bin/env python3
"""Execute unchanged Step5 precision modules and retain native tensor content."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time
import traceback

SOURCE_BATCHES = {
    'native_runtime_sources': 'a4c4722e16f0da3d4968f06b8ffde9cc978e834fc28b865da329de9487b82308',
    'native_precision_sources': '651f041ffff31f6d0b120edf35f31f600872f9d94beb2b9bc74812711eb2334f',
    'native_ffn_execution_sources': '835e6dc80780d614deac7edae954e59beceb448df021b67d3f30312d8417e333',
    'native_execution_sources': 'e91531fa6c308212ab355d6065ab957fa54e454800df27bc5592af017242468c',
    'native_workspace_sources': '83a3b8d9f5bd54614e26b539042adbe924bfea387d4f2c70492cc74b503d48e6',
}
CLOUD_ROOT = Path('/mnt/codesign-exp/ycfeng/step5-preview/g1_precision_ffn_20261017_01')
CASES = ('dense', 'moe_fp8', 'moe_excluded', 'swa_projection', 'dsa_projection', 'output_projection')


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(torch):
    columns = torch.arange(4096, device='cuda', dtype=torch.float32)
    rows = torch.stack([((columns * (2*r+1) + 7*r) % 257 - 128) / 128 for r in range(4)]).bfloat16()
    return [('zero', torch.zeros_like(rows)), ('nonconstant', rows), ('large', rows * 16384)]


def retain(torch, tensors, directory, collector):
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for name, tensor in tensors:
        pieces = tensor.unbind(0) if tensor.ndim == 3 and tensor.shape[0] == 352 else (tensor,)
        for index, piece in enumerate(pieces):
            path = directory / f'{len(records):05d}.pt'
            with path.open('xb') as stream:
                torch.save(piece.detach().cpu(), stream)
            records.append(dict(name=name, expert=index if len(pieces) == 352 else None,
                shape=list(piece.shape), dtype=str(piece.dtype), stride=list(piece.stride()),
                path=str(path), sha256=collector.sha256_file(path), size_bytes=path.stat().st_size))
    collector.dump_json(directory/'manifest.json', dict(tensors=records))
    return records


def construct(torch, name, native):
    from vllm.models.step5.nvidia.model import Step5MLP, Step5MoE, Step5FusedQKVIndexerLinear
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
    text, quant = native.model_config.hf_config, native.quant_config
    if name == 'dense':
        return Step5MLP(text, 4096, 13824, 'silu', quant_config=quant, prefix='model.layers.0.mlp')
    if name in ('moe_fp8', 'moe_excluded'):
        return Step5MoE(native, prefix=f'model.layers.{3 if name == "moe_fp8" else 88}.moe')
    if name == 'swa_projection':
        return MergedColumnParallelLinear(4096, [12288, 768, 768, 64], bias=False,
            quant_config=quant, prefix='model.layers.0.self_attn.qkv_proj')
    if name == 'dsa_projection':
        return Step5FusedQKVIndexerLinear(4096, 192, 64, 4, indexer_q_output_size=4096,
            indexer_kv_output_size=256, gate_output_size=64, proxy_dim=256, index_tp_size=4,
            quant_config=quant, prefix='model.layers.3.self_attn.qkv_indexer_proj')
    return RowParallelLinear(12288, 4096, bias=False, quant_config=quant, prefix='model.layers.0.self_attn.o_proj')


def execute(task, output, collector, case):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (init_distributed_environment, initialize_model_parallel,
        get_world_group, get_tp_group, get_dp_group, get_pp_group, get_ep_group)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
    from vllm.model_executor.model_loader.utils import configure_quant_config, process_weights_after_loading
    from vllm.models.step5.nvidia.model import Step5ForCausalLM
    from vllm.v1.worker.workspace import init_workspace_manager
    config = EngineArgs(model=str(task/'reference_inputs'), load_format='dummy', skip_tokenizer_init=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, max_num_seqs=4,
        max_num_batched_tokens=512, max_model_len=8192).create_engine_config()
    native = config.with_hf_config(config.model_config.hf_text_config)
    configure_quant_config(native.quant_config, Step5ForCausalLM)
    text = native.model_config.hf_config
    assert text.num_hidden_layers == 92 and text.moe_num_experts == 352 and len(text.layer_types_with_mtp) == 95
    assert len(native.quant_config.ignored_layers) == 1406
    torch.set_default_dtype(native.model_config.dtype)
    torch.cuda.set_device(0)
    assert CLOUD_ROOT.is_dir()
    with set_current_vllm_config(native), torch.device('cuda'), torch.no_grad():
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
            distributed_init_method=f'file:///tmp/step5_precision_dist_{case}', backend='nccl')
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device('cuda'))
        groups = {n: dict(world_size=g().world_size, rank=g().rank,
            backend=torch.distributed.get_backend(g().device_group)) for n, g in
            [('world', get_world_group), ('tp', get_tp_group), ('dp', get_dp_group), ('pp', get_pp_group), ('ep', get_ep_group)]}
        assert all(row['world_size'] == 1 and row['rank'] == 0 for row in groups.values())
        collector.dump_json(output/'configuration.json', dict(layers=92, declarations=95, experts=352,
            dtype=str(native.model_config.dtype), quantization=type(native.quant_config).__name__, groups=groups,
            exclusion_count=1406, weight_block_size=native.quant_config.weight_block_size, cloud_root=str(CLOUD_ROOT)))
        results = []
        for name in (case,):
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            row = dict(case=name)
            try:
                module = construct(torch, name, native)
                print('PRECISION_PHASE', name, 'constructed', flush=True)
                record_metadata_for_reloading(module)
                DummyModelLoader(native.load_config).load_weights(module, native.model_config)
                print('PRECISION_PHASE', name, 'dummy_loaded', flush=True)
                tensors = list(module.named_parameters()) + list(module.named_buffers())
                row['pre'] = retain(torch, tensors, CLOUD_ROOT/name/'pre', collector)
                del tensors
                process_weights_after_loading(module, native.model_config, torch.device('cuda'))
                print('PRECISION_PHASE', name, 'postprocessed', flush=True)
                row['post'] = retain(torch, list(module.named_parameters()) + list(module.named_buffers()), CLOUD_ROOT/name/'post', collector)
                row['methods'] = {n: type(m.quant_method).__name__ for n, m in module.named_modules() if hasattr(m, 'quant_method')}
                row['tensor_metadata'] = {n: dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()))
                    for n, t in list(module.named_parameters()) + list(module.named_buffers())}
                if name.startswith('moe_'):
                    routed = module.experts.routed_experts
                    expected = torch.float8_e4m3fn if name == 'moe_fp8' else torch.bfloat16
                    assert routed.w13_weight.dtype == expected and routed.w2_weight.dtype == expected
                    assert routed.w13_weight.shape[0] == 352 and routed.w2_weight.shape[0] == 352
                    assert module.gate.weight.dtype == torch.float32 and module.router_bias.dtype == torch.float32
                    row['backend'] = str(getattr(routed.quant_method, 'fp8_backend', getattr(routed.quant_method, 'unquantized_backend', None)))
                else:
                    assert all(t.dtype == torch.bfloat16 for t in module.parameters())
                captures = {}
                def capture(n):
                    def hook(m, x, y):
                        captures[n] = (y[0] if isinstance(y, tuple) else y).detach().clone()
                    return hook
                hooks = [m.register_forward_hook(capture(n)) for n, m in module.named_modules()
                    if n.endswith(('gate_up_proj', 'act_fn', 'down_proj')) or n == 'gate']
                row['frames'] = []
                for label, x in inputs(torch):
                    if name == 'output_projection':
                        x = x.repeat(1, 3)
                    captures.clear()
                    with set_forward_context(None, native, num_tokens=x.shape[0]):
                        y = module(x)
                        ys = y if isinstance(y, tuple) else (y,)
                        actual = [('input', x)] + [(f'output_{i}', value) for i, value in enumerate(ys) if value is not None]
                        actual += list(captures.items())
                        if name == 'dsa_projection':
                            actual += [(f'indexer_{i}', value) for i, value in enumerate(module.split_indexer(y[1]))]
                        if name.startswith('moe_'):
                            logits = captures['gate']
                            weights, ids = module.experts.router.select_experts(x, logits,
                                topk_indices_dtype=module.experts.routed_experts.quant_method.topk_indices_dtype)
                            actual += [('routing_logits', logits), ('routing_weights', weights), ('routing_ids', ids)]
                    checks = dict(finite=all(torch.isfinite(v).all().item() for v in ys if v is not None),
                        zero=label != 'zero' or all((v == 0).all().item() for v in ys if v is not None))
                    frame = dict(input=label, checks=checks, tensors=retain(torch, actual, CLOUD_ROOT/name/label, collector))
                    frame['clamp_witnesses'] = {n: dict(gate_silu_over_limit=(torch.nn.functional.silu(t.float().chunk(2, dim=-1)[0]) > 7).sum().item(),
                        up_over_limit=(t.float().chunk(2, dim=-1)[1].abs() > 7).sum().item())
                        for n, t in captures.items() if n.endswith('gate_up_proj')}
                    row['frames'].append(frame)
                    assert all(checks.values()), checks
                for hook in hooks:
                    hook.remove()
                if name.startswith('moe_'):
                    method = module.experts.routed_experts.quant_method
                    row['backend'] = str(getattr(method, 'fp8_backend', getattr(method, 'unquantized_backend', None)))
                    row['experts_class'] = method.experts_cls.__name__
                    row['executed_kernel'] = type(method.moe_kernel.fused_experts).__name__
                    row['routing_scope'] = 'separate native exposed router call on actual logits; not an internal monolithic observation'
                row['status'] = 'executed_finite_zero_checked_general_content_unverified'
                del module, captures, hooks, actual, ys, y, x
                torch.cuda.empty_cache()
            except Exception:
                row.update(status='failed', error=traceback.format_exc())
            row.update(wall_seconds=time.perf_counter()-started, host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            results.append(row)
            collector.dump_json(output/(name+'.json'), row)
            print('PRECISION_MODULE', name, row['status'], flush=True)
        collector.dump_json(output/'results.json', dict(cases=results))
        assert all(row['status'] != 'failed' for row in results), 'Native module failures retained'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'run'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    task, output = args.task_root.resolve(), args.output_dir.resolve()
    launcher = load(task/'native_source_export/submit_export.py')
    launcher.inspection.locked_inputs()
    assert task == launcher.inspection.TASK_ROOT
    assert output.is_relative_to(task) and output != task, 'Output must be a new directory inside the current task'
    assert not output.exists()
    collector = load(task/'native_g1/run_g1_cases.py')
    _, manifest, _ = collector.verify_inputs(task/'g1_preparation.json')
    files = list(manifest['files'])
    for batch, digest in SOURCE_BATCHES.items():
        path = task/batch/'observed/source_manifest.json'
        assert collector.sha256_file(path) == digest
        files += [dict(relative_path=r['relative_path'], source_sha256=r['sha256'], source_size_bytes=r['size_bytes'])
            for r in json.loads(path.read_text())['files'] if r['status'] == 'copied']
    collector.IMPORTED_FILES = [row['relative_path'] for row in files]
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND'] == 'rjob'
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(launcher.inspection.WORKTREE)
        os.environ['EXP_ID'] = 'step5-g1-precision-ffn'
        command = 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; exec python3 -B '+shlex.join([str(Path(__file__).resolve()), 'run', '--task-root', str(task), '--output-dir', str(output)])
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()),
            **launcher.spawn_arguments('timeout 2400 bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    output.mkdir(parents=True, exist_ok=False)
    collector.verify_installed(dict(files=files))
    collector.record_environment(output)
    collector.dump_json(output/'worker.json', dict(sha256=collector.sha256_file(Path(__file__))))
    if args.case:
        try:
            execute(task, output, collector, args.case)
        except Exception:
            collector.dump_json(output/'failure.json', dict(error=traceback.format_exc(),
                host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            raise
        return
    assert os.path.ismount('/mnt/codesign-exp'), 'Required cloud volume is not mounted'
    CLOUD_ROOT.mkdir(parents=True, exist_ok=False)
    statuses = []
    for case in CASES:
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'run',
            '--task-root', str(task), '--output-dir', str(output/case), '--case', case])
        statuses.append(dict(case=case, exit_code=result.returncode))
        collector.dump_json(output/'case_statuses.json', dict(cases=statuses))
        retained_bytes = sum(path.stat().st_size for path in CLOUD_ROOT.rglob('*.pt'))
        assert retained_bytes <= 64 * 1024**3, 'Frozen raw-tensor storage budget exceeded'
    assert all(row['exit_code'] == 0 for row in statuses), 'Native failures retained in independent case processes'


if __name__ == '__main__':
    main()
