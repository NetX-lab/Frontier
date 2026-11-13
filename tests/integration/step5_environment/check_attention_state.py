#!/usr/bin/env python3
"""Retain actual native Step5 attention continuation and isolated decode replay."""
import argparse
import copy
import dataclasses
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

def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


precision = load(Path(__file__).with_name('check_precision_ffn.py'))
retain, SOURCE_BATCHES = precision.retain, precision.SOURCE_BATCHES

CASES = ('dsa_short', 'swa', 'dsa_long')
CLOUD_ROOT = Path('/mnt/codesign-exp/ycfeng/step5-preview/g1_attention_state_20261017_03')


def memory(torch):
    return dict(host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(),
        cuda_peak_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())


def execute(task, output, collector, case):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, get_world_group, get_tp_group, get_dp_group, get_pp_group, get_ep_group
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
    from vllm.model_executor.model_loader.utils import configure_quant_config, process_weights_after_loading
    from vllm.models.step5.nvidia.model import Step5Attention, Step5ForCausalLM, _get_step5_layer_types
    from vllm.models.step5.nvidia.layernorm import get_step5_norm_dtype
    from vllm.models.step5.nvidia.config import get_step5_sparse_config
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheTensor, create_kv_cache_views
    from vllm.v1.worker.workspace import init_workspace_manager
    native = EngineArgs(model=str(task/'reference_inputs'), load_format='dummy', skip_tokenizer_init=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, max_num_seqs=4,
        max_num_batched_tokens=512, max_model_len=8192).create_engine_config()
    native = native.with_hf_config(native.model_config.hf_text_config)
    configure_quant_config(native.quant_config, Step5ForCausalLM)
    text, sparse = native.model_config.hf_config, get_step5_sparse_config(native.model_config.hf_config)
    assert text.num_hidden_layers == 92 and text.moe_num_experts == 352 and len(text.layer_types_with_mtp) == 95
    torch.set_default_dtype(native.model_config.dtype)
    torch.cuda.set_device(0)
    phases, frames, started = [], [], time.perf_counter()
    def phase(name):
        phases.append(dict(phase=name, **memory(torch)))
        collector.dump_json(output/'phases.json', dict(phases=phases))
    with set_current_vllm_config(native), torch.device('cuda'), torch.no_grad():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method=f'file:///tmp/step5_attention_{case}', backend='nccl')
        initialize_model_parallel(1, 1)
        init_workspace_manager(torch.device('cuda'))
        groups = {n: dict(size=g().world_size, rank=g().rank, backend=torch.distributed.get_backend(g().device_group))
            for n, g in [('world', get_world_group), ('tp', get_tp_group), ('dp', get_dp_group), ('pp', get_pp_group), ('ep', get_ep_group)]}
        assert len(native.quant_config.ignored_layers) == 1406
        assert all(row['size'] == 1 and row['rank'] == 0 for row in groups.values())
        phase('configured')
        index = 0 if case == 'swa' else 3
        cache_config = copy.copy(native.cache_config)
        cache_config.sliding_window = None
        module = Step5Attention(hidden_size=text.hidden_size, num_heads=text.num_attention_heads,
            num_kv_heads=text.num_attention_groups, max_position=text.max_position_embeddings, head_dim=text.head_dim,
            rms_norm_eps=text.rms_norm_eps, qkv_bias=getattr(text, 'attention_bias', False), rope_theta=text.rope_theta,
            cache_config=cache_config, quant_config=native.quant_config, rope_scaling=getattr(text, 'rope_scaling', None),
            prefix=f'model.layers.{index}.self_attn', sliding_window=text.sliding_window,
            use_head_wise_attn_gate=text.use_head_wise_attn_gate, layer_types=_get_step5_layer_types(text),
            use_rope_layers=getattr(text, 'use_rope_layers', []), yarn_only_types=getattr(text, 'yarn_only_types', []),
            swa_num_attention_heads=getattr(text, 'swa_num_attention_heads', None), partial_rotary_factor=text.partial_rotary_factors[index],
            zero_centered=text.zero_centered, vllm_config=native, sparse_config=None if case == 'swa' else sparse,
            norm_dtype=get_step5_norm_dtype(text))
        phase('constructed')
        record_metadata_for_reloading(module)
        DummyModelLoader(native.load_config).load_weights(module, native.model_config)
        phase('dummy_loaded')
        pre_weights = retain(torch, list(module.named_parameters())+list(module.named_buffers()), CLOUD_ROOT/case/'pre', collector)
        process_weights_after_loading(module, native.model_config, torch.device('cuda'))
        phase('postprocessed')
        weights = retain(torch, list(module.named_parameters())+list(module.named_buffers()), CLOUD_ROOT/case/'post', collector)
        layer, r, budget = module.attn, sparse.region_block_size, native.scheduler_config.max_num_batched_tokens
        chunks = [text.sliding_window-1, 2, 1] if case == 'swa' else ([2*r-1, 2, 1] if case == 'dsa_short' else [budget]*7+[budget-1, 2, r, 1])
        spec = layer.get_kv_cache_spec(native)
        pages = (sum(chunks)+4+spec.block_size-1)//spec.block_size
        raw = torch.zeros(pages*spec.page_size_bytes, dtype=torch.int8, device='cuda')
        layer.kv_cache = create_kv_cache_views(raw, spec, pages, layer.attn_backend.supported_kv_cache_layouts()[0],
            KVCacheTensor(size=raw.numel(), layers=[layer.layer_name], layer_stride=raw.numel(), block_stride=spec.page_size_bytes))[0]
        if module.sparse_summary_cache is not None:
            module.sparse_summary_cache.bind_kv_cache_side_storage(native.compilation_config.static_forward_context)
        assert layer.kv_cache.dtype == torch.bfloat16
        assert list(layer.kv_cache.shape) == [pages, module.num_kv_heads, spec.block_size, 2*module.head_dim]
        assert layer.kv_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert not module.fuse_swa_gate
        initial = retain(torch, [('raw_cache', layer.kv_cache)]+([] if case == 'swa' else
            [(n, getattr(module.sparse_summary_cache._cache, n)) for n in ('mean', 'numerator', 'denominator', 'maximum', 'last_slot')]), CLOUD_ROOT/case/'initial', collector)
        builder = layer.attn_backend.get_builder_cls()(kv_cache_spec=spec, layer_names=[layer.layer_name], vllm_config=native, device=torch.device('cuda'))
        table = torch.arange(pages-1, -1, -1, dtype=torch.int32, device='cuda')[None]
        captures = []
        def observe(owner, name, capture):
            original = getattr(owner, name)
            def call(*args, **kwargs):
                result = original(*args, **kwargs)
                captures.extend((n, t.detach().clone()) for n, t in capture(args, kwargs, result))
                return result
            setattr(owner, name, call)
        observe(layer.impl, 'forward', lambda a, k, y: [('query', a[1]), ('key', a[2]), ('value', a[3]), ('attention_output', k['output'] if 'output' in k else a[6])]
            + [(n, k[n]) for n in ('dsa_proxy_query', 'dsa_proxy_key', 'dsa_proxy_weights', 'dsa_proxy_z', 'dsa_gate') if k.get(n) is not None])
        if case != 'swa':
            assert layer.impl.adapter is not None
            observe(layer.impl.adapter, 'logits', lambda a, k, y: [('scores', y)])
            observe(layer.impl.adapter, 'decode_logits', lambda a, k, y: [('scores', y[0]), ('current', y[1].current), ('topk_lengths', y[1].topk_lengths)])
            observe(layer.impl.adapter, 'select_regions', lambda a, k, y: [('selected', y), ('current', a[1]), ('topk_lengths', k['topk_lengths'])])
        phase('cache_bound')
        previous, graph = 0, None
        for step, count in enumerate(chunks+[1]*4):
            positions = torch.arange(previous, previous+count, device='cuda')
            slots = table[0, positions//spec.block_size].long()*spec.block_size+positions%spec.block_size
            starts = torch.tensor([0, count], dtype=torch.int32, device='cpu')
            lengths = torch.tensor([previous+count], dtype=torch.int32, device='cpu')
            common = CommonAttentionMetadata(query_start_loc=starts.cuda(), query_start_loc_cpu=starts, seq_lens=lengths.cuda(),
                num_reqs=1, num_actual_tokens=count, max_query_len=count, max_seq_len=sum(chunks)+4,
                block_table_tensor=table, slot_mapping=slots, positions=positions, seq_lens_cpu_upper_bound=lengths)
            with torch.device('cpu'):
                metadata = builder.build(0, common, fast_build=False)
            x = (((positions[:, None]+1)*(torch.arange(text.hidden_size, device='cuda')[None]+3))%257-128).float().div(128).bfloat16()
            before = layer.kv_cache.clone()
            if graph is None:
                captures.clear()
                with set_forward_context({layer.layer_name: metadata}, native, num_tokens=count, slot_mapping={layer.layer_name: slots}):
                    if step == len(chunks):
                        graph = torch.cuda.CUDAGraph()
                        graph_positions, graph_x, graph_slots, graph_lengths = positions, x, slots, common.seq_lens
                        graph_common, graph_metadata = common, metadata
                        with torch.cuda.graph(graph):
                            y = module(positions, x)
                        graph.replay()
                    else:
                        y = module(positions, x)
            else:
                graph_positions.copy_(positions); graph_x.copy_(x); graph_slots.copy_(slots); graph_lengths.copy_(common.seq_lens)
                graph.replay()
            torch.cuda.synchronize()
            k, v = layer.impl._split_kv_cache(layer.kv_cache)
            prepared = {n: t for n, t in captures if n in ('key', 'value')}
            exact_key = torch.equal(k[slots//spec.block_size, slots%spec.block_size], prepared['key'].view(count, module.num_kv_heads, module.head_dim))
            exact_value = torch.equal(v[slots//spec.block_size, slots%spec.block_size], prepared['value'].view(count, module.num_kv_heads, module.head_dim))
            changed = torch.zeros((pages, spec.block_size), dtype=torch.bool, device='cuda'); changed[slots//spec.block_size, slots%spec.block_size] = True
            untouched = torch.equal(layer.kv_cache.transpose(1, 2)[~changed], before.transpose(1, 2)[~changed])
            tensors = [('input', x), ('positions', positions), ('slots', slots), ('block_table', table), ('output', y), ('raw_cache', layer.kv_cache)]+captures
            if graph is not None:
                for name in ('query_start_loc', 'seq_lens', 'scheduler_metadata'):
                    value = getattr(graph_metadata, name, None)
                    if value is not None:
                        tensors.append(('graph_'+name, value))
            plan = metadata.step5_runtime if case != 'swa' else None
            if plan is not None:
                for chunk_index, entry in enumerate(plan[4]):
                    tensors += [(f'plan_{chunk_index}_{n}', getattr(entry, n)) for n in ('current', 'topk_lengths', 'selected') if getattr(entry, n) is not None]
            if module.sparse_summary_cache is not None:
                cache = module.sparse_summary_cache._cache
                tensors += [(n, getattr(cache, n)) for n in ('mean', 'numerator', 'denominator', 'maximum', 'last_slot')]
            records = retain(torch, tensors, CLOUD_ROOT/case/f'step_{step:02}', collector)
            frames.append(dict(step=step, previous=previous, count=count, total=previous+count, graph='capture' if step == len(chunks) else ('replay' if step > len(chunks) else 'eager'),
                tensors=records, finite=torch.isfinite(y).all().item(), exact_key=exact_key, exact_value=exact_value, untouched=untouched, native_plan=plan is not None,
                staged=plan is not None and plan[7] is not None, paged_prefill=plan is not None and plan[5] is not None,
                graph_metadata_pointers={} if graph is None else {name: value.data_ptr()
                    for name in ('query_start_loc', 'seq_lens', 'scheduler_metadata')
                    if (value := getattr(graph_metadata, name, None)) is not None}, **memory(torch)))
            collector.dump_json(output/'frames.json', dict(frames=frames))
            assert frames[-1]['finite'] and exact_key and exact_value and untouched
            previous += count
        spec_record = dataclasses.asdict(spec)
        spec_record['dtype'] = str(spec.dtype)
        collector.dump_json(output/(case+'.json'), dict(case=case, groups=groups, layers=92, declarations=95, experts=352,
            pre_weights=pre_weights, weights=weights, initial=initial, phases=phases, frames=frames, spec=spec_record, backend=layer.attn_backend.get_name(),
            gpu_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
            builder=type(builder).__name__, fuse_swa_gate=module.fuse_swa_gate,
            compilation_cudagraph_mode=str(native.compilation_config.cudagraph_mode),
            builder_use_full_cuda_graph=getattr(builder, 'use_full_cuda_graph', None),
            builder_aot_schedule=getattr(builder, 'aot_schedule', None),
            builder_swa_tile_alignment=getattr(builder, 'swa_tile_alignment', None),
            flash_attn_version=getattr(layer.impl, 'vllm_flash_attn_version', None),
            wall_seconds=time.perf_counter()-started, **memory(torch)))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=['submit', 'run'])
    parser.add_argument('--task-root', type=Path, required=True); parser.add_argument('--output-dir', type=Path, required=True); parser.add_argument('--case', choices=CASES)
    parser.add_argument('--cases', choices=CASES, nargs='+', default=CASES)
    args = parser.parse_args(); task, output = args.task_root.resolve(), args.output_dir.resolve()
    launcher, collector = load(task/'native_source_export/submit_export.py'), load(task/'native_g1/run_g1_cases.py')
    launcher.inspection.locked_inputs(); assert task == launcher.inspection.TASK_ROOT and output.is_relative_to(task) and output != task and not output.exists()
    assert collector.sha256_file(Path(precision.__file__)) == '7cb9acf7b030fd61dd545abe9dc989c1934b148684c5f8775f7625c6c9cabeb7'
    _, manifest, _ = collector.verify_inputs(task/'g1_preparation.json'); files = list(manifest['files'])
    for batch, digest in SOURCE_BATCHES.items():
        path = task/batch/'observed/source_manifest.json'; assert collector.sha256_file(path) == digest
        files += [dict(relative_path=r['relative_path'], source_sha256=r['sha256'], source_size_bytes=r['size_bytes']) for r in json.loads(path.read_text())['files'] if r['status'] == 'copied']
    collector.IMPORTED_FILES = [r['relative_path'] for r in files]
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND'] == 'rjob'
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(launcher.inspection.WORKTREE); os.environ['EXP_ID'] = 'step5-g1-attention-state'
        command = 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; exec python3 -B '+shlex.join([str(Path(__file__).resolve()), 'run', '--task-root', str(task), '--output-dir', str(output), '--cases', *args.cases])
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()), **launcher.spawn_arguments('timeout 2400 bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True); return
    output.mkdir(parents=True); collector.verify_installed(dict(files=files)); collector.record_environment(output)
    collector.dump_json(output/'worker.json', dict(sha256=collector.sha256_file(Path(__file__))))
    try:
        if args.case:
            execute(task, output, collector, args.case); return
        assert os.path.ismount('/mnt/codesign-exp'); CLOUD_ROOT.mkdir(parents=True, exist_ok=False)
        statuses = []
        for case in args.cases:
            code = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'run', '--task-root', str(task), '--output-dir', str(output/case), '--case', case]).returncode
            statuses.append(dict(case=case, exit_code=code)); collector.dump_json(output/'case_statuses.json', dict(cases=statuses))
            assert sum(p.stat().st_size for p in CLOUD_ROOT.rglob('*.pt')) <= 4*1024**3
        assert all(r['exit_code'] == 0 for r in statuses)
    except Exception:
        collector.dump_json(output/'failure.json', dict(error=traceback.format_exc())); raise
    finally:
        import torch
        collector.dump_json(output/'resources.json', memory(torch))


if __name__ == '__main__':
    main()
