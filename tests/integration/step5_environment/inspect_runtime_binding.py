#!/usr/bin/env python3
"""Inspect native Step5 engine configuration, attention specs and cache binding."""
import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import shlex
import time
import traceback


RUNTIME_SOURCE_MANIFEST_SHA256 = 'a4c4722e16f0da3d4968f06b8ffde9cc978e834fc28b865da329de9487b82308'


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect(task, output, collector):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.models.step5.nvidia.model import Step5ForCausalLM
    from vllm.models.step5.nvidia.config import get_step5_sparse_config
    from vllm.models.step5.nvidia.sparse_attention import Step5DSAAttentionBackend, Step5SplitKVFlashAttentionBackend
    from vllm.models.step5.nvidia.sparse_summary_cache import Step5SparseSummaryCacheLayer
    from vllm.v1.kv_cache_interface import KVCacheTensor, create_kv_cache_views

    supplied_arguments = dict(model=str(task/'reference_inputs'), load_format='dummy',
                              skip_tokenizer_init=True, enable_prefix_caching=False,
                              enable_chunked_prefill=True, max_num_seqs=4,
                              max_num_batched_tokens=512, max_model_len=8192)
    arguments = EngineArgs(**supplied_arguments)
    config = arguments.create_engine_config()
    text = config.model_config.hf_text_config
    native = config.with_hf_config(text)
    configure_quant_config(native.quant_config, Step5ForCausalLM)
    assert text.num_hidden_layers == 92 and text.moe_num_experts == 352
    assert text.num_nextn_predict_layers == 3
    assert len(text.layer_types) == 92 and len(text.layer_types_with_mtp) == 95
    sparse = get_step5_sparse_config(text)
    rows = []
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(native.model_config.dtype)
    try:
        with set_current_vllm_config(native):
            for index, backend in ((0, Step5SplitKVFlashAttentionBackend), (3, Step5DSAAttentionBackend)):
                window = text.sliding_window if text.layer_types[index] == 'sliding_attention' else None
                prefix = f'model.layers.{index}.self_attn.attn'
                extra = dict(sparse_config=sparse, max_model_len=native.model_config.max_model_len,
                             max_num_seqs=native.scheduler_config.max_num_seqs) if backend is Step5DSAAttentionBackend else {}
                layer = Attention(text.num_attention_heads, text.head_dim, text.head_dim**-0.5,
                                  num_kv_heads=text.num_attention_groups,
                                  cache_config=native.cache_config, quant_config=None,
                                  prefix=prefix, per_layer_sliding_window=window,
                                  attn_backend=backend, **extra)
                spec = layer.get_kv_cache_spec(native)
                layouts = backend.supported_kv_cache_layouts()
                assert len(layouts) == 1
                layout = layouts[0]
                pages = 4
                raw = torch.zeros(pages*spec.page_size_bytes, device='cuda', dtype=torch.int8)
                tensor_spec = KVCacheTensor(size=raw.numel(), layers=[prefix],
                                           layer_stride=pages*spec.page_size_bytes,
                                           block_stride=spec.page_size_bytes)
                layer.kv_cache = create_kv_cache_views(raw, spec, pages, layout, tensor_spec)[0]
                summary = None
                if backend is Step5DSAAttentionBackend:
                    side = Step5SparseSummaryCacheLayer(prefix=f'model.layers.{index}.self_attn.summary_cache',
                        target_impl=layer.impl, main_layer_name=prefix,
                        static_forward_context=native.compilation_config.static_forward_context)
                    side.bind_kv_cache_side_storage(native.compilation_config.static_forward_context)
                    cache = side._cache
                    summary = dict(mean_shape=list(cache.mean.shape), mean_dtype=str(cache.mean.dtype),
                        numerator_shape=list(cache.numerator.shape), denominator_shape=list(cache.denominator.shape),
                        maximum_shape=list(cache.maximum.shape), last_slot_shape=list(cache.last_slot.shape),
                        pages=cache.pages, block_size=cache.block_size,
                        layer_alias=side.kv_cache.untyped_storage().data_ptr() == cache.mean.untyped_storage().data_ptr(),
                        target_alias=layer.impl.summary_cache is cache)
                scales = {name: dict(shape=list(getattr(layer, name).shape), dtype=str(getattr(layer, name).dtype),
                                    value=getattr(layer, name).item()) for name in ('_k_scale', '_v_scale', '_q_scale')}
                assert layer.kv_cache.dtype == native.model_config.dtype
                assert list(layer.kv_cache.shape) == [pages, text.num_attention_groups, spec.block_size, 2*text.head_dim]
                assert layer.kv_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
                assert all(torch.isfinite(getattr(layer, name)).all().item() for name in ('_k_scale', '_v_scale', '_q_scale'))
                assert summary is None or (summary['layer_alias'] and summary['target_alias'])
                rows.append(dict(layer=index, backend=backend.get_name(), impl=type(layer.impl).__name__,
                    heads=layer.num_heads, kv_heads=layer.num_kv_heads, dim=layer.head_size,
                    window=window, impl_window=list(layer.impl.sliding_window),
                    spec=json.loads(json.dumps(dataclasses.asdict(spec), default=str)),
                    cache_shape=list(layer.kv_cache.shape), cache_dtype=str(layer.kv_cache.dtype),
                    layout=layout.name,
                    spec_scope='standalone Attention; Step5 model recompute, zeroing and side-storage budget hooks not applied',
                    layout_scope='standalone sole supported backend layout; not engine-global allocation',
                    raw_alias=layer.kv_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr(),
                    cache_numel=layer.kv_cache.numel(), raw_bytes=raw.numel(), scales=scales, summary=summary))
                del raw, layer
    finally:
        torch.set_default_dtype(previous_dtype)
    collector.dump_json(output/'binding.json', dict(scope='native standalone TP1 attention binding; not full model or serving engine',
        engine_arguments=supplied_arguments, backbone_layers=text.num_hidden_layers,
        declared_layers=len(text.layer_types_with_mtp), experts=text.moe_num_experts,
        model_dtype=str(native.model_config.dtype), quantization=type(native.quant_config).__name__,
        exclusion_count=len(native.quant_config.ignored_layers), weight_block_size=native.quant_config.weight_block_size,
        cache_config=json.loads(json.dumps(dataclasses.asdict(native.cache_config), default=str)), rows=rows))
    print('NATIVE_RUNTIME_BINDING', len(rows), 'attention classes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['submit', 'run'])
    parser.add_argument('--task-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.task_root = args.task_root.resolve()
    args.output_dir = args.output_dir.resolve()
    launcher = load(args.task_root/'native_source_export/submit_export.py')
    launcher.inspection.locked_inputs()
    assert args.task_root == launcher.inspection.TASK_ROOT
    collector = load(args.task_root/'native_g1/run_g1_cases.py')
    _, manifest, _ = collector.verify_inputs(args.task_root/'g1_preparation.json')
    owner_path = args.task_root/'native_runtime_sources/observed/source_manifest.json'
    assert collector.sha256_file(owner_path) == RUNTIME_SOURCE_MANIFEST_SHA256
    owners = json.loads(owner_path.read_text())
    owner_manifest = [dict(relative_path=row['relative_path'], source_sha256=row['sha256'],
                           source_size_bytes=row['size_bytes']) for row in owners['files']]
    collector.IMPORTED_FILES = [r['relative_path'] for r in manifest['files']]+[r['relative_path'] for r in owner_manifest]
    manifest = dict(files=manifest['files']+owner_manifest)
    assert not args.output_dir.exists()
    if args.mode == 'submit':
        assert os.environ['STEPMIND_BACKEND'] == 'rjob'
        from steptron.exp.base_exp import ResourceConfig
        from steptron.utils.stepmind import spawn_tasks
        os.chdir(launcher.inspection.WORKTREE)
        os.environ['EXP_ID'] = 'step5-g1-runtime-binding'
        command = 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ENABLE_CUDA_COMPATIBILITY=1 LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH; exec python3 -B '+shlex.join([str(Path(__file__).resolve()), 'run', '--task-root', str(args.task_root), '--output-dir', str(args.output_dir)])
        workers = spawn_tasks(ResourceConfig(**launcher.resource_config_arguments()),
                              **launcher.spawn_arguments('timeout 600 bash -c '+shlex.quote(command)))
        print('JOB_ID', workers.rjob.meta.name, 'FINAL_STATUS', workers.poll(), flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    collector.verify_installed(manifest)
    environment = collector.record_environment(args.output_dir)
    started = time.perf_counter()
    try:
        inspect(args.task_root, args.output_dir, collector)
    except Exception:
        collector.dump_json(args.output_dir/'failure.json', dict(error=traceback.format_exc()))
        raise
    finally:
        import resource
        import torch
        collector.dump_json(args.output_dir/'summary.json', dict(environment=environment,
            worker_sha256=collector.sha256_file(Path(__file__)), wall_seconds=time.perf_counter()-started,
            host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_bytes=torch.cuda.max_memory_allocated()))


if __name__ == '__main__':
    main()
