"""Measure independent eager FFN roots; graph and attention coverage remain unmeasured."""
import json
from pathlib import Path
import time

from local_scope_manifest import admit


def samples(plan, case):
    labels = {'dense': ('boundary_new_7', 'boundary_new_8', 'boundary_new_9', 'holdout_subsequent_19'),
              'moe_fp8': ('precision_fp8_routed',), 'moe_excluded': ('precision_bf16_excluded',)}
    if case not in labels:
        raise ValueError('Only preregistered dense and routed FFN roots are supported')
    rows = {row['label']: row for row in plan['cases']}
    return [rows[label] for label in labels[case]]


def validate_topology(native, groups):
    parallel = native.parallel_config
    sizes = [parallel.tensor_parallel_size, parallel.data_parallel_size, parallel.pipeline_parallel_size]
    if sizes != [1, 1, 1] or parallel.enable_expert_parallel:
        raise ValueError('Preregistered FFN roots require native TP1/DP1/PP1 with EP disabled')
    if any(group.world_size != 1 or group.rank != 0 for group in groups):
        raise ValueError('Actual distributed groups disagree with the rank-local contract')


def linear_dispatch(module):
    result = {}
    for name in ('gate_up_proj', 'down_proj'):
        layer = getattr(module, name)
        method = layer.quant_method
        if type(method).__name__ != 'UnquantizedLinearMethod':
            raise ValueError('Both dense projections must preserve native excluded BF16 precision')
        gemm = method._gemm_impl
        result[name] = dict(method=type(method).__name__, gemm=gemm.__module__+'.'+gemm.__qualname__,
                            dtype=str(layer.weight.dtype), shape=list(layer.weight.shape),
                            stride=list(layer.weight.stride()))
    return result


def measure(task, output, native, construct, case, *, warmups, repeats):
    if warmups < 1 or repeats < 2:
        raise ValueError('Positive warmups and at least two measured repeats required')
    plan = json.loads((task/'local_scope_sampling_01.json').read_text())
    admit(plan, task)
    selected = samples(plan, case)
    import torch
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.reload.layerwise import record_metadata_for_reloading
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from frontier.profiling.utils.record_function_tracer import RecordFunctionTracer
    from vllm.models.step5.nvidia.model import Step5DecoderLayer
    from vllm.distributed.parallel_state import get_world_group, get_tp_group, get_dp_group, get_pp_group, get_ep_group
    validate_topology(native, [get_world_group(), get_tp_group(), get_dp_group(), get_pp_group(), get_ep_group()])
    module = construct(torch, case, native)
    record_metadata_for_reloading(module)
    DummyModelLoader(native.load_config).load_weights(module, native.model_config)
    process_weights_after_loading(module, native.model_config, torch.device('cuda'))
    scope = {'dense': 'dense_ffn', 'moe_fp8': 'fp8_moe_root', 'moe_excluded': 'bf16_moe_root'}[case]
    forward = module
    if case != 'dense':
        # Native FFN method owns branch combination; no attention module is allocated.
        ffn = Step5DecoderLayer.__new__(Step5DecoderLayer)
        torch.nn.Module.__init__(ffn)
        ffn.use_moe, ffn.moe = True, module
        ffn.fp32_residual_connection = native.model_config.hf_config.fp32_residual_connection
        assert module.fuse_all_reduce is False
        forward = ffn._forward_ffn
        routed = module.experts.routed_experts
        dtype = torch.float8_e4m3fn if case == 'moe_fp8' else torch.bfloat16
        assert routed.w13_weight.shape[0] == routed.w2_weight.shape[0] == 352
        assert routed.w13_weight.dtype == routed.w2_weight.dtype == dtype
        assert module.gate.weight.dtype == module.router_bias.dtype == torch.float32
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in selected:
        cell = output/sample['label']; (cell/'profiler_traces').mkdir(parents=True, exist_ok=False)
        generator = torch.Generator(device='cuda').manual_seed(sample['content_seed'])
        x = torch.randn((sample['query'], native.model_config.hf_config.hidden_size), generator=generator,
                        device='cuda', dtype=torch.bfloat16)
        with set_forward_context(None, native, num_tokens=x.shape[0]):
            for _ in range(warmups):
                y = forward(x)
            torch.cuda.synchronize()
            outputs = y if isinstance(y, tuple) else (y,)
            assert all(torch.isfinite(value).all().item() for value in outputs if value is not None), \
                'Independent shape outputs must be finite'
            if case != 'dense':
                method = module.experts.routed_experts.quant_method
                backend = str(getattr(method, 'fp8_backend', getattr(method, 'unquantized_backend', None)))
                expected = next(s['backend'] for s in plan['scopes'] if s['id'] == scope)
                assert backend+'/'+method.moe_kernel.fused_experts.__class__.__name__ == expected
            else:
                backend = type(module.gate_up_proj.quant_method).__name__
                dispatch = linear_dispatch(module)
                assert all(row['dtype'] == 'torch.bfloat16' for row in dispatch.values())
            with (cell/'content.pt').open('xb') as stream:
                torch.save(dict(input=x.cpu(), outputs=[value.cpu() if value is not None else None
                                                       for value in outputs]), stream)
            launches = []
            with RecordFunctionTracer(str(cell)) as tracer:
                for _ in range(repeats):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    with torch.profiler.record_function('vidur_'+scope):
                        y = forward(x)
                    launches.append((time.perf_counter_ns()-start)/1e6)
            stats = tracer.get_operation_time_stats()[scope]
            assert stats['count'] == repeats
        row = dict(scope=scope, sample=sample, backend=backend,
                   dispatch=dispatch if case == 'dense' else dict(root='Step5DecoderLayer._forward_ffn',
                        branches='Step5MoE.forward', combine_dtype=str(y.dtype)), eager=stats,
                   host_profiled_root_launch_ms=launches, trace=tracer.trace_path,
                   content=str(cell/'content.pt'), charges=1,
                   graph_status='unmeasured', content_status='finite_only_FP64_comparison_pending',
                   gap_status='unmeasured_not_pure_device_gap', measurement_status='raw_eager_observation_not_admitted')
        rows.append(row)
        with (cell/'observation.json').open('x') as stream:
            json.dump(row, stream, indent=2, default=float); stream.write('\n')
    return rows
