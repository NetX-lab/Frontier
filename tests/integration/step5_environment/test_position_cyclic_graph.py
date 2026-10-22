"""Thin outer-setter/opaque-MoE/router PW/FULL proof, plus native eager modules.

Run from a neutral cwd in the derived image under the approved TP8/DP2 torchrun
allocation. --cases supplies two graph replays and explicit per-DP module positions
and masks; this script never submits jobs. CPU/static checks are not T2 execution.
"""
import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path


def origins(installed):
    import importlib
    names = ('vllm.models.step5.nvidia.model', 'vllm.forward_context',
             'vllm.model_executor.layers.fused_moe.router.router_factory',
             'vllm.model_executor.layers.fused_moe.router.routing_simulator_router')
    result = {}
    for name in names:
        module = importlib.import_module(name)
        path = installed/(name.removeprefix('vllm.').replace('.', '/')+'.py')
        assert Path(module.__file__).resolve() == path.resolve()
        assert Path(module.__spec__.origin).resolve() == path.resolve()
        result[name] = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def expected(torch, positions, mask, logits, top_k, normalize, scale):
    # Independent bounded test oracle, evaluated after capture/replay only.
    ids = torch.tensor([[(int(p)*top_k+s)%logits.shape[1] for s in range(top_k)]
                        for p in positions.cpu()], device=logits.device, dtype=torch.int64)
    weights = logits.float().gather(1, ids).sigmoid()
    if normalize:
        weights /= weights.sum(1, keepdim=True)
    weights *= scale
    return weights.masked_fill(mask[:, None], 0), ids.masked_fill(mask[:, None], -1).int()


def graph_binding(config, supplied, device):
    import torch
    import vllm.envs as envs
    from vllm.config import CUDAGraphMode, CompilationMode
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphWrapper, eager_break_during_capture)
    from vllm.forward_context import BatchDescriptor, ForwardContext, override_forward_context
    from vllm.models.step5.nvidia.model import Step5ForCausalLM
    from vllm.model_executor.layers.fused_moe.router.router_factory import create_fused_moe_router
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    assert envs.VLLM_USE_BREAKABLE_CUDAGRAPH
    assert config.compilation_config.mode == CompilationMode.NONE
    assert config.kernel_config.moe_backend == 'auto'
    assert not config.model_config.enforce_eager
    text = config.model_config.hf_text_config
    top_k, experts = text.moe_top_k, text.moe_num_experts
    assert text.moe_router_activation == 'sigmoid'
    assert len(supplied) == 2
    rows = len(supplied[0]['positions'])
    assert all(len(row['positions']) == len(row['mask']) == rows for row in supplied)
    positions = torch.zeros(rows, dtype=torch.int64, device=device)
    mask = torch.ones(rows, dtype=torch.bool, device=device)
    logits = torch.arange(rows*experts, device=device).reshape(rows, experts).float()%31-15
    hidden = torch.zeros(rows, top_k, device=device)
    callback_input = torch.zeros_like(hidden)
    callback_output, before_break = torch.zeros_like(hidden), torch.zeros_like(hidden)
    output, weights = torch.zeros_like(hidden), torch.zeros_like(hidden)
    ids = torch.empty(rows, top_k, device=device, dtype=torch.int32)
    counts = dict(outer=0, model=0, router=0, callback=0)
    router = create_fused_moe_router(top_k, experts, scoring_func=text.moe_router_activation,
        renormalize=text.norm_expert_weight, routed_scaling_factor=text.moe_router_scaling_factor)
    def capture_ids(value):
        counts['router'] += 1
        ids.copy_(value)
    router.set_capture_fn(capture_ids)

    @eager_break_during_capture
    def eager_callback(value, destination):
        counts['callback'] += 1
        destination.copy_(value)

    class RouteSink(MoERunner):
        """Thin opaque-op fixture only, never full expert/module evidence."""
        def __init__(self):
            torch.nn.Module.__init__(self)
        def _forward_impl(self, hidden_states, router_logits, shared_input, input_ids):
            selected, _ = router.select_experts(hidden_states, router_logits, torch.int32)
            weights.copy_(selected)
            output.copy_(weights + callback_output)
            return output
    context = ForwardContext({'t2.position_route': RouteSink()}, {}, {}, is_padding=mask)
    class TextSink(torch.nn.Module):
        def forward(self, input_ids, positions, intermediate, embeds):
            counts['model'] += 1
            before_break.copy_(hidden+1)
            eager_callback(callback_input, callback_output)
            return torch.ops.vllm.moe_forward(hidden, logits, None, None, 't2.position_route', 0)
    model = Step5ForCausalLM.__new__(Step5ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = TextSink()
    def outer(input_ids, position_arg):
        counts['outer'] += 1
        return model(input_ids, position_arg)
    buffers = [positions, mask, callback_input, callback_output, before_break, output, weights, ids]
    addresses = [value.data_ptr() for value in buffers]
    report = {}
    with torch.inference_mode(), override_forward_context(context):
        # Warm up ordinary native opaque-op routing on a side stream.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            outer(hidden, positions)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        assert context.routing_positions is None
        for mode in ('NONE', 'PIECEWISE', 'FULL'):
            positions.zero_(); mask.fill_(True); callback_input.zero_()
            context.cudagraph_runtime_mode = CUDAGraphMode.NONE
            if mode == 'PIECEWISE':
                descriptor = BatchDescriptor(rows)
                context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
                context.batch_descriptor = descriptor
                wrapper = BreakableCUDAGraphWrapper(outer, config, runtime_mode=CUDAGraphMode.PIECEWISE)
                wrapper(hidden, positions)
                capture = wrapper.entries[descriptor].capture
                assert capture.num_eager_breaks > 0 and capture.num_graphs >= 2
                replay = lambda: wrapper(hidden, positions)
                report[mode] = dict(num_eager_breaks=capture.num_eager_breaks, num_graphs=capture.num_graphs)
            elif mode == 'FULL':
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outer(hidden, positions)  # Native FULL capture also enters context mode NONE.
                replay = graph.replay
                report[mode] = dict(capture_context_mode='NONE')
            else:
                replay = lambda: outer(hidden, positions)
                report[mode] = {}
            # Capture records writes but does not execute them. Materialize dummy
            # results before inspecting buffers or taking replay counter baselines.
            replay()
            torch.cuda.synchronize(device)
            assert torch.equal(ids, torch.full_like(ids, -1)) and torch.equal(weights, torch.zeros_like(weights))
            assert context.routing_positions is None
            baseline = counts.copy()
            replay_counts = []
            for index, case in enumerate(supplied):
                positions.copy_(torch.tensor(case['positions'], device=device))
                mask.copy_(torch.tensor(case['mask'], device=device))
                callback_input.fill_(index+3)
                result = replay()
                torch.cuda.synchronize(device)
                wanted_w, wanted_i = expected(torch, positions, mask, logits, top_k,
                                            text.norm_expert_weight, text.moe_router_scaling_factor)
                assert torch.equal(ids, wanted_i)
                torch.testing.assert_close(weights, wanted_w, rtol=0, atol=0)
                torch.testing.assert_close(output, wanted_w+callback_input, rtol=0, atol=0)
                assert addresses == [value.data_ptr() for value in buffers]
                if mode != 'NONE':
                    assert all(counts[key] == baseline[key] for key in ('outer', 'model', 'router'))
                    expected_callbacks = baseline['callback']+(index+1 if mode == 'PIECEWISE' else 0)
                    assert counts['callback'] == expected_callbacks
                if mode != 'FULL':
                    assert result.data_ptr() == output.data_ptr()
                replay_counts.append(counts.copy())
            report[mode].update(addresses=addresses, replay_baseline_counts=baseline, replay_counts=replay_counts)
        context.cudagraph_runtime_mode = CUDAGraphMode.NONE
        from vllm.forward_context import set_routing_positions
        with set_routing_positions(positions):
            try:
                with set_routing_positions(positions[:1]):
                    raise LookupError('test-only restoration')
            except LookupError:
                assert context.routing_positions is positions
        assert context.routing_positions is None
    return report


def native_modules(args, config, plan, cases, device):
    import torch
    from vllm.distributed import parallel_state as ps
    from vllm.forward_context import set_forward_context, set_routing_positions
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
    from frontier.profiling.collectives.step5_dispatch_probe import (
        DispatchObserver, load_native_layer, native_backend, diagnostic_scope)
    results = []
    dp = config.parallel_config.data_parallel_rank
    for selected in plan['cases']:
        layer = load_native_layer(config, selected, device)
        for case in cases:
            local = case['dp'][dp]
            rows = len(local['positions'])
            assert rows == len(local['mask']) > 0
            positions = torch.tensor(local['positions'], dtype=torch.int64, device=device)
            mask = torch.tensor(local['mask'], dtype=torch.bool, device=device)
            sizes = torch.tensor([len(row['positions']) for row in case['dp']], dtype=torch.int32)
            inputs = torch.randn(rows, config.model_config.hf_text_config.hidden_size,
                                 device=device, dtype=config.model_config.dtype)
            report = dict(case=case['name'], layer=selected['layer_id'], enforce_eager=True,
                          route_checks=[], observer_errors=[])
            observer = DispatchObserver(lambda event: report.setdefault('events', []).append(event))
            with diagnostic_scope(observer.errors, 'native_backend'):
                report['backend'] = native_backend(layer)
                report['e8m0'] = is_deep_gemm_e8m0_used()
                report['moe_backend'] = config.kernel_config.moe_backend
            router = layer.experts.router
            original = router.select_experts
            captured = []
            def route(hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
                value = original(hidden_states=hidden_states, router_logits=router_logits,
                                 topk_indices_dtype=topk_indices_dtype, input_ids=input_ids)
                with diagnostic_scope(observer.errors, 'route_capture'):
                    captured.append((value[0].detach().clone(), value[1].detach().clone(), router_logits.detach().clone()))
                return value
            router.select_experts = route
            with ExitStack() as observation:
                with diagnostic_scope(observer.errors, 'collectives'):
                    observation.enter_context(observer.collectives(ps.GroupCoordinator))
                with diagnostic_scope(observer.errors, 'experts'):
                    observation.enter_context(observer.experts(layer.experts.routed_experts.quant_method.moe_kernel.fused_experts))
                with torch.inference_mode(), set_forward_context(None, config, num_tokens=rows,
                        num_tokens_across_dp=sizes, is_padding=mask), set_routing_positions(positions):
                    layer(inputs)  # Native execution errors propagate unchanged.
                torch.cuda.synchronize(device)
            router.select_experts = original
            with diagnostic_scope(observer.errors, 'route_values'):
                assert captured
                for weights, ids, logits in captured:
                    wanted_w, wanted_i = expected(torch, positions, mask, logits, router.top_k,
                                                router.renormalize, router.routed_scaling_factor)
                    assert torch.equal(ids, wanted_i)
                    torch.testing.assert_close(weights, wanted_w, rtol=0, atol=0)
                report['route_checks'] = ['IDs/weights/padding PASS']
                assert str(router.routing_method_type).endswith('Simulated')
                assert observer.expert_calls
            report.update(executed_experts_classes=observer.expert_calls, collectives=observer.calls,
                          observer_errors=observer.errors, routing_method=str(router.routing_method_type),
                          routing_strategy='position_cyclic', module_only=True, g3_qualified=False)
            results.append(report)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'cases', 'output', 'installed-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['VLLM_MOE_ROUTING_SIMULATION_STRATEGY'] == 'position_cyclic'
    args.tensor_parallel_size, args.data_parallel_size, args.gpus_per_node = 8, 2, 8
    args.disable_custom_all_reduce = False
    supplied = json.loads(args.cases.read_text())
    args.tokens_per_dp_rank = tuple(len(row['positions']) for row in supplied['modules'][0]['dp'])
    from frontier.profiling.collectives.step5_dispatch_probe import (
        create_native_config, probe_plan, validate_launcher, diagnostic_scope)
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
    from vllm.v1.worker.workspace import init_workspace_manager
    plan = probe_plan(args)
    validate_launcher(plan['topology'], os.environ)
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    report = dict(rank=rank, g3_qualified=False, mrv2_serving_qualified=False,
                   real_backend_graphs_qualified=False, diagnostic_errors=[],
                  torch=torch.__version__, python=__import__('sys').version)
    with diagnostic_scope(report['diagnostic_errors'], 'source_origins'):
        report['origins'] = origins(args.installed_root)
    config = create_native_config(args)
    with set_current_vllm_config(config):
        init_worker_distributed_environment(config, rank, 'env://', local)
        init_workspace_manager(device)
        report['native_modules'] = native_modules(args, config, plan, supplied['modules'], device)
    graph_config = EngineArgs(model=str(args.model_path), load_format='dummy',
        language_model_only=True, skip_tokenizer_init=True, tensor_parallel_size=8,
        data_parallel_size=2, enable_expert_parallel=True,
        distributed_executor_backend='external_launcher').create_engine_config()
    graph_config = graph_config.with_hf_config(graph_config.model_config.hf_text_config,
                                             architectures=['Step5ForCausalLM'])
    with set_current_vllm_config(graph_config):
        with diagnostic_scope(report['diagnostic_errors'], 'graph_binding'):
            report['graph_binding'] = graph_binding(graph_config, supplied['graph'], device)
    with diagnostic_scope(report['diagnostic_errors'], 'output'):
        report['status'] = ('UNVERIFIED' if report['diagnostic_errors'] or any(
            row['observer_errors'] for row in report['native_modules']) else 'PASS_THIN_GRAPH_AND_EAGER_MODULES')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/f'rank_{rank}.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
