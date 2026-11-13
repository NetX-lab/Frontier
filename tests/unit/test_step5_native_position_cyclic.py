"""Actual exported native methods on CPU; not CUDA or native expert qualification."""
import ast
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest
import torch


ROUTER = 'model_executor/layers/fused_moe/router/'


def nodes(root, path, names):
    return [node for node in ast.parse((root/path).read_text()).body
            if getattr(node, 'name', None) in names]


def execute(body, namespace):
    exec(compile(ast.Module(body=body, type_ignores=[]), '<actual-native-source>', 'exec'), namespace)
    return namespace


@pytest.fixture
def native():
    root = os.environ.get('STEP5_NATIVE_SOURCE_ROOT')
    if not root:
        pytest.skip('Set STEP5_NATIVE_SOURCE_ROOT to the derived vllm package')
    root = Path(root)
    env = NS(VLLM_MOE_ROUTING_SIMULATION_STRATEGY='position_cyclic', VLLM_MOE_SKIP_PADDING=True)
    namespace = dict(torch=torch, ABC=ABC, abstractmethod=abstractmethod, Any=Any, Callable=Callable,
                     contextmanager=contextmanager, dataclass=dataclass, field=field,
                     CUDAGraphMode=type('Mode', (), {'NONE': type('NoneMode', (), {
                         'is_valid_runtime_mode': lambda self: True})()}),
                     DPMetadata=Any, AttentionMetadata=Any, BatchDescriptor=Any, UBatchSlices=Any,
                     envs=env, EplbLayerState=Any, RoutingMethodType=NS(Simulated='Simulated'),
                     logger=NS(warning_once=lambda *args: None), _forward_context=None)
    execute(nodes(root, 'forward_context.py', ['ForwardContext', 'get_forward_context',
                                              'set_routing_positions']), namespace)
    context = namespace['ForwardContext']({}, {}, {})
    namespace['_forward_context'] = context
    # Only the superclass constructor is a dependency fixture; native selection
    # template, mapping/dtype conversion and complete strategy bodies are executed.
    class FusedMoERouter:
        def __init__(self, eplb_state=None, **kwargs):
            self.eplb_state, self.capture_fn = eplb_state, None
    namespace['FusedMoERouter'] = FusedMoERouter
    execute(nodes(root, ROUTER+'base_router.py', ['BaseRouter']), namespace)
    execute(nodes(root, ROUTER+'routing_simulator_router.py', [
        'RoutingStrategy', 'DistributionBasedRouting', 'PositionCyclicRouting',
        'RoutingSimulator', 'RoutingSimulatorRouter']), namespace)
    namespace.update(current_platform=NS(is_cpu=lambda: True), CPURouter=FusedMoERouter,
                     Callable=Any)
    execute(nodes(root, ROUTER+'router_factory.py', ['create_fused_moe_router']), namespace)
    return NS(root=root, ns=namespace, env=env, context=context,
              setter=namespace['set_routing_positions'], factory=namespace['create_fused_moe_router'])


def route(native, positions, mask=None, normalize=True, scale=2.75, logits=None):
    logits = logits if logits is not None else torch.arange(len(positions)*352).reshape(-1, 352).float()%31-15
    router = native.factory(8, 352, scoring_func='sigmoid', renormalize=normalize,
                            routed_scaling_factor=scale, e_score_correction_bias=torch.ones(352))
    native.context.is_padding = mask
    with native.setter(positions):
        return router._select_experts(torch.zeros(len(positions), 2), logits, torch.int32)


@pytest.mark.parametrize('normalize', [False, True])
def test_selected_unbiased_scores_mask_dtype_and_scale(native, normalize):
    positions = torch.tensor([0, 43, 44, 6128, 6144, 999999], dtype=torch.int32)
    mask = torch.tensor([False]*5+[True])
    logits = torch.arange(6*352).reshape(6, 352).float()%31-15
    weights, ids = route(native, positions, mask, normalize, logits=logits)
    expected_ids = torch.tensor([[(int(p)*8+s)%352 for s in range(8)] for p in positions])
    expected = logits.gather(1, expected_ids).sigmoid()
    if normalize:
        expected /= expected.sum(1, keepdim=True)
    expected *= 2.75
    expected[mask] = 0
    expected_ids[mask] = -1
    assert torch.equal(ids, expected_ids) and ids.dtype == torch.int32
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    assert weights.dtype == torch.float32 and (ids[:5] >= 0).sum() == 40
    native.env.VLLM_MOE_SKIP_PADDING = False
    assert (route(native, positions, mask)[1][-1] >= 0).all()


def test_pd_chunk_q1_and_repeated_position_invariance(native):
    positions = torch.arange(6144)
    full = route(native, positions)[1]
    for sizes in ([6128, 16], [2048, 2048, 2032, 16]):
        split = torch.cat([route(native, part)[1] for part in positions.split(sizes)])
        assert torch.equal(full, split)
    assert torch.equal(torch.bincount(full[:44].flatten().long()), torch.ones(352, dtype=torch.int64))
    assert route(native, torch.arange(6144, 6148))[1][:, 0].tolist() == [224, 232, 240, 248]
    repeated = route(native, torch.tensor([44, 44]))[1]
    assert torch.equal(repeated[0], repeated[1])


@pytest.mark.parametrize('positions,message', [
    (None, 'requires scoped'), (torch.zeros(2), 'signed integer'),
    (torch.zeros(2, 1, dtype=torch.int64), 'signed integer'),
    (torch.zeros(2, device='meta', dtype=torch.int64), 'same device'),
])
def test_invalid_positions_fail_without_fallback(native, positions, message):
    with pytest.raises(ValueError, match=message):
        route(native, positions, logits=torch.zeros(2, 352)) if positions is not None else native.ns[
            'PositionCyclicRouting']().route_tokens(torch.zeros(2, 1), torch.zeros(2, 352), 8)


def test_factory_scoring_and_old_seeded_distribution_controls(native):
    with pytest.raises(ValueError, match='sigmoid'):
        native.factory(8, 352)
    for mode in ('uniform_random', 'normal_routing'):
        native.env.VLLM_MOE_ROUTING_SIMULATION_STRATEGY = mode
        distribution = native.ns['RoutingSimulator']._routing_strategies[mode]
        torch.manual_seed(97)
        expected = distribution.route_tokens(torch.zeros(2, 1), torch.zeros(2, 352), 8, torch.int32)
        torch.manual_seed(97)
        actual = native.factory(8, 352)._select_experts(torch.zeros(2, 1), torch.zeros(2, 352), torch.int32)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    native.env.VLLM_MOE_ROUTING_SIMULATION_STRATEGY = ''
    assert type(native.factory(8, 352)).__name__ == 'FusedMoERouter'


def test_actual_outer_forward_identity_nested_exception_and_counter(native):
    model = nodes(native.root, 'models/step5/nvidia/model.py', ['Step5ForCausalLM'])[0]
    forward = next(node for node in model.body if getattr(node, 'name', None) == 'forward')
    namespace = native.ns | {'IntermediateTensors': Any}
    execute([forward], namespace)
    positions, nested = torch.arange(2), torch.arange(3)
    context, setter = native.context, native.setter
    context.moe_layer_index = 16
    def sink(*args):
        assert context is native.ns['_forward_context'] and context.routing_positions is positions
        with setter(nested):
            assert context.routing_positions is nested and context.moe_layer_index == 16
        assert context.routing_positions is positions
        raise LookupError('native failure')
    with pytest.raises(LookupError, match='native failure'):
        namespace['forward'](NS(model=sink), torch.ones(2), positions)
    assert context.routing_positions is None and context.moe_layer_index == 16
    native.env.VLLM_MOE_ROUTING_SIMULATION_STRATEGY = ''
    assert namespace['forward'](NS(model=lambda *args: args[1]), torch.ones(2), positions) is positions
    text = (native.root/'models/step5/nvidia/model.py').read_text()
    assert not model.decorator_list and 'support_torch_compile' not in ast.unparse(model)
    config = (native.root/'config/vllm.py').read_text()
    assert 'Step5ForCausalLM' in config and 'breakable' in config
    routing = ast.unparse(nodes(native.root, ROUTER+'routing_simulator_router.py', ['PositionCyclicRouting'])[0])
    assert not any(name in routing for name in ('.item(', '.cpu(', '.any(', 'all_reduce'))
    oracle = (native.root/'model_executor/layers/fused_moe/experts/triton_deep_gemm_moe.py').read_text()
    assert 'is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2)' in oracle


@pytest.mark.parametrize('rows,experts,top_k', [(2, 352, 0), (2, 352, 353)])
def test_invalid_top_k_and_logits_shape(native, rows, experts, top_k):
    strategy = native.ns['PositionCyclicRouting']()
    positions = torch.arange(rows)
    with pytest.raises(ValueError, match='1 <= top_k'):
        strategy.route_tokens(torch.zeros(rows, 1), torch.zeros(rows, experts), top_k, positions=positions)
    with pytest.raises(ValueError, match='shape'):
        strategy.route_tokens(torch.zeros(rows, 1), torch.zeros(rows), 1, positions=positions)


def test_t2_capture_accepts_actual_moe_keyword_invocation(native):
    from tests.integration.step5_environment.test_position_cyclic_graph import native_modules
    tree = ast.parse(Path(__import__('inspect').getsourcefile(native_modules)).read_text())
    wrapper = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'route')
    captured, errors, calls = [], [], []
    result = torch.ones(2, 8), torch.zeros(2, 8, dtype=torch.int32)
    failure = []
    @contextmanager
    def diagnostic_scope(errors, label):
        try: yield
        except Exception as error: errors.append(error)
    def original(hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        calls.append((hidden_states, router_logits, input_ids))
        if failure: raise failure[0]
        return result
    namespace = dict(original=original, captured=captured, observer=NS(errors=errors), diagnostic_scope=diagnostic_scope)
    execute([wrapper], namespace)
    runner = nodes(native.root, 'model_executor/layers/fused_moe/runner/moe_runner.py', ['MoERunner'])[0]
    method = next(node for node in runner.body if getattr(node, 'name', None) == '_apply_quant_method')
    namespace.update(torch=torch, SharedExpertsOrder=NS(NO_OVERLAP=None), UnfinalizedMoEOutput=Any)
    execute([method], namespace)
    inputs, logits = torch.zeros(2, 1), torch.zeros(2, 352)
    value = namespace['_apply_quant_method'](NS(_maybe_apply_shared_experts=lambda *a: None,
        routed_experts=NS(quant_method=NS(is_monolithic=False), forward_modular=lambda **kw: kw['x']),
        _quant_method=NS(topk_indices_dtype=torch.int32), _shared_experts=None,
        router=NS(select_experts=namespace['route'])), inputs, logits, None)
    assert value[1] is inputs and len(calls) == len(captured) == 1 and not errors
    assert torch.equal(captured[0][2], logits)
    assert namespace['route'](inputs, logits) is result and len(calls) == 2
    failure.append(LookupError('native routing failure'))
    with pytest.raises(LookupError) as caught: namespace['route'](hidden_states=inputs, router_logits=logits)
    assert caught.value is failure[0] and len(calls) == 3 and len(captured) == 2
    failure.clear(); namespace['captured'] = NS(append=lambda value: (_ for _ in ()).throw(OSError('capture failure')))
    assert namespace['route'](hidden_states=inputs, router_logits=logits) is result
    assert len(calls) == 4 and len(errors) == 1


def test_t2_replays_dummy_before_assertion_and_baseline(native):
    path = Path(__file__).parents[1]/'integration/step5_environment/test_position_cyclic_graph.py'
    tree = ast.parse(path.read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'mode')
    dispatch = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.If))
    remainder = loop.body[dispatch+1:]
    counts = dict(outer=1, model=1, router=1, callback=1)
    wrapper = nodes(native.root, 'compilation/breakable_cudagraph.py', ['BreakableCUDAGraphWrapper'])[0]
    methods = [node for node in wrapper.body if getattr(node, 'name', None) in ('__call__', '_replay')]
    context = NS(batch_descriptor=2, cudagraph_runtime_mode='PIECEWISE')
    namespace = execute(methods, dict(Any=Any, Hashable=Any, CUDAGraphMode=NS(NONE='NONE'),
        is_forward_context_available=lambda: True, get_forward_context=lambda: context,
        _BreakableEntry=lambda **kw: NS(capture=None, **kw),
        get_offloader=lambda: NS(sync_prev_onload=lambda: None)))
    for mode in ('PIECEWISE', 'FULL'):
        ids, weights = torch.ones(2, 8, dtype=torch.int32), torch.ones(2, 8)
        calls = []
        def writes():
            calls.append(mode); ids.fill_(-1); weights.zero_()
            if mode == 'PIECEWISE': counts['callback'] += 1
        if mode == 'PIECEWISE':
            def capture(entry, args, kwargs):
                entry.capture, entry.output = NS(replay=writes), weights
                return weights
            state = NS(runtime_mode='PIECEWISE', key_fn=None, entries={}, is_debugging_mode=False,
                       _capture=capture, _replay=lambda *args: namespace['_replay'](state, *args))
            replay = lambda: namespace['__call__'](state)
            replay()  # Actual pinned dispatch enters capture, not _replay.
        else:
            replay = writes  # FULL graph capture requires CUDA; only post-capture sequencing is tested.
        assert not calls and (ids == 1).all() and (weights == 1).all()
        # Actual post-capture statements, stopping before the real-case loop.
        body = remainder[:next(i for i, node in enumerate(remainder) if isinstance(node, ast.For))]
        execution = dict(torch=NS(cuda=NS(synchronize=lambda *a: None), equal=torch.equal,
            full_like=torch.full_like, zeros_like=torch.zeros_like), device=None, replay=replay,
            ids=ids, weights=weights, context=NS(routing_positions=None), counts=counts, mode=mode)
        execute(body, execution)
        assert calls == [mode] and execution['baseline'] == counts


@pytest.mark.parametrize('drift', [None, 'modules', 'graph'])
def test_t2_entrypoint_supplies_and_observes_final_engine_limits(tmp_path, monkeypatch, drift):
    import json
    import sys
    from tests.integration.step5_environment import test_position_cyclic_graph as t2
    limits = dict(max_model_len=7168, max_num_batched_tokens=8192, max_num_seqs=128)
    cases = tmp_path/'cases.json'
    cases.write_text(json.dumps(dict(engine_limits=limits, modules=[dict(dp=[dict(positions=[0])]*2)], graph=[])))
    calls, failures = [], []
    class Config:
        def __init__(self, kwargs, final=False):
            self.kwargs, self.final = kwargs, final
            self.model_config = NS(hf_text_config=NS(), max_model_len=kwargs['max_model_len'])
            self.scheduler_config = NS(**{key: kwargs[key] for key in limits if key != 'max_model_len'})
        def with_hf_config(self, *args, **kwargs):
            value = Config(self.kwargs, final=True)
            if ('modules' if self.kwargs.get('enforce_eager') else 'graph') == drift:
                value.scheduler_config.max_num_seqs += 1
            return value
    class EngineArgs:
        def __init__(self, **kwargs):
            calls.append(kwargs); self.kwargs = kwargs
        def create_engine_config(self):
            if failures: raise failures[0]
            return Config(self.kwargs)
    @contextmanager
    def current(config):
        assert config.final
        yield
    from frontier.profiling.collectives import step5_dispatch_probe as probe
    monkeypatch.setitem(sys.modules, 'vllm.config', NS(set_current_vllm_config=current))
    monkeypatch.setitem(sys.modules, 'vllm.engine.arg_utils', NS(EngineArgs=EngineArgs))
    monkeypatch.setitem(sys.modules, 'vllm.models.step5.nvidia.model', NS(Step5ForCausalLM=object))
    monkeypatch.setitem(sys.modules, 'vllm.model_executor.model_loader.utils', NS(configure_quant_config=lambda *a: None))
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.gpu_worker', NS(init_worker_distributed_environment=lambda *a: None))
    monkeypatch.setitem(sys.modules, 'vllm.v1.worker.workspace', NS(init_workspace_manager=lambda *a: None))
    monkeypatch.setitem(sys.modules, 'torch', NS(__version__='host-fixture', device=lambda *a: None,
                                              cuda=NS(set_device=lambda *a: None)))
    monkeypatch.setattr(Config, 'quant_config', None, raising=False)
    def plan(args):
        assert args.routing_input_file is None
        return dict(topology={})
    monkeypatch.setattr(probe, 'probe_plan', plan)
    monkeypatch.setattr(probe, 'validate_launcher', lambda *a: None)
    monkeypatch.setattr(t2, 'origins', lambda *a: {})
    monkeypatch.setattr(t2, 'native_modules', lambda *a: [])
    monkeypatch.setattr(t2, 'graph_binding', lambda *a: {})
    monkeypatch.setenv('VLLM_MOE_ROUTING_SIMULATION_STRATEGY', 'position_cyclic')
    monkeypatch.setenv('RANK', '0'); monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setattr(sys, 'argv', ['t2', '--model-path', str(tmp_path), '--cases', str(cases),
                                   '--output', str(tmp_path/'output'), '--installed-root', str(tmp_path)])
    t2.main()
    report = json.loads((tmp_path/'output/rank_0.json').read_text())
    assert len(calls) == 2 and all({key: row[key] for key in limits} == limits for row in calls)
    assert calls[0]['enforce_eager'] is True and 'enforce_eager' not in calls[1]
    for role in ('modules', 'graph'):
        assert report['engine_limits'][role] == limits | {'max_num_seqs': 129 if role == drift else 128}
    assert bool(report['diagnostic_errors']) == bool(drift)
    assert report['status'] == ('UNVERIFIED' if drift else 'PASS_THIN_GRAPH_AND_EAGER_MODULES')
    cases.write_text(json.dumps(dict(modules=[], graph=[])))
    with pytest.raises(KeyError, match='engine_limits'): t2.main()
    assert len(calls) == 2
    cases.write_text(json.dumps(dict(engine_limits=limits, modules=[dict(dp=[dict(positions=[0])]*2)], graph=[])))
    failures.append(LookupError('native constructor failure'))
    with pytest.raises(LookupError) as caught: t2.main()
    assert caught.value is failures[0] and len(calls) == 3
