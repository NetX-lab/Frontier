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
