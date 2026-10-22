"""CPU fault injection for explicit standalone gating construction."""

from unittest.mock import Mock
from types import ModuleType, SimpleNamespace
from contextlib import nullcontext
import sys

import pytest
import torch

from frontier.profiling.moe import moe_impl


@pytest.fixture
def gating_runtime(monkeypatch):
    monkeypatch.setattr(moe_impl, "HAS_VLLM", True)
    monkeypatch.setattr(moe_impl, "HAS_VLLM_REPLICATED_LINEAR", True)
    monkeypatch.setattr(moe_impl, "CudaTimer", lambda *args: None)
    monkeypatch.setattr(moe_impl, "raise_if_fp8_requested", lambda *args: None)
    monkeypatch.setattr(moe_impl, "ensure_vllm_single_rank_parallel_state", lambda: None)


@pytest.mark.parametrize("error", [AssertionError("bad weight"), RuntimeError("allocation failed")])
def test_native_constructor_errors_are_not_replaced_by_torch(gating_runtime, monkeypatch, error):
    constructor = Mock(side_effect=error)
    monkeypatch.setattr(moe_impl, "ReplicatedLinear", constructor)
    with pytest.raises(type(error), match=str(error)):
        moe_impl.MoEGatingNetwork(64, 2, 1)
    constructor.assert_called_once()


def test_disabled_native_routing_selects_torch_before_native_constructor(gating_runtime, monkeypatch):
    constructor = Mock(side_effect=AssertionError("must not construct native layer"))
    monkeypatch.setattr(moe_impl, "ReplicatedLinear", constructor)
    gate = moe_impl.MoEGatingNetwork(64, 2, 1, use_vllm_fused_topk=False)
    assert isinstance(gate.gate, torch.nn.Linear)
    assert gate.linear_backend == "torch_linear"
    constructor.assert_not_called()


def test_native_linear_receives_standalone_tp_disable(gating_runtime, monkeypatch):
    native_layer = torch.nn.Linear(64, 2, bias=False)
    constructor = Mock(return_value=native_layer)
    monkeypatch.setattr(moe_impl, "ReplicatedLinear", constructor)
    gate = moe_impl.MoEGatingNetwork(64, 2, 1)
    assert gate.gate is native_layer
    assert gate.linear_backend == "vllm_replicated_linear"
    constructor.assert_called_once_with(64, 2, bias=False, disable_tp=True)


def test_grouped_sigmoid_uses_native_gate_and_fp32_logits(gating_runtime, monkeypatch):
    class NativeGate(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.ones(hidden_states.shape[0], 8, dtype=torch.float32), None

    native = ModuleType("vllm.model_executor.layers.fused_moe")
    native.GateLinear = Mock(return_value=NativeGate())
    router = ModuleType("vllm.model_executor.layers.fused_moe.router.grouped_topk_router")
    router.grouped_topk = Mock(return_value=(torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.int32)))
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, router.__name__, router)
    monkeypatch.setattr(moe_impl, "CudaTimer", lambda *_: nullcontext())
    config = SimpleNamespace(n_group=2, topk_group=1, norm_topk_prob=True,
                             routed_scaling_factor=2.5, dtype=torch.bfloat16)
    gate = moe_impl.MoEGatingNetwork(64, 8, 2, routing_runtime_path="grouped_topk_sigmoid", router_config=config)
    gate(torch.ones(2, 64, dtype=torch.bfloat16))
    native.GateLinear.assert_called_once_with(64, 8, out_dtype=torch.float32, params_dtype=torch.bfloat16)
    kwargs = router.grouped_topk.call_args.kwargs
    assert kwargs["gating_output"].dtype is torch.float32
    assert kwargs["e_score_correction_bias"].dtype is torch.float32
    assert kwargs["scoring_func"] == "sigmoid"
    assert kwargs["num_expert_group"] == 2
    assert kwargs["topk_group"] == 1
    assert kwargs["renormalize"] is True
    assert kwargs["routed_scaling_factor"] == 2.5
    assert gate.linear_backend == "vllm_gate_linear"
