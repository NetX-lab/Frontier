"""CPU fault injection for explicit standalone gating construction."""

from unittest.mock import Mock

import pytest
import torch

from frontier.profiling.moe import moe_impl


@pytest.fixture
def gating_runtime(monkeypatch):
    monkeypatch.setattr(moe_impl, "HAS_VLLM", True)
    monkeypatch.setattr(moe_impl, "HAS_VLLM_REPLICATED_LINEAR", True)
    monkeypatch.setattr(moe_impl, "CudaTimer", lambda *args: None)
    monkeypatch.setattr(moe_impl, "raise_if_fp8_requested", lambda *args: None)


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
