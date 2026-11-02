"""DeepSeek-V4 linear-op producer: sequence-parallel shared-expert rows only."""

from __future__ import annotations

import json
import os
import sys

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op import main as linear_main
from frontier.profiling.linear_op.profiling_plan import tensor_parallel_sweep_exclusion

DSV4 = "deepseek-ai/DeepSeek-V4-Flash"
GLM = "zai-org/GLM-5.3"
SHARED_EXPERT_OPS = ["share_expert_up_proj", "share_expert_down_proj", "share_expert_act"]


@pytest.fixture(scope="module")
def dsv4_config() -> ModelConfig:
    return ModelConfig.from_model_name(DSV4)


def _run_dry_run(monkeypatch, *extra: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "linear_op.main", "--dry_run", "--models", DSV4,
            "--num_tensor_parallel_workers", "1", "8",
            "--max_tokens", "8192", "--num_tokens_list", "1", "256", "8192",
            *extra,
        ],
    )
    linear_main.main()


def test_only_deepseek_v4_is_excluded_from_tensor_parallel_sweep(dsv4_config):
    assert "DeepSeek-V4" in tensor_parallel_sweep_exclusion(dsv4_config)
    assert tensor_parallel_sweep_exclusion(ModelConfig.from_model_name(GLM)) is None


def test_dry_run_lists_only_the_sequence_parallel_plan(monkeypatch, capsys):
    _run_dry_run(monkeypatch, "--sequence_parallel_moe_topology", "8", "4", "32")
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert "Skipping the TP sweep: DeepSeek-V4" in captured.err

    model = summary["models"][DSV4]
    assert model["quantization"] == {"quant_method": "fp8", "weight_block_size": [128, 128]}
    assert model["profiled_steps"] == 3
    (plan,) = model["plans"]
    assert plan["tp_size"] == 1
    assert plan["enabled_ops"] == SHARED_EXPERT_OPS
    assert plan["padded_n_expanded_embd"] == 2048
    assert {
        (contract["tensor_parallel_mode"], contract["effective_ffn_width"])
        for contract in plan["operator_contracts"].values()
    } == {("replicated", 2048)}


def test_dry_run_without_sequence_parallel_topology_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="requires --sequence_parallel_moe_topology"):
        _run_dry_run(monkeypatch)


def _patch_vllm_activation_ops(monkeypatch, torch):
    calls = []

    def silu_and_mul_with_clamp(out, x, limit, alpha, beta):
        calls.append((tuple(out.shape), tuple(x.shape), limit, alpha, beta))

    monkeypatch.setattr(torch.ops._C, "silu_and_mul", lambda out, x: None, raising=False)
    monkeypatch.setattr(
        torch.ops._C, "silu_and_mul_with_clamp", silu_and_mul_with_clamp, raising=False
    )
    return calls


def test_shared_expert_uses_the_clamped_swiglu_of_vllm(monkeypatch, dsv4_config):
    """DeepSeek-V4's shared expert passes the same clamp arguments as vLLM's layer."""

    torch = pytest.importorskip("torch")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    pytest.importorskip("vllm")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import (
        SiluAndMulWithClamp as VllmSiluAndMulWithClamp,
    )

    from frontier.profiling.common.layers.activation import SiluAndMul, SiluAndMulWithClamp
    from frontier.profiling.linear_op import linear_op_impl

    calls = _patch_vllm_activation_ops(monkeypatch, torch)
    monkeypatch.setattr(linear_op_impl, "ColumnParallelLinear", lambda *a, **k: None)
    monkeypatch.setattr(linear_op_impl, "RowParallelLinear", lambda *a, **k: None)

    dsv4_act = linear_op_impl.ShareExpertMLP(dsv4_config, 1).act
    glm_act = linear_op_impl.ShareExpertMLP(ModelConfig.from_model_name(GLM), 1).act
    with set_current_vllm_config(VllmConfig()):
        reference = VllmSiluAndMulWithClamp(dsv4_config.swiglu_limit)

    assert isinstance(dsv4_act, SiluAndMulWithClamp)
    assert type(glm_act) is SiluAndMul
    output = dsv4_act(torch.empty(5, 2 * dsv4_config.share_expert_dim))
    assert tuple(output.shape) == (5, dsv4_config.share_expert_dim)
    assert calls == [(
        (5, dsv4_config.share_expert_dim),
        (5, 2 * dsv4_config.share_expert_dim),
        reference.swiglu_limit, reference.alpha, reference.beta,
    )]
