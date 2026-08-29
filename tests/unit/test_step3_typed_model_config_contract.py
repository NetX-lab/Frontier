"""Regression coverage for the real Step3 mixed-layer configuration contract."""

from __future__ import annotations

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.model_architectures import (
    ExpertParallelMode,
    LayerKind,
    TensorParallelMode,
)
from frontier.profiling.common.model_config import ModelConfig as ProfilingModelConfig


def _step3_moe_layers() -> str:
    return ",".join(str(layer_id) for layer_id in range(4, 60))


@pytest.mark.parametrize("loader", ["runtime", "profiling"])
def test_real_step3_loaders_expose_typed_widths_and_layer_domains(loader: str) -> None:
    config = (
        BaseModelConfig.create_from_name("step3-moe-noquant")
        if loader == "runtime"
        else ProfilingModelConfig.from_model_name("step3-moe-noquant")
    )

    assert config.num_layers == 61
    assert config.dense_mlp_hidden_dim == 18432
    assert config.routed_mlp_hidden_dim == 5120
    # Keep the historical field for existing routed-MoE consumers.
    assert config.mlp_hidden_dim == 5120
    assert config.get_moe_layer_ids() == list(range(4, 60))

    profile = config.get_model_architecture_profile()
    dense = profile.resolve_layer_contract(
        config,
        layer_id=0,
        operator_name="mlp_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    routed = profile.resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="moe_grouped_gemm",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )
    shared = profile.resolve_layer_contract(
        config,
        layer_id=4,
        operator_name="share_expert_up_proj",
        attention_tp_size=8,
        moe_tp_size=1,
        expert_parallel_size=8,
    )

    assert (dense.layer_kind, dense.width, dense.tensor_parallel_size) == (
        LayerKind.DENSE,
        18432,
        8,
    )
    assert dense.expert_parallel_mode is ExpertParallelMode.OFF
    assert (routed.layer_kind, routed.width, routed.tensor_parallel_size) == (
        LayerKind.ROUTED,
        5120,
        1,
    )
    assert routed.expert_parallel_mode is ExpertParallelMode.ON
    assert routed.expert_parallel_size == 8
    assert (shared.layer_kind, shared.width, shared.tensor_parallel_size) == (
        LayerKind.SHARED,
        5120,
        8,
    )
    assert shared.expert_parallel_mode is ExpertParallelMode.OFF


@pytest.mark.parametrize("loader", ["runtime", "profiling"])
@pytest.mark.parametrize("malformed", ["4,4,5", "-1,4", "4,61", "bad,4"])
def test_real_step3_loaders_reject_malformed_layer_maps(
    loader: str, malformed: str
) -> None:
    config = (
        BaseModelConfig.create_from_name("step3-moe-noquant")
        if loader == "runtime"
        else ProfilingModelConfig.from_model_name("step3-moe-noquant")
    )
    config.moe_layers_enum = malformed

    with pytest.raises(ValueError, match="moe_layers_enum"):
        config.get_moe_layer_ids()
