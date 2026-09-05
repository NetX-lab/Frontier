from __future__ import annotations

from dataclasses import fields
import sys

import pytest

from frontier.config.config import ReplicaConfig
from frontier.config.config import SimulationConfig
from frontier.config.flat_dataclass import create_flat_dataclass
from frontier.config.parallel_semantics import FrontierParallelismMapping


def test_replica_config_exposes_canonical_attention_dp_name_only() -> None:
    field_names = {field.name for field in fields(ReplicaConfig)}
    assert "attn_dp" in field_names
    assert "attn_data_parallel_size" not in field_names

    config = ReplicaConfig(
        model_name="Phi-tiny-MoE-instruct",
        attn_tensor_parallel_size=2,
        attn_dp=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=2,
        total_expert_num=4,
    )
    assert config.attn_dp == 1

    with pytest.raises(TypeError):
        ReplicaConfig(
            model_name="Phi-tiny-MoE-instruct",
            attn_data_parallel_size=1,
        )


def test_parallel_mapping_uses_attn_dp_name_only() -> None:
    mapping = FrontierParallelismMapping(
        cluster_num_replicas=2,
        attn_tensor_parallel_size=2,
        attn_dp=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=2,
    )
    assert mapping.attn_dp == 1
    assert "attn_data_parallel_size" not in mapping.to_dict()


def test_attn_dp_is_available_through_flattened_cli() -> None:
    flat_config = create_flat_dataclass(SimulationConfig)
    assert "replica_config_attn_dp" in flat_config.metadata_mapping

    original_argv = sys.argv
    try:
        sys.argv = ["frontier.main", "--replica_config_attn_dp", "2"]
        parsed = flat_config.create_from_cli_args()
    finally:
        sys.argv = original_argv

    assert parsed.replica_config_attn_dp == 2
