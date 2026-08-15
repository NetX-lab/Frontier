from __future__ import annotations

from dataclasses import fields

import pytest

from frontier.config.config import ReplicaConfig
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
