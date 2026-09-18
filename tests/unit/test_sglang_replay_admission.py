"""CPU regressions for serialized replay and packed routed-expert admission."""

from types import SimpleNamespace

import pytest

from frontier.profiling.experimental.sglang.graph_replay import (
    build_replay_plan, local_rank_for_visibility, validate_replay_plan,
)
from frontier.profiling.experimental.sglang.moe import (
    moe_routed_spec, validate_moe_routed_spec,
)


@pytest.mark.parametrize("field,value", [
    ("primitive", "unknown"), ("logical_sizes", [5]),
    ("logical_sizes", []), ("context_lengths", [0]),
    ("context_lengths", []),
])
def test_serialized_plan_revalidates_constructor_contract(field, value):
    plan = build_replay_plan(primitive="gemma_norm", sizes=[4], invocations=[2],
                             repetitions=5, split="validation", environment={})
    plan[field] = value
    with pytest.raises(ValueError):
        validate_replay_plan(plan)


def test_visible_devices_do_not_extend_world_size():
    with pytest.raises(ValueError, match="world_size"):
        local_rank_for_visibility(rank=1, world_size=1,
                                  environment={"HIP_VISIBLE_DEVICES": "4,5"})


def test_serialized_valid_plan_round_trip():
    plan = build_replay_plan(primitive="gemma_norm", sizes=[4], invocations=[2],
                             repetitions=5, split="validation", logical_sizes=[3],
                             context_lengths=[16], environment={"HIP_VISIBLE_DEVICES": "4,5"},
                             rank=1, world_size=2)
    validate_replay_plan(plan)


@pytest.mark.parametrize("hidden", [63, 65, 80])
def test_routed_builder_rejects_truncated_packed_dimensions(hidden):
    model = SimpleNamespace(is_moe=True, routed_mlp_hidden_dim=256,
                            embedding_dim=hidden, num_experts=2, num_experts_per_tok=1)
    with pytest.raises(ValueError):
        moe_routed_spec(model, 1)


@pytest.mark.parametrize("hidden,local", [(65, 256), (64, 96)])
def test_serialized_routed_spec_rechecks_native_alignment(hidden, local):
    model = SimpleNamespace(is_moe=True, routed_mlp_hidden_dim=256,
                            embedding_dim=64, num_experts=2, num_experts_per_tok=1)
    spec = moe_routed_spec(model, 1)
    spec.update(hidden_size=hidden, local_intermediate_size=local,
                global_intermediate_size=local,
                w13_packed_shape=(2, 2 * local, hidden // 2),
                w2_packed_shape=(2, hidden, local // 2),
                w13_scale_shape=(2, 2 * local, hidden // 32),
                w2_scale_shape=(2, hidden, local // 32))
    with pytest.raises(ValueError):
        validate_moe_routed_spec(spec, 1)
