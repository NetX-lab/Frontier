import pytest
import torch

from frontier.profiling.moe.moe_wrapper import MoEWrapper
from frontier.profiling.moe.load_distribution import generate_expert_routing


def _cpu_wrapper(*, expert_parallel_size: int = 1) -> MoEWrapper:
    wrapper = MoEWrapper.__new__(MoEWrapper)
    wrapper.expert_parallel_size = expert_parallel_size
    wrapper.num_experts = 8
    wrapper.num_experts_per_device = 8 // expert_parallel_size
    wrapper.router_topk = 2
    return wrapper


def test_prepare_routing_inputs_rejects_counts_inconsistent_with_generated_route() -> None:
    wrapper = _cpu_wrapper(expert_parallel_size=2)

    with pytest.raises(ValueError, match="expert_token_counts.*match"):
        wrapper._prepare_routing_inputs(
            num_tokens=17,
            load_distribution="skewed",
            seed=3,
            expert_token_counts=[34, 0, 0, 0],
        )


def test_prepare_routing_inputs_accepts_matching_local_counts() -> None:
    wrapper = _cpu_wrapper(expert_parallel_size=2)
    generated = wrapper._prepare_routing_inputs(
        num_tokens=17,
        load_distribution="skewed",
        seed=3,
    )

    reused = wrapper._prepare_routing_inputs(
        num_tokens=17,
        load_distribution="skewed",
        seed=3,
        expert_token_counts=generated["expert_token_counts"],
    )

    assert reused["expert_token_counts"] == generated["expert_token_counts"]


def test_prepare_routing_inputs_rejects_integral_float_counts() -> None:
    wrapper = _cpu_wrapper(expert_parallel_size=1)
    generated = wrapper._prepare_routing_inputs(
        num_tokens=17,
        load_distribution="uniform",
        seed=3,
    )
    supplied = list(generated["expert_token_counts"])
    supplied[0] = float(supplied[0])

    with pytest.raises(ValueError, match="non-negative integers"):
        wrapper._prepare_routing_inputs(
            num_tokens=17,
            load_distribution="uniform",
            seed=3,
            expert_token_counts=supplied,
        )


def test_extremely_skewed_preserves_a_hot_subset_when_topk_is_wide() -> None:
    _, topk_ids = generate_expert_routing(
        num_tokens=512,
        num_experts=16,
        top_k=8,
        load_distribution="extremely_skewed",
        seed=42,
    )

    hot_rows = (topk_ids < 8).all(dim=1)

    assert int(hot_rows.sum()) >= 350


def test_skewed_distribution_keeps_expert_zero_reachable() -> None:
    _, topk_ids = generate_expert_routing(
        num_tokens=512,
        num_experts=8,
        top_k=2,
        load_distribution="skewed",
        seed=42,
    )

    assert bool((topk_ids == 0).any())


@pytest.mark.parametrize("num_experts, top_k", [(16, 8), (128, 8), (8, 2)])
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 33, 20448])
def test_balanced_rows_carry_the_loads_of_the_simulator_balanced_routing(
    num_experts: int, top_k: int, num_tokens: int
) -> None:
    from frontier.moe_ep_workload import (
        generate_moe_routing_ratios,
        materialize_expert_token_counts,
    )
    from frontier.profiling.moe.load_distribution import compute_expert_token_counts

    _, topk_ids = generate_expert_routing(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        load_distribution="balanced",
        seed=0,
    )
    simulator_counts = materialize_expert_token_counts(
        routing_ratios=generate_moe_routing_ratios(
            total_expert_num=num_experts, distribution_type="balanced", seed=0, layer_id=0,
        ),
        total_routed_assignments=num_tokens * top_k,
        total_expert_num=num_experts,
    )

    assert compute_expert_token_counts(topk_ids, num_experts) == [
        simulator_counts[expert_id] for expert_id in range(num_experts)
    ]


def test_balanced_rows_at_ep2_profile_the_first_lane_of_the_simulator() -> None:
    wrapper = _cpu_wrapper(expert_parallel_size=2)
    wrapper.num_experts, wrapper.num_experts_per_device, wrapper.router_topk = 16, 8, 8

    counts = wrapper._prepare_routing_inputs(
        num_tokens=3, load_distribution="balanced", seed=0,
    )["expert_token_counts"]

    assert counts == [2] * 8
