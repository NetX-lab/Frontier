"""Tests for the runtime-legal MoE expert-parallel sampling envelope."""

from __future__ import annotations

import sys

import pytest

from frontier.profiling.moe.main import parse_args
from frontier.profiling.moe.moe_input import (
    get_default_moe_profiling_config,
    get_runtime_legal_expert_parallel_sizes,
    resolve_moe_expert_parallel_sizes,
)


def test_runtime_legal_ep_sizes_include_every_positive_divisor() -> None:
    assert get_runtime_legal_expert_parallel_sizes(60) == [
        1,
        2,
        3,
        4,
        5,
        6,
        10,
        12,
        15,
        20,
        30,
        60,
    ]


def test_default_moe_profile_maps_all_ep_sizes_to_local_widths() -> None:
    config = get_default_moe_profiling_config(num_experts=60)

    assert config.num_experts_per_device_list == [
        60,
        30,
        20,
        15,
        12,
        10,
        6,
        5,
        4,
        3,
        2,
        1,
    ]


def test_explicit_ep_sizes_are_deduplicated_and_validated() -> None:
    assert resolve_moe_expert_parallel_sizes(60, [20, 2, 20, 1]) == [20, 2, 1]

    with pytest.raises(ValueError, match="divisor"):
        resolve_moe_expert_parallel_sizes(60, [7])

    with pytest.raises(ValueError, match="at least one"):
        resolve_moe_expert_parallel_sizes(60, [])


@pytest.mark.parametrize("num_experts", [0, -1])
def test_ep_domain_rejects_non_positive_expert_count(num_experts: int) -> None:
    with pytest.raises(ValueError, match="num_experts"):
        get_runtime_legal_expert_parallel_sizes(num_experts)


def test_moe_cli_uses_model_derived_ep_domain_when_unspecified(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["frontier.profiling.moe.main", "--device", "a100"])

    args, _ = parse_args()

    assert args.expert_parallel_sizes is None
