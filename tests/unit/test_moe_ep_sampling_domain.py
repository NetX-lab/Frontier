"""Tests for the runtime-legal MoE expert-parallel sampling envelope."""

from __future__ import annotations

import sys

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.moe import main as moe_main
from frontier.profiling.moe.main import parse_args
from frontier.profiling.moe.moe_input import (
    get_default_moe_profiling_config,
    get_runtime_legal_expert_parallel_sizes,
    resolve_moe_expert_parallel_sizes,
)
from frontier.profiling.utils import get_num_tokens_to_profile


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


def test_moe_confirmation_reports_all_model_domains_and_aggregate_total(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["frontier.profiling.moe.main", "--device", "a100"])
    args, _ = parse_args()
    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)
    model_configs = {
        model: ModelConfig.from_model_name(model) for model in args.models
    }
    resolved_ep_sizes_by_model = {
        model: resolve_moe_expert_parallel_sizes(
            model_config.num_experts,
            args.expert_parallel_sizes,
        )
        for model, model_config in model_configs.items()
    }

    sections = moe_main._build_moe_confirmation_sections(
        args=args,
        model_configs=model_configs,
        resolved_ep_sizes_by_model=resolved_ep_sizes_by_model,
        num_tokens_count=len(num_tokens_to_profile),
        use_vllm_kernel=True,
    )

    section_map = {section_name: dict(rows) for section_name, rows in sections}
    model_domains = section_map["Resolved Model Profiling Domains"]
    assert "[1, 2, 4, 8]" in model_domains["mixtral_8x7b_moe"]
    assert "12,432 configurations" in model_domains["mixtral_8x7b_moe"]
    assert "[1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]" in model_domains[
        "qwen2_moe_example"
    ]
    assert "37,296 configurations" in model_domains["qwen2_moe_example"]
    assert "49,728" in section_map["Profiling Matrix"]["Total Configurations"]


def test_moe_configuration_count_includes_load_distribution_samples() -> None:
    cases_per_parallel_point = 3 * 2

    assert moe_main._count_moe_configurations(
        num_tokens_count=3,
        tensor_parallel_count=3,
        expert_parallel_count=4,
        cases_per_parallel_point=cases_per_parallel_point,
    ) == 216
    assert moe_main._count_moe_configurations(
        num_tokens_count=3,
        tensor_parallel_count=1,
        expert_parallel_count=4,
        cases_per_parallel_point=cases_per_parallel_point,
    ) == 72
