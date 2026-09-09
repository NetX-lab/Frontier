from __future__ import annotations

import torch

from frontier.profiling.common.layers.rotary_embedding import _call_vllm_get_rope


def test_vllm_legacy_rope_factory_contract() -> None:
    captured = {}

    def legacy_get_rope(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling,
        dtype,
    ):
        captured.update(locals())
        return "legacy"

    result = _call_vllm_get_rope(
        legacy_get_rope,
        head_size=128,
        rotary_dim=64,
        max_position=8192,
        base=1_000_000,
        is_neox_style=True,
        rope_scaling=None,
        dtype=torch.bfloat16,
    )

    assert result == "legacy"
    assert captured["rotary_dim"] == 64
    assert captured["base"] == 1_000_000


def test_vllm_current_rope_factory_contract() -> None:
    captured = {}

    def current_get_rope(
        head_size,
        max_position,
        is_neox_style=True,
        rope_parameters=None,
        dtype=None,
        dual_chunk_attention_config=None,
    ):
        captured.update(locals())
        return "current"

    result = _call_vllm_get_rope(
        current_get_rope,
        head_size=128,
        rotary_dim=64,
        max_position=8192,
        base=1_000_000,
        is_neox_style=False,
        rope_scaling={"rope_type": "linear", "factor": 2.0},
        dtype=torch.bfloat16,
    )

    assert result == "current"
    assert captured["rope_parameters"] == {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 1_000_000,
        "rope_dim": 64,
    }
