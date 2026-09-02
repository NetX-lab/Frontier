from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.model_architectures import (
    LayerKind,
    ModelArchitectureProfile,
)
from frontier.operators.binding import build_operator_manifest
from tests.e2e.operator_parity.op_family_coverage_oracle import (
    _manifest_model_config,
)


def _pure_moe_manifest_config(*, share_expert_dim: int) -> object:
    return _manifest_model_config(
        {
            "model_type": "phimoe",
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "hidden_size": 128,
            "num_experts": 16,
            "share_expert_dim": share_expert_dim,
            "use_mla": False,
        }
    )


def test_zero_shared_width_is_an_inactive_optional_contract() -> None:
    config = _pure_moe_manifest_config(share_expert_dim=0)
    profile = config.get_model_architecture_profile()

    assert profile.supports_share_expert(config) is False

    manifest = build_operator_manifest(config)
    assert {binding.family_id for binding in manifest.family_bindings} == {
        "dense_attention",
        "memory",
        "moe",
    }


def test_explicit_shared_contract_rejects_zero_width() -> None:
    config = SimpleNamespace(
        is_moe=True,
        num_layers=1,
        num_experts=16,
        share_expert_dim=0,
    )
    profile = ModelArchitectureProfile.generic()

    with pytest.raises(ValueError, match="share_expert_dim layer width.*positive int"):
        profile.resolve_layer_contract(config, layer_kind=LayerKind.SHARED)


def test_negative_shared_width_remains_malformed() -> None:
    config = _pure_moe_manifest_config(share_expert_dim=-1)
    profile = config.get_model_architecture_profile()

    with pytest.raises(ValueError, match="share_expert_dim layer width"):
        profile.supports_share_expert(config)
