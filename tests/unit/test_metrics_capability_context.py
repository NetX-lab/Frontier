from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from frontier.model_architectures import ModelArchitectureProfile
from frontier.types import ClusterType


class _ProfileOnlyStep3ModelConfig:
    def get_model_architecture_profile(self) -> ModelArchitectureProfile:
        return ModelArchitectureProfile.step3_text()


def test_capability_context_derives_step3_decode_skip_and_ep_alltoall_from_profile() -> None:
    from frontier.metrics.capability_context import CapabilityContext

    context = CapabilityContext.from_replica_config(
        cluster_type=ClusterType.DECODE_FFN,
        replica_config=SimpleNamespace(
            model_config=_ProfileOnlyStep3ModelConfig(),
            moe_expert_parallel_size=2,
        ),
    )

    assert context.skip_decode_ffn_attn_norm_residual is True
    assert context.skip_decode_attn_residual is False
    assert context.uses_profile_ep_alltoall is True


def test_capability_context_decode_attn_skip_is_cluster_specific() -> None:
    from frontier.metrics.capability_context import CapabilityContext

    context = CapabilityContext.from_replica_config(
        cluster_type=ClusterType.DECODE_ATTN,
        replica_config=SimpleNamespace(
            model_config=_ProfileOnlyStep3ModelConfig(),
            moe_expert_parallel_size=2,
        ),
    )

    assert context.skip_decode_ffn_attn_norm_residual is False
    assert context.skip_decode_attn_residual is True


def test_capability_context_requires_model_config_without_generic_fallback() -> None:
    from frontier.metrics.capability_context import CapabilityContext

    try:
        CapabilityContext.from_replica_config(
            cluster_type=ClusterType.DECODE_FFN,
            replica_config=SimpleNamespace(
                model_config=None,
                moe_expert_parallel_size=2,
            ),
        )
    except ValueError as exc:
        assert "model_config" in str(exc)
    else:
        raise AssertionError("CapabilityContext must not fall back to generic")


def test_metrics_store_consumes_capability_context_instead_of_resolving_profiles_directly() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "frontier/metrics/metrics_store.py").read_text(
        encoding="utf-8"
    )

    assert "get_model_architecture_profile(" not in source
