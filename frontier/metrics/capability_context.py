"""Metrics-local read-only architecture capability context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from frontier.model_architectures import (
    ModelArchitectureProfile,
    get_model_architecture_profile,
)
from frontier.types import ClusterType


@dataclass(frozen=True)
class CapabilityContext:
    """Derived profile/topology capabilities used by metrics emission."""

    cluster_type: ClusterType
    model_config: Any
    replica_config: Any
    architecture_profile: ModelArchitectureProfile
    expected_ep_size: int

    @classmethod
    def from_replica_config(
        cls,
        *,
        cluster_type: ClusterType,
        replica_config: Any,
    ) -> "CapabilityContext":
        """Build a read-only capability view from a replica config."""

        model_config = getattr(replica_config, "model_config", None)
        if model_config is None:
            raise ValueError("CapabilityContext requires replica_config.model_config")

        profile_getter = getattr(model_config, "get_model_architecture_profile", None)
        architecture_profile = (
            profile_getter()
            if callable(profile_getter)
            else get_model_architecture_profile(model_config)
        )
        if not isinstance(architecture_profile, ModelArchitectureProfile):
            raise TypeError(
                "model_config architecture profile must be ModelArchitectureProfile"
            )

        return cls(
            cluster_type=cluster_type,
            model_config=model_config,
            replica_config=replica_config,
            architecture_profile=architecture_profile,
            expected_ep_size=int(
                getattr(replica_config, "moe_expert_parallel_size", 1)
            ),
        )

    @property
    def skip_decode_ffn_attn_norm_residual(self) -> bool:
        """Return whether decode-FFN op traces skip attn/norm residual add."""

        return (
            self.cluster_type == ClusterType.DECODE_FFN
            and self.architecture_profile.skip_decode_ffn_attn_norm_residual
        )

    @property
    def skip_decode_attn_residual(self) -> bool:
        """Return whether decode-attention op traces skip attention residual add."""

        return (
            self.cluster_type == ClusterType.DECODE_ATTN
            and self.architecture_profile.skip_decode_attn_residual
        )

    @property
    def uses_profile_ep_alltoall(self) -> bool:
        """Return whether the profile declares EP alltoall for this cluster."""

        return self.architecture_profile.uses_expert_parallel_alltoall(
            self.cluster_type,
            self.expected_ep_size,
        )
