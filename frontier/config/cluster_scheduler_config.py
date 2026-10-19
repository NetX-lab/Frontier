"""Cluster scheduler configuration: how a cluster picks a replica."""

from __future__ import annotations

from dataclasses import dataclass, field

from frontier.config.base_poly_config import BasePolyConfig
from frontier.types import ClusterSchedulerType


@dataclass
class BaseClusterSchedulerConfig(BasePolyConfig):
    pass


@dataclass
class RandomClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.RANDOM


@dataclass
class RoundRobinClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.ROUND_ROBIN


@dataclass
class LORClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.LOR


@dataclass
class StickyRoundRobinClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.STICKY_ROUND_ROBIN


@dataclass
class StickyLORClusterSchedulerConfig(BaseClusterSchedulerConfig):
    @staticmethod
    def get_type():
        return ClusterSchedulerType.STICKY_LOR


@dataclass
class VllmLoadBalancingClusterSchedulerConfig(BaseClusterSchedulerConfig):
    """Select the reference version of vLLM's internal DP placement."""

    policy_version: str = field(
        default="v0.10.2",
        metadata={"help": "Internal DP policy reference: v0.10.2 or v0.28.0."},
    )

    def __post_init__(self) -> None:
        if self.policy_version not in ("v0.10.2", "v0.28.0"):
            raise ValueError(
                "vllm_load_balancing policy_version must be v0.10.2 or v0.28.0, "
                f"got {self.policy_version!r}"
            )

    @staticmethod
    def get_type():
        return ClusterSchedulerType.VLLM_LOAD_BALANCING
