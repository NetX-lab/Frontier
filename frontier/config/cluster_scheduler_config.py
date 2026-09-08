"""Configuration families for cluster request-routing policies."""

from dataclasses import dataclass

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
    @staticmethod
    def get_type():
        return ClusterSchedulerType.VLLM_LOAD_BALANCING
