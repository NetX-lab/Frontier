from frontier.scheduler.cluster_scheduler.lor_cluster_scheduler import LORClusterScheduler
from frontier.scheduler.cluster_scheduler.random_cluster_scheduler import (
    RandomClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.sticky_lor_cluster_scheduler import (
    StickyLORClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.sticky_round_robin_cluster_scheduler import (
    StickyRoundRobinClusterScheduler,
)
from frontier.types.cluster_scheduler_type import ClusterSchedulerType
from frontier.utils.base_registry import BaseRegistry


class ClusterSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ClusterSchedulerType:
        return ClusterSchedulerType.from_str(key_str)


ClusterSchedulerRegistry.register(ClusterSchedulerType.ROUND_ROBIN, RoundRobinClusterScheduler)
ClusterSchedulerRegistry.register(ClusterSchedulerType.RANDOM, RandomClusterScheduler)
ClusterSchedulerRegistry.register(ClusterSchedulerType.LOR, LORClusterScheduler)
ClusterSchedulerRegistry.register(
    ClusterSchedulerType.STICKY_ROUND_ROBIN,
    StickyRoundRobinClusterScheduler,
)
ClusterSchedulerRegistry.register(ClusterSchedulerType.STICKY_LOR, StickyLORClusterScheduler)
