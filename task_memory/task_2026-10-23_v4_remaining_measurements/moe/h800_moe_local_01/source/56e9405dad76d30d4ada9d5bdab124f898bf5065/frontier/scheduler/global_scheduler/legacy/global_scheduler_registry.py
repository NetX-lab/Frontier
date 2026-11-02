from frontier.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from frontier.scheduler.global_scheduler.random_global_scheduler import (
    RandomGlobalScheduler,
)
from frontier.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)
from frontier.types import GlobalSchedulerType
from frontier.utils.base_registry import BaseRegistry


class GlobalSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:
        return GlobalSchedulerType.from_str(key_str)


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)
GlobalSchedulerRegistry.register(
    GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler
)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)
