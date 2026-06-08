from frontier.scheduler.replica_scheduler.faster_transformer_replica_scheduler import (
    FasterTransformerReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.lightllm_replica_scheduler import (
    LightLLMReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.orca_replica_scheduler import (
    OrcaReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_fastserve_lite_replica_scheduler import (
    SJ2QFastServeLiteReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_penalty_only_replica_scheduler import (
    SJ2QPenaltyOnlyReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sj2q_bounded_carryover_replica_scheduler import (
    SJ2QBoundedCarryoverReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.sglang_style_replica_scheduler import (
    SGLangStyleReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_replica_scheduler import (
    VLLMReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ReplicaSchedulerType
from frontier.utils.base_registry import BaseRegistry


class ReplicaSchedulerRegistry(BaseRegistry):
    pass


ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.FASTER_TRANSFORMER, FasterTransformerReplicaScheduler
)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.ORCA, OrcaReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.SARATHI, SarathiReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.VLLM, VLLMReplicaScheduler)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.LIGHTLLM, LightLLMReplicaScheduler
)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.VLLM_V1, VLLMv1EngineReplicaScheduler
)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE, SJ2QFastServeLiteReplicaScheduler
)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.SJ2Q_PENALTY_ONLY, SJ2QPenaltyOnlyReplicaScheduler
)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
    SJ2QBoundedCarryoverReplicaScheduler,
)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.SGLANG, SGLangStyleReplicaScheduler
)
