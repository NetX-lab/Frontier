"""Construct replica-local scheduler maps for a cluster scheduler."""

from typing import Any

from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.types import ClusterType, ReplicaSchedulerType


def _create_scheduler(
    *,
    scheduler_type: ReplicaSchedulerType,
    replica_config: Any,
    scheduler_config: Any,
    request_generator_config: Any,
    replica: Any,
    predictor: Any,
    cluster_type: ClusterType,
    replica_local_id: int | None,
    af_pipeline_num_micro_batch: int,
    cluster_scheduler: Any,
    registry: Any,
) -> Any:
    """Create one child scheduler with the shared runtime dependencies."""

    return registry.get(
        scheduler_type,
        replica_config=replica_config,
        replica_scheduler_config=scheduler_config,
        request_generator_config=request_generator_config,
        replica=replica,
        predictor=predictor,
        cluster_type=cluster_type,
        replica_local_id=replica_local_id,
        af_pipeline_num_micro_batch=af_pipeline_num_micro_batch,
        cluster_scheduler=cluster_scheduler,
    )


def build_replica_scheduler_maps(
    *,
    cluster: Any,
    cluster_type: ClusterType,
    scheduler_type: ReplicaSchedulerType,
    replica_config: Any,
    scheduler_config: Any,
    request_generator_config: Any,
    predictor: Any,
    af_pipeline_num_micro_batch: int,
    cluster_scheduler: Any,
    dp_size: int | None,
    ep_size: int | None,
    registry: Any = ReplicaSchedulerRegistry,
) -> tuple[dict[tuple[int, int | None], Any], dict[int, Any]]:
    """Build lane and full-stage scheduler maps for one physical cluster.

    A DECODE_FFN replica owns one child per EP lane plus a separate full-stage
    child for dense work. A DECODE_ATTN replica owns one full-stage child. All
    other cluster types own one child per attention-DP lane and use lane zero
    as the compatibility full-stage lookup.
    """

    if cluster_type == ClusterType.DECODE_FFN:
        if type(ep_size) is not int or ep_size <= 0:
            raise ValueError("DECODE_FFN requires a positive EP size")
    elif cluster_type == ClusterType.DECODE_ATTN:
        if dp_size != 1:
            raise ValueError("DECODE_ATTN requires a DP size of one")
    elif type(dp_size) is not int or dp_size <= 0:
        raise ValueError(
            f"{cluster_type.name} requires a positive attention-DP size"
        )

    lane_schedulers: dict[tuple[int, int | None], Any] = {}
    full_stage_schedulers: dict[int, Any] = {}
    for replica_id, replica in cluster.replicas.items():
        if cluster_type == ClusterType.DECODE_FFN:
            for ep_id in range(ep_size):
                lane_schedulers[(replica_id, ep_id)] = _create_scheduler(
                    scheduler_type=scheduler_type,
                    replica_config=replica_config,
                    scheduler_config=scheduler_config,
                    request_generator_config=request_generator_config,
                    replica=replica,
                    predictor=predictor,
                    cluster_type=cluster_type,
                    replica_local_id=ep_id,
                    af_pipeline_num_micro_batch=af_pipeline_num_micro_batch,
                    cluster_scheduler=cluster_scheduler,
                    registry=registry,
                )
            full_stage_schedulers[replica_id] = _create_scheduler(
                scheduler_type=scheduler_type,
                replica_config=replica_config,
                scheduler_config=scheduler_config,
                request_generator_config=request_generator_config,
                replica=replica,
                predictor=predictor,
                cluster_type=cluster_type,
                replica_local_id=None,
                af_pipeline_num_micro_batch=af_pipeline_num_micro_batch,
                cluster_scheduler=cluster_scheduler,
                registry=registry,
            )
        elif cluster_type == ClusterType.DECODE_ATTN:
            full_stage_scheduler = _create_scheduler(
                scheduler_type=scheduler_type,
                replica_config=replica_config,
                scheduler_config=scheduler_config,
                request_generator_config=request_generator_config,
                replica=replica,
                predictor=predictor,
                cluster_type=cluster_type,
                replica_local_id=None,
                af_pipeline_num_micro_batch=af_pipeline_num_micro_batch,
                cluster_scheduler=cluster_scheduler,
                registry=registry,
            )
            full_stage_schedulers[replica_id] = full_stage_scheduler
            lane_schedulers[(replica_id, None)] = full_stage_scheduler
        else:
            for dp_id in range(dp_size):
                lane_schedulers[(replica_id, dp_id)] = _create_scheduler(
                    scheduler_type=scheduler_type,
                    replica_config=replica_config,
                    scheduler_config=scheduler_config,
                    request_generator_config=request_generator_config,
                    replica=replica,
                    predictor=predictor,
                    cluster_type=cluster_type,
                    replica_local_id=dp_id,
                    af_pipeline_num_micro_batch=af_pipeline_num_micro_batch,
                    cluster_scheduler=cluster_scheduler,
                    registry=registry,
                )
            full_stage_schedulers[replica_id] = lane_schedulers[(replica_id, 0)]

    return lane_schedulers, full_stage_schedulers
