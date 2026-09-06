"""Replica scheduler map initialization for cluster schedulers."""

from typing import Any

from frontier.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from frontier.types import ClusterType, ReplicaSchedulerType
from frontier.scheduler.utils.replica_schedulers import build_replica_scheduler_maps


def initialize_replica_schedulers(
    scheduler: Any, request_generator_config: Any, logger
) -> None:
    """Validate cluster scheduler config and build child scheduler maps."""
    scheduler._replica_schedulers = {}
    scheduler._full_stage_replica_schedulers = {}
    cluster_specific_config = scheduler._get_cluster_specific_replica_scheduler_config(
        scheduler._config, scheduler._cluster_type
    )
    scheduler._replica_scheduler_type = cluster_specific_config.get_type()
    if type(scheduler._replica_scheduler_type) is not ReplicaSchedulerType:
        raise TypeError(
            "Cluster replica scheduler type must be an exact "
            f"ReplicaSchedulerType, got {scheduler._replica_scheduler_type!r}"
        )
    scheduler._validate_prefix_cache_cluster_config(cluster_specific_config)
    if (
        scheduler._cluster_type is ClusterType.DECODE_FFN
        and cluster_specific_config.get_type() is not ReplicaSchedulerType.ORCA
    ):
        raise ValueError(
            "DECODE_FFN cluster requires 'orca' scheduler, "
            f"got '{cluster_specific_config.get_type()}'. "
            "Reason: DECODE_FFN uses EP-based workload grouping which is only "
            "implemented in OrcaReplicaScheduler."
        )
    if scheduler._cluster_type is ClusterType.DECODE_FFN:
        scheduler._replica_ep_size = scheduler._config.replica_config.moe_expert_parallel_size
    scheduler._replica_schedulers, scheduler._full_stage_replica_schedulers = (
        build_replica_scheduler_maps(
            cluster=scheduler._cluster,
            cluster_type=scheduler._cluster_type,
            scheduler_type=cluster_specific_config.get_type(),
            replica_config=scheduler._config.replica_config,
            scheduler_config=cluster_specific_config,
            request_generator_config=request_generator_config,
            predictor=scheduler._predictor,
            af_pipeline_num_micro_batch=getattr(
                scheduler._config, "af_pipeline_num_micro_batch", -1
            ),
            cluster_scheduler=scheduler,
            dp_size=getattr(scheduler, "_replica_dp_size", None),
            ep_size=getattr(scheduler, "_replica_ep_size", None),
            registry=ReplicaSchedulerRegistry,
        )
    )
