"""Resolve cluster-local replica scheduler configuration."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

from frontier.config import BaseReplicaSchedulerConfig
from frontier.types import ClusterType, ReplicaSchedulerType


_CLUSTER_PREFIX = {
    ClusterType.PREFILL: "prefill",
    ClusterType.DECODE: "decode",
    ClusterType.DECODE_ATTN: "decode_attn",
    ClusterType.DECODE_FFN: "decode_ffn",
}

_SCHEDULER_TYPES = {
    "vllm": ReplicaSchedulerType.VLLM,
    "vllm_v1": ReplicaSchedulerType.VLLM_V1,
    "sj2q_fastserve_lite": ReplicaSchedulerType.SJ2Q_FASTSERVE_LITE,
    "sj2q_penalty_only": ReplicaSchedulerType.SJ2Q_PENALTY_ONLY,
    "sj2q_bounded_carryover": ReplicaSchedulerType.SJ2Q_BOUNDED_CARRYOVER,
    "sglang": ReplicaSchedulerType.SGLANG,
    "orca": ReplicaSchedulerType.ORCA,
    "sarathi": ReplicaSchedulerType.SARATHI,
    "lightllm": ReplicaSchedulerType.LIGHTLLM,
    "faster_transformer": ReplicaSchedulerType.FASTER_TRANSFORMER,
}

_OVERRIDE_FIELDS = (
    "batch_size_cap",
    "max_tokens_in_batch",
    "enable_chunked_prefill",
    "long_prefill_token_threshold",
    "num_blocks",
    "block_size",
    "watermark_blocks_fraction",
)


def resolve_replica_scheduler_config(config: Any, cluster_type: ClusterType) -> Any:
    """Return a copied scheduler config with cluster-local overrides applied."""

    base_config = config.replica_scheduler_config
    prefix = _CLUSTER_PREFIX.get(cluster_type)
    if prefix is None:
        return copy.deepcopy(base_config)

    type_field = f"{prefix}_replica_scheduler_config_type"
    override_type_name = (
        getattr(config, type_field, None)
        if hasattr(config, type_field)
        else None
    )
    if override_type_name is None:
        cluster_config = copy.deepcopy(base_config)
    else:
        try:
            override_type = _SCHEDULER_TYPES[override_type_name.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Invalid scheduler type '{override_type_name}' for {cluster_type.name}. "
                f"Valid options: {list(_SCHEDULER_TYPES)}"
            ) from exc
        cluster_config = BaseReplicaSchedulerConfig.create_from_type(override_type)
        if not is_dataclass(base_config) or not is_dataclass(cluster_config):
            raise TypeError(
                "Replica scheduler configs must be dataclasses for field-copy behavior"
            )
        base_names = {field.name for field in fields(base_config)}
        cluster_names = {field.name for field in fields(cluster_config)}
        for field_name in sorted(base_names & cluster_names):
            setattr(cluster_config, field_name, getattr(base_config, field_name))

    for field_name in _OVERRIDE_FIELDS:
        config_name = f"{prefix}_replica_scheduler_config_{field_name}"
        if hasattr(config, config_name):
            value = getattr(config, config_name)
            if value is not None and hasattr(cluster_config, field_name):
                setattr(cluster_config, field_name, value)
    return cluster_config
