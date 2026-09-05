from __future__ import annotations

import pytest

from frontier.cc_backend.cc_backend_config import (
    AstraSimAnalyticalCCBackendConfig,
    CollectiveSimCCBackendConfig,
)
from frontier.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig, ReplicaConfig
from frontier.entities.cluster import Cluster


@pytest.mark.parametrize(
    ("backend_config_type", "materializer_name"),
    [
        (CollectiveSimCCBackendConfig, "_materialize_collective_sim_cc_config"),
        (AstraSimAnalyticalCCBackendConfig, "_materialize_astra_sim_analytical_cc_config"),
    ],
)
def test_runtime_collective_layout_is_local_to_one_replica_pod(
    backend_config_type: type,
    materializer_name: str,
) -> None:
    replica_config = ReplicaConfig(
        model_name="llama3.1-8b",
        attn_tensor_parallel_size=4,
        attn_dp=2,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=8,
    )
    cluster = Cluster(
        ClusterConfig(
            num_replicas=3,
            replica_config=replica_config,
            cc_backend_config=backend_config_type(),
        ),
        metrics_config=MetricsConfig(),
        generator_config=BaseRequestGeneratorConfig(),
    )

    materializer = getattr(cluster, materializer_name)
    materialized = materializer(
        backend_config_type(),
        replica_config,
        replica_config.world_size,
    )

    assert materialized.runtime_num_replicas == 1
    assert materialized.cluster_servers * materialized.cluster_gpus_per_server == (
        replica_config.world_size
    )
    assert materialized.runtime_attn_dp == 2
    assert materialized.runtime_attn_tensor_parallel_size == 4
    assert materialized.runtime_moe_tensor_parallel_size == 1
    assert materialized.runtime_moe_expert_parallel_size == 8


@pytest.mark.parametrize(
    "backend_config_type",
    [CollectiveSimCCBackendConfig, AstraSimAnalyticalCCBackendConfig],
)
def test_runtime_parallel_dimensions_do_not_merge_outer_replicas(
    backend_config_type: type,
) -> None:
    config = backend_config_type(
        cluster_servers=1,
        cluster_gpus_per_server=8,
        runtime_num_replicas=3,
        runtime_num_pipeline_stages=1,
        runtime_attn_tensor_parallel_size=4,
        runtime_attn_dp=2,
        runtime_moe_tensor_parallel_size=1,
        runtime_moe_expert_parallel_size=8,
    )
    if backend_config_type is CollectiveSimCCBackendConfig:
        from frontier.cc_backend.backends.collective_sim_cc_backend import CollectiveSimCCBackend

        backend_cls = CollectiveSimCCBackend
    else:
        from frontier.cc_backend.backends.astra_sim_analytical_cc_backend import AstraSimAnalyticalCCBackend

        backend_cls = AstraSimAnalyticalCCBackend

    # Bypass the optional predictor initialization and inspect the pure layout
    # helpers that receive materialized runtime metadata.
    backend = object.__new__(backend_cls)
    backend._config = config

    assert backend._get_attention_parallel_dim_sizes() == {
        "TP": 4,
        "CP": 1,
        "DP": 2,
        "EP": 1,
    }
    assert backend._get_moe_parallel_dim_sizes() == {
        "TP": 1,
        "CP": 1,
        "DP": 1,
        "EP": 8,
    }
