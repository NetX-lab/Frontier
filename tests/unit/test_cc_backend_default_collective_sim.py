from frontier.cc_backend.cc_backend_config import AstraSimAnalyticalCCBackendConfig
from frontier.config import ClusterConfig, ReplicaConfig


def test_cluster_config_defaults_to_astra_sim_analytical_backend() -> None:
    cluster_config = ClusterConfig(
        num_replicas=1,
        replica_config=ReplicaConfig(model_name="llama3.1-8b"),
    )

    assert isinstance(cluster_config.cc_backend_config, AstraSimAnalyticalCCBackendConfig)
    assert cluster_config.cc_backend_config.get_name() == "astra_sim_analytical"
