"""Sequential runs honor ``SimulationConfig.time_limit``."""

from pathlib import Path

from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
from frontier.config import (
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    RoundRobinClusterSchedulerConfig,
    SimulationConfig,
    SyntheticRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.simulator import Simulator

TIME_LIMIT_S = 10


def _run(root: Path, time_limit: int) -> Simulator:
    config = SimulationConfig(
        simulation_mode="online",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        time_limit=time_limit,
        cluster_config=ClusterConfig(
            replica_config=ReplicaConfig(
                model_name="meta-llama/Llama-2-7b-hf",
                device="a100",
                network_device="a100_pairwise_nvlink",
            ),
            replica_scheduler_config=VllmV1SchedulerConfig(),
            cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
            cc_backend_config=AnalyticalCCBackendConfig(),
        ),
        metrics_config=MetricsConfig(
            output_dir=str(root / "metrics"),
            cache_dir=str(root / "cache"),
            write_metrics=True,
            store_request_metrics=True,
            store_plots=False,
            enable_chrome_trace=False,
            write_json_trace=False,
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            num_requests=10,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=64, decode_tokens=16
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1),
        ),
    )
    simulator = Simulator(config)
    simulator.run()
    return simulator


def _completion_times(simulator: Simulator) -> list:
    requests = sorted(simulator._all_requests, key=lambda request: request.arrived_at)
    return [request.completed_at if request.completed else None for request in requests]


def test_sequential_run_stops_at_the_first_event_past_the_time_limit(tmp_path):
    full = _run(tmp_path / "full", time_limit=0)
    full_completions = _completion_times(full)
    assert all(time is not None for time in full_completions)
    assert full._time > TIME_LIMIT_S

    limited = _run(tmp_path / "limited", time_limit=TIME_LIMIT_S)

    # The run handles the first event past the limit, then stops; up to that
    # event it is the same run as the unlimited one.
    assert TIME_LIMIT_S < limited._time < full._time
    expected = [time if time <= limited._time else None for time in full_completions]
    assert _completion_times(limited) == expected
    assert None in expected and any(time is not None for time in expected)
    assert (Path(limited._config.metrics_config.output_dir) / "request_metrics.csv").is_file()
