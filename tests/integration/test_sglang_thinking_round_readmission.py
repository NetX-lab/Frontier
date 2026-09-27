"""An SGLang thinking round waits for its previous round's terminal release.

With MONOLITHIC PP >= 4 a completed round keeps its KV and its scheduler
frontier for extra iterations. With no tool latency the next round is queued
at once. The vllm_v1 waiting pass admits nothing while a release is pending;
the SGLang prefill-first pass must at least skip that request, or it schedules
the new round from the old round's frontier (0 tokens).
"""

from __future__ import annotations

import pytest

from frontier.config import global_vars
from frontier.config import (
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SglangSchedulerConfig,
    SimulationConfig,
    SyntheticRequestGeneratorConfig,
)
from frontier.simulator import Simulator


@pytest.fixture(autouse=True)
def _fresh_global_vars():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


@pytest.mark.parametrize("num_requests", [1, 6])
def test_a_thinking_round_queued_during_its_terminal_release_completes(
    tmp_path, num_requests
):
    config = SimulationConfig(
        simulation_mode="offline",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        enable_thinking_mode=True,
        thinking_depth=2,
        tool_call_latency=0.0,
        thinking_round_prefill_tokens=[64],
        thinking_round_decode_tokens=[1],
        cluster_config=ClusterConfig(
            replica_config=ReplicaConfig(
                model_name="llama2_7b_dense_example",
                num_pipeline_stages=4,
                attn_tensor_parallel_size=1,
            ),
            replica_scheduler_config=SglangSchedulerConfig(),
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
        ),
        metrics_config=MetricsConfig(
            output_dir=str(tmp_path / "metrics"),
            cache_dir=str(tmp_path / "cache"),
            write_metrics=False,
            store_plots=False,
            enable_chrome_trace=False,
            write_json_trace=False,
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            num_requests=num_requests,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=32, decode_tokens=4
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
        ),
    )
    simulator = Simulator(config)
    simulator.run()

    requests = list(simulator._all_requests)
    assert len(requests) == num_requests
    for request in requests:
        assert request.completed, request.id
        assert request.num_processed_decode_tokens == request.num_decode_tokens
