"""An SGLang thinking round waits for its previous round's terminal release.

With MONOLITHIC PP >= 4 a completed round keeps its KV and its scheduler
frontier for extra iterations. With no tool latency the next round is queued
at once. The vllm_v1 waiting pass admits nothing while a release is pending.
The SGLang prefill-first pass must skip that request. Scheduled from the old
round's frontier, the new round gets 0 tokens when the old round was at least
as long (the run stops), and otherwise runs from the wrong frontier and never
completes. Requests queued behind it are still admitted.
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
    TraceRequestGeneratorConfig,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator


@pytest.fixture(autouse=True)
def _fresh_global_vars():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def _config(
    tmp_path,
    *,
    simulation_mode,
    hidden_round_prefill_tokens,
    request_generator_config,
    replica_scheduler_config,
):
    return SimulationConfig(
        simulation_mode=simulation_mode,
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        enable_thinking_mode=True,
        thinking_depth=2,
        tool_call_latency=0.0,
        thinking_round_prefill_tokens=[hidden_round_prefill_tokens],
        thinking_round_decode_tokens=[1],
        cluster_config=ClusterConfig(
            replica_config=ReplicaConfig(
                model_name="llama2_7b_dense_example",
                num_pipeline_stages=4,
                attn_tensor_parallel_size=1,
            ),
            replica_scheduler_config=replica_scheduler_config,
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
        request_generator_config=request_generator_config,
    )


def _assert_every_request_completes(simulator, num_requests):
    requests = list(simulator._all_requests)
    assert len(requests) == num_requests
    for request in requests:
        assert request.completed, request.id
        assert request.num_processed_decode_tokens == request.num_decode_tokens


# The hidden round is shorter or longer than the 32-token final prompt.
@pytest.mark.parametrize("hidden_round_prefill_tokens", [16, 64])
@pytest.mark.parametrize("num_requests", [1, 6])
def test_a_thinking_round_queued_during_its_terminal_release_completes(
    tmp_path, hidden_round_prefill_tokens, num_requests
):
    simulator = Simulator(
        _config(
            tmp_path,
            simulation_mode="offline",
            hidden_round_prefill_tokens=hidden_round_prefill_tokens,
            request_generator_config=SyntheticRequestGeneratorConfig(
                num_requests=num_requests,
                length_generator_config=FixedRequestLengthGeneratorConfig(
                    prefill_tokens=32, decode_tokens=4
                ),
                interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                    qps=1.0
                ),
            ),
            replica_scheduler_config=SglangSchedulerConfig(),
        )
    )
    simulator.run()

    _assert_every_request_completes(simulator, num_requests)


def test_a_request_queued_behind_a_round_awaiting_its_release_is_admitted(
    tmp_path, monkeypatch
):
    # Request 1 arrives during request 0's first round. Final-round priority
    # queues request 0's second round ahead of it while that round waits for
    # its release.
    trace = tmp_path / "requests.csv"
    trace.write_text(
        "arrived_at,num_prefill_tokens,num_decode_tokens\n0.0,32,4\n0.3,32,4\n"
    )
    passes = []
    schedule_waiting_requests = VLLMv1EngineReplicaScheduler._schedule_waiting_requests

    def observed_schedule_waiting_requests(self, token_budget):
        queued = [request.id for request in self._get_sorted_waiting_queue()]
        pending = set(self._get_monolithic_pp_pending_terminal_release_iters())
        result = schedule_waiting_requests(self, token_budget)
        passes.append((queued, pending, [request.id for request in result[1]]))
        return result

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler,
        "_schedule_waiting_requests",
        observed_schedule_waiting_requests,
    )
    simulator = Simulator(
        _config(
            tmp_path,
            simulation_mode="online",
            hidden_round_prefill_tokens=64,
            request_generator_config=TraceRequestGeneratorConfig(
                trace_file=str(trace)
            ),
            replica_scheduler_config=SglangSchedulerConfig(
                enable_thinking_round_priority=True
            ),
        )
    )
    simulator.run()

    passes_behind_a_release = [
        (queued, pending, scheduled)
        for queued, pending, scheduled in passes
        if queued
        and queued[0] in pending
        and any(request_id not in pending for request_id in queued)
    ]
    assert passes_behind_a_release, passes
    for queued, pending, scheduled in passes_behind_a_release:
        assert any(
            request_id not in pending for request_id in scheduled
        ), (queued, pending, scheduled)
    _assert_every_request_completes(simulator, 2)
