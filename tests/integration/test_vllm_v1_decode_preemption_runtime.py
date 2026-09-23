"""A monolithic request preempted during decode resumes and completes.

vLLM v1 preemption discards a request's computed KV but keeps its generated
output. Three requests arrive together with KV for fewer tokens than they reach
together, so decode growth preempts a request that has finished its prefill.
The real `Simulator` loop must bring that request back and finish every request
with all of its output tokens.
"""

from __future__ import annotations

from frontier.config import (
    ClusterConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    TraceRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator

# Each request ends at 60 tokens, four 16-token blocks; eight blocks hold two.
TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,30,30
0.0,30,30
0.0,30,30
"""


def _config(root):
    trace = root / "decode_preemption.csv"
    trace.write_text(TRACE)
    return SimulationConfig(
        simulation_mode="online",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=ClusterConfig(
            replica_config=ReplicaConfig(
                model_name="llama2_7b_dense_example",
                device="a100",
                network_device="a100_pairwise_nvlink",
            ),
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=8,
                block_size=16,
                batch_size_cap=4,
                max_tokens_in_batch=64,
                enable_chunked_prefill=True,
            ),
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
        ),
        metrics_config=MetricsConfig(
            output_dir=str(root / "metrics"),
            cache_dir=str(root / "cache"),
            write_metrics=False,
            store_request_metrics=False,
            store_batch_metrics=False,
            store_operation_metrics=False,
            store_utilization_metrics=False,
            store_plots=False,
            enable_chrome_trace=False,
            write_json_trace=False,
        ),
        request_generator_config=TraceRequestGeneratorConfig(trace_file=str(trace)),
    )


def test_a_request_preempted_during_decode_resumes_and_completes(tmp_path, monkeypatch):
    decode_preemptions = []
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request

    def observed_preempt_request(self, victim, preempted_requests):
        processed_before = victim.num_processed_tokens
        preempt_request(self, victim, preempted_requests)
        if victim.is_prefill_complete:
            decode_preemptions.append(
                (victim.id, processed_before, victim.num_processed_tokens)
            )

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt_request
    )
    simulator = Simulator(_config(tmp_path))
    simulator.run()

    # The run has to reach a preemption after prefill, or it proves nothing.
    assert decode_preemptions
    for request_id, processed_before, processed_after in decode_preemptions:
        assert processed_after == processed_before, request_id
    requests = list(simulator._all_requests)
    assert len(requests) == 3
    for request in requests:
        assert request.completed, request.id
        assert request.num_processed_decode_tokens == request.num_decode_tokens
