"""A prompt's next chunk is scheduled while its previous chunk is in flight.

vLLM 0.10.2 advances `num_computed_tokens` when it schedules a step
(`Scheduler._update_after_schedule`, `vllm/v1/core/sched/scheduler.py`), and
its running loop skips a request only when no prompt token is left to
schedule. At PP>1 the next iteration therefore schedules the next chunk while
the previous one is still in the pipeline, and the worker builds that chunk's
attention from the `num_computed_tokens` of the scheduler output: the tokens
of every earlier chunk, finished or not.

A 96-token prompt with a 32-token budget runs as three chunks. At PP2 the
first two fill the pipeline at once and the third follows the first chunk's
end; at PP1 each chunk follows the previous one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config import (
    ClusterConfig,
    MetricsConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    TraceRequestGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType

CHUNK_TOKENS = 32
TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,96,4
"""


def _replica(num_pipeline_stages=1):
    return ReplicaConfig(
        model_name="llama2_7b_dense_example",
        device="a100",
        network_device="a100_pairwise_nvlink",
        attn_tensor_parallel_size=1,
        num_pipeline_stages=num_pipeline_stages,
    )


def _monolithic_cluster(num_pipeline_stages):
    return dict(replica_config=_replica(num_pipeline_stages))


def _pdd_clusters(num_pipeline_stages):
    return dict(
        prefill_cluster_num_replicas=1,
        decode_cluster_num_replicas=1,
        replica_config=_replica(),
        prefill_replica_config_num_pipeline_stages=num_pipeline_stages,
    )


CASES = {
    "monolithic": ("co-location", _monolithic_cluster, ClusterType.MONOLITHIC),
    "pdd_prefill": ("pd-disaggregation", _pdd_clusters, ClusterType.PREFILL),
}


def _config(root, case, num_pipeline_stages):
    sys_arch, cluster_fields, _ = CASES[case]
    trace = root / "trace.csv"
    trace.write_text(TRACE)
    return SimulationConfig(
        simulation_mode="online",
        sys_arch=sys_arch,
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=ClusterConfig(
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=64,
                block_size=16,
                batch_size_cap=4,
                max_tokens_in_batch=CHUNK_TOKENS,
                enable_chunked_prefill=True,
            ),
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
            **cluster_fields(num_pipeline_stages),
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


@pytest.mark.parametrize("num_pipeline_stages", [2, 1])
@pytest.mark.parametrize("case", ["monolithic", "pdd_prefill"])
def test_the_next_prompt_chunk_is_scheduled_while_the_previous_one_is_in_flight(
    case, num_pipeline_stages, tmp_path, monkeypatch
):
    cluster_type = CASES[case][2]
    chunks = []
    prefill_attention_params = []
    predictor = SimpleNamespace(_config=SimpleNamespace(kv_cache_prediction_granularity=1))
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end

    def recorded_on_batch_end(self, batch):
        if self._cluster_type == cluster_type and batch.num_prefill_tokens:
            chunks.append(batch)
            prefill_attention_params.append(
                (
                    batch.requests[0].num_context_tokens,
                    SklearnExecutionTimePredictor._get_batch_prefill_attention_params(
                        predictor, batch
                    ),
                )
            )
        on_batch_end(self, batch)

    monkeypatch.setattr(VLLMv1EngineReplicaScheduler, "on_batch_end", recorded_on_batch_end)
    simulator = Simulator(_config(tmp_path, case, num_pipeline_stages))
    simulator.run()
    (request,) = simulator._all_requests
    chunks.sort(key=lambda batch: batch.scheduled_at)

    assert request.completed
    assert [batch.num_tokens for batch in chunks] == [[CHUNK_TOKENS]] * 3
    first, second, third = chunks
    if num_pipeline_stages == 2:
        assert second.scheduled_at == first.scheduled_at
        assert third.scheduled_at == first.completed_at
    else:
        assert second.scheduled_at == first.completed_at
        assert third.scheduled_at == second.completed_at

    # Each chunk attends to every token scheduled before it, and its prefill
    # attention is priced with that context rather than the Request's state:
    # when the second chunk ends, the Request has computed 64 tokens.
    assert [batch.num_context_tokens for batch in chunks] == [[0], [32], [64]]
    assert prefill_attention_params[1] == (64, [(32, CHUNK_TOKENS)])
