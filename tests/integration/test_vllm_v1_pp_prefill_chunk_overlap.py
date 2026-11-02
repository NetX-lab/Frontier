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
end; at PP1 each chunk follows the previous one by default. Explicit queue
capacity two models v0.28 V1 asynchronous prefill admission at PP1, while
the existing stage remains serial and decode waits for consumed output.
"""

from __future__ import annotations

from dataclasses import replace
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
from frontier.config import global_vars
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.entities.batch_stage import BatchStage
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType

CHUNK_TOKENS = 32
TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,96,4
"""


@pytest.fixture(autouse=True)
def reset_simulation_globals():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


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


def _config(root, case, num_pipeline_stages, replica_config=None):
    sys_arch, cluster_fields, _ = CASES[case]
    clusters = cluster_fields(num_pipeline_stages)
    if replica_config is not None:
        clusters["replica_config"] = replica_config
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
            **clusters,
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


@pytest.mark.parametrize("graph_mode", ["none", "full_decode_only", "piecewise"])
@pytest.mark.parametrize("model_name,device,network", [
    ("llama2_7b_dense_example", "a100", "a100_pairwise_nvlink"),
    ("zai-org/GLM-5.3", "h800", "h800_dgx"),
])
def test_pp1_capacity_two_admits_prompt_chunks_before_consumption_and_runs_serially(
    tmp_path, monkeypatch, graph_mode, model_name, device, network
):
    config = _config(tmp_path, "monolithic", 1, replica_config=ReplicaConfig(
        model_name=model_name, device=device, network_device=network,
        attn_tensor_parallel_size=1, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
    ))
    config.decode_cuda_graph_mode = graph_mode
    config.cudagraph_capture_sizes = None if graph_mode == "none" else [1, 32]
    config.__post_init__()
    config.cluster_config.replica_scheduler_config = replace(
        config.cluster_config.replica_scheduler_config,
        max_concurrent_batches=2, block_size=64, num_blocks_mode="explicit",
    )
    batches = []
    stages = []
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end
    on_stage_end = BatchStage.on_stage_end

    def recorded_batch_end(self, batch):
        batches.append(batch)
        on_batch_end(self, batch)

    def recorded_stage_end(self, time):
        on_stage_end(self, time)
        stages.append(self)

    monkeypatch.setattr(VLLMv1EngineReplicaScheduler, "on_batch_end", recorded_batch_end)
    monkeypatch.setattr(BatchStage, "on_stage_end", recorded_stage_end)
    simulator = Simulator(config)
    simulator.run()

    (request,) = simulator._all_requests
    assert request.completed
    assert request.num_prefill_tokens == 96
    assert request.num_processed_decode_tokens == request.num_decode_tokens == 4
    chunks = [batch for batch in batches if batch.num_prefill_tokens]
    assert [batch.num_tokens for batch in chunks] == [[32], [32], [32]]
    assert [batch.num_context_tokens for batch in chunks] == [[0], [32], [64]]
    first, second, third = chunks
    assert second.scheduled_at == first.scheduled_at < first.completed_at
    assert third.scheduled_at == first.completed_at
    assert {stage.pipeline_stage for stage in stages} == {0}
    assert len(stages) == len(batches) == 6
    for previous, following in zip(stages, stages[1:]):
        assert previous.scheduled_at < previous.completed_at <= following.scheduled_at
    assert stages[1].scheduled_at == stages[0].completed_at
    if graph_mode == "none":
        assert all(batch.decode_cuda_graph_metadata is None for batch in batches)
    else:
        expected_modes = (
            ["NONE"] * 3 + ["FULL"] * 3
            if graph_mode == "full_decode_only" else ["PIECEWISE"] * 6
        )
        metadata = [batch.decode_cuda_graph_metadata for batch in batches]
        assert [item.runtime_mode for item in metadata] == expected_modes
        assert [item.original_total_tokens for item in metadata] == [32, 32, 32, 1, 1, 1]
        assert [item.original_decode_batch_size for item in metadata] == [0, 0, 0, 1, 1, 1]
