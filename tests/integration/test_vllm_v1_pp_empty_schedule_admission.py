"""An engine that scheduled nothing waits for its oldest batch before admitting.

vLLM 0.10.2 appends an empty scheduler output to the batch queue and then
blocks on the oldest in-flight batch, even when the queue still has room
(`EngineCore.step_with_batch_queue`, `vllm/v1/engine/core.py:364-424`). A
request that arrives meanwhile waits in the input queue until the next busy-loop
iteration drains it, after that oldest output is applied (`:804-856`). With
PP=1 every iteration blocks on its own batch.

The MONOLITHIC case and the unified PDD DECODE case follow the same rule. Both
use one attention-DP lane, so no DP dummy forward follows the empty iteration
(`:1183-1190`).
"""

from __future__ import annotations

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
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType

# r0's batch is the only one in flight when r1 reaches the cluster under test,
# long after the host time of the engine iteration that scheduled r0's batch:
# an arrival inside that window would be drained by the next iteration without
# a block. Under dummy timing, r0's 16-token prompt is one MONOLITHIC batch from
# 0 to 0.464 s at PP2. In the PDD case two prefill replicas keep r1's prefill
# from queueing behind r0's, so r1 reaches the DECODE replica at about 0.459 s,
# inside r0's first decode batch (about 0.359 to 0.723 s at PP2).
TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,16,4
0.1,16,4
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
        prefill_cluster_num_replicas=2,
        decode_cluster_num_replicas=1,
        replica_config=_replica(),
        decode_replica_config_num_pipeline_stages=num_pipeline_stages,
    )


CASES = {
    "monolithic": ("co-location", _monolithic_cluster, ClusterType.MONOLITHIC),
    "pdd_decode": ("pd-disaggregation", _pdd_clusters, ClusterType.DECODE),
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
                max_tokens_in_batch=64,
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
@pytest.mark.parametrize("case", ["monolithic", "pdd_decode"])
def test_an_arrival_during_a_blocked_iteration_waits_for_the_oldest_batch(
    case, num_pipeline_stages, tmp_path, monkeypatch
):
    cluster_type = CASES[case][2]
    batches = []
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end

    def recorded_on_batch_end(self, batch):
        if self._cluster_type == cluster_type:
            batches.append((batch.scheduled_at, batch.completed_at, batch.request_ids))
        on_batch_end(self, batch)

    monkeypatch.setattr(VLLMv1EngineReplicaScheduler, "on_batch_end", recorded_on_batch_end)
    simulator = Simulator(_config(tmp_path, case, num_pipeline_stages))
    simulator.run()
    first, second = sorted(simulator._all_requests, key=lambda request: request.id)
    batches.sort()

    assert first.completed and second.completed
    # Premise: r0's first batch on this cluster runs alone, and r1 reaches the
    # cluster while it is in flight.
    first_scheduled, first_ended, first_ids = batches[0]
    arrival = (
        second.arrived_at
        if cluster_type == ClusterType.MONOLITHIC
        else second.kv_cache_transfer_end_time
    )
    assert first_ids == [first.id]
    assert first_scheduled < arrival < first_ended
    # At PP2 the engine's next iteration found nothing to schedule and blocked
    # on that batch; at PP1 it blocked on it anyway.
    second_admission = min(
        scheduled for scheduled, _, ids in batches if second.id in ids
    )
    assert second_admission == first_ended
