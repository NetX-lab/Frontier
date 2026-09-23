"""P1(b): record Frontier's admission and completion boundaries under PP.

No source change. The probe wraps `_get_next_batch` (one call per admission,
after `_running_requests` has grown) and the inert `on_replica_batch_end` seam,
and reads the candidate report key -- the Replica's next forward id held by
`ForwardSyncState` -- at each boundary.

Run with ``PYTHONPATH`` set to the Frontier tree under test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def build_config(root: Path, *, is_moe: bool, attn_dp: int, moe_ep: int, stages: int):
    from frontier.config import (
        BaseModelConfig, ClusterConfig, FixedRequestLengthGeneratorConfig,
        MetricsConfig, PoissonRequestIntervalGeneratorConfig,
        RandomForrestExecutionTimePredictorConfig, ReplicaConfig,
        RoundRobinClusterSchedulerConfig, SimulationConfig,
        SyntheticRequestGeneratorConfig, VllmV1SchedulerConfig,
    )
    from frontier.types import ActivationType, NormType

    model = BaseModelConfig(
        num_layers=6, num_q_heads=4, num_kv_heads=2, embedding_dim=256,
        mlp_hidden_dim=64, max_position_embeddings=4096, use_gated_mlp=True,
        use_bias=False, use_qkv_bias=False, activation=ActivationType.SILU,
        norm=NormType.RMS_NORM, post_attn_norm=True, vocab_size=1024,
        is_moe=is_moe, num_experts=8 if is_moe else 0,
        num_experts_per_tok=2 if is_moe else 0, torch_dtype="bfloat16",
    )
    model._model_name = f"w9_probe_{'moe' if is_moe else 'dense'}"
    original = BaseModelConfig.create_from_name
    BaseModelConfig.create_from_name = classmethod(
        lambda cls, name: model if name == model._model_name else original(name)
    )
    moe_fields = dict(
        moe_tensor_parallel_size=1, moe_expert_parallel_size=moe_ep,
        total_expert_num=8, router_topk=2,
    ) if is_moe else {}
    replica = ReplicaConfig(
        model_name=model._model_name, device="a100",
        network_device="a100_pairwise_nvlink", num_pipeline_stages=stages,
        attn_tensor_parallel_size=1, attn_dp=attn_dp,
        memory_margin_fraction=0.1, **moe_fields,
    )
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=128, block_size=16, batch_size_cap=4,
            max_tokens_in_batch=16, enable_chunked_prefill=True,
        ),
        cluster_scheduler_config=RoundRobinClusterSchedulerConfig(),
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True
        ),
    )
    return SimulationConfig(
        simulation_mode="offline", sys_arch="co-location",
        enable_parallel_clusters=False, decode_cuda_graph_mode="none",
        cluster_config=cluster,
        metrics_config=MetricsConfig(
            output_dir=str(root / "metrics"), cache_dir=str(root / "cache"),
            run_id="w9_probe", write_metrics=False, store_request_metrics=False,
            store_batch_metrics=False, store_operation_metrics=False,
            store_utilization_metrics=False, store_plots=False,
            enable_chrome_trace=False, write_json_trace=False,
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            num_requests=6,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=16, decode_tokens=3
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1e6),
        ),
    )


def run(root: Path, *, is_moe: bool, attn_dp: int, moe_ep: int, stages: int):
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        BaseClusterScheduler,
    )
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
        VLLMv1EngineReplicaScheduler,
    )
    from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
    from frontier.simulator import Simulator

    events: list[dict] = []
    schedulers: dict = {}

    def next_forward_id(cluster_scheduler, replica_id):
        state = cluster_scheduler._forward_sync_state
        return int(state._next_step_id_by_replica.get(replica_id, 0))

    original_next_batch = VLLMv1EngineReplicaScheduler._get_next_batch

    def observed_next_batch(self, is_micro_batch=False):
        batch = original_next_batch(self, is_micro_batch=is_micro_batch)
        if batch is not None:
            schedulers[(self._replica_id, self._replica_local_id)] = self
            events.append({
                "kind": "admit",
                "time": round(float(self._current_schedule_time), 9),
                "lane": self._replica_local_id,
                "batch": batch.id,
                "provisional": batch._forward_cohort_provisional_id,
                "running_batches_before": self._num_running_batches,
                "stages": self._num_stages,
                "load": list(self.get_request_load()),
                "candidate_key": next_forward_id(self._cluster_scheduler, self._replica_id),
            })
        return batch

    original_batch_end = BaseClusterScheduler.on_replica_batch_end

    def observed_batch_end(self, time, replica_id, replica_local_id, batch):
        result = original_batch_end(self, time, replica_id, replica_local_id, batch)
        lane = self.get_replica_scheduler(replica_id, replica_local_id)
        events.append({
            "kind": "complete",
            "time": round(float(time), 9),
            "lane": replica_local_id,
            "batch": batch.id,
            "provisional": batch._forward_cohort_provisional_id,
            "resolved": ForwardSyncState.get_step_id(batch),
            "running_batches_after": lane.num_running_batches,
            "load": list(lane.get_request_load()),
            "candidate_key": next_forward_id(self, replica_id),
        })
        return result

    VLLMv1EngineReplicaScheduler._get_next_batch = observed_next_batch
    BaseClusterScheduler.on_replica_batch_end = observed_batch_end
    try:
        config = build_config(root, is_moe=is_moe, attn_dp=attn_dp,
                              moe_ep=moe_ep, stages=stages)
        simulator = Simulator(config)
        simulator.run()
        requests = list(simulator._all_requests)
    finally:
        VLLMv1EngineReplicaScheduler._get_next_batch = original_next_batch
        BaseClusterScheduler.on_replica_batch_end = original_batch_end
    return {
        "completed": sum(1 for r in requests if r.completed),
        "requests": len(requests),
        "events": events,
    }


if __name__ == "__main__":
    root = Path(sys.argv[1])
    summary = {}
    for label, shape in {
        "moe_dp2_pp1": dict(is_moe=True, attn_dp=2, moe_ep=2, stages=1),
        "moe_dp2_pp2": dict(is_moe=True, attn_dp=2, moe_ep=2, stages=2),
        "moe_dp2_pp3": dict(is_moe=True, attn_dp=2, moe_ep=2, stages=3),
        "dense_dp1_pp2": dict(is_moe=False, attn_dp=1, moe_ep=1, stages=2),
    }.items():
        case_root = root / label
        case_root.mkdir(parents=True, exist_ok=True)
        result = run(case_root, **shape)
        summary[label] = result
        events = result["events"]
        print(f"\n=== {label}: {result['completed']}/{result['requests']} completed, "
              f"{len(events)} boundaries")
        print(f"{'time':>9} {'kind':>8} {'lane':>4} {'batch':>5} {'prov':>4} "
              f"{'resolved':>8} {'load':>8} {'key':>4} {'slots':>6}")
        for e in events[:28]:
            slots = (f"{e['running_batches_before']}/{e['stages']}" if e["kind"] == "admit"
                     else f"{e['running_batches_after']}")
            print(f"{e['time']:>9.5f} {e['kind']:>8} {str(e['lane']):>4} {e['batch']:>5} "
                  f"{e['provisional']:>4} {str(e.get('resolved', '')):>8} "
                  f"{str(tuple(e['load'])):>8} {e['candidate_key']:>4} {slots:>6}")
    (root / "frontier_boundaries.json").write_text(json.dumps(summary, indent=1))
