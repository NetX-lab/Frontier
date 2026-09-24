"""Rejected drafts leave the vllm_v1 scheduler frontier when their step ends.

vLLM schedules a speculative step with its whole verify width and, when the
step's output arrives, subtracts the rejected drafts from num_computed_tokens.
After every decode step the frontier therefore stands one token behind the
request's tokens: the last sampled token is not computed yet. A frontier that
keeps the rejected drafts runs ahead of the request, inflates its KV accounting
and, near max_model_len, leaves the request with nothing it may schedule.
"""

from __future__ import annotations

import pytest

from frontier.config import global_vars
from frontier.config import (
    ClusterConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SimulationConfig,
    SpeculativeDecodingConfig,
    SyntheticRequestGeneratorConfig,
    UniformRequestLengthGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator

# Two drafts with two committed tokens per step reject one draft every step.
SPEC_DECODE = SpeculativeDecodingConfig(
    enabled=True,
    method="ngram",
    num_speculative_tokens=2,
    committed_tokens_per_iteration=2,
)

# Each case ended with a request stranded at max_model_len (96 tokens) before
# the fix. There is no preemption: 64 blocks hold every request.
CASES = {
    "dense": dict(
        replica=dict(model_name="llama2_7b_dense_example"),
        num_requests=12,
        seed=7,
    ),
    "moe_dp2_ep2": dict(
        replica=dict(
            model_name="Qwen3-30B-A3B-tiny",
            attn_dp=2,
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=2,
            total_expert_num=16,
            router_topk=8,
        ),
        num_requests=6,
        seed=5,
    ),
}


@pytest.fixture(autouse=True)
def _fresh_global_vars():
    # A run sets process-wide model flags once, and these cases mix dense and
    # MoE models in one process.
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def _config(root, case):
    seed = case["seed"]
    return SimulationConfig(
        simulation_mode="online",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=ClusterConfig(
            replica_config=ReplicaConfig(
                device="a100",
                network_device="a100_pairwise_nvlink",
                attn_tensor_parallel_size=1,
                speculative_decoding_config=SPEC_DECODE,
                **case["replica"],
            ),
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=64,
                block_size=16,
                batch_size_cap=4,
                max_tokens_in_batch=16,
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
        request_generator_config=SyntheticRequestGeneratorConfig(
            num_requests=case["num_requests"],
            seed=seed,
            length_generator_config=UniformRequestLengthGeneratorConfig(
                min_tokens=8,
                max_tokens=96,
                prefill_to_decode_ratio=4.0,
                seed=seed,
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                qps=200.0, seed=seed
            ),
        ),
    )


def _observe_decode_frontiers(monkeypatch):
    """Record (frontier, tokens) of each unfinished decode request at step end."""

    frontiers = []
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end

    def observed_on_batch_end(self, batch):
        on_batch_end(self, batch)
        if batch.spec_decode_metadata is None:
            return
        for request in batch.current_execution_requests:
            if request.completed or not request.is_prefill_complete:
                continue
            frontiers.append(dict(
                request_id=request.id,
                frontier=self._get_scheduler_num_computed_tokens(request),
                tokens=request.num_processed_tokens,
            ))

    monkeypatch.setattr(VLLMv1EngineReplicaScheduler, "on_batch_end", observed_on_batch_end)
    return frontiers


@pytest.mark.parametrize("name", CASES)
def test_a_speculative_decode_step_returns_its_rejected_drafts(name, tmp_path, monkeypatch):
    case = CASES[name]
    frontiers = _observe_decode_frontiers(monkeypatch)
    simulator = Simulator(_config(tmp_path, case))
    simulator.run()

    assert frontiers
    for step in frontiers:
        assert step["frontier"] == step["tokens"] - 1, step
    requests = list(simulator._all_requests)
    assert len(requests) == case["num_requests"]
    for request in requests:
        assert request.completed, request.id
        assert request.num_processed_decode_tokens == request.num_decode_tokens
