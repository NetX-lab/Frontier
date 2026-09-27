"""Rejected drafts leave the vllm_v1 scheduler frontier when their step ends.

vLLM advances num_computed_tokens by a speculative step's scheduled width and,
when the step's output arrives, subtracts the scheduled tokens that produced no
output. After every decode step the frontier therefore stands one token behind
the request's tokens: the last sampled token is not computed yet. A frontier that
keeps the rejected drafts runs ahead of the request, inflates its KV accounting
and, near max_model_len, leaves the request with nothing it may schedule.

A MONOLITHIC target-embedded MTP request schedules one token fewer than its
verify width on its first decode step, so the rollback is taken against the
scheduled width, not the verify width.

On the PDD DECODE role the first output token arrives with the KV handoff and
is not part of the request's processed tokens there, so the frontier equals
them after every decode step.
"""

from __future__ import annotations

import dataclasses
import json

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
    TraceRequestGeneratorConfig,
    UniformRequestLengthGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities import Request
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType

# Two drafts with two committed tokens per step reject one draft every step.
NGRAM = SpeculativeDecodingConfig(
    enabled=True,
    method="ngram",
    num_speculative_tokens=2,
    committed_tokens_per_iteration=2,
)
QWEN3_NEXT_MTP = SpeculativeDecodingConfig(
    enabled=True,
    method="qwen3_next_mtp",
    num_speculative_tokens=2,
    committed_tokens_per_iteration=2,
    mtp_n_predict=1,
    mtp_num_layers=1,
)

# The ngram cases ended with a request stranded at max_model_len (96 tokens)
# before the rollback; the MTP case left the frontier two tokens behind the
# request after every step. There is no preemption: 64 blocks hold every request.
CASES = {
    "dense": dict(
        spec_decode=NGRAM,
        replica=dict(model_name="llama2_7b_dense_example"),
        num_requests=12,
        seed=7,
    ),
    "dense_qwen3_next_mtp": dict(
        spec_decode=QWEN3_NEXT_MTP,
        replica=dict(model_name="llama2_7b_dense_example"),
        num_requests=12,
        seed=7,
    ),
    "moe_dp2_ep2": dict(
        spec_decode=NGRAM,
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


def _config(root, case, sys_arch="co-location"):
    seed = case["seed"]
    role_args = {}
    if sys_arch == "pd-disaggregation":
        role_args = dict(prefill_cluster_num_replicas=1, decode_cluster_num_replicas=1)
    return SimulationConfig(
        simulation_mode="online",
        sys_arch=sys_arch,
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=ClusterConfig(
            replica_config=ReplicaConfig(
                device="a100",
                network_device="a100_pairwise_nvlink",
                attn_tensor_parallel_size=1,
                speculative_decoding_config=case["spec_decode"],
                **case["replica"],
            ),
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=64,
                block_size=16,
                batch_size_cap=4,
                max_tokens_in_batch=16,
                max_model_len=96,
                enable_chunked_prefill=True,
            ),
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
            **role_args,
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
    """Record (frontier, tokens, rejected) of each unfinished decode request at step end."""

    frontiers = []
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end

    def observed_on_batch_end(self, batch):
        on_batch_end(self, batch)
        if batch.spec_decode_metadata is None:
            return
        rejected_by_request_id = {
            request.id: scheduled - committed
            for request, scheduled, committed in zip(
                batch.requests,
                batch.num_tokens,
                batch.spec_decode_metadata.committed_tokens_per_request,
            )
        }
        for request in batch.current_execution_requests:
            if request.completed or not request.is_prefill_complete:
                continue
            frontiers.append(dict(
                cluster_type=self._cluster_type,
                request_id=request.id,
                frontier=self._get_scheduler_num_computed_tokens(request),
                tokens=request.num_processed_tokens,
                rejected=rejected_by_request_id[request.id],
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


def test_a_pdd_decode_step_returns_its_rejected_drafts(tmp_path, monkeypatch):
    # Request i commits 1 + (i % 3) tokens per step, so its steps reject
    # 2, 1 or 0 of their two drafts. The trace is keyed by request id, and
    # ids count per process.
    monkeypatch.setattr(Request, "_id", -1)
    trace = tmp_path / "acceptance_trace.json"
    trace.write_text(json.dumps({
        "per_request_committed_tokens_per_iteration": {
            str(i): [1 + (i % 3)] * 96 for i in range(CASES["dense"]["num_requests"])
        }
    }))
    case = dict(
        CASES["dense"],
        spec_decode=SpeculativeDecodingConfig(
            enabled=True,
            method="ngram",
            num_speculative_tokens=2,
            acceptance_trace_file=str(trace),
        ),
    )
    frontiers = _observe_decode_frontiers(monkeypatch)
    simulator = Simulator(_config(tmp_path, case, sys_arch="pd-disaggregation"))
    simulator.run()

    decode_steps = [
        step for step in frontiers if step["cluster_type"] == ClusterType.DECODE
    ]
    assert len(decode_steps) == len(frontiers)
    assert {step["rejected"] for step in decode_steps} == {0, 1, 2}
    for step in decode_steps:
        assert step["frontier"] == step["tokens"], step
    requests = list(simulator._all_requests)
    assert len(requests) == case["num_requests"]
    for request in requests:
        assert request.completed, request.id
        assert request.num_emitted_decode_tokens == request.num_decode_tokens


@pytest.mark.parametrize(
    "sys_arch, last_step_width",
    [("co-location", 1), ("pd-disaggregation", 2)],
)
def test_a_speculative_step_leaves_room_for_the_token_it_samples(
    tmp_path, monkeypatch, sys_arch, last_step_width
):
    # 80 + 16 tokens fill max_model_len (96). The trace schedules two drafts on
    # every step and commits two tokens, so the frontier reaches 94 with three
    # tokens wanted. The clamp keeps max_model_len - 1 - num_computed_tokens of
    # them, as vLLM's running phase does. On co-location the frontier is
    # num_computed_tokens: one token. The DECODE frontier is one token ahead of
    # num_computed_tokens, so its cap is max_model_len - frontier: two tokens.
    requests = tmp_path / "requests.csv"
    requests.write_text("arrived_at,num_prefill_tokens,num_decode_tokens\n0.0,80,16\n")
    acceptance_trace = tmp_path / "acceptance_trace.json"
    acceptance_trace.write_text(json.dumps({
        "committed_tokens_per_iteration": [2] * 16,
        "scheduled_draft_tokens_per_iteration": [2] * 16,
    }))
    steps = []
    advance = VLLMv1EngineReplicaScheduler._advance_scheduler_num_computed_tokens

    def observed_advance(self, request, num_scheduled_tokens):
        steps.append((self._get_scheduler_num_computed_tokens(request), num_scheduled_tokens))
        advance(self, request, num_scheduled_tokens)

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler,
        "_advance_scheduler_num_computed_tokens",
        observed_advance,
    )
    case = dict(
        CASES["dense"],
        num_requests=1,
        spec_decode=SpeculativeDecodingConfig(
            enabled=True,
            method="ngram",
            num_speculative_tokens=2,
            acceptance_trace_file=str(acceptance_trace),
        ),
    )
    config = dataclasses.replace(
        _config(tmp_path, case, sys_arch=sys_arch),
        request_generator_config=TraceRequestGeneratorConfig(trace_file=str(requests)),
    )
    simulator = Simulator(config)
    simulator.run()

    assert steps[-1] == (94, last_step_width), steps
    (request,) = simulator._all_requests
    assert request.completed
    assert request.num_emitted_decode_tokens == 16
