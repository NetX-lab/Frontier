"""A monolithic request preempted during decode resumes and completes.

vLLM v1 preemption discards a request's computed KV but keeps its generated
output. Three requests arrive together with KV for fewer tokens than they reach
together, so decode growth preempts a request that has finished its prefill.
The real `Simulator` loop must bring that request back and finish every request
with all of its output tokens.

With pipeline stages, the victim can also be preempted while an earlier batch
still carries it through a later stage, or after it finished but before deep PP
releases it. The pipelined cases below drive those shapes through the same loop.
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
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    UniformRequestLengthGeneratorConfig,
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

DENSE_REPLICA = dict(model_name="llama2_7b_dense_example")
MOE_DP2_EP2_REPLICA = dict(
    model_name="Qwen3-30B-A3B-tiny",
    attn_dp=2,
    moe_tensor_parallel_size=1,
    moe_expert_parallel_size=2,
    total_expert_num=16,
    router_topk=8,
)

# Poisson arrivals of 8-96 token requests into KV for a few of them, so decode
# growth preempts requests that an earlier batch still carries through a later
# stage. Each case failed before the fix named in its comment.
PIPELINED_CASES = {
    # The in-flight batch drops the victim at a later stage, whole or as a copy
    # without it. The victim's active mark used to outlive that batch, so the
    # running phase skipped the victim for good and the run stalled.
    "dense_pp4": dict(
        replica=DENSE_REPLICA, num_pipeline_stages=4, num_blocks=8,
        num_requests=6, seed=2,
    ),
    # The victim is preempted part-way through a decode step. The rest of the
    # stale step used to credit its layers, and the resumed step then overran
    # the layer count.
    "moe_dp2_ep2_pp2": dict(
        replica=MOE_DP2_EP2_REPLICA, num_pipeline_stages=2, num_blocks=8,
        num_requests=12, seed=42,
    ),
    # PP4 holds a finished request in running for one more iteration. Chosen
    # as the victim there, it used to re-enter the waiting queue.
    "dense_pp4_finished_victim": dict(
        replica=DENSE_REPLICA, num_pipeline_stages=4, num_blocks=10,
        num_requests=24, seed=33,
    ),
}


@pytest.fixture(autouse=True)
def _fresh_global_vars():
    # A run sets process-wide model flags once, and these cases mix dense and
    # MoE models in one process.
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def _config(root, replica_config, replica_scheduler_config, request_generator_config):
    return SimulationConfig(
        simulation_mode="online",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=ClusterConfig(
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
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
        request_generator_config=request_generator_config,
    )


def _trace_config(root):
    trace = root / "decode_preemption.csv"
    trace.write_text(TRACE)
    return _config(
        root,
        ReplicaConfig(
            model_name="llama2_7b_dense_example",
            device="a100",
            network_device="a100_pairwise_nvlink",
        ),
        VllmV1SchedulerConfig(
            num_blocks=8,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=64,
            enable_chunked_prefill=True,
        ),
        TraceRequestGeneratorConfig(trace_file=str(trace)),
    )


def _pipelined_config(root, case):
    return _config(
        root,
        ReplicaConfig(
            device="a100",
            network_device="a100_pairwise_nvlink",
            num_pipeline_stages=case["num_pipeline_stages"],
            attn_tensor_parallel_size=1,
            **case["replica"],
        ),
        VllmV1SchedulerConfig(
            num_blocks=case["num_blocks"],
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=16,
            enable_chunked_prefill=True,
        ),
        SyntheticRequestGeneratorConfig(
            num_requests=case["num_requests"],
            seed=case["seed"],
            length_generator_config=UniformRequestLengthGeneratorConfig(
                min_tokens=8,
                max_tokens=96,
                prefill_to_decode_ratio=4.0,
                seed=case["seed"],
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                qps=200.0, seed=case["seed"]
            ),
        ),
    )


def _observe_preemptions(monkeypatch):
    """Record each victim's state as the scheduler preempts it."""

    victims = []
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request

    def observed_preempt_request(self, victim, preempted_requests):
        victims.append(dict(
            request_id=victim.id,
            decoding=victim.is_prefill_complete,
            finished=victim.completed,
            in_flight=self._is_request_active_in_batch(victim),
            processed_before=victim.num_processed_tokens,
        ))
        preempt_request(self, victim, preempted_requests)
        victims[-1]["processed_after"] = victim.num_processed_tokens

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt_request
    )
    return victims


def _observe_rescheduling(monkeypatch):
    """Record each request scheduled while a batch in flight still runs it."""

    rescheduled = []
    in_flight = {}
    create_batch = VLLMv1EngineReplicaScheduler._create_batch
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end

    def observed_create_batch(self, requests, num_tokens):
        batches = in_flight.setdefault(id(self), {})
        running = {
            request.id
            for batch in batches.values()
            for request in batch.current_execution_requests
            if not request.completed
        }
        rescheduled.extend(request.id for request in requests if request.id in running)
        batch = create_batch(self, requests, num_tokens)
        batches[batch.id] = batch
        return batch

    def observed_on_batch_end(self, batch):
        in_flight[id(self)].pop(batch.id)
        on_batch_end(self, batch)

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_create_batch", observed_create_batch
    )
    monkeypatch.setattr(VLLMv1EngineReplicaScheduler, "on_batch_end", observed_on_batch_end)
    return rescheduled


def _assert_every_request_completes(simulator, num_requests):
    requests = list(simulator._all_requests)
    assert len(requests) == num_requests
    for request in requests:
        assert request.completed, request.id
        assert request.num_processed_decode_tokens == request.num_decode_tokens


def test_a_request_preempted_during_decode_resumes_and_completes(tmp_path, monkeypatch):
    victims = _observe_preemptions(monkeypatch)
    simulator = Simulator(_trace_config(tmp_path))
    simulator.run()

    # The run has to reach a preemption after prefill, or it proves nothing.
    decode_victims = [victim for victim in victims if victim["decoding"]]
    assert decode_victims
    for victim in decode_victims:
        assert victim["processed_after"] == victim["processed_before"], victim
    _assert_every_request_completes(simulator, 3)


@pytest.mark.parametrize("name", PIPELINED_CASES)
def test_a_pipelined_run_completes_every_request_across_decode_preemptions(
    name, tmp_path, monkeypatch
):
    case = PIPELINED_CASES[name]
    victims = _observe_preemptions(monkeypatch)
    rescheduled = _observe_rescheduling(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, case))
    simulator.run()

    # Each case has to reach its shape, or it proves nothing.
    if name == "dense_pp4_finished_victim":
        assert any(victim["finished"] for victim in victims), victims
    else:
        assert any(
            victim["decoding"] and victim["in_flight"] and not victim["finished"]
            for victim in victims
        ), victims
    # A resumed victim's new batch keeps it active: the end of the batch that
    # carried it before preemption must not release it a second time.
    assert rescheduled == []
    _assert_every_request_completes(simulator, case["num_requests"])
