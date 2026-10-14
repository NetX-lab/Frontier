"""A monolithic request preempted during decode resumes and completes.

vLLM v1 preemption discards a request's computed KV but keeps its generated
output. Three requests arrive together with KV for fewer tokens than they reach
together, so decode growth preempts a request that has finished its prefill.
The real `Simulator` loop must bring that request back and finish every request
with all of its output tokens.

With pipeline stages, the victim can also be preempted while an earlier batch
still carries it through a later stage, or after it finished but before deep PP
releases it. The pipelined cases below drive those shapes through the same loop.
A victim preempted with a step still in flight keeps that step's sample, as in
vLLM; Frontier applies it when that step ends, even if the victim was admitted
again before then.
"""

from __future__ import annotations

import collections
import dataclasses
from types import SimpleNamespace

import pytest

from frontier.config import global_vars
from frontier.config import (
    ClusterConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    RandomForrestExecutionTimePredictorConfig,
    ReplicaConfig,
    SglangSchedulerConfig,
    SimulationConfig,
    SpeculativeDecodingConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    UniformRequestLengthGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities.batch import Batch
from frontier.entities.request import Request
from frontier.errors import FrontierMemoryOOMError
from frontier.events import global_batch_end_event
from frontier.events.batch_stage_end_event import BatchStageEndEvent
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.metrics.metrics_store import MetricsStore
from frontier.scheduler.replica_scheduler.sglang_style_replica_scheduler import (
    SGLangStyleReplicaScheduler,
)
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
    ReplicaStageScheduler,
)
from frontier.simulator import Simulator
from frontier.types import ClusterType

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
DENSE_SPEC_DECODE_REPLICA = dict(
    DENSE_REPLICA,
    speculative_decoding_config=SpeculativeDecodingConfig(
        enabled=True,
        method="ngram",
        num_speculative_tokens=2,
        committed_tokens_per_iteration=2,
    ),
)

# Poisson arrivals of 8-96 token requests into KV for a few of them, so decode
# growth preempts requests that an earlier batch still carries through a later
# stage. Each comment names the fix its case guards. With prompt chunks
# scheduled while the previous chunk is in flight, the request being chunked
# is the usual victim; the seeds were picked to reach each decode shape.
PIPELINED_CASES = {
    # The in-flight batch drops the victim at a later stage, whole or as a copy
    # without it. The victim's active mark used to outlive that batch, so the
    # running phase skipped the victim for good and the run stalled.
    "dense_pp4": dict(
        replica=DENSE_REPLICA, num_pipeline_stages=4, num_blocks=8,
        num_requests=6, seed=1,
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
        num_requests=8, seed=7,
    ),
    # A later stage keeps only the live rows of a speculative decode step.
    # That copy used to keep the whole batch's draft plan, so its batch end
    # rejected the plan's length.
    "dense_pp4_spec_decode": dict(
        replica=DENSE_SPEC_DECODE_REPLICA, num_pipeline_stages=4, num_blocks=7,
        num_requests=6, seed=7,
    ),
}

# dense PP4, 8 blocks, 24 requests, seed 5, with token-proportional stage
# times. A schedule pass preempts a decode row of a batch it formed earlier
# before a stage runs that batch. Found by a probe over PP 2/4, 8-12 blocks,
# long-prefill thresholds 0/6 and seeds 0-5; a fixed dummy time never reaches it.
SCHEDULED_DECODE_VICTIM_CASE = dict(
    replica=DENSE_REPLICA,
    num_pipeline_stages=4,
    num_blocks=8,
    num_requests=24,
    seed=5,
)

# dense PP4, 6 blocks, 6 requests, seed 2. A victim is preempted again while
# its kept tokens are still being recomputed. Found on the first probe of that
# shape family (num_blocks in 6-12, num_requests in 6-24, seeds 2/7/33/42).
RECOMPUTE_RESTART_CASE = dict(
    replica=DENSE_REPLICA,
    num_pipeline_stages=4,
    num_blocks=6,
    num_requests=6,
    seed=2,
)


@pytest.fixture(autouse=True)
def _fresh_global_vars():
    # A run sets process-wide model flags once, and these cases mix dense and
    # MoE models in one process.
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def _config(root, cluster_config, request_generator_config, sys_arch="co-location"):
    return SimulationConfig(
        simulation_mode="online",
        sys_arch=sys_arch,
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=cluster_config,
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


def _colocation_cluster(replica_config, replica_scheduler_config):
    return ClusterConfig(
        replica_config=replica_config,
        replica_scheduler_config=replica_scheduler_config,
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True
        ),
    )


def _trace_config(
    root,
    *,
    trace_text=TRACE,
    num_blocks=8,
    max_tokens_in_batch=64,
    enable_chunked_prefill=True,
    long_prefill_token_threshold=0,
    enable_prefix_caching=False,
    scheduler_config_cls=VllmV1SchedulerConfig,
    scheduling_policy="fcfs",
    replica_config=None,
):
    trace = root / "decode_preemption.csv"
    trace.write_text(trace_text)
    if replica_config is None:
        replica_config = ReplicaConfig(
            model_name="llama2_7b_dense_example",
            device="a100",
            network_device="a100_pairwise_nvlink",
        )
    return _config(
        root,
        _colocation_cluster(
            replica_config,
            scheduler_config_cls(
                num_blocks=num_blocks,
                block_size=16,
                batch_size_cap=4,
                max_tokens_in_batch=max_tokens_in_batch,
                enable_chunked_prefill=enable_chunked_prefill,
                long_prefill_token_threshold=long_prefill_token_threshold,
                enable_prefix_caching=enable_prefix_caching,
                scheduling_policy=scheduling_policy,
            ),
        ),
        TraceRequestGeneratorConfig(trace_file=str(trace)),
    )


def _synthetic_requests(case):
    return SyntheticRequestGeneratorConfig(
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
    )


def _pipelined_config(root, case):
    return _config(
        root,
        _colocation_cluster(
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
        ),
        _synthetic_requests(case),
    )


def _pdd_config(root, case):
    """PD-disaggregation with pipeline stages on the unified decode replica."""

    return _config(
        root,
        ClusterConfig(
            prefill_cluster_num_replicas=case["num_prefill_replicas"],
            decode_cluster_num_replicas=1,
            replica_config=ReplicaConfig(
                device="a100",
                network_device="a100_pairwise_nvlink",
                attn_tensor_parallel_size=1,
                **case["replica"],
            ),
            decode_replica_config_num_pipeline_stages=case["num_pipeline_stages"],
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=64,
                block_size=16,
                batch_size_cap=case["batch_size_cap"],
                max_tokens_in_batch=16,
                enable_chunked_prefill=True,
            ),
            decode_replica_scheduler_config_num_blocks=case["num_blocks"],
            decode_replica_scheduler_config_max_tokens_in_batch=16,
            execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
                enable_dummy_mode=True
            ),
        ),
        _synthetic_requests(case),
        sys_arch="pd-disaggregation",
    )


def _observe_preemptions(monkeypatch):
    """Record each victim's state as the scheduler preempts it."""

    victims = []
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request

    def observed_preempt_request(self, victim, preempted_requests):
        victims.append(dict(
            request_id=victim.id,
            decoding=victim.is_decoding,
            finished=victim.completed,
            in_flight=self._is_request_active_in_batch(victim),
            processed_before=victim.num_processed_tokens,
            cluster_type=self._cluster_type,
            layers_before=victim.completed_layer_count,
        ))
        preempt_request(self, victim, preempted_requests)
        victims[-1]["processed_after"] = victim.num_processed_tokens
        victims[-1]["recomputing"] = victim.is_recomputing

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt_request
    )
    return victims


def _observe_rescheduling(monkeypatch):
    """Record each request scheduled while a batch in flight runs its decode step.

    A prompt chunk is scheduled while the previous chunk is in flight, as in
    vLLM; a decode step waits for the one in flight.
    """

    rescheduled = []
    in_flight = {}
    create_batch = VLLMv1EngineReplicaScheduler._create_batch
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end

    def observed_create_batch(self, requests, num_tokens):
        batches = in_flight.setdefault(id(self), {})
        decoding = {
            request.id
            for batch, decoding_ids in batches.values()
            for request in batch.current_execution_requests
            if request.id in decoding_ids and not request.completed
        }
        rescheduled.extend(request.id for request in requests if request.id in decoding)
        decoding_ids = {request.id for request in requests if request.is_decoding}
        batch = create_batch(self, requests, num_tokens)
        batches[batch.id] = (batch, decoding_ids)
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


def _observe_recompute(monkeypatch):
    """Record preemptions, scheduled rows, and batch-end logical lengths.

    Rows come from the arguments of `_create_batch`. The cursor is snapshotted
    after preemption for the post-run assertion; row classification uses the
    logical length and whether prefill has completed.
    """

    events = []
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request
    create_batch = VLLMv1EngineReplicaScheduler._create_batch
    on_batch_end = VLLMv1EngineReplicaScheduler.on_batch_end

    def observed_preempt_request(self, victim, preempted_requests):
        processed_before = victim.num_processed_tokens
        decoding = victim.is_prefill_complete
        finished = victim.completed
        prefill_completed_at = victim.prefill_completed_at
        first_decode_at = victim.first_decode_token_completed_at
        cached = victim.num_prefill_tokens_cached
        spec_iterations = victim.spec_total_iterations
        preempt_request(self, victim, preempted_requests)
        events.append((
            "preempt",
            {
                "request_id": victim.id,
                "decoding": decoding,
                "finished": finished,
                "processed_before": processed_before,
                "processed_after": victim.num_processed_tokens,
                "prefill_completed_at": prefill_completed_at,
                "first_decode_at": first_decode_at,
                "cached": cached,
                "spec_iterations": spec_iterations,
                "cursor_after": getattr(victim, "_num_recomputed_tokens", None),
            },
        ))

    def observed_create_batch(self, requests, num_tokens):
        batch = create_batch(self, requests, num_tokens)
        metadata = batch.spec_decode_metadata
        for index, (request, width) in enumerate(zip(requests, num_tokens)):
            planned = verify = committed = None
            if metadata is not None:
                planned = int(metadata.planned_draft_tokens_per_request[index])
                verify = int(metadata.verify_tokens_per_request[index])
                committed = int(metadata.committed_tokens_per_request[index])
            events.append((
                "row",
                {
                    "batch_id": batch.id,
                    "request_id": request.id,
                    "width": int(width),
                    "processed": request.num_processed_tokens,
                    "prefill_complete": request.is_prefill_complete,
                    "planned": planned,
                    "verify": verify,
                    "committed": committed,
                    "spec_iterations": request.spec_total_iterations,
                },
            ))
        return batch

    def observed_on_batch_end(self, batch):
        on_batch_end(self, batch)
        for request in batch.requests:
            events.append((
                "end",
                {
                    "batch_id": batch.id,
                    "request_id": request.id,
                    "processed": request.num_processed_tokens,
                    "prefill_completed_at": request.prefill_completed_at,
                    "first_decode_at": request.first_decode_token_completed_at,
                    "cached": request.num_prefill_tokens_cached,
                    "spec_iterations": request.spec_total_iterations,
                },
            ))

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt_request
    )
    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_create_batch", observed_create_batch
    )
    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "on_batch_end", observed_on_batch_end
    )
    return events


def _recompute_episodes(events):
    """Rows of one victim from a post-prefill preemption until it commits or is preempted again."""

    open_episodes = {}
    episodes = []
    for kind, payload in events:
        request_id = payload["request_id"]
        if kind == "preempt":
            if request_id in open_episodes:
                episode = open_episodes.pop(request_id)
                episode["ended_by"] = "preempt"
                episodes.append(episode)
            if payload["decoding"] and not payload["finished"]:
                open_episodes[request_id] = {
                    "request_id": request_id,
                    "processed_before": payload["processed_before"],
                    "prefill_completed_at": payload["prefill_completed_at"],
                    "first_decode_at": payload["first_decode_at"],
                    "cached": payload["cached"],
                    "spec_iterations": payload["spec_iterations"],
                    "rows": [],
                    "ended_by": None,
                }
            continue
        episode = open_episodes.get(request_id)
        if episode is None:
            continue
        if kind == "row":
            episode["rows"].append(dict(payload))
            continue
        if not episode["rows"] or episode["rows"][-1]["batch_id"] != payload["batch_id"]:
            continue
        if episode["rows"][-1].get("processed_after") is not None:
            continue
        episode["rows"][-1]["processed_after"] = payload["processed"]
        episode["rows"][-1]["prefill_completed_at_after"] = payload["prefill_completed_at"]
        episode["rows"][-1]["first_decode_at_after"] = payload["first_decode_at"]
        episode["rows"][-1]["cached_after"] = payload["cached"]
        episode["rows"][-1]["spec_iterations_after"] = payload["spec_iterations"]
        if payload["processed"] != episode["processed_before"]:
            episode["ended_by"] = "commit"
            episodes.append(open_episodes.pop(request_id))
    return episodes


def _assert_completed_recompute(episode, *, max_width):
    rows = episode["rows"]
    assert rows, episode
    assert all(row.get("processed_after") is not None for row in rows), episode
    widths = [row["width"] for row in rows]
    assert sum(widths) == episode["processed_before"], episode
    assert all(0 < width <= max_width for width in widths), episode
    for row in rows[:-1]:
        assert row["processed_after"] == episode["processed_before"], row
    assert rows[-1]["processed_after"] == episode["processed_before"] + 1, rows[-1]
    assert rows[-1]["prefill_completed_at_after"] == episode["prefill_completed_at"]
    assert rows[-1]["first_decode_at_after"] == episode["first_decode_at"]


def test_a_decode_victim_recomputes_its_kept_tokens(tmp_path, monkeypatch):
    events = _observe_recompute(monkeypatch)
    simulator = Simulator(_trace_config(tmp_path))
    simulator.run()

    episodes = [
        episode
        for episode in _recompute_episodes(events)
        if episode["ended_by"] == "commit"
    ]
    assert episodes
    for episode in episodes:
        _assert_completed_recompute(episode, max_width=64)
    _assert_every_request_completes(simulator, 3)


def test_a_recompute_restarts_when_preempted_again(tmp_path, monkeypatch):
    events = _observe_recompute(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, RECOMPUTE_RESTART_CASE))
    simulator.run()

    open_processed = {}
    restarts = []
    for kind, payload in events:
        if kind == "end" and payload["request_id"] in open_processed:
            if payload["processed"] != open_processed[payload["request_id"]]:
                del open_processed[payload["request_id"]]
            continue
        if kind != "preempt" or not payload["decoding"] or payload["finished"]:
            continue
        request_id = payload["request_id"]
        if (
            request_id in open_processed
            and payload["processed_before"] == open_processed[request_id]
        ):
            restarts.append(payload)
        open_processed[request_id] = payload["processed_before"]

    assert restarts
    for restart in restarts:
        assert restart["processed_after"] == restart["processed_before"], restart
        assert restart["cursor_after"] == 0, restart
    _assert_every_request_completes(simulator, RECOMPUTE_RESTART_CASE["num_requests"])


def test_a_recompute_chunk_respects_the_long_prefill_threshold(tmp_path, monkeypatch):
    events = _observe_recompute(monkeypatch)
    simulator = Simulator(
        _trace_config(tmp_path, long_prefill_token_threshold=16)
    )
    simulator.run()

    episodes = [
        episode
        for episode in _recompute_episodes(events)
        if episode["ended_by"] == "commit"
    ]
    assert episodes
    for episode in episodes:
        _assert_completed_recompute(episode, max_width=16)
        assert episode["processed_before"] > 16, episode
    _assert_every_request_completes(simulator, 3)


def test_a_recompute_is_one_chunk_without_chunked_prefill(tmp_path, monkeypatch):
    events = _observe_recompute(monkeypatch)
    simulator = Simulator(
        _trace_config(
            tmp_path,
            enable_chunked_prefill=False,
            max_tokens_in_batch=128,
        )
    )
    simulator.run()

    episodes = [
        episode
        for episode in _recompute_episodes(events)
        if episode["ended_by"] == "commit"
    ]
    assert episodes
    for episode in episodes:
        assert len(episode["rows"]) == 1, episode
        assert episode["rows"][0]["width"] == episode["processed_before"], episode
        _assert_completed_recompute(episode, max_width=128)
    _assert_every_request_completes(simulator, 3)


# A 16-token prefill chunk and a 32-token budget. Once the pool is full, the
# first running request is the priority victim of its own next chunk: the pass
# preempts it, forms no rows and leaves a higher-priority request running with
# no batch in flight. The PP1 case admits the lower-priority request first; the
# PP2 case reaches the state after eight earlier preempting passes, because the
# engine batch queue holds most passes until the oldest batch ends. Both stalled
# before the follow-up poll; a grid over pool size, budget, arrival order and PP
# found the first, and a random search over priority traces the second.
EMPTY_PREEMPTING_PASS_CASES = {
    "pp1_lower_priority_first": dict(
        trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.0,48,1,1
0.2,48,1,0
""",
        num_blocks=3,
        num_pipeline_stages=1,
    ),
    "pp2_after_repeated_preemptions": dict(
        trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.0,27,1,1
0.724428,54,7,0
2.867778,62,1,1
3.702085,59,7,2
4.628865,59,1,2
7.829212,28,4,0
8.613820,16,5,2
8.759862,38,1,2
""",
        num_blocks=6,
        num_pipeline_stages=2,
    ),
}


def _priority_chunk_config(root, *, trace_text, num_blocks, num_pipeline_stages):
    return _trace_config(
        root,
        trace_text=trace_text,
        num_blocks=num_blocks,
        max_tokens_in_batch=32,
        long_prefill_token_threshold=16,
        scheduling_policy="priority",
        replica_config=ReplicaConfig(
            model_name="llama2_7b_dense_example",
            device="a100",
            network_device="a100_pairwise_nvlink",
            num_pipeline_stages=num_pipeline_stages,
            attn_tensor_parallel_size=1,
        ),
    )


def _observe_empty_preempting_passes(monkeypatch):
    """The replica state after each MONOLITHIC pass that preempts and forms no batch."""

    passes = []
    pass_victims = []
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request
    schedule_two_phase = VLLMv1EngineReplicaScheduler._schedule_two_phase

    def observed_preempt_request(self, victim, preempted_requests):
        pass_victims.append(victim.id)
        preempt_request(self, victim, preempted_requests)

    def observed_schedule_two_phase(self):
        pass_victims.clear()
        batch = schedule_two_phase(self)
        if batch is None and pass_victims:
            passes.append(dict(
                victims=list(pass_victims),
                running=[request.id for request in self._running_requests],
                batches_in_flight=self._num_running_batches,
            ))
        return batch

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt_request
    )
    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_schedule_two_phase", observed_schedule_two_phase
    )
    return passes


@pytest.mark.parametrize("case", EMPTY_PREEMPTING_PASS_CASES)
def test_an_empty_pass_that_preempts_runs_the_next_step_at_once(
    tmp_path, monkeypatch, case
):
    passes = _observe_empty_preempting_passes(monkeypatch)
    trace_text = EMPTY_PREEMPTING_PASS_CASES[case]["trace_text"]
    simulator = Simulator(
        _priority_chunk_config(tmp_path, **EMPTY_PREEMPTING_PASS_CASES[case])
    )
    simulator.run()

    assert any(
        observed["running"] and observed["batches_in_flight"] == 0
        for observed in passes
    ), passes
    _assert_every_request_completes(simulator, len(trace_text.splitlines()) - 1)
    (replica_scheduler,) = simulator._global_scheduler.get_cluster_scheduler(
        ClusterType.MONOLITHIC
    )._full_stage_replica_schedulers.values()
    assert replica_scheduler._allocation_map == {}
    assert replica_scheduler._num_allocated_blocks == 0


@pytest.mark.parametrize("num_pipeline_stages", [1, 2, 4])
def test_a_request_larger_than_the_kv_pool_is_refused(tmp_path, num_pipeline_stages):
    # vLLM refuses to start when its KV cache cannot hold a max_model_len
    # request. This request's 64-token context needs 4 of the 3 blocks; it
    # would otherwise preempt itself and be admitted again without end.
    simulator = Simulator(
        _priority_chunk_config(
            tmp_path,
            trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.0,64,1,0
""",
            num_blocks=3,
            num_pipeline_stages=num_pipeline_stages,
        )
    )

    with pytest.raises(
        FrontierMemoryOOMError, match="needs 4 KV blocks for its 64-token context"
    ):
        simulator.run()


def test_a_request_whose_context_fills_the_kv_pool_completes(tmp_path):
    # 63 prompt tokens and 2 output tokens: the last output token is never
    # written to the cache, so the context peaks at the pool's 64 tokens.
    simulator = Simulator(
        _priority_chunk_config(
            tmp_path,
            trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.0,63,2,0
""",
            num_blocks=4,
            num_pipeline_stages=2,
        )
    )
    simulator.run()

    _assert_every_request_completes(simulator, 1)


@pytest.mark.parametrize("num_pipeline_stages", [1, 2])
def test_a_priority_tie_preempts_the_later_arrival(
    tmp_path, monkeypatch, num_pipeline_stages
):
    # Both requests share one trace timestamp and priority, and the pool holds
    # only one of them. vLLM stamps each arrival, so the second row is the later
    # arrival and every victim; the first row finishes before it resumes.
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request

    def observed_preempt_request(self, victim, preempted_requests):
        assert victim.num_prefill_tokens == 56, victim.id
        preempt_request(self, victim, preempted_requests)

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt_request
    )
    simulator = Simulator(
        _priority_chunk_config(
            tmp_path,
            trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.0,40,1,0
0.0,56,3,0
""",
            num_blocks=4,
            num_pipeline_stages=num_pipeline_stages,
        )
    )
    simulator.run()

    _assert_every_request_completes(simulator, 2)


PREFIX_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids
0.0,32,30,7,11|22
0.0,32,30,7,11|22
0.0,32,30,7,11|22
"""


def test_a_recompute_uses_whole_prompt_blocks_from_the_prefix_cache(
    tmp_path, monkeypatch
):
    events = _observe_recompute(monkeypatch)
    simulator = Simulator(
        _trace_config(
            tmp_path,
            trace_text=PREFIX_TRACE,
            num_blocks=4,
            enable_prefix_caching=True,
        )
    )
    simulator.run()

    episodes = [
        episode
        for episode in _recompute_episodes(events)
        if episode["ended_by"] == "commit"
    ]
    assert episodes
    starts = []
    for episode in episodes:
        rows = episode["rows"]
        assert rows, episode
        start_context = episode["processed_before"] - sum(row["width"] for row in rows)
        starts.append(start_context)
        assert start_context > 0, episode
        assert start_context < episode["processed_before"], episode
        assert start_context % 16 == 0, episode
        assert rows[-1]["cached_after"] == episode["cached"], episode
    # A one-token resume starts at processed_before - 1. A cached prompt block
    # starts strictly earlier, so at least one episode must.
    assert any(
        start < episode["processed_before"] - 1
        for start, episode in zip(starts, episodes)
    )
    _assert_every_request_completes(simulator, 3)


# The hashes name three blocks; each 20-token prompt fills one. Decode fills
# the other two, so they keep no hash, and a recompute hits only the prompt
# block.
LONG_HASH_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens,session_id,block_hash_ids
0.0,20,30,1,11|22|33
0.0,20,30,1,11|22|33
0.0,20,30,1,11|22|33
"""


def test_a_recompute_hits_only_the_blocks_of_its_prompt(tmp_path, monkeypatch):
    hits = []
    on_cache_hit = Request.on_cache_hit

    def observed_on_cache_hit(self, num_tokens_cached):
        hits.append((self.is_recomputing, num_tokens_cached))
        on_cache_hit(self, num_tokens_cached)

    monkeypatch.setattr(Request, "on_cache_hit", observed_on_cache_hit)
    simulator = Simulator(
        _trace_config(
            tmp_path,
            trace_text=LONG_HASH_TRACE,
            num_blocks=4,
            enable_prefix_caching=True,
        )
    )
    simulator.run()

    assert (True, 16) in hits, hits
    assert max(cached for _, cached in hits) == 16, hits
    _assert_every_request_completes(simulator, 3)

def test_a_recompute_step_carries_no_drafts(tmp_path, monkeypatch):
    events = _observe_recompute(monkeypatch)
    simulator = Simulator(
        _trace_config(
            tmp_path,
            replica_config=ReplicaConfig(
                model_name="llama2_7b_dense_example",
                device="a100",
                network_device="a100_pairwise_nvlink",
                speculative_decoding_config=SpeculativeDecodingConfig(
                    enabled=True,
                    method="ngram",
                    num_speculative_tokens=2,
                    committed_tokens_per_iteration=2,
                ),
            ),
        )
    )
    simulator.run()

    episodes = [
        episode
        for episode in _recompute_episodes(events)
        if episode["ended_by"] == "commit"
    ]
    assert episodes
    for episode in episodes:
        for row in episode["rows"]:
            if row["planned"] is None:
                continue
            assert row["planned"] == 0, row
            assert row["verify"] == 0, row
            assert row["committed"] == row["width"], row
        assert episode["rows"][-1]["spec_iterations_after"] == episode["spec_iterations"]
        _assert_completed_recompute(episode, max_width=64)
    _assert_every_request_completes(simulator, 3)


def test_an_sglang_victim_recomputes_its_kept_tokens(tmp_path, monkeypatch):
    events = _observe_recompute(monkeypatch)
    simulator = Simulator(
        _trace_config(tmp_path, scheduler_config_cls=SglangSchedulerConfig)
    )
    simulator.run()

    episodes = [
        episode
        for episode in _recompute_episodes(events)
        if episode["ended_by"] == "commit"
    ]
    assert episodes
    for episode in episodes:
        assert episode["rows"][0]["width"] > 1, episode
    _assert_every_request_completes(simulator, 3)


def _observe_prefill_first_victims(monkeypatch):
    """Where each victim of an SGLang prefill-first pass sits once that pass forms no rows."""

    victims = []
    schedule_prefill_stage_first = (
        SGLangStyleReplicaScheduler._schedule_prefill_stage_first
    )

    def observed_schedule_prefill_stage_first(
        self, token_budget, preempted_requests
    ):
        result = schedule_prefill_stage_first(self, token_budget, preempted_requests)
        _, waiting_scheduled, _, running_scheduled, _ = result
        if not waiting_scheduled and not running_scheduled:
            waiting = [*self._preempted_requests, *self._request_queue]
            victims.extend(
                dict(
                    request_id=victim.id,
                    in_running=victim in self._running_requests,
                    waiting_entries=waiting.count(victim),
                    blocks=self._allocation_map.get(victim.id, 0),
                )
                for victim in preempted_requests
            )
        return result

    monkeypatch.setattr(
        SGLangStyleReplicaScheduler,
        "_schedule_prefill_stage_first",
        observed_schedule_prefill_stage_first,
    )
    return victims


@pytest.mark.parametrize("scheduling_policy", ["fcfs", "priority"])
def test_an_sglang_prefill_victim_waits_when_its_pass_forms_no_rows(
    tmp_path, monkeypatch, scheduling_policy
):
    # A 16-token budget makes a recomputing victim resume in chunks, so its
    # next chunk can preempt it inside the prefill-only view while the
    # decoding requests hold the rest of the pool. The pass then forms no
    # rows, and the victim must stay freed and queued, as it was preempted.
    victims = _observe_prefill_first_victims(monkeypatch)
    simulator = Simulator(
        _trace_config(
            tmp_path,
            max_tokens_in_batch=16,
            scheduler_config_cls=SglangSchedulerConfig,
            scheduling_policy=scheduling_policy,
        )
    )
    simulator.run()

    assert victims
    for victim in victims:
        assert victim == dict(
            request_id=victim["request_id"],
            in_running=False,
            waiting_entries=1,
            blocks=0,
        )
    _assert_every_request_completes(simulator, 3)
    for request in simulator._all_requests:
        assert not request._is_waiting[ClusterType.MONOLITHIC], request.id


class _InflightRemovals:
    """A preempted request's in-flight row at its step's end, and its state after.

    The step runs every stage whole and its end applies the sample, so the
    state is read after the batch-end event returns. A victim admitted again
    before its row ends is covered by the readmission tests instead. A
    prefill or recompute victim can have several chunks in
    flight; the rows before the one that completes its tokens take no sample
    and are skipped.
    """

    def __init__(self):
        self.removals = []
        self.rows = []
        self.end_ids = []
        self.open = {}
        self._pending = {}

    def note(self, batch, index):
        request = batch.requests[index]
        episode = self.open.get(request.id)
        if episode is None or request.id in self._pending:
            return
        width = int(batch.num_tokens[index])
        context = int(batch.num_context_tokens[index])
        if not episode["decoding"]:
            sample_seq_len = (
                episode["processed_before"]
                if episode["prefill_complete"]
                else request.num_prefill_tokens
            )
            if context + width < sample_seq_len:
                return
        metadata = batch.spec_decode_metadata
        self._pending[request.id] = dict(
            width=width,
            context=context,
            committed=(
                width
                if metadata is None
                else int(metadata.committed_tokens_per_request[index])
            ),
            spec=metadata is not None,
        )

    def finish(self, event, scheduler):
        if not self._pending:
            return
        replica_scheduler = scheduler.get_cluster_scheduler(
            event._cluster_type
        ).get_replica_scheduler(event._replica_id, event._replica_local_id)
        waiting = [
            *replica_scheduler._request_queue,
            *replica_scheduler._preempted_requests,
            *replica_scheduler._waiting_requests,
        ]
        for request_id, removal in self._pending.items():
            episode = self.open.pop(request_id)
            request = episode["request"]
            episode.update(
                removal,
                time=event.time,
                processed_after=request.num_processed_tokens,
                cursor_after=request._num_recomputed_tokens,
                recomputing=request.is_recomputing,
                completed_after=request.completed,
                prefill_completed_at=request._prefill_completed_at,
                first_decode_at=request.first_decode_token_completed_at,
                in_waiting_after=request in waiting,
                num_rows_before=len(self.rows),
            )
            self.removals.append(episode)
        self._pending.clear()

    def rows_after(self, removal):
        return [
            row
            for row in self.rows[removal["num_rows_before"]:]
            if row["request_id"] == removal["request"].id
        ]


def _observe_inflight_removals(monkeypatch):
    recorder = _InflightRemovals()
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request
    create_batch = VLLMv1EngineReplicaScheduler._create_batch
    global_handle = GlobalBatchEndEvent.handle_event
    on_request_end = MetricsStore._on_request_end

    def observed_preempt(self, victim, preempted_requests):
        episode = dict(
            request=victim,
            decoding=victim.is_decoding,
            prefill_complete=victim.is_prefill_complete,
            processed_before=victim.num_processed_tokens,
        )
        in_flight = self._is_request_active_in_batch(victim) and not victim.completed
        preempt_request(self, victim, preempted_requests)
        if in_flight:
            recorder.open[victim.id] = episode

    def observed_create_batch(self, requests, num_tokens):
        batch = create_batch(self, requests, num_tokens)
        for request, width in zip(requests, num_tokens):
            recorder.open.pop(request.id, None)
            recorder.rows.append(dict(
                request_id=request.id,
                width=int(width),
                processed=request.num_processed_tokens,
                recomputing=request.is_recomputing,
                cursor=request._num_recomputed_tokens,
            ))
        return batch

    def observed_global(self, scheduler, metrics_store):
        for index, request in enumerate(self._batch.requests):
            signature = Batch._get_request_execution_signature(request)
            if signature != self._request_execution_signatures[index]:
                recorder.note(self._batch, index)
        events = global_handle(self, scheduler, metrics_store)
        recorder.finish(self, scheduler)
        return events

    def observed_end(self, time, request):
        recorder.end_ids.append(request.id)
        return on_request_end(self, time, request)

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt
    )
    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_create_batch", observed_create_batch
    )
    monkeypatch.setattr(GlobalBatchEndEvent, "handle_event", observed_global)
    monkeypatch.setattr(MetricsStore, "_on_request_end", observed_end)
    return recorder


def _decode_removals(recorder):
    return [removal for removal in recorder.removals if removal["decoding"]]


def _assert_gains_the_committed_tokens(removal):
    expected = removal["processed_before"] + removal["committed"]
    assert removal["processed_after"] == expected, removal


def _assert_resumes_with_a_recompute(recorder, removal):
    assert removal["recomputing"] and removal["cursor_after"] == 0, removal
    rows = recorder.rows_after(removal)
    assert rows, removal
    assert rows[0]["recomputing"] and rows[0]["cursor"] == 0, rows[0]
    assert rows[0]["processed"] == removal["processed_after"], rows[0]


# A decode victim's row is still in flight when the pass preempts it, at PP4
# and at PP2. A step now runs every stage whole, so the PP4 victim is chosen
# where its row ends before it is admitted again (a probe over 6-12 blocks,
# 12-48 requests and 24 seeds).
INFLIGHT_DECODE_CASES = {
    "dense_pp4": dict(
        replica=DENSE_REPLICA,
        num_pipeline_stages=4,
        num_blocks=8,
        num_requests=12,
        seed=1,
    ),
    "dense_pp2": dict(
        replica=DENSE_REPLICA,
        num_pipeline_stages=2,
        num_blocks=8,
        num_requests=24,
        seed=0,
    ),
}

# A victim's final prompt chunk is in flight behind an earlier chunk of the
# same prompt, and that row ends before the victim is admitted again.
FINAL_CHUNK_CASE = dict(
    replica=DENSE_REPLICA,
    num_pipeline_stages=4,
    num_blocks=10,
    num_requests=24,
    seed=9,
)

# A victim's in-flight decode sample reaches its length stop.
LENGTH_STOP_CASE = dict(
    replica=DENSE_REPLICA,
    num_pipeline_stages=2,
    num_blocks=12,
    num_requests=24,
    seed=45,
)

# A decode victim on the unified PDD decode replica. Every running decode fits
# one step, so each pass forms one batch and the engine blocks on it: no victim
# is in flight when a later pass preempts it.
PDD_PREEMPTION_CASE = dict(
    replica=DENSE_REPLICA,
    num_prefill_replicas=1,
    batch_size_cap=4,
    num_pipeline_stages=2,
    num_blocks=6,
    num_requests=8,
    seed=7,
)

# Four prefill replicas keep more 3-token speculative decodes running than one
# 16-token step holds, so the decodes split over two in-flight batches. When the
# older batch ends, the next pass can preempt a request of the other one. Found
# by a probe over 1-4 prefill replicas, 24-48 requests, 20-28 blocks and seeds 0-3.
PDD_INFLIGHT_CASE = dict(
    replica=DENSE_SPEC_DECODE_REPLICA,
    num_prefill_replicas=4,
    batch_size_cap=16,
    num_pipeline_stages=2,
    num_blocks=28,
    num_requests=48,
    seed=1,
)

# MoE PDD with speculative decodes split over two in-flight batches on a PP2
# decode replica. A decode victim is preempted after the first stage credited
# its layers, while its old batch still runs the second stage and ends there.
PDD_MOE_INFLIGHT_CASE = dict(
    replica=dict(
        model_name="Qwen3-30B-A3B-tiny",
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        total_expert_num=16,
        router_topk=8,
        speculative_decoding_config=DENSE_SPEC_DECODE_REPLICA[
            "speculative_decoding_config"
        ],
    ),
    num_prefill_replicas=4,
    batch_size_cap=16,
    num_pipeline_stages=2,
    num_blocks=28,
    num_requests=48,
    seed=1,
)

# Each request thinks one 48-token round with a one-token answer first. A
# request is preempted while the last chunk of that round is in flight, and
# the step's end applies the sample that ends the round. A random search over
# short priority traces found this one.
THINKING_INFLIGHT_CASE = dict(
    trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.28,16,2,2
0.34,32,1,2
""",
    num_blocks=4,
    num_pipeline_stages=4,
)


@pytest.mark.parametrize("name", INFLIGHT_DECODE_CASES)
def test_an_inflight_decode_victim_keeps_the_sample_of_its_removed_row(
    name, tmp_path, monkeypatch
):
    case = INFLIGHT_DECODE_CASES[name]
    recorder = _observe_inflight_removals(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, case))
    simulator.run()

    removals = _decode_removals(recorder)
    assert removals, recorder.removals
    for removal in removals:
        _assert_gains_the_committed_tokens(removal)
        if not removal["completed_after"]:
            _assert_resumes_with_a_recompute(recorder, removal)
    _assert_every_request_completes(simulator, case["num_requests"])


def test_an_inflight_final_prefill_chunk_grants_the_first_token_and_recomputes(
    tmp_path, monkeypatch
):
    recorder = _observe_inflight_removals(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, FINAL_CHUNK_CASE))
    simulator.run()

    finals = [
        removal
        for removal in recorder.removals
        if not removal["prefill_complete"]
        and removal["context"] + removal["width"]
        >= removal["request"].num_prefill_tokens
    ]
    assert finals, recorder.removals
    # The final chunk was scheduled while an earlier chunk was in flight, and
    # that chunk's end took no sample.
    assert any(
        removal["context"] > removal["processed_before"] for removal in finals
    ), finals
    for removal in finals:
        assert removal["processed_after"] == removal["request"].num_prefill_tokens + 1
        assert removal["prefill_completed_at"] == removal["time"], removal
        assert removal["first_decode_at"] == removal["time"], removal
        _assert_resumes_with_a_recompute(recorder, removal)
    _assert_every_request_completes(simulator, FINAL_CHUNK_CASE["num_requests"])


def test_a_victim_that_stops_on_its_inflight_sample_leaves_waiting(
    tmp_path, monkeypatch
):
    recorder = _observe_inflight_removals(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, LENGTH_STOP_CASE))
    simulator.run()

    stopped = [
        removal for removal in _decode_removals(recorder) if removal["completed_after"]
    ]
    assert stopped, _decode_removals(recorder)
    for removal in stopped:
        request = removal["request"]
        _assert_gains_the_committed_tokens(removal)
        assert removal["processed_after"] == request.total_tokens, removal
        assert not removal["in_waiting_after"], removal
        assert recorder.rows_after(removal) == [], removal
        assert recorder.end_ids.count(request.id) == 1, removal
    _assert_every_request_completes(simulator, LENGTH_STOP_CASE["num_requests"])


def test_an_inflight_sample_that_ends_a_thinking_round_requeues_it_once(
    tmp_path, monkeypatch
):
    sample_stops = []
    requeues = []
    ends = []
    apply_samples = Batch.apply_preempted_step_samples
    requeue_event = global_batch_end_event.thinking_round_requeue_event
    on_request_end = MetricsStore._on_request_end

    def observed_apply_samples(self, time, cluster_type, **signatures):
        stopped = apply_samples(self, time, cluster_type, **signatures)
        sample_stops.extend((request.id, time) for _, request in stopped)
        return stopped

    def observed_requeue_event(time, request, round_started_at):
        event = requeue_event(time, request, round_started_at)
        if event is not None:
            requeues.append((request.id, time))
        return event

    def observed_request_end(self, time, request):
        ends.append(request.id)
        return on_request_end(self, time, request)

    monkeypatch.setattr(Batch, "apply_preempted_step_samples", observed_apply_samples)
    monkeypatch.setattr(
        global_batch_end_event, "thinking_round_requeue_event", observed_requeue_event
    )
    monkeypatch.setattr(MetricsStore, "_on_request_end", observed_request_end)
    simulator = Simulator(
        dataclasses.replace(
            _priority_chunk_config(tmp_path, **THINKING_INFLIGHT_CASE),
            enable_thinking_mode=True,
            thinking_depth=2,
            tool_call_latency=0.01,
            thinking_round_prefill_tokens=[48],
            thinking_round_decode_tokens=[1],
        )
    )
    simulator.run()

    assert set(sample_stops) & set(requeues), (sample_stops, requeues)
    requests = list(simulator._all_requests)
    assert collections.Counter(request_id for request_id, _ in requeues) == {
        request.id: 1 for request in requests
    }
    assert sorted(ends) == sorted(request.id for request in requests)
    _assert_every_request_completes(
        simulator, len(THINKING_INFLIGHT_CASE["trace_text"].splitlines()) - 1
    )


def test_a_pdd_decode_victim_resumes_with_one_token(tmp_path, monkeypatch):
    victims = _observe_preemptions(monkeypatch)
    simulator = Simulator(_pdd_config(tmp_path, PDD_PREEMPTION_CASE))
    simulator.run()

    # The decode role never recomputes: a victim keeps its tokens and resumes
    # with its next decode step.
    decode_victims = [victim for victim in victims if victim["decoding"]]
    assert decode_victims, victims
    for victim in decode_victims:
        assert victim["processed_after"] == victim["processed_before"], victim
        assert not victim["recomputing"], victim
    _assert_every_request_completes(simulator, PDD_PREEMPTION_CASE["num_requests"])


def test_an_inflight_pdd_decode_victim_resumes_with_a_decode_step(
    tmp_path, monkeypatch
):
    recorder = _observe_inflight_removals(monkeypatch)
    simulator = Simulator(_pdd_config(tmp_path, PDD_INFLIGHT_CASE))
    simulator.run()

    resumed = [
        removal
        for removal in _decode_removals(recorder)
        if not removal["completed_after"]
    ]
    assert resumed, _decode_removals(recorder)
    for removal in resumed:
        _assert_gains_the_committed_tokens(removal)
        assert not removal["recomputing"] and removal["cursor_after"] is None
        rows = recorder.rows_after(removal)
        assert rows, removal
        # A decode step, never wider than the one the victim had in flight.
        assert rows[0]["width"] <= removal["width"], rows[0]
        assert not rows[0]["recomputing"], rows[0]
        assert rows[0]["processed"] == removal["processed_after"], rows[0]
    _assert_every_request_completes(simulator, PDD_INFLIGHT_CASE["num_requests"])


def test_a_pdd_moe_victim_preempted_mid_step_lets_its_old_batch_end(
    tmp_path, monkeypatch
):
    victims = _observe_preemptions(monkeypatch)
    simulator = Simulator(_pdd_config(tmp_path, PDD_MOE_INFLIGHT_CASE))
    simulator.run()

    num_layers = simulator._config.cluster_config.replica_config.model_config.num_layers
    mid_step = [
        victim
        for victim in victims
        if victim["cluster_type"] == ClusterType.DECODE
        and 0 < victim["layers_before"] < num_layers
    ]
    assert mid_step, victims
    _assert_every_request_completes(simulator, PDD_MOE_INFLIGHT_CASE["num_requests"])


def test_an_inflight_speculative_victim_gains_its_rows_committed_tokens(
    tmp_path, monkeypatch
):
    case = PIPELINED_CASES["dense_pp4_spec_decode"]
    recorder = _observe_inflight_removals(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, case))
    simulator.run()

    removals = [removal for removal in _decode_removals(recorder) if removal["spec"]]
    assert removals, _decode_removals(recorder)
    for removal in removals:
        _assert_gains_the_committed_tokens(removal)
        if not removal["completed_after"]:
            _assert_resumes_with_a_recompute(recorder, removal)
    _assert_every_request_completes(simulator, case["num_requests"])


def _observe_blocking_steps(monkeypatch):
    """Each step the engine blocked on: the stages it ran before leaving the
    engine batch queue, and how many of its rows were still live."""

    stages_run = collections.defaultdict(list)
    blocking_steps = []
    stage_end = BatchStageEndEvent.handle_event
    leave_queue = VLLMv1EngineReplicaScheduler._leave_engine_batch_queue

    def observed_stage_end(self, scheduler, metrics_store):
        stages_run[self._batch.id].append(self._stage_id)
        return stage_end(self, scheduler, metrics_store)

    def observed_leave(self, batch):
        if self._has_engine_batch_queue and batch.id == self._blocking_batch_id:
            blocking_steps.append(dict(
                batch_id=batch.id,
                stages=list(stages_run[batch.id]),
                num_stages=self._num_stages,
                live_rows=sum(
                    batch._request_execution_matches_snapshot(index)
                    for index in range(len(batch.requests))
                ),
            ))
        leave_queue(self, batch)

    monkeypatch.setattr(BatchStageEndEvent, "handle_event", observed_stage_end)
    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_leave_engine_batch_queue", observed_leave
    )
    return blocking_steps


# Dense PP4 under KV pressure: every row of a step the engine blocks on is
# preempted while that step is in flight.
BLOCKING_PREEMPTED_STEP_CASE = dict(
    replica=DENSE_REPLICA,
    num_pipeline_stages=4,
    num_blocks=8,
    num_requests=6,
    seed=1,
)


def test_the_engine_stays_blocked_until_its_step_runs_every_stage(
    tmp_path, monkeypatch
):
    blocking_steps = _observe_blocking_steps(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, BLOCKING_PREEMPTED_STEP_CASE))
    simulator.run()

    assert any(step["live_rows"] == 0 for step in blocking_steps), blocking_steps
    for step in blocking_steps:
        # vLLM waits for the step's output, which its last PP rank returns.
        assert step["stages"] == list(range(step["num_stages"])), step
    _assert_every_request_completes(
        simulator, BLOCKING_PREEMPTED_STEP_CASE["num_requests"]
    )


def _scale_stage_time_with_tokens(monkeypatch):
    dummy_execution_time = BaseExecutionTimePredictor._get_dummy_execution_time

    def token_scaled(self, batch, pipeline_stage):
        unit_time = self._dummy_execution_time
        self._dummy_execution_time = unit_time * max(1, batch.total_num_tokens)
        try:
            return dummy_execution_time(self, batch, pipeline_stage)
        finally:
            self._dummy_execution_time = unit_time

    monkeypatch.setattr(
        BaseExecutionTimePredictor, "_get_dummy_execution_time", token_scaled
    )


def _observe_popped_decode_victims(monkeypatch):
    """Price each popped stage batch that holds a decode row preempted since it formed.

    The phase of every row is recorded when its batch is built, apart from the
    batch; the pricing reads run at the stage pop, before the batch ends.
    """
    scheduled_phases = {}
    batch_init = Batch.__init__

    def recorded_init(self, *args, **kwargs):
        batch_init(self, *args, **kwargs)
        scheduled_phases[self.id] = [request.is_decoding for request in self.requests]

    monkeypatch.setattr(Batch, "__init__", recorded_init)
    predictor = SimpleNamespace(
        _config=SimpleNamespace(kv_cache_prediction_granularity=1)
    )
    priced = []
    pop_batch = ReplicaStageScheduler.pop_batch_if_not_busy

    def recorded_pop(self):
        batch = pop_batch(self)
        if batch is None or batch.id not in scheduled_phases:
            return batch
        phases = scheduled_phases[batch.id]
        if any(
            was_decoding and request.is_recomputing
            for was_decoding, request in zip(phases, batch.requests)
        ):
            priced.append(
                dict(
                    batch=batch,
                    scheduled_phases=phases,
                    mla_shape=SklearnExecutionTimePredictor._get_mla_batch_runtime_shape_components(
                        batch
                    ),
                    prefill_attention_params=SklearnExecutionTimePredictor._get_batch_prefill_attention_params(
                        predictor, batch
                    ),
                )
            )
        return batch

    monkeypatch.setattr(ReplicaStageScheduler, "pop_batch_if_not_busy", recorded_pop)
    return priced


def test_a_scheduled_decode_row_is_priced_as_decode_after_a_later_preemption(
    tmp_path, monkeypatch
):
    # vLLM runs the step in its batch queue as it was scheduled; the victim's
    # recompute is a later step of its own.
    _scale_stage_time_with_tokens(monkeypatch)
    priced = _observe_popped_decode_victims(monkeypatch)
    case = SCHEDULED_DECODE_VICTIM_CASE
    simulator = Simulator(_pipelined_config(tmp_path, case))
    simulator.run()

    assert priced
    for entry in priced:
        batch, phases = entry["batch"], entry["scheduled_phases"]
        assert batch.request_is_decoding == phases
        decode_tokens = [
            tokens for tokens, is_decoding in zip(batch.num_tokens, phases) if is_decoding
        ]
        prefill_tokens = [
            tokens for tokens, is_decoding in zip(batch.num_tokens, phases) if not is_decoding
        ]
        assert entry["mla_shape"]["decode_active_token_counts"] == decode_tokens
        assert entry["mla_shape"]["prefill_active_token_counts"] == prefill_tokens
        assert [chunk for _, chunk in entry["prefill_attention_params"]] == prefill_tokens
    _assert_every_request_completes(simulator, case["num_requests"])


class _Readmissions:
    """Each victim preempted with a sampling row in flight, and its next row.

    A prompt or recompute can have several chunks in flight; the newest row
    reaches furthest, so it is the one that samples if any does: a decode step,
    or a chunk that reaches the end of the prompt or of the recompute.
    """

    def __init__(self):
        self.episodes = []
        self._newest_rows = {}
        self._awaiting_row = {}

    def on_row(self, request, row):
        episode = self._awaiting_row.pop(request.id, None)
        if episode is not None:
            episode.update(next_row=row, readmitted_first=not episode["row_ended"])
        self._newest_rows[request.id] = row

    def on_preempted(self, victim, processed_before, remaining_before):
        row = self._newest_rows[victim.id]
        if not row["prefill_complete"]:
            kind = "prefill"
            samples = row["context"] + row["width"] >= victim.num_prefill_tokens
        elif row["recomputing"]:
            kind = "recompute"
            samples = row["context"] + row["width"] >= row["processed"]
        else:
            kind, samples = "decode", True
        if samples:
            episode = dict(
                request=victim, kind=kind, processed_before=processed_before,
                remaining_before=remaining_before, batch_id=row["batch_id"],
                row_ended=False, at_head_before_the_row_ends=False,
            )
            self.episodes.append(episode)
            self._awaiting_row[victim.id] = episode

    def on_waiting_pass(self, head):
        episode = self._awaiting_row.get(head.id)
        if episode is not None and not episode["row_ended"]:
            episode["at_head_before_the_row_ends"] = True

    def on_rows_end(self, batch, time):
        for request in batch.requests:
            episode = self._episode_of_row(request, batch.id)
            if episode is not None:
                episode.update(row_ended=True, row_end_time=time)

    def on_sample(self, request, time, context_before):
        episode = next(
            (
                episode
                for episode in reversed(self.episodes)
                if episode["request"] is request
            ),
            None,
        )
        if episode is None:
            return
        episode.setdefault("samples", []).append(dict(
            time=time,
            context_before=context_before,
            context_after=request.num_context_tokens,
            processed=request.num_processed_tokens,
            prefill_completed_at=request._prefill_completed_at,
            first_decode_at=request.first_decode_token_completed_at,
        ))

    def _episode_of_row(self, request, batch_id):
        for episode in reversed(self.episodes):
            if episode["request"] is request and episode["batch_id"] == batch_id:
                return episode
        return None


def _observe_readmissions(monkeypatch):
    recorder = _Readmissions()
    preempt_request = VLLMv1EngineReplicaScheduler._preempt_request
    create_batch = VLLMv1EngineReplicaScheduler._create_batch
    schedule_waiting = VLLMv1EngineReplicaScheduler._schedule_waiting_requests
    apply_samples = Batch.apply_preempted_step_samples
    apply_sample = Request.on_preempted_step_end

    def observed_preempt(self, victim, preempted_requests):
        in_flight = self._is_request_active_in_batch(victim) and not victim.completed
        processed_before = victim.num_processed_tokens
        remaining_before = victim.remaining_decode_tokens
        preempt_request(self, victim, preempted_requests)
        if in_flight:
            recorder.on_preempted(victim, processed_before, remaining_before)

    def observed_schedule_waiting(self, token_budget):
        waiting = self._get_sorted_waiting_queue()
        if waiting and token_budget > 0:
            recorder.on_waiting_pass(waiting[0])
        return schedule_waiting(self, token_budget)

    def observed_create_batch(self, requests, num_tokens):
        batch = create_batch(self, requests, num_tokens)
        for request, width, context in zip(
            requests, num_tokens, batch.num_context_tokens
        ):
            recorder.on_row(request, dict(
                time=self._current_schedule_time,
                batch_id=batch.id,
                width=int(width),
                context=int(context),
                processed=request.num_processed_tokens,
                prefill_complete=request.is_prefill_complete,
                recomputing=request.is_recomputing,
                cursor=request._num_recomputed_tokens,
                prefill_completed_at=request._prefill_completed_at,
            ))
        return batch

    def observed_apply_samples(
        self, time, cluster_type, request_execution_signatures=None
    ):
        recorder.on_rows_end(self, time)
        return apply_samples(self, time, cluster_type, request_execution_signatures)

    def observed_sample(self, time, cluster_type):
        context_before = self.num_context_tokens
        apply_sample(self, time, cluster_type)
        recorder.on_sample(self, time, context_before)

    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_preempt_request", observed_preempt
    )
    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler, "_create_batch", observed_create_batch
    )
    monkeypatch.setattr(
        VLLMv1EngineReplicaScheduler,
        "_schedule_waiting_requests",
        observed_schedule_waiting,
    )
    monkeypatch.setattr(Batch, "apply_preempted_step_samples", observed_apply_samples)
    monkeypatch.setattr(Request, "on_preempted_step_end", observed_sample)
    return recorder


# Dense PP4 runs. A decode victim, a victim with the last chunk of its prompt
# in flight, and one with the last chunk of its recompute in flight are each
# admitted again before that row ends; each case names the victims it reaches.
# Found by a probe over dense and MoE, PP2/PP4, 6-12 blocks, 12-48 requests
# and 24 seeds, where no one configuration reaches all three.
READMISSION_CASES = {
    "decode_and_prefill": (
        dict(
            replica=DENSE_REPLICA,
            num_pipeline_stages=4,
            num_blocks=6,
            num_requests=12,
            seed=14,
        ),
        {"decode", "prefill"},
    ),
    "decode_and_recompute": (
        dict(
            replica=DENSE_REPLICA,
            num_pipeline_stages=4,
            num_blocks=12,
            num_requests=24,
            seed=3,
        ),
        {"decode", "recompute"},
    ),
}


@pytest.mark.parametrize("name", READMISSION_CASES)
def test_a_victim_admitted_again_before_its_inflight_row_ends_keeps_its_sample(
    name, tmp_path, monkeypatch
):
    case, readmitted_kinds = READMISSION_CASES[name]
    recorder = _observe_readmissions(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, case))
    simulator.run()

    readmitted = collections.Counter(
        episode["kind"]
        for episode in recorder.episodes
        if episode.get("readmitted_first")
    )
    assert readmitted_kinds <= set(readmitted), recorder.episodes
    for episode in recorder.episodes:
        request, row = episode["request"], episode.get("next_row")
        # The sample lands once, when the row that carries it ends.
        assert episode["row_ended"], episode
        [sample] = episode["samples"]
        assert sample["time"] == episode["row_end_time"], episode
        if episode["kind"] == "prefill":
            tokens_before_the_sample = request.num_prefill_tokens
            assert sample["prefill_completed_at"] == sample["time"], episode
            assert sample["first_decode_at"] == sample["time"], episode
        else:
            tokens_before_the_sample = episode["processed_before"]
        assert sample["processed"] == tokens_before_the_sample + 1, episode
        if row is None:
            # The sample ended the request when its row ended.
            assert request.completed, episode
            continue
        if episode["readmitted_first"]:
            # As in vLLM, the admission is sized before the sample arrives,
            # and the tokens it has computed by then count toward the
            # recompute of the sample.
            assert row["width"] <= tokens_before_the_sample, episode
            assert sample["context_after"] == sample["context_before"], episode
        else:
            assert row["recomputing"] and row["cursor"] == 0, episode
            assert row["processed"] == tokens_before_the_sample + 1, episode
    _assert_every_request_completes(simulator, case["num_requests"])



# Three-request priority traces at PP4. A victim whose one remaining output
# token is the sample of its in-flight row heads the waiting queue, with blocks
# for it, before that row ends: a decode victim and a victim with the last chunk
# of its prompt in flight in the first trace, a victim with the last chunk of
# its recompute in flight in the second. A random search over short priority
# traces found both.
SAMPLE_ENDS_THE_VICTIM_CASES = {
    "decode_and_prefill": dict(
        trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.09,16,3,2
0.92,32,1,1
0.93,32,3,0
""",
        num_blocks=4,
        num_pipeline_stages=4,
    ),
    "recompute": dict(
        trace_text="""arrived_at,num_prefill_tokens,num_decode_tokens,priority
0.01,32,2,0
0.61,16,2,2
0.97,48,1,1
""",
        num_blocks=4,
        num_pipeline_stages=4,
    ),
}


@pytest.mark.parametrize("name", SAMPLE_ENDS_THE_VICTIM_CASES)
def test_a_victim_that_its_inflight_sample_ends_is_not_admitted_again(
    name, tmp_path, monkeypatch
):
    case = SAMPLE_ENDS_THE_VICTIM_CASES[name]
    recorder = _observe_readmissions(monkeypatch)
    simulator = Simulator(_priority_chunk_config(tmp_path, **case))
    simulator.run()

    ended_by_the_sample = [
        episode for episode in recorder.episodes if episode["remaining_before"] == 1
    ]
    assert any(
        episode["at_head_before_the_row_ends"] for episode in ended_by_the_sample
    ), recorder.episodes
    for episode in ended_by_the_sample:
        # vLLM would admit it again only to free it when the sample arrives.
        assert "next_row" not in episode, episode
        assert episode["row_ended"] and episode["request"].completed, episode
    _assert_every_request_completes(
        simulator, len(case["trace_text"].splitlines()) - 1
    )
