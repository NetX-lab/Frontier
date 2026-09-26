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
vLLM; Frontier applies it where it removes the victim's row.
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
    SglangSchedulerConfig,
    SimulationConfig,
    SpeculativeDecodingConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    UniformRequestLengthGeneratorConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities.batch import Batch
from frontier.events.global_batch_end_event import GlobalBatchEndEvent
from frontier.events.replica_stage_schedule_event import ReplicaStageScheduleEvent
from frontier.metrics.metrics_store import MetricsStore
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
    ReplicaStageScheduler,
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
            prefill_cluster_num_replicas=1,
            decode_cluster_num_replicas=1,
            replica_config=ReplicaConfig(
                model_name="llama2_7b_dense_example",
                device="a100",
                network_device="a100_pairwise_nvlink",
                attn_tensor_parallel_size=1,
            ),
            decode_replica_config_num_pipeline_stages=case["num_pipeline_stages"],
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=64,
                block_size=16,
                batch_size_cap=4,
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


class _InflightRemovals:
    """Where Frontier removes a preempted request's in-flight row, and its state after.

    The sample is applied inside the stage-boundary event and the batch-end
    event, so the state is read after those events return. A victim scheduled
    again before its row is removed takes no sample and is not tracked. A
    prefill or recompute victim can have several chunks in flight; the rows
    before the one that completes its tokens take no sample and are skipped.
    """

    def __init__(self):
        self.removals = []
        self.rows = []
        self.end_ids = []
        self.open = {}
        self._pending = {}

    def note(self, kind, batch, index):
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
            kind=kind,
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
                prefill_completed_at=request.prefill_completed_at,
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
    materialize = ReplicaStageScheduler._materialize_runtime_live_batch
    stage_handle = ReplicaStageScheduleEvent.handle_event
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

    def observed_materialize(self, batch):
        stale = [
            index
            for index in range(len(batch.requests))
            if not batch._request_execution_matches_snapshot(index)
        ]
        live = materialize(self, batch)
        if live is not batch:
            kind = "whole_drop" if live is None else "stage_boundary"
            for index in stale:
                recorder.note(kind, batch, index)
        return live

    def observed_stage(self, scheduler, metrics_store):
        events = stage_handle(self, scheduler, metrics_store)
        recorder.finish(self, scheduler)
        return events

    def observed_global(self, scheduler, metrics_store):
        for index, request in enumerate(self._batch.requests):
            signature = Batch._get_request_execution_signature(request)
            if signature != self._request_execution_signatures[index]:
                recorder.note("batch_end", self._batch, index)
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
    monkeypatch.setattr(
        ReplicaStageScheduler, "_materialize_runtime_live_batch", observed_materialize
    )
    monkeypatch.setattr(ReplicaStageScheduleEvent, "handle_event", observed_stage)
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


# PP4 reaches a stage-boundary removal. PP2 removes in-flight decode rows at
# batch end: with equal stage times the next pop is ordered before the
# schedule that preempts, so two stages leave no later stage to drop the row.
INFLIGHT_DECODE_CASES = {
    "dense_pp4": dict(
        replica=DENSE_REPLICA,
        num_pipeline_stages=4,
        num_blocks=10,
        num_requests=24,
        seed=11,
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
# same prompt.
FINAL_CHUNK_CASE = dict(
    replica=DENSE_REPLICA,
    num_pipeline_stages=4,
    num_blocks=10,
    num_requests=24,
    seed=3,
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
PDD_PREEMPTION_CASE = dict(num_pipeline_stages=2, num_blocks=6, num_requests=8, seed=7)


@pytest.mark.parametrize("name", INFLIGHT_DECODE_CASES)
def test_an_inflight_decode_victim_keeps_the_sample_of_its_removed_row(
    name, tmp_path, monkeypatch
):
    case = INFLIGHT_DECODE_CASES[name]
    recorder = _observe_inflight_removals(monkeypatch)
    simulator = Simulator(_pipelined_config(tmp_path, case))
    simulator.run()

    removals = _decode_removals(recorder)
    kind = "stage_boundary" if case["num_pipeline_stages"] > 2 else "batch_end"
    assert any(removal["kind"] == kind for removal in removals), removals
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
