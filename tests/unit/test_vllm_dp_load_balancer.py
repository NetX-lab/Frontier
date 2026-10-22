"""Behavior tests for the opt-in vLLM-style DP placement policy.

Three layers are covered here: the pure selection and publication state
machine, the real `vllm_v1` load accessor it consumes, and the policy's
capability guard built through the real cluster-scheduler constructor. The
real-runtime evidence -- that `ClusterScheduleEvent` supplies the time, that
`GlobalBatchEndEvent` reports post-step load, and that the policy keeps no
simulation alive -- lives in
`tests/integration/test_vllm_dp_placement_runtime.py`.

Expected values are derived from vLLM v0.10.2 `core_client.py` and
`coordinator.py`, cited in the module under test. They are not produced by
re-running the implementation.
"""

from __future__ import annotations

import sys

import pytest

from frontier.config import (
    BaseModelConfig,
    ClusterConfig,
    FixedRequestLengthGeneratorConfig,
    LORClusterSchedulerConfig,
    MetricsConfig,
    PoissonRequestIntervalGeneratorConfig,
    RandomClusterSchedulerConfig,
    ReplicaConfig,
    RoundRobinClusterSchedulerConfig,
    SarathiSchedulerConfig,
    SimulationConfig,
    StickyLORClusterSchedulerConfig,
    StickyRoundRobinClusterSchedulerConfig,
    SyntheticRequestGeneratorConfig,
    TraceRequestGeneratorConfig,
    VllmLoadBalancingClusterSchedulerConfig,
    VllmV1SchedulerConfig,
)
from frontier.config import global_vars
from frontier.config.cluster_scheduler_config import BaseClusterSchedulerConfig
from frontier.config.flat_dataclass import create_flat_dataclass
from frontier.config.utils import get_all_subclasses
from frontier.entities import Cluster, Request
from frontier.scheduler.cluster_scheduler.cluster_scheduler_registry import (
    ClusterSchedulerRegistry,
)
from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
from frontier.types import (
    ActivationType,
    ClusterSchedulerType,
    ClusterType,
    NormType,
)


@pytest.fixture(autouse=True)
def reset_simulation_globals():
    """Keep the process-global model latch out of it.

    Building a `SimulationConfig` latches `IS_MOE` for the whole process, and
    this file builds both dense and MoE shapes. Resetting on both sides is what
    `tests/unit/test_config_owned_contracts.py` already does.
    """

    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


# --------------------------------------------------------------------------
# Selection: reference core_client.py:1139-1153
# --------------------------------------------------------------------------


def test_selection_weights_waiting_four_times_against_running() -> None:
    balancer = VllmDPLoadBalancer(2)
    # Engine 0 scores 4*1 + 0 = 4; engine 1 scores 4*0 + 3 = 3.
    balancer.frontend_counts = [RequestLoad(1, 0), RequestLoad(0, 3)]

    assert balancer.select(0.0) == 1


def test_selection_weight_is_exactly_four_at_the_boundary() -> None:
    balancer = VllmDPLoadBalancer(2)
    # 4*1 + 0 == 4*0 + 4, so the tie falls to the lowest index.
    balancer.frontend_counts = [RequestLoad(1, 0), RequestLoad(0, 4)]
    assert balancer.select(0.0) == 0

    balancer = VllmDPLoadBalancer(2)
    # One more running unit on engine 1 breaks the tie the other way.
    balancer.frontend_counts = [RequestLoad(1, 0), RequestLoad(0, 5)]
    assert balancer.select(0.0) == 0

    balancer = VllmDPLoadBalancer(2)
    balancer.frontend_counts = [RequestLoad(1, 1), RequestLoad(0, 4)]
    assert balancer.select(0.0) == 1


def test_equal_scores_always_choose_the_lowest_engine_index() -> None:
    balancer = VllmDPLoadBalancer(4)
    balancer.frontend_counts = [RequestLoad(2, 1)] * 4

    assert balancer.select(0.0) == 0


def test_local_reservations_spread_requests_between_snapshots() -> None:
    balancer = VllmDPLoadBalancer(2)

    # Counts start empty, and nothing is published in between, so the local
    # waiting reservation is the only thing that moves the choice.
    assert [balancer.select(0.0) for _ in range(5)] == [0, 1, 0, 1, 0]


def test_a_published_snapshot_replaces_local_reservations() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.select(0.0)
    balancer.select(0.0)
    assert balancer.frontend_counts == [RequestLoad(1, 0), RequestLoad(1, 0)]

    # A report makes engine 1 the loaded one, and the publish that follows
    # replaces the estimate wholesale rather than adjusting the reservations.
    balancer.report(0.01, 1, 0, RequestLoad(0, 6))
    assert balancer.select(1.0) == 0
    assert balancer.frontend_counts == [RequestLoad(1, 0), RequestLoad(0, 6)]


def test_one_engine_always_selects_lane_zero() -> None:
    balancer = VllmDPLoadBalancer(1)

    assert [balancer.select(0.0), balancer.select(0.5), balancer.select(9.0)] == [0] * 3


# --------------------------------------------------------------------------
# Publication: reference coordinator.py:192-218, :293-310
# --------------------------------------------------------------------------


def test_the_first_publish_waits_only_the_collection_interval() -> None:
    balancer = VllmDPLoadBalancer(2)

    # 50 ms, because the reference's `wait_for - elapsed` is deeply negative on
    # its first iteration and `min_timeout` decides.
    assert balancer.next_publish_ms == 50


def test_changed_counts_republish_after_the_changed_interval() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 0, RequestLoad(1, 0))
    # First publish consumes the collection wait.
    balancer.select(0.06)
    assert balancer.last_publish_ms == 50

    balancer.report(0.06, 0, 1, RequestLoad(2, 0))

    # stats_changed, so 100 ms after the last publish.
    assert balancer.next_publish_ms == 150


def test_unchanged_counts_republish_after_the_idle_interval() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 0, RequestLoad(1, 0))
    balancer.select(0.06)

    # The publish cleared stats_changed, so the next deadline is the idle one.
    assert balancer.stats_changed is False
    assert balancer.next_publish_ms == 50 + 5000


def test_a_newer_step_latches_the_previous_counts_when_changes_are_pending() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 0, RequestLoad(1, 0))
    assert balancer.last_step_counts is None

    balancer.report(0.001, 1, 1, RequestLoad(5, 5))

    # The step advanced with an unpublished change, so the counts as of the
    # previous step are held back for the next publish.
    assert balancer.last_step_counts == [RequestLoad(1, 0), RequestLoad(0, 0)]
    assert balancer.engine_counts == [RequestLoad(1, 0), RequestLoad(5, 5)]

    # That snapshot, not the newer counts, is what the frontend sees first.
    # Under the previous-step snapshot engine 1 is empty and wins; under the
    # newer counts engine 0 would have won with 4 against 25.
    assert balancer.select(0.06) == 1
    assert balancer.last_step_counts is None
    # The published snapshot plus this selection's local reservation.
    assert balancer.frontend_counts == [RequestLoad(1, 0), RequestLoad(1, 0)]


def test_a_newer_step_without_pending_changes_latches_nothing() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 0, RequestLoad(1, 0))
    balancer.select(0.06)
    assert balancer.stats_changed is False

    balancer.report(0.06, 0, 4, RequestLoad(2, 0))

    assert balancer.last_step_counts is None
    assert balancer.last_report_step == 4


def test_peer_engines_reporting_one_forward_do_not_latch_a_snapshot() -> None:
    """Both attention-DP lanes of one shared forward carry the same key."""

    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 3, RequestLoad(1, 0))
    assert balancer.last_report_step == 3

    balancer.report(0.0, 1, 3, RequestLoad(0, 2))

    # The equal key takes neither branch, exactly as coordinator.py:293-300.
    assert balancer.last_step_counts is None
    assert balancer.last_report_step == 3
    assert balancer.engine_counts == [RequestLoad(1, 0), RequestLoad(0, 2)]


def test_an_unchanged_report_changes_nothing_at_all() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 0, RequestLoad(1, 1))
    before = (
        list(balancer.engine_counts),
        balancer.last_report_step,
        balancer.next_publish_ms,
        balancer.stats_changed,
    )

    # The reference engine compares against its own last counts and sends no
    # message, so nothing downstream moves -- not even the step latch.
    balancer.report(0.02, 0, 9, RequestLoad(1, 1))

    assert (
        list(balancer.engine_counts),
        balancer.last_report_step,
        balancer.next_publish_ms,
        balancer.stats_changed,
    ) == before


def test_an_out_of_order_step_warns_and_still_applies_the_counts(caplog) -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 7, RequestLoad(1, 0))

    with caplog.at_level("WARNING"):
        balancer.report(0.0, 1, 2, RequestLoad(3, 4))

    assert "out-of-order step 2" in caplog.text
    # Applied unconditionally, as coordinator.py:308-310 does. Nothing aborts.
    assert balancer.engine_counts[1] == RequestLoad(3, 4)
    assert balancer.last_report_step == 7


# --------------------------------------------------------------------------
# Ordering and time validation
# --------------------------------------------------------------------------


def test_a_report_on_its_deadline_is_processed_before_that_publish() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 0, 0, RequestLoad(1, 0))
    assert balancer.next_publish_ms == 50

    # A real poller returns the waiting message instead of timing out, so the
    # report is applied and then moves the deadline.
    balancer.report(0.05, 1, 1, RequestLoad(9, 9))

    assert balancer.frontend_counts == [RequestLoad(0, 0), RequestLoad(0, 0)]
    assert balancer.engine_counts[1] == RequestLoad(9, 9)
    assert balancer.last_step_counts == [RequestLoad(1, 0), RequestLoad(0, 0)]


def test_a_selection_on_a_passed_deadline_sees_the_new_snapshot() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.report(0.0, 1, 0, RequestLoad(0, 6))
    assert balancer.frontend_counts == [RequestLoad(0, 0), RequestLoad(0, 0)]

    # At the deadline itself a selection publishes first, then chooses.
    assert balancer.select(0.05) == 0
    assert balancer.last_publish_ms == 50


def test_several_reports_inside_one_millisecond_stay_in_one_publish_window() -> None:
    balancer = VllmDPLoadBalancer(3)
    for engine in range(3):
        balancer.report(0.0001 * engine, engine, 0, RequestLoad(engine + 1, 0))

    assert balancer.frontend_counts == [RequestLoad(0, 0)] * 3
    assert balancer.next_publish_ms == 50
    assert balancer.engine_counts == [
        RequestLoad(1, 0),
        RequestLoad(2, 0),
        RequestLoad(3, 0),
    ]


# --------------------------------------------------------------------------
# The load accessor, on the real vllm_v1 scheduler
# --------------------------------------------------------------------------


def _model(*, is_moe: bool, num_layers: int = 4) -> BaseModelConfig:
    model = BaseModelConfig(
        num_layers=num_layers,
        num_q_heads=4,
        num_kv_heads=2,
        embedding_dim=256,
        mlp_hidden_dim=64,
        max_position_embeddings=4096,
        use_gated_mlp=True,
        use_bias=False,
        use_qkv_bias=False,
        activation=ActivationType.SILU,
        norm=NormType.RMS_NORM,
        post_attn_norm=True,
        vocab_size=1024,
        is_moe=is_moe,
        num_experts=8 if is_moe else 0,
        num_experts_per_tok=2 if is_moe else 0,
        torch_dtype="bfloat16",
    )
    model._model_name = f"w4_dp_{'moe' if is_moe else 'dense'}_{num_layers}l"
    return model


def _policy_scheduler(
    patch,
    *,
    is_moe: bool = True,
    attn_dp: int = 2,
    moe_ep: int = 2,
    num_replicas: int = 1,
    num_pipeline_stages: int = 1,
    num_layers: int = 4,
    cluster_type: ClusterType = ClusterType.MONOLITHIC,
    replica_scheduler_config=None,
    cluster_scheduler_config=None,
    request_generator_config=None,
):
    """Build a real cluster scheduler through the real constructor path."""

    model = _model(is_moe=is_moe, num_layers=num_layers)
    original = BaseModelConfig.create_from_name
    patch.setattr(
        BaseModelConfig,
        "create_from_name",
        classmethod(
            lambda cls, name: model if name == model._model_name else original(name)
        ),
    )
    moe_fields = (
        dict(
            moe_tensor_parallel_size=1,
            moe_expert_parallel_size=moe_ep,
            total_expert_num=8,
            router_topk=2,
        )
        if is_moe
        else {}
    )
    replica_config = ReplicaConfig(
        model_name=model._model_name,
        device="a100",
        network_device="a100_pairwise_nvlink",
        num_pipeline_stages=num_pipeline_stages,
        attn_tensor_parallel_size=1,
        attn_dp=attn_dp,
        memory_margin_fraction=0.1,
        **moe_fields,
    )
    generator_config = request_generator_config or SyntheticRequestGeneratorConfig(
        num_requests=2,
        length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=8, decode_tokens=2
        ),
        interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1.0),
    )
    policy_config = cluster_scheduler_config or VllmLoadBalancingClusterSchedulerConfig()
    cluster_config = ClusterConfig(
        cluster_type=cluster_type,
        num_replicas=num_replicas,
        replica_config=replica_config,
        replica_scheduler_config=replica_scheduler_config
        or VllmV1SchedulerConfig(
            num_blocks=64,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=16,
            enable_chunked_prefill=True,
        ),
        cluster_scheduler_config=policy_config,
    )
    metrics_config = MetricsConfig(
        write_metrics=False,
        store_plots=False,
        enable_chrome_trace=False,
        write_json_trace=False,
    )
    cluster = Cluster(cluster_config, metrics_config, generator_config)
    return ClusterSchedulerRegistry.get(
        policy_config.get_type(),
        config=cluster_config,
        cluster=cluster,
        request_generator_config=generator_config,
        predictor=None,
    )


def _request(tokens: int = 8) -> Request:
    return Request(arrived_at=0.0, num_prefill_tokens=tokens, num_decode_tokens=2)


def test_the_load_accessor_separates_waiting_from_admitted_running(monkeypatch) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(patch)
        lane = scheduler.get_replica_scheduler(scheduler._serving_replica_id, 0)

        assert lane.get_request_load() == RequestLoad(0, 0)

        for _ in range(3):
            lane.add_request(_request())
        # Queued but not admitted: waiting only.
        assert lane.get_request_load() == RequestLoad(3, 0)

        batch = lane.on_schedule(0.0)
        load = lane.get_request_load()
        # A 16-token budget admits two of the three 8-token prompts, so one waits.
        assert load == RequestLoad(1, 2)
        assert load.running + load.waiting == 3
        assert batch is not None


def test_a_preempted_request_is_waiting_again(monkeypatch) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(patch)
        lane = scheduler.get_replica_scheduler(scheduler._serving_replica_id, 0)
        for _ in range(2):
            lane.add_request(_request())
        lane.on_schedule(0.0)
        admitted = lane.get_request_load()

        victim = lane._running_requests[0]
        lane._preempted_requests.append(victim)
        lane._running_requests.remove(victim)

        assert lane.get_request_load() == RequestLoad(
            admitted.waiting + 1, admitted.running - 1
        )


def test_a_scheduler_without_a_serving_load_definition_says_which_one() -> None:
    class _Bare(BaseReplicaScheduler):
        def _get_next_batch(self, *args, **kwargs):
            return None

        def on_batch_end(self, *args, **kwargs):
            return None

    bare = _Bare.__new__(_Bare)

    with pytest.raises(NotImplementedError, match="_Bare does not expose"):
        bare.get_request_load()


# --------------------------------------------------------------------------
# Capability guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "kwargs", "message"),
    [
        ("two_replicas", dict(num_replicas=2), "one co-location Replica"),
        (
            "wrong_replica_scheduler",
            dict(is_moe=False, attn_dp=1, moe_ep=1,
                 replica_scheduler_config=SarathiSchedulerConfig(
                     num_blocks=64, block_size=16, batch_size_cap=4,
                     chunk_size=16,
                 )),
            "one co-location Replica",
        ),
        (
            "moe_lanes_without_the_engine_loop",
            dict(replica_scheduler_config=SarathiSchedulerConfig(
                num_blocks=64, block_size=16, batch_size_cap=4,
                chunk_size=16,
            )),
            "require a vLLM v1 engine replica scheduler",
        ),
        (
            "dense_multi_lane",
            dict(is_moe=False, attn_dp=2),
            "share no stage-0 forward",
        ),
        (
            "dense_multi_lane_pipeline_parallel",
            dict(is_moe=False, attn_dp=2, num_pipeline_stages=2),
            "share no stage-0 forward",
        ),
        (
            "uneven_layer_partition",
            dict(num_pipeline_stages=3),
            "evenly divisible",
        ),
    ],
)
def test_each_unsupported_topology_is_rejected_at_construction(
    label, kwargs, message
) -> None:
    with pytest.MonkeyPatch.context() as patch:
        with pytest.raises(ValueError, match=message):
            _policy_scheduler(patch, **kwargs)


@pytest.mark.parametrize(
    ("is_moe", "attn_dp", "moe_ep", "num_pipeline_stages", "num_layers"),
    [
        (True, 2, 2, 1, 4),
        (True, 1, 1, 1, 4),
        (False, 1, 1, 1, 4),
        (True, 2, 2, 2, 4),
        (True, 1, 1, 2, 4),
        (False, 1, 1, 2, 4),
        (True, 2, 2, 3, 6),
        (True, 1, 1, 3, 6),
    ],
)
def test_the_supported_shapes_construct(
    is_moe, attn_dp, moe_ep, num_pipeline_stages, num_layers
) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch,
            is_moe=is_moe,
            attn_dp=attn_dp,
            moe_ep=moe_ep,
            num_pipeline_stages=num_pipeline_stages,
            num_layers=num_layers,
        )

    assert scheduler._load_balancer is not None
    assert len(scheduler._load_balancer.engine_counts) == attn_dp


@pytest.mark.parametrize("is_moe,attn_dp", [(True, 4), (True, 1), (False, 1)])
def test_decode_pp1_shapes_construct(is_moe, attn_dp) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch, cluster_type=ClusterType.DECODE,
            is_moe=is_moe, attn_dp=attn_dp, moe_ep=attn_dp,
        )
    assert len(scheduler._load_balancer.engine_counts) == attn_dp


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(cluster_type=ClusterType.PREFILL, num_pipeline_stages=2), "PP1 only"),
        (dict(cluster_type=ClusterType.DECODE, num_replicas=2), "one DECODE Replica"),
        (dict(cluster_type=ClusterType.DECODE, num_pipeline_stages=2), "PP1 only"),
        (dict(cluster_type=ClusterType.DECODE, is_moe=False), "share no stage-0 forward"),
    ],
)
def test_decode_extension_keeps_unsupported_shapes_rejected(kwargs, message) -> None:
    with pytest.MonkeyPatch.context() as patch:
        with pytest.raises(ValueError, match=message):
            _policy_scheduler(patch, **kwargs)


def test_decode_admission_holds_its_forward_key_until_completion() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch, cluster_type=ClusterType.DECODE,
            cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(
                policy_version="v0.28.0"
            ),
        )
        request = _request()
        request._num_processed_tokens = request.num_prefill_tokens
        request._is_prefill_complete = True
        request._prefill_completed_at = 0.0
        request.on_arrival(0.0, ClusterType.DECODE)
        replica_id, lane_id = scheduler.select_kv_transfer_target(0.0, request)
        lane = scheduler.get_replica_scheduler(replica_id, lane_id)
        lane.add_request(request)
        balancer = scheduler._load_balancer

        [batch] = lane.on_schedule(0.0)
        assert scheduler._held_key[lane_id] == 0
        assert balancer.engine_counts[lane_id] == RequestLoad(0, 0)
        batch.on_batch_end(0.01, ClusterType.DECODE)
        lane.on_batch_end(batch)
        scheduler.on_replica_batch_end(0.01, replica_id, lane_id, batch)
        assert scheduler._held_key[lane_id] is None
        assert balancer.last_report_step == 0
        assert balancer.engine_counts[lane_id] == lane.get_request_load()
        assert balancer.engine_counts[lane_id].running == 1
        assert balancer.request_owners == {request.id: lane_id}

        [batch] = lane.on_schedule(0.02)
        batch.on_batch_end(0.03, ClusterType.DECODE)
        lane.on_batch_end(batch)
        scheduler.on_replica_batch_end(0.03, replica_id, lane_id, batch)
        assert request.completed
        assert balancer.request_owners == {}
        assert balancer.engine_inflight == [0, 0]


@pytest.mark.parametrize("attn_dp", [1, 4])
def test_prefill_pp1_routes_and_releases_ownership_after_the_final_chunk(attn_dp) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch, cluster_type=ClusterType.PREFILL, attn_dp=attn_dp, moe_ep=attn_dp,
            cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(policy_version="v0.28.0"),
        )
        request = _request(tokens=32)
        request.on_arrival(0.0, ClusterType.PREFILL)
        scheduler.add_request(request)
        [(replica_id, lane_id, routed)] = scheduler.schedule_at(0.0)
        assert routed is request
        lane = scheduler.get_replica_scheduler(replica_id, lane_id)
        lane.add_request(request)
        balancer = scheduler._load_balancer

        [first] = lane.on_schedule(0.0)
        first.on_batch_end(0.01, ClusterType.PREFILL)
        lane.on_batch_end(first)
        scheduler.on_replica_batch_end(0.01, replica_id, lane_id, first)
        assert not request.is_prefill_complete
        assert balancer.request_owners == {request.id: lane_id}
        [last] = lane.on_schedule(0.02)
        last.on_batch_end(0.03, ClusterType.PREFILL)
        lane.on_batch_end(last)
        scheduler.on_replica_batch_end(0.03, replica_id, lane_id, last)
        assert request.is_prefill_complete
        assert not request.completed
        assert balancer.request_owners == {}
        assert balancer.engine_inflight == [0] * attn_dp
        assert lane._allocation_map[request.id] > 0


def test_routing_without_a_time_is_refused_rather_than_reusing_a_stale_one() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(patch)
        scheduler.add_request(_request())

        with pytest.raises(RuntimeError, match="route through schedule_at"):
            scheduler.schedule()

        # The queue is untouched, so the refusal loses no work.
        assert len(scheduler._request_queue) == 1


def test_routing_places_every_queued_request_on_the_serving_replica() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(patch)
        requests = [_request() for _ in range(4)]
        for request in requests:
            scheduler.add_request(request)

        mapping = scheduler.schedule_at(0.0)

    replica_id = scheduler._serving_replica_id
    assert [(rid, lane) for rid, lane, _ in mapping] == [
        (replica_id, 0),
        (replica_id, 1),
        (replica_id, 0),
        (replica_id, 1),
    ]
    assert [request for _, _, request in mapping] == requests
    assert scheduler._request_queue == []


@pytest.mark.parametrize(
    "seam", ["on_replica_batch_scheduled", "on_replica_batch_end"]
)
def test_an_unknown_lane_identity_is_rejected_at_the_report_boundary(seam) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(patch)

        with pytest.raises(ValueError, match="exact lane index"):
            getattr(scheduler, seam)(0.0, scheduler._serving_replica_id, None, None)


# --------------------------------------------------------------------------
# Report timing and keys under pipeline parallelism
#
# Expected reports are derived from the reference engine iteration
# (`core.py`, `step_with_batch_queue`): an iteration that schedules while its
# batch queue still has room publishes at once, one that fills the queue
# publishes after applying its oldest output, and the key is the index of the
# forward the iteration launches, shared by peer engines. The lane readings
# are scripted; the stage-0 forward groups go through the real context.
# --------------------------------------------------------------------------


class _ScriptedLane:
    """The two lane readings the policy takes, set by the test."""

    def __init__(self, waiting: int):
        self.num_running_batches = 0
        self.waiting = waiting
        self.running = 0

    def get_request_load(self) -> RequestLoad:
        return RequestLoad(self.waiting, self.running)


class _ScriptedReplica:
    """Drive one policy scheduler as its lanes and stage 0 would."""

    def __init__(self, patch, *, waiting: list[int], **shape):
        self.scheduler = _policy_scheduler(patch, attn_dp=len(waiting), **shape)
        self.replica_id = self.scheduler._serving_replica_id
        self.lanes = [_ScriptedLane(count) for count in waiting]
        for lane_id, lane in enumerate(self.lanes):
            self.scheduler._replica_schedulers[(self.replica_id, lane_id)] = lane
        self.stage0 = self.scheduler.get_stage_execution_context(self.replica_id, 0)
        self._stage0_owners = []
        self._operations = 0
        self.reports: list[tuple[int, int, RequestLoad]] = []
        balancer_report = self.scheduler._load_balancer.report

        def record(time, engine, step, load):
            self.reports.append((engine, step, load))
            balancer_report(time, engine, step, load)

        patch.setattr(self.scheduler._load_balancer, "report", record)

    def admit(self, lane_id: int, time: float = 0.0) -> None:
        """Admit one single-request batch, as the lane's admission loop does."""

        lane = self.lanes[lane_id]
        lane.num_running_batches += 1
        lane.waiting -= 1
        lane.running += 1
        self.scheduler.on_replica_batch_scheduled(time, self.replica_id, lane_id, None)

    def complete(self, lane_id: int, time: float, *, finished: int = 0) -> None:
        """Finish one batch on the last stage, as `GlobalBatchEndEvent` does."""

        lane = self.lanes[lane_id]
        lane.num_running_batches -= 1
        lane.running -= finished
        self.scheduler.on_replica_batch_end(time, self.replica_id, lane_id, None)

    def run_stage0_forward(self, num_lanes: int) -> int:
        """Start one shared stage-0 forward, run it to its end, return its group."""

        for _ in range(num_lanes):
            ticket = self.stage0.enqueue_full_stage(operation_id=self._operations)
            self._operations += 1
            assert self.stage0.try_acquire(ticket)
            group = self.stage0.bind_forward_group(ticket)
            self._stage0_owners.append(ticket)
        for ticket in self._stage0_owners:
            self.stage0.release(ticket)
        self._stage0_owners.clear()
        return group


def test_pp1_admissions_are_reported_only_with_the_completion_they_join() -> None:
    with pytest.MonkeyPatch.context() as patch:
        replica = _ScriptedReplica(patch, waiting=[3, 3])
        for iteration in range(3):
            for lane_id in (0, 1):
                replica.admit(lane_id, time=0.1 * iteration)
            # The single pipeline slot is full after every admission.
            assert len(replica.reports) == 2 * iteration
            assert replica.run_stage0_forward(2) == iteration
            for lane_id in (0, 1):
                replica.complete(
                    lane_id, 0.1 * iteration + 0.05, finished=min(iteration, 1)
                )

    # One report per lane per iteration, each keyed by the forward that
    # completed, as the reference's atomic PP=1 step publishes.
    assert replica.reports == [
        (0, 0, RequestLoad(2, 1)),
        (1, 0, RequestLoad(2, 1)),
        (0, 1, RequestLoad(1, 1)),
        (1, 1, RequestLoad(1, 1)),
        (0, 2, RequestLoad(0, 1)),
        (1, 2, RequestLoad(0, 1)),
    ]


@pytest.mark.parametrize("lane_order", [(0, 1), (1, 0)])
def test_a_pp2_cold_fill_keys_peer_lanes_by_the_forward_they_share(
    lane_order,
) -> None:
    with pytest.MonkeyPatch.context() as patch:
        replica = _ScriptedReplica(patch, waiting=[3, 3], num_pipeline_stages=2)
        for lane_id in lane_order:
            replica.admit(lane_id)
            replica.admit(lane_id)
        # The first admission leaves a slot and is published; the second
        # fills the pipeline and waits for the first completion.
        assert replica.reports == [
            (lane_id, 0, RequestLoad(2, 1)) for lane_id in lane_order
        ]
        assert replica.run_stage0_forward(2) == 0
        assert replica.run_stage0_forward(2) == 1
        # Inside the first collection wait, so both lanes' new counts are
        # still unpublished when forward 1 is reported.
        for lane_id in lane_order:
            replica.complete(lane_id, 0.02)

    assert replica.reports[2:] == [
        (lane_id, 1, RequestLoad(1, 2)) for lane_id in lane_order
    ]
    # The first report of forward 1 latched forward 0 for both lanes; the
    # peer's report of the same forward latched nothing partial.
    assert replica.scheduler._load_balancer.last_step_counts == [
        RequestLoad(2, 1),
        RequestLoad(2, 1),
    ]


@pytest.mark.parametrize("lane_order", [(0, 1), (1, 0)])
def test_pp3_publishes_two_admission_only_iterations_before_any_completion(
    lane_order,
) -> None:
    with pytest.MonkeyPatch.context() as patch:
        replica = _ScriptedReplica(
            patch, waiting=[4, 4], num_pipeline_stages=3, num_layers=6
        )
        for lane_id in lane_order:
            for _ in range(3):
                replica.admit(lane_id)
        assert replica.reports == [
            report
            for lane_id in lane_order
            for report in (
                (lane_id, 0, RequestLoad(3, 1)),
                (lane_id, 1, RequestLoad(2, 2)),
            )
        ]
        assert [replica.run_stage0_forward(2) for _ in range(3)] == [0, 1, 2]
        for lane_id in lane_order:
            replica.complete(lane_id, 0.3)

    # The third admission filled the pipeline in forward 2.
    assert replica.reports[4:] == [
        (lane_id, 2, RequestLoad(1, 3)) for lane_id in lane_order
    ]


@pytest.mark.parametrize("is_moe", [True, False])
def test_a_full_pipeline_makes_one_report_per_iteration(is_moe) -> None:
    with pytest.MonkeyPatch.context() as patch:
        replica = _ScriptedReplica(
            patch, waiting=[3], is_moe=is_moe, moe_ep=1, num_pipeline_stages=2
        )

        def forward(group: int) -> None:
            # A dense Replica binds no forward group; its key is the lane's
            # admission count.
            if is_moe:
                assert replica.run_stage0_forward(1) == group

        replica.admit(0)
        replica.admit(0)
        forward(0)
        forward(1)
        replica.complete(0, 0.2, finished=1)
        replica.admit(0, 0.2)
        forward(2)
        replica.complete(0, 0.3, finished=1)
        # Nothing left to admit: the next completion is the lane's next
        # iteration on its own.
        replica.complete(0, 0.4, finished=1)

    assert replica.reports == [
        (0, 0, RequestLoad(2, 1)),
        (0, 1, RequestLoad(1, 1)),
        (0, 2, RequestLoad(0, 1)),
        (0, 3, RequestLoad(0, 0)),
    ]
    assert replica.scheduler._held_key == [None]


def test_a_completion_and_the_admission_it_makes_room_for_share_one_key() -> None:
    with pytest.MonkeyPatch.context() as patch:
        replica = _ScriptedReplica(
            patch, waiting=[1], moe_ep=1, num_pipeline_stages=2
        )
        replica.admit(0)
        assert replica.run_stage0_forward(1) == 0
        replica.lanes[0].waiting += 1
        replica.complete(0, 0.2)
        replica.admit(0, 0.2)
        balancer = replica.scheduler._load_balancer

    # The reference publishes this iteration once, after scheduling the new
    # request and applying the ready output. Two reports under one key leave
    # the coordinator the same state: nothing latched in between.
    assert replica.reports == [
        (0, 0, RequestLoad(0, 1)),
        (0, 1, RequestLoad(1, 1)),
        (0, 1, RequestLoad(0, 2)),
    ]
    assert balancer.last_step_counts is None
    assert balancer.engine_counts == [RequestLoad(0, 2)]


def test_a_lane_that_drains_to_zero_reports_it_and_new_work_keys_coherently() -> None:
    with pytest.MonkeyPatch.context() as patch:
        replica = _ScriptedReplica(patch, waiting=[1, 0], num_pipeline_stages=2)
        replica.admit(0)
        assert replica.run_stage0_forward(1) == 0
        replica.complete(0, 0.2, finished=1)
        balancer = replica.scheduler._load_balancer
        assert balancer.engine_counts == [RequestLoad(0, 0), RequestLoad(0, 0)]

        # New work on both lanes after a quiet interval joins one forward.
        for lane_id in (1, 0):
            replica.lanes[lane_id].waiting += 1
            replica.admit(lane_id, 10.0)
        assert replica.run_stage0_forward(2) == 1

    assert replica.reports == [
        (0, 0, RequestLoad(0, 1)),
        (0, 1, RequestLoad(0, 0)),
        (1, 1, RequestLoad(0, 1)),
        (0, 1, RequestLoad(0, 1)),
    ]


def test_the_admission_loop_reports_the_state_after_each_admission() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(patch, num_pipeline_stages=2)
        reports = []
        balancer_report = scheduler._load_balancer.report
        patch.setattr(
            scheduler._load_balancer,
            "report",
            lambda time, engine, step, load: (
                reports.append((engine, step, load)),
                balancer_report(time, engine, step, load),
            ),
        )
        lane = scheduler.get_replica_scheduler(scheduler._serving_replica_id, 0)
        # Each request fills the 16-token budget, so each batch holds one.
        for _ in range(3):
            lane.add_request(_request(tokens=16))

        batches = lane.on_schedule(0.0)

    assert len(batches) == 2
    # One call admitted two batches; only the first left a slot, and it is
    # reported with the state after it alone.
    assert reports == [(0, 0, RequestLoad(2, 1))]


# --------------------------------------------------------------------------
# Configuration discovery
# --------------------------------------------------------------------------


def test_all_cluster_policy_configs_and_registry_entries_remain_discoverable() -> None:
    configs = {
        subclass.get_type(): subclass
        for subclass in get_all_subclasses(BaseClusterSchedulerConfig)
    }

    assert set(configs) == set(ClusterSchedulerType)
    for scheduler_type in ClusterSchedulerType:
        assert ClusterSchedulerRegistry.get_class(scheduler_type) is not None


def test_the_accepted_policy_tokens_are_the_previous_five_plus_one() -> None:
    tokens = {
        str(subclass.get_type())
        for subclass in get_all_subclasses(BaseClusterSchedulerConfig)
    }

    assert tokens == {
        "round_robin",
        "random",
        "lor",
        "sticky_round_robin",
        "sticky_lor",
        "vllm_load_balancing",
    }


def test_the_generated_cli_selects_the_policy_and_keeps_the_old_default() -> None:
    flat_config = create_flat_dataclass(SimulationConfig)
    assert "cluster_scheduler_config_type" in flat_config.metadata_mapping

    original_argv = sys.argv
    try:
        sys.argv = ["frontier.main"]
        default_parsed = flat_config.create_from_cli_args()
        sys.argv = [
            "frontier.main",
            "--cluster_scheduler_config_type",
            "vllm_load_balancing",
        ]
        selected = flat_config.create_from_cli_args()
    finally:
        sys.argv = original_argv

    default_config = default_parsed.reconstruct_original_dataclass()
    selected_config = selected.reconstruct_original_dataclass()
    assert isinstance(
        default_config.cluster_config.cluster_scheduler_config,
        RoundRobinClusterSchedulerConfig,
    )
    assert isinstance(
        selected_config.cluster_config.cluster_scheduler_config,
        VllmLoadBalancingClusterSchedulerConfig,
    )


@pytest.mark.parametrize(
    "config_type",
    [
        RoundRobinClusterSchedulerConfig,
        RandomClusterSchedulerConfig,
        LORClusterSchedulerConfig,
        StickyRoundRobinClusterSchedulerConfig,
        StickyLORClusterSchedulerConfig,
        VllmLoadBalancingClusterSchedulerConfig,
    ],
)
def test_every_policy_config_round_trips_through_copy_and_dict(config_type) -> None:
    from copy import deepcopy

    from frontier.config.utils import dataclass_to_dict

    config = config_type()
    assert deepcopy(config) == config
    # The serialized identity is the selector token, which is what a config
    # artifact has to round-trip through.
    assert dataclass_to_dict(config)["name"] == str(config_type.get_type())


def test_v028_low_pressure_counts_waiting_once() -> None:
    config = VllmLoadBalancingClusterSchedulerConfig(policy_version="v0.28.0")
    balancer = VllmDPLoadBalancer(2, policy_version=config.policy_version)
    balancer.report(0.0, 0, 0, RequestLoad(1, 0, kv_cache_usage=0.5))
    balancer.report(0.0, 1, 0, RequestLoad(0, 2, kv_cache_usage=1.0))

    # Source oracle A: scores 1 and 2; legacy scores 4 and 2.
    assert balancer.select(0.06, request_id=100) == 0
    assert balancer.frontend_counts[0].kv_cache_usage == 0.5


def test_load_metadata_keeps_the_legacy_count_sequence_and_equality() -> None:
    load = RequestLoad(2, 1, kv_cache_usage=0.75)

    assert list(load) == [2, 1]
    assert tuple(load) == (2, 1)
    assert load == RequestLoad(2, 1, kv_cache_usage=0.5)
    assert load.kv_cache_usage == 0.75


@pytest.mark.parametrize(("usage", "expected"), [(0.5, 0), (0.75, 1), (1.0, 1)])
def test_v028_kv_pressure_penalizes_only_waiting_requests(usage, expected) -> None:
    balancer = VllmDPLoadBalancer(2, policy_version="v0.28.0")
    balancer.report(0.0, 0, 0, RequestLoad(2, 1, usage))
    balancer.report(0.0, 1, 0, RequestLoad(0, 4, 1.0))

    # Source oracle B: lane 0 scores 3, 6, or 9; lane 1 stays at 4.
    assert balancer.select(0.06, request_id=100) == expected


def test_v028_snapshot_replacement_retains_unfinished_request_ownership() -> None:
    balancer = VllmDPLoadBalancer(2, policy_version="v0.28.0")
    balancer.report(0.0, 1, 0, RequestLoad(0, 7))
    assert balancer.select(0.06, request_id=100) == 0
    assert balancer.select(0.06, request_id=101) == 0

    # An empty delayed snapshot predates the two requests' engine arrival.
    balancer.report(0.07, 1, 1, RequestLoad(0, 0))
    assert [balancer.select(0.16, request_id=i) for i in (102, 103, 104)] == [1, 1, 0]
    assert balancer.engine_inflight == [3, 2]
    assert balancer.request_owners == {100: 0, 101: 0, 102: 1, 103: 1, 104: 0}


def test_v028_ties_rotate_after_terminal_output_and_fresh_empty_snapshot() -> None:
    balancer = VllmDPLoadBalancer(2, policy_version="v0.28.0")
    choices = []
    for request_id, time in enumerate((0.06, 5.1, 10.2)):
        choices.append(balancer.select(time, request_id=request_id))
        balancer.finish_requests([])
        assert sum(balancer.engine_inflight) == 1
        balancer.finish_requests([request_id, request_id, -1])
        assert balancer.engine_inflight == [0, 0]
        assert balancer.request_owners == {}

    assert choices == [0, 1, 0]


def test_v028_selection_requires_a_real_request_identity() -> None:
    balancer = VllmDPLoadBalancer(2, policy_version="v0.28.0")
    with pytest.raises(ValueError, match="request_id"):
        balancer.select(0.0)


def test_v028_kv_only_changes_do_not_refresh_or_replace_latched_pressure() -> None:
    balancer = VllmDPLoadBalancer(2, policy_version="v0.28.0")
    balancer.report(0.0, 0, 0, RequestLoad(1, 1, 0.5))
    balancer.report(0.001, 1, 0, RequestLoad(0, 1, 0.75))
    balancer.report(0.002, 0, 9, RequestLoad(1, 1, 1.0))
    assert balancer.last_report_step == 0
    assert balancer.engine_counts[0].kv_cache_usage == 0.5

    balancer.report(0.003, 0, 1, RequestLoad(2, 1, 0.75))
    balancer.select(0.06, request_id=100)
    assert [load.kv_cache_usage for load in balancer.frontend_counts] == [0.5, 0.75]
    balancer.select(0.16, request_id=101)
    assert balancer.frontend_counts[0].kv_cache_usage == 0.75

    deadline = balancer.next_publish_ms
    balancer.report(0.17, 0, 10, RequestLoad(2, 1, 1.0))
    assert balancer.engine_counts[0].kv_cache_usage == 0.75
    assert balancer.next_publish_ms == deadline


@pytest.mark.parametrize("used_blocks", [2, 3])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_load_accessor_reports_allocatable_capacity_usage(used_blocks, prefix_caching) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch,
            replica_scheduler_config=VllmV1SchedulerConfig(
                num_blocks=4, block_size=16, batch_size_cap=4,
                max_tokens_in_batch=64, enable_chunked_prefill=True,
                enable_prefix_caching=prefix_caching,
            ),
            request_generator_config=TraceRequestGeneratorConfig(
                trace_file="examples/fixtures/prefix_cache_shared_session_trace.csv"
            ) if prefix_caching else None,
        )
        lane = scheduler.get_replica_scheduler(scheduler._serving_replica_id, 0)
        lane.add_request(Request(
            arrived_at=0.0, num_prefill_tokens=used_blocks * 16, num_decode_tokens=2,
            session_id=1, block_hash_ids=list(range(used_blocks)),
        ))
        [batch] = lane.on_schedule(0.0)
        assert batch.size == 1
        assert lane.num_allocated_blocks == used_blocks
        # Frontier's ordinary pool has four allocatable blocks. Native total
        # five with one null block has the same 0.5 / 0.75 usage, not used/5.
        assert lane.get_request_load().kv_cache_usage == used_blocks / 4


def test_v028_config_binds_real_requests_and_releases_terminal_outputs() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch,
            cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(
                policy_version="v0.28.0"
            ),
        )
        request = _request(tokens=24)
        scheduler.add_request(request)
        [(replica_id, lane_id, assigned)] = scheduler.schedule_at(0.0)
        assert assigned is request
        balancer = scheduler._load_balancer
        assert balancer.policy_version == "v0.28.0"
        assert balancer.request_owners == {request.id: lane_id}
        lane = scheduler.get_replica_scheduler(replica_id, lane_id)
        lane.add_request(request)

        completions = []
        for step in range(3):
            [batch] = lane.on_schedule(step * 0.1)
            batch.on_schedule(step * 0.1, ClusterType.MONOLITHIC)
            batch.on_batch_end(step * 0.1 + 0.01, ClusterType.MONOLITHIC)
            lane.on_batch_end(batch)
            scheduler.on_replica_batch_end(step * 0.1 + 0.01, replica_id, lane_id, batch)
            completions.append(request.completed)
            assert sum(balancer.engine_inflight) == (0 if request.completed else 1)

        assert completions == [False, False, True]
        assert balancer.request_owners == {}
        scheduler.on_replica_batch_end(0.3, replica_id, lane_id, batch)
        assert balancer.engine_inflight == [0, 0]


@pytest.mark.parametrize("used_blocks", [2, 3])
def test_grouped_load_usage_excludes_the_reserved_null_block(used_blocks) -> None:
    from tests.unit.test_grouped_kv_cache_scheduler import _scheduler

    lane = _scheduler(blocks=5, watermark=0)
    pool = lane._kv_cache_manager.block_pool
    allocated = pool.get_new_blocks(used_blocks)

    assert pool.reserved_blocks == 1
    assert all(block.block_id != 0 for block in allocated.blocks)
    assert pool.get_num_free_blocks() == 4 - used_blocks
    assert lane.get_request_load().kv_cache_usage == used_blocks / 4


def test_v028_preemption_and_stale_drop_keep_request_ownership() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch,
            cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(
                policy_version="v0.28.0"
            ),
        )
        request = _request(tokens=24)
        scheduler.add_request(request)
        [(replica_id, lane_id, _)] = scheduler.schedule_at(0.0)
        balancer = scheduler._load_balancer
        lane = scheduler.get_replica_scheduler(replica_id, lane_id)
        lane.add_request(request)
        [stale_batch] = lane.on_schedule(0.0)
        stale_batch.on_schedule(0.0, ClusterType.MONOLITHIC)
        lane._preempt_request(request, [])

        assert not request.completed
        assert lane.get_request_load() == RequestLoad(1, 0)
        lane.on_stale_batch_drop(stale_batch)
        scheduler.on_replica_batch_end(0.01, replica_id, lane_id, stale_batch)
        assert balancer.request_owners == {request.id: lane_id}
        assert sum(balancer.engine_inflight) == 1


def test_v028_request_ownership_survives_replayed_prefill_after_preemption() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch,
            cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(
                policy_version="v0.28.0"
            ),
        )
        request = _request(tokens=24)
        scheduler.add_request(request)
        [(replica_id, lane_id, _)] = scheduler.schedule_at(0.0)
        balancer = scheduler._load_balancer
        lane = scheduler.get_replica_scheduler(replica_id, lane_id)
        lane.add_request(request)
        [first] = lane.on_schedule(0.0)
        first.on_schedule(0.0, ClusterType.MONOLITHIC)
        first.on_batch_end(0.01, ClusterType.MONOLITHIC)
        lane.on_batch_end(first)
        scheduler.on_replica_batch_end(0.01, replica_id, lane_id, first)
        assert request.num_processed_tokens == 16
        lane._preempt_request(request, [])
        assert lane.get_request_load() == RequestLoad(1, 0)
        assert balancer.request_owners == {request.id: lane_id}

        for step in range(1, 4):
            [batch] = lane.on_schedule(step * 0.1)
            batch.on_schedule(step * 0.1, ClusterType.MONOLITHIC)
            batch.on_batch_end(step * 0.1 + 0.01, ClusterType.MONOLITHIC)
            lane.on_batch_end(batch)
            scheduler.on_replica_batch_end(step * 0.1 + 0.01, replica_id, lane_id, batch)
            assert sum(balancer.engine_inflight) == (0 if request.completed else 1)

        assert request.completed
        assert balancer.request_owners == {}


def test_v028_terminal_preempted_sample_releases_the_request_owner() -> None:
    from frontier.entities import Batch
    from tests.unit.test_request_preempted_step import _decode_request

    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch,
            cluster_scheduler_config=VllmLoadBalancingClusterSchedulerConfig(
                policy_version="v0.28.0"
            ),
        )
        request = _decode_request(processed=8, prefill=8, decode=1)
        scheduler.add_request(request)
        [(replica_id, lane_id, _)] = scheduler.schedule_at(0.0)
        batch = Batch(replica_id=replica_id, requests=[request], num_tokens=[1], is_moe=True)
        request.on_preempted(recompute=True, scheduler_num_computed_tokens=8)
        request.on_enter_waiting_queue(0.05, ClusterType.MONOLITHIC)
        stopped = batch.apply_preempted_step_samples(0.1, ClusterType.MONOLITHIC)
        batch.on_batch_end(0.1, ClusterType.MONOLITHIC)

        assert stopped == [(0, request)]
        assert batch.completed
        assert request.completed
        assert batch.completed_requests == []
        scheduler.on_replica_batch_end(0.1, replica_id, lane_id, batch)
        assert scheduler._load_balancer.request_owners == {}
        assert scheduler._load_balancer.engine_inflight == [0, 0]


def test_v028_policy_round_trips_through_generated_cli_copy_and_dict(monkeypatch) -> None:
    from copy import deepcopy
    from frontier.config.utils import dataclass_to_dict

    flat_config = create_flat_dataclass(SimulationConfig)
    monkeypatch.setattr(sys, "argv", [
        "frontier.main", "--cluster_scheduler_config_type", "vllm_load_balancing",
        "--vllm_load_balancing_cluster_scheduler_config_policy_version", "v0.28.0",
    ])
    selected = flat_config.create_from_cli_args().reconstruct_original_dataclass()
    config = selected.cluster_config.cluster_scheduler_config
    assert config.policy_version == "v0.28.0"
    assert deepcopy(config) == config
    assert dataclass_to_dict(config)["policy_version"] == "v0.28.0"
    assert VllmLoadBalancingClusterSchedulerConfig().policy_version == "v0.10.2"


def test_an_unknown_dp_policy_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="policy_version must be"):
        VllmLoadBalancingClusterSchedulerConfig(policy_version="unknown")


def test_default_zero_block_config_resolves_capacity_before_load_sampling() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch,
            replica_scheduler_config=VllmV1SchedulerConfig(num_blocks=0),
        )
        lane = scheduler.get_replica_scheduler(scheduler._serving_replica_id, 0)
        assert lane._config.num_blocks > 0
        assert lane.get_request_load() == RequestLoad(0, 0)
        assert lane.get_request_load().kv_cache_usage == 0.0
