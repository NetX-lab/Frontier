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
    VllmLoadBalancingClusterSchedulerConfig,
    VllmV1SchedulerConfig,
)
from frontier.config.cluster_scheduler_config import BaseClusterSchedulerConfig
from frontier.config.flat_dataclass import create_flat_dataclass
from frontier.config.utils import get_all_subclasses
from frontier.entities import Cluster, Request
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.cluster_scheduler_registry import (
    ClusterSchedulerRegistry,
)
from frontier.scheduler.cluster_scheduler.lor_cluster_scheduler import (
    LORClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.random_cluster_scheduler import (
    RandomClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.sticky_lor_cluster_scheduler import (
    StickyLORClusterScheduler,
)
from frontier.scheduler.cluster_scheduler.sticky_round_robin_cluster_scheduler import (
    StickyRoundRobinClusterScheduler,
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

    assert balancer.last_publish_ms == -5000
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

    assert balancer.time_ms == 0
    assert balancer.last_publish_ms == -5000
    assert balancer.engine_counts == [
        RequestLoad(1, 0),
        RequestLoad(2, 0),
        RequestLoad(3, 0),
    ]


def test_time_must_not_move_backwards() -> None:
    balancer = VllmDPLoadBalancer(2)
    balancer.select(1.0)

    with pytest.raises(ValueError, match="cannot move backwards"):
        balancer.select(0.5)


@pytest.mark.parametrize("time", [-1.0, float("nan"), float("inf")])
def test_time_must_be_finite_and_nonnegative(time) -> None:
    balancer = VllmDPLoadBalancer(2)

    with pytest.raises(ValueError, match="finite nonnegative time"):
        balancer.select(time)


@pytest.mark.parametrize("engine", [-1, 2, True, 1.0, None])
def test_an_unknown_engine_is_rejected(engine) -> None:
    balancer = VllmDPLoadBalancer(2)

    with pytest.raises(ValueError, match="unknown engine"):
        balancer.report(0.0, engine, 0, RequestLoad(1, 0))


@pytest.mark.parametrize(
    ("step", "load"),
    [(-1, RequestLoad(1, 0)), (0, RequestLoad(-1, 0)), (0, RequestLoad(0, -2))],
)
def test_negative_steps_and_counts_are_rejected(step, load) -> None:
    balancer = VllmDPLoadBalancer(2)

    with pytest.raises(ValueError, match="nonnegative"):
        balancer.report(0.0, 0, step, load)


@pytest.mark.parametrize("num_engines", [0, -1, 1.0, None])
def test_the_engine_count_must_be_a_positive_int(num_engines) -> None:
    with pytest.raises(ValueError, match="at least one engine"):
        VllmDPLoadBalancer(num_engines)


# --------------------------------------------------------------------------
# The load accessor, on the real vllm_v1 scheduler
# --------------------------------------------------------------------------


def _model(*, is_moe: bool) -> BaseModelConfig:
    model = BaseModelConfig(
        num_layers=4,
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
    model._model_name = f"w4_dp_{'moe' if is_moe else 'dense'}"
    return model


def _policy_scheduler(
    patch,
    *,
    is_moe: bool = True,
    attn_dp: int = 2,
    moe_ep: int = 2,
    num_replicas: int = 1,
    num_pipeline_stages: int = 1,
    cluster_type: ClusterType = ClusterType.MONOLITHIC,
    replica_scheduler_config=None,
    cluster_scheduler_config=None,
):
    """Build a real cluster scheduler through the real constructor path."""

    model = _model(is_moe=is_moe)
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
    generator_config = SyntheticRequestGeneratorConfig(
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
        # Admission moves requests into running; the accessor must not count a
        # request twice, and the two populations must add up.
        assert load.running == len(lane._running_requests)
        assert load.waiting == len(lane._request_queue) + len(
            lane._preempted_requests
        )
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
        ("pipeline_parallel", dict(num_pipeline_stages=2), "one co-location Replica"),
        (
            "wrong_replica_scheduler",
            dict(replica_scheduler_config=SarathiSchedulerConfig(
                num_blocks=64, block_size=16, batch_size_cap=4,
                chunk_size=16,
            )),
            "one co-location Replica",
        ),
        (
            "dense_multi_lane",
            dict(is_moe=False, attn_dp=2),
            "monotonic per Replica",
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
    ("is_moe", "attn_dp", "moe_ep"),
    [(True, 2, 2), (True, 1, 1), (False, 1, 1)],
)
def test_the_supported_shapes_construct(is_moe, attn_dp, moe_ep) -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(
            patch, is_moe=is_moe, attn_dp=attn_dp, moe_ep=moe_ep
        )

    assert scheduler._load_balancer is not None
    assert len(scheduler._load_balancer.engine_counts) == attn_dp


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


def test_an_unknown_lane_identity_is_rejected_at_the_report_boundary() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler = _policy_scheduler(patch)

        with pytest.raises(ValueError, match="exact lane index"):
            scheduler.on_replica_batch_end(0.0, scheduler._serving_replica_id, None, None)


# --------------------------------------------------------------------------
# The two base seams stay inert for every existing policy
# --------------------------------------------------------------------------


def _bare_policy(scheduler_type, requests):
    from types import SimpleNamespace

    scheduler = scheduler_type.__new__(scheduler_type)
    # The rotation ordinal W2 made persistent, and the sticky policies' session
    # table and counter, are normally set up by __init__.
    scheduler._request_counter = 0
    scheduler._session_counter = 0
    scheduler._session_to_target_map = {}
    scheduler._cluster_type = ClusterType.MONOLITHIC
    scheduler._num_replicas = 2
    scheduler._replica_dp_size = 2
    scheduler._cluster = SimpleNamespace(replicas={0: object(), 1: object()})
    scheduler._request_queue = list(requests)
    scheduler._replica_schedulers = {
        (replica_id, dp_id): SimpleNamespace(num_pending_requests=0)
        for replica_id in (0, 1)
        for dp_id in (0, 1)
    }
    return scheduler


@pytest.mark.parametrize(
    "scheduler_type",
    [
        RoundRobinClusterScheduler,
        LORClusterScheduler,
        RandomClusterScheduler,
        StickyRoundRobinClusterScheduler,
        StickyLORClusterScheduler,
    ],
)
def test_every_existing_policy_routes_identically_through_schedule_at(
    scheduler_type, monkeypatch
) -> None:
    monkeypatch.setattr(
        "frontier.scheduler.cluster_scheduler.random_cluster_scheduler.randint",
        lambda _low, _high: 0,
    )
    # The sticky policies route by session, so every request carries one.
    requests = [
        Request(
            arrived_at=float(index),
            num_prefill_tokens=8,
            num_decode_tokens=2,
            session_id=index % 3,
        )
        for index in range(6)
    ]

    direct = _bare_policy(scheduler_type, requests).schedule()
    through_seam = _bare_policy(scheduler_type, requests).schedule_at(12.5)

    assert [(rid, lane, request.id) for rid, lane, request in through_seam] == [
        (rid, lane, request.id) for rid, lane, request in direct
    ]


def test_the_default_batch_end_seam_is_inert() -> None:
    class _Bare(BaseClusterScheduler):
        def schedule(self):
            return []

    bare = _Bare.__new__(_Bare)
    before = dict(vars(bare))

    assert bare.on_replica_batch_end(1.0, 0, 0, None) is None
    assert dict(vars(bare)) == before


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
