"""DeepSeek V4 DP synchronization against the native vLLM V2 runner (A13 a, c, d).

The fixture holds the distinct DP synchronization tuples recorded by rank 0 of
every PREFILL and DECODE node of P2-32, P2-64 and P0: the post-all-reduce
vectors, the recording DP rank's request count, its descriptor and
num_tokens_across_dp (built by build_dp_sync_fixture.py under
task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/
evidence/routing/).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.config import global_vars
from frontier.config.config import SimulationConfig
from frontier.config.model_config import BaseModelConfig
from frontier.entities import Batch, DummyForwardBatch, Request
from frontier.entities.batch import DecodeCudaGraphMetadata
from frontier.execution_time_predictor.deepseek_v4 import source_physical_rows
from frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy import (
    IterationSchedulingPolicy,
)
from frontier.scheduler.utils.dp_dummy_forward import (
    _synchronize_dp_descriptors,
    defer_first_layer,
    resolve_first_layer_inputs,
)
from frontier.types import ClusterType

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "dsv4_native_dp_sync_rows.json").read_text()
)
NATIVE_ROWS = [dict(zip(FIXTURE["fields"], row)) for row in FIXTURE["rows"]]
TP = 8


def v4_config():
    profile = SimpleNamespace(profile_id="deepseek_v4")
    return SimpleNamespace(replica_config=SimpleNamespace(
        model_config=SimpleNamespace(
            is_moe=True, get_model_architecture_profile=lambda: profile,
        )
    ))


@pytest.fixture
def native_graphs(monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_and_piecewise")
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", FIXTURE["capture_sizes"])


def policy(cluster_type=ClusterType.DECODE):
    owner = IterationSchedulingPolicy()
    owner._cluster_type = cluster_type
    owner._max_num_running_reqs = FIXTURE["max_full_num_reqs"]
    return owner


def lane_scheduler(owner):
    return SimpleNamespace(get_replica_scheduler=lambda replica_id, lane_id: owner)


def padded_num_reqs(batch):
    metadata = batch.decode_cuda_graph_metadata
    return metadata.padded_decode_batch_size - metadata.original_decode_batch_size + batch.size


class RecordedLane:
    """A DP lane known only through the native all-reduce vectors."""

    def __init__(self, size, metadata):
        self.size = size
        self.decode_cuda_graph_metadata = metadata

    get_decode_cuda_graph_runtime_mode = Batch.get_decode_cuda_graph_runtime_mode


def recorded_lanes(row):
    lanes = {}
    for lane_id, (tokens, desired, uniform) in enumerate(zip(
        row["local_num_tokens"], row["desired_modes"], row["uniform_token_counts"]
    )):
        if lane_id == row["recording_dp_rank"]:
            size = row["recording_num_reqs"]
        else:
            # Only the recording lane's request count is known. A PIECEWISE
            # or eager lane keeps its own count, which the row does not check.
            size = tokens if uniform == 1 else 1
        decode_rows = size if uniform == 1 else 0
        lanes[lane_id] = RecordedLane(size, DecodeCudaGraphMetadata(
            config_mode="full_and_piecewise",
            runtime_mode=FIXTURE["mode_order"][desired],
            capture_hit=desired != 0,
            is_mixed_batch=uniform == 0,
            original_total_tokens=tokens,
            padded_total_tokens=tokens,
            original_decode_batch_size=decode_rows,
            padded_decode_batch_size=decode_rows,
        ))
    return lanes


def test_fixture_covers_the_three_cases_and_both_roles():
    assert len(NATIVE_ROWS) == 2080
    assert sum(row["forwards"] for row in NATIVE_ROWS) == 245088
    runs = {name.split("/")[0] for name in FIXTURE["source_names"]}
    assert runs == {
        "dsv4-h200-p2-profile-20261025-01",
        "dsv4-h200-p2-64-profile-20261025-01",
        "dsv4-h800-p0-profile-20261024-04",
    }
    assert {name.split("/", 1)[1] for name in FIXTURE["source_names"]} == {
        "decode/dummy", "decode/serving", "prefill/dummy", "prefill/serving",
    }


def test_synchronization_reproduces_every_recorded_descriptor(native_graphs):
    """A13 (a): every serving and dummy forward of the three cases."""
    owner = lane_scheduler(policy())
    mismatches = []
    for index, row in enumerate(NATIVE_ROWS):
        lanes = recorded_lanes(row)
        uniform_lanes = [
            lane for lane, uniform in zip(lanes.values(), row["uniform_token_counts"])
            if uniform == 1
        ]
        assert all(lane.size == lane.decode_cuda_graph_metadata.original_total_tokens
                   for lane in uniform_lanes)
        _synchronize_dp_descriptors(owner, 0, lanes)
        recorder = lanes[row["recording_dp_rank"]]
        widths = [
            lane.decode_cuda_graph_metadata.get_effective_total_tokens_for_compute()
            for lane in lanes.values()
        ]
        # A PIECEWISE descriptor carries no request count: it pads tokens only.
        expected_reqs = row["descriptor_num_reqs"] or row["recording_num_reqs"]
        observed = (
            [lane.get_decode_cuda_graph_runtime_mode() for lane in lanes.values()],
            widths,
            widths[row["recording_dp_rank"]],
            padded_num_reqs(recorder),
        )
        expected = (
            [row["descriptor_mode"]] * len(lanes),
            row["num_tokens_across_dp"],
            row["descriptor_num_tokens"],
            expected_reqs,
        )
        if observed != expected:
            mismatches.append((index, observed, expected))
    assert mismatches == []


def decode_requests(count, context=512):
    requests = [Request(0.0, context, 64) for _ in range(count)]
    for request in requests:
        request._is_prefill_complete = True
        request._num_processed_tokens = context
    return requests


def decode_lane(num_decodes, *, first_tails=0, context=512):
    """A DECODE lane: steady decodes beside one-token first tails.

    A first tail is the prompt's last token, computed on the decode role after
    the KV transfer of the rest of the prompt.
    """
    tails = [Request(0.0, context + 1, 64) for _ in range(first_tails)]
    for request in tails:
        request._num_processed_tokens = context
    requests = tails + decode_requests(num_decodes, context)
    return Batch(
        0, requests, [1] * len(requests), is_moe=True,
        num_context_tokens=[context] * len(requests),
        request_is_decoding=[False] * first_tails + [True] * num_decodes,
    )


def prefill_lane(tokens):
    return Batch(
        0, [Request(0.0, tokens, 64)], [tokens], is_moe=True,
        num_context_tokens=[0], request_is_decoding=[False],
    )


def dummy_lane():
    return DummyForwardBatch(0, 0, 0)


def native_tuples(lanes):
    """The recorded tuples of the forwards whose lanes had these local inputs."""
    key = (
        [lane.decode_cuda_graph_metadata.original_total_tokens for lane in lanes.values()],
        [FIXTURE["mode_order"].index(lane.get_decode_cuda_graph_runtime_mode())
         for lane in lanes.values()],
        [0 if lane.decode_cuda_graph_metadata.is_mixed_batch else 1
         for lane in lanes.values()],
    )
    return [
        row for row in NATIVE_ROWS
        if (row["local_num_tokens"], row["desired_modes"], row["uniform_token_counts"]) == key
    ]


def synchronized_forward(lanes, cluster_type=ClusterType.DECODE):
    owner = policy(cluster_type)
    for lane in lanes.values():
        lane.decode_cuda_graph_metadata = owner._build_decode_cuda_graph_metadata(lane)
    native = native_tuples(lanes)
    _synchronize_dp_descriptors(lane_scheduler(owner), 0, lanes)
    for row in native:
        assert [lane.get_decode_cuda_graph_runtime_mode() for lane in lanes.values()] == (
            [row["descriptor_mode"]] * len(lanes)
        )
        assert [lane.get_effective_total_tokens_for_compute(ClusterType.MONOLITHIC)
                for lane in lanes.values()] == row["num_tokens_across_dp"]
    rows = source_physical_rows(
        SimpleNamespace(_replica_config=SimpleNamespace(
            attn_dp=len(lanes), attn_tensor_parallel_size=TP,
        )),
        lanes,
    )
    return {
        "modes": [lane.get_decode_cuda_graph_runtime_mode() for lane in lanes.values()],
        "widths": [lane.get_effective_total_tokens_for_compute(ClusterType.MONOLITHIC)
                   for lane in lanes.values()],
        "requests": [padded_num_reqs(lane) for lane in lanes.values()],
        "routed_rows": rows,
        "native_tuples": len(native),
    }


# A13 (c): Frontier batches through the local classifier, the synchronization
# and the routed-row count of the V4 source cohort.
FORWARDS = {
    # The tuple of counter row 428: a first tail on lane 0 makes the forward
    # PIECEWISE, so lane 1's steady decodes pad tokens and not requests.
    "row_428": (
        lambda: {0: decode_lane(11, first_tails=1), 1: decode_lane(8)},
        ClusterType.DECODE,
        {"modes": ["PIECEWISE"] * 2, "widths": [16, 16], "requests": [12, 8],
         "routed_rows": 32},
    ),
    "steady_decodes": (
        lambda: {0: decode_lane(12), 1: decode_lane(8)},
        ClusterType.DECODE,
        {"modes": ["FULL"] * 2, "widths": [16, 16], "requests": [16, 16],
         "routed_rows": 32},
    ),
    "first_tail_alone": (
        lambda: {0: decode_lane(0, first_tails=1), 1: dummy_lane()},
        ClusterType.DECODE,
        {"modes": ["PIECEWISE"] * 2, "widths": [1, 1], "requests": [1, 1],
         "routed_rows": 2 * TP},
    ),
    "first_tail_with_decodes": (
        lambda: {0: decode_lane(3, first_tails=1), 1: decode_lane(5)},
        ClusterType.DECODE,
        {"modes": ["PIECEWISE"] * 2, "widths": [8, 8], "requests": [4, 5],
         "routed_rows": 16},
    ),
    "dummy_beside_decodes": (
        lambda: {0: decode_lane(5), 1: dummy_lane()},
        ClusterType.DECODE,
        {"modes": ["FULL"] * 2, "widths": [8, 8], "requests": [8, 8],
         "routed_rows": 16},
    ),
    "all_dummies": (
        lambda: {0: dummy_lane(), 1: dummy_lane()},
        ClusterType.DECODE,
        {"modes": ["FULL"] * 2, "widths": [1, 1], "requests": [1, 1],
         "routed_rows": 2 * TP},
    ),
    "prefill_chunk_beside_dummy": (
        lambda: {0: prefill_lane(6144), 1: dummy_lane()},
        ClusterType.PREFILL,
        {"modes": ["NONE"] * 2, "widths": [6144, 1], "requests": [1, 1],
         "routed_rows": 6144 + TP},
    ),
    "prefill_dummies": (
        lambda: {0: dummy_lane(), 1: dummy_lane()},
        ClusterType.PREFILL,
        {"modes": ["FULL"] * 2, "widths": [1, 1], "requests": [1, 1],
         "routed_rows": 2 * TP},
    ),
    "short_prefill_beside_dummy": (
        lambda: {0: prefill_lane(100), 1: dummy_lane()},
        ClusterType.PREFILL,
        {"modes": ["PIECEWISE"] * 2, "widths": [104, 104], "requests": [1, 1],
         "routed_rows": 208},
    ),
    "four_lanes_unequal_capture_sizes": (
        lambda: {0: decode_lane(3), 1: decode_lane(17), 2: dummy_lane(), 3: decode_lane(40)},
        ClusterType.DECODE,
        {"modes": ["FULL"] * 4, "widths": [40] * 4, "requests": [40] * 4,
         "routed_rows": 160},
    ),
    "four_prefill_lanes_unequal_capture_sizes": (
        lambda: {0: prefill_lane(100), 1: prefill_lane(30), 2: dummy_lane(), 3: dummy_lane()},
        ClusterType.PREFILL,
        {"modes": ["PIECEWISE"] * 4, "widths": [104] * 4, "requests": [1] * 4,
         "routed_rows": 4 * 104},
    ),
    "four_prefill_lanes_one_eager": (
        lambda: {0: prefill_lane(100), 1: prefill_lane(30), 2: dummy_lane(),
                 3: prefill_lane(6144)},
        ClusterType.PREFILL,
        {"modes": ["NONE"] * 4, "widths": [100, 30, 1, 6144], "requests": [1] * 4,
         "routed_rows": 104 + 32 + TP + 6144},
    ),
}


@pytest.mark.parametrize("name", FORWARDS)
def test_frontier_forward_takes_the_native_descriptor(name, native_graphs):
    build, cluster_type, expected = FORWARDS[name]
    result = synchronized_forward(build(), cluster_type)
    assert {key: result[key] for key in expected} == expected


def test_row_428_and_its_steady_decodes_are_recorded_natively(native_graphs):
    """The two decode forwards above match recorded P2-32 serving tuples."""
    for name in ("row_428", "steady_decodes", "first_tail_alone", "all_dummies"):
        build, cluster_type, _ = FORWARDS[name]
        assert synchronized_forward(build(), cluster_type)["native_tuples"] > 0, name


# A13 (d): the first layer of every lane is placed after the DP rendezvous.
def rendezvous_room(lanes, cluster_type=ClusterType.DECODE):
    """`lanes` maps a lane to (batch, entry s, pre-DP ms, attention ms, preparation ms)."""
    costs = {id(batch): spec for batch, *spec in lanes.values()}

    def predict(batch, *args, **kwargs):
        pre_dp, attention, preparation = costs[id(batch)][1:]
        return SimpleNamespace(
            get_single_layer_pre_dp_time=lambda: pre_dp,
            get_single_layer_attention_scope_time=lambda: attention,
            forward_preparation_time=preparation,
            forward_launch_time=0.5,
        )

    scheduler = SimpleNamespace(
        _cluster_type=cluster_type,
        _sync_kind=cluster_type.name.lower(),
        _config=v4_config(),
        _predictor=SimpleNamespace(
            _num_layers_per_pipeline_stage=4, predict_stage_execution_time=predict,
        ),
        get_pipeline_stage_layer_bounds=lambda stage_id, num_layers: (0, num_layers),
    )
    room = {"batches": {}, "arrival_times": {}}
    for lane_id, (batch, entry, *_) in lanes.items():
        defer_first_layer(scheduler, entry, 0, 0, batch, lane_id,
                          prefill=cluster_type == ClusterType.PREFILL)
        room["batches"][lane_id] = batch
        room["arrival_times"][lane_id] = entry
    resolve_first_layer_inputs(scheduler, 0, 0, 0, room)
    return room


@pytest.mark.parametrize("lanes,arrivals", [
    # F009: an early lane's first layer waits for the late peer.
    pytest.param({0: (0.0, 0.0, 10.0, 0.0), 1: (0.100, 0.0, 10.0, 0.0)},
                 [0.110, 0.110], id="peer_wait"),
    # Each lane prepares its inputs after the rendezvous.
    pytest.param({0: (0.0, 0.0, 10.0, 2.0), 1: (0.100, 0.0, 10.0, 1.0)},
                 [0.112, 0.111], id="preparation"),
    # F017: 30 ms zeroing from stage start is hidden by the wait (110 ms, not 140 ms).
    pytest.param({0: (0.0, 30.0, 10.0, 0.0), 1: (0.100, 0.0, 10.0, 0.0)},
                 [0.110, 0.110], id="zeroing_hidden"),
    # With no peer wait the zeroing is still paid; a lane without new blocks is not.
    pytest.param({0: (0.0, 30.0, 10.0, 0.0), 1: (0.0, 0.0, 10.0, 0.0)},
                 [0.040, 0.010], id="zeroing_paid"),
    # Zeroing partly hidden by the wait.
    pytest.param({0: (0.0, 120.0, 10.0, 0.0), 1: (0.100, 0.0, 10.0, 0.0)},
                 [0.130, 0.110], id="zeroing_partly_hidden"),
])
def test_first_layer_starts_after_the_dp_rendezvous(lanes, arrivals):
    batches = {0: decode_lane(4), 1: decode_lane(2)}
    room = rendezvous_room({
        lane_id: (batches[lane_id], *spec) for lane_id, spec in lanes.items()
    })
    assert [room["arrival_times"][lane_id] for lane_id in (0, 1)] == pytest.approx(arrivals)
    rendezvous = max(entry for entry, *_ in lanes.values())
    for lane_id, (_, _, _, preparation) in lanes.items():
        batch = batches[lane_id]
        assert batch._forward_launch_start_time == pytest.approx(
            rendezvous + preparation * 1e-3
        )
        assert batch._forward_launch_time == pytest.approx(0.5e-3)
        assert not batch._first_layer_pricing_deferred


def test_real_lane_zeroing_beside_dummy_lanes():
    real, dummy, idle = decode_lane(4), dummy_lane(), dummy_lane()
    room = rendezvous_room({
        0: (real, 0.0, 80.0, 10.0, 0.0),
        1: (dummy, 0.050, 0.0, 3.0, 0.0),
        2: (idle, 0.020, 0.0, 3.0, 0.0),
    })
    # The rendezvous is the latest entry, 50 ms; the real lane's zeroing ends at 80 ms.
    assert [room["arrival_times"][lane] for lane in (0, 1, 2)] == pytest.approx(
        [0.090, 0.053, 0.053]
    )
    assert dummy.requests == [] and idle.requests == []


def test_only_real_prefill_lanes_extend_the_prefill_ledger():
    real, dummy = prefill_lane(6144), dummy_lane()
    real._prefill_model_execution_components_ms_by_stage = {0: []}
    rendezvous_room(
        {0: (real, 0.0, 30.0, 40.0, 0.0), 1: (dummy, 0.010, 0.0, 3.0, 0.0)},
        ClusterType.PREFILL,
    )
    # Each first-layer component is charged once: zeroing, then attention.
    assert real._prefill_model_execution_components_ms_by_stage == {0: [30.0, 40.0]}
    assert not hasattr(dummy, "_prefill_model_execution_components_ms_by_stage")


def test_v4_lanes_keep_their_own_launch_start_through_the_room(monkeypatch):
    """The V4 room replaces the launch-start alignment of the other models."""
    from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
        RoundRobinClusterScheduler,
    )
    from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
    from frontier.scheduler.utils.sync_state import initialize_sync_waiting_rooms

    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "none")
    real, dummy = decode_lane(4), dummy_lane()
    costs = {id(real): (0.0, 10.0, 2.0, 4.0), id(dummy): (0.0, 10.0, 1.0, 0.0)}

    def predict(batch, *args, **kwargs):
        pre_dp, attention, preparation, launch = costs[id(batch)]
        return SimpleNamespace(
            get_single_layer_pre_dp_time=lambda: pre_dp,
            get_single_layer_attention_scope_time=lambda: attention,
            forward_preparation_time=preparation,
            forward_launch_time=launch,
        )

    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = ClusterType.DECODE
    scheduler._config = v4_config()
    initialize_sync_waiting_rooms(scheduler)
    scheduler._forward_sync_state = ForwardSyncState()
    scheduler._replica_dp_size = 2
    scheduler._predictor = SimpleNamespace(
        _num_layers_per_pipeline_stage=4, predict_stage_execution_time=predict,
    )
    scheduler._uses_shared_decode_layer_path = lambda *_: True
    scheduler._replica_schedulers = {
        (0, 0): SimpleNamespace(engine_loop_idle=False),
        (0, 1): SimpleNamespace(engine_loop_idle=False),
    }
    waves = []
    scheduler._on_decode_ep_wave_ready = lambda **kwargs: waves.append(kwargs["time"]) or []
    entries = {0: (real, 0.0), 1: (dummy, 0.100)}
    for lane_id, (batch, entry) in entries.items():
        batch.set_global_id(lane_id)
        batch._forward_cohort_id = 0
        batch._forward_cohort_provisional_id = 0
        (event,) = defer_first_layer(scheduler, entry, 0, 0, batch, lane_id, prefill=False)
        assert event.time == entry
    for lane_id, (batch, entry) in entries.items():
        scheduler.on_decode_sync(entry, 0, 0, batch, lane_id, "pre_moe", 0, 0.0)
    assert real._forward_launch_start_time == pytest.approx(0.102)
    assert dummy._forward_launch_start_time == pytest.approx(0.101)
    # Device arrival 0.112 s; the launch bound, 0.102 + 1/4 * 4 ms, is earlier.
    assert waves == [pytest.approx(0.112)]


def validation_owner(model_name, *, capture_sizes=(1, 2, 4), spec_decode=False):
    model_config = BaseModelConfig.create_from_name(model_name)
    replica_config = SimpleNamespace(
        speculative_decoding_config=SimpleNamespace(enabled=spec_decode),
    )
    return SimpleNamespace(
        decode_cuda_graph_mode="full_and_piecewise",
        cudagraph_capture_sizes=None if capture_sizes is None else list(capture_sizes),
        allow_spec_decode_cuda_graph_diagnostic=False,
        use_cuda_graph=False,
        sys_arch="pd-disaggregation",
        cluster_config=SimpleNamespace(replica_config=replica_config),
        _get_model_config_for_is_moe=lambda: model_config,
    )


@pytest.mark.parametrize("overrides,message", [
    ({"model_name": "zai-org/GLM-5.3"}, "deepseek_v4 architecture profile only"),
    ({"capture_sizes": None}, "requires explicit cudagraph_capture_sizes"),
    ({"spec_decode": True}, "speculative decoding is not supported"),
])
def test_full_and_piecewise_configuration_is_v4_only(overrides, message):
    options = {"model_name": "deepseek-ai/DeepSeek-V4-Flash", **overrides}
    with pytest.raises(ValueError, match=message):
        SimulationConfig._validate_cuda_graph_config(validation_owner(**{
            "model_name": options.pop("model_name"), **options,
        }))


def test_full_and_piecewise_configuration_accepts_v4():
    SimulationConfig._validate_cuda_graph_config(
        validation_owner("deepseek-ai/DeepSeek-V4-Flash")
    )
