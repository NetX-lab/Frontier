"""Real-runtime wiring for the opt-in vLLM-style DP placement policy.

The state machine itself is covered by `tests/unit/test_vllm_dp_load_balancer.py`
against reference-derived expectations. What can only be shown by running the
simulator is the wiring: that `ClusterScheduleEvent` is what supplies the
routing time, that each report stands for one reference engine iteration --
an admission while the pipeline has room, or a completion -- with the lane's
post-step load and the key of the forward it describes, that a lane with a
batch in flight admits only when the reference engine runs an iteration, and
that the policy introduces no event that keeps a drained run alive.

The child process runs each configuration twice, once with the policy and once
with a comparison run that differs only in the cluster scheduler policy or, for
the discriminating case, only in how the policy reports. Execution time comes
from the dummy predictor: placement here is decided by the balancer, not by
latency realism, and both runs see the same durations.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_child(tmp_path: Path, case: str) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), case],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(repo_root), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    (tmp_path / f"{case}.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-15000:]
    return json.loads((tmp_path / f"{case}_evidence.json").read_text())


def _assert_reports_follow_engine_iterations(run: dict, num_pipeline_stages: int) -> int:
    """Check every report against the reference engine iteration it stands for.

    Reference `step_with_batch_queue`: an iteration that schedules a batch and
    still has pipeline room publishes at once; one that fills the pipeline
    publishes after applying its oldest output; one with nothing new to
    schedule applies the oldest output by itself. The key is the forward the
    iteration launches, which peer engines share. Returns the number of
    admissions published on their own.
    """

    records = run["records"]
    forward: dict[int, int] = {}
    starts_by_lane: dict[int, int] = defaultdict(int)
    for record in records:
        if record["kind"] != "stage0":
            continue
        if record["group"] is None:
            # A dense Replica shares no forward; a lane's forwards are its own
            # stage-0 starts.
            forward[record["batch"]] = starts_by_lane[record["lane"]]
            starts_by_lane[record["lane"]] += 1
        else:
            forward[record["batch"]] = record["group"]

    held: dict[int, int] = {}
    last_forward: dict[int, int] = defaultdict(lambda: -1)
    last_key: dict[int, int] = defaultdict(lambda: -1)
    admission_only = 0
    for record in records:
        if record["kind"] == "stage0":
            continue
        lane = record["lane"]
        if record["kind"] == "scheduled":
            last_forward[lane] = forward[record["batch"]]
            if record["running_after"] >= num_pipeline_stages:
                assert record["report"] is None, record
                assert lane not in held, record
                held[lane] = record["batch"]
                continue
            assert record["report"] is not None, record
            admission_only += 1
            expected = forward[record["batch"]]
        else:
            assert record["report"] is not None, record
            expected = forward[held.pop(lane)] if lane in held else None
        key = record["report"][0]
        if expected is None:
            # An iteration after the lane's last forward, with nothing new.
            assert key > last_forward[lane], (record, last_forward[lane])
        else:
            assert key == expected, (record, expected)
        assert key >= last_key[lane], (record, last_key[lane])
        last_key[lane] = key
    assert not held, held
    return admission_only


def _admissions_while_the_engine_blocks(run: dict) -> list[dict]:
    """Return the admissions a reference engine could not make at that instant.

    Reference `step_with_batch_queue` (`vllm/v1/engine/core.py:364-424`): an
    iteration that schedules nothing, or fills the pipeline, blocks on the
    oldest in-flight batch, and a request that arrives meanwhile waits for the
    next iteration. So a lane with a batch in flight admits only in the pass of
    a previous admission or at the end of its oldest batch; an idle lane admits
    at any time. Iterations take no host time here, as in Frontier. A stale
    drop of the oldest batch removes its remaining work, so it ends the block
    as its completion would.
    """

    in_flight: dict[int, list[int]] = defaultdict(list)
    last_iteration: dict[int, float] = {}
    blocked = []
    for record in run["records"]:
        if record["kind"] == "stage0":
            continue
        lane, time = record["lane"], record["time"]
        batches = in_flight[lane]
        if record["kind"] == "end":
            if record["batch"] in batches:
                if record["batch"] == batches[0]:
                    last_iteration[lane] = time
                batches.remove(record["batch"])
            continue
        if batches and last_iteration[lane] != time:
            blocked.append(record)
        batches.append(record["batch"])
        last_iteration[lane] = time
    return blocked


def _count_admissions_into_peer_forwards(run: dict) -> int:
    """Count published admissions whose batch joins a forward a peer lane
    already started at stage 0, the one case where the key is the open forward
    rather than the next one."""

    forward = {
        record["batch"]: record["group"]
        for record in run["records"]
        if record["kind"] == "stage0"
    }
    opened_by: dict[int, int] = {}
    joins = 0
    for record in run["records"]:
        if record["kind"] == "stage0":
            opened_by.setdefault(record["group"], record["lane"])
        elif record["kind"] == "scheduled" and record["report"] is not None:
            opener = opened_by.get(forward[record["batch"]])
            joins += opener is not None and opener != record["lane"]
    return joins


def _assert_completions_report_post_step_load(run: dict) -> None:
    """Every completion report carries that completion's post-step load. When a
    deep pipeline defers every release, the terminal-release report carries the
    load after the deferred free."""

    completions = [
        record
        for record in run["records"]
        if record["kind"] == "end" and record["source"] == "completion"
    ]
    assert completions
    assert all(record["report"][1] == record["post_step"] for record in completions)
    if any(record["pre_step"] != record["post_step"] for record in completions):
        return
    # Deep pipelines defer every release past the completion, and the
    # terminal-release report then publishes the change.
    terminal_releases = [
        record
        for record in run["records"]
        if record["kind"] == "end" and record["source"] == "terminal_release"
    ]
    assert terminal_releases
    assert all(
        record["report"][1] == record["lane_load"] for record in terminal_releases
    )


def _assert_every_held_key_is_reported(run: dict) -> list[str]:
    """A held admission key is published by the next end record, and by no later one.

    Walks every record except stage-0 starts. Returns the source of each end
    record that published a key its lane was holding.
    """

    pending: dict[int, int] = {}
    last_key: dict[int, int] = defaultdict(lambda: -1)
    consumed: list[str] = []
    for record in run["records"]:
        if record["kind"] == "stage0":
            continue
        lane = record["lane"]
        if record["kind"] == "scheduled":
            if record["report"] is None:
                assert lane not in pending, record
                pending[lane] = record["held_key"]
            else:
                key = record["report"][0]
                assert key >= last_key[lane], (record, last_key[lane])
                last_key[lane] = key
            continue
        assert record["report"] is not None, record
        key = record["report"][0]
        if lane in pending:
            assert key == pending.pop(lane), record
            consumed.append(record["source"])
        assert key >= last_key[lane], (record, last_key[lane])
        last_key[lane] = key
    assert not pending, pending
    return consumed


def _assert_final_counts_match_lanes(run: dict) -> None:
    assert run["final_engine_counts"] == run["final_lane_loads"]


def _assert_pp1_keys_are_ordered_forwards(run: dict) -> None:
    """At PP=1 the key is ordered as the capability guard assumes: never
    decreasing, and equal only for peer lanes of one shared forward."""

    reports = [
        record
        for record in run["records"]
        if record["kind"] != "stage0" and record["report"] is not None
    ]
    keys = [record["report"][0] for record in reports]
    assert keys == sorted(keys)
    lanes_by_key: dict[int, list[int]] = defaultdict(list)
    for record in reports:
        lanes_by_key[record["report"][0]].append(record["lane"])
    for key, lanes in lanes_by_key.items():
        assert len(lanes) == len(set(lanes)), (key, lanes)


def _assert_run_conserves_work(run: dict) -> None:
    assert run["completed_requests"] == run["num_requests"] > 0
    assert run["tokens_conserved"]
    assert run["lanes_released"]
    assert run["stage_contexts_released"]


def test_dp_placement_runs_and_reports_post_step_load(tmp_path):
    evidence = _run_child(tmp_path, "moe_dp2")
    policy = evidence["vllm_load_balancing"]
    baseline = evidence["round_robin"]

    # The run has to reach the shape under test.
    assert policy["num_lanes"] == 2
    _assert_run_conserves_work(policy)

    # 1. The routing time comes from ClusterScheduleEvent, not retained state.
    #    Every recorded routing time is one of the cluster schedule times, and
    #    the policy's own `schedule()` would have raised had it been used.
    assert policy["routing_times"]
    assert policy["routing_times"] == policy["cluster_schedule_times"]

    # 2. Placement, derived independently: four requests arrive together, no
    #    report can precede the first routing, so the estimate is empty and only
    #    the local reservation moves the choice.
    assert policy["first_four_lanes"] == [0, 1, 0, 1]

    # 3. At PP=1 no admission leaves a pipeline slot, so every report is a
    #    completion keyed by the forward that completed.
    assert _assert_reports_follow_engine_iterations(policy, 1) == 0
    _assert_pp1_keys_are_ordered_forwards(policy)

    # 4. The reported load is the post-step state.
    _assert_completions_report_post_step_load(policy)

    # 5. The run drains rather than being kept alive by a heartbeat.
    assert policy["makespan"] > 0
    _assert_run_conserves_work(baseline)


def test_placement_follows_published_load_where_round_robin_cannot(tmp_path):
    """Spread the arrivals so snapshots land between them, and skew the load.

    This is the discriminating case at PP=1: the two policies see identical
    arrivals, identical durations and identical lane capacity, so any
    difference in placement comes from reading the published load.
    """

    evidence = _run_child(tmp_path, "moe_dp2_online")
    policy = evidence["vllm_load_balancing"]
    baseline = evidence["round_robin"]

    assert policy["completed_requests"] == policy["num_requests"] == 6
    assert baseline["completed_requests"] == baseline["num_requests"] == 6

    # 1. The routing time is the cluster schedule time, now at several distinct
    #    instants rather than one, so a retained or stale time would show up.
    assert len(set(policy["cluster_schedule_times"])) > 1
    assert policy["routing_times"] == policy["cluster_schedule_times"]
    assert baseline["routing_times"] == baseline["cluster_schedule_times"]

    # 2. Round-robin alternates because it cannot see load. The policy places
    #    strictly fewer requests on the lane that is still draining the one long
    #    request, which is the whole point of reading the snapshot.
    assert baseline["placements"] == [0, 1, 0, 1, 0, 1]
    assert policy["placements"] != baseline["placements"]
    assert policy["placements"][:2] == [0, 1]
    assert policy["placements"].count(0) < baseline["placements"].count(0)

    # 3. The report key stays ordered across a much longer run, and peer lanes of
    #    one shared forward remain the only source of equal keys.
    assert _assert_reports_follow_engine_iterations(policy, 1) == 0
    _assert_pp1_keys_are_ordered_forwards(policy)
    assert sum(1 for record in policy["records"] if record["kind"] == "end") > 10

    # 4. Still the post-step load.
    _assert_completions_report_post_step_load(policy)


def test_a_single_lane_shape_routes_everything_to_lane_zero(tmp_path):
    evidence = _run_child(tmp_path, "dense_dp1")
    policy = evidence["vllm_load_balancing"]

    assert policy["num_lanes"] == 1
    _assert_run_conserves_work(policy)
    assert set(policy["first_four_lanes"]) == {0}
    assert _assert_reports_follow_engine_iterations(policy, 1) == 0


@pytest.mark.parametrize(
    ("case", "num_pipeline_stages"),
    [
        ("dense_dp1_pp2", 2),
        ("moe_dp2_pp2", 2),
        ("moe_dp2_pp2_online", 2),
        ("moe_dp1_pp3", 3),
        ("moe_dp2_pp3", 3),
        ("moe_dp2_pp4", 4),
    ],
)
def test_pipeline_parallel_shapes_report_once_per_engine_iteration(
    tmp_path, case, num_pipeline_stages
):
    evidence = _run_child(tmp_path, case)
    policy = evidence["vllm_load_balancing"]
    baseline = evidence["round_robin"]

    # 1. Every request completes, every token is processed once, and no lane
    #    or stage context still owns work.
    _assert_run_conserves_work(policy)
    _assert_run_conserves_work(baseline)

    # 2. The routing time is still the cluster schedule time.
    assert policy["routing_times"] == policy["cluster_schedule_times"]

    # 3. Each report stands for one engine iteration under its forward's key,
    #    and every lane's cold fill publishes an admission before anything
    #    completes.
    admission_only = _assert_reports_follow_engine_iterations(
        policy, num_pipeline_stages
    )
    assert admission_only >= policy["num_lanes"]

    # 4. With a batch in flight, a lane admits only when the reference engine
    #    runs an iteration.
    assert _admissions_while_the_engine_blocks(policy) == []

    # 5. Completions still carry the post-step load.
    _assert_completions_report_post_step_load(policy)
    _assert_every_held_key_is_reported(policy)
    _assert_final_counts_match_lanes(policy)


@pytest.mark.parametrize(
    ("case", "num_pipeline_stages"),
    [("moe_dp2_pp2_stagger", 2), ("moe_dp2_pp3_stagger", 3)],
)
def test_an_admission_into_a_started_forward_reports_under_that_forward(
    tmp_path, case, num_pipeline_stages
):
    evidence = _run_child(tmp_path, case)
    policy = evidence["vllm_load_balancing"]

    _assert_run_conserves_work(policy)
    # Premise: staggered arrivals make a lane publish an admission into a
    # forward its peer has already started. The burst shapes above never do.
    assert _count_admissions_into_peer_forwards(policy) >= 1
    _assert_reports_follow_engine_iterations(policy, num_pipeline_stages)
    assert _admissions_while_the_engine_blocks(policy) == []


def test_schedule_time_reports_decide_a_probe_that_completion_reports_cannot(
    tmp_path,
):
    """Plan §18.6 under PP=2, against the completion-reporting control.

    The control differs from the policy only in its two seams, so the probe's
    placement differs only because of what was published before it arrived.
    """

    evidence = _run_child(tmp_path, "moe_dp2_pp2_discriminating")
    fixed = evidence["vllm_load_balancing"]
    control = evidence["completion_reporting_control"]

    for run in (fixed, control):
        _assert_run_conserves_work(run)
        # Premise: the burst is routed from reservations alone, after the
        # first collection publish of empty counts ...
        assert run["selections"][0]["snapshot"] == [[0, 0], [0, 0]]
        assert run["placements"][:5] == [0, 1, 0, 1, 0]
        # ... and nothing completes before the probe arrives.
        first_completion = min(
            record["time"] for record in run["records"] if record["kind"] == "end"
        )
        assert first_completion > 1.1
        # The probe reaches lane 0 while its batch is in flight, and waits.
        assert _admissions_while_the_engine_blocks(run) == []

    probe = {name: run["selections"][5] for name, run in evidence.items()}
    assert probe["vllm_load_balancing"]["time"] == pytest.approx(1.1)
    assert probe["completion_reporting_control"]["time"] == pytest.approx(1.1)

    # The policy published each lane's first admission: lane 0 runs its three
    # short requests (score 3); lane 1 runs one chunk of the long prompt while
    # the short request waits (score 4 + 1). The probe goes to lane 0.
    first_admissions = {}
    for record in fixed["records"]:
        if record["kind"] == "scheduled" and record["report"] is not None:
            first_admissions.setdefault(record["lane"], record["report"][1])
    assert first_admissions == {0: [0, 3], 1: [1, 1]}
    assert probe["vllm_load_balancing"]["snapshot"] == [[0, 3], [1, 1]]
    assert probe["vllm_load_balancing"]["engine"] == 0

    # The control reported nothing yet, so the frontend still holds its own
    # reservations (score 12 against 8) and sends the probe to lane 1.
    assert all(
        record["report"] is None
        for record in control["records"]
        if record["kind"] == "scheduled"
    )
    assert probe["completion_reporting_control"]["snapshot"] == [[3, 0], [2, 0]]
    assert probe["completion_reporting_control"]["engine"] == 1


def test_a_lane_joining_after_its_placeholder_completes_the_forward(tmp_path):
    """Issue W9-04 on the policy's four-lane MoE shape, under both policies.

    Lane 0 waits in the first MoE room of forward 0 while lane 1 is still in
    attention, so the room places placeholders for the idle lanes 2 and 3.
    Request 2 then reaches lane 2, which joins forward 0 because it is not yet
    sealed. The room used to count lane 2's stale placeholder and dispatch when
    lane 1 arrived. Lane 2's own batch then opened a room that its busy peers
    never enter, and both runs stalled with no request complete.
    """

    evidence = _run_child(tmp_path, "moe_dp4_late_join")

    for run in evidence.values():
        assert run["placements"] == [0, 1, 2]
        first_groups = {}
        for record in run["records"]:
            if record["kind"] == "stage0":
                first_groups.setdefault(record["lane"], record["group"])
        # The race is reached: lane 2 joined forward 0, and the placeholder it
        # had been given there was withdrawn.
        assert first_groups == {0: 0, 1: 0, 2: 0}
        assert run["withdrawn_placeholder_lanes"] == [2]
        _assert_run_conserves_work(run)


@pytest.mark.parametrize(
    "case",
    ["dense_dp1_pp4_kv_pressure", "moe_dp2_pp4_kv_pressure"],
)
def test_a_stale_dropped_batch_reports_the_key_its_admission_held(tmp_path, case):
    evidence = _run_child(tmp_path, case)
    policy = evidence["vllm_load_balancing"]
    baseline = evidence["round_robin"]

    sources = _assert_every_held_key_is_reported(policy)
    assert policy["num_preemptions"] >= 1
    assert policy["stale_drops"]
    assert "stale_drop" in sources
    assert _admissions_while_the_engine_blocks(policy) == []
    _assert_run_conserves_work(policy)
    _assert_run_conserves_work(baseline)
    _assert_final_counts_match_lanes(policy)
    assert policy["routing_times"] == policy["cluster_schedule_times"]


def test_a_deferred_terminal_release_is_reported(tmp_path):
    pp4 = _run_child(tmp_path, "moe_dp2_pp4_release")["vllm_load_balancing"]
    pp2 = _run_child(tmp_path, "moe_dp2_pp2_release")["vllm_load_balancing"]

    assert any(
        record["kind"] == "end" and record["source"] == "terminal_release"
        for record in pp4["records"]
    )
    assert not any(
        record["kind"] == "end" and record["source"] == "terminal_release"
        for record in pp2["records"]
    )
    assert pp4["placements"] == [0, 0]
    assert pp2["placements"] == [0, 0]
    for run in (pp4, pp2):
        assert _admissions_while_the_engine_blocks(run) == []
        _assert_final_counts_match_lanes(run)
        _assert_every_held_key_is_reported(run)
        _assert_run_conserves_work(run)


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

REQUEST_SHAPES = 4


def _model(*, is_moe: bool, num_layers: int):
    from frontier.config import BaseModelConfig
    from frontier.types import ActivationType, NormType

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
    model._model_name = f"w4_runtime_{'moe' if is_moe else 'dense'}_{num_layers}l"
    return model


def _config(
    root: Path,
    patch,
    *,
    is_moe: bool,
    attn_dp: int,
    moe_ep: int,
    policy,
    trace: str | None = None,
    num_pipeline_stages: int = 1,
    num_layers: int = 4,
    analytical_backend: bool = False,
    dummy_execution_time_ms: float | None = None,
    num_blocks: int = 128,
):
    from frontier.cc_backend.cc_backend_config import AnalyticalCCBackendConfig
    from frontier.config import (
        BaseModelConfig,
        ClusterConfig,
        FixedRequestLengthGeneratorConfig,
        MetricsConfig,
        PoissonRequestIntervalGeneratorConfig,
        RandomForrestExecutionTimePredictorConfig,
        ReplicaConfig,
        SimulationConfig,
        SyntheticRequestGeneratorConfig,
        TraceRequestGeneratorConfig,
        VllmV1SchedulerConfig,
    )

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
    replica = ReplicaConfig(
        model_name=model._model_name,
        device="a100",
        network_device="a100_pairwise_nvlink",
        num_pipeline_stages=num_pipeline_stages,
        attn_tensor_parallel_size=1,
        attn_dp=attn_dp,
        memory_margin_fraction=0.1,
        **moe_fields,
    )
    # Placement here is decided by the balancer, not by latency realism, so the
    # predictor only has to be deterministic and identical across the policies
    # being compared.
    predictor = RandomForrestExecutionTimePredictorConfig(
        enable_dummy_mode=True,
        **(
            {}
            if dummy_execution_time_ms is None
            else dict(dummy_execution_time_ms=dummy_execution_time_ms)
        ),
    )
    backend = (
        dict(cc_backend_config=AnalyticalCCBackendConfig())
        if analytical_backend
        else {}
    )
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=num_blocks,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=16,
            enable_chunked_prefill=True,
        ),
        cluster_scheduler_config=policy(),
        execution_time_predictor_config=predictor,
        **backend,
    )
    generator = (
        TraceRequestGeneratorConfig(trace_file=trace)
        if trace is not None
        else SyntheticRequestGeneratorConfig(
            num_requests=REQUEST_SHAPES,
            length_generator_config=FixedRequestLengthGeneratorConfig(
                prefill_tokens=16, decode_tokens=3
            ),
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(qps=1e6),
        )
    )
    return SimulationConfig(
        simulation_mode="online" if trace is not None else "offline",
        sys_arch="co-location",
        enable_parallel_clusters=False,
        decode_cuda_graph_mode="none",
        cluster_config=cluster,
        metrics_config=MetricsConfig(
            output_dir=str(root / "metrics"),
            cache_dir=str(root / "cache"),
            run_id="dp_placement",
            write_metrics=False,
            store_request_metrics=False,
            store_batch_metrics=False,
            store_operation_metrics=False,
            store_utilization_metrics=False,
            store_plots=False,
            enable_chrome_trace=False,
            write_json_trace=False,
        ),
        request_generator_config=generator,
    )


def run_case(
    root: Path,
    *,
    policy_name: str,
    completion_reporting_control: bool = False,
    build_config=None,
    observe_placeholder_withdrawal: bool = False,
    **shape,
):
    """Run one configuration and return what only the event loop can show.

    `completion_reporting_control` swaps the policy's two seams for the
    reporting it had before schedule-time reports: nothing at admission, and
    each completion keyed by `ForwardSyncState.get_step_id`. It exists only
    here, as the control the discriminating case is measured against.

    `build_config(policy)` replaces this module's small test model and
    scheduler with a caller's configuration; the calibration case under
    `tests/comparison/dp_placement_pp/` uses it to observe the same seams.
    `shape` then only has to name `attn_dp`.
    """

    from frontier.config import (
        RoundRobinClusterSchedulerConfig,
        VllmLoadBalancingClusterSchedulerConfig,
    )
    from frontier.events.cluster_schedule_event import ClusterScheduleEvent
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        BaseClusterScheduler,
    )
    from frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler import (  # noqa: E501
        VllmLoadBalancingClusterScheduler,
    )
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (  # noqa: E501
        VLLMv1EngineReplicaScheduler,
    )
    from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import (
        ReplicaStageScheduler,
    )
    from frontier.scheduler.utils import sync_entry
    from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
    from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer
    from frontier.simulator import Simulator
    from frontier.types import ClusterType

    policy = {
        "vllm_load_balancing": VllmLoadBalancingClusterSchedulerConfig,
        "round_robin": RoundRobinClusterSchedulerConfig,
    }[policy_name]

    cluster_schedule_times: list[float] = []
    routing_times: list[float] = []
    placements: list[int] = []
    placement_request_ids: list[int] = []
    releases: dict[int, dict] = {}
    # Seam calls, their reports, and stage-0 forward starts, in event order.
    records: list[dict] = []
    reports: list[list] = []
    selections: list[dict] = []
    withdrawn_placeholder_lanes: list[int] = []
    stale_drops: list[dict] = []

    original_cluster_schedule = ClusterScheduleEvent.handle_event
    original_report = VllmDPLoadBalancer.report
    original_select = VllmDPLoadBalancer.select
    original_stage_pop = ReplicaStageScheduler.pop_batch_if_not_busy
    original_consume = ReplicaStageScheduler.consume_last_stale_drops

    def observed_cluster_schedule(self, scheduler, metrics_store):
        cluster_schedule_times.append(float(self.time))
        return original_cluster_schedule(self, scheduler, metrics_store)

    def observing_schedule_at(original):
        def observed(self, time):
            routing_times.append(float(time))
            mapping = original(self, time)
            placements.extend(lane for _, lane, _ in mapping)
            placement_request_ids.extend(request.id for _, _, request in mapping)
            return mapping

        return observed

    def observing_release(original):
        """Bracket the lane's own release so pre- and post-step load differ."""

        def observed(self, batch):
            before = list(self.get_request_load())
            result = original(self, batch)
            releases[batch.id] = {
                "before": before,
                "after": list(self.get_request_load()),
            }
            return result

        return observed

    def observed_report(self, time, engine, step, load):
        reports.append([step, list(load)])
        return original_report(self, time, engine, step, load)

    def observed_select(self, time):
        # Applying the passed deadlines first changes nothing: `select` does
        # the same before choosing. It exposes the snapshot the choice reads.
        self._advance(time)
        snapshot = [list(load) for load in self.frontend_counts]
        engine = original_select(self, time)
        selections.append({"time": float(time), "snapshot": snapshot, "engine": engine})
        return engine

    def observed_stage_pop(self):
        batch = original_stage_pop(self)
        if batch is not None and self._stage_id == 0:
            records.append(
                {
                    "kind": "stage0",
                    "lane": self._replica_local_id,
                    "batch": batch.id,
                    # A MoE lane batch carries the shared forward group bound
                    # at this start; a dense stage binds none.
                    "group": batch._forward_cohort_provisional_id
                    if self._is_moe
                    else None,
                }
            )
        return batch

    def observed_consume(self):
        dropped = original_consume(self)
        if dropped:
            stale_drops.append(
                {
                    "lane": self._replica_local_id,
                    "stage": self._stage_id,
                    "batches": [batch.id for batch in dropped],
                }
            )
        return dropped

    def observing_seam(kind, original):
        def observed(self, time, replica_id, replica_local_id, batch):
            before = len(reports)
            result = original(self, time, replica_id, replica_local_id, batch)
            made = reports[before:]
            assert len(made) <= 1, made
            lane = self.get_replica_scheduler(replica_id, replica_local_id)
            release = None if batch is None else releases.get(batch.id)
            if kind == "scheduled":
                source = None
            elif batch is None:
                source = "terminal_release"
            elif release is not None:
                source = "completion"
            else:
                source = "stale_drop"
            records.append(
                {
                    "kind": kind,
                    "time": float(time),
                    "lane": replica_local_id,
                    "batch": None if batch is None else batch.id,
                    "running_after": lane.num_running_batches,
                    "report": made[0] if made else None,
                    "released": release is not None,
                    "pre_step": release["before"] if release else None,
                    "post_step": release["after"] if release else None,
                    "source": source,
                    "held_key": self._held_key[replica_local_id],
                    "lane_load": list(lane.get_request_load()),
                }
            )
            return result

        return observed

    def completion_reporting(self, time, replica_id, replica_local_id, batch):
        if batch is None:
            return
        lane = self.get_replica_scheduler(replica_id, replica_local_id)
        self._load_balancer.report(
            time,
            replica_local_id,
            ForwardSyncState.get_step_id(batch),
            lane.get_request_load(),
        )

    with pytest.MonkeyPatch.context() as patch:
        config = (
            _config(root, patch, policy=policy, **shape)
            if build_config is None
            else build_config(policy)
        )
        patch.setattr(ClusterScheduleEvent, "handle_event", observed_cluster_schedule)
        patch.setattr(
            VLLMv1EngineReplicaScheduler,
            "on_batch_end",
            observing_release(vars(VLLMv1EngineReplicaScheduler)["on_batch_end"]),
        )
        patch.setattr(VllmDPLoadBalancer, "report", observed_report)
        patch.setattr(VllmDPLoadBalancer, "select", observed_select)
        patch.setattr(ReplicaStageScheduler, "pop_batch_if_not_busy", observed_stage_pop)
        patch.setattr(ReplicaStageScheduler, "consume_last_stale_drops", observed_consume)
        if observe_placeholder_withdrawal:
            original_withdraw = sync_entry._withdraw_idle_batches_of_joined_lanes

            def observed_withdraw(scheduler, sync_room, replica_id, stage_id):
                placed = {
                    lane
                    for lane, batch in sync_room["batches"].items()
                    if batch.is_idle
                }
                original_withdraw(scheduler, sync_room, replica_id, stage_id)
                withdrawn_placeholder_lanes.extend(
                    sorted(placed - set(sync_room["batches"]))
                )

            patch.setattr(
                sync_entry, "_withdraw_idle_batches_of_joined_lanes", observed_withdraw
            )
        # Both the inert base seam and the policy's override have to be
        # wrapped for routing: patching only the base would silently observe
        # nothing on the very policy under test.
        for owner in (BaseClusterScheduler, VllmLoadBalancingClusterScheduler):
            if "schedule_at" in vars(owner):
                patch.setattr(
                    owner,
                    "schedule_at",
                    observing_schedule_at(vars(owner)["schedule_at"]),
                )
        seams = {
            "on_replica_batch_scheduled": vars(VllmLoadBalancingClusterScheduler)[
                "on_replica_batch_scheduled"
            ],
            "on_replica_batch_end": vars(VllmLoadBalancingClusterScheduler)[
                "on_replica_batch_end"
            ],
        }
        if completion_reporting_control:
            seams = {
                "on_replica_batch_scheduled": vars(BaseClusterScheduler)[
                    "on_replica_batch_scheduled"
                ],
                "on_replica_batch_end": completion_reporting,
            }
        for name, kind in (
            ("on_replica_batch_scheduled", "scheduled"),
            ("on_replica_batch_end", "end"),
        ):
            patch.setattr(
                VllmLoadBalancingClusterScheduler,
                name,
                observing_seam(kind, seams[name]),
            )
        simulator = Simulator(config)
        simulator.run()
        requests = list(simulator._all_requests)
        cluster_scheduler = simulator._global_scheduler.get_cluster_scheduler(
            ClusterType.MONOLITHIC
        )
        lanes = [
            cluster_scheduler.get_replica_scheduler(replica_id, lane_id)
            for replica_id in cluster_scheduler._cluster.replicas
            for lane_id in range(shape["attn_dp"])
        ]
        contexts = list(cluster_scheduler._stage_execution_contexts.values())

    return {
        "num_lanes": shape["attn_dp"],
        "num_requests": len(requests),
        "completed_requests": sum(1 for request in requests if request.completed),
        "tokens_conserved": all(
            request.num_processed_tokens
            == request.num_prefill_tokens + request.num_decode_tokens
            for request in requests
        ),
        "lanes_released": all(
            lane.num_running_batches == 0 and not lane._running_requests
            for lane in lanes
        ),
        "stage_contexts_released": all(
            context.is_idle and context.queued_tickets == () for context in contexts
        ),
        "makespan": max((request.completed_at for request in requests), default=0.0),
        "cluster_schedule_times": cluster_schedule_times,
        "routing_times": routing_times,
        "first_four_lanes": placements[:4],
        "placements": placements,
        "placement_request_ids": placement_request_ids,
        "records": records,
        "selections": selections,
        "withdrawn_placeholder_lanes": withdrawn_placeholder_lanes,
        "stale_drops": stale_drops,
        "num_preemptions": sum(
            request.get_total_preemption_count() for request in requests
        ),
        "final_engine_counts": (
            None
            if policy_name == "round_robin"
            else [list(load) for load in cluster_scheduler._load_balancer.engine_counts]
        ),
        "final_lane_loads": [list(lane.get_request_load()) for lane in lanes],
    }


# One long request holds lane 0 while the rest are short, and the arrivals are
# spread far enough apart that published snapshots land between them. Load-blind
# round-robin cannot react to that asymmetry; a load-balancing policy can.
ASYMMETRIC_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,16,40
0.0,16,1
0.4,16,1
0.6,16,1
0.8,16,1
1.0,16,1
"""

# Plan §18.6. A burst after the first collection publish, so the frontend
# routes it from local reservations: lanes 0, 1, 0, 1, 0. Lane 0's three short
# prompts fit one batch; lane 1's long prompt takes the whole token budget, so
# its short request waits. A probe then arrives after the publication of those
# admissions and before any batch completes.
DISCRIMINATING_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
1.0,4,8
1.0,32,8
1.0,4,8
1.0,4,8
1.0,4,8
1.1,4,1
"""

# Issue W9-04. Four MoE lanes; requests 0 and 1 start lanes 0 and 1 two
# milliseconds apart, and request 2 arrives while lane 0 already waits in the
# first MoE room of that forward and lane 1 has not reached it yet.
LATE_JOIN_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,8,2
0.002,24,2
0.008,48,2
"""

# Short prompts arriving half a millisecond apart, so a lane often schedules
# while its peer's stage-0 forward is open and joins it.
STAGGER_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,16,6
0.0005,16,6
0.001,16,6
0.0015,16,6
0.002,16,6
0.0025,16,6
0.003,16,6
0.0035,16,6
"""

# SyntheticRequestGeneratorConfig(num_requests=16, seed=5): uniform lengths
# 8 to 96, prefill:decode 4.0, Poisson qps 200, all generators seeded with 5.
KV_PRESSURE_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.004876,58,15
0.012805,71,19
0.019538,71,18
0.019685,38,10
0.034040,51,14
0.045598,13,4
0.048764,23,6
0.052688,46,12
0.052754,21,6
0.054393,70,18
0.061649,17,5
0.069625,15,5
0.074430,15,4
0.074439,67,17
0.075614,20,6
0.090614,67,17
"""

RELEASE_TRACE = """arrived_at,num_prefill_tokens,num_decode_tokens
0.0,16,2
2.0,16,2
"""

TRACES = {
    "asymmetric": ASYMMETRIC_TRACE,
    "discriminating": DISCRIMINATING_TRACE,
    "late_join": LATE_JOIN_TRACE,
    "stagger": STAGGER_TRACE,
    "kv_pressure": KV_PRESSURE_TRACE,
    "release": RELEASE_TRACE,
}

CASES = {
    "moe_dp2": dict(is_moe=True, attn_dp=2, moe_ep=2),
    "dense_dp1": dict(is_moe=False, attn_dp=1, moe_ep=1),
    "moe_dp2_online": dict(is_moe=True, attn_dp=2, moe_ep=2, trace="asymmetric"),
    "dense_dp1_pp2": dict(
        is_moe=False, attn_dp=1, moe_ep=1, num_pipeline_stages=2,
        analytical_backend=True,
    ),
    "moe_dp2_pp2": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=2,
        analytical_backend=True,
    ),
    "moe_dp2_pp2_online": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=2,
        analytical_backend=True, trace="asymmetric",
    ),
    "moe_dp1_pp3": dict(
        is_moe=True, attn_dp=1, moe_ep=1, num_pipeline_stages=3, num_layers=6,
        analytical_backend=True,
    ),
    "moe_dp2_pp3": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=3, num_layers=6,
        analytical_backend=True,
    ),
    "moe_dp2_pp2_stagger": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=2,
        analytical_backend=True, trace="stagger",
    ),
    "moe_dp2_pp3_stagger": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=3, num_layers=6,
        analytical_backend=True, trace="stagger",
    ),
    "moe_dp4_late_join": dict(
        is_moe=True, attn_dp=4, moe_ep=4, trace="late_join",
        dummy_execution_time_ms=1.0, observe_placeholder_withdrawal=True,
    ),
    "moe_dp2_pp4": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=4, num_layers=4,
        analytical_backend=True,
    ),
    "dense_dp1_pp4_kv_pressure": dict(
        is_moe=False, attn_dp=1, moe_ep=1, num_pipeline_stages=4, num_layers=4,
        analytical_backend=True, num_blocks=8, trace="kv_pressure",
    ),
    "moe_dp2_pp4_kv_pressure": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=4, num_layers=4,
        analytical_backend=True, num_blocks=8, trace="kv_pressure",
    ),
    "moe_dp2_pp2_release": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=2, num_layers=4,
        analytical_backend=True, trace="release",
    ),
    "moe_dp2_pp4_release": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=4, num_layers=4,
        analytical_backend=True, trace="release",
    ),
    # Stages long enough that no batch completes before the probe (plan D-e).
    "moe_dp2_pp2_discriminating": dict(
        is_moe=True, attn_dp=2, moe_ep=2, num_pipeline_stages=2,
        analytical_backend=True, trace="discriminating",
        dummy_execution_time_ms=10.0,
    ),
}

# The policy run's comparison run: a load-blind baseline, or for the
# discriminating case the completion-reporting control.
COMPARISONS = {"moe_dp2_pp2_discriminating": "completion_reporting_control"}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    case = sys.argv[2]
    shape = CASES[case]
    comparison = COMPARISONS.get(case, "round_robin")
    evidence = {}
    for run_name in ("vllm_load_balancing", comparison):
        case_root = root / case / run_name
        case_root.mkdir(parents=True, exist_ok=True)
        arguments = dict(shape)
        if "trace" in arguments:
            trace_path = case_root / f"{arguments['trace']}_arrivals.csv"
            trace_path.write_text(TRACES[arguments["trace"]])
            arguments["trace"] = str(trace_path)
        evidence[run_name] = run_case(
            case_root,
            policy_name="round_robin"
            if run_name == "round_robin"
            else "vllm_load_balancing",
            completion_reporting_control=run_name == "completion_reporting_control",
            **arguments,
        )
    (root / f"{case}_evidence.json").write_text(json.dumps(evidence, indent=1))
    print(json.dumps({k: v["completed_requests"] for k, v in evidence.items()}))
