"""Real-runtime wiring for the opt-in vLLM-style DP placement policy.

The state machine itself is covered by `tests/unit/test_vllm_dp_load_balancer.py`
against reference-derived expectations. What can only be shown by running the
simulator is the wiring: that `ClusterScheduleEvent` is what supplies the
routing time, that `GlobalBatchEndEvent` reports the lane's **post**-step load,
that the report key is ordered the way the policy's guard assumes, and that the
policy introduces no event that keeps a drained run alive.

The child process runs two configurations that differ only in the cluster
scheduler policy, so the comparison isolates the policy. Execution time comes
from the dummy predictor: placement here is decided by the balancer, not by
latency realism, and both policies see the same durations.
"""

from __future__ import annotations

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


def test_dp_placement_runs_and_reports_post_step_load(tmp_path):
    evidence = _run_child(tmp_path, "moe_dp2")
    policy = evidence["vllm_load_balancing"]
    baseline = evidence["round_robin"]

    # The run has to reach the shape under test.
    assert policy["num_lanes"] == 2
    assert policy["completed_requests"] == policy["num_requests"]

    # 1. The routing time comes from ClusterScheduleEvent, not retained state.
    #    Every recorded routing time is one of the cluster schedule times, and
    #    the policy's own `schedule()` would have raised had it been used.
    assert policy["routing_times"]
    assert policy["routing_times"] == policy["cluster_schedule_times"]

    # 2. Placement, derived independently: four requests arrive together, no
    #    report can precede the first routing, so the estimate is empty and only
    #    the local reservation moves the choice.
    assert policy["first_four_lanes"] == [0, 1, 0, 1]

    # 3. The report key is ordered as the capability guard assumes: never
    #    decreasing, and equal only for peer lanes of one shared forward.
    keys = policy["report_keys"]
    assert keys == sorted(keys)
    for key, lanes in policy["lanes_by_key"].items():
        assert len(lanes) == len(set(lanes)), (key, lanes)

    # 4. The reported load is the post-step state. The lane's own release is
    #    bracketed, so pre- and post-step load are distinct values for at least
    #    some batches; every report matches the post-step one and the reports are
    #    not merely the pre-step values.
    assert policy["reports_after_the_lane_released_the_batch"] == policy["num_reports"]
    assert policy["reports_where_the_release_changed_the_load"] > 0
    assert policy["reports_matching_post_step"] == policy["num_reports"]
    assert policy["reports_matching_pre_step"] < policy["num_reports"]

    # 5. No event type is introduced, and the run drains rather than being kept
    #    alive by a heartbeat.
    assert set(policy["event_types"]) == set(baseline["event_types"])
    assert policy["makespan"] > 0
    assert baseline["completed_requests"] == baseline["num_requests"]


def test_placement_follows_published_load_where_round_robin_cannot(tmp_path):
    """Spread the arrivals so snapshots land between them, and skew the load.

    This is the discriminating case: the two policies see identical arrivals,
    identical durations and identical lane capacity, so any difference in
    placement comes from reading the published load.
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
    keys = policy["report_keys"]
    assert keys == sorted(keys)
    assert len(keys) > 10
    for key, lanes in policy["lanes_by_key"].items():
        assert len(lanes) == len(set(lanes)), (key, lanes)

    # 4. Still the post-step load, and still no new event type.
    assert policy["reports_after_the_lane_released_the_batch"] == policy["num_reports"]
    assert policy["reports_where_the_release_changed_the_load"] > 0
    assert policy["reports_matching_post_step"] == policy["num_reports"]
    assert policy["reports_matching_pre_step"] < policy["num_reports"]
    assert set(policy["event_types"]) == set(baseline["event_types"])


def test_a_single_lane_shape_routes_everything_to_lane_zero(tmp_path):
    evidence = _run_child(tmp_path, "dense_dp1")
    policy = evidence["vllm_load_balancing"]

    assert policy["num_lanes"] == 1
    assert policy["completed_requests"] == policy["num_requests"]
    assert set(policy["first_four_lanes"]) == {0}


# ---------------------------------------------------------------------------
# Child process
# ---------------------------------------------------------------------------

REQUEST_SHAPES = 4


def _model(*, is_moe: bool):
    from frontier.config import BaseModelConfig
    from frontier.types import ActivationType, NormType

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
    model._model_name = f"w4_runtime_{'moe' if is_moe else 'dense'}"
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
):
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
    replica = ReplicaConfig(
        model_name=model._model_name,
        device="a100",
        network_device="a100_pairwise_nvlink",
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        attn_dp=attn_dp,
        memory_margin_fraction=0.1,
        **moe_fields,
    )
    cluster = ClusterConfig(
        replica_config=replica,
        replica_scheduler_config=VllmV1SchedulerConfig(
            num_blocks=128,
            block_size=16,
            batch_size_cap=4,
            max_tokens_in_batch=16,
            enable_chunked_prefill=True,
        ),
        cluster_scheduler_config=policy(),
        # Placement here is decided by the balancer, not by latency realism, so
        # the predictor only has to be deterministic and identical across the
        # two policies being compared.
        execution_time_predictor_config=RandomForrestExecutionTimePredictorConfig(
            enable_dummy_mode=True
        ),
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
    is_moe: bool,
    attn_dp: int,
    moe_ep: int,
    policy_name: str,
    trace: str | None = None,
):
    """Run one configuration and return what only the event loop can show."""

    from frontier.config import (
        RoundRobinClusterSchedulerConfig,
        VllmLoadBalancingClusterSchedulerConfig,
    )
    from frontier.events.cluster_schedule_event import ClusterScheduleEvent
    from frontier.events.global_batch_end_event import GlobalBatchEndEvent
    from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
        BaseClusterScheduler,
    )
    from frontier.scheduler.cluster_scheduler.vllm_load_balancing_cluster_scheduler import (  # noqa: E501
        VllmLoadBalancingClusterScheduler,
    )
    from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (  # noqa: E501
        VLLMv1EngineReplicaScheduler,
    )
    from frontier.scheduler.utils.forward_sync_state import ForwardSyncState
    from frontier.simulator import Simulator

    policy = {
        "vllm_load_balancing": VllmLoadBalancingClusterSchedulerConfig,
        "round_robin": RoundRobinClusterSchedulerConfig,
    }[policy_name]

    cluster_schedule_times: list[float] = []
    routing_times: list[float] = []
    placements: list[int] = []
    reports: list[dict] = []
    releases: dict[int, dict] = {}
    event_types: set[str] = set()

    original_cluster_schedule = ClusterScheduleEvent.handle_event
    original_batch_end = GlobalBatchEndEvent.handle_event

    def observed_cluster_schedule(self, scheduler, metrics_store):
        cluster_schedule_times.append(float(self.time))
        events = original_cluster_schedule(self, scheduler, metrics_store)
        event_types.update(type(event).__name__ for event in events or [])
        return events

    def observing_schedule_at(original):
        def observed(self, time):
            routing_times.append(float(time))
            mapping = original(self, time)
            placements.extend(lane for _, lane, _ in mapping)
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

    def observing_hook(original):
        def observed(self, time, replica_id, replica_local_id, batch):
            result = original(self, time, replica_id, replica_local_id, batch)
            balancer = getattr(self, "_load_balancer", None)
            if balancer is not None and type(replica_local_id) is int:
                # `None` means the lane had not yet released this batch when the
                # hook ran, which is itself the ordering evidence.
                release = releases.get(batch.id)
                reports.append(
                    {
                        "lane": replica_local_id,
                        "key": ForwardSyncState.get_step_id(batch),
                        "released": release is not None,
                        "pre_step": release["before"] if release else None,
                        "post_step": release["after"] if release else None,
                        "reported": list(balancer.engine_counts[replica_local_id]),
                    }
                )
            return result

        return observed

    def observed_batch_end(self, scheduler, metrics_store):
        events = original_batch_end(self, scheduler, metrics_store)
        event_types.update(type(event).__name__ for event in events or [])
        return events

    with pytest.MonkeyPatch.context() as patch:
        config = _config(
            root,
            patch,
            is_moe=is_moe,
            attn_dp=attn_dp,
            moe_ep=moe_ep,
            policy=policy,
            trace=trace,
        )
        patch.setattr(ClusterScheduleEvent, "handle_event", observed_cluster_schedule)
        patch.setattr(GlobalBatchEndEvent, "handle_event", observed_batch_end)
        patch.setattr(
            VLLMv1EngineReplicaScheduler,
            "on_batch_end",
            observing_release(vars(VLLMv1EngineReplicaScheduler)["on_batch_end"]),
        )
        # Both the inert base seam and the policy's override have to be
        # wrapped: patching only the base would silently observe nothing on the
        # very policy under test.
        for owner in (BaseClusterScheduler, VllmLoadBalancingClusterScheduler):
            if "schedule_at" in vars(owner):
                patch.setattr(
                    owner,
                    "schedule_at",
                    observing_schedule_at(vars(owner)["schedule_at"]),
                )
            if "on_replica_batch_end" in vars(owner):
                patch.setattr(
                    owner,
                    "on_replica_batch_end",
                    observing_hook(vars(owner)["on_replica_batch_end"]),
                )
        simulator = Simulator(config)
        simulator.run()
        requests = list(simulator._all_requests)

    lanes_by_key: dict[str, list[int]] = {}
    for report in reports:
        lanes_by_key.setdefault(str(report["key"]), []).append(report["lane"])

    return {
        "num_lanes": attn_dp,
        "num_requests": len(requests),
        "completed_requests": sum(1 for request in requests if request.completed),
        "makespan": max((request.completed_at for request in requests), default=0.0),
        "cluster_schedule_times": cluster_schedule_times,
        "routing_times": routing_times,
        "first_four_lanes": placements[:4],
        "placements": placements,
        "report_keys": [report["key"] for report in reports],
        "lanes_by_key": lanes_by_key,
        "num_reports": len(reports),
        "reports_after_the_lane_released_the_batch": sum(
            1 for report in reports if report["released"]
        ),
        "reports_matching_post_step": sum(
            1 for report in reports if report["post_step"] == report["reported"]
        ),
        "reports_matching_pre_step": sum(
            1 for report in reports if report["pre_step"] == report["reported"]
        ),
        "reports_where_the_release_changed_the_load": sum(
            1 for report in reports if report["pre_step"] != report["post_step"]
        ),
        "event_types": sorted(event_types),
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


def _trace_file(root: Path) -> str:
    path = root / "asymmetric_arrivals.csv"
    path.write_text(ASYMMETRIC_TRACE)
    return str(path)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    case = sys.argv[2]
    shape = {
        "moe_dp2": dict(is_moe=True, attn_dp=2, moe_ep=2),
        "dense_dp1": dict(is_moe=False, attn_dp=1, moe_ep=1),
        "moe_dp2_online": dict(is_moe=True, attn_dp=2, moe_ep=2, trace=True),
    }[case]
    evidence = {}
    for policy_name in ("vllm_load_balancing", "round_robin"):
        case_root = root / case / policy_name
        case_root.mkdir(parents=True, exist_ok=True)
        arguments = dict(shape)
        if arguments.pop("trace", False):
            arguments["trace"] = _trace_file(case_root)
        evidence[policy_name] = run_case(
            case_root, policy_name=policy_name, **arguments
        )
    (root / f"{case}_evidence.json").write_text(json.dumps(evidence, indent=1))
    print(json.dumps({k: v["completed_requests"] for k, v in evidence.items()}))
