from __future__ import annotations

import json
from pathlib import Path

import pytest

from frontier.config import global_vars
from frontier.simulator import Simulator
from tests.performance.sim_walltime_scaling.pd_moe_lifecycle_reproducer import (
    _capture_generated_requests,
    _capture_claimed_event_priorities,
    collect_lifecycle_state,
    summarize_claim_order,
    validate_lifecycle_state,
)
from tests.performance.sim_walltime_scaling.run_case import (
    CaseSpec,
    ParallelShape,
    _default_config_factory,
    build_frontier_argv,
    validate_measurement_config,
)


@pytest.fixture(autouse=True)
def _reset_frontier_global_state():
    global_vars.reset_global_vars()
    yield
    global_vars.reset_global_vars()


def _moe_case(
    *,
    mode: str = "parallel",
    num_requests: int = 1,
) -> CaseSpec:
    return CaseSpec.for_scale(
        model="moe",
        total_gpus=32,
        mode=mode,
        attempt_index=0,
        shape=ParallelShape(
            attn_tp=4,
            attn_dp=2,
            moe_tp=1,
            moe_ep=8,
            pp=2,
        ),
        num_requests=num_requests,
        qps=8.0,
        prefill_tokens=16,
        decode_tokens=2,
    )


def _spec_decode_argv(trace_path: Path) -> list[str]:
    return [
        "--speculative_decoding_config_enabled",
        "--speculative_decoding_config_method",
        "qwen3_moe_mtp",
        "--speculative_decoding_config_num_speculative_tokens",
        "1",
        "--speculative_decoding_config_acceptance_trace_file",
        str(trace_path),
        "--speculative_decoding_config_mtp_n_predict",
        "1",
        "--speculative_decoding_config_mtp_num_layers",
        "1",
    ]


def _run_moe_case(
    case: CaseSpec,
    output_root: Path,
    *,
    extra_argv: list[str] | None = None,
) -> dict:
    argv = build_frontier_argv(case, output_root / "simulator-configs")
    if extra_argv:
        argv.extend(extra_argv)
    config = _default_config_factory(argv)
    validate_measurement_config(config)

    with _capture_generated_requests() as requests:
        simulator = Simulator(config)
    simulator.run()

    state = collect_lifecycle_state(
        simulator,
        requests,
        case=case,
        run_error=None,
    )
    state["validation_errors"] = validate_lifecycle_state(state)
    return state


def _timestamp_signature(state: dict) -> list[tuple[float, float, float, float]]:
    fields = (
        "arrived_at",
        "scheduled_at",
        "prefill_completed_at",
        "completed_at",
    )
    return [
        tuple(float(request[field]) for field in fields)
        for request in sorted(
            state["requests"],
            key=lambda request: float(request["arrived_at"]),
        )
    ]


def _normalized_claim_signature(
    claimed_priorities: list[tuple],
) -> list[tuple[float, int, int]]:
    first_event_id = int(claimed_priorities[0][1])
    return [
        (
            float(priority[0]),
            int(priority[1]) - first_event_id,
            int(priority[2]),
        )
        for priority in claimed_priorities
    ]


def test_parallel_distributed_moe_completes_after_initial_zero_commit(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "zero-commit-trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "committed_tokens_per_iteration": [0, 1, 1],
                "scheduled_draft_tokens_per_iteration": [1, 0, 0],
            }
        ),
        encoding="utf-8",
    )

    case = _moe_case()
    state = _run_moe_case(
        case,
        tmp_path,
        extra_argv=_spec_decode_argv(trace_path),
    )

    assert state["completed_requests"] == state["expected_requests"] == 1
    assert state["validation_errors"] == []

    request_state = state["requests"][0]
    assert request_state.get("spec_total_iterations") == 3
    assert request_state.get("spec_total_committed_tokens") == 2
    assert request_state["spec_last_committed_tokens"] == 1
    assert request_state["completed"] is True
    assert request_state["num_processed_tokens"] == request_state["total_tokens"] == 18
    assert request_state["completed_layer_count"] == 0
    assert request_state["active_memberships"] == []

    inter_cluster_stats = state["inter_cluster_stats"]
    assert inter_cluster_stats["events_sent"] == inter_cluster_stats["events_delivered"]
    assert inter_cluster_stats["events_sent"] > 0
    assert inter_cluster_stats["queue_size"] == 0
    assert inter_cluster_stats["total_buffered_events"] == 0


def test_parallel_distributed_moe_claims_follow_sequential_des_priority(
    tmp_path: Path,
) -> None:
    case = _moe_case(num_requests=8)
    with _capture_claimed_event_priorities() as claimed_priorities:
        state = _run_moe_case(case, tmp_path)
    state["claim_order"] = summarize_claim_order(claimed_priorities)
    state["validation_errors"] = validate_lifecycle_state(state)

    assert state["completed_requests"] == state["expected_requests"] == 8
    assert state["validation_errors"] == []
    assert state["claim_order"]["claim_count"] > 0
    assert state["claim_order"]["priority_inversion_count"] == 0
    assert state["claim_order"]["monotonic"] is True


def test_parallel_distributed_moe_64_request_lifecycle_is_clean(
    tmp_path: Path,
) -> None:
    case = _moe_case(num_requests=64)
    with _capture_claimed_event_priorities() as claimed_priorities:
        state = _run_moe_case(case, tmp_path)
    state["claim_order"] = summarize_claim_order(claimed_priorities)
    state["validation_errors"] = validate_lifecycle_state(state)

    assert state["completed_requests"] == state["expected_requests"] == 64
    assert len(state["requests"]) == 64
    assert state["validation_errors"] == []
    assert state["global_scheduler_is_empty"] is True
    assert state["claim_order"]["claim_count"] > 0
    assert state["claim_order"]["priority_inversion_count"] == 0
    assert state["claim_order"]["monotonic"] is True

    assert all(request["completed"] for request in state["requests"])
    assert {
        request["completed_layer_count"] for request in state["requests"]
    } == {0}
    assert all(
        request["active_memberships"] == [] for request in state["requests"]
    )

    parallel_runtime = state["event_queues"]["parallel_cluster_runtime"]
    assert set(parallel_runtime) == {"PREFILL", "DECODE"}
    for runtime_state in parallel_runtime.values():
        assert runtime_state["queue_size"] == 0
        assert runtime_state["is_processing_event"] is False
        assert runtime_state["is_running"] is False
        assert runtime_state["current_event_priority"] is None
        assert runtime_state["next_event_priority"] is None

    for cluster_state in state["cluster_schedulers"].values():
        assert cluster_state["is_empty"] is True
        for replica_state in cluster_state["replicas"].values():
            assert replica_state["active_batch_request_counts"] == {}
            assert replica_state["num_running_batches"] == 0
            assert replica_state["allocation_map"]["count"] == 0
            assert replica_state["request_queue"]["count"] == 0
            assert replica_state["waiting_requests"]["count"] == 0
            assert replica_state["running_requests"]["count"] == 0
            for stage_state in replica_state["stage_schedulers"].values():
                assert stage_state["is_busy"] is False
                assert stage_state["is_empty"] is True
                assert stage_state["batch_queue"]["count"] == 0

    inter_cluster_stats = state["inter_cluster_stats"]
    assert inter_cluster_stats["events_sent"] == 64
    assert inter_cluster_stats["events_delivered"] == 64
    assert inter_cluster_stats["queue_full_count"] == 0
    assert inter_cluster_stats["queue_size"] == 0
    assert inter_cluster_stats["total_buffered_events"] == 0
    assert set(inter_cluster_stats["buffer_sizes"].values()) == {0}

    global_vars.reset_global_vars()
    sequential_state = _run_moe_case(
        _moe_case(mode="sequential", num_requests=64),
        tmp_path / "sequential",
    )

    assert sequential_state["completed_requests"] == 64
    assert sequential_state["validation_errors"] == []
    assert _timestamp_signature(state) == _timestamp_signature(sequential_state)


def test_parallel_distributed_moe_matches_sequential_request_timestamps(
    tmp_path: Path,
) -> None:
    parallel_state = _run_moe_case(
        _moe_case(num_requests=8),
        tmp_path / "parallel",
    )
    global_vars.reset_global_vars()
    sequential_state = _run_moe_case(
        _moe_case(mode="sequential", num_requests=8),
        tmp_path / "sequential",
    )

    parallel_timestamps = _timestamp_signature(parallel_state)
    sequential_timestamps = _timestamp_signature(sequential_state)
    deltas = [
        abs(parallel_value - sequential_value)
        for parallel_request, sequential_request in zip(
            parallel_timestamps,
            sequential_timestamps,
        )
        for parallel_value, sequential_value in zip(
            parallel_request,
            sequential_request,
        )
    ]
    mismatches = [delta for delta in deltas if delta != 0.0]

    assert parallel_state["validation_errors"] == []
    assert sequential_state["validation_errors"] == []
    assert len(parallel_timestamps) == len(sequential_timestamps) == 8
    assert not mismatches, (
        f"mismatched_values={len(mismatches)}, "
        f"max_timestamp_delta={max(deltas, default=0.0)}"
    )


def test_parallel_distributed_moe_is_repeat_deterministic(
    tmp_path: Path,
) -> None:
    case = _moe_case(num_requests=8)
    with _capture_claimed_event_priorities() as first_claims:
        first_state = _run_moe_case(case, tmp_path / "first")
    global_vars.reset_global_vars()
    with _capture_claimed_event_priorities() as second_claims:
        second_state = _run_moe_case(case, tmp_path / "second")

    first_claim_order = summarize_claim_order(first_claims)
    second_claim_order = summarize_claim_order(second_claims)

    assert first_state["validation_errors"] == []
    assert second_state["validation_errors"] == []
    assert _timestamp_signature(first_state) == _timestamp_signature(second_state)
    assert first_claim_order["priority_inversion_count"] == 0
    assert second_claim_order["priority_inversion_count"] == 0
    assert _normalized_claim_signature(first_claims) == (
        _normalized_claim_signature(second_claims)
    )
