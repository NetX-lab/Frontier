"""Contract tests for the real-data MoE EP matrix harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.moe_ep_non_dummy_matrix import (
    build_matrix,
    build_shell_command,
    check_case_log,
    validate_profile_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_matrix_has_required_cross_architecture_coverage() -> None:
    cases = build_matrix(REPO_ROOT)

    assert len(cases) >= 100
    assert {case.architecture for case in cases} == {
        "co-location",
        "pd-disaggregation",
        "pd-af-disaggregation",
    }
    assert {case.model_kind for case in cases} == {"dense", "moe", "mixed"}
    assert {case.routing_distribution for case in cases if case.model_kind != "dense"} >= {
        "balanced",
        "random",
        "skewed",
        "zipf",
    }
    assert {case.ep_size for case in cases if case.model_kind != "dense"} >= {1, 2, 4}
    assert {case.workload_kind for case in cases} >= {
        "prefill-heavy",
        "decode-heavy",
        "mixed",
        "zero-routed",
    }


def test_matrix_enforces_dense_topology_and_card_limit() -> None:
    cases = build_matrix(REPO_ROOT)

    assert all(case.total_cards <= 32 for case in cases)
    assert all(case.total_cards > 0 for case in cases)
    assert all(case.ep_size == 1 for case in cases if case.model_kind == "dense")
    assert all(case.moe_tensor_parallel_size == 1 for case in cases)


def test_non_dummy_command_has_no_dummy_switch() -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )

    command, env = build_shell_command(case, REPO_ROOT, Path("/data/ycfeng/tmp/matrix"))

    assert "--random_forrest_execution_time_predictor_config_enable_dummy_mode" not in command
    assert "--replica_config_device h800" in command
    assert env["ENABLE_DUMMY_MODE"] == "false"
    assert env["DECODE_CUDA_GRAPH_MODE"] == "none"


def test_profile_validation_is_fail_fast(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.model_kind == "moe"
    )
    with pytest.raises(FileNotFoundError, match="moe.csv"):
        validate_profile_inputs(case, tmp_path)


def test_log_checker_requires_layer_trace_and_finite_metrics(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "pd-af-disaggregation" and case.model_kind == "moe"
    )
    log_path = tmp_path / "case.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "ttft_statistics": {"mean": 1.25},
                "request_e2e_time_statistics": {"mean": 2.5},
            }
        ),
        encoding="utf-8",
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][DECODE_FFN][MOE][moe_shuffling] batch_id=1, layer_id=0, predicted_time_ms=0.1",
                "[OP-TRACE][DECODE_FFN][MOE][moe_grouped_gemm] batch_id=1, layer_id=0, predicted_time_ms=0.2",
                "[OP-TRACE][DECODE_FFN][MOE][TOTAL] batch_id=1, layer_id=0, total_moe_time_ms=1.0",
                "[DECODE_FFN] per_expert_tokens extracted: {0: 1, 1: 0}",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["layer_ids"] == [0]
    assert result["numeric_metric_count"] == 2


def test_log_checker_rejects_traceback(tmp_path: Path) -> None:
    case = next(iter(build_matrix(REPO_ROOT)))
    log_path = tmp_path / "case.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text("{}", encoding="utf-8")
    log_path.write_text(
        "Dummy Mode: false\nTraceback (most recent call last):\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "Traceback" in result["errors"]


def test_dense_checker_does_not_require_moe_layer_granularity(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "dense"
    )
    log_path = tmp_path / "dense.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "Dummy Mode: false\nSimulation completed successfully.\n"
        "[OP-TRACE][MONOLITHIC][ATTENTION] batch_id=0, layer_id=0, num_tokens=1\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "PASS"
