"""Contract tests for the real-data MoE EP matrix harness."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from tests.e2e.moe_ep_non_dummy_matrix import (
    _find_metrics_dir,
    _merge_result_rows,
    _parse_ep_conservation_records,
    _parse_ep_barrier_records,
    _parse_ep_workload_records,
    _validate_result_ledger_provenance,
    build_matrix,
    build_shell_command,
    check_case_log,
    validate_case_parallel_semantics,
    validate_profile_inputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_matrix_has_required_cross_architecture_coverage() -> None:
    cases = build_matrix(REPO_ROOT)

    assert len(cases) >= 100
    assert Counter(case.architecture for case in cases) == Counter(
        {
            "co-location": 50,
            "pd-disaggregation": 50,
            "pd-af-disaggregation": 10,
        }
    )
    assert {case.model_kind for case in cases} == {"dense", "moe", "mixed"}
    assert {case.routing_distribution for case in cases if case.model_kind != "dense"} >= {
        "balanced",
        "random",
        "skewed",
        "zipf",
    }
    assert all(
        case.model_name == "step-moe-noquant-small"
        and case.routing_distribution == "random"
        for case in cases
        if case.model_kind == "mixed"
    )
    assert all(
        case.model_name == "qwen3-a3b-30b-moe"
        and case.device == "a800"
        for case in cases
        if case.model_kind == "moe" and case.routing_distribution != "random"
    )
    assert {case.ep_size for case in cases if case.model_kind != "dense"} >= {1, 2, 4}
    assert {case.workload_kind for case in cases} >= {
        "prefill-heavy",
        "decode-heavy",
        "mixed",
        "zero-routed",
    }


def test_matrix_uses_frontier_vllm_parallel_semantics() -> None:
    for case in build_matrix(REPO_ROOT):
        validate_case_parallel_semantics(case)


def test_matrix_enforces_dense_topology_and_card_limit() -> None:
    cases = build_matrix(REPO_ROOT)

    assert all(case.total_cards <= 32 for case in cases)
    assert all(case.total_cards > 0 for case in cases)
    assert all(case.prefill_tokens > 1 for case in cases)
    assert all(case.ep_size == 1 for case in cases if case.model_kind == "dense")
    assert all(
        case.moe_tensor_parallel_size == (4 if case.model_kind == "mixed" else 1)
        for case in cases
    )
    assert all(case.pipeline_stages == 1 for case in cases)


def test_mixed_matrix_shapes_stay_within_step_profile_tp_domain() -> None:
    mixed_cases = [case for case in build_matrix(REPO_ROOT) if case.model_kind == "mixed"]

    assert mixed_cases
    assert all(case.moe_tensor_parallel_size == 4 for case in mixed_cases)
    assert all(case.ep_size <= 2 for case in mixed_cases)
    assert all(case.attn_tensor_parallel_size <= 8 for case in mixed_cases)


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


def test_non_dummy_command_uses_output_scoped_predictor_cache(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )
    output_root = tmp_path / "matrix-output"

    command, _ = build_shell_command(case, REPO_ROOT, output_root)

    expected_cache = (output_root / "_predictor_cache").resolve()
    assert "--metrics_config_cache_dir" in command
    assert str(expected_cache) in command


def test_profile_validation_is_fail_fast(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.model_kind == "moe"
    )
    with pytest.raises(FileNotFoundError, match="moe.csv"):
        validate_profile_inputs(case, tmp_path)


def test_profile_validation_rejects_missing_architecture_metadata(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )
    model_dir = tmp_path / case.device / case.model_name
    model_dir.mkdir(parents=True)
    (model_dir / "attention.csv").write_text(
        "profiling_precision,model_arch,quant_signature,measurement_type\n"
        "BF16,generic,none,CUDA_EVENT\n",
        encoding="utf-8",
    )
    (model_dir / "linear_op.csv").write_text(
        "profiling_precision,model_arch,quant_signature,measurement_type\n"
        "BF16,generic,none,CUDA_EVENT\n",
        encoding="utf-8",
    )
    (model_dir / "moe.csv").write_text(
        "profiling_precision,model_arch,quant_signature,measurement_type,"
        "model_architecture_profile,routing_runtime_path\n"
        "BF16,generic,none,CUDA_EVENT,generic,standard_fused_topk\n"
        "BF16,generic,none,CUDA_EVENT,generic,uniform_topk\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_architecture_profile"):
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
                "[EP-WORKLOAD][DECODE_FFN] batch_id=1, layer_id=0, ep_id=0, moe_ep_size=1, per_expert_tokens={0: 1, 1: 0}, lane_compute_ms=0.2, lane_comm_ms=0.0",
                "[EP-BARRIER][DECODE_FFN] batch_id=1, layer_id=0, phase=combine, expected_ep_ids=[0], arrived_ep_ids=[0], max_lane_time_ms=0.2, barrier_time_ms=0.2, barrier_end_time_s=0.001",
                "[EP-CONSERVATION][DECODE_FFN] batch_id=1, layer_id=0, routing_token_count=1, router_topk=1, total_routed_assignments=1, per_ep_routed_tokens={0: 1}",
                "[DECODE_FFN] per_expert_tokens extracted: {0: 1, 1: 0}",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "PASS"
    assert result["layer_ids"] == [0]
    assert result["ep_workload_records"] == 1
    assert result["ep_barrier_records"] == 1
    assert result["numeric_metric_count"] == 2


def test_shared_moe_checker_requires_ep_workload_trace(tmp_path: Path) -> None:
    case = next(
        case
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location" and case.model_kind == "moe"
    )
    log_path = tmp_path / "shared_moe.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "Dummy Mode: false\nSimulation completed successfully.\n"
        "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] batch_id=1, layer_id=0, predicted_time_ms=0.2\n",
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir)

    assert result["status"] == "FAIL"
    assert "EP workload" in result["errors"]


def test_shared_moe_checker_requires_dispatch_and_combine_barriers(
    tmp_path: Path,
) -> None:
    case = replace(
        next(
            case
            for case in build_matrix(REPO_ROOT)
            if case.architecture == "co-location"
            and case.model_kind == "moe"
            and case.ep_size == 1
        ),
        num_layers=1,
        moe_layer_ids=(0,),
    )
    log_path = tmp_path / "shared_barrier.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] batch_id=1, layer_id=0, predicted_time_ms=0.1",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] batch_id=1, layer_id=0, predicted_time_ms=0.2",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=1, layer_id=0, ep_id=0, moe_ep_size=1, per_expert_tokens={0: 1}, lane_compute_ms=0.2, lane_comm_ms=0.0",
                "[EP-CONSERVATION][MONOLITHIC] batch_id=1, layer_id=0, routing_token_count=1, router_topk=1, total_routed_assignments=1, per_ep_routed_tokens={0: 1}",
                "[EP-BARRIER][MONOLITHIC] batch_id=1, layer_id=0, phase=combine, expected_ep_ids=[0], arrived_ep_ids=[0], max_lane_time_ms=0.2, barrier_time_ms=0.2, barrier_end_time_s=0.001",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "cluster=MONOLITHIC phase=dispatch layers=[0]" in result["errors"]


def test_ep_workload_parser_accepts_logger_prefix() -> None:
    records = _parse_ep_workload_records(
        "INFO 12:00:00 scheduler.py:1] "
        "[EP-WORKLOAD][DECODE] batch_id=7, layer_id=3, ep_id=1, "
        "moe_ep_size=2, per_expert_tokens={0: 0, 1: 4}, "
        "lane_compute_ms=1.25, lane_comm_ms=0.5"
    )

    assert records == [
        {
            "cluster": "DECODE",
            "batch_id": 7,
            "layer_id": 3,
            "ep_id": 1,
            "moe_ep_size": 2,
            "per_expert_tokens": {0: 0, 1: 4},
            "lane_compute_ms": 1.25,
            "lane_comm_ms": 0.5,
        }
    ]


def test_ep_barrier_parser_accepts_logger_prefix() -> None:
    records = _parse_ep_barrier_records(
        "INFO 12:00:00 scheduler.py:1] "
        "[EP-BARRIER][PREFILL] batch_id=7, layer_id=3, phase=combine, "
        "expected_ep_ids=[0, 1], arrived_ep_ids=[0, 1], "
        "max_lane_time_ms=4.0, barrier_time_ms=4.0, barrier_end_time_s=0.008"
    )

    assert records == [
        {
            "cluster": "PREFILL",
            "batch_id": 7,
            "layer_id": 3,
            "phase": "combine",
            "expected_ep_ids": [0, 1],
            "arrived_ep_ids": [0, 1],
            "max_lane_time_ms": 4.0,
            "barrier_time_ms": 4.0,
            "barrier_end_time_s": 0.008,
        }
    ]


def test_ep_conservation_parser_accepts_logger_prefix() -> None:
    records = _parse_ep_conservation_records(
        "INFO 12:00:00 scheduler.py:1] "
        "[EP-CONSERVATION][DECODE_FFN] batch_id=7, layer_id=3, "
        "routing_token_count=2, router_topk=2, total_routed_assignments=4, "
        "per_ep_routed_tokens={0: 1, 1: 3}"
    )

    assert records == [
        {
            "cluster": "DECODE_FFN",
            "batch_id": 7,
            "layer_id": 3,
            "routing_token_count": 2,
            "router_topk": 2,
            "total_routed_assignments": 4,
            "per_ep_routed_tokens": {0: 1, 1: 3},
        }
    ]


def test_strict_checker_does_not_merge_ep_ids_from_different_waves(tmp_path: Path) -> None:
    case = next(
        replace(case, num_layers=1, moe_layer_ids=(0,))
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "moe"
        and case.ep_size == 2
    )
    log_path = tmp_path / "split_wave.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id=0",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id=0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=0, "
                "moe_ep_size=2, per_expert_tokens={0: 1}, lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=11, layer_id=0, ep_id=1, "
                "moe_ep_size=2, per_expert_tokens={1: 1}, lane_compute_ms=1.0, lane_comm_ms=0.0",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "participants are incomplete" in result["errors"]


def test_strict_shared_checker_requires_layer_barrier_evidence(tmp_path: Path) -> None:
    case = next(
        replace(case, num_layers=1, moe_layer_ids=(0,))
        for case in build_matrix(REPO_ROOT)
        if case.architecture == "co-location"
        and case.model_kind == "moe"
        and case.ep_size == 2
    )
    log_path = tmp_path / "missing_barrier.log"
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "system_metrics.json").write_text(
        json.dumps({"ttft_statistics": {"mean": 1.0}}), encoding="utf-8"
    )
    log_path.write_text(
        "\n".join(
            [
                "Dummy Mode: false",
                "Simulation completed successfully.",
                "[OP-TRACE][MONOLITHIC][MOE][moe_shuffling] layer_id=0",
                "[OP-TRACE][MONOLITHIC][MOE][moe_grouped_gemm] layer_id=0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=0, "
                "moe_ep_size=2, per_expert_tokens={0: 1}, lane_compute_ms=1.0, lane_comm_ms=0.0",
                "[EP-WORKLOAD][MONOLITHIC] batch_id=10, layer_id=0, ep_id=1, "
                "moe_ep_size=2, per_expert_tokens={1: 1}, lane_compute_ms=2.0, lane_comm_ms=0.0",
                "[EP-CONSERVATION][MONOLITHIC] batch_id=10, layer_id=0, routing_token_count=1, router_topk=2, total_routed_assignments=2, per_ep_routed_tokens={0: 1, 1: 1}",
            ]
        ),
        encoding="utf-8",
    )

    result = check_case_log(case, log_path, metrics_dir, strict_layers=True)

    assert result["status"] == "FAIL"
    assert "missing EP barrier evidence" in result["errors"]


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


def test_result_ledger_merges_partial_runs_without_erasing_prior_cases() -> None:
    existing = [
        {"case_id": "case-a", "status": "PASS", "attempt": 1},
        {"case_id": "case-b", "status": "FAIL", "attempt": 1},
    ]
    rerun = [{"case_id": "case-b", "status": "PASS", "attempt": 2}]

    merged = _merge_result_rows(
        existing,
        rerun,
        expected_case_ids=("case-a", "case-b", "case-c"),
    )

    assert [row["case_id"] for row in merged] == ["case-a", "case-b"]
    assert merged[0]["attempt"] == 1
    assert merged[1]["status"] == "PASS"
    assert merged[1]["attempt"] == 2


def test_result_ledger_rejects_rows_without_canonical_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance"):
        _validate_result_ledger_provenance(
            [{"case_id": "case-a", "status": "PASS"}],
            repo_root=tmp_path / "repo",
            output_root=tmp_path / "output",
            results_path=tmp_path / "results.jsonl",
        )


def test_result_ledger_rejects_rows_from_another_output_root(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    results_path = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="output_root"):
        _validate_result_ledger_provenance(
            [
                {
                    "case_id": "case-a",
                    "status": "PASS",
                    "repo_root": str(tmp_path / "repo"),
                    "output_root": str(tmp_path / "old-output"),
                    "results_path": str(results_path),
                    "log_path": str(output_root / "case-a" / "case-a.log"),
                    "metrics_path": str(output_root / "case-a" / "metrics"),
                }
            ],
            repo_root=tmp_path / "repo",
            output_root=output_root,
            results_path=results_path,
        )


def test_result_ledger_rejects_external_log_and_metrics_paths(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    case_root = output_root / "case-a"
    results_path = tmp_path / "results.jsonl"
    with pytest.raises(ValueError, match="log_path"):
        _validate_result_ledger_provenance(
            [
                {
                    "case_id": "case-a",
                    "status": "PASS",
                    "repo_root": str(tmp_path / "repo"),
                    "output_root": str(output_root),
                    "results_path": str(results_path),
                    "log_path": str(tmp_path / "old.log"),
                    "metrics_path": str(case_root / "metrics"),
                }
            ],
            repo_root=tmp_path / "repo",
            output_root=output_root,
            results_path=results_path,
        )


def test_find_metrics_dir_rejects_stale_metrics(tmp_path: Path) -> None:
    case = next(iter(build_matrix(REPO_ROOT)))
    case_root = tmp_path / case.case_id / "metrics" / "run"
    case_root.mkdir(parents=True)
    metrics_path = case_root / "system_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="fresh"):
        _find_metrics_dir(tmp_path, case, started_at_ns=metrics_path.stat().st_mtime_ns + 1)
