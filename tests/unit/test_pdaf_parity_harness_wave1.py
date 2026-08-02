import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import pytest

from tests.e2e.pd_af_parity import harness


def _complete_row() -> dict[str, int | float]:
    return {
        "request_num_prefill_tokens": 512,
        "request_num_decode_tokens": 128,
        "request_num_tokens": 640,
        "request_num_restarts": 0,
        "request_thinking_round_count": 0,
        "request_e2e_time": 10.0,
        "request_execution_time": 9.0,
        "prefill_e2e_time": 4.0,
        "decode_e2e_time": 6.0,
        "ttft": 4.0,
        "tpot": 0.05,
        "transfer_kv_cache": 0.1,
        "transfer_m2n_total": 0.2,
        "transfer_m2n_attn_to_ffn": 0.1,
        "transfer_m2n_ffn_to_attn": 0.1,
        "cluster_prefill_computation": 3.0,
        "cluster_decode_attn_computation": 2.0,
        "cluster_decode_ffn_computation": 4.0,
        "cross_branch_first_token_ttft_ms": 4.5,
    }


def _write_metrics_csv(
    directory: Path,
    rows: list[dict[str, object]],
    *,
    header: list[str] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request_metrics.csv"
    fieldnames = header or ["Request Id", *rows[0].keys()]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for request_id, row in enumerate(rows):
            writer.writerow({"Request Id": request_id, **row})
    return path


def _write_raw_csv(directory: Path, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request_metrics.csv"
    path.write_text(content, encoding="utf-8")
    return path


def _write_ground_truth(directory: Path, records: list[dict[str, object]]) -> Path:
    path = directory / "metrics_ground_truth.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _write_reference_lifecycle(
    directory: Path,
    *,
    request_id: int = 0,
    arrived_at_s: float = 0.0,
    prefill_completed_at_s: float = 1.0,
    raw_decode_execution_completed_at_s: float = 2.0,
    resolved_global_end_time_s: float = 2.0,
) -> Path:
    payload = {
        "schema_version": "frontier.pdaf.reference-first-real-decode/v1",
        "producer": {
            "branch_kind": "reference",
            "reference_repo_root": (
                "/data/ycfeng/stepfun-performance-optimization/Frontier/"
                "worktrees/ref-afd-readonly"
            ),
            "reference_git_head": (
                "dcb1cc8ee160a9c3c5412293d93b64042960aa4d"
            ),
            "python_executable": "/test/python",
            "argv_sha256": "a" * 64,
            "observer_source_sha256": hashlib.sha256(
                Path(harness.__file__)
                .with_name("reference_lifecycle_observer.py")
                .read_bytes()
            ).hexdigest(),
            "bootstrap_source_sha256": hashlib.sha256(
                Path(harness.__file__)
                .with_name("reference_observer_bootstrap.py")
                .read_bytes()
            ).hexdigest(),
            "request_source_sha256": (
                "4cff6da775a1b04ba4c252ccc679a3f2919ed5bfc98f1c039dff1519b9bc42b0"
            ),
            "cluster_scheduler_source_sha256": (
                "5a28d18a7cfdcfc04b2848a9861973c652947005b356fa9b837b90356329fb6d"
            ),
            "global_batch_end_event_source_sha256": (
                "5366bd739c9765ef57b06448ce719d013795273535fd623aa17ed064279021b0"
            ),
            "candidate_hook": (
                "BaseClusterScheduler."
                "resolve_decode_attn_boundary_first_mixed_global_end_time"
            ),
            "transition_hook": "GlobalBatchEndEvent.handle_event",
            "transition_contract": "num_processed_decode_tokens:0->1",
            "timestamp_contract": (
                "resolver_input_time_before_observation_delay"
            ),
        },
        "request_count": 1,
        "request_ids_sha256": hashlib.sha256(
            json.dumps([request_id], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "requests": [
            {
                "request_id": request_id,
                "cluster_type": "DECODE_ATTN",
                "arrived_at_s": arrived_at_s,
                "prefill_completed_at_s": prefill_completed_at_s,
                "raw_decode_execution_completed_at_s": (
                    raw_decode_execution_completed_at_s
                ),
                "resolved_global_end_time_s": resolved_global_end_time_s,
                "processed_decode_tokens_before": 0,
                "processed_decode_tokens_after": 1,
            }
        ],
    }
    path = directory / harness.REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _write_minimal_event_logs(directory: Path) -> None:
    log_dir = directory / "logs" / "cluster_events"
    log_dir.mkdir(parents=True, exist_ok=True)
    for index, role in enumerate(("PREFILL", "DECODE_ATTN", "DECODE_FFN")):
        event_id = index + 1
        event_time = float(index)
        common = (
            f"event_time: {event_time} | cluster: {role} | "
            f"event_type: RequestArrivalEvent | event_id: {event_id} | "
            f"target_cluster: {role}"
        )
        path = log_dir / f"{role.lower()}_wave1.log"
        path.write_text(
            "\n".join(
                [
                    "=== VIDUR CLUSTER EVENT LOG ===",
                    f"Cluster Type: {role}",
                    "Start Time: 2026-07-11 00:00:00",
                    "Log Level: INFO",
                    f"Log File: {path}",
                    "==================================================",
                    "",
                    f"[2026-07-11 00:00:00.000] START RequestArrivalEvent | ID: {event_id} | {common}",
                    f"[2026-07-11 00:00:00.001] COMPLETE RequestArrivalEvent | ID: {event_id} | Duration: 0.1ms | {common} | request_id: {index} | new_events_generated: 1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def _completion_record(
    request_id: int = 0,
    *,
    arrived_at: float = 1.0,
    prefill_completed_at: float = 1.004,
    first_decode_token_completed_at: float = 1.0045,
) -> dict[str, object]:
    return {
        "event_type": "request_completion",
        "request_id": request_id,
        "arrived_at": arrived_at,
        "prefill_completed_at": prefill_completed_at,
        "first_decode_token_completed_at": first_decode_token_completed_at,
    }


def _case(num_requests: int = 1, layer: harness.ParityLayer = harness.ParityLayer.L2_TRAINED) -> harness.ParityCaseConfig:
    return harness.ParityCaseConfig(
        case_id="wave1",
        model="test-model",
        mode="offline",
        scale_gpu=24,
        prefill_tokens=512,
        decode_tokens=128,
        num_requests=num_requests,
        layer=layer,
    )


def _required_callable(name: str) -> Callable[..., object]:
    function = getattr(harness, name, None)
    assert callable(function), f"Required Wave 1 API is missing: {name}"
    return function


def _comparison_field(
    comparison: harness.RequestComparison,
    field_name: str,
) -> harness.FieldComparison:
    return next(field for field in comparison.fields if field.field_name == field_name)


def test_load_request_metrics_rejects_header_only_csv(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path, "Request Id,request_e2e_time\n")

    with pytest.raises(ValueError, match="no request rows"):
        harness.load_request_metrics(str(tmp_path))


def test_load_request_metrics_rejects_duplicate_headers(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path, "Request Id,ttft,ttft\n0,4.0,4.0\n")

    with pytest.raises(ValueError, match="duplicate.*header"):
        harness.load_request_metrics(str(tmp_path))


def test_load_request_metrics_rejects_missing_request_id_header(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path, "request_e2e_time\n10.0\n")

    with pytest.raises(ValueError, match="Request Id"):
        harness.load_request_metrics(str(tmp_path))


def test_load_request_metrics_rejects_non_integer_request_id(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path, "Request Id,request_e2e_time\nnot-an-id,10.0\n")

    with pytest.raises(ValueError, match="Request Id"):
        harness.load_request_metrics(str(tmp_path))


def test_load_request_metrics_rejects_duplicate_request_id(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path, "Request Id,request_e2e_time\n0,10.0\n0,11.0\n")

    with pytest.raises(ValueError, match="duplicate Request Id"):
        harness.load_request_metrics(str(tmp_path))


def test_load_request_metrics_rejects_rows_with_extra_columns(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path, "Request Id,request_e2e_time\n0,10.0,unexpected\n")

    with pytest.raises(harness.ParityInputError, match="extra columns.*line 2"):
        harness.load_request_metrics(str(tmp_path))


def test_load_request_metrics_rejects_ambiguous_recursive_matches(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path / "run-a", "Request Id,request_e2e_time\n0,10.0\n")
    _write_raw_csv(tmp_path / "run-b", "Request Id,request_e2e_time\n0,10.0\n")

    with pytest.raises(ValueError, match="(?i)ambiguous"):
        harness.load_request_metrics(str(tmp_path))


def test_load_request_metrics_accepts_explicit_root_file(tmp_path: Path) -> None:
    _write_raw_csv(tmp_path, "Request Id,request_e2e_time\n0,10.0\n")
    _write_raw_csv(tmp_path / "old-run", "Request Id,request_e2e_time\n0,99.0\n")

    metrics = harness.load_request_metrics(str(tmp_path))

    assert metrics[0]["request_e2e_time"] == 10.0


def test_generate_report_rejects_both_empty_outputs(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_raw_csv(main_dir, "Request Id,request_e2e_time\n")
    _write_raw_csv(ref_dir, "Request Id,request_e2e_time\n")
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    with pytest.raises(ValueError, match="no request rows"):
        harness.generate_report(_case(), str(main_dir), str(ref_dir))


def test_generate_report_fails_identically_truncated_request_sets(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_metrics_csv(main_dir, [_complete_row()])
    _write_metrics_csv(ref_dir, [_complete_row()])
    _write_ground_truth(main_dir, [_completion_record()])
    _write_reference_lifecycle(ref_dir)
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    report = harness.generate_report(_case(num_requests=2), str(main_dir), str(ref_dir))

    assert report.overall_pass is False
    assert report.total_mismatches >= 1


def test_generate_report_fails_unexpected_request_ids(tmp_path: Path) -> None:
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    row = _complete_row()
    path = _write_metrics_csv(main_dir, [row])
    content = path.read_text(encoding="utf-8").replace("\n0,", "\n1,")
    path.write_text(content, encoding="utf-8")
    path = _write_metrics_csv(ref_dir, [row])
    content = path.read_text(encoding="utf-8").replace("\n0,", "\n1,")
    path.write_text(content, encoding="utf-8")
    _write_ground_truth(main_dir, [_completion_record(request_id=1)])
    _write_reference_lifecycle(ref_dir, request_id=1)
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    report = harness.generate_report(_case(num_requests=1), str(main_dir), str(ref_dir))

    assert report.overall_pass is False
    assert any(
        comparison.first_divergence_field == "UNEXPECTED_REQUEST_ID"
        for comparison in report.request_comparisons
    )


def test_compare_requests_marks_equal_nonnumeric_metric_invalid() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    main_row["request_e2e_time"] = "invalid"
    ref_row["request_e2e_time"] = "invalid"

    [comparison] = harness.compare_requests(
        {0: main_row}, {0: ref_row}, harness.ParityLayer.L2_TRAINED
    )

    field = _comparison_field(comparison, "request_e2e_time")
    assert field.result.value == "invalid_value"
    assert comparison.passed is False


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_compare_requests_marks_nonfinite_metric_invalid(invalid_value: float) -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    main_row["request_e2e_time"] = invalid_value

    [comparison] = harness.compare_requests(
        {0: main_row}, {0: ref_row}, harness.ParityLayer.L2_TRAINED
    )

    field = _comparison_field(comparison, "request_e2e_time")
    assert field.result.value == "invalid_value"
    assert comparison.passed is False


def test_l1_ignores_numeric_metric_mismatch() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    main_row["request_e2e_time"] = 99.0

    [comparison] = harness.compare_requests(
        {0: main_row}, {0: ref_row}, harness.ParityLayer.L1_DUMMY
    )

    assert comparison.passed is True
    assert {field.field_name for field in comparison.fields} == set(harness.DISCRETE_FIELDS)


def test_l2_compares_numeric_metric_mismatch() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    main_row["request_e2e_time"] = 99.0

    [comparison] = harness.compare_requests(
        {0: main_row}, {0: ref_row}, harness.ParityLayer.L2_TRAINED
    )

    assert comparison.passed is False
    assert _comparison_field(comparison, "request_e2e_time").result.value == "mismatch"


def test_l2_compares_cross_branch_ttft_not_raw_ttft() -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    main_row["ttft"] = 4.0
    ref_row["ttft"] = 4.5
    main_row["cross_branch_first_token_ttft_ms"] = 4.5
    ref_row["cross_branch_first_token_ttft_ms"] = 4.5

    [comparison] = harness.compare_requests(
        {0: main_row}, {0: ref_row}, harness.ParityLayer.L2_TRAINED
    )

    assert comparison.passed is True
    compared_names = {field.field_name for field in comparison.fields}
    assert "cross_branch_first_token_ttft_ms" in compared_names
    assert "ttft" not in compared_names


def test_report_counts_missing_fields_as_mismatches(tmp_path: Path) -> None:
    main_row = _complete_row()
    ref_row = _complete_row()
    del main_row["request_e2e_time"]
    del ref_row["request_e2e_time"]
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_metrics_csv(main_dir, [main_row])
    _write_metrics_csv(ref_dir, [ref_row])
    _write_ground_truth(main_dir, [_completion_record()])
    _write_reference_lifecycle(ref_dir)
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    report = harness.generate_report(_case(), str(main_dir), str(ref_dir))

    assert report.overall_pass is False
    assert report.total_mismatches >= 1


def test_markdown_renders_missing_field_details(tmp_path: Path) -> None:
    row = _complete_row()
    del row["request_e2e_time"]
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_metrics_csv(main_dir, [row])
    _write_metrics_csv(ref_dir, [row])
    _write_ground_truth(main_dir, [_completion_record()])
    _write_reference_lifecycle(ref_dir)
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    markdown = harness.report_to_markdown(
        harness.generate_report(_case(), str(main_dir), str(ref_dir))
    )

    assert "request_e2e_time" in markdown
    assert "missing_field" in markdown


def test_markdown_renders_invalid_field_details() -> None:
    row = _complete_row()
    row["request_e2e_time"] = "invalid"
    [comparison] = harness.compare_requests(
        {0: row}, {0: dict(row)}, harness.ParityLayer.L2_TRAINED
    )
    report = harness.ParityReport(
        case_config=_case(),
        main_output_dir="main",
        ref_output_dir="ref",
        request_comparisons=[comparison],
        overall_pass=False,
        total_fields_compared=len(comparison.fields),
        total_mismatches=1,
    )

    markdown = harness.report_to_markdown(report)

    assert "request_e2e_time" in markdown
    assert "invalid_value" in markdown


def test_main_adapter_derives_both_ttft_metrics(tmp_path: Path) -> None:
    row = _complete_row()
    row["ttft"] = 4.0
    _write_metrics_csv(tmp_path, [row])
    _write_ground_truth(tmp_path, [_completion_record()])
    adapter = _required_callable("load_main_request_metrics")

    metrics = adapter(str(tmp_path))

    assert metrics[0]["canonical_main_ttft_ms"] == pytest.approx(4.0)
    assert metrics[0]["cross_branch_first_token_ttft_ms"] == pytest.approx(4.5)


def test_main_adapter_requires_ground_truth_file(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    adapter = _required_callable("load_main_request_metrics")

    with pytest.raises(ValueError, match="metrics_ground_truth.jsonl"):
        adapter(str(tmp_path))


def test_main_adapter_rejects_malformed_ground_truth_json(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    (tmp_path / "metrics_ground_truth.jsonl").write_text("{not-json}\n", encoding="utf-8")
    adapter = _required_callable("load_main_request_metrics")

    with pytest.raises(ValueError, match="(?i)malformed.*JSON"):
        adapter(str(tmp_path))


def test_main_adapter_rejects_duplicate_completion_record(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    record = _completion_record()
    _write_ground_truth(tmp_path, [record, record])
    adapter = _required_callable("load_main_request_metrics")

    with pytest.raises(ValueError, match="(?i)duplicate.*completion"):
        adapter(str(tmp_path))


def test_main_adapter_rejects_missing_ttft_timestamp(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    record = _completion_record()
    del record["first_decode_token_completed_at"]
    _write_ground_truth(tmp_path, [record])
    adapter = _required_callable("load_main_request_metrics")

    with pytest.raises(ValueError, match="first_decode_token_completed_at"):
        adapter(str(tmp_path))


def test_main_adapter_rejects_nonfinite_ttft_timestamp(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    _write_ground_truth(
        tmp_path,
        [_completion_record(first_decode_token_completed_at=float("nan"))],
    )
    adapter = _required_callable("load_main_request_metrics")

    with pytest.raises(ValueError, match="finite"):
        adapter(str(tmp_path))


def test_main_adapter_rejects_ground_truth_request_set_mismatch(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, [_complete_row(), _complete_row()])
    _write_ground_truth(tmp_path, [_completion_record(request_id=0)])
    adapter = _required_callable("load_main_request_metrics")

    with pytest.raises(ValueError, match="request.*set"):
        adapter(str(tmp_path))


def test_main_adapter_rejects_csv_canonical_ttft_mismatch(tmp_path: Path) -> None:
    row = _complete_row()
    row["ttft"] = 9.0
    _write_metrics_csv(tmp_path, [row])
    _write_ground_truth(tmp_path, [_completion_record()])
    adapter = _required_callable("load_main_request_metrics")

    with pytest.raises(ValueError, match="canonical.*ttft"):
        adapter(str(tmp_path))


def test_reference_adapter_requires_first_real_decode_lifecycle_sidecar(
    tmp_path: Path,
) -> None:
    row = _complete_row()
    row["ttft"] = 4.5
    _write_metrics_csv(tmp_path, [row])
    adapter = _required_callable("load_reference_request_metrics")

    with pytest.raises(
        harness.ParityInputError,
        match="reference_first_real_decode_lifecycle.json",
    ):
        adapter(str(tmp_path))


def test_reference_adapter_uses_direct_execution_timestamp(tmp_path: Path) -> None:
    row = _complete_row()
    row["ttft"] = 1000.0
    _write_metrics_csv(tmp_path, [row])
    _write_reference_lifecycle(tmp_path)
    adapter = _required_callable("load_reference_request_metrics")

    metrics = adapter(str(tmp_path))

    assert metrics[0]["cross_branch_first_token_ttft_ms"] == 2000.0
    assert "canonical_main_ttft_ms" not in metrics[0]


def test_reference_adapter_rejects_nonfinite_raw_ttft(tmp_path: Path) -> None:
    row = _complete_row()
    row["ttft"] = float("inf")
    _write_metrics_csv(tmp_path, [row])
    _write_reference_lifecycle(tmp_path)
    adapter = _required_callable("load_reference_request_metrics")

    with pytest.raises(ValueError, match="finite"):
        adapter(str(tmp_path))


def test_reference_adapter_rejects_derived_ttft_overflow(tmp_path: Path) -> None:
    row = _complete_row()
    row["ttft"] = 1000.0
    _write_metrics_csv(tmp_path, [row])
    _write_reference_lifecycle(
        tmp_path,
        raw_decode_execution_completed_at_s=1e308,
        resolved_global_end_time_s=1e308,
    )

    with pytest.raises(ValueError, match="derived.*TTFT.*finite"):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize(
    "field_name",
    ["observer_source_sha256", "bootstrap_source_sha256"],
)
def test_reference_adapter_rejects_live_control_source_hash_mismatch(
    tmp_path: Path,
    field_name: str,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["producer"][field_name] = "0" * 64
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match=field_name):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_uses_raw_execution_before_delayed_observation(
    tmp_path: Path,
) -> None:
    row = _complete_row()
    row["ttft"] = 5500.0
    _write_metrics_csv(tmp_path, [row])
    _write_reference_lifecycle(
        tmp_path,
        prefill_completed_at_s=4.0,
        raw_decode_execution_completed_at_s=4.125,
        resolved_global_end_time_s=5.5,
    )

    metrics = harness.load_reference_request_metrics(str(tmp_path))

    assert metrics[0]["cross_branch_first_token_ttft_ms"] == 4125.0
    assert metrics[0]["cross_branch_first_token_ttft_ms"] != 5500.0


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("top", "schema_version", "wrong", "schema_version"),
        ("top", "unexpected", True, "unexpected"),
        ("producer", "branch_kind", "main", "branch_kind"),
        ("producer", "reference_repo_root", "/wrong", "reference_repo_root"),
        ("producer", "reference_git_head", "0" * 40, "reference_git_head"),
        ("producer", "request_source_sha256", "0" * 64, "request_source_sha256"),
        (
            "producer",
            "cluster_scheduler_source_sha256",
            "0" * 64,
            "cluster_scheduler_source_sha256",
        ),
        (
            "producer",
            "global_batch_end_event_source_sha256",
            "0" * 64,
            "global_batch_end_event_source_sha256",
        ),
        ("producer", "candidate_hook", "wrong", "candidate_hook"),
        ("producer", "transition_hook", "wrong", "transition_hook"),
        ("producer", "transition_contract", "0->2", "transition_contract"),
        ("producer", "timestamp_contract", "observation", "timestamp_contract"),
        ("request", "cluster_type", "PREFILL", "cluster_type"),
        ("request", "processed_decode_tokens_before", 1, "processed_decode_tokens_before"),
        ("request", "processed_decode_tokens_after", 2, "processed_decode_tokens_after"),
        ("request", "unexpected", True, "unexpected"),
    ],
)
def test_reference_adapter_rejects_invalid_lifecycle_contract(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if section == "top":
        target = payload
    elif section == "request":
        target = payload["requests"][0]
    else:
        target = payload[section]
    target[key] = value
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(harness.ParityInputError, match=message):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("top", "producer"),
        ("producer", "transition_contract"),
        ("request", "request_id"),
    ],
)
def test_reference_adapter_rejects_missing_lifecycle_key(
    tmp_path: Path,
    section: str,
    key: str,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if section == "top":
        target = payload
    elif section == "request":
        target = payload["requests"][0]
    else:
        target = payload[section]
    del target[key]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="missing"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_rejects_non_object_lifecycle_json(
    tmp_path: Path,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = tmp_path / harness.REFERENCE_FIRST_REAL_DECODE_LIFECYCLE_FILENAME
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="must contain an object"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_rejects_empty_lifecycle_requests(
    tmp_path: Path,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["requests"] = []
    payload["request_count"] = 0
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="must contain requests"):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize("request_id", [True, -1, 0.5])
def test_reference_adapter_rejects_invalid_lifecycle_request_id(
    tmp_path: Path,
    request_id: object,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["requests"][0]["request_id"] = request_id
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="request_id"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_rejects_duplicate_lifecycle_request_id(
    tmp_path: Path,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["requests"].append(dict(payload["requests"][0]))
    payload["request_count"] = 2
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="duplicate request_id"):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize("python_executable", ["", "relative/python", 3])
def test_reference_adapter_rejects_invalid_python_executable(
    tmp_path: Path,
    python_executable: object,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["producer"]["python_executable"] = python_executable
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="python_executable"):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("argv_sha256", "A" * 64),
        ("observer_source_sha256", "abc"),
        ("bootstrap_source_sha256", 3),
    ],
)
def test_reference_adapter_rejects_invalid_producer_sha256(
    tmp_path: Path,
    field_name: str,
    field_value: object,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["producer"][field_name] = field_value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match=field_name):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arrived_at_s", float("nan")),
        ("prefill_completed_at_s", float("inf")),
        ("raw_decode_execution_completed_at_s", float("-inf")),
        ("resolved_global_end_time_s", float("nan")),
        ("arrived_at_s", -1.0),
    ],
)
def test_reference_adapter_rejects_invalid_lifecycle_timestamp(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["requests"][0][field] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match=field):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize("value", [True, "2.0"])
def test_reference_adapter_rejects_non_json_numeric_lifecycle_timestamp(
    tmp_path: Path,
    value: object,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["requests"][0]["raw_decode_execution_completed_at_s"] = value
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        harness.ParityInputError,
        match="raw_decode_execution_completed_at_s",
    ):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize(
    ("before", "after"),
    [(False, True), (0.0, 1.0), ("0", "1")],
)
def test_reference_adapter_rejects_non_integer_transition_evidence(
    tmp_path: Path,
    before: object,
    after: object,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["requests"][0]["processed_decode_tokens_before"] = before
    payload["requests"][0]["processed_decode_tokens_after"] = after
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(harness.ParityInputError, match="processed_decode_tokens"):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize(
    ("arrived", "prefill", "execution", "resolved"),
    [
        (2.0, 1.0, 3.0, 3.0),
        (0.0, 2.0, 1.5, 3.0),
        (0.0, 1.0, 3.0, 2.0),
    ],
)
def test_reference_adapter_rejects_lifecycle_timestamp_order_violation(
    tmp_path: Path,
    arrived: float,
    prefill: float,
    execution: float,
    resolved: float,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    _write_reference_lifecycle(
        tmp_path,
        arrived_at_s=arrived,
        prefill_completed_at_s=prefill,
        raw_decode_execution_completed_at_s=execution,
        resolved_global_end_time_s=resolved,
    )

    with pytest.raises(harness.ParityInputError, match="timestamp order"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_rejects_lifecycle_request_count_mismatch(
    tmp_path: Path,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["request_count"] = 2
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="request_count"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_rejects_lifecycle_request_id_hash_mismatch(
    tmp_path: Path,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["request_ids_sha256"] = "0" * 64
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(harness.ParityInputError, match="request_ids_sha256"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_rejects_duplicate_json_key(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            '"schema_version":',
            '"schema_version":"duplicate","schema_version":',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(harness.ParityInputError, match="duplicate.*schema_version"):
        harness.load_reference_request_metrics(str(tmp_path))


@pytest.mark.parametrize("suffix", ["\n", " "])
def test_reference_adapter_rejects_noncanonical_lifecycle_bytes(
    tmp_path: Path,
    suffix: str,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row()])
    path = _write_reference_lifecycle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(payload, sort_keys=False, indent=2) + "\n" + suffix,
        encoding="utf-8",
    )

    with pytest.raises(harness.ParityInputError, match="canonical"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_reference_adapter_rejects_csv_sidecar_request_set_mismatch(
    tmp_path: Path,
) -> None:
    _write_metrics_csv(tmp_path, [_complete_row(), _complete_row()])
    _write_reference_lifecycle(tmp_path)

    with pytest.raises(harness.ParityInputError, match="request sets differ"):
        harness.load_reference_request_metrics(str(tmp_path))


def test_generate_report_uses_branch_specific_ttft_adapters(tmp_path: Path) -> None:
    main_row = _complete_row()
    main_row["ttft"] = 4.0
    ref_row = _complete_row()
    ref_row["ttft"] = 4.5
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_metrics_csv(main_dir, [main_row])
    _write_ground_truth(main_dir, [_completion_record()])
    _write_metrics_csv(ref_dir, [ref_row])
    _write_reference_lifecycle(
        ref_dir,
        arrived_at_s=1.0,
        prefill_completed_at_s=1.004,
        raw_decode_execution_completed_at_s=1.0045,
        resolved_global_end_time_s=1.0045,
    )
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    report = harness.generate_report(_case(), str(main_dir), str(ref_dir))

    assert report.overall_pass is True
    [comparison] = report.request_comparisons
    assert comparison.canonical_main_ttft_ms == pytest.approx(4.0)
    assert comparison.canonical_main_ttft_provenance == "metrics_ground_truth.jsonl"


def test_markdown_exposes_ttft_provenance_and_source_directories(tmp_path: Path) -> None:
    main_row = _complete_row()
    main_row["ttft"] = 4.0
    ref_row = _complete_row()
    ref_row["ttft"] = 4.5
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    _write_metrics_csv(main_dir, [main_row])
    _write_ground_truth(main_dir, [_completion_record()])
    _write_metrics_csv(ref_dir, [ref_row])
    _write_reference_lifecycle(
        ref_dir,
        arrived_at_s=1.0,
        prefill_completed_at_s=1.004,
        raw_decode_execution_completed_at_s=1.0045,
        resolved_global_end_time_s=1.0045,
    )
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    markdown = harness.report_to_markdown(
        harness.generate_report(_case(), str(main_dir), str(ref_dir))
    )

    assert str(main_dir) in markdown
    assert str(ref_dir) in markdown
    assert "canonical_main_ttft_ms" in markdown
    assert "metrics_ground_truth.jsonl" in markdown


def test_markdown_always_exposes_cross_branch_ttft_evidence_pass(
    tmp_path: Path,
) -> None:
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    main_row = _complete_row()
    main_row["ttft"] = 1000.0
    ref_row = dict(main_row)
    _write_metrics_csv(main_dir, [main_row])
    _write_ground_truth(
        main_dir,
        [
            _completion_record(
                arrived_at=0.0,
                prefill_completed_at=1.0,
                first_decode_token_completed_at=2.0,
            )
        ],
    )
    _write_metrics_csv(ref_dir, [ref_row])
    _write_reference_lifecycle(ref_dir)
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    report = harness.generate_report(_case(), str(main_dir), str(ref_dir))
    markdown = harness.report_to_markdown(report)

    assert report.overall_pass is True
    assert "## Cross-branch First Real Decode TTFT" in markdown
    assert "| 0 | 2000.0 | 2000.0 | 0.0 | 0.0 | PASS |" in markdown
    assert (
        "metrics_ground_truth.jsonl#request_completion."
        "first_decode_token_completed_at" in markdown
    )
    assert harness.REFERENCE_GIT_HEAD in markdown


def test_markdown_always_exposes_cross_branch_ttft_evidence_fail(
    tmp_path: Path,
) -> None:
    main_dir = tmp_path / "main"
    ref_dir = tmp_path / "ref"
    main_row = _complete_row()
    main_row["ttft"] = 1000.0
    ref_row = dict(main_row)
    _write_metrics_csv(main_dir, [main_row])
    _write_ground_truth(
        main_dir,
        [
            _completion_record(
                arrived_at=0.0,
                prefill_completed_at=1.0,
                first_decode_token_completed_at=2.0,
            )
        ],
    )
    _write_metrics_csv(ref_dir, [ref_row])
    _write_reference_lifecycle(
        ref_dir,
        prefill_completed_at_s=4.0,
        raw_decode_execution_completed_at_s=4.125,
        resolved_global_end_time_s=5.5,
    )
    _write_minimal_event_logs(main_dir)
    _write_minimal_event_logs(ref_dir)

    report = harness.generate_report(_case(), str(main_dir), str(ref_dir))
    markdown = harness.report_to_markdown(report)

    assert report.overall_pass is False
    assert "## Cross-branch First Real Decode TTFT" in markdown
    assert (
        "| 0 | 2000.0 | 4125.0 | 2125.0 | "
        "0.5151515151515151 | FAIL |" in markdown
    )
    assert harness.REFERENCE_TTFT_PROVENANCE in markdown


def test_complete_l2_rows_are_finite_and_comparable() -> None:
    row = _complete_row()

    [comparison] = harness.compare_requests(
        {0: row}, {0: dict(row)}, harness.ParityLayer.L2_TRAINED
    )

    assert comparison.passed is True
    for field in comparison.fields:
        if field.abs_delta is not None:
            assert math.isfinite(field.abs_delta)
