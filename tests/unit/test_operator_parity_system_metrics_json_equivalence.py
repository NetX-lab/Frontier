from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.operator_parity.system_metrics_json_equivalence import (
    compare_system_metrics_roots,
)


def _write_system_metrics(root: Path, relative_case_dir: str, payload: object) -> None:
    path = root / relative_case_dir / "system_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_compare_system_metrics_roots_passes_when_all_json_values_match_within_tolerance(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    case_dir = "model/offline_batch/model_offline_batch_co_location_cuda_event"
    _write_system_metrics(
        reference,
        case_dir,
        {
            "simulation_metadata": {"mode": "offline", "requests": 1},
            "ttft_statistics": {"mean": 1.0, "p95": [2.0, 3.0]},
            "notes": None,
        },
    )
    _write_system_metrics(
        candidate,
        case_dir,
        {
            "simulation_metadata": {"mode": "offline", "requests": 1},
            "ttft_statistics": {"mean": 1.0 + 5e-13, "p95": [2.0, 3.0]},
            "notes": None,
        },
    )

    report = compare_system_metrics_roots(reference, candidate)

    assert report["status"] == "PASS"
    assert report["case_count"] == 1
    assert report["numeric_fields_compared"] == 4
    assert report["mismatch_count"] == 0
    assert report["max_abs_delta"] == pytest.approx(5e-13)
    assert report["max_relative_delta_pct"] <= 1e-7


def test_compare_system_metrics_roots_accepts_relative_tolerance_when_absolute_delta_is_tiny(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    case_dir = "model/offline_batch/model_offline_batch_co_location_kernel_only"
    _write_system_metrics(
        reference,
        case_dir,
        {"throughput_metrics": {"tokens_per_second": 3286.643659492857}},
    )
    _write_system_metrics(
        candidate,
        case_dir,
        {"throughput_metrics": {"tokens_per_second": 3286.6436594928587}},
    )

    report = compare_system_metrics_roots(reference, candidate)

    assert report["status"] == "PASS"
    assert report["mismatch_count"] == 0
    assert report["numeric_fields_compared"] == 1
    assert report["max_abs_delta"] == pytest.approx(1.8189894035458565e-12)
    assert report["max_relative_delta_pct"] <= 1e-7


def test_compare_system_metrics_roots_reports_numeric_mismatch_outside_tolerance(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    case_dir = "model/online_serving/model_online_serving_pd_disaggregation_kernel_only"
    _write_system_metrics(reference, case_dir, {"throughput_metrics": {"qps": 100.0}})
    _write_system_metrics(candidate, case_dir, {"throughput_metrics": {"qps": 100.0001}})

    report = compare_system_metrics_roots(reference, candidate)

    assert report["status"] == "FAIL"
    assert report["mismatch_count"] == 1
    assert report["numeric_fields_compared"] == 1
    assert report["max_abs_delta"] == pytest.approx(0.0001)
    assert report["mismatches"][0]["path"] == "throughput_metrics.qps"


def test_compare_system_metrics_roots_reports_non_numeric_mismatch(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    case_dir = "model/offline_batch/model_offline_batch_co_location_kernel_only"
    _write_system_metrics(reference, case_dir, {"simulation_metadata": {"sys_arch": "co-location"}})
    _write_system_metrics(candidate, case_dir, {"simulation_metadata": {"sys_arch": "pd-disaggregation"}})

    report = compare_system_metrics_roots(reference, candidate)

    assert report["status"] == "FAIL"
    assert report["mismatch_count"] == 1
    assert report["numeric_fields_compared"] == 0
    assert report["mismatches"][0]["kind"] == "value_mismatch"


def test_compare_system_metrics_roots_fails_fast_when_case_sets_differ(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    candidate.mkdir(parents=True)
    _write_system_metrics(reference, "model/offline_batch/reference_only", {"value": 1})

    with pytest.raises(ValueError, match="system_metrics case set mismatch"):
        compare_system_metrics_roots(reference, candidate)
