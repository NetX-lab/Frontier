from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from tests.e2e.attention_equivalence.measurement_csv_equivalence import (
    compare_measurement_csv,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_measurement_equivalence_is_keyed_and_float_format_independent(tmp_path: Path) -> None:
    fieldnames = [
        "op_name",
        "batch_size",
        "time_stats.attn_prefill.mean",
        "time_stats.attn_prefill.median",
        "attention_backend",
    ]
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        reference,
        [
            {
                "op_name": "attn_prefill",
                "batch_size": "1",
                "time_stats.attn_prefill.mean": "1.0",
                "time_stats.attn_prefill.median": "1.0",
                "attention_backend": "FLASHINFER",
            },
            {
                "op_name": "attn_prefill",
                "batch_size": "2",
                "time_stats.attn_prefill.mean": "2.00",
                "time_stats.attn_prefill.median": "2.0",
                "attention_backend": "FLASHINFER",
            },
        ],
        fieldnames,
    )
    _write_csv(
        candidate,
        [
            {
                "op_name": "attn_prefill",
                "batch_size": "2",
                "time_stats.attn_prefill.mean": "2.0",
                "time_stats.attn_prefill.median": "2.000",
                "attention_backend": "FLASHINFER",
            },
            {
                "op_name": "attn_prefill",
                "batch_size": "1",
                "time_stats.attn_prefill.mean": "1.000000",
                "time_stats.attn_prefill.median": "1",
                "attention_backend": "FLASHINFER",
            },
        ],
        fieldnames,
    )

    report = compare_measurement_csv(
        reference,
        candidate,
        key_columns=("op_name", "batch_size"),
        case_manifest={"raw_capture_sha256": "same-input"},
    )

    assert report["status"] == "PASS"
    assert report["mismatch_count"] == 0
    assert report["row_count"] == {"reference": 2, "candidate": 2, "delta": 0}
    assert report["per_op_statistics"]["attn_prefill"]["reference"]["sample_count"] == 2
    assert report["per_op_statistics"]["attn_prefill"]["reference"]["mean_of_means_ms"] == pytest.approx(1.5)
    assert report["per_op_statistics"]["attn_prefill"]["candidate"]["p95_of_means_ms"] == pytest.approx(2.0)
    assert len(report["numeric_comparisons"]) == 4


def test_measurement_equivalence_fails_fast_on_duplicate_keys(tmp_path: Path) -> None:
    fieldnames = ["op_name", "batch_size", "time_stats.attn_decode.mean"]
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    rows = [
        {"op_name": "attn_decode", "batch_size": "1", "time_stats.attn_decode.mean": "1.0"},
        {"op_name": "attn_decode", "batch_size": "1", "time_stats.attn_decode.mean": "1.0"},
    ]
    _write_csv(reference, rows[:1], fieldnames)
    _write_csv(candidate, rows, fieldnames)

    with pytest.raises(ValueError, match="duplicate key"):
        compare_measurement_csv(reference, candidate, key_columns=("op_name", "batch_size"))


def test_measurement_equivalence_fails_fast_on_schema_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, [{"op_name": "attn", "latency_ms": "1.0"}], ["op_name", "latency_ms"])
    _write_csv(candidate, [{"op_name": "attn", "latency_ms": "1.0", "extra": "x"}], ["op_name", "latency_ms", "extra"])

    with pytest.raises(ValueError, match="schema mismatch"):
        compare_measurement_csv(reference, candidate, key_columns=("op_name",))


def test_measurement_equivalence_fails_fast_on_empty_data_rows(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, [], ["op_name", "time_stats.attn_prefill.mean"])
    _write_csv(candidate, [], ["op_name", "time_stats.attn_prefill.mean"])

    with pytest.raises(ValueError, match="must contain at least one data row"):
        compare_measurement_csv(reference, candidate, key_columns=("op_name",))


def test_measurement_equivalence_reports_missing_and_extra_keys(tmp_path: Path) -> None:
    fieldnames = ["op_name", "batch_size", "time_stats.attn_prefill.mean"]
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        reference,
        [{"op_name": "attn_prefill", "batch_size": "1", "time_stats.attn_prefill.mean": "1.0"}],
        fieldnames,
    )
    _write_csv(
        candidate,
        [{"op_name": "attn_prefill", "batch_size": "2", "time_stats.attn_prefill.mean": "1.0"}],
        fieldnames,
    )

    report = compare_measurement_csv(reference, candidate, key_columns=("op_name", "batch_size"))

    assert report["status"] == "FAIL"
    assert report["mismatch_count"] == 2
    assert report["missing_keys"] == [{"op_name": "attn_prefill", "batch_size": "1"}]
    assert report["extra_keys"] == [{"op_name": "attn_prefill", "batch_size": "2"}]


def test_measurement_equivalence_reports_numeric_mismatch_and_accepts_explicit_tolerance(
    tmp_path: Path,
) -> None:
    fieldnames = ["op_name", "time_stats.attn_prefill.mean"]
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, [{"op_name": "attn_prefill", "time_stats.attn_prefill.mean": "1.0"}], fieldnames)
    _write_csv(candidate, [{"op_name": "attn_prefill", "time_stats.attn_prefill.mean": "1.01"}], fieldnames)

    strict_report = compare_measurement_csv(reference, candidate, key_columns=("op_name",))

    assert strict_report["status"] == "FAIL"
    assert strict_report["mismatch_count"] == 1
    mismatch = strict_report["numeric_comparisons"][0]
    assert mismatch["reference_value"] == 1.0
    assert mismatch["candidate_value"] == 1.01
    assert mismatch["absolute_delta"] == pytest.approx(0.01)
    assert mismatch["relative_delta_pct"] == pytest.approx(1.0)
    assert mismatch["passed"] is False

    tolerance_report = compare_measurement_csv(
        reference,
        candidate,
        key_columns=("op_name",),
        tolerance_allowlist={"time_stats.attn_prefill.mean": {"absolute": 0.02}},
    )

    assert tolerance_report["status"] == "PASS"
    assert tolerance_report["mismatch_count"] == 0
    assert tolerance_report["numeric_comparisons"][0]["tolerance"] == {
        "absolute": 0.02,
        "relative_pct": 0.0,
    }


def test_measurement_equivalence_accepts_explicit_relative_tolerance(tmp_path: Path) -> None:
    fieldnames = ["op_name", "time_stats.attn_prefill.mean"]
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(reference, [{"op_name": "attn_prefill", "time_stats.attn_prefill.mean": "100.0"}], fieldnames)
    _write_csv(candidate, [{"op_name": "attn_prefill", "time_stats.attn_prefill.mean": "101.0"}], fieldnames)

    report = compare_measurement_csv(
        reference,
        candidate,
        key_columns=("op_name",),
        tolerance_allowlist={"time_stats.attn_prefill.mean": {"relative_pct": 1.0}},
    )

    assert report["status"] == "PASS"
    assert report["mismatch_count"] == 0
    assert report["numeric_comparisons"][0]["relative_delta_pct"] == pytest.approx(1.0)


def test_measurement_equivalence_handles_nan_and_inf_explicitly(tmp_path: Path) -> None:
    fieldnames = ["op_name", "time_stats.attn_prefill.mean", "time_stats.attn_decode.mean"]
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        reference,
        [{"op_name": "attn", "time_stats.attn_prefill.mean": "nan", "time_stats.attn_decode.mean": "inf"}],
        fieldnames,
    )
    _write_csv(
        candidate,
        [{"op_name": "attn", "time_stats.attn_prefill.mean": "NaN", "time_stats.attn_decode.mean": "inf"}],
        fieldnames,
    )

    same_specials = compare_measurement_csv(reference, candidate, key_columns=("op_name",))

    assert same_specials["status"] == "PASS"
    assert same_specials["mismatch_count"] == 0

    _write_csv(
        candidate,
        [{"op_name": "attn", "time_stats.attn_prefill.mean": "0.0", "time_stats.attn_decode.mean": "inf"}],
        fieldnames,
    )

    mismatch = compare_measurement_csv(reference, candidate, key_columns=("op_name",))

    assert mismatch["status"] == "FAIL"
    assert mismatch["mismatch_count"] == 1
    assert math.isnan(mismatch["numeric_comparisons"][0]["reference_value"])


def test_measurement_equivalence_fails_fast_on_missing_file(tmp_path: Path) -> None:
    reference = tmp_path / "missing.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(candidate, [{"op_name": "attn", "latency_ms": "1.0"}], ["op_name", "latency_ms"])

    with pytest.raises(FileNotFoundError):
        compare_measurement_csv(reference, candidate, key_columns=("op_name",))
