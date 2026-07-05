import csv
from pathlib import Path

import pytest

from tests.e2e.operator_parity.audit_true_mixed_attention_stage import audit_stage


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _true_mixed_row(**overrides: str) -> dict[str, str]:
    row = {
        "num_tensor_parallel_workers": "1",
        "is_true_mixed_batch": "True",
        "decode_batch_size": "1",
        "decode_avg_kv_cache_size": "16",
        "num_prefill_seqs": "1",
        "total_prefill_tokens": "16",
        "total_batch_size": "2",
        "batch_composition_ratio": "0.5",
        "total_tokens": "17",
        "time_stats.attn_decode.median": "3.25",
    }
    row.update(overrides)
    return row


def _write_stage_pair(stage_root: Path, model: str, rows: list[dict[str, str]]) -> None:
    _write_csv(stage_root / model / "attention_true_mixed.csv", rows)
    _write_csv(stage_root / model / "attention_true_mixed_kernel_only.csv", rows)


def test_audit_stage_accepts_complete_true_mixed_rows(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    _write_stage_pair(stage_root, "model_a", [_true_mixed_row()])

    summary = audit_stage(
        stage_root=stage_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "PASS"
    assert summary["expected_file_count"] == 2
    assert summary["pass_count"] == 2
    assert summary["fail_count"] == 0
    assert summary["total_true_mixed_row_count"] == 2


def test_audit_stage_rejects_missing_true_mixed_file(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    _write_csv(stage_root / "model_a" / "attention_true_mixed.csv", [_true_mixed_row()])

    summary = audit_stage(
        stage_root=stage_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "FAIL"
    assert summary["observed_file_count"] == 1
    assert summary["fail_count"] == 1
    assert summary["reports"][1]["reason"] == "missing file"


def test_audit_stage_rejects_partial_decode_timings(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    _write_stage_pair(
        stage_root,
        "model_a",
        [_true_mixed_row(), _true_mixed_row(**{"time_stats.attn_decode.median": ""})],
    )

    summary = audit_stage(
        stage_root=stage_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=2,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "FAIL"
    assert summary["fail_count"] == 2
    assert "valid attn_decode rows 1 != true-mixed rows 2" in summary["reports"][0]["reasons"]


def test_audit_stage_rejects_non_numeric_true_mixed_features(tmp_path: Path) -> None:
    stage_root = tmp_path / "stage"
    _write_stage_pair(
        stage_root,
        "model_a",
        [_true_mixed_row(decode_batch_size="not-a-number")],
    )

    summary = audit_stage(
        stage_root=stage_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "FAIL"
    assert summary["reports"][0]["true_mixed_required_numeric_invalid_columns"] == {
        "decode_batch_size": 1
    }


@pytest.mark.parametrize("attn_decode_median", ["0", "-1"])
def test_audit_stage_rejects_non_positive_decode_timing(
    tmp_path: Path,
    attn_decode_median: str,
) -> None:
    stage_root = tmp_path / "stage"
    _write_stage_pair(
        stage_root,
        "model_a",
        [_true_mixed_row(**{"time_stats.attn_decode.median": attn_decode_median})],
    )

    summary = audit_stage(
        stage_root=stage_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "FAIL"
    assert "time_stats.attn_decode.median" in summary["reports"][0][
        "true_mixed_required_numeric_invalid_columns"
    ]


def test_audit_stage_rejects_inconsistent_true_mixed_batch_fields(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "stage"
    _write_stage_pair(
        stage_root,
        "model_a",
        [
            _true_mixed_row(
                total_batch_size="99",
                batch_composition_ratio="0.25",
                total_tokens="99",
            )
        ],
    )

    summary = audit_stage(
        stage_root=stage_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "FAIL"
    reasons = " ".join(summary["reports"][0]["reasons"])
    assert "inconsistent true-mixed rows" in reasons


def test_audit_stage_rejects_duplicate_true_mixed_profile_keys(
    tmp_path: Path,
) -> None:
    stage_root = tmp_path / "stage"
    _write_stage_pair(
        stage_root,
        "model_a",
        [
            _true_mixed_row(**{"time_stats.attn_decode.median": "3.25"}),
            _true_mixed_row(**{"time_stats.attn_decode.median": "3.50"}),
        ],
    )

    summary = audit_stage(
        stage_root=stage_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=2,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "FAIL"
    assert summary["reports"][0]["duplicate_true_mixed_profile_key_count"] == 1
    assert "duplicate true-mixed profile keys 1" in summary["reports"][0]["reasons"]
