import csv
from pathlib import Path

import pytest

from tests.e2e.operator_parity.build_true_mixed_profile_overlay import build_overlay


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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def test_build_overlay_maps_true_mixed_sources_to_canonical_supplement_names(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(canonical_root / "model_a" / "attention_kernel_only.csv", [{"k": "base"}])
    _write_csv(stage_root / "model_a" / "attention_true_mixed.csv", [_true_mixed_row()])
    _write_csv(
        stage_root / "model_a" / "attention_true_mixed_kernel_only.csv",
        [_true_mixed_row()],
    )

    summary = build_overlay(
        canonical_root=canonical_root,
        stage_root=stage_root,
        overlay_root=overlay_root,
        supplement_root=supplement_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
    )

    assert summary["status"] == "PASS"
    assert summary["mapped_file_count"] == 2
    assert summary["total_supplement_rows"] == 2
    assert (overlay_root / "model_a" / "attention.csv").is_file()
    assert _read_rows(supplement_root / "model_a" / "attention.csv")[0][
        "is_true_mixed_batch"
    ] == "True"
    assert _read_rows(supplement_root / "model_a" / "attention_kernel_only.csv")[0][
        "is_true_mixed_batch"
    ] == "True"


def test_build_overlay_fails_if_overlay_root_already_exists(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    overlay_root.mkdir()

    with pytest.raises(FileExistsError, match="overlay root already exists"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
        )


def test_build_overlay_rejects_non_true_mixed_source_rows(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(stage_root / "model_a" / "attention_true_mixed.csv", [_true_mixed_row(is_true_mixed_batch="False")])
    _write_csv(stage_root / "model_a" / "attention_true_mixed_kernel_only.csv", [_true_mixed_row()])

    with pytest.raises(ValueError, match="stage audit failed"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
        )


def test_build_overlay_reuses_full_stage_audit_for_invalid_timing(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(
        stage_root / "model_a" / "attention_true_mixed.csv",
        [_true_mixed_row(**{"time_stats.attn_decode.median": "0"})],
    )
    _write_csv(
        stage_root / "model_a" / "attention_true_mixed_kernel_only.csv",
        [_true_mixed_row()],
    )

    with pytest.raises(ValueError, match="stage audit failed"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
        )


def test_build_overlay_reuses_full_stage_audit_for_wrong_tp_coverage(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(stage_root / "model_a" / "attention_true_mixed.csv", [_true_mixed_row()])
    _write_csv(
        stage_root / "model_a" / "attention_true_mixed_kernel_only.csv",
        [_true_mixed_row()],
    )

    with pytest.raises(ValueError, match="stage audit failed"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1, 2),
        )
