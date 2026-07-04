from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.operator_parity.profiling_csv_equivalence import (
    GOLDEN_CONFIG_FILENAMES,
    compare_profiling_csv_roots,
    main,
    _required_relative_paths,
)
from tests.e2e.operator_parity.profile_prerequisite_audit import (
    REQUIRED_BASE_PROFILE_FILES,
    REQUIRED_MOE_PROFILE_FILES,
)


def _write_config(config_root: Path, name: str, payload: dict[str, object]) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_profile(
    root: Path,
    model_name: str,
    filename: str,
    rows: list[dict[str, str]],
    columns: list[str] | None = None,
) -> None:
    path = root / model_name / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else ["op", "time_stats.mean"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row.get(column, "") for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_all_required_profiles(
    root: Path,
    model_name: str,
    required_files: tuple[str, ...] = REQUIRED_BASE_PROFILE_FILES,
    rows: list[dict[str, str]] | None = None,
    columns: list[str] | None = None,
) -> None:
    if rows is None:
        rows = [{"op": "attn", "time_stats.mean": "1.25"}]
    for filename in required_files:
        _write_profile(root, model_name, filename, rows, columns)


def test_compare_profiling_csv_roots_passes_for_toy_dense_and_moe_required_set(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    _write_config(config_root, "moe", {"model_type": "qwen3_moe", "num_experts": 8})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(reference, "dense")
    _write_all_required_profiles(candidate, "dense")
    _write_all_required_profiles(
        reference,
        "moe",
        REQUIRED_BASE_PROFILE_FILES + REQUIRED_MOE_PROFILE_FILES,
    )
    _write_all_required_profiles(
        candidate,
        "moe",
        REQUIRED_BASE_PROFILE_FILES + REQUIRED_MOE_PROFILE_FILES,
    )

    report = compare_profiling_csv_roots(
        config_root=config_root,
        reference_profile_root=reference,
        candidate_profile_root=candidate,
        config_filenames=("dense.json", "moe.json"),
    )

    assert report["status"] == "PASS"
    assert report["required_file_count"] == 10
    assert report["file_count"] == 10
    assert report["row_count"] == 10
    assert report["numeric_cells_compared"] == 10
    assert report["mismatch_count"] == 0
    assert report["max_abs_delta"] == 0.0
    assert report["max_relative_delta_pct"] == 0.0


def test_required_relative_paths_for_real_golden_configs_returns_34_files() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    required_paths = _required_relative_paths(
        config_root=repo_root / "data/config/models",
        config_filenames=GOLDEN_CONFIG_FILENAMES,
    )

    assert len(GOLDEN_CONFIG_FILENAMES) == 6
    assert len(required_paths) == 34
    assert len(set(required_paths)) == 34
    assert sum(path.endswith("/moe.csv") for path in required_paths) == 5
    assert sum(path.endswith("/moe_kernel_only.csv") for path in required_paths) == 5


def test_compare_profiling_csv_roots_reports_extra_csvs_without_failing_required_set(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(reference, "dense")
    _write_all_required_profiles(candidate, "dense")
    _write_profile(
        reference,
        "dense",
        "attention_combined.csv",
        [{"op": "combined", "time_stats.mean": "2.0"}],
    )
    _write_profile(candidate, "dense", "moe.csv", [{"op": "moe", "time_stats.mean": "2.0"}])

    report = compare_profiling_csv_roots(
        config_root=config_root,
        reference_profile_root=reference,
        candidate_profile_root=candidate,
        config_filenames=("dense.json",),
    )

    assert report["status"] == "PASS"
    assert report["required_file_count"] == 4
    assert report["reference_extra_csv_count"] == 1
    assert report["candidate_extra_csv_count"] == 1
    assert report["reference_extra_csv_files"] == ["dense/attention_combined.csv"]
    assert report["candidate_extra_csv_files"] == ["dense/moe.csv"]


def test_compare_profiling_csv_roots_fails_fast_for_missing_empty_schema_and_row_count(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(reference, "dense")
    _write_all_required_profiles(candidate, "dense")

    candidate_missing_required = tmp_path / "candidate_missing_required"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        if filename != "linear_op.csv":
            _write_profile(
                candidate_missing_required,
                "dense",
                filename,
                [{"op": "attn", "time_stats.mean": "1.25"}],
            )
    with pytest.raises(FileNotFoundError, match="required profiling CSV missing"):
        compare_profiling_csv_roots(
            config_root=config_root,
            reference_profile_root=reference,
            candidate_profile_root=candidate_missing_required,
            config_filenames=("dense.json",),
        )

    _write_profile(candidate, "dense", "linear_op.csv", [])
    with pytest.raises(ValueError, match="CSV is empty"):
        compare_profiling_csv_roots(
            config_root=config_root,
            reference_profile_root=reference,
            candidate_profile_root=candidate,
            config_filenames=("dense.json",),
        )

    _write_profile(
        candidate,
        "dense",
        "linear_op.csv",
        [{"op": "linear", "time_stats.mean": "1.25", "extra": "x"}],
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        compare_profiling_csv_roots(
            config_root=config_root,
            reference_profile_root=reference,
            candidate_profile_root=candidate,
            config_filenames=("dense.json",),
        )

    _write_profile(
        candidate,
        "dense",
        "linear_op.csv",
        [
            {"op": "linear", "time_stats.mean": "1.25"},
            {"op": "linear", "time_stats.mean": "1.25"},
        ],
    )
    with pytest.raises(ValueError, match="row-count mismatch"):
        compare_profiling_csv_roots(
            config_root=config_root,
            reference_profile_root=reference,
            candidate_profile_root=candidate,
            config_filenames=("dense.json",),
        )


def test_compare_profiling_csv_roots_preserves_duplicate_attention_row_order_and_reports_row_index(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    duplicate_rows = [
        {"op": "attention", "shape": "same", "time_stats.mean": "1.0"},
        {"op": "attention", "shape": "same", "time_stats.mean": "2.0"},
    ]
    _write_all_required_profiles(reference, "dense", rows=duplicate_rows)
    _write_all_required_profiles(candidate, "dense", rows=duplicate_rows)
    _write_profile(candidate, "dense", "attention.csv", list(reversed(duplicate_rows)))

    report = compare_profiling_csv_roots(
        config_root=config_root,
        reference_profile_root=reference,
        candidate_profile_root=candidate,
        config_filenames=("dense.json",),
    )

    assert report["status"] == "FAIL"
    assert report["mismatch_count"] == 2
    assert [mismatch["row_index"] for mismatch in report["mismatches"]] == [0, 1]
    assert {mismatch["kind"] for mismatch in report["mismatches"]} == {"numeric_mismatch"}
    assert report["max_abs_delta"] == 1.0
    assert report["max_relative_delta_pct"] == 100.0



def test_compare_profiling_csv_roots_accepts_matching_sparse_time_stats_blanks(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    rows = [
        {"op": "linear", "time_stats.emb.mean": "1.0", "time_stats.attn.mean": ""},
        {"op": "linear", "time_stats.emb.mean": "", "time_stats.attn.mean": "2.0"},
    ]
    columns = ["op", "time_stats.emb.mean", "time_stats.attn.mean"]
    _write_all_required_profiles(reference, "dense", rows=rows, columns=columns)
    _write_all_required_profiles(candidate, "dense", rows=rows, columns=columns)

    report = compare_profiling_csv_roots(
        config_root=config_root,
        reference_profile_root=reference,
        candidate_profile_root=candidate,
        config_filenames=("dense.json",),
    )

    assert report["status"] == "PASS"
    assert report["numeric_cells_compared"] == 16
    assert report["time_stats_blank_cells_matched"] == 8
    assert report["mismatch_count"] == 0


def test_compare_profiling_csv_roots_reports_sparse_time_stats_blank_mismatch(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(
        reference,
        "dense",
        rows=[{"op": "linear", "time_stats.emb.mean": ""}],
    )
    _write_all_required_profiles(
        candidate,
        "dense",
        rows=[{"op": "linear", "time_stats.emb.mean": "1.0"}],
    )

    report = compare_profiling_csv_roots(
        config_root=config_root,
        reference_profile_root=reference,
        candidate_profile_root=candidate,
        config_filenames=("dense.json",),
    )

    assert report["status"] == "FAIL"
    assert report["mismatch_count"] == 4
    assert report["time_stats_blank_mismatch_count"] == 4
    assert report["mismatches"][0]["kind"] == "time_stats_blank_mismatch"
    assert report["mismatches"][0]["row_index"] == 0

def test_compare_profiling_csv_roots_reports_text_and_numeric_mismatches(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(
        reference,
        "dense",
        rows=[{"op": "linear", "dtype": "bf16", "time_stats.mean": "10.0"}],
    )
    _write_all_required_profiles(
        candidate,
        "dense",
        rows=[{"op": "linear", "dtype": "fp8", "time_stats.mean": "12.5"}],
    )

    report = compare_profiling_csv_roots(
        config_root=config_root,
        reference_profile_root=reference,
        candidate_profile_root=candidate,
        config_filenames=("dense.json",),
    )

    assert report["status"] == "FAIL"
    assert report["mismatch_count"] == 8
    assert report["numeric_mismatch_count"] == 4
    assert report["text_mismatch_count"] == 4
    assert report["max_abs_delta"] == 2.5
    assert report["max_relative_delta_pct"] == 25.0
    first_text = next(m for m in report["mismatches"] if m["kind"] == "text_mismatch")
    assert first_text["column"] == "dtype"
    assert first_text["reference_value"] == "bf16"
    assert first_text["candidate_value"] == "fp8"


def test_compare_profiling_csv_roots_allows_explicit_candidate_metadata_column_migration(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(reference, "dense")
    _write_all_required_profiles(
        candidate,
        "dense",
        rows=[{"op": "attn", "time_stats.mean": "1.25", "model_architecture_profile": "generic"}],
        columns=["op", "model_architecture_profile", "time_stats.mean"],
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        compare_profiling_csv_roots(
            config_root=config_root,
            reference_profile_root=reference,
            candidate_profile_root=candidate,
            config_filenames=("dense.json",),
        )

    report = compare_profiling_csv_roots(
        config_root=config_root,
        reference_profile_root=reference,
        candidate_profile_root=candidate,
        config_filenames=("dense.json",),
        ignore_candidate_extra_metadata_columns=("model_architecture_profile",),
    )

    assert report["status"] == "PASS"
    assert report["ignored_candidate_extra_metadata_columns"] == ["model_architecture_profile"]
    assert report["ignored_candidate_extra_metadata_cell_count"] == 4


def test_compare_profiling_csv_roots_validates_ignored_candidate_metadata_values(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(reference, "dense")
    _write_all_required_profiles(
        candidate,
        "dense",
        rows=[
            {
                "op": "attn",
                "time_stats.mean": "1.25",
                "model_architecture_profile": "not_generic",
            }
        ],
        columns=["op", "model_architecture_profile", "time_stats.mean"],
    )

    with pytest.raises(ValueError, match="ignored candidate metadata mismatch"):
        compare_profiling_csv_roots(
            config_root=config_root,
            reference_profile_root=reference,
            candidate_profile_root=candidate,
            config_filenames=("dense.json",),
            ignore_candidate_extra_metadata_columns=("model_architecture_profile",),
        )


def test_cli_writes_json_report_and_returns_mismatch_exit_code(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    _write_config(config_root, "dense", {"model_type": "llama"})
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_all_required_profiles(reference, "dense")
    _write_all_required_profiles(candidate, "dense", rows=[{"op": "attn", "time_stats.mean": "1.5"}])
    output_json = tmp_path / "report.json"

    exit_code = main(
        [
            "--config-root",
            str(config_root),
            "--reference-profile-root",
            str(reference),
            "--candidate-profile-root",
            str(candidate),
            "--output-json",
            str(output_json),
            "--config-filename",
            "dense.json",
        ]
    )

    assert exit_code == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["mismatch_count"] == 4
    assert payload["max_abs_delta"] == 0.25
