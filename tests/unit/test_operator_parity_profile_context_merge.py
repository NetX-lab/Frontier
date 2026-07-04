import csv
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.operator_parity.merge_profile_csv_contexts import merge_profile_csvs


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


def test_merge_profile_csvs_keeps_prefill_hot_and_standalone_legacy_rows(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "moe.csv"
    supplement = tmp_path / "stage" / "moe.csv"
    output = tmp_path / "merged" / "moe.csv"
    common = {
        "num_tensor_parallel_workers": "1",
        "expert_parallel_size": "1",
        "num_tokens": "1",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical,
        [
            {
                **common,
                "gating_runtime_context": "prefill_hot",
                "time_stats.moe_gating_linear.median": "3.0",
            }
        ],
    )
    _write_csv(
        supplement,
        [
            {
                **common,
                "gating_runtime_context": "standalone_legacy",
                "time_stats.moe_gating_linear.median": "2.0",
            }
        ],
    )

    report = merge_profile_csvs(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
    )

    rows = _read_rows(output)
    assert report["base_row_count"] == 1
    assert report["supplement_row_count"] == 1
    assert report["merged_row_count"] == 2
    assert [row["gating_runtime_context"] for row in rows] == [
        "prefill_hot",
        "standalone_legacy",
    ]


def test_merge_profile_csvs_fails_on_conflicting_duplicate_profile_key(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "moe.csv"
    supplement = tmp_path / "stage" / "moe.csv"
    output = tmp_path / "merged" / "moe.csv"
    row_key = {
        "num_tensor_parallel_workers": "1",
        "expert_parallel_size": "1",
        "num_tokens": "1",
        "measurement_type": "CUDA_EVENT",
        "gating_runtime_context": "standalone_legacy",
    }
    _write_csv(
        canonical,
        [{**row_key, "time_stats.moe_gating_linear.median": "2.0"}],
    )
    _write_csv(
        supplement,
        [{**row_key, "time_stats.moe_gating_linear.median": "2.5"}],
    )

    with pytest.raises(ValueError, match="Conflicting duplicate profiling row"):
        merge_profile_csvs(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
        )



def test_merge_profile_csvs_preserves_existing_canonical_repeated_measurements(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "attention.csv"
    supplement = tmp_path / "stage" / "attention.csv"
    output = tmp_path / "merged" / "attention.csv"
    base_key = {
        "num_tensor_parallel_workers": "1",
        "batch_size": "1",
        "is_prefill": "True",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical,
        [
            {**base_key, "time_stats.attn_decode.median": "1.0"},
            {**base_key, "time_stats.attn_decode.median": "1.1"},
        ],
    )
    _write_csv(
        supplement,
        [
            {
                "num_tensor_parallel_workers": "1",
                "is_true_mixed_batch": "True",
                "decode_batch_size": "1",
                "decode_avg_kv_cache_size": "16",
                "num_prefill_seqs": "1",
                "total_prefill_tokens": "16",
                "total_batch_size": "2",
                "batch_composition_ratio": "0.5",
                "total_tokens": "17",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_decode.median": "3.0",
            }
        ],
    )

    report = merge_profile_csvs(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
    )

    rows = _read_rows(output)
    assert report["base_row_count"] == 2
    assert report["supplement_row_count"] == 1
    assert report["merged_row_count"] == 3
    assert [row["time_stats.attn_decode.median"] for row in rows].count("1.0") == 1
    assert [row["time_stats.attn_decode.median"] for row in rows].count("1.1") == 1
    assert [row["time_stats.attn_decode.median"] for row in rows].count("3.0") == 1


def test_merge_profile_csvs_skips_fully_identical_supplement_rows(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "attention.csv"
    supplement = tmp_path / "stage" / "attention.csv"
    output = tmp_path / "merged" / "attention.csv"
    canonical_row = {
        "num_tensor_parallel_workers": "1",
        "is_true_mixed_batch": "False",
        "measurement_type": "CUDA_EVENT",
        "time_stats.attn_decode.median": "1.0",
    }
    supplement_row = {
        "num_tensor_parallel_workers": "1",
        "is_true_mixed_batch": "True",
        "measurement_type": "CUDA_EVENT",
        "time_stats.attn_decode.median": "3.0",
    }
    _write_csv(canonical, [canonical_row])
    _write_csv(supplement, [supplement_row, supplement_row])

    report = merge_profile_csvs(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
    )

    rows = _read_rows(output)
    assert report["base_row_count"] == 1
    assert report["supplement_row_count"] == 2
    assert report["merged_row_count"] == 2
    assert report["duplicate_identical_count"] == 1
    assert rows == [canonical_row, supplement_row]


def test_merge_profile_csvs_supports_attention_true_mixed_supplement_filename(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "attention.csv"
    supplement = tmp_path / "stage" / "attention_true_mixed.csv"
    output = tmp_path / "merged" / "attention.csv"
    _write_csv(
        canonical,
        [
            {
                "num_tensor_parallel_workers": "1",
                "is_true_mixed_batch": "False",
                "batch_size": "1",
                "time_stats.attn_decode.median": "1.0",
            }
        ],
    )
    _write_csv(
        supplement,
        [
            {
                "num_tensor_parallel_workers": "1",
                "is_true_mixed_batch": "True",
                "decode_batch_size": "1",
                "decode_avg_kv_cache_size": "16",
                "num_prefill_seqs": "1",
                "total_prefill_tokens": "16",
                "total_batch_size": "2",
                "batch_composition_ratio": "0.5",
                "total_tokens": "17",
                "time_stats.attn_decode.median": "3.0",
            }
        ],
    )

    report = merge_profile_csvs(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
    )

    rows = _read_rows(output)
    assert report["base_row_count"] == 1
    assert report["supplement_row_count"] == 1
    assert report["merged_row_count"] == 2
    assert [row["is_true_mixed_batch"] for row in rows] == ["False", "True"]
    assert rows[1]["decode_batch_size"] == "1"


def test_merge_cli_refuses_in_place_without_explicit_allow_flag(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(supplement_root / "model_a" / "attention.csv", [{"k": "supplement"}])

    result = subprocess.run(
        [
            sys.executable,
            "tests/e2e/operator_parity/merge_profile_csv_contexts.py",
            "--canonical-root",
            str(canonical_root),
            "--supplement-root",
            str(supplement_root),
            "--models",
            "model_a",
            "--filenames",
            "attention.csv",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Refusing in-place merge" in result.stderr
    assert _read_rows(canonical_root / "model_a" / "attention.csv") == [{"k": "base"}]


def test_merge_cli_writes_output_root_without_mutating_canonical_input(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(supplement_root / "model_a" / "attention.csv", [{"k": "supplement"}])

    result = subprocess.run(
        [
            sys.executable,
            "tests/e2e/operator_parity/merge_profile_csv_contexts.py",
            "--canonical-root",
            str(canonical_root),
            "--supplement-root",
            str(supplement_root),
            "--output-root",
            str(output_root),
            "--models",
            "model_a",
            "--filenames",
            "attention.csv",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _read_rows(canonical_root / "model_a" / "attention.csv") == [{"k": "base"}]
    assert _read_rows(output_root / "model_a" / "attention.csv") == [
        {"k": "base"},
        {"k": "supplement"},
    ]


def test_merge_cli_allows_explicit_in_place_write(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(supplement_root / "model_a" / "attention.csv", [{"k": "supplement"}])

    result = subprocess.run(
        [
            sys.executable,
            "tests/e2e/operator_parity/merge_profile_csv_contexts.py",
            "--canonical-root",
            str(canonical_root),
            "--supplement-root",
            str(supplement_root),
            "--allow-in-place",
            "--models",
            "model_a",
            "--filenames",
            "attention.csv",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _read_rows(canonical_root / "model_a" / "attention.csv") == [
        {"k": "base"},
        {"k": "supplement"},
    ]
