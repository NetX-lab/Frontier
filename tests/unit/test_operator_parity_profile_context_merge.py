import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.operator_parity.merge_profile_csv_contexts import (
    enrich_profile_csv_columns,
    merge_profile_csvs,
)


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


def test_merge_profile_csvs_normalizes_legacy_gating_context_rows(
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

    with pytest.warns(FutureWarning):
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
        "direct",
        "prefill_warmed",
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
        "gating_runtime_context": "direct",
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


def test_merge_profile_csvs_writes_lf_line_endings(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical" / "attention.csv"
    supplement = tmp_path / "stage" / "attention.csv"
    output = tmp_path / "merged" / "attention.csv"
    _write_csv(canonical, [{"key": "base"}])
    _write_csv(supplement, [{"key": "supplement"}])

    merge_profile_csvs(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
    )

    output_bytes = output.read_bytes()
    assert b"\r\n" not in output_bytes
    assert output_bytes.count(b"\n") == 3


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


def test_enrich_profile_csv_columns_preserves_canonical_timings(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    _write_csv(
        canonical,
        [
            {
                "num_tensor_parallel_workers": "4",
                "num_tokens": "1",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_pre_proj.median": "1.0",
            },
            {
                "num_tensor_parallel_workers": "4",
                "num_tokens": "2",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_pre_proj.median": "2.0",
            },
        ],
    )
    _write_csv(
        supplement,
        [
            {
                "num_tensor_parallel_workers": "4",
                "num_tokens": "1",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_pre_proj.median": "9.0",
                "time_stats.mtp_fusion_proj.median": "3.0",
                "time_stats.lm_head_linear.median": "4.0",
            }
        ],
    )

    report = enrich_profile_csv_columns(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
        target_columns=[
            "time_stats.mtp_fusion_proj.median",
            "time_stats.lm_head_linear.median",
        ],
    )

    rows = _read_rows(output)
    assert report["enriched_row_count"] == 1
    assert rows[0]["time_stats.attn_pre_proj.median"] == "1.0"
    assert rows[0]["time_stats.mtp_fusion_proj.median"] == "3.0"
    assert rows[0]["time_stats.lm_head_linear.median"] == "4.0"
    assert rows[1]["time_stats.attn_pre_proj.median"] == "2.0"
    assert rows[1]["time_stats.mtp_fusion_proj.median"] == ""
    assert rows[1]["time_stats.lm_head_linear.median"] == ""


def test_enrich_profile_csv_columns_writes_lf_line_endings(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    key = {
        "num_tokens": "1",
        "num_tensor_parallel_workers": "4",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical,
        [{**key, "time_stats.attn_pre_proj.median": "1.0"}],
    )
    _write_csv(
        supplement,
        [{**key, "time_stats.mtp_fusion_proj.median": "3.0"}],
    )

    enrich_profile_csv_columns(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
        target_columns=["time_stats.mtp_fusion_proj.median"],
    )

    output_bytes = output.read_bytes()
    assert b"\r\n" not in output_bytes
    assert output_bytes.count(b"\n") == 2


def test_enrich_profile_csv_columns_accepts_reordered_key_columns(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    key = {
        "n_head": "16",
        "n_kv_head": "2",
        "n_embd": "2048",
        "num_tokens": "1",
        "num_tensor_parallel_workers": "4",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical,
        [{**key, "time_stats.attn_pre_proj.median": "1.0"}],
    )
    supplement.parent.mkdir(parents=True, exist_ok=True)
    supplement_fields = [
        "n_head",
        "n_kv_head",
        "n_embd",
        "num_tokens",
        "num_tensor_parallel_workers",
        "measurement_type",
        "time_stats.mtp_fusion_proj.median",
    ]
    with supplement.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=supplement_fields)
        writer.writeheader()
        writer.writerow(
            {
                **key,
                "time_stats.mtp_fusion_proj.median": "3.0",
            }
        )

    report = enrich_profile_csv_columns(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
        target_columns=["time_stats.mtp_fusion_proj.median"],
    )

    assert report["populated_cell_count"] == 1
    assert _read_rows(output) == [
        {
            **key,
            "time_stats.attn_pre_proj.median": "1.0",
            "time_stats.mtp_fusion_proj.median": "3.0",
        }
    ]


def test_enrich_profile_csv_columns_drops_explicit_constant_legacy_key(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    canonical_row = {
        "num_tokens": "1",
        "num_tensor_parallel_workers": "4",
        "is_step2_mini": "False",
        "measurement_type": "CUDA_EVENT",
        "time_stats.attn_pre_proj.median": "1.0",
    }
    supplement_row = {
        "num_tokens": "1",
        "num_tensor_parallel_workers": "4",
        "measurement_type": "CUDA_EVENT",
        "time_stats.mtp_fusion_proj.median": "3.0",
    }
    _write_csv(canonical, [canonical_row])
    _write_csv(supplement, [supplement_row])

    report = enrich_profile_csv_columns(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
        target_columns=["time_stats.mtp_fusion_proj.median"],
        drop_canonical_key_values={"is_step2_mini": "False"},
    )

    rows = _read_rows(output)
    assert report["dropped_canonical_key_values"] == {"is_step2_mini": "False"}
    assert rows == [
        {
            "num_tokens": "1",
            "num_tensor_parallel_workers": "4",
            "measurement_type": "CUDA_EVENT",
            "time_stats.attn_pre_proj.median": "1.0",
            "time_stats.mtp_fusion_proj.median": "3.0",
        }
    ]


def test_enrich_profile_csv_columns_rejects_unexpected_legacy_key_value(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    _write_csv(
        canonical,
        [
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "4",
                "is_step2_mini": "True",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_pre_proj.median": "1.0",
            }
        ],
    )
    _write_csv(
        supplement,
        [
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "4",
                "measurement_type": "CUDA_EVENT",
                "time_stats.mtp_fusion_proj.median": "3.0",
            }
        ],
    )

    with pytest.raises(ValueError, match="expected canonical key value"):
        enrich_profile_csv_columns(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
            target_columns=["time_stats.mtp_fusion_proj.median"],
            drop_canonical_key_values={"is_step2_mini": "False"},
        )


def test_enrich_profile_csv_columns_filters_explicit_supplement_key_values(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    _write_csv(
        canonical,
        [
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "4",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_pre_proj.median": "1.0",
            }
        ],
    )
    _write_csv(
        supplement,
        [
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "1",
                "measurement_type": "CUDA_EVENT",
                "time_stats.mtp_fusion_proj.median": "",
            },
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "4",
                "measurement_type": "CUDA_EVENT",
                "time_stats.mtp_fusion_proj.median": "3.0",
            },
        ],
    )

    report = enrich_profile_csv_columns(
        canonical_csv=canonical,
        supplement_csv=supplement,
        output_csv=output,
        target_columns=["time_stats.mtp_fusion_proj.median"],
        supplement_key_values={"num_tensor_parallel_workers": "4"},
    )

    assert report["supplement_row_count"] == 2
    assert report["selected_supplement_row_count"] == 1
    assert report["excluded_supplement_row_count"] == 1
    assert report["supplement_key_values"] == {
        "num_tensor_parallel_workers": "4"
    }
    assert _read_rows(output)[0]["time_stats.mtp_fusion_proj.median"] == "3.0"


@pytest.mark.parametrize(
    ("supplement_key_values", "error"),
    [
        ({"unknown_key": "4"}, "missing supplement filter key"),
        (
            {"time_stats.mtp_fusion_proj.median": "3.0"},
            "Cannot filter supplement by timing column",
        ),
    ],
)
def test_enrich_profile_csv_columns_rejects_invalid_supplement_filter_keys(
    tmp_path: Path,
    supplement_key_values: dict[str, str],
    error: str,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    key = {
        "num_tokens": "1",
        "num_tensor_parallel_workers": "4",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical,
        [{**key, "time_stats.attn_pre_proj.median": "1.0"}],
    )
    _write_csv(
        supplement,
        [{**key, "time_stats.mtp_fusion_proj.median": "3.0"}],
    )

    with pytest.raises(ValueError, match=error):
        enrich_profile_csv_columns(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
            target_columns=["time_stats.mtp_fusion_proj.median"],
            supplement_key_values=supplement_key_values,
        )


def test_enrich_profile_csv_columns_rejects_empty_supplement_filter_result(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    key = {
        "num_tokens": "1",
        "num_tensor_parallel_workers": "4",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical,
        [{**key, "time_stats.attn_pre_proj.median": "1.0"}],
    )
    _write_csv(
        supplement,
        [{**key, "time_stats.mtp_fusion_proj.median": "3.0"}],
    )

    with pytest.raises(ValueError, match="matched no supplement rows"):
        enrich_profile_csv_columns(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
            target_columns=["time_stats.mtp_fusion_proj.median"],
            supplement_key_values={"num_tensor_parallel_workers": "8"},
        )


def test_enrich_profile_csv_columns_rejects_unknown_supplement_key(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    _write_csv(
        canonical,
        [
            {
                "num_tensor_parallel_workers": "4",
                "num_tokens": "1",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_pre_proj.median": "1.0",
            }
        ],
    )
    _write_csv(
        supplement,
        [
            {
                "num_tensor_parallel_workers": "4",
                "num_tokens": "2",
                "measurement_type": "CUDA_EVENT",
                "time_stats.mtp_fusion_proj.median": "3.0",
            }
        ],
    )

    with pytest.raises(ValueError, match="has no canonical row"):
        enrich_profile_csv_columns(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
            target_columns=["time_stats.mtp_fusion_proj.median"],
        )


@pytest.mark.parametrize("duplicate_side", ["canonical", "supplement"])
def test_enrich_profile_csv_columns_rejects_duplicate_keys(
    tmp_path: Path,
    duplicate_side: str,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    canonical_row = {
        "num_tensor_parallel_workers": "4",
        "num_tokens": "1",
        "measurement_type": "CUDA_EVENT",
        "time_stats.attn_pre_proj.median": "1.0",
    }
    supplement_row = {
        "num_tensor_parallel_workers": "4",
        "num_tokens": "1",
        "measurement_type": "CUDA_EVENT",
        "time_stats.mtp_fusion_proj.median": "3.0",
    }
    _write_csv(
        canonical,
        [canonical_row, canonical_row] if duplicate_side == "canonical" else [canonical_row],
    )
    _write_csv(
        supplement,
        [supplement_row, supplement_row]
        if duplicate_side == "supplement"
        else [supplement_row],
    )

    with pytest.raises(ValueError, match=f"Duplicate {duplicate_side} profiling key"):
        enrich_profile_csv_columns(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
            target_columns=["time_stats.mtp_fusion_proj.median"],
        )


def test_enrich_profile_csv_columns_rejects_conflicting_existing_target(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    key = {
        "num_tensor_parallel_workers": "4",
        "num_tokens": "1",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical,
        [{**key, "time_stats.mtp_fusion_proj.median": "3.0"}],
    )
    _write_csv(
        supplement,
        [{**key, "time_stats.mtp_fusion_proj.median": "3.5"}],
    )

    with pytest.raises(ValueError, match="Conflicting profiling column"):
        enrich_profile_csv_columns(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
            target_columns=["time_stats.mtp_fusion_proj.median"],
        )


@pytest.mark.parametrize("invalid_value", ["", "nan", "inf", "not-a-number"])
def test_enrich_profile_csv_columns_rejects_invalid_target_values(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    canonical = tmp_path / "canonical" / "linear_op.csv"
    supplement = tmp_path / "stage" / "linear_op.csv"
    output = tmp_path / "merged" / "linear_op.csv"
    key = {
        "num_tensor_parallel_workers": "4",
        "num_tokens": "1",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(canonical, [{**key, "time_stats.attn_pre_proj.median": "1.0"}])
    _write_csv(
        supplement,
        [{**key, "time_stats.mtp_fusion_proj.median": invalid_value}],
    )

    with pytest.raises(ValueError, match="profiling column"):
        enrich_profile_csv_columns(
            canonical_csv=canonical,
            supplement_csv=supplement,
            output_csv=output,
            target_columns=["time_stats.mtp_fusion_proj.median"],
        )


def test_enrich_profile_cli_uses_explicit_target_columns_without_mutating_canonical(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    key = {
        "num_tensor_parallel_workers": "4",
        "num_tokens": "1",
        "measurement_type": "CUDA_EVENT",
    }
    _write_csv(
        canonical_root / "model_a" / "linear_op.csv",
        [{**key, "time_stats.attn_pre_proj.median": "1.0"}],
    )
    _write_csv(
        supplement_root / "model_a" / "linear_op.csv",
        [
            {
                **key,
                "time_stats.attn_pre_proj.median": "8.0",
                "time_stats.mtp_fusion_proj.median": "3.0",
            }
        ],
    )

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
            "linear_op.csv",
            "--enrich-columns",
            "time_stats.mtp_fusion_proj.median",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _read_rows(canonical_root / "model_a" / "linear_op.csv") == [
        {**key, "time_stats.attn_pre_proj.median": "1.0"}
    ]
    assert _read_rows(output_root / "model_a" / "linear_op.csv") == [
        {
            **key,
            "time_stats.attn_pre_proj.median": "1.0",
            "time_stats.mtp_fusion_proj.median": "3.0",
        }
    ]


def test_enrich_profile_cli_drops_legacy_key_and_filters_supplement_rows(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    _write_csv(
        canonical_root / "model_a" / "linear_op.csv",
        [
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "4",
                "is_step2_mini": "False",
                "measurement_type": "CUDA_EVENT",
                "time_stats.attn_pre_proj.median": "1.0",
            }
        ],
    )
    _write_csv(
        supplement_root / "model_a" / "linear_op.csv",
        [
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "1",
                "measurement_type": "CUDA_EVENT",
                "time_stats.mtp_fusion_proj.median": "",
            },
            {
                "num_tokens": "1",
                "num_tensor_parallel_workers": "4",
                "measurement_type": "CUDA_EVENT",
                "time_stats.mtp_fusion_proj.median": "3.0",
            }
        ],
    )

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
            "linear_op.csv",
            "--enrich-columns",
            "time_stats.mtp_fusion_proj.median",
            "--drop-canonical-key",
            "is_step2_mini=False",
            "--supplement-key",
            "num_tensor_parallel_workers=4",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)["merged_files"][0]
    assert report["selected_supplement_row_count"] == 1
    assert report["excluded_supplement_row_count"] == 1
    assert _read_rows(output_root / "model_a" / "linear_op.csv") == [
        {
            "num_tokens": "1",
            "num_tensor_parallel_workers": "4",
            "measurement_type": "CUDA_EVENT",
            "time_stats.attn_pre_proj.median": "1.0",
            "time_stats.mtp_fusion_proj.median": "3.0",
        }
    ]
