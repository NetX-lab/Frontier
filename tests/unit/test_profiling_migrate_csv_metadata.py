from pathlib import Path

import pandas as pd
import pytest

from frontier.profiling.migrate_csv_metadata import migrate_csv_metadata


def _write_legacy_profile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "num_tensor_parallel_workers,time_stats.mean\n"
        "1,2.5\n",
        encoding="utf-8",
    )


def test_migrate_csv_metadata_writes_model_architecture_profile(tmp_path: Path) -> None:
    csv_path = tmp_path / "linear_op.csv"
    _write_legacy_profile(csv_path)

    migrated = migrate_csv_metadata(
        input_csv=str(csv_path),
        output_csv=None,
        profiling_precision="FP16",
        model_arch="generic",
        model_architecture_profile="step3_text",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
    )

    assert migrated["model_architecture_profile"].unique().tolist() == ["step3_text"]
    persisted = pd.read_csv(csv_path)
    assert persisted["model_architecture_profile"].unique().tolist() == ["step3_text"]
    assert persisted["time_stats.mean"].tolist() == [2.5]


def test_migrate_csv_metadata_rejects_architecture_profile_conflict(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "linear_op.csv"
    csv_path.write_text(
        "num_tensor_parallel_workers,model_architecture_profile,time_stats.mean\n"
        "1,generic,2.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model_architecture_profile"):
        migrate_csv_metadata(
            input_csv=str(csv_path),
            output_csv=None,
            profiling_precision="FP16",
            model_arch="generic",
            model_architecture_profile="step3_text",
            quant_signature="none",
            measurement_type="CUDA_EVENT",
        )



def _read_csv_text_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_migrate_csv_metadata_preserves_high_precision_timing_strings(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "linear_op_kernel_only.csv"
    high_precision = "1.9919839356781918e-05"
    csv_path.write_text(
        "num_tensor_parallel_workers,time_stats.attn_rope.std,time_stats.empty\n"
        f"1,{high_precision},\n",
        encoding="utf-8",
    )

    migrate_csv_metadata(
        input_csv=str(csv_path),
        output_csv=None,
        profiling_precision="FP16",
        model_arch="generic",
        model_architecture_profile="generic",
        quant_signature="none",
        measurement_type="KERNEL_ONLY",
    )

    [row] = _read_csv_text_rows(csv_path)
    assert row["time_stats.attn_rope.std"] == high_precision
    assert row["time_stats.empty"] == ""
    assert row["model_architecture_profile"] == "generic"


def test_migrate_csv_metadata_is_idempotent_for_half_migrated_sparse_csv(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "attention.csv"
    csv_path.write_text(
        "batch_size,model_architecture_profile,time_stats.attn_prefill.mean,"
        "time_stats.attn_decode.mean\n"
        "1,step3_text,3.2500000000000004,\n"
        "2,step3_text,,4.5000000000000001\n",
        encoding="utf-8",
    )
    before_rows = _read_csv_text_rows(csv_path)

    migrate_csv_metadata(
        input_csv=str(csv_path),
        output_csv=None,
        profiling_precision="FP16",
        model_arch="generic",
        model_architecture_profile="step3_text",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
    )
    first_rows = _read_csv_text_rows(csv_path)
    migrate_csv_metadata(
        input_csv=str(csv_path),
        output_csv=None,
        profiling_precision="FP16",
        model_arch="generic",
        model_architecture_profile="step3_text",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
    )
    second_rows = _read_csv_text_rows(csv_path)

    for rows in (first_rows, second_rows):
        assert [row["time_stats.attn_prefill.mean"] for row in rows] == [
            row["time_stats.attn_prefill.mean"] for row in before_rows
        ]
        assert [row["time_stats.attn_decode.mean"] for row in rows] == [
            row["time_stats.attn_decode.mean"] for row in before_rows
        ]
    assert first_rows == second_rows


def test_migrate_csv_metadata_directory_handles_mixed_idempotent_manifest(
    tmp_path: Path,
) -> None:
    from frontier.profiling.migrate_csv_metadata import migrate_csv_metadata_directory

    profile_dir = tmp_path / "profiles"
    missing_profile = profile_dir / "attention.csv"
    existing_profile = profile_dir / "linear_op.csv"
    missing_profile.parent.mkdir(parents=True, exist_ok=True)
    missing_profile.write_text(
        "num_tensor_parallel_workers,time_stats.mean\n"
        "1,1.0000000000000002\n",
        encoding="utf-8",
    )
    existing_profile.write_text(
        "num_tensor_parallel_workers,profiling_precision,model_arch,"
        "model_architecture_profile,quant_signature,measurement_type,time_stats.mean\n"
        "1,FP16,generic,generic,none,CUDA_EVENT,2.0000000000000004\n",
        encoding="utf-8",
    )

    records = migrate_csv_metadata_directory(
        input_dir=str(profile_dir),
        output_dir=None,
        profiling_precision="FP16",
        model_arch="generic",
        model_architecture_profile="generic",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
    )

    records_by_name = {Path(str(record["input_csv"])).name: record for record in records}
    assert records_by_name["attention.csv"]["row_count"] == 1
    assert "model_architecture_profile" in records_by_name["attention.csv"]["added_columns"]
    assert records_by_name["linear_op.csv"]["added_columns"] == ""
    assert _read_csv_text_rows(missing_profile)[0]["time_stats.mean"] == "1.0000000000000002"
    assert _read_csv_text_rows(existing_profile)[0]["time_stats.mean"] == "2.0000000000000004"


def test_migrate_csv_metadata_cli_requires_model_architecture_profile(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    csv_path = tmp_path / "linear_op.csv"
    _write_legacy_profile(csv_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "frontier.profiling.migrate_csv_metadata",
            "--input_csv",
            str(csv_path),
            "--profiling_precision",
            "FP16",
            "--model_arch",
            "generic",
            "--quant_signature",
            "none",
            "--measurement_type",
            "CUDA_EVENT",
        ],
        cwd=Path.cwd(),
        env={"PYTHONPATH": str(Path.cwd()), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "model_architecture_profile" in result.stderr
