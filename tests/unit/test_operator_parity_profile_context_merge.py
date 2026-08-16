import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from frontier.profiling.attention import provenance as attention_provenance
from frontier.profiling.attention.provenance import (
    publish_attention_union_and_alias,
    validate_attention_merge_sidecar,
    write_attention_partition_run_sidecar,
)
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
    _write_csv(canonical_root / "model_a" / "moe.csv", [{"k": "base"}])
    _write_csv(supplement_root / "model_a" / "moe.csv", [{"k": "supplement"}])

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
            "moe.csv",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _read_rows(canonical_root / "model_a" / "moe.csv") == [{"k": "base"}]
    assert _read_rows(output_root / "model_a" / "moe.csv") == [
        {"k": "base"},
        {"k": "supplement"},
    ]


def test_merge_cli_allows_explicit_in_place_write(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "moe.csv", [{"k": "base"}])
    _write_csv(supplement_root / "model_a" / "moe.csv", [{"k": "supplement"}])

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
            "moe.csv",
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _read_rows(canonical_root / "model_a" / "moe.csv") == [
        {"k": "base"},
        {"k": "supplement"},
    ]


def _publish_attention_source(
    root: Path,
    *,
    run_id: str,
    total_tokens: int,
    provenance_model: str = "model_a",
    measurement_type: str = "CUDA_EVENT",
) -> dict[str, Path]:
    frame = pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 1,
                "prefill_chunk_size": total_tokens,
                "kv_cache_size": 0,
                "batch_size": 1,
                "is_prefill": True,
                "measurement_type": measurement_type,
                "profiling_precision": "BF16",
                "quant_signature": "none",
                "model_architecture_profile": "generic",
                "attention_backend": "FLASHINFER",
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": 18,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
                "allocated_max_num_blocks": 18,
                "allocated_kv_token_capacity": 288,
                "block_size": 16,
                "time_stats.attn_kv_cache_save.median": total_tokens / 1000,
            }
        ]
    )
    return publish_attention_union_and_alias(
        output_dir=root / "model_a",
        standard_df=frame,
        mixed_df=pd.DataFrame(),
        true_mixed_df=pd.DataFrame(),
        run_id=run_id,
        provenance={
            "model": provenance_model,
            "device": "h800",
            "tensor_parallel_sizes": [1],
            "measurement_type": measurement_type,
            "profiling_precision": "BF16",
            "quant_signature": "none",
            "model_architecture_profile": "generic",
            "attention_backend": "FLASHINFER",
            "allocation_by_tp_semantics": "per_tp_column_max_v1",
            "allocation_by_tp": {
                "1": {
                    "physical_max_num_blocks": 100,
                    "requested_max_num_blocks": 18,
                    "selected_max_num_blocks": 18,
                    "required_max_num_blocks": 18,
                    "allocated_max_num_blocks": 18,
                    "allocated_kv_token_capacity": 288,
                    "block_size": 16,
                }
            },
        },
    )


def test_merge_cli_requires_sidecars_for_attention_publication(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    _publish_attention_source(canonical_root, run_id="base", total_tokens=8)
    _publish_attention_source(supplement_root, run_id="supplement", total_tokens=16)

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

    assert result.returncode != 0
    assert "requires --canonical-sidecar and --supplement-sidecar" in result.stderr
    assert not (output_root / "model_a" / "attention.csv").exists()
    assert not (output_root / "model_a" / "attention_combined.csv").exists()


def test_merge_cli_publishes_attention_canonical_sidecar_and_alias(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    base = _publish_attention_source(
        canonical_root,
        run_id="base",
        total_tokens=8,
    )
    supplement = _publish_attention_source(
        supplement_root,
        run_id="supplement",
        total_tokens=16,
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
            "attention.csv",
            "--canonical-sidecar",
            str(base["sidecar"]),
            "--supplement-sidecar",
            str(supplement["sidecar"]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    canonical = output_root / "model_a" / "attention.csv"
    alias = output_root / "model_a" / "attention_combined.csv"
    sidecar = output_root / "model_a" / "attention.merge_provenance.json"
    assert canonical.read_bytes() == alias.read_bytes()
    validate_attention_merge_sidecar(sidecar_path=sidecar)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("attention_combined.csv", "output-only"),
        ("attention_combined_kernel_only.csv", "output-only"),
        ("attention_mixed.csv", "partition"),
        ("attention_mixed_kernel_only.csv", "partition"),
        ("attention_true_mixed.csv", "partition"),
        ("attention_true_mixed_kernel_only.csv", "partition"),
        ("./attention.csv", "plain CSV basename"),
    ],
)
def test_merge_cli_rejects_attention_filename_bypasses_before_writing(
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    source_name = Path(filename).name
    _write_csv(canonical_root / "model_a" / source_name, [{"k": "base"}])
    _write_csv(
        supplement_root / "model_a" / source_name,
        [{"k": "supplement"}],
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
            filename,
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not (output_root / "model_a" / source_name).exists()


@pytest.mark.parametrize(
    "collision_kind",
    [
        "source_csv",
        "source_sidecar",
        "bound_source_artifact",
        "output_canonical",
        "output_canonical_descendant",
        "output_alias",
        "output_sidecar",
        "output_root",
        "output_model_descendant",
    ],
)
def test_merge_cli_rejects_json_out_artifact_collisions_before_writing(
    tmp_path: Path,
    collision_kind: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    base = _publish_attention_source(
        canonical_root,
        run_id="base",
        total_tokens=8,
    )
    supplement = _publish_attention_source(
        supplement_root,
        run_id="supplement",
        total_tokens=16,
    )
    output_canonical = output_root / "model_a" / "attention.csv"
    output_alias = output_root / "model_a" / "attention_combined.csv"
    output_sidecar = output_root / "model_a" / "attention.merge_provenance.json"
    base_payload = json.loads(base["sidecar"].read_text(encoding="utf-8"))
    bound_source_artifact = Path(base_payload["artifact_csv"])
    collisions = {
        "source_csv": base["canonical"],
        "source_sidecar": base["sidecar"],
        "bound_source_artifact": bound_source_artifact,
        "output_canonical": output_canonical,
        "output_canonical_descendant": output_canonical / "report.json",
        "output_alias": output_alias,
        "output_sidecar": output_sidecar,
        "output_root": output_root,
        "output_model_descendant": output_root / "model_a" / "report.json",
    }
    source_snapshots = {
        base["canonical"]: base["canonical"].read_bytes(),
        base["sidecar"]: base["sidecar"].read_bytes(),
        bound_source_artifact: bound_source_artifact.read_bytes(),
        supplement["canonical"]: supplement["canonical"].read_bytes(),
        supplement["sidecar"]: supplement["sidecar"].read_bytes(),
    }

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
            "--canonical-sidecar",
            str(base["sidecar"]),
            "--supplement-sidecar",
            str(supplement["sidecar"]),
            "--json-out",
            str(collisions[collision_kind]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--json-out" in result.stderr
    for path, expected_bytes in source_snapshots.items():
        assert path.read_bytes() == expected_bytes
    assert not output_canonical.exists()
    assert not output_alias.exists()
    assert not output_sidecar.exists()


def test_merge_cli_rejects_output_bound_artifact_collision_before_writing(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    base = _publish_attention_source(
        canonical_root,
        run_id="base",
        total_tokens=8,
    )
    supplement = _publish_attention_source(
        supplement_root,
        run_id="supplement",
        total_tokens=16,
    )
    output_canonical = output_root / "model_a" / "attention.csv"
    base_payload = json.loads(base["sidecar"].read_text(encoding="utf-8"))
    base_payload["artifact_csv"] = str(output_canonical)
    base_payload["config_sha256"] = attention_provenance._config_digest(
        base_payload
    )
    base["sidecar"].write_text(
        json.dumps(base_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
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
            "attention.csv",
            "--canonical-sidecar",
            str(base["sidecar"]),
            "--supplement-sidecar",
            str(supplement["sidecar"]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "bound source artifact" in result.stderr
    assert not output_canonical.exists()
    assert not (output_root / "model_a" / "attention_combined.csv").exists()
    assert not (
        output_root / "model_a" / "attention.merge_provenance.json"
    ).exists()


def test_merge_cli_rejects_directory_alias_before_writing_other_outputs(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    base = _publish_attention_source(
        canonical_root,
        run_id="base",
        total_tokens=8,
    )
    supplement = _publish_attention_source(
        supplement_root,
        run_id="supplement",
        total_tokens=16,
    )
    output_alias = output_root / "model_a" / "attention_combined.csv"
    output_alias.mkdir(parents=True)

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
            "--canonical-sidecar",
            str(base["sidecar"]),
            "--supplement-sidecar",
            str(supplement["sidecar"]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "directory" in result.stderr
    assert not (output_root / "model_a" / "attention.csv").exists()
    assert output_alias.is_dir()
    assert not (
        output_root / "model_a" / "attention.merge_provenance.json"
    ).exists()


@pytest.mark.parametrize(
    "collision_field",
    ["source_run_csv", "source_run_sidecar"],
)
def test_merge_cli_rejects_json_out_partition_parent_collisions_before_writing(
    tmp_path: Path,
    collision_field: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    parent_root = tmp_path / "partition-parent"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    base = _publish_attention_source(
        canonical_root,
        run_id="base",
        total_tokens=8,
    )
    parent = _publish_attention_source(
        parent_root,
        run_id="partition-parent",
        total_tokens=16,
    )
    partition_csv = supplement_root / "model_a" / "attention.csv"
    partition_csv.parent.mkdir(parents=True, exist_ok=True)
    partition_csv.write_bytes(parent["run_csv"].read_bytes())
    partition_sidecar = (
        supplement_root / "model_a" / "attention.run_provenance.json"
    )
    write_attention_partition_run_sidecar(
        source_sidecar_path=parent["sidecar"],
        partition_csv=partition_csv,
        sidecar_path=partition_sidecar,
        partition="standard",
        expected_model="model_a",
        expected_measurement_type="CUDA_EVENT",
    )
    partition_payload = json.loads(
        partition_sidecar.read_text(encoding="utf-8")
    )
    collision_path = Path(partition_payload[collision_field])
    source_snapshots = {
        base["canonical"]: base["canonical"].read_bytes(),
        base["sidecar"]: base["sidecar"].read_bytes(),
        partition_csv: partition_csv.read_bytes(),
        partition_sidecar: partition_sidecar.read_bytes(),
        collision_path: collision_path.read_bytes(),
    }

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
            "--canonical-sidecar",
            str(base["sidecar"]),
            "--supplement-sidecar",
            str(partition_sidecar),
            "--json-out",
            str(collision_path),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--json-out" in result.stderr
    for path, expected_bytes in source_snapshots.items():
        assert path.read_bytes() == expected_bytes
    assert not (output_root / "model_a" / "attention.csv").exists()
    assert not (output_root / "model_a" / "attention_combined.csv").exists()
    assert not (
        output_root / "model_a" / "attention.merge_provenance.json"
    ).exists()


def test_merge_cli_rejects_non_native_sources_for_formal_publication(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    base = _publish_attention_source(
        canonical_root,
        run_id="base",
        total_tokens=8,
    )
    supplement = _publish_attention_source(
        supplement_root,
        run_id="supplement",
        total_tokens=16,
    )
    for paths in (base, supplement):
        payload = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
        allocation = payload.pop("allocation_by_tp")["1"]
        payload.pop("allocation_by_tp_semantics")
        payload["is_native_profile_allocation"] = False
        payload.pop("tensor_parallel_sizes")
        payload["tensor_parallel_size"] = 2
        payload.update(allocation)
        payload["config_sha256"] = attention_provenance._config_digest(payload)
        paths["sidecar"].write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
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
            "attention.csv",
            "--canonical-sidecar",
            str(base["sidecar"]),
            "--supplement-sidecar",
            str(supplement["sidecar"]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "native allocation provenance" in result.stderr
    assert not (output_root / "model_a" / "attention.csv").exists()


@pytest.mark.parametrize(
    ("provenance_model", "measurement_type", "message"),
    [
        ("model_b", "CUDA_EVENT", "model identity"),
        ("model_a", "KERNEL_ONLY", "measurement family"),
    ],
)
def test_merge_cli_binds_model_and_measurement_family_before_writing(
    tmp_path: Path,
    provenance_model: str,
    measurement_type: str,
    message: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    base = _publish_attention_source(
        canonical_root,
        run_id="base",
        total_tokens=8,
        provenance_model=provenance_model,
        measurement_type=measurement_type,
    )
    supplement = _publish_attention_source(
        supplement_root,
        run_id="supplement",
        total_tokens=16,
        provenance_model=provenance_model,
        measurement_type=measurement_type,
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
            "attention.csv",
            "--canonical-sidecar",
            str(base["sidecar"]),
            "--supplement-sidecar",
            str(supplement["sidecar"]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stderr
    assert not (output_root / "model_a" / "attention.csv").exists()
