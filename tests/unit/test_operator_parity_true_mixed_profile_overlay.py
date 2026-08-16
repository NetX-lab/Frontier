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
    validate_attention_run_sidecar,
    validate_attention_merge_sidecar,
    write_attention_partition_run_sidecar,
)
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
        "prefill_seq_lens": "[16]",
        "prefill_kv_cache_sizes": "[0]",
        "decode_kv_cache_sizes": "[16]",
        "decode_batch_size": "1",
        "decode_avg_kv_cache_size": "16",
        "num_prefill_seqs": "1",
        "total_prefill_tokens": "16",
        "total_batch_size": "2",
        "batch_composition_ratio": "0.5",
        "total_tokens": "17",
        "profiling_precision": "BF16",
        "quant_signature": "none",
        "model_architecture_profile": "generic",
        "attention_backend": "FLASHINFER",
        "physical_max_num_blocks": "100",
        "requested_max_num_blocks": "18",
        "selected_max_num_blocks": "18",
        "required_max_num_blocks": "18",
        "allocated_max_num_blocks": "18",
        "allocated_kv_token_capacity": "288",
        "block_size": "16",
        "time_stats.attn_decode.median": "3.25",
    }
    row.update(overrides)
    return row


def _publish_true_mixed_stage_sources(
    stage_root: Path,
    *,
    provenance_model: str = "model_a",
) -> dict[tuple[str, str], Path]:
    source_sidecars: dict[tuple[str, str], Path] = {}
    for measurement_type, canonical_name, alias_name, source_name in (
        (
            "CUDA_EVENT",
            "attention.csv",
            "attention_combined.csv",
            "attention_true_mixed.csv",
        ),
        (
            "KERNEL_ONLY",
            "attention_kernel_only.csv",
            "attention_combined_kernel_only.csv",
            "attention_true_mixed_kernel_only.csv",
        ),
    ):
        row = _true_mixed_row(measurement_type=measurement_type)
        published = publish_attention_union_and_alias(
            output_dir=stage_root / "model_a",
            standard_df=pd.DataFrame(),
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame([row]),
            run_id=f"{measurement_type.lower()}-source",
            canonical_name=canonical_name,
            alias_name=alias_name,
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
        source = stage_root / "model_a" / source_name
        source.write_bytes(published["run_csv"].read_bytes())
        source_sidecars[("model_a", source_name)] = published["sidecar"]
    return source_sidecars


def _publish_canonical_source(canonical_root: Path):
    frame = pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 1,
                "prefill_chunk_size": 8,
                "kv_cache_size": 0,
                "batch_size": 1,
                "is_prefill": True,
                "measurement_type": "CUDA_EVENT",
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
                "time_stats.attn_kv_cache_save.median": 0.008,
            }
        ]
    )
    return publish_attention_union_and_alias(
        output_dir=canonical_root / "model_a",
        standard_df=frame,
        mixed_df=pd.DataFrame(),
        true_mixed_df=pd.DataFrame(),
        run_id="base",
        provenance={
            "model": "model_a",
            "device": "h800",
            "tensor_parallel_sizes": [1],
            "measurement_type": "CUDA_EVENT",
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


def test_build_overlay_maps_true_mixed_sources_to_canonical_supplement_names(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _write_csv(canonical_root / "model_a" / "attention.csv", [{"k": "base"}])
    _write_csv(canonical_root / "model_a" / "attention_kernel_only.csv", [{"k": "base"}])
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)

    overlay_report = tmp_path / "overlay_report.json"
    overlay_result = subprocess.run(
        [
            sys.executable,
            "tests/e2e/operator_parity/build_true_mixed_profile_overlay.py",
            "--canonical-root",
            str(canonical_root),
            "--stage-root",
            str(stage_root),
            "--overlay-root",
            str(overlay_root),
            "--supplement-root",
            str(supplement_root),
            "--models",
            "model_a",
            "--expected-true-mixed-rows-per-file",
            "1",
            "--expected-tp-values",
            "1",
            "--source-sidecar",
            "model_a",
            "attention_true_mixed.csv",
            str(source_sidecars[("model_a", "attention_true_mixed.csv")]),
            "--source-sidecar",
            "model_a",
            "attention_true_mixed_kernel_only.csv",
            str(
                source_sidecars[
                    ("model_a", "attention_true_mixed_kernel_only.csv")
                ]
            ),
            "--json-out",
            str(overlay_report),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert overlay_result.returncode == 0, overlay_result.stderr
    summary = json.loads(overlay_report.read_text(encoding="utf-8"))

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
    for report in summary["reports"]:
        assert Path(report["target_sidecar"]).is_file()


def test_build_overlay_outputs_sidecars_consumable_by_formal_merge_cli(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    output_root = tmp_path / "merged"
    canonical = _publish_canonical_source(canonical_root)
    _write_csv(
        canonical_root / "model_a" / "attention_kernel_only.csv",
        [{"k": "base"}],
    )
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)

    summary = build_overlay(
        canonical_root=canonical_root,
        stage_root=stage_root,
        overlay_root=overlay_root,
        supplement_root=supplement_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
        source_sidecars=source_sidecars,
    )
    eager_report = next(
        report
        for report in summary["reports"]
        if report["target_filename"] == "attention.csv"
    )

    result = subprocess.run(
        [
            sys.executable,
            "tests/e2e/operator_parity/merge_profile_csv_contexts.py",
            "--canonical-root",
            str(overlay_root),
            "--supplement-root",
            str(supplement_root),
            "--output-root",
            str(output_root),
            "--models",
            "model_a",
            "--filenames",
            "attention.csv",
            "--canonical-sidecar",
            str(canonical["sidecar"]),
            "--supplement-sidecar",
            str(eager_report["target_sidecar"]),
        ],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    validate_attention_merge_sidecar(
        sidecar_path=(
            output_root
            / "model_a"
            / "attention.merge_provenance.json"
        )
    )


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


@pytest.mark.parametrize(
    "layout",
    [
        "same",
        "supplement_inside_overlay",
        "overlay_inside_supplement",
    ],
)
def test_build_overlay_rejects_overlapping_output_roots_before_writing(
    tmp_path: Path,
    layout: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    if layout == "same":
        overlay_root = supplement_root = tmp_path / "published"
    elif layout == "supplement_inside_overlay":
        overlay_root = tmp_path / "published"
        supplement_root = overlay_root / "supplement"
    else:
        supplement_root = tmp_path / "published"
        overlay_root = supplement_root / "overlay"
    _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)

    with pytest.raises(ValueError, match="roots must be disjoint"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
            source_sidecars=source_sidecars,
        )

    assert not overlay_root.exists()
    assert not supplement_root.exists()


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


def test_build_overlay_rejects_partition_timing_not_present_in_source_run(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)
    tampered_source = stage_root / "model_a" / "attention_true_mixed.csv"
    tampered_rows = _read_rows(tampered_source)
    tampered_rows[0]["time_stats.attn_decode.median"] = "9.99"
    _write_csv(tampered_source, tampered_rows)

    with pytest.raises(ValueError, match="complete normalized rows"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
            source_sidecars=source_sidecars,
        )

    assert not overlay_root.exists()
    assert not supplement_root.exists()


@pytest.mark.parametrize("parent_artifact", ["csv", "sidecar"])
def test_partition_sidecar_revalidates_parent_provenance(
    tmp_path: Path,
    parent_artifact: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)
    summary = build_overlay(
        canonical_root=canonical_root,
        stage_root=stage_root,
        overlay_root=overlay_root,
        supplement_root=supplement_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
        source_sidecars=source_sidecars,
    )
    eager_report = next(
        report
        for report in summary["reports"]
        if report["target_filename"] == "attention.csv"
    )
    child_csv = Path(eager_report["target"])
    child_sidecar = Path(eager_report["target_sidecar"])
    source_sidecar = source_sidecars[
        ("model_a", "attention_true_mixed.csv")
    ]
    source_payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    if parent_artifact == "csv":
        source_csv = Path(source_payload["artifact_csv"])
        source_csv.write_bytes(source_csv.read_bytes() + b"\n")
    else:
        source_sidecar.write_text(
            source_sidecar.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="parent .*sha256 mismatch"):
        validate_attention_run_sidecar(
            csv_path=child_csv,
            sidecar_path=child_sidecar,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("run_id", "inherited metadata"),
        ("allocation_by_tp", "per-TP column maxima"),
    ],
)
def test_partition_sidecar_rejects_inherited_metadata_drift(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)
    summary = build_overlay(
        canonical_root=canonical_root,
        stage_root=stage_root,
        overlay_root=overlay_root,
        supplement_root=supplement_root,
        models=("model_a",),
        expected_true_mixed_rows_per_file=1,
        expected_tp_values=(1,),
        source_sidecars=source_sidecars,
    )
    eager_report = next(
        report
        for report in summary["reports"]
        if report["target_filename"] == "attention.csv"
    )
    child_csv = Path(eager_report["target"])
    child_sidecar = Path(eager_report["target_sidecar"])
    payload = json.loads(child_sidecar.read_text(encoding="utf-8"))
    if field == "run_id":
        payload["run_id"] = "forged-child"
    else:
        payload["allocation_by_tp"]["1"]["physical_max_num_blocks"] = 101
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    child_sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        validate_attention_run_sidecar(
            csv_path=child_csv,
            sidecar_path=child_sidecar,
        )


def test_build_overlay_rejects_derived_partition_source_before_writing(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)
    eager_source = stage_root / "model_a" / "attention_true_mixed.csv"
    derived_sidecar = stage_root / "model_a" / "derived.run_provenance.json"
    write_attention_partition_run_sidecar(
        source_sidecar_path=source_sidecars[
            ("model_a", "attention_true_mixed.csv")
        ],
        partition_csv=eager_source,
        sidecar_path=derived_sidecar,
        partition="true_mixed",
        expected_model="model_a",
        expected_measurement_type="CUDA_EVENT",
    )
    source_sidecars[("model_a", "attention_true_mixed.csv")] = derived_sidecar

    with pytest.raises(ValueError, match="direct profiling run"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
            source_sidecars=source_sidecars,
        )

    assert not overlay_root.exists()
    assert not supplement_root.exists()


def test_build_overlay_preflights_all_required_sidecars_before_writing(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)
    source_sidecars.pop(
        ("model_a", "attention_true_mixed_kernel_only.csv")
    )

    with pytest.raises(ValueError, match="required for every true-mixed partition"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
            source_sidecars=source_sidecars,
        )

    assert not overlay_root.exists()
    assert not supplement_root.exists()


def test_build_overlay_binds_directory_model_to_sidecar_model_before_writing(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(
        stage_root,
        provenance_model="model_b",
    )

    with pytest.raises(ValueError, match="model identity"):
        build_overlay(
            canonical_root=canonical_root,
            stage_root=stage_root,
            overlay_root=overlay_root,
            supplement_root=supplement_root,
            models=("model_a",),
            expected_true_mixed_rows_per_file=1,
            expected_tp_values=(1,),
            source_sidecars=source_sidecars,
        )

    assert not overlay_root.exists()
    assert not supplement_root.exists()


@pytest.mark.parametrize(
    "collision_kind",
    [
        "canonical_source",
        "stage_source",
        "source_sidecar",
        "bound_source_artifact",
        "overlay_alias",
        "supplement_csv",
        "supplement_sidecar",
        "overlay_root",
        "overlay_model_dir",
        "overlay_model_descendant",
        "overlay_alias_descendant",
        "supplement_root",
    ],
)
def test_overlay_cli_rejects_json_out_artifact_collisions_before_writing(
    tmp_path: Path,
    collision_kind: str,
) -> None:
    canonical_root = tmp_path / "canonical"
    stage_root = tmp_path / "stage"
    overlay_root = tmp_path / "overlay"
    supplement_root = tmp_path / "supplement"
    canonical = _publish_canonical_source(canonical_root)
    source_sidecars = _publish_true_mixed_stage_sources(stage_root)
    eager_source = stage_root / "model_a" / "attention_true_mixed.csv"
    eager_sidecar = source_sidecars[
        ("model_a", "attention_true_mixed.csv")
    ]
    eager_payload = json.loads(eager_sidecar.read_text(encoding="utf-8"))
    bound_source_artifact = Path(eager_payload["artifact_csv"])
    collisions = {
        "canonical_source": canonical["canonical"],
        "stage_source": eager_source,
        "source_sidecar": eager_sidecar,
        "bound_source_artifact": bound_source_artifact,
        "overlay_alias": (
            overlay_root / "model_a" / "attention_combined.csv"
        ),
        "supplement_csv": supplement_root / "model_a" / "attention.csv",
        "supplement_sidecar": (
            supplement_root
            / "model_a"
            / "attention.run_provenance.json"
        ),
        "overlay_root": overlay_root,
        "overlay_model_dir": overlay_root / "model_a",
        "overlay_model_descendant": (
            overlay_root / "model_a" / "report.json"
        ),
        "overlay_alias_descendant": (
            overlay_root
            / "model_a"
            / "attention_combined.csv"
            / "report.json"
        ),
        "supplement_root": supplement_root,
    }
    source_snapshots = {
        canonical["canonical"]: canonical["canonical"].read_bytes(),
        eager_source: eager_source.read_bytes(),
        eager_sidecar: eager_sidecar.read_bytes(),
        bound_source_artifact: bound_source_artifact.read_bytes(),
    }

    result = subprocess.run(
        [
            sys.executable,
            "tests/e2e/operator_parity/build_true_mixed_profile_overlay.py",
            "--canonical-root",
            str(canonical_root),
            "--stage-root",
            str(stage_root),
            "--overlay-root",
            str(overlay_root),
            "--supplement-root",
            str(supplement_root),
            "--models",
            "model_a",
            "--expected-true-mixed-rows-per-file",
            "1",
            "--expected-tp-values",
            "1",
            "--source-sidecar",
            "model_a",
            "attention_true_mixed.csv",
            str(eager_sidecar),
            "--source-sidecar",
            "model_a",
            "attention_true_mixed_kernel_only.csv",
            str(
                source_sidecars[
                    ("model_a", "attention_true_mixed_kernel_only.csv")
                ]
            ),
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
    assert not overlay_root.exists()
    assert not supplement_root.exists()
