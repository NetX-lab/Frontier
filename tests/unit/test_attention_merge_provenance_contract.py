from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from frontier.profiling.attention import provenance as attention_provenance
from frontier.profiling.attention.provenance import (
    publish_attention_union_and_alias,
    write_attention_partition_run_sidecar,
)
from tests.e2e.operator_parity.merge_profile_csv_contexts import merge_profile_csvs


def _publish_source(
    tmp_path,
    name: str,
    *,
    total_tokens: int,
    measurement_type: str = "CUDA_EVENT",
    identity_overrides: dict[str, str] | None = None,
    row_identity_overrides: dict[str, str] | None = None,
):
    profile_identity = {
        "profiling_precision": "BF16",
        "quant_signature": "none",
        "model_architecture_profile": "generic",
        "attention_backend": "FLASHINFER",
    }
    profile_identity.update(identity_overrides or {})
    row_profile_identity = {
        **profile_identity,
        **(row_identity_overrides or {}),
    }
    standard = pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 1,
                "prefill_chunk_size": total_tokens,
                "kv_cache_size": 0,
                "batch_size": 1,
                "is_prefill": True,
                "measurement_type": measurement_type,
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": 18,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
                "allocated_max_num_blocks": 18,
                "allocated_kv_token_capacity": 288,
                "block_size": 16,
                "time_stats.attn_kv_cache_save.median": total_tokens / 1000,
                **row_profile_identity,
            }
        ]
    )
    return publish_attention_union_and_alias(
        output_dir=tmp_path / name,
        standard_df=standard,
        mixed_df=pd.DataFrame(),
        true_mixed_df=pd.DataFrame(),
        run_id=name,
        provenance={
            "model": "probe",
            "device": "h800",
            "tensor_parallel_sizes": [1],
            "measurement_type": measurement_type,
            **profile_identity,
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


def _write_merge(
    tmp_path,
    *,
    base_measurement_type: str = "CUDA_EVENT",
    supplement_measurement_type: str = "CUDA_EVENT",
    supplement_identity_overrides: dict[str, str] | None = None,
):
    sources = _merge_sources(
        tmp_path,
        base_measurement_type=base_measurement_type,
        supplement_measurement_type=supplement_measurement_type,
        supplement_identity_overrides=supplement_identity_overrides,
    )
    base = sources["base"]
    supplement = sources["supplement"]
    output_csv = tmp_path / "merged" / "attention.csv"
    output_alias = tmp_path / "merged" / "attention_combined.csv"
    report = merge_profile_csvs(
        canonical_csv=base["canonical"],
        supplement_csv=supplement["canonical"],
        output_csv=output_csv,
    )
    output_alias.write_bytes(output_csv.read_bytes())
    sidecar_path = tmp_path / "merged" / "attention.merge.json"

    return {
        **sources,
        "output_csv": output_csv,
        "output_alias": output_alias,
        "report": report,
        "sidecar_path": sidecar_path,
    }


def _merge_sources(
    tmp_path,
    *,
    base_measurement_type: str = "CUDA_EVENT",
    supplement_measurement_type: str = "CUDA_EVENT",
    supplement_identity_overrides: dict[str, str] | None = None,
):
    base = _publish_source(
        tmp_path,
        "base",
        total_tokens=8,
        measurement_type=base_measurement_type,
    )
    supplement = _publish_source(
        tmp_path,
        "supplement",
        total_tokens=16,
        measurement_type=supplement_measurement_type,
        identity_overrides=supplement_identity_overrides,
    )

    return {
        "base": base,
        "supplement": supplement,
        "sources": [
            {
                "label": "base",
                "csv_path": base["canonical"],
                "sidecar_path": base["sidecar"],
            },
            {
                "label": "supplement",
                "csv_path": supplement["canonical"],
                "sidecar_path": supplement["sidecar"],
            },
        ],
    }


def _publication_request(tmp_path, **source_kwargs):
    request = _merge_sources(tmp_path, **source_kwargs)
    request.update(
        {
            "output_csv": tmp_path / "merged" / "attention.csv",
            "output_alias": tmp_path / "merged" / "attention_combined.csv",
            "sidecar_path": tmp_path / "merged" / "attention.merge.json",
        }
    )
    return request


def _publish_merge_sidecar(merge) -> None:
    attention_provenance.write_attention_merge_sidecar(
        output_csv=merge["output_csv"],
        alias_csv=merge["output_alias"],
        sidecar_path=merge["sidecar_path"],
        sources=merge["sources"],
        merge_report=merge["report"],
    )


def test_attention_merge_sidecar_binds_sources_allocations_and_output(
    tmp_path,
) -> None:
    merge = _write_merge(tmp_path)
    _publish_merge_sidecar(merge)
    attention_provenance.validate_attention_merge_sidecar(
        sidecar_path=merge["sidecar_path"]
    )

    payload = json.loads(merge["sidecar_path"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "frontier.attention.merge_provenance/v1"
    assert payload["measurement_type"] == "CUDA_EVENT"
    assert payload["model"] == "probe"
    assert payload["device"] == "h800"
    assert payload["profiling_precision"] == "BF16"
    assert payload["quant_signature"] == "none"
    assert payload["model_architecture_profile"] == "generic"
    assert payload["attention_backend"] == "FLASHINFER"
    assert payload["row_identity_schema"] == "normalized_csv_row_multiset/v1"
    assert payload["output"]["row_count"] == 2
    assert payload["output"]["alias_csv"] == str(merge["output_alias"])
    assert payload["output"]["csv_sha256"] == payload["output"][
        "alias_csv_sha256"
    ]
    assert [source["label"] for source in payload["sources"]] == [
        "base",
        "supplement",
    ]
    assert payload["sources"][0]["allocation_by_tp"]["1"][
        "selected_max_num_blocks"
    ] == 18
    assert payload["sources"][0]["allocation_by_tp_semantics"] == (
        "per_tp_column_max_v1"
    )


def test_publish_attention_merge_rejects_missing_allocation_semantics(
    tmp_path,
) -> None:
    request = _publication_request(tmp_path)
    source_sidecar = request["base"]["sidecar"]
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload.pop("allocation_by_tp_semantics", None)
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    source_sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allocation_by_tp_semantics"):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["output_alias"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    assert not request["output_csv"].exists()


def test_attention_merge_sidecar_rejects_mismatched_measurement_families(
    tmp_path,
) -> None:
    merge = _write_merge(
        tmp_path,
        supplement_measurement_type="KERNEL_ONLY",
    )

    with pytest.raises(ValueError, match="must share complete profiling identity"):
        _publish_merge_sidecar(merge)


@pytest.mark.parametrize(
    ("field", "different_value"),
    [
        ("profiling_precision", "FP16"),
        ("quant_signature", "fp8"),
        ("model_architecture_profile", "mla"),
        ("attention_backend", "FLASHINFER_MLA"),
    ],
)
def test_attention_merge_sidecar_rejects_mismatched_profile_identity(
    tmp_path,
    field,
    different_value,
) -> None:
    merge = _write_merge(
        tmp_path,
        supplement_identity_overrides={field: different_value},
    )

    with pytest.raises(ValueError, match="must share complete profiling identity"):
        _publish_merge_sidecar(merge)


def test_attention_run_publication_rejects_profile_identity_row_mismatch(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="profiling identity mismatch.*profiling_precision",
    ):
        _publish_source(
            tmp_path,
            "mismatched",
            total_tokens=8,
            row_identity_overrides={"profiling_precision": "FP16"},
        )


def test_attention_merge_sidecar_rejects_alias_byte_drift(tmp_path) -> None:
    merge = _write_merge(tmp_path)
    _publish_merge_sidecar(merge)
    merge["output_alias"].write_bytes(merge["output_alias"].read_bytes() + b"\n")

    with pytest.raises(ValueError, match="must be byte-identical"):
        attention_provenance.validate_attention_merge_sidecar(
            sidecar_path=merge["sidecar_path"]
        )


def test_attention_merge_sidecar_rejects_output_row_identity_drift(
    tmp_path,
) -> None:
    merge = _write_merge(tmp_path)
    _publish_merge_sidecar(merge)
    output = pd.read_csv(merge["output_csv"])
    output.loc[0, "time_stats.attn_kv_cache_save.median"] = 9.0
    output.to_csv(merge["output_csv"], index=False, lineterminator="\n")
    merge["output_alias"].write_bytes(merge["output_csv"].read_bytes())

    with pytest.raises(ValueError, match="normalized row identities"):
        attention_provenance.validate_attention_merge_sidecar(
            sidecar_path=merge["sidecar_path"]
        )


def test_attention_merge_sidecar_rejects_source_allocation_drift(
    tmp_path,
) -> None:
    merge = _write_merge(tmp_path)
    _publish_merge_sidecar(merge)
    source_sidecar = merge["base"]["sidecar"]
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload["allocation_by_tp"]["1"]["physical_max_num_blocks"] = 101
    source_sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV per-TP column maxima"):
        attention_provenance.validate_attention_merge_sidecar(
            sidecar_path=merge["sidecar_path"]
        )


def test_publish_attention_merge_rejects_source_csv_tp_allocation_tp_mismatch(
    tmp_path,
) -> None:
    request = _publication_request(tmp_path)
    source_sidecar = request["base"]["sidecar"]
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload["tensor_parallel_sizes"] = [2]
    payload["allocation_by_tp"] = {
        "2": payload["allocation_by_tp"].pop("1")
    }
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    source_sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tensor-parallel binding mismatch"):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["output_alias"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    assert not request["output_csv"].exists()
    assert not request["output_alias"].exists()
    assert not request["sidecar_path"].exists()


def test_publish_attention_merge_rejects_all_none_scalar_allocation(
    tmp_path,
) -> None:
    request = _publication_request(tmp_path)
    source_sidecar = request["base"]["sidecar"]
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload["is_native_profile_allocation"] = False
    payload.pop("allocation_by_tp")
    payload.pop("allocation_by_tp_semantics")
    for field in (
        "physical_max_num_blocks",
        "requested_max_num_blocks",
        "selected_max_num_blocks",
        "required_max_num_blocks",
    ):
        payload[field] = None
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    source_sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="native allocation provenance"):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["output_alias"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    assert not request["output_csv"].exists()
    assert not request["output_alias"].exists()
    assert not request["sidecar_path"].exists()


def test_publish_attention_merge_rejects_non_native_allocation_escape(
    tmp_path,
) -> None:
    request = _publication_request(tmp_path)
    for source in (request["base"], request["supplement"]):
        source_sidecar = source["sidecar"]
        payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
        allocation = payload.pop("allocation_by_tp")["1"]
        payload.pop("allocation_by_tp_semantics")
        payload["is_native_profile_allocation"] = False
        payload.pop("tensor_parallel_sizes")
        payload["tensor_parallel_size"] = 2
        payload.update(allocation)
        payload["config_sha256"] = attention_provenance._config_digest(payload)
        source_sidecar.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="native allocation provenance"):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["output_alias"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    assert not request["output_csv"].exists()
    assert not request["output_alias"].exists()
    assert not request["sidecar_path"].exists()


def test_attention_merge_sidecar_rejects_normalized_source_identity_drift(
    tmp_path,
) -> None:
    merge = _write_merge(tmp_path)
    _publish_merge_sidecar(merge)
    _publish_source(tmp_path, "supplement", total_tokens=32)

    with pytest.raises(ValueError, match="normalized row identities"):
        attention_provenance.validate_attention_merge_sidecar(
            sidecar_path=merge["sidecar_path"]
        )


def test_publish_attention_merge_validates_before_publishing_outputs(tmp_path) -> None:
    request = _publication_request(
        tmp_path,
        supplement_measurement_type="KERNEL_ONLY",
    )

    with pytest.raises(ValueError, match="must share complete profiling identity"):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["output_alias"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    assert not request["output_csv"].exists()
    assert not request["output_alias"].exists()
    assert not request["sidecar_path"].exists()


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "device",
        "measurement_type",
        "profiling_precision",
        "quant_signature",
        "model_architecture_profile",
        "attention_backend",
        "run_id",
        "requested_tuple_schema",
        "requested_tuple_digest",
        "structural_identity_digest",
    ],
)
def test_publish_attention_merge_requires_complete_source_identity(
    tmp_path,
    field,
) -> None:
    request = _publication_request(tmp_path)
    source_sidecar = request["base"]["sidecar"]
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload.pop(field)
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    source_sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    request["output_alias"].parent.mkdir(parents=True, exist_ok=True)
    request["output_alias"].write_bytes(b"existing-alias\n")

    with pytest.raises(
        ValueError,
        match=rf"source 'base'.*required field {field!r}",
    ):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["output_alias"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    assert request["output_alias"].read_bytes() == b"existing-alias\n"
    assert not request["output_csv"].exists()
    assert not request["sidecar_path"].exists()


def test_publish_attention_merge_writes_validated_canonical_sidecar_then_alias(
    tmp_path,
) -> None:
    request = _publication_request(tmp_path)

    result = attention_provenance.publish_attention_merge(
        output_csv=request["output_csv"],
        alias_csv=request["output_alias"],
        sidecar_path=request["sidecar_path"],
        sources=request["sources"],
    )

    attention_provenance.validate_attention_merge_sidecar(
        sidecar_path=request["sidecar_path"]
    )
    assert request["output_csv"].read_bytes() == request["output_alias"].read_bytes()
    assert result["report"]["base_row_count"] == 1
    assert result["report"]["supplement_row_count"] == 1
    assert result["report"]["merged_row_count"] == 2
    assert result["canonical"] == request["output_csv"]
    assert result["alias"] == request["output_alias"]
    assert result["sidecar"] == request["sidecar_path"]


def test_publish_attention_merge_rejects_any_output_overwriting_a_source(
    tmp_path,
) -> None:
    request = _publication_request(tmp_path)
    source_bytes = request["base"]["canonical"].read_bytes()

    with pytest.raises(ValueError, match="cannot overwrite a bound source artifact"):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["base"]["canonical"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    assert request["base"]["canonical"].read_bytes() == source_bytes
    assert not request["output_csv"].exists()
    assert not request["sidecar_path"].exists()


@pytest.mark.parametrize(
    "directory_target",
    ["output_csv", "output_alias", "sidecar_path"],
)
def test_publish_attention_merge_rejects_directory_targets_before_writing(
    tmp_path,
    directory_target,
) -> None:
    request = _publication_request(tmp_path)
    targets = {
        "output_csv": request["output_csv"],
        "output_alias": request["output_alias"],
        "sidecar_path": request["sidecar_path"],
    }
    targets[directory_target].mkdir(parents=True)

    with pytest.raises(ValueError, match="directory"):
        attention_provenance.publish_attention_merge(
            output_csv=request["output_csv"],
            alias_csv=request["output_alias"],
            sidecar_path=request["sidecar_path"],
            sources=request["sources"],
        )

    for name, path in targets.items():
        if name == directory_target:
            assert path.is_dir()
        else:
            assert not path.exists()


@pytest.mark.parametrize(
    "collision_kind",
    [
        "direct_artifact_csv",
        "derived_parent_csv",
        "derived_parent_sidecar",
    ],
)
def test_publish_attention_merge_rejects_transitive_source_artifact_collisions(
    tmp_path,
    collision_kind,
) -> None:
    request = _publication_request(tmp_path)
    parent = _publish_source(
        tmp_path,
        "partition-parent",
        total_tokens=32,
    )
    partition_csv = tmp_path / "partition" / "attention.csv"
    partition_csv.parent.mkdir(parents=True)
    partition_csv.write_bytes(parent["run_csv"].read_bytes())
    partition_sidecar = tmp_path / "partition" / "attention.run.json"
    write_attention_partition_run_sidecar(
        source_sidecar_path=parent["sidecar"],
        partition_csv=partition_csv,
        sidecar_path=partition_sidecar,
        partition="standard",
        expected_model="probe",
        expected_measurement_type="CUDA_EVENT",
    )
    request["sources"][1] = {
        "label": "supplement",
        "csv_path": partition_csv,
        "sidecar_path": partition_sidecar,
    }
    base_payload = json.loads(
        request["base"]["sidecar"].read_text(encoding="utf-8")
    )
    partition_payload = json.loads(
        partition_sidecar.read_text(encoding="utf-8")
    )
    collisions = {
        "direct_artifact_csv": base_payload["artifact_csv"],
        "derived_parent_csv": partition_payload["source_run_csv"],
        "derived_parent_sidecar": partition_payload["source_run_sidecar"],
    }
    collision_path = Path(collisions[collision_kind])
    collision_bytes = collision_path.read_bytes()
    output_alias = tmp_path / "collision-output" / "attention_combined.csv"
    output_sidecar = tmp_path / "collision-output" / "attention.merge.json"

    with pytest.raises(ValueError, match="bound source artifact"):
        attention_provenance.publish_attention_merge(
            output_csv=collision_path,
            alias_csv=output_alias,
            sidecar_path=output_sidecar,
            sources=request["sources"],
        )

    assert collision_path.read_bytes() == collision_bytes
    assert not output_alias.exists()
    assert not output_sidecar.exists()


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("key_columns", ["incorrect"]),
        ("key_column_count", 0),
        ("duplicate_identical_count", 99),
        ("supplement_duplicate_identical_count", 99),
    ],
)
def test_attention_merge_sidecar_rejects_incorrect_merge_report(
    tmp_path,
    field,
    invalid_value,
) -> None:
    merge = _write_merge(tmp_path)
    merge["report"][field] = invalid_value

    with pytest.raises(ValueError, match=rf"{field} mismatch"):
        attention_provenance.write_attention_merge_sidecar(
            output_csv=merge["output_csv"],
            alias_csv=merge["output_alias"],
            sidecar_path=merge["sidecar_path"],
            sources=merge["sources"],
            merge_report=merge["report"],
        )
