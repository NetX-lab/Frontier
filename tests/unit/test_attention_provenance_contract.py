from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from frontier.profiling.attention import provenance as attention_provenance
from frontier.profiling.attention.provenance import (
    publish_attention_union_and_alias,
    validate_attention_run_sidecar,
    write_attention_partition_run_sidecar,
    write_attention_run_sidecar,
)


def _partitions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standard = pd.DataFrame(
        [
            {
                "row_id": "standard",
                "num_tensor_parallel_workers": 1,
                "prefill_chunk_size": 0,
                "kv_cache_size": 0,
                "batch_size": 1,
                "is_prefill": False,
                "time_stats.attn_kv_cache_save.median": 0.1,
            }
        ]
    )
    mixed = pd.DataFrame(
        [
            {
                "row_id": "mixed",
                "num_tensor_parallel_workers": 1,
                "seq_lens": [2, 2],
                "kv_cache_size": 0,
                "mode": "online_grid_balanced",
                "is_prefill": True,
                "time_stats.attn_kv_cache_save.median": 0.2,
            }
        ]
    )
    true_mixed = pd.DataFrame(
        [
            {
                "row_id": "true",
                "num_tensor_parallel_workers": 1,
                "prefill_seq_lens": [2],
                "prefill_kv_cache_sizes": [0],
                "decode_kv_cache_sizes": [4],
                "time_stats.attn_kv_cache_save.median": 0.3,
            }
        ]
    )
    return standard, mixed, true_mixed


def _complete_allocation_record(
    *,
    physical: int = 100,
    requested: int | None = 18,
    selected: int = 18,
    required: int = 18,
    block_size: int = 16,
) -> dict[str, int | None]:
    return {
        "physical_max_num_blocks": physical,
        "requested_max_num_blocks": requested,
        "selected_max_num_blocks": selected,
        "required_max_num_blocks": required,
        "allocated_max_num_blocks": selected,
        "allocated_kv_token_capacity": selected * block_size,
        "block_size": block_size,
    }


def _attach_allocation_columns(
    frame: pd.DataFrame,
    allocation_by_tp: dict[str, dict[str, int | None]],
) -> pd.DataFrame:
    output = frame.copy()
    if output.empty:
        return output
    for field in next(iter(allocation_by_tp.values())):
        output[field] = [
            allocation_by_tp[str(int(tp_size))][field]
            for tp_size in output["num_tensor_parallel_workers"]
        ]
    return output


def _varying_allocation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 1,
                "prefill_chunk_size": total_tokens,
                "kv_cache_size": 0,
                "batch_size": 1,
                "is_prefill": True,
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": selected,
                "selected_max_num_blocks": selected,
                "required_max_num_blocks": required,
                "allocated_max_num_blocks": selected,
                "allocated_kv_token_capacity": selected * 16,
                "block_size": 16,
                "time_stats.attn_kv_cache_save.median": total_tokens / 1000,
            }
            for total_tokens, selected, required in (
                (8, 18, 17),
                (16, 32, 31),
            )
        ]
    )


def _varying_allocation_provenance() -> dict[str, object]:
    return {
        "model": "probe",
        "device": "h800",
        "tensor_parallel_sizes": [1],
        "measurement_type": "CUDA_EVENT",
        "allocation_by_tp_semantics": "per_tp_column_max_v1",
        "allocation_by_tp": {
            "1": {
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": 32,
                "selected_max_num_blocks": 32,
                "required_max_num_blocks": 31,
                "allocated_max_num_blocks": 32,
                "allocated_kv_token_capacity": 512,
                "block_size": 16,
            }
        },
    }


def _publish_varying_allocation_run(
    tmp_path,
    *,
    provenance: dict[str, object] | None = None,
):
    return publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=_varying_allocation_rows(),
        mixed_df=pd.DataFrame(),
        true_mixed_df=pd.DataFrame(),
        run_id="varying-allocation",
        provenance=provenance or _varying_allocation_provenance(),
    )


@pytest.mark.parametrize(
    "record_updates",
    [
        {"physical_max_num_blocks": 101},
        {"requested_max_num_blocks": None},
        {"required_max_num_blocks": 30},
        {
            "requested_max_num_blocks": 33,
            "selected_max_num_blocks": 33,
            "allocated_max_num_blocks": 33,
            "allocated_kv_token_capacity": 528,
        },
        {
            "allocated_kv_token_capacity": 256,
            "block_size": 8,
        },
    ],
)
def test_native_attention_run_rejects_per_tp_column_max_drift(
    tmp_path,
    record_updates: dict[str, object],
) -> None:
    provenance = _varying_allocation_provenance()
    provenance["allocation_by_tp"]["1"].update(record_updates)
    output_dir = tmp_path / "output"

    with pytest.raises(
        ValueError,
        match="allocation_by_tp.*CSV per-TP column maxima",
    ):
        _publish_varying_allocation_run(
            output_dir,
            provenance=provenance,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "semantics",
    [None, "per_tp_max_envelope"],
)
def test_attention_run_validator_requires_per_tp_column_max_semantics(
    tmp_path,
    semantics: str | None,
) -> None:
    paths = _publish_varying_allocation_run(tmp_path)
    sidecar = paths["sidecar"]
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if semantics is None:
        payload.pop("allocation_by_tp_semantics")
    else:
        payload["allocation_by_tp_semantics"] = semantics
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allocation_by_tp_semantics"):
        validate_attention_run_sidecar(
            csv_path=paths["canonical"],
            sidecar_path=sidecar,
        )


def test_partition_sidecar_recomputes_per_tp_column_maxima(tmp_path) -> None:
    source = _publish_varying_allocation_run(tmp_path / "source")
    source_rows = pd.read_csv(source["canonical"])
    partition_csv = tmp_path / "partition.csv"
    source_rows[source_rows["prefill_chunk_size"] == 8].to_csv(
        partition_csv,
        index=False,
    )

    partition_sidecar = write_attention_partition_run_sidecar(
        source_sidecar_path=source["sidecar"],
        partition_csv=partition_csv,
        sidecar_path=tmp_path / "partition.json",
        partition="standard",
    )

    payload = json.loads(partition_sidecar.read_text(encoding="utf-8"))
    assert payload["allocation_by_tp_semantics"] == "per_tp_column_max_v1"
    assert payload["allocation_by_tp"]["1"] == {
        "physical_max_num_blocks": 100,
        "requested_max_num_blocks": 18,
        "selected_max_num_blocks": 18,
        "required_max_num_blocks": 17,
        "allocated_max_num_blocks": 18,
        "allocated_kv_token_capacity": 288,
        "block_size": 16,
    }


def test_publish_attention_union_and_alias_is_deterministic(tmp_path) -> None:
    standard, mixed, true_mixed = _partitions()
    paths = publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=standard,
        mixed_df=mixed,
        true_mixed_df=true_mixed,
        run_id="run-a",
        provenance={
            "model": "probe",
            "device": "h800",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "physical_max_num_blocks": 100,
            "requested_max_num_blocks": None,
            "selected_max_num_blocks": 18,
            "required_max_num_blocks": 18,
        },
    )

    assert paths["canonical"].read_bytes() == paths["alias"].read_bytes()
    output = pd.read_csv(paths["canonical"])
    assert len(output) == 3
    assert output["is_mixed_batch"].tolist() == [False, True, False]
    assert output["is_true_mixed_batch"].tolist() == [False, False, True]
    assert paths["sidecar"].exists()


def test_publish_attention_union_is_byte_stable_under_worker_row_order(tmp_path) -> None:
    standard, mixed, true_mixed = _partitions()
    standard = pd.concat([standard, standard.assign(row_id="standard-2", kv_cache_size=64)])
    mixed = pd.concat([mixed, mixed.assign(row_id="mixed-2", seq_lens=[[3, 1]])])
    provenance = {
        "model": "probe",
        "device": "h800",
        "tensor_parallel_size": 1,
        "measurement_type": "CUDA_EVENT",
        "physical_max_num_blocks": 100,
        "requested_max_num_blocks": None,
        "selected_max_num_blocks": 18,
        "required_max_num_blocks": 18,
    }
    first = publish_attention_union_and_alias(
        output_dir=tmp_path / "first",
        standard_df=standard,
        mixed_df=mixed,
        true_mixed_df=true_mixed,
        run_id="stable",
        provenance=provenance,
    )
    second = publish_attention_union_and_alias(
        output_dir=tmp_path / "second",
        standard_df=standard.iloc[::-1],
        mixed_df=mixed.iloc[::-1],
        true_mixed_df=true_mixed.iloc[::-1],
        run_id="stable",
        provenance=provenance,
    )
    assert first["run_csv"].read_bytes() == second["run_csv"].read_bytes()


def test_requested_tuple_digest_uses_workload_shape_not_full_row_identity(
    tmp_path,
) -> None:
    standard, mixed, true_mixed = _partitions()
    provenance = {
        "model": "probe",
        "device": "h800",
        "tensor_parallel_size": 1,
        "measurement_type": "CUDA_EVENT",
        "physical_max_num_blocks": 100,
        "requested_max_num_blocks": None,
        "selected_max_num_blocks": 18,
        "required_max_num_blocks": 18,
    }
    first = publish_attention_union_and_alias(
        output_dir=tmp_path / "first",
        standard_df=standard.assign(profile_max_seq_len=128),
        mixed_df=mixed,
        true_mixed_df=true_mixed,
        run_id="same-workload",
        provenance=provenance,
    )
    second = publish_attention_union_and_alias(
        output_dir=tmp_path / "second",
        standard_df=standard.assign(profile_max_seq_len=256),
        mixed_df=mixed,
        true_mixed_df=true_mixed,
        run_id="same-workload",
        provenance=provenance,
    )

    first_payload = json.loads(first["sidecar"].read_text(encoding="utf-8"))
    second_payload = json.loads(second["sidecar"].read_text(encoding="utf-8"))

    assert first_payload["requested_tuple_schema"] == (
        "attention_workload_tuple_multiset_v1"
    )
    assert first_payload["requested_tuple_digest"] == second_payload[
        "requested_tuple_digest"
    ]
    assert first_payload["structural_identity_digest"] != second_payload[
        "structural_identity_digest"
    ]


def test_requested_tuple_digest_is_order_independent_and_multiplicity_sensitive(
    tmp_path,
) -> None:
    standard, _, _ = _partitions()
    one = standard.assign(num_tensor_parallel_workers=1)
    two_tps = pd.concat(
        [
            standard.assign(num_tensor_parallel_workers=1),
            standard.assign(num_tensor_parallel_workers=2),
        ],
        ignore_index=True,
    )
    other_two_tps = pd.concat(
        [
            standard.assign(num_tensor_parallel_workers=1),
            standard.assign(num_tensor_parallel_workers=4),
        ],
        ignore_index=True,
    )
    def publish(name: str, frame: pd.DataFrame) -> dict[str, object]:
        tensor_parallel_sizes = sorted(
            int(value)
            for value in frame["num_tensor_parallel_workers"].unique().tolist()
        )
        allocation_by_tp = {
            str(tp): _complete_allocation_record(requested=None)
            for tp in tensor_parallel_sizes
        }
        paths = publish_attention_union_and_alias(
            output_dir=tmp_path / name,
            standard_df=_attach_allocation_columns(frame, allocation_by_tp),
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame(),
            run_id=name,
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_sizes": tensor_parallel_sizes,
                "measurement_type": "CUDA_EVENT",
                "allocation_by_tp_semantics": "per_tp_column_max_v1",
                "allocation_by_tp": allocation_by_tp,
            },
        )
        return json.loads(paths["sidecar"].read_text(encoding="utf-8"))

    single_payload = publish("single", one)
    double_payload = publish("double", two_tps)
    reversed_payload = publish("reversed", two_tps.iloc[::-1])
    other_tp_payload = publish("other-tp", other_two_tps)

    assert single_payload["requested_tuple_digest"] != double_payload[
        "requested_tuple_digest"
    ]
    assert double_payload["requested_tuple_digest"] == reversed_payload[
        "requested_tuple_digest"
    ]
    assert double_payload["requested_tuple_digest"] == other_tp_payload[
        "requested_tuple_digest"
    ]
    assert double_payload["structural_identity_digest"] != other_tp_payload[
        "structural_identity_digest"
    ]


def test_publish_imported_mla_rows_uses_structural_schema_without_native_fields(
    tmp_path,
) -> None:
    """Imported MLA rows must not be classified by the dense native tuple subset."""

    imported = pd.DataFrame(
        [
            {
                "model_name": "deepseek-ai/DeepSeek-V2-Lite",
                "model_arch": "deepseek_v2",
                "batch_size": 1,
                "batch_num_tokens": 1,
                "batch_num_prefill_tokens": 0,
                "batch_num_decode_tokens": 1,
                "batch_request_num_tokens": [1],
                "max_seqlen_q": 1,
                "max_seqlen_k": 65,
                "num_actual_tokens": 1,
                "is_prefill": False,
                "is_mla_profile_import": True,
                "time_stats.attn_mla_decode.median": 0.05,
            }
        ]
    )

    paths = publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=imported,
        mixed_df=pd.DataFrame(),
        true_mixed_df=pd.DataFrame(),
        run_id="mla-import",
        provenance={
            "model": "deepseek-ai/DeepSeek-V2-Lite",
            "device": "h100",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "is_native_profile_allocation": False,
        },
    )

    payload = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    assert payload["requested_tuple_schema"] == (
        "attention_structural_row_multiset_v1"
    )
    validate_attention_run_sidecar(
        csv_path=paths["run_csv"], sidecar_path=paths["sidecar"]
    )


def _imported_mla_rows(*, max_seqlen_k: int = 65) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "deepseek-ai/DeepSeek-V2-Lite",
                "model_arch": "deepseek_v2",
                "batch_size": 1,
                "batch_num_tokens": 1,
                "batch_num_prefill_tokens": 0,
                "batch_num_decode_tokens": 1,
                "batch_request_num_tokens": [1],
                "max_seqlen_q": 1,
                "max_seqlen_k": max_seqlen_k,
                "num_actual_tokens": 1,
                "is_prefill": False,
                "is_mla_profile_import": True,
                "time_stats.attn_mla_decode.median": 0.05,
            }
        ]
    )


def _publish_imported_rows(tmp_path, name: str, frame: pd.DataFrame) -> dict:
    paths = publish_attention_union_and_alias(
        output_dir=tmp_path / name,
        standard_df=frame,
        mixed_df=pd.DataFrame(),
        true_mixed_df=pd.DataFrame(),
        run_id=name,
        provenance={
            "model": "deepseek-ai/DeepSeek-V2-Lite",
            "device": "h100",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "is_native_profile_allocation": False,
        },
    )
    return json.loads(paths["sidecar"].read_text(encoding="utf-8")), paths


def test_imported_structural_digest_is_order_independent_and_metadata_bound(
    tmp_path,
) -> None:
    first_payload, _ = _publish_imported_rows(
        tmp_path, "first", _imported_mla_rows()
    )
    reversed_rows = pd.concat(
        [_imported_mla_rows(max_seqlen_k=66), _imported_mla_rows()],
        ignore_index=True,
    )
    reordered_rows = reversed_rows.iloc[::-1].reset_index(drop=True)
    reordered_payload, _ = _publish_imported_rows(
        tmp_path, "reordered", reordered_rows
    )

    assert first_payload["requested_tuple_schema"] == (
        "attention_structural_row_multiset_v1"
    )
    # Reordering preserves the multiset, while changing dynamic context metadata
    # changes the structural identity and therefore the imported digest.
    assert reordered_payload["requested_tuple_digest"] != first_payload[
        "requested_tuple_digest"
    ]

    duplicate_order_a = pd.concat(
        [_imported_mla_rows(), _imported_mla_rows(max_seqlen_k=66)],
        ignore_index=True,
    )
    duplicate_order_b = duplicate_order_a.iloc[::-1].reset_index(drop=True)
    order_a_payload, _ = _publish_imported_rows(
        tmp_path, "order-a", duplicate_order_a
    )
    order_b_payload, _ = _publish_imported_rows(
        tmp_path, "order-b", duplicate_order_b
    )
    assert order_a_payload["requested_tuple_digest"] == order_b_payload[
        "requested_tuple_digest"
    ]


def test_imported_structural_digest_tampering_is_rejected(tmp_path) -> None:
    payload, paths = _publish_imported_rows(
        tmp_path, "tampered", _imported_mla_rows()
    )
    payload["requested_tuple_digest"] = "0" * 64
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    paths["sidecar"].write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requested_tuple_digest mismatch"):
        validate_attention_run_sidecar(
            csv_path=paths["run_csv"], sidecar_path=paths["sidecar"]
        )


def test_publish_native_mixed_only_rows_keeps_workload_tuple_schema(tmp_path) -> None:
    mixed = pd.DataFrame(
        [
            {
                "num_tensor_parallel_workers": 1,
                "seq_lens": [2, 4],
                "kv_cache_size": 8,
                "mode": "online_grid_balanced",
                "is_prefill": True,
                "is_mixed_batch": True,
                "time_stats.attn_kv_cache_save.median": 0.2,
            }
        ]
    )
    paths = publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=pd.DataFrame(),
        mixed_df=mixed,
        true_mixed_df=pd.DataFrame(),
        run_id="mixed-only",
        provenance={
            "model": "probe",
            "device": "h800",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "physical_max_num_blocks": 100,
            "requested_max_num_blocks": None,
            "selected_max_num_blocks": 18,
            "required_max_num_blocks": 18,
        },
    )
    payload = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    assert payload["requested_tuple_schema"] == (
        "attention_workload_tuple_multiset_v1"
    )
    validate_attention_run_sidecar(
        csv_path=paths["run_csv"], sidecar_path=paths["sidecar"]
    )


def test_publish_attention_union_rejects_duplicate_partition_key(tmp_path) -> None:
    standard, mixed, true_mixed = _partitions()
    duplicate = pd.concat([standard, standard], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate.*standard"):
        publish_attention_union_and_alias(
            output_dir=tmp_path,
            standard_df=duplicate,
            mixed_df=mixed,
            true_mixed_df=true_mixed,
            run_id="run-duplicate",
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_size": 1,
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": None,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
            },
        )


def test_attention_run_sidecar_binds_csv_digest_and_rejects_mutation(tmp_path) -> None:
    csv_path = tmp_path / "attention.csv"
    csv_path.write_text(
        "num_tensor_parallel_workers,a,b\n1,1,2\n",
        encoding="utf-8",
    )
    sidecar = write_attention_run_sidecar(
        csv_path=csv_path,
        sidecar_path=tmp_path / "attention.run-a.json",
        payload={
            "run_id": "run-a",
            "model": "probe",
            "device": "h800",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "physical_max_num_blocks": 100,
            "requested_max_num_blocks": None,
            "selected_max_num_blocks": 18,
            "required_max_num_blocks": 18,
            "requested_tuple_digest": "abc",
        },
    )
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert "requested_tuple_schema" not in data
    validate_attention_run_sidecar(csv_path=csv_path, sidecar_path=sidecar)

    csv_path.write_text(
        "num_tensor_parallel_workers,a,b\n1,1,3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="csv_sha256"):
        validate_attention_run_sidecar(csv_path=csv_path, sidecar_path=sidecar)


def test_attention_run_sidecar_recomputes_requested_workload_tuple_digest(
    tmp_path,
) -> None:
    standard, mixed, true_mixed = _partitions()
    paths = publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=standard,
        mixed_df=mixed,
        true_mixed_df=true_mixed,
        run_id="tuple-binding",
        provenance={
            "model": "probe",
            "device": "h800",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "physical_max_num_blocks": 100,
            "requested_max_num_blocks": None,
            "selected_max_num_blocks": 18,
            "required_max_num_blocks": 18,
        },
    )
    payload = json.loads(paths["sidecar"].read_text(encoding="utf-8"))
    payload["requested_tuple_schema"] = "attention_workload_tuple_multiset_v1"
    payload["requested_tuple_digest"] = "0" * 64
    payload["config_sha256"] = attention_provenance._config_digest(payload)
    paths["sidecar"].write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requested_tuple_digest mismatch"):
        validate_attention_run_sidecar(
            csv_path=paths["run_csv"],
            sidecar_path=paths["sidecar"],
        )


def test_attention_run_sidecar_rejects_string_block_sentinel(tmp_path) -> None:
    csv_path = tmp_path / "attention.csv"
    csv_path.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="numeric"):
        write_attention_run_sidecar(
            csv_path=csv_path,
            sidecar_path=tmp_path / "attention.run-b.json",
            payload={
                "run_id": "run-b",
                "model": "probe",
                "device": "h800",
                "tensor_parallel_size": 1,
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": "physical_max",
                "requested_max_num_blocks": None,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
            },
        )


def test_publish_attention_union_supports_kernel_only_artifact_names(tmp_path) -> None:
    standard, mixed, true_mixed = _partitions()
    paths = publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=standard,
        mixed_df=mixed,
        true_mixed_df=true_mixed,
        run_id="kernel-run",
        canonical_name="attention_kernel_only.csv",
        alias_name="attention_combined_kernel_only.csv",
        provenance={
            "model": "probe",
            "device": "h800",
            "tensor_parallel_size": 1,
            "measurement_type": "KERNEL_ONLY",
            "physical_max_num_blocks": 100,
            "requested_max_num_blocks": None,
            "selected_max_num_blocks": 18,
            "required_max_num_blocks": 18,
        },
    )
    assert paths["canonical"].name == "attention_kernel_only.csv"
    assert paths["alias"].name == "attention_combined_kernel_only.csv"
    assert paths["canonical"].read_bytes() == paths["alias"].read_bytes()


def test_publish_attention_union_keeps_run_scoped_csv_artifacts(tmp_path) -> None:
    standard, mixed, true_mixed = _partitions()
    kwargs = {
        "output_dir": tmp_path,
        "standard_df": standard,
        "mixed_df": mixed,
        "true_mixed_df": true_mixed,
        "provenance": {
            "model": "probe",
            "device": "h800",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "physical_max_num_blocks": 100,
            "requested_max_num_blocks": None,
            "selected_max_num_blocks": 18,
            "required_max_num_blocks": 18,
        },
    }
    first = publish_attention_union_and_alias(run_id="run-a", **kwargs)
    second = publish_attention_union_and_alias(run_id="run-b", **kwargs)
    assert first["run_csv"].is_file()
    assert second["run_csv"].is_file()
    assert first["run_csv"] != second["run_csv"]
    validate_attention_run_sidecar(
        csv_path=first["run_csv"], sidecar_path=first["sidecar"]
    )


def test_publish_attention_union_rejects_fractional_allocation_provenance(tmp_path) -> None:
    standard, mixed, true_mixed = _partitions()
    with pytest.raises(ValueError, match="positive integer.*numeric"):
        publish_attention_union_and_alias(
            output_dir=tmp_path,
            standard_df=standard,
            mixed_df=mixed,
            true_mixed_df=true_mixed,
            run_id="fractional",
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_size": 1,
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": 100.0,
                "requested_max_num_blocks": None,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
            },
        )


def test_publish_attention_union_accepts_per_tp_numeric_allocations(tmp_path) -> None:
    standard, mixed, true_mixed = _partitions()
    standard = pd.concat(
        [
            standard,
            standard.assign(
                row_id="standard-tp2",
                num_tensor_parallel_workers=2,
            ),
        ],
        ignore_index=True,
    )
    allocation_by_tp = {
        "1": _complete_allocation_record(requested=None),
        "2": _complete_allocation_record(
            physical=80,
            requested=20,
            selected=20,
            required=19,
        ),
    }
    paths = publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=_attach_allocation_columns(standard, allocation_by_tp),
        mixed_df=_attach_allocation_columns(mixed, allocation_by_tp),
        true_mixed_df=_attach_allocation_columns(true_mixed, allocation_by_tp),
        run_id="multi-tp",
        provenance={
            "model": "probe",
            "device": "h800",
            "tensor_parallel_sizes": [1, 2],
            "measurement_type": "CUDA_EVENT",
            "allocation_by_tp_semantics": "per_tp_column_max_v1",
            "allocation_by_tp": allocation_by_tp,
        },
    )
    validate_attention_run_sidecar(
        csv_path=paths["run_csv"], sidecar_path=paths["sidecar"]
    )


def test_native_attention_run_sidecar_rejects_csv_tp_allocation_tp_mismatch(
    tmp_path,
) -> None:
    standard, _, _ = _partitions()

    with pytest.raises(ValueError, match="tensor-parallel binding mismatch"):
        publish_attention_union_and_alias(
            output_dir=tmp_path,
            standard_df=standard,
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame(),
            run_id="tp-mismatch",
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_sizes": [2],
                "measurement_type": "CUDA_EVENT",
                "allocation_by_tp_semantics": "per_tp_column_max_v1",
                "allocation_by_tp": {
                    "2": _complete_allocation_record()
                },
            },
        )

    assert not (tmp_path / "attention.csv").exists()
    assert not (tmp_path / "attention_combined.csv").exists()
    assert not (
        tmp_path / "runs" / "tp-mismatch" / "attention.csv"
    ).exists()
    assert not (tmp_path / "attention.tp-mismatch.json").exists()


@pytest.mark.parametrize(
    ("provenance_update", "message"),
    [
        ({"tensor_parallel_sizes": [1, 1]}, "duplicate"),
        ({"tensor_parallel_sizes": [0]}, "positive integer"),
        (
            {
                "tensor_parallel_sizes": [1],
                    "allocation_by_tp": {
                        "0": _complete_allocation_record()
                    },
                },
            "allocation_by_tp.*positive integer",
        ),
    ],
)
def test_native_attention_run_sidecar_rejects_invalid_tp_identity(
    tmp_path,
    provenance_update,
    message,
) -> None:
    standard, _, _ = _partitions()
    provenance = {
        "model": "probe",
        "device": "h800",
        "tensor_parallel_sizes": [1],
        "measurement_type": "CUDA_EVENT",
        "allocation_by_tp_semantics": "per_tp_column_max_v1",
        "allocation_by_tp": {
            "1": _complete_allocation_record()
        },
    }
    provenance.update(provenance_update)

    with pytest.raises(ValueError, match=message):
        publish_attention_union_and_alias(
            output_dir=tmp_path,
            standard_df=standard,
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame(),
            run_id="invalid-tp",
            provenance=provenance,
        )


def test_native_attention_run_sidecar_requires_declared_tp_identity(tmp_path) -> None:
    standard, _, _ = _partitions()

    with pytest.raises(ValueError, match="tensor_parallel_sizes.*tensor_parallel_size"):
        publish_attention_union_and_alias(
            output_dir=tmp_path,
            standard_df=standard,
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame(),
            run_id="missing-tp",
            provenance={
                "model": "probe",
                "device": "h800",
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": 18,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_max_num_blocks", 101, "exceeds physical_max_num_blocks"),
        ("selected_max_num_blocks", 17, "cannot cover required_max_num_blocks"),
        ("requested_max_num_blocks", 17, "must equal requested_max_num_blocks"),
    ],
)
def test_publish_attention_union_rejects_invalid_per_tp_allocation_relationships(
    tmp_path, field, value, message
) -> None:
    standard, mixed, true_mixed = _partitions()
    record = _complete_allocation_record(requested=None)
    record[field] = value
    with pytest.raises(ValueError, match=message):
        publish_attention_union_and_alias(
            output_dir=tmp_path / str(value),
            standard_df=standard,
            mixed_df=mixed,
            true_mixed_df=true_mixed,
            run_id="invalid-per-tp",
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_sizes": [1],
                "measurement_type": "CUDA_EVENT",
                "allocation_by_tp_semantics": "per_tp_column_max_v1",
                "allocation_by_tp": {"1": record},
            },
        )


def test_publish_attention_union_allows_nullable_allocation_for_imported_profile(
    tmp_path,
) -> None:
    standard, _, _ = _partitions()
    paths = publish_attention_union_and_alias(
        output_dir=tmp_path,
        standard_df=standard,
        mixed_df=pd.DataFrame(),
        true_mixed_df=pd.DataFrame(),
        run_id="imported",
        provenance={
            "model": "probe",
            "device": "h800",
            "tensor_parallel_size": 1,
            "measurement_type": "CUDA_EVENT",
            "is_native_profile_allocation": False,
            "physical_max_num_blocks": None,
            "requested_max_num_blocks": None,
            "selected_max_num_blocks": None,
            "required_max_num_blocks": None,
        },
    )
    validate_attention_run_sidecar(
        csv_path=paths["run_csv"], sidecar_path=paths["sidecar"]
    )


@pytest.mark.parametrize("invalid_run_id", ["..", r"nested\run"])
def test_publish_attention_union_rejects_unsafe_run_id_before_writing(
    tmp_path,
    invalid_run_id,
) -> None:
    standard, _, _ = _partitions()
    output_dir = tmp_path / "publication"

    with pytest.raises(ValueError, match="run_id"):
        publish_attention_union_and_alias(
            output_dir=output_dir,
            standard_df=standard,
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame(),
            run_id=invalid_run_id,
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_size": 1,
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": None,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
            },
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "directory_target",
    ["canonical", "alias", "run_csv", "sidecar"],
)
def test_publish_attention_union_rejects_directory_targets_before_writing(
    tmp_path,
    directory_target,
) -> None:
    standard, _, _ = _partitions()
    output_dir = tmp_path / directory_target
    targets = {
        "canonical": output_dir / "attention.csv",
        "alias": output_dir / "attention_combined.csv",
        "run_csv": output_dir / "runs" / "directory-run" / "attention.csv",
        "sidecar": output_dir / "attention.directory-run.json",
    }
    targets[directory_target].mkdir(parents=True)

    with pytest.raises(ValueError, match="directory"):
        publish_attention_union_and_alias(
            output_dir=output_dir,
            standard_df=standard,
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame(),
            run_id="directory-run",
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_size": 1,
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": None,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
            },
        )

    for name, path in targets.items():
        if name == directory_target:
            assert path.is_dir()
        else:
            assert not path.exists()


def test_publish_attention_union_requires_distinct_output_paths(tmp_path) -> None:
    standard, _, _ = _partitions()

    with pytest.raises(ValueError, match="distinct"):
        publish_attention_union_and_alias(
            output_dir=tmp_path,
            standard_df=standard,
            mixed_df=pd.DataFrame(),
            true_mixed_df=pd.DataFrame(),
            run_id="distinct-run",
            canonical_name="attention.csv",
            alias_name="attention.csv",
            provenance={
                "model": "probe",
                "device": "h800",
                "tensor_parallel_size": 1,
                "measurement_type": "CUDA_EVENT",
                "physical_max_num_blocks": 100,
                "requested_max_num_blocks": None,
                "selected_max_num_blocks": 18,
                "required_max_num_blocks": 18,
            },
        )

    assert not (tmp_path / "attention.csv").exists()
    assert not (tmp_path / "attention.distinct-run.json").exists()
