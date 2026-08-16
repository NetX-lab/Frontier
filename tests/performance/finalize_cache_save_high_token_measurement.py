#!/usr/bin/env python3
"""Publish one measured high-token attention run with native provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from frontier.profiling.attention.provenance import (
    publish_attention_union_and_alias,
    validate_attention_run_id,
)


SUPPORTED_MEASUREMENT_SCHEMAS = {
    "frontier.attention.cache_save_high_token_training_measurement/v1": (
        "high-token-training"
    ),
    "frontier.attention.mixed_prefill_tail_training_measurement/v1": (
        "mixed-prefill-tail-training"
    ),
}
REQUIRED_TP_SIZES = (1, 2, 4, 8)
REQUIRED_ALLOCATION_FIELDS = (
    "physical",
    "requested",
    "required",
    "selected",
    "workspace_bytes",
)
ROW_ALLOCATION_FIELDS = {
    "physical": "physical_max_num_blocks",
    "requested": "requested_max_num_blocks",
    "required": "required_max_num_blocks",
    "selected": "selected_max_num_blocks",
    "workspace_bytes": "backend_workspace_reservation_bytes",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid measurement provenance JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("measurement provenance must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


def _require_positive_integer(value: Any, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not float(value).is_integer()
        or int(value) <= 0
    ):
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return int(value)


def _unique_integer(frame: pd.DataFrame, column: str, *, tp_size: int) -> int:
    if column not in frame.columns:
        raise ValueError(f"measurement CSV is missing allocation column {column!r}")
    values = pd.to_numeric(frame[column], errors="raise")
    unique = sorted({float(value) for value in values.tolist()})
    if len(unique) != 1 or not unique[0].is_integer():
        raise ValueError(
            f"measurement CSV {column!r} must have one integer value for "
            f"TP={tp_size}, got {unique!r}"
        )
    return int(unique[0])


def _validate_source_binding(
    *,
    measurement_csv: Path,
    measurement_sidecar: Path,
    payload: Mapping[str, Any],
    frame: pd.DataFrame,
) -> None:
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_MEASUREMENT_SCHEMAS:
        raise ValueError(
            "unsupported measurement schema_version: "
            f"{schema_version!r}"
        )
    expected_point_set = SUPPORTED_MEASUREMENT_SCHEMAS[str(schema_version)]
    if payload.get("point_set") != expected_point_set:
        raise ValueError(
            f"measurement point_set must be {expected_point_set!r}, "
            f"got {payload.get('point_set')!r}"
        )
    if payload.get("csv") != measurement_csv.name:
        raise ValueError(
            "measurement sidecar CSV filename mismatch: "
            f"sidecar={payload.get('csv')!r}, actual={measurement_csv.name!r}"
        )
    actual_csv_sha = _sha256(measurement_csv)
    if payload.get("csv_sha256") != actual_csv_sha:
        raise ValueError(
            "measurement csv_sha256 mismatch: "
            f"expected={payload.get('csv_sha256')!r}, actual={actual_csv_sha!r}"
        )
    if payload.get("csv_rows") != len(frame):
        raise ValueError(
            "measurement csv_rows mismatch: "
            f"expected={payload.get('csv_rows')!r}, actual={len(frame)!r}"
        )
    required_columns = {
        "num_tensor_parallel_workers",
        "total_tokens",
        "kv_cache_size",
        "batch_size",
        "is_prefill",
        "is_mixed_batch",
        "measurement_type",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"measurement CSV is missing required columns: {missing!r}")
    requested_tuples = [
        [
            int(row.num_tensor_parallel_workers),
            int(row.total_tokens),
            int(row.kv_cache_size),
            int(row.batch_size),
        ]
        for row in frame.itertuples(index=False)
    ]
    if payload.get("requested_tuples") != requested_tuples:
        raise ValueError("measurement requested_tuples do not match CSV row order")
    requested_digest = hashlib.sha256(
        json.dumps(
            requested_tuples,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("requested_tuple_sha256") != requested_digest:
        raise ValueError(
            "measurement requested_tuple_sha256 mismatch: "
            f"expected={payload.get('requested_tuple_sha256')!r}, "
            f"actual={requested_digest!r}"
        )
    observed_tp = tuple(
        sorted(
            pd.to_numeric(
                frame["num_tensor_parallel_workers"], errors="raise"
            ).astype(int).unique()
        )
    )
    declared_tp = tuple(payload.get("tp_sizes", ()))
    if observed_tp != REQUIRED_TP_SIZES or declared_tp != REQUIRED_TP_SIZES:
        raise ValueError(
            "high-token measurement TP binding must be exactly "
            f"{REQUIRED_TP_SIZES!r}: observed={observed_tp!r}, "
            f"declared={declared_tp!r}"
        )
    if not frame["is_prefill"].astype(bool).all():
        raise ValueError("high-token measurement rows must all be prefill rows")
    if not frame["is_mixed_batch"].astype(bool).all():
        raise ValueError("high-token measurement rows must all be mixed-batch rows")
    measurement_types = set(frame["measurement_type"].astype(str))
    if measurement_types != {payload.get("measurement_type")}:
        raise ValueError(
            "measurement_type mismatch between CSV and sidecar: "
            f"csv={sorted(measurement_types)!r}, "
            f"sidecar={payload.get('measurement_type')!r}"
        )
    profile_identity_fields = {
        "precision": "profiling_precision",
        "quant_signature": "quant_signature",
        "model_architecture_profile": "model_architecture_profile",
        "attention_backend": "attention_backend",
    }
    for sidecar_field, csv_field in profile_identity_fields.items():
        expected = payload.get(sidecar_field)
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(
                "measurement profile identity is incomplete: "
                f"{sidecar_field!r} must be a non-empty string"
            )
        if csv_field not in frame.columns:
            raise ValueError(
                "measurement profile identity column is missing from CSV: "
                f"{csv_field!r}"
            )
        actual_values = sorted(
            {
                str(value)
                for value in frame[csv_field].dropna().unique()
                if str(value).strip()
            }
        )
        if actual_values != [expected]:
            raise ValueError(
                "measurement profile identity mismatch for "
                f"{csv_field!r}: sidecar={expected!r}, csv={actual_values!r}"
            )
    validate_attention_run_id(str(payload.get("run_id", "")))
    if not measurement_sidecar.is_file():
        raise FileNotFoundError(measurement_sidecar)


def _formal_allocations(
    *,
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    raw_allocations = payload.get("allocations_by_tp")
    if not isinstance(raw_allocations, Mapping):
        raise ValueError("measurement allocations_by_tp must be a mapping")
    if set(raw_allocations) != {str(value) for value in REQUIRED_TP_SIZES}:
        raise ValueError(
            "measurement allocations_by_tp keys must be exactly "
            f"{[str(value) for value in REQUIRED_TP_SIZES]!r}"
        )
    block_size = _require_positive_integer(
        payload.get("block_size"), field="block_size"
    )
    formal: dict[str, dict[str, int]] = {}
    for tp_size in REQUIRED_TP_SIZES:
        raw_record = raw_allocations[str(tp_size)]
        if not isinstance(raw_record, Mapping):
            raise ValueError(
                f"measurement allocations_by_tp[{tp_size!r}] must be a mapping"
            )
        tp_rows = frame[
            pd.to_numeric(
                frame["num_tensor_parallel_workers"], errors="raise"
            ).astype(int)
            == tp_size
        ]
        record: dict[str, int] = {}
        for source_field in REQUIRED_ALLOCATION_FIELDS:
            sidecar_value = _require_positive_integer(
                raw_record.get(source_field),
                field=f"allocations_by_tp[{tp_size}].{source_field}",
            )
            csv_value = _unique_integer(
                tp_rows,
                ROW_ALLOCATION_FIELDS[source_field],
                tp_size=tp_size,
            )
            if sidecar_value != csv_value:
                raise ValueError(
                    "measurement allocation mismatch for "
                    f"TP={tp_size}, field={source_field!r}: "
                    f"sidecar={sidecar_value}, csv={csv_value}"
                )
            if source_field != "workspace_bytes":
                record[f"{source_field}_max_num_blocks"] = sidecar_value
        record["physical_max_num_blocks"] = record.pop(
            "physical_max_num_blocks"
        )
        record["requested_max_num_blocks"] = record.pop(
            "requested_max_num_blocks"
        )
        record["required_max_num_blocks"] = record.pop(
            "required_max_num_blocks"
        )
        record["selected_max_num_blocks"] = record.pop(
            "selected_max_num_blocks"
        )
        selected = record["selected_max_num_blocks"]
        allocated = _unique_integer(
            tp_rows, "allocated_max_num_blocks", tp_size=tp_size
        )
        capacity = _unique_integer(
            tp_rows, "allocated_kv_token_capacity", tp_size=tp_size
        )
        if allocated != selected or capacity != selected * block_size:
            raise ValueError(
                "measurement allocated capacity mismatch for "
                f"TP={tp_size}: selected={selected}, allocated={allocated}, "
                f"capacity={capacity}, block_size={block_size}"
            )
        record.update(
            {
                "allocated_max_num_blocks": allocated,
                "allocated_kv_token_capacity": capacity,
                "block_size": block_size,
            }
        )
        formal[str(tp_size)] = record
    return formal


def finalize_measurement(
    *,
    measurement_csv: str | Path,
    measurement_sidecar: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Validate a purpose-specific measurement and publish a native run."""

    csv_path = Path(measurement_csv)
    sidecar_path = Path(measurement_sidecar)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"formal output directory already exists: {output}")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(sidecar_path)
    payload = _read_payload(sidecar_path)
    frame = pd.read_csv(csv_path)
    _validate_source_binding(
        measurement_csv=csv_path,
        measurement_sidecar=sidecar_path,
        payload=payload,
        frame=frame,
    )
    allocations = _formal_allocations(frame=frame, payload=payload)
    workspace_values = {
        int(record["workspace_bytes"])
        for record in payload["allocations_by_tp"].values()
    }
    if len(workspace_values) != 1:
        raise ValueError(
            "measurement workspace_bytes must be identical across TP values"
        )
    published_frame = frame.drop(columns=["scenario"], errors="raise")
    provenance = {
        "model": payload["model"],
        "device": payload["device"],
        "measurement_type": payload["measurement_type"],
        "tensor_parallel_sizes": list(REQUIRED_TP_SIZES),
        "allocation_by_tp_semantics": "per_tp_column_max_v1",
        "allocation_by_tp": allocations,
        "backend_workspace_reservation_bytes": workspace_values.pop(),
        "block_size": _require_positive_integer(
            payload.get("block_size"), field="block_size"
        ),
        "max_model_len": _require_positive_integer(
            payload.get("max_model_len"), field="max_model_len"
        ),
        "profile_max_seq_len": _require_positive_integer(
            payload.get("profile_max_seq_len"), field="profile_max_seq_len"
        ),
        "profile_input_grid_max_seq_len": _unique_integer(
            frame,
            "profile_input_grid_max_seq_len",
            tp_size=0,
        ),
        "is_native_profile_allocation": True,
        "profile_method": payload["profile_method"],
        "attention_backend": payload["attention_backend"],
        "profiling_precision": payload["precision"],
        "quant_signature": payload["quant_signature"],
        "model_architecture_profile": payload["model_architecture_profile"],
        "command": payload.get("command"),
        "environment": payload.get("environment"),
        "source_measurement_schema_version": payload["schema_version"],
        "source_measurement_csv_sha256": _sha256(csv_path),
        "source_measurement_sidecar_sha256": _sha256(sidecar_path),
    }
    return publish_attention_union_and_alias(
        output_dir=output,
        standard_df=pd.DataFrame(),
        mixed_df=published_frame,
        true_mixed_df=pd.DataFrame(),
        run_id=str(payload["run_id"]),
        provenance=provenance,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a validated high-token attention measurement into the "
            "native attention run/sidecar publication format."
        )
    )
    parser.add_argument("--measurement-csv", type=Path, required=True)
    parser.add_argument("--measurement-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    published = finalize_measurement(
        measurement_csv=args.measurement_csv,
        measurement_sidecar=args.measurement_sidecar,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {key: str(value) for key, value in published.items()},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
