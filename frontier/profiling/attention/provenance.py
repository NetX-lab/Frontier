"""Artifact-scoped provenance for native attention profiling runs.

The profiler emits several attention partitions (standard, mixed, and true
mixed).  This module gives those partitions one deterministic canonical CSV,
keeps the historical ``attention_combined.csv`` name as a byte-identical
compatibility alias, and binds both files to a run sidecar containing the
allocation and coverage facts used to produce them.

The helpers are intentionally independent of the GPU wrappers.  They can be
tested with small dataframes and are also used by the production entrypoint
after all workers have completed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ATTENTION_PROVENANCE_SCHEMA_VERSION = 1
ATTENTION_REQUESTED_TUPLE_SCHEMA = "attention_workload_tuple_multiset_v1"
ATTENTION_IMPORTED_ROW_SCHEMA = "attention_structural_row_multiset_v1"
ATTENTION_MERGE_PROVENANCE_SCHEMA = "frontier.attention.merge_provenance/v1"
ATTENTION_MERGE_ROW_IDENTITY_SCHEMA = "normalized_csv_row_multiset/v1"
ATTENTION_ALLOCATION_BY_TP_SEMANTICS = "per_tp_column_max_v1"
_ATTENTION_PROFILE_IDENTITY_FIELDS = (
    "profiling_precision",
    "quant_signature",
    "model_architecture_profile",
    "attention_backend",
)
_ATTENTION_MERGE_SOURCE_TEXT_FIELDS = (
    "model",
    "device",
    "measurement_type",
    *_ATTENTION_PROFILE_IDENTITY_FIELDS,
    "run_id",
    "requested_tuple_schema",
)
_ATTENTION_MERGE_SOURCE_DIGEST_FIELDS = (
    "requested_tuple_digest",
    "structural_identity_digest",
)
_ATTENTION_PARTITION_PARENT_FIELDS = (
    "partition",
    "source_run_csv",
    "source_run_csv_sha256",
    "source_run_sidecar",
    "source_run_sidecar_sha256",
)
_ATTENTION_RUN_BOUND_PATH_FIELDS = (
    "artifact_csv",
    "source_run_csv",
    "source_run_sidecar",
)
_BLOCK_FIELDS = (
    "physical_max_num_blocks",
    "requested_max_num_blocks",
    "selected_max_num_blocks",
    "required_max_num_blocks",
)
_POSITIVE_BLOCK_FIELDS = (
    "physical_max_num_blocks",
    "selected_max_num_blocks",
    "required_max_num_blocks",
)
_ALLOCATION_BY_TP_COLUMN_FIELDS = (
    "physical_max_num_blocks",
    "requested_max_num_blocks",
    "selected_max_num_blocks",
    "required_max_num_blocks",
    "allocated_max_num_blocks",
    "allocated_kv_token_capacity",
    "block_size",
)
_GENERATED_FIELDS = {
    "csv_sha256",
    "canonical_csv_sha256",
    "config_sha256",
    "row_count",
    "requested_row_count",
    "artifact_csv",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if not hasattr(result, "__len__") else False


def _jsonable(value: Any) -> Any:
    """Convert dataframe/object values to a deterministic JSON-compatible form."""

    if _is_missing(value):
        return None
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        # Keep integral floats canonical without changing non-integral timings.
        return int(value) if value.is_integer() else float(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _freeze(value: Any) -> Any:
    """Return a hashable form of a JSON-compatible value."""

    value = _jsonable(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items()))
    return value


def _natural_sort_atom(value: Any) -> tuple[Any, ...]:
    """Return a comparable key that keeps numeric values in numeric order."""

    value = _jsonable(value)
    if value is None:
        return (4, "")
    if isinstance(value, bool):
        return (2, int(value))
    if isinstance(value, (int, float)):
        return (0, float(value))
    if isinstance(value, list):
        return (3, tuple(_natural_sort_atom(item) for item in value))
    if isinstance(value, dict):
        return (
            3,
            tuple(
                (str(key), _natural_sort_atom(item))
                for key, item in sorted(value.items())
            ),
        )
    return (1, str(value))


def _partition_marker_defaults(partition: str) -> tuple[bool, bool]:
    if partition == "standard":
        return False, False
    if partition == "mixed":
        return True, False
    if partition == "true_mixed":
        return False, True
    raise ValueError(f"Unsupported attention partition {partition!r}.")


def _normalize_marker(
    frame: pd.DataFrame,
    *,
    column: str,
    expected: bool,
    partition: str,
) -> None:
    if column not in frame.columns:
        frame[column] = bool(expected)
        return
    values = []
    for value in frame[column].tolist():
        if _is_missing(value):
            values.append(bool(expected))
            continue
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                value = True
            elif normalized in {"false", "0", "no"}:
                value = False
            else:
                raise ValueError(
                    f"{partition} attention marker {column!r} has invalid value "
                    f"{value!r}."
                )
        normalized = bool(value)
        if normalized != expected:
            raise ValueError(
                f"{partition} attention partition has conflicting {column}="
                f"{normalized!r}; expected {expected!r}."
            )
        values.append(normalized)
    frame[column] = pd.Series(values, index=frame.index, dtype=bool)


def _normalize_partition(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{partition}_df must be a pandas.DataFrame.")
    result = frame.copy()
    if result.empty:
        # Preserve no arbitrary columns for an empty partition; the union schema
        # is determined by non-empty partitions.
        return result
    mixed_default, true_mixed_default = _partition_marker_defaults(partition)
    _normalize_marker(
        result,
        column="is_mixed_batch",
        expected=mixed_default,
        partition=partition,
    )
    _normalize_marker(
        result,
        column="is_true_mixed_batch",
        expected=true_mixed_default,
        partition=partition,
    )
    return result


def _row_identity(row: Mapping[str, Any], partition: str) -> tuple[Any, ...]:
    """Build a full structural identity, excluding measured timing values."""

    # ``row_id`` is a fixture/debug label, not a workload dimension.  All other
    # non-timing fields are part of the physical/profile contract, including TP,
    # model metadata, sequence shapes, and partition markers.
    fields = sorted(
        name
        for name in row
        if name != "row_id"
        and not str(name).startswith("time_stats.")
        and name not in {"latency", "latency_ms"}
    )
    return (
        partition,
        tuple((name, _freeze(row[name])) for name in fields),
    )


def _sort_identity(identity: tuple[Any, ...]) -> tuple[Any, ...]:
    partition, fields = identity
    return (
        {"standard": 0, "mixed": 1, "true_mixed": 2}[partition],
        tuple((name, _natural_sort_atom(value)) for name, value in fields),
    )


def _required_row_field(
    row: Mapping[str, Any],
    field: str,
    *,
    partition: str,
) -> Any:
    if field not in row or _is_missing(row[field]):
        raise ValueError(
            f"{partition} attention workload tuple is missing field {field!r}."
        )
    return row[field]


def _workload_integer(value: Any, *, field: str, partition: str) -> int:
    value = _jsonable(value)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not float(value).is_integer()
    ):
        raise ValueError(
            f"{partition} attention workload field {field!r} must be an integer, "
            f"got {value!r}."
        )
    return int(value)


def _workload_bool(value: Any, *, field: str, partition: str) -> bool:
    value = _jsonable(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(
        f"{partition} attention workload field {field!r} must be boolean, "
        f"got {value!r}."
    )


def _workload_integer_sequence(
    value: Any,
    *,
    field: str,
    partition: str,
) -> tuple[int, ...]:
    value = _jsonable(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{partition} attention workload field {field!r} must be a "
                f"JSON integer list, got {value!r}."
            ) from exc
    if not isinstance(value, list):
        raise ValueError(
            f"{partition} attention workload field {field!r} must be an integer "
            f"list, got {value!r}."
        )
    return tuple(
        _workload_integer(item, field=field, partition=partition)
        for item in value
    )


def _partition_from_row(row: Mapping[str, Any]) -> str:
    mixed = _workload_bool(
        _required_row_field(row, "is_mixed_batch", partition="attention"),
        field="is_mixed_batch",
        partition="attention",
    )
    true_mixed = _workload_bool(
        _required_row_field(row, "is_true_mixed_batch", partition="attention"),
        field="is_true_mixed_batch",
        partition="attention",
    )
    if mixed and true_mixed:
        raise ValueError(
            "Attention workload row cannot be both mixed and true_mixed."
        )
    if true_mixed:
        return "true_mixed"
    if mixed:
        return "mixed"
    return "standard"


def _workload_tuple_identity(
    row: Mapping[str, Any],
    partition: str,
) -> tuple[Any, ...]:
    """Return only the requested shape, excluding execution/provenance context."""

    if partition == "standard":
        return (
            "standard",
            _workload_integer(
                _required_row_field(row, "prefill_chunk_size", partition=partition),
                field="prefill_chunk_size",
                partition=partition,
            ),
            _workload_integer(
                _required_row_field(row, "kv_cache_size", partition=partition),
                field="kv_cache_size",
                partition=partition,
            ),
            _workload_integer(
                _required_row_field(row, "batch_size", partition=partition),
                field="batch_size",
                partition=partition,
            ),
            _workload_bool(
                _required_row_field(row, "is_prefill", partition=partition),
                field="is_prefill",
                partition=partition,
            ),
        )
    if partition == "mixed":
        mode = _jsonable(_required_row_field(row, "mode", partition=partition))
        if not isinstance(mode, str) or not mode:
            raise ValueError(
                f"{partition} attention workload field 'mode' must be a "
                f"non-empty string, got {mode!r}."
            )
        return (
            "mixed_prefill",
            _workload_integer_sequence(
                _required_row_field(row, "seq_lens", partition=partition),
                field="seq_lens",
                partition=partition,
            ),
            _workload_integer(
                _required_row_field(row, "kv_cache_size", partition=partition),
                field="kv_cache_size",
                partition=partition,
            ),
            mode,
        )
    if partition == "true_mixed":
        return (
            "true_mixed",
            _workload_integer_sequence(
                _required_row_field(row, "prefill_seq_lens", partition=partition),
                field="prefill_seq_lens",
                partition=partition,
            ),
            _workload_integer_sequence(
                _required_row_field(
                    row, "prefill_kv_cache_sizes", partition=partition
                ),
                field="prefill_kv_cache_sizes",
                partition=partition,
            ),
            _workload_integer_sequence(
                _required_row_field(row, "decode_kv_cache_sizes", partition=partition),
                field="decode_kv_cache_sizes",
                partition=partition,
            ),
        )
    raise ValueError(f"Unsupported attention partition {partition!r}.")


def _workload_tuple_identities(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [
        _workload_tuple_identity(row, _partition_from_row(row))
        for row in frame.to_dict(orient="records")
    ]


def _requested_tuple_digest(identities: list[tuple[Any, ...]]) -> str:
    """Hash a canonical workload-tuple multiset.

    Ordering is intentionally ignored, while duplicate requested shapes remain
    represented.  Model, TP, measurement, allocation, and other execution
    context belong to the surrounding provenance/config contract rather than
    this shape-only digest.
    """

    canonical_identities = sorted(
        (_jsonable(identity) for identity in identities),
        key=_canonical_json,
    )
    payload = {
        "schema": ATTENTION_REQUESTED_TUPLE_SCHEMA,
        "tuples": canonical_identities,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _structural_row_digest(identities: list[tuple[Any, ...]]) -> str:
    """Hash imported attention rows whose workload schema is not dense-native."""

    canonical_identities = sorted(
        (_jsonable(identity) for identity in identities),
        key=_canonical_json,
    )
    payload = {
        "schema": ATTENTION_IMPORTED_ROW_SCHEMA,
        "rows": canonical_identities,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _imported_row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Build a schema-independent identity for an imported attention row.

    vLLM MLA imports expose dynamic token/context fields rather than the native
    ``prefill_chunk_size``/``kv_cache_size`` tuple.  Their provenance must bind
    the complete non-timing row, without requiring native partition markers.
    """

    fields = sorted(
        name
        for name in row
        if name != "row_id"
        and not str(name).startswith("time_stats.")
        and name not in {"latency", "latency_ms"}
    )
    return (
        "imported",
        tuple((name, _freeze(row[name])) for name in fields),
    )


def _imported_row_identities(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    return [
        _imported_row_identity(row)
        for row in frame.to_dict(orient="records")
    ]


def _is_imported_row_schema(frame: pd.DataFrame) -> bool:
    """Return whether every row explicitly declares the MLA import schema."""

    marker = "is_mla_profile_import"
    if marker not in frame.columns:
        return False
    values = frame[marker].tolist()
    if not values:
        return False
    normalized = [
        _workload_bool(value, field=marker, partition="imported")
        for value in values
    ]
    if not any(normalized):
        return False
    if not all(normalized):
        raise ValueError(
            "Attention provenance cannot mix imported MLA rows with native rows."
        )
    return True


def _resolve_requested_tuple_binding(
    frame: pd.DataFrame,
) -> tuple[str, list[tuple[Any, ...]], str]:
    """Select an explicit digest schema for native or imported rows.

    Native attention rows have the four workload fields used by the runtime
    tuple contract.  Imported MLA rows intentionally have a different shape
    schema (batch token counts and runtime metadata), so treating them as
    dense tuples would be false provenance.  They use a separately named full
    structural-row multiset digest instead.  A partially present dense schema
    fails fast rather than silently switching semantics.
    """

    # Imported MLA rows also contain ``batch_size`` and ``is_prefill``.  Check
    # their explicit marker before looking for the native tuple subset.
    if _is_imported_row_schema(frame):
        identities = _imported_row_identities(frame)
        return ATTENTION_IMPORTED_ROW_SCHEMA, identities, _structural_row_digest(
            identities
        )

    partition_fields = {"is_mixed_batch", "is_true_mixed_batch"}
    present_partition = partition_fields.intersection(frame.columns)
    if present_partition:
        missing_partition = partition_fields - set(frame.columns)
        if missing_partition:
            raise ValueError(
                "Attention native partition schema is incomplete; missing fields: "
                f"{sorted(missing_partition)!r}."
            )
        identities = _workload_tuple_identities(frame)
        return ATTENTION_REQUESTED_TUPLE_SCHEMA, identities, _requested_tuple_digest(
            identities
        )

    dense_fields = {
        "prefill_chunk_size",
        "kv_cache_size",
        "batch_size",
        "is_prefill",
    }
    present_dense = dense_fields.intersection(frame.columns)
    if present_dense:
        missing_dense = dense_fields - set(frame.columns)
        if missing_dense:
            raise ValueError(
                "Attention native workload schema is incomplete; missing fields: "
                f"{sorted(missing_dense)!r}."
            )
        identities = _workload_tuple_identities(frame)
        return ATTENTION_REQUESTED_TUPLE_SCHEMA, identities, _requested_tuple_digest(
            identities
        )

    raise ValueError(
        "Attention provenance cannot infer a native workload schema: missing "
        "the complete native tuple or explicit is_mla_profile_import marker."
    )


def _normalize_dataframe_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Make object/list columns stable across equivalent worker outputs."""

    result = frame.copy()
    for column in result.columns:
        if column in {"is_mixed_batch", "is_true_mixed_batch"}:
            result[column] = result[column].astype(bool)
            continue
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: (
                    _canonical_json(value)
                    if isinstance(value, (list, tuple, dict))
                    else (None if _is_missing(value) else value)
                )
            )
    return result


def _stable_union(
    standard_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    true_mixed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[Any, ...]]]:
    partitions = (
        ("standard", _normalize_partition(standard_df, "standard")),
        ("mixed", _normalize_partition(mixed_df, "mixed")),
        ("true_mixed", _normalize_partition(true_mixed_df, "true_mixed")),
    )
    records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    for partition, frame in partitions:
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            row_dict = row.to_dict()
            identity = _row_identity(row_dict, partition)
            records.append((partition, identity, row_dict))

    identities = [identity for _partition, identity, _row in records]
    seen: set[tuple[Any, ...]] = set()
    duplicates: list[tuple[Any, ...]] = []
    for identity in identities:
        if identity in seen:
            duplicates.append(identity)
        seen.add(identity)
    if duplicates:
        partitions_seen = sorted({str(identity[0]) for identity in duplicates})
        raise ValueError(
            "duplicate attention profiling structural key(s) in partition "
            f"{', '.join(partitions_seen)}: {duplicates[:3]!r}."
        )

    records.sort(key=lambda item: _sort_identity(item[1]))
    if not records:
        return pd.DataFrame(), []

    # A sorted column union makes output independent of input dataframe column
    # ordering while retaining familiar structural/marker columns up front.
    all_columns = sorted(
        {column for _partition, _identity, row in records for column in row},
        key=lambda column: (str(column)),
    )
    preferred = [
        "num_tensor_parallel_workers",
        "prefill_chunk_size",
        "kv_cache_size",
        "batch_size",
        "seq_lens",
        "prefill_seq_lens",
        "prefill_kv_cache_sizes",
        "decode_kv_cache_sizes",
        "is_prefill",
        "is_mixed_batch",
        "is_true_mixed_batch",
    ]
    ordered_columns = [
        column for column in preferred if column in all_columns
    ] + [column for column in all_columns if column not in preferred]
    union = pd.DataFrame(
        [{column: row.get(column, None) for column in ordered_columns} for _, _, row in records],
        columns=ordered_columns,
    )
    union = _normalize_dataframe_values(union)
    return union, [identity for _partition, identity, _row in records]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_attention_run_id(run_id: str) -> str:
    """Return a safe single-component run identifier or fail fast."""

    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).name != run_id
    ):
        raise ValueError(
            "run_id must be a non-empty path-safe identifier and cannot be "
            "'.', '..', or contain path separators."
        )
    return run_id


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.resolve()
    second_resolved = second.resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _validate_publication_file_targets(
    *,
    context: str,
    targets: Mapping[str, Path],
) -> None:
    items = list(targets.items())
    for name, path in items:
        if path.exists() and path.is_dir():
            raise ValueError(
                f"{context} target {name!r} must be a file path, not a "
                f"directory: {path}."
            )
    for index, (first_name, first_path) in enumerate(items):
        for second_name, second_path in items[index + 1 :]:
            if _paths_overlap(first_path, second_path):
                raise ValueError(
                    f"{context} targets must be distinct and non-overlapping: "
                    f"{first_name}={first_path}, {second_name}={second_path}."
                )


def _attention_run_bound_artifact_paths(sidecar_path: Path) -> set[Path]:
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid attention merge source sidecar: {sidecar_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Attention merge source sidecar must contain a JSON object: "
            f"{sidecar_path}."
        )
    paths: set[Path] = set()
    for field in _ATTENTION_RUN_BOUND_PATH_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Attention merge source sidecar field {field!r} must be a "
                "non-empty path."
            )
        paths.add(Path(value).resolve())
    return paths


def _validate_positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(
            f"{name} must be a positive integer (numeric), got {value!r}."
        )
    return int(value)


def _normalize_tensor_parallel_value(
    value: Any,
    *,
    field: str,
    allow_numeric_string: bool,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    if isinstance(value, Integral):
        normalized = int(value)
    elif allow_numeric_string and isinstance(value, str):
        stripped = value.strip()
        try:
            numeric = float(stripped)
        except ValueError as exc:
            raise ValueError(
                f"{field} must be a positive integer, got {value!r}."
            ) from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{field} must be a positive integer, got {value!r}.")
        normalized = int(numeric)
    else:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    if normalized <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}.")
    return normalized


def _normalize_declared_tensor_parallel_values(
    payload: Mapping[str, Any],
    *,
    required: bool,
) -> tuple[int, ...] | None:
    has_sizes = (
        "tensor_parallel_sizes" in payload
        and payload.get("tensor_parallel_sizes") is not None
    )
    has_size = (
        "tensor_parallel_size" in payload
        and payload.get("tensor_parallel_size") is not None
    )
    if has_sizes and has_size:
        raise ValueError(
            "Use tensor_parallel_sizes or tensor_parallel_size, not both."
        )
    if not has_sizes and not has_size:
        if required:
            raise ValueError(
                "Native attention provenance requires tensor_parallel_sizes "
                "or tensor_parallel_size."
            )
        return None
    if has_size:
        return (
            _normalize_tensor_parallel_value(
                payload["tensor_parallel_size"],
                field="tensor_parallel_size",
                allow_numeric_string=False,
            ),
        )

    raw_sizes = payload["tensor_parallel_sizes"]
    if (
        not isinstance(raw_sizes, Sequence)
        or isinstance(raw_sizes, (str, bytes))
        or not raw_sizes
    ):
        raise ValueError("tensor_parallel_sizes must be a non-empty sequence.")
    normalized = tuple(
        _normalize_tensor_parallel_value(
            value,
            field=f"tensor_parallel_sizes[{index}]",
            allow_numeric_string=False,
        )
        for index, value in enumerate(raw_sizes)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"tensor_parallel_sizes contains duplicate values: {normalized!r}."
        )
    return normalized


def _normalize_allocation_tensor_parallel_keys(
    allocation_by_tp: Mapping[Any, Any],
) -> tuple[tuple[int, Any], ...]:
    normalized: list[tuple[int, Any]] = []
    seen: set[int] = set()
    for raw_key, record in allocation_by_tp.items():
        tp = _normalize_tensor_parallel_value(
            raw_key,
            field=f"allocation_by_tp key {raw_key!r}",
            allow_numeric_string=True,
        )
        if tp in seen:
            raise ValueError(
                "allocation_by_tp contains duplicate normalized tensor-parallel "
                f"keys for TP={tp}."
            )
        seen.add(tp)
        normalized.append((tp, record))
    return tuple(normalized)


def _observed_csv_tensor_parallel_values(csv_path: Path) -> tuple[int, ...]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if "num_tensor_parallel_workers" not in fieldnames:
            raise ValueError(
                "Native attention CSV is missing "
                "'num_tensor_parallel_workers'."
            )
        values: set[int] = set()
        for row_index, row in enumerate(reader, start=2):
            values.add(
                _normalize_tensor_parallel_value(
                    row.get("num_tensor_parallel_workers"),
                    field=(
                        "CSV num_tensor_parallel_workers "
                        f"at line {row_index}"
                    ),
                    allow_numeric_string=True,
                )
            )
    if not values:
        raise ValueError("Native attention CSV has no tensor-parallel rows.")
    return tuple(sorted(values))


def _observed_frame_tensor_parallel_values(
    frame: pd.DataFrame,
) -> tuple[int, ...]:
    if "num_tensor_parallel_workers" not in frame.columns:
        raise ValueError(
            "Native attention frame is missing "
            "'num_tensor_parallel_workers'."
        )
    values = {
        _normalize_tensor_parallel_value(
            value,
            field="frame num_tensor_parallel_workers",
            allow_numeric_string=True,
        )
        for value in frame["num_tensor_parallel_workers"].tolist()
    }
    if not values:
        raise ValueError("Native attention frame has no tensor-parallel rows.")
    return tuple(sorted(values))


def _validate_native_tensor_parallel_values(
    *,
    observed: tuple[int, ...],
    payload: Mapping[str, Any],
) -> None:
    if payload.get("is_native_profile_allocation", True) is not True:
        return
    declared = _normalize_declared_tensor_parallel_values(
        payload,
        required=True,
    )
    if declared is None:
        raise AssertionError("Native tensor-parallel declaration was not normalized.")
    allocation_by_tp = payload.get("allocation_by_tp")
    if allocation_by_tp is None:
        if len(declared) != 1:
            raise ValueError(
                "Native multi-TP attention provenance requires allocation_by_tp."
            )
        allocation = declared
    else:
        allocation = tuple(
            sorted(
                tp
                for tp, _record in _normalize_allocation_tensor_parallel_keys(
                    allocation_by_tp
                )
            )
        )
    normalized_declared = tuple(sorted(declared))
    if observed != normalized_declared or observed != allocation:
        raise ValueError(
            "Native attention tensor-parallel binding mismatch: "
            f"csv={list(observed)}, declared={list(normalized_declared)}, "
            f"allocation_by_tp={list(allocation)}."
        )


def _validate_native_tensor_parallel_binding(
    *,
    csv_path: Path,
    payload: Mapping[str, Any],
) -> None:
    if payload.get("is_native_profile_allocation", True) is not True:
        return
    _validate_native_tensor_parallel_values(
        observed=_observed_csv_tensor_parallel_values(csv_path),
        payload=payload,
    )


def _validate_native_tensor_parallel_frame_binding(
    *,
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
) -> None:
    if payload.get("is_native_profile_allocation", True) is not True:
        return
    _validate_native_tensor_parallel_values(
        observed=_observed_frame_tensor_parallel_values(frame),
        payload=payload,
    )


def _per_tp_column_max_allocations(
    frame: pd.DataFrame,
) -> dict[str, dict[str, int | None]]:
    missing_columns = sorted(
        set(_ALLOCATION_BY_TP_COLUMN_FIELDS) - set(frame.columns)
    )
    if missing_columns:
        raise ValueError(
            "Native attention allocation columns are missing from CSV: "
            f"{missing_columns!r}."
        )

    allocations: dict[str, dict[str, int | None]] = {}
    tp_values = pd.to_numeric(
        frame["num_tensor_parallel_workers"],
        errors="raise",
    )
    for tp_size in _observed_frame_tensor_parallel_values(frame):
        tp_rows = frame[tp_values == tp_size]
        record: dict[str, int | None] = {}
        for field in _ALLOCATION_BY_TP_COLUMN_FIELDS:
            numeric = pd.to_numeric(tp_rows[field], errors="raise").dropna()
            if numeric.empty:
                if field == "requested_max_num_blocks":
                    record[field] = None
                    continue
                raise ValueError(
                    "Native attention allocation column has no value for "
                    f"TP={tp_size}, field={field!r}."
                )
            values: list[int] = []
            for raw_value in numeric.tolist():
                numeric_value = float(raw_value)
                if (
                    not math.isfinite(numeric_value)
                    or not numeric_value.is_integer()
                    or int(numeric_value) <= 0
                ):
                    raise ValueError(
                        "Native attention allocation values must be positive "
                        f"integers: TP={tp_size}, field={field!r}, "
                        f"value={raw_value!r}."
                    )
                values.append(int(numeric_value))
            record[field] = max(values)
        allocations[str(tp_size)] = record
    return allocations


def _validate_allocation_by_tp_frame_binding(
    *,
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
) -> None:
    allocation_by_tp = payload.get("allocation_by_tp")
    if allocation_by_tp is None:
        return
    if payload.get("is_native_profile_allocation", True) is not True:
        return

    observed = _per_tp_column_max_allocations(frame)
    expected = {
        str(tp): dict(record)
        for tp, record in _normalize_allocation_tensor_parallel_keys(
            allocation_by_tp
        )
    }
    for tp_key in sorted(expected, key=int):
        for field in _ALLOCATION_BY_TP_COLUMN_FIELDS:
            sidecar_value = expected[tp_key].get(field)
            csv_value = observed[tp_key][field]
            if sidecar_value != csv_value:
                raise ValueError(
                    "allocation_by_tp does not match CSV per-TP column maxima: "
                    f"TP={tp_key}, field={field!r}, "
                    f"sidecar={sidecar_value!r}, csv_max={csv_value!r}."
                )


def _validate_provenance_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("attention provenance payload must be a mapping.")
    raw_payload = {str(key): value for key, value in payload.items()}
    raw_native = raw_payload.get("is_native_profile_allocation", True)
    if not isinstance(raw_native, bool):
        raise ValueError(
            "is_native_profile_allocation must be a boolean, "
            f"got {raw_native!r}."
        )
    # Validate numeric provenance before JSON normalization.  Converting an
    # integral float (or a string sentinel) first would silently erase the
    # distinction this contract is meant to protect.
    allocation_by_tp = raw_payload.get("allocation_by_tp")
    normalized_allocation_by_tp: dict[str, Any] | None = None
    if allocation_by_tp is not None:
        semantics = raw_payload.get("allocation_by_tp_semantics")
        if semantics != ATTENTION_ALLOCATION_BY_TP_SEMANTICS:
            raise ValueError(
                "allocation_by_tp_semantics must be "
                f"{ATTENTION_ALLOCATION_BY_TP_SEMANTICS!r} when "
                "allocation_by_tp is present."
            )
        if not isinstance(allocation_by_tp, Mapping) or not allocation_by_tp:
            raise ValueError("allocation_by_tp must be a non-empty mapping.")
        normalized_allocation_by_tp = {}
        for tp_value, record in _normalize_allocation_tensor_parallel_keys(
            allocation_by_tp
        ):
            tp_key = str(tp_value)
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"allocation_by_tp[{tp_key!r}] must be a mapping."
                )
            missing_fields = sorted(
                set(_ALLOCATION_BY_TP_COLUMN_FIELDS) - set(record)
            )
            if missing_fields:
                raise ValueError(
                    f"allocation_by_tp[{tp_key!r}] is incomplete; "
                    f"missing={missing_fields!r}."
                )
            for field in _POSITIVE_BLOCK_FIELDS:
                _validate_positive_integer(
                    f"allocation_by_tp[{tp_key!r}].{field}", record.get(field)
                )
            requested_value = record.get("requested_max_num_blocks")
            if requested_value is not None:
                _validate_positive_integer(
                    f"allocation_by_tp[{tp_key!r}].requested_max_num_blocks",
                    requested_value,
                )
            for field in (
                "allocated_max_num_blocks",
                "allocated_kv_token_capacity",
                "block_size",
            ):
                _validate_positive_integer(
                    f"allocation_by_tp[{tp_key!r}].{field}",
                    record.get(field),
                )
            physical_value = int(record["physical_max_num_blocks"])
            selected_value = int(record["selected_max_num_blocks"])
            required_value = int(record["required_max_num_blocks"])
            allocated_value = int(record["allocated_max_num_blocks"])
            capacity_value = int(record["allocated_kv_token_capacity"])
            block_size_value = int(record["block_size"])
            if selected_value > physical_value:
                raise ValueError(
                    f"allocation_by_tp[{tp_key!r}].selected_max_num_blocks "
                    "exceeds physical_max_num_blocks: "
                    f"selected={selected_value}, physical={physical_value}."
                )
            if selected_value < required_value:
                raise ValueError(
                    f"allocation_by_tp[{tp_key!r}].selected_max_num_blocks "
                    "cannot cover required_max_num_blocks: "
                    f"selected={selected_value}, required={required_value}."
                )
            if requested_value is not None and int(requested_value) != selected_value:
                raise ValueError(
                    f"allocation_by_tp[{tp_key!r}].selected_max_num_blocks must "
                    "equal requested_max_num_blocks when an explicit allocation "
                    f"request is recorded: requested={int(requested_value)}, "
                    f"selected={selected_value}."
                )
            if allocated_value != selected_value:
                raise ValueError(
                    f"allocation_by_tp[{tp_key!r}].allocated_max_num_blocks "
                    "must equal selected_max_num_blocks: "
                    f"allocated={allocated_value}, selected={selected_value}."
                )
            if capacity_value != allocated_value * block_size_value:
                raise ValueError(
                    f"allocation_by_tp[{tp_key!r}].allocated_kv_token_capacity "
                    "must equal allocated_max_num_blocks * block_size: "
                    f"capacity={capacity_value}, allocated={allocated_value}, "
                    f"block_size={block_size_value}."
                )
            normalized_allocation_by_tp[tp_key] = _jsonable(record)
    elif raw_payload.get("allocation_by_tp_semantics") is not None:
        raise ValueError(
            "allocation_by_tp_semantics is valid only when allocation_by_tp "
            "is present."
        )
    declared_tp_values = _normalize_declared_tensor_parallel_values(
        raw_payload,
        required=raw_native,
    )
    if (
        raw_native
        and normalized_allocation_by_tp is not None
        and set(normalized_allocation_by_tp)
        != {str(value) for value in declared_tp_values or ()}
    ):
        raise ValueError(
            "Native attention tensor-parallel binding mismatch between "
            f"declared values={list(declared_tp_values or ())} and "
            "allocation_by_tp keys="
            f"{sorted(normalized_allocation_by_tp, key=int)}."
        )
    for field in _POSITIVE_BLOCK_FIELDS:
        if field in raw_payload and raw_payload[field] is not None:
            _validate_positive_integer(field, raw_payload[field])
        elif raw_native and allocation_by_tp is None:
            raise ValueError(
                f"{field} must be a positive integer (numeric) for a native profile."
            )
    if "requested_max_num_blocks" in raw_payload and raw_payload["requested_max_num_blocks"] is not None:
        _validate_positive_integer(
            "requested_max_num_blocks", raw_payload["requested_max_num_blocks"]
        )
    normalized = {str(key): _jsonable(value) for key, value in raw_payload.items()}
    if declared_tp_values is not None:
        if "tensor_parallel_sizes" in raw_payload:
            normalized["tensor_parallel_sizes"] = list(declared_tp_values)
        else:
            normalized["tensor_parallel_size"] = declared_tp_values[0]
    if normalized_allocation_by_tp is not None:
        normalized["allocation_by_tp"] = normalized_allocation_by_tp
    if allocation_by_tp is None and raw_native:
        for field in _POSITIVE_BLOCK_FIELDS:
            normalized[field] = _validate_positive_integer(field, normalized.get(field))
    else:
        # Multi-TP artifacts carry per-TP allocation facts.  Scalar fields are
        # optional in that case and must not be synthesized from a misleading
        # min/max aggregate.
        for field in _POSITIVE_BLOCK_FIELDS:
            if normalized.get(field) is not None:
                normalized[field] = _validate_positive_integer(field, normalized[field])
    requested = normalized.get("requested_max_num_blocks")
    if allocation_by_tp is None and requested is not None:
        normalized["requested_max_num_blocks"] = _validate_positive_integer(
            "requested_max_num_blocks", requested
        )
        if normalized["requested_max_num_blocks"] != normalized["selected_max_num_blocks"]:
            raise ValueError(
                "selected_max_num_blocks must equal requested_max_num_blocks when "
                "an explicit allocation request is recorded."
            )
    elif requested is None:
        normalized["requested_max_num_blocks"] = None
    if (
        allocation_by_tp is None
        and raw_native
        and normalized["selected_max_num_blocks"] > normalized["physical_max_num_blocks"]
    ):
        raise ValueError(
            "selected_max_num_blocks exceeds physical_max_num_blocks: "
            f"selected={normalized['selected_max_num_blocks']}, "
            f"physical={normalized['physical_max_num_blocks']}."
        )
    if (
        allocation_by_tp is None
        and raw_native
        and normalized["selected_max_num_blocks"] < normalized["required_max_num_blocks"]
    ):
        raise ValueError(
            "selected_max_num_blocks cannot cover required_max_num_blocks: "
            f"selected={normalized['selected_max_num_blocks']}, "
            f"required={normalized['required_max_num_blocks']}."
        )

    for field in ("backend_workspace_reservation_bytes",):
        if field in normalized:
            value = normalized[field]
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
                raise ValueError(f"{field} must be a non-negative integer, got {value!r}.")
            normalized[field] = int(value)
    if "block_size" in normalized:
        normalized["block_size"] = _validate_positive_integer(
            "block_size", normalized["block_size"]
        )
        if normalized.get("selected_max_num_blocks") is not None:
            normalized.setdefault(
                "allocated_kv_token_capacity",
                normalized["selected_max_num_blocks"] * normalized["block_size"],
            )
    if normalized.get("selected_max_num_blocks") is not None:
        normalized.setdefault(
            "allocated_max_num_blocks", normalized["selected_max_num_blocks"]
        )
    for field in ("run_id", "model", "device", "measurement_type"):
        if field in normalized and (
            not isinstance(normalized[field], str) or not normalized[field].strip()
        ):
            raise ValueError(f"{field} must be a non-empty string.")
    return normalized


def _config_digest(payload: Mapping[str, Any]) -> str:
    config = {
        key: value
        for key, value in payload.items()
        if key not in _GENERATED_FIELDS
    }
    return _sha256_bytes(_canonical_json(config).encode("utf-8"))


def _validate_attention_profile_identity_binding(
    *,
    frame: pd.DataFrame,
    payload: Mapping[str, Any],
) -> None:
    present_fields = [
        field for field in _ATTENTION_PROFILE_IDENTITY_FIELDS if field in payload
    ]
    if not present_fields:
        return
    if len(present_fields) != len(_ATTENTION_PROFILE_IDENTITY_FIELDS):
        missing_fields = [
            field
            for field in _ATTENTION_PROFILE_IDENTITY_FIELDS
            if field not in payload
        ]
        raise ValueError(
            "Attention profiling identity is incomplete; "
            f"missing={missing_fields!r}."
        )
    for field in _ATTENTION_PROFILE_IDENTITY_FIELDS:
        expected = payload[field]
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(
                f"Attention profiling identity field {field!r} must be non-empty."
            )
        if field not in frame.columns:
            raise ValueError(
                "Attention profiling identity column is missing from CSV: "
                f"{field!r}."
            )
        actual_values = sorted(
            {
                str(value)
                for value in frame[field].dropna().unique()
                if str(value).strip()
            }
        )
        if actual_values != [expected]:
            raise ValueError(
                "Attention profiling identity mismatch for "
                f"{field!r}: sidecar={expected!r}, csv={actual_values!r}."
            )


def _csv_row_count(csv_path: Path) -> int:
    try:
        return int(len(pd.read_csv(csv_path)))
    except Exception as exc:  # pragma: no cover - error context is the contract
        raise ValueError(f"Unable to read attention CSV for provenance: {csv_path}") from exc


def _validate_requested_tuple_binding(
    *,
    csv_path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Validate new workload-tuple digests while accepting unmarked legacy ones."""

    tuple_schema = payload.get("requested_tuple_schema")
    if tuple_schema is None:
        # Schema-v1 sidecars written before the workload/full-identity split
        # used this field for a full row identity.  Keep those readable, but do
        # not silently assign the new semantics to them.
        return
    if tuple_schema not in {
        ATTENTION_REQUESTED_TUPLE_SCHEMA,
        ATTENTION_IMPORTED_ROW_SCHEMA,
    }:
        raise ValueError(
            "Unsupported attention requested_tuple_schema: "
            f"{tuple_schema!r}."
        )
    expected_digest = payload.get("requested_tuple_digest")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError(
            "attention requested_tuple_digest must be a 64-character SHA-256 "
            f"hex digest, got {expected_digest!r}."
        )
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - error context is the contract
        raise ValueError(
            f"Unable to read attention CSV workload tuples: {csv_path}"
        ) from exc
    actual_schema, identities, actual_digest = _resolve_requested_tuple_binding(frame)
    if actual_schema != tuple_schema:
        raise ValueError(
            "attention requested_tuple_schema does not match CSV workload schema: "
            f"sidecar={tuple_schema!r}, csv={actual_schema!r}."
        )
    requested_row_count = payload.get("requested_row_count")
    if requested_row_count != len(identities):
        raise ValueError(
            "attention provenance requested_row_count mismatch: "
            f"expected={requested_row_count!r}, actual={len(identities)!r}."
        )
    if expected_digest != actual_digest:
        raise ValueError(
            "attention provenance requested_tuple_digest mismatch: "
            f"expected={expected_digest!r}, actual={actual_digest!r}."
        )


def write_attention_run_sidecar(
    *,
    csv_path: str | Path,
    sidecar_path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write a validated sidecar bound to one CSV's current bytes."""

    csv = Path(csv_path)
    sidecar = Path(sidecar_path)
    if not csv.is_file():
        raise FileNotFoundError(f"Attention CSV does not exist: {csv}")
    normalized = _validate_provenance_payload(payload)
    normalized.setdefault("schema_version", ATTENTION_PROVENANCE_SCHEMA_VERSION)
    if normalized["schema_version"] != ATTENTION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported attention provenance schema_version: "
            f"{normalized['schema_version']!r}."
        )
    frame = pd.read_csv(csv)
    _validate_native_tensor_parallel_frame_binding(
        frame=frame,
        payload=normalized,
    )
    _validate_allocation_by_tp_frame_binding(
        frame=frame,
        payload=normalized,
    )
    _validate_attention_profile_identity_binding(
        frame=frame,
        payload=normalized,
    )
    normalized["csv_sha256"] = _sha256_bytes(csv.read_bytes())
    normalized["row_count"] = _csv_row_count(csv)
    normalized.setdefault("requested_row_count", normalized["row_count"])
    _validate_requested_tuple_binding(csv_path=csv, payload=normalized)
    normalized["config_sha256"] = _config_digest(normalized)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(normalized, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sidecar


def validate_attention_run_sidecar(
    *,
    csv_path: str | Path,
    sidecar_path: str | Path,
) -> None:
    """Fail fast if a run sidecar no longer matches its CSV or schema."""

    csv = Path(csv_path)
    sidecar = Path(sidecar_path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Attention provenance sidecar does not exist: {sidecar}")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid attention provenance sidecar: {sidecar}") from exc
    normalized = _validate_provenance_payload(payload)
    if normalized.get("schema_version") != ATTENTION_PROVENANCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported attention provenance schema_version: "
            f"{normalized.get('schema_version')!r}."
        )
    if not csv.is_file():
        raise FileNotFoundError(f"Attention CSV does not exist: {csv}")
    frame = pd.read_csv(csv)
    _validate_native_tensor_parallel_frame_binding(
        frame=frame,
        payload=normalized,
    )
    _validate_allocation_by_tp_frame_binding(
        frame=frame,
        payload=normalized,
    )
    _validate_attention_profile_identity_binding(
        frame=frame,
        payload=normalized,
    )
    actual_sha = _sha256_bytes(csv.read_bytes())
    if payload.get("csv_sha256") != actual_sha:
        raise ValueError(
            "attention provenance csv_sha256 mismatch: "
            f"expected={payload.get('csv_sha256')!r}, actual={actual_sha!r}."
        )
    expected_rows = payload.get("row_count")
    actual_rows = _csv_row_count(csv)
    if expected_rows != actual_rows:
        raise ValueError(
            "attention provenance row_count mismatch: "
            f"expected={expected_rows!r}, actual={actual_rows!r}."
        )
    _validate_requested_tuple_binding(csv_path=csv, payload=normalized)
    actual_config_sha = _config_digest(payload)
    if payload.get("config_sha256") != actual_config_sha:
        raise ValueError(
            "attention provenance config_sha256 mismatch: "
            f"expected={payload.get('config_sha256')!r}, actual={actual_config_sha!r}."
        )
    _validate_attention_partition_parent_binding(
        partition_csv=csv,
        payload=normalized,
    )


def publish_attention_union_and_alias(
    *,
    output_dir: str | Path,
    standard_df: pd.DataFrame,
    mixed_df: pd.DataFrame,
    true_mixed_df: pd.DataFrame,
    run_id: str,
    canonical_name: str = "attention.csv",
    alias_name: str = "attention_combined.csv",
    provenance: Mapping[str, Any],
) -> dict[str, Path]:
    """Publish canonical union, compatibility alias, and a run sidecar."""

    run_id = validate_attention_run_id(run_id)
    for name, value in (("canonical_name", canonical_name), ("alias_name", alias_name)):
        if (
            not isinstance(value, str)
            or not value.strip()
            or Path(value).name != value
            or not value.endswith(".csv")
        ):
            raise ValueError(f"{name} must be a path-safe CSV filename, got {value!r}.")
    output = Path(output_dir)
    canonical = output / canonical_name
    alias = output / alias_name
    run_csv = output / "runs" / run_id / canonical_name
    sidecar = output / f"{Path(canonical_name).stem}.{run_id}.json"
    _validate_publication_file_targets(
        context="Attention union publication",
        targets={
            "canonical": canonical,
            "alias": alias,
            "run_csv": run_csv,
            "sidecar": sidecar,
        },
    )

    union, identities = _stable_union(standard_df, mixed_df, true_mixed_df)
    if union.empty:
        raise ValueError("Cannot publish an empty attention profiling union.")

    payload = dict(provenance)
    payload["run_id"] = run_id
    tuple_schema, workload_identities, workload_digest = (
        _resolve_requested_tuple_binding(union)
    )
    payload["requested_row_count"] = len(workload_identities)
    payload["artifact_csv"] = str(run_csv)
    payload["requested_tuple_schema"] = tuple_schema
    payload["requested_tuple_digest"] = workload_digest
    # Preserve the pre-split value under an explicit name.  It remains useful
    # for auditing the emitted full row structure, but is not a request-shape
    # identity and therefore must not occupy ``requested_tuple_digest``.
    payload["structural_identity_digest"] = _sha256_bytes(
        _canonical_json(identities).encode("utf-8")
    )
    normalized_payload = _validate_provenance_payload(payload)
    _validate_native_tensor_parallel_frame_binding(
        frame=union,
        payload=normalized_payload,
    )
    _validate_allocation_by_tp_frame_binding(
        frame=union,
        payload=normalized_payload,
    )
    _validate_attention_profile_identity_binding(
        frame=union,
        payload=normalized_payload,
    )

    output.mkdir(parents=True, exist_ok=True)
    # Explicit line terminator avoids platform-dependent bytes.
    run_csv.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(run_csv, index=False, lineterminator="\n")
    canonical.write_bytes(run_csv.read_bytes())
    alias.write_bytes(run_csv.read_bytes())

    write_attention_run_sidecar(
        csv_path=run_csv,
        sidecar_path=sidecar,
        payload=payload,
    )
    # Verify the alias immediately; this catches partial writes before callers
    # consume the output directory.
    if alias.read_bytes() != canonical.read_bytes():
        raise RuntimeError("attention_combined.csv is not byte-identical to attention.csv.")
    return {
        "canonical": canonical,
        "alias": alias,
        "run_csv": run_csv,
        "sidecar": sidecar,
    }


def _prepare_attention_partition_run_payload(
    *,
    source_sidecar_path: str | Path,
    partition_csv: str | Path,
    partition: str,
    expected_model: str | None = None,
    expected_measurement_type: str | None = None,
) -> dict[str, Any]:
    if partition not in {"standard", "mixed", "true_mixed"}:
        raise ValueError(
            "Attention partition must be one of standard, mixed, or true_mixed."
        )
    source_sidecar = Path(source_sidecar_path)
    partition_path = Path(partition_csv)
    if not source_sidecar.is_file():
        raise FileNotFoundError(
            f"Attention source run sidecar does not exist: {source_sidecar}"
        )
    try:
        source_payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid attention source run sidecar: {source_sidecar}"
        ) from exc
    if any(
        field in source_payload for field in _ATTENTION_PARTITION_PARENT_FIELDS
    ):
        raise ValueError(
            "Attention partition source must be a direct profiling run, not "
            "another derived partition."
        )
    source_csv_value = source_payload.get("artifact_csv")
    if not isinstance(source_csv_value, str) or not source_csv_value.strip():
        raise ValueError(
            "Attention source run sidecar must declare a non-empty artifact_csv."
        )
    source_csv = Path(source_csv_value)
    validate_attention_run_sidecar(
        csv_path=source_csv,
        sidecar_path=source_sidecar,
    )
    if source_payload.get("is_native_profile_allocation", True) is not True:
        raise ValueError(
            "Attention partition publication requires native allocation provenance."
        )
    if (
        expected_model is not None
        and source_payload.get("model") != expected_model
    ):
        raise ValueError(
            "Attention partition source model identity mismatch: "
            f"expected={expected_model!r}, "
            f"actual={source_payload.get('model')!r}."
        )
    if (
        expected_measurement_type is not None
        and source_payload.get("measurement_type")
        != expected_measurement_type
    ):
        raise ValueError(
            "Attention partition source measurement family mismatch: "
            f"expected={expected_measurement_type!r}, "
            f"actual={source_payload.get('measurement_type')!r}."
        )
    if not partition_path.is_file():
        raise FileNotFoundError(
            f"Attention partition CSV does not exist: {partition_path}"
        )

    source_frame = pd.read_csv(source_csv)
    partition_frame = pd.read_csv(partition_path)
    if partition_frame.empty:
        raise ValueError(
            f"Attention partition CSV is empty: {partition_path}"
        )
    source_schema, _source_identities, _source_digest = (
        _resolve_requested_tuple_binding(source_frame)
    )
    partition_schema, partition_identities, partition_digest = (
        _resolve_requested_tuple_binding(partition_frame)
    )
    if source_schema != partition_schema:
        raise ValueError(
            "Attention partition workload schema does not match the source run: "
            f"source={source_schema!r}, partition={partition_schema!r}."
        )
    _validate_attention_partition_complete_rows(
        source_csv=source_csv,
        partition_csv=partition_path,
    )

    partition_tps = _observed_csv_tensor_parallel_values(partition_path)
    derived_payload = {
        key: value
        for key, value in source_payload.items()
        if key
        not in {
            *_GENERATED_FIELDS,
            "requested_tuple_schema",
            "requested_tuple_digest",
            "structural_identity_digest",
        }
    }
    source_allocation = source_payload.get("allocation_by_tp")
    if source_allocation is not None:
        normalized_source_allocation = {
            str(tp): record
            for tp, record in _normalize_allocation_tensor_parallel_keys(
                source_allocation
            )
        }
        missing_allocation_tps = [
            tp for tp in partition_tps if str(tp) not in normalized_source_allocation
        ]
        if missing_allocation_tps:
            raise ValueError(
                "Attention partition TP values have no source allocation records: "
                f"{missing_allocation_tps}."
            )
        derived_payload.pop("tensor_parallel_size", None)
        derived_payload["tensor_parallel_sizes"] = list(partition_tps)
        derived_payload["allocation_by_tp"] = _per_tp_column_max_allocations(
            partition_frame
        )
    else:
        declared = _normalize_declared_tensor_parallel_values(
            source_payload,
            required=True,
        )
        if declared != partition_tps:
            raise ValueError(
                "Attention partition TP values do not match scalar source "
                f"allocation identity: partition={list(partition_tps)}, "
                f"source={list(declared or ())}."
            )

    partition_frames = {
        "standard": (partition_frame, pd.DataFrame(), pd.DataFrame()),
        "mixed": (pd.DataFrame(), partition_frame, pd.DataFrame()),
        "true_mixed": (pd.DataFrame(), pd.DataFrame(), partition_frame),
    }
    _union, structural_identities = _stable_union(*partition_frames[partition])
    derived_payload.update(
        {
            "artifact_csv": str(partition_path.resolve()),
            "partition": partition,
            "source_run_csv": str(source_csv.resolve()),
            "source_run_csv_sha256": _sha256_bytes(source_csv.read_bytes()),
            "source_run_sidecar": str(source_sidecar.resolve()),
            "source_run_sidecar_sha256": _sha256_bytes(
                source_sidecar.read_bytes()
            ),
            "requested_row_count": len(partition_identities),
            "requested_tuple_schema": partition_schema,
            "requested_tuple_digest": partition_digest,
            "structural_identity_digest": _sha256_bytes(
                _canonical_json(structural_identities).encode("utf-8")
            ),
        }
    )
    return derived_payload


def validate_attention_partition_source(
    *,
    source_sidecar_path: str | Path,
    partition_csv: str | Path,
    partition: str,
    expected_model: str | None = None,
    expected_measurement_type: str | None = None,
) -> None:
    """Validate one partition against its authenticated parent without writing."""

    _prepare_attention_partition_run_payload(
        source_sidecar_path=source_sidecar_path,
        partition_csv=partition_csv,
        partition=partition,
        expected_model=expected_model,
        expected_measurement_type=expected_measurement_type,
    )


def write_attention_partition_run_sidecar(
    *,
    source_sidecar_path: str | Path,
    partition_csv: str | Path,
    sidecar_path: str | Path,
    partition: str,
    expected_model: str | None = None,
    expected_measurement_type: str | None = None,
) -> Path:
    """Bind one emitted partition to its validated native profiling run."""

    partition_path = Path(partition_csv)
    derived_payload = _prepare_attention_partition_run_payload(
        source_sidecar_path=source_sidecar_path,
        partition_csv=partition_path,
        partition=partition,
        expected_model=expected_model,
        expected_measurement_type=expected_measurement_type,
    )
    written = write_attention_run_sidecar(
        csv_path=partition_path,
        sidecar_path=sidecar_path,
        payload=derived_payload,
    )
    validate_attention_run_sidecar(
        csv_path=partition_path,
        sidecar_path=written,
    )
    return written


def _read_csv_rows_as_strings(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Attention merge CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        if not fieldnames:
            raise ValueError(f"Attention merge CSV has no header: {path}")
        return fieldnames, [dict(row) for row in reader]


def _merge_csv_fieldnames(*fieldname_sets: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for fieldnames in fieldname_sets:
        for fieldname in fieldnames:
            if fieldname not in merged:
                merged.append(fieldname)
    return merged


def _normalized_csv_rows(
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> list[tuple[str, ...]]:
    return [
        tuple(str(row.get(fieldname, "")) for fieldname in fieldnames)
        for row in rows
    ]


def _normalized_csv_row_digest(
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> str:
    identities = sorted(_normalized_csv_rows(rows, fieldnames))
    payload = {
        "schema": ATTENTION_MERGE_ROW_IDENTITY_SCHEMA,
        "fieldnames": list(fieldnames),
        "rows": identities,
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _validate_attention_partition_complete_rows(
    *,
    source_csv: Path,
    partition_csv: Path,
) -> None:
    source_fieldnames, source_rows = _read_csv_rows_as_strings(source_csv)
    partition_fieldnames, partition_rows = _read_csv_rows_as_strings(
        partition_csv
    )
    fieldnames = _merge_csv_fieldnames(
        source_fieldnames,
        partition_fieldnames,
    )
    missing_rows = Counter(
        _normalized_csv_rows(partition_rows, fieldnames)
    ) - Counter(_normalized_csv_rows(source_rows, fieldnames))
    if missing_rows:
        raise ValueError(
            "Attention partition contains complete normalized rows not present "
            "in the validated source run: "
            f"count={sum(missing_rows.values())}."
        )


def _validate_attention_partition_parent_binding(
    *,
    partition_csv: Path,
    payload: Mapping[str, Any],
) -> None:
    present = {
        field for field in _ATTENTION_PARTITION_PARENT_FIELDS if field in payload
    }
    if not present:
        return
    missing = set(_ATTENTION_PARTITION_PARENT_FIELDS) - present
    if missing:
        raise ValueError(
            "Attention partition parent provenance is incomplete; missing "
            f"fields={sorted(missing)!r}."
        )
    if payload.get("partition") not in {"standard", "mixed", "true_mixed"}:
        raise ValueError(
            "Attention partition parent provenance has an invalid partition: "
            f"{payload.get('partition')!r}."
        )

    source_csv = Path(str(payload["source_run_csv"])).resolve()
    source_sidecar = Path(str(payload["source_run_sidecar"])).resolve()
    if not source_csv.is_file():
        raise FileNotFoundError(
            f"Attention partition parent CSV does not exist: {source_csv}"
        )
    if not source_sidecar.is_file():
        raise FileNotFoundError(
            "Attention partition parent sidecar does not exist: "
            f"{source_sidecar}"
        )
    actual_source_csv_sha = _sha256_bytes(source_csv.read_bytes())
    if payload.get("source_run_csv_sha256") != actual_source_csv_sha:
        raise ValueError(
            "Attention partition parent CSV sha256 mismatch: "
            f"expected={payload.get('source_run_csv_sha256')!r}, "
            f"actual={actual_source_csv_sha!r}."
        )
    actual_source_sidecar_sha = _sha256_bytes(source_sidecar.read_bytes())
    if (
        payload.get("source_run_sidecar_sha256")
        != actual_source_sidecar_sha
    ):
        raise ValueError(
            "Attention partition parent sidecar sha256 mismatch: "
            f"expected={payload.get('source_run_sidecar_sha256')!r}, "
            f"actual={actual_source_sidecar_sha!r}."
        )
    try:
        source_payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid attention partition parent sidecar: {source_sidecar}"
        ) from exc
    if any(
        field in source_payload for field in _ATTENTION_PARTITION_PARENT_FIELDS
    ):
        raise ValueError(
            "Attention partition parent must be a direct profiling run, not "
            "another derived partition."
        )
    source_artifact = source_payload.get("artifact_csv")
    if (
        not isinstance(source_artifact, str)
        or Path(source_artifact).resolve() != source_csv
    ):
        raise ValueError(
            "Attention partition parent sidecar artifact_csv does not match "
            f"source_run_csv: {source_artifact!r}."
        )
    for field in ("model", "device", "measurement_type"):
        if source_payload.get(field) != payload.get(field):
            raise ValueError(
                "Attention partition parent identity mismatch for "
                f"{field!r}: parent={source_payload.get(field)!r}, "
                f"partition={payload.get(field)!r}."
            )
    if source_payload.get("is_native_profile_allocation", True) is not True:
        raise ValueError(
            "Attention partition parent requires native allocation provenance."
        )
    validate_attention_run_sidecar(
        csv_path=source_csv,
        sidecar_path=source_sidecar,
    )
    _validate_attention_partition_complete_rows(
        source_csv=source_csv,
        partition_csv=partition_csv,
    )
    expected_payload = _prepare_attention_partition_run_payload(
        source_sidecar_path=source_sidecar,
        partition_csv=partition_csv,
        partition=str(payload["partition"]),
    )
    actual_artifact = payload.get("artifact_csv")
    if (
        not isinstance(actual_artifact, str)
        or Path(actual_artifact).resolve() != partition_csv.resolve()
    ):
        raise ValueError(
            "Attention partition inherited metadata mismatch for artifact_csv: "
            f"expected={str(partition_csv.resolve())!r}, "
            f"actual={actual_artifact!r}."
        )
    expected_config_sha = _config_digest(expected_payload)
    actual_config_sha = _config_digest(payload)
    if actual_config_sha != expected_config_sha:
        raise ValueError(
            "Attention partition inherited metadata does not match its live "
            "parent run: "
            f"expected_config_sha256={expected_config_sha!r}, "
            f"actual_config_sha256={actual_config_sha!r}."
        )


def _merge_config_digest(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("config_sha256", None)
    return _sha256_bytes(_canonical_json(normalized).encode("utf-8"))


def _source_allocation_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("is_native_profile_allocation", True) is not True:
        raise ValueError(
            "Attention merge source requires native allocation provenance."
        )
    scalar = {
        field: payload.get(field)
        for field in _BLOCK_FIELDS
        if field in payload and not _is_missing(payload.get(field))
    }
    if scalar and any(field not in scalar for field in _POSITIVE_BLOCK_FIELDS):
        raise ValueError(
            "Attention merge source has incomplete scalar allocation provenance."
        )
    allocation_by_tp = payload.get("allocation_by_tp")
    semantics = payload.get("allocation_by_tp_semantics")
    if (
        allocation_by_tp is not None
        and semantics != ATTENTION_ALLOCATION_BY_TP_SEMANTICS
    ):
        raise ValueError(
            "Attention merge source allocation_by_tp_semantics must be "
            f"{ATTENTION_ALLOCATION_BY_TP_SEMANTICS!r}."
        )
    return {
        "allocation_by_tp": allocation_by_tp,
        "allocation_by_tp_semantics": semantics,
        "scalar_allocation": scalar or None,
    }


def _expected_merged_row_identities(
    *,
    source_rows: Sequence[Sequence[Mapping[str, str]]],
    fieldnames: Sequence[str],
) -> Counter[tuple[str, ...]]:
    if len(source_rows) != 2:
        raise ValueError(
            "Attention merge provenance v1 requires exactly two ordered sources."
        )
    key_indexes = [
        index
        for index, fieldname in enumerate(fieldnames)
        if not str(fieldname).startswith("time_stats.")
    ]
    if not key_indexes:
        raise ValueError(
            "Attention merge provenance requires at least one non-time_stats key column."
        )

    base_identities = _normalized_csv_rows(source_rows[0], fieldnames)
    accepted = list(base_identities)
    accepted_identity_set = set(base_identities)
    accepted_by_key: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for identity in base_identities:
        key = tuple(identity[index] for index in key_indexes)
        accepted_by_key.setdefault(key, []).append(identity)

    for identity in _normalized_csv_rows(source_rows[1], fieldnames):
        if identity in accepted_identity_set:
            continue
        key = tuple(identity[index] for index in key_indexes)
        if accepted_by_key.get(key):
            raise ValueError(
                "Conflicting duplicate attention merge source row for normalized "
                f"key={key!r}."
            )
        accepted.append(identity)
        accepted_identity_set.add(identity)
        accepted_by_key.setdefault(key, []).append(identity)
    return Counter(accepted)


def _expected_merge_report(
    *,
    source_rows: Sequence[Sequence[Mapping[str, str]]],
    fieldnames: Sequence[str],
    merged_row_count: int,
) -> dict[str, Any]:
    key_columns = [
        str(fieldname)
        for fieldname in fieldnames
        if not str(fieldname).startswith("time_stats.")
    ]
    base_identities = _normalized_csv_rows(source_rows[0], fieldnames)
    base_identity_set = set(base_identities)
    accepted_identity_set = set(base_identities)
    duplicate_identical_count = 0
    supplement_duplicate_identical_count = 0
    for identity in _normalized_csv_rows(source_rows[1], fieldnames):
        if identity in accepted_identity_set:
            duplicate_identical_count += 1
            if identity not in base_identity_set:
                supplement_duplicate_identical_count += 1
            continue
        accepted_identity_set.add(identity)
    return {
        "base_row_count": len(source_rows[0]),
        "supplement_row_count": len(source_rows[1]),
        "merged_row_count": merged_row_count,
        "key_column_count": len(key_columns),
        "key_columns": key_columns,
        "duplicate_identical_count": duplicate_identical_count,
        "supplement_duplicate_identical_count": (
            supplement_duplicate_identical_count
        ),
    }


def _attention_merge_source_path(
    source: Mapping[str, Any],
    *,
    index: int,
    field: str,
) -> Path:
    raw_path = source.get(field)
    if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
        raise ValueError(
            f"Attention merge source {index} requires a non-empty {field} path."
        )
    return Path(raw_path).resolve()


def _validate_attention_merge_source_identity(
    *,
    label: str,
    payload: Mapping[str, Any],
) -> None:
    for field in _ATTENTION_MERGE_SOURCE_TEXT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Attention merge source {label!r} required field {field!r} "
                "must be a non-empty string."
            )
    for field in _ATTENTION_MERGE_SOURCE_DIGEST_FIELDS:
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError(
                f"Attention merge source {label!r} required field {field!r} "
                "must be a 64-character SHA-256 hex digest."
            )


def _prepare_attention_merge_output(
    *,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, str]], bytes, dict[str, Any]]:
    if len(sources) != 2:
        raise ValueError(
            "Attention merge provenance v1 requires exactly two ordered sources."
        )
    source_fieldnames: list[list[str]] = []
    source_rows: list[list[dict[str, str]]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise TypeError(f"Attention merge source {index} must be a mapping.")
        csv_path = _attention_merge_source_path(
            source,
            index=index,
            field="csv_path",
        )
        fields, rows = _read_csv_rows_as_strings(csv_path)
        source_fieldnames.append(fields)
        source_rows.append(rows)

    fieldnames = _merge_csv_fieldnames(*source_fieldnames)
    key_columns = [
        fieldname
        for fieldname in fieldnames
        if not fieldname.startswith("time_stats.")
    ]
    if not key_columns:
        raise ValueError(
            "Attention merge provenance requires at least one non-time_stats key column."
        )

    normalized_sources = [
        [
            {fieldname: str(row.get(fieldname, "")) for fieldname in fieldnames}
            for row in rows
        ]
        for rows in source_rows
    ]
    normalized_base_rows, normalized_supplement_rows = normalized_sources
    accepted_rows_by_key: dict[tuple[str, ...], list[dict[str, str]]] = {}
    accepted_row_identities: set[tuple[str, ...]] = set()
    base_row_identities: set[tuple[str, ...]] = set()
    for row in normalized_base_rows:
        identity = tuple(row[fieldname] for fieldname in fieldnames)
        key = tuple(row[fieldname] for fieldname in key_columns)
        accepted_rows_by_key.setdefault(key, []).append(row)
        accepted_row_identities.add(identity)
        base_row_identities.add(identity)

    supplement_rows_to_append: list[dict[str, str]] = []
    duplicate_identical_count = 0
    supplement_duplicate_identical_count = 0
    for row in normalized_supplement_rows:
        identity = tuple(row[fieldname] for fieldname in fieldnames)
        if identity in accepted_row_identities:
            duplicate_identical_count += 1
            if identity not in base_row_identities:
                supplement_duplicate_identical_count += 1
            continue
        key = tuple(row[fieldname] for fieldname in key_columns)
        if accepted_rows_by_key.get(key):
            raise ValueError(
                "Conflicting duplicate attention merge source row for normalized "
                f"key={key!r}."
            )
        supplement_rows_to_append.append(row)
        accepted_rows_by_key.setdefault(key, []).append(row)
        accepted_row_identities.add(identity)

    merged_rows = sorted(
        [*normalized_base_rows, *supplement_rows_to_append],
        key=lambda row: tuple(row[fieldname] for fieldname in key_columns),
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\r\n",
    )
    writer.writeheader()
    writer.writerows(merged_rows)
    output_bytes = buffer.getvalue().encode("utf-8")
    report = {
        "base_row_count": len(source_rows[0]),
        "supplement_row_count": len(source_rows[1]),
        "merged_row_count": len(merged_rows),
        "key_column_count": len(key_columns),
        "key_columns": key_columns,
        "duplicate_identical_count": duplicate_identical_count,
        "supplement_duplicate_identical_count": (
            supplement_duplicate_identical_count
        ),
    }
    return fieldnames, merged_rows, output_bytes, report


def _build_attention_merge_sidecar_payload(
    *,
    output_csv: Path,
    alias_csv: Path,
    sources: Sequence[Mapping[str, Any]],
    merge_report: Mapping[str, Any],
    output_fieldnames: Sequence[str] | None = None,
    output_rows: Sequence[Mapping[str, str]] | None = None,
    output_bytes: bytes | None = None,
    alias_bytes: bytes | None = None,
) -> dict[str, Any]:
    if len(sources) != 2:
        raise ValueError(
            "Attention merge provenance v1 requires exactly two ordered sources."
    )
    output_csv = output_csv.resolve()
    alias_csv = alias_csv.resolve()
    supplied_output = (
        output_fieldnames,
        output_rows,
        output_bytes,
        alias_bytes,
    )
    if all(value is None for value in supplied_output):
        output_fieldnames, output_rows = _read_csv_rows_as_strings(output_csv)
        if not alias_csv.is_file():
            raise FileNotFoundError(
                f"Attention merge compatibility alias does not exist: {alias_csv}"
            )
        output_bytes = output_csv.read_bytes()
        alias_bytes = alias_csv.read_bytes()
    elif any(value is None for value in supplied_output):
        raise ValueError(
            "In-memory attention merge validation requires fieldnames, rows, "
            "canonical bytes, and alias bytes together."
        )
    else:
        output_fieldnames = list(output_fieldnames)
        output_rows = [dict(row) for row in output_rows]
        output_bytes = bytes(output_bytes)
        alias_bytes = bytes(alias_bytes)
    if alias_bytes != output_bytes:
        raise ValueError(
            "Attention merge compatibility alias must be byte-identical to the "
            "merged canonical CSV."
        )

    normalized_sources: list[dict[str, Any]] = []
    source_rows: list[list[dict[str, str]]] = []
    source_fieldnames: list[list[str]] = []
    labels: set[str] = set()
    identity: dict[str, Any] | None = None
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise TypeError(f"Attention merge source {index} must be a mapping.")
        label = source.get("label")
        if (
            not isinstance(label, str)
            or not label.strip()
            or label in labels
        ):
            raise ValueError(
                f"Attention merge source labels must be unique and non-empty: {label!r}."
            )
        labels.add(label)
        csv_path = _attention_merge_source_path(
            source,
            index=index,
            field="csv_path",
        )
        sidecar_path = _attention_merge_source_path(
            source,
            index=index,
            field="sidecar_path",
        )
        try:
            sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid attention merge source sidecar: {sidecar_path}"
            ) from exc
        _validate_attention_merge_source_identity(
            label=label,
            payload=sidecar_payload,
        )
        validate_attention_run_sidecar(
            csv_path=csv_path,
            sidecar_path=sidecar_path,
        )
        current_identity = {
            "model": sidecar_payload.get("model"),
            "device": sidecar_payload.get("device"),
            "measurement_type": sidecar_payload.get("measurement_type"),
            **{
                field: sidecar_payload.get(field)
                for field in _ATTENTION_PROFILE_IDENTITY_FIELDS
            },
        }
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise ValueError(
                "Attention merge sources must share complete profiling identity: "
                f"expected={identity!r}, actual={current_identity!r}."
            )
        fields, rows = _read_csv_rows_as_strings(csv_path)
        source_fieldnames.append(fields)
        source_rows.append(rows)
        allocation = _source_allocation_binding(sidecar_payload)
        if (
            allocation["allocation_by_tp"] is None
            and allocation["scalar_allocation"] is None
        ):
            raise ValueError(
                f"Attention merge source {label!r} has no allocation provenance."
            )
        normalized_sources.append(
            {
                "label": label,
                "csv_path": str(csv_path),
                "csv_sha256": _sha256_bytes(csv_path.read_bytes()),
                "row_count": len(rows),
                "sidecar_path": str(sidecar_path),
                "sidecar_sha256": _sha256_bytes(sidecar_path.read_bytes()),
                "sidecar_schema_version": sidecar_payload.get("schema_version"),
                "run_id": sidecar_payload.get("run_id"),
                **{
                    field: sidecar_payload.get(field)
                    for field in _ATTENTION_PROFILE_IDENTITY_FIELDS
                },
                "requested_tuple_schema": sidecar_payload.get(
                    "requested_tuple_schema"
                ),
                "requested_tuple_digest": sidecar_payload.get(
                    "requested_tuple_digest"
                ),
                "structural_identity_digest": sidecar_payload.get(
                    "structural_identity_digest"
                ),
                **allocation,
            }
        )

    if identity is None:
        raise AssertionError("Attention merge source identity was not initialized.")
    merged_fieldnames = _merge_csv_fieldnames(*source_fieldnames)
    if output_fieldnames != merged_fieldnames:
        raise ValueError(
            "Attention merge output field order does not match the normalized source "
            f"union: expected={merged_fieldnames!r}, actual={output_fieldnames!r}."
        )
    expected_identities = _expected_merged_row_identities(
        source_rows=source_rows,
        fieldnames=merged_fieldnames,
    )
    actual_identities = Counter(
        _normalized_csv_rows(output_rows, merged_fieldnames)
    )
    if actual_identities != expected_identities:
        raise ValueError(
            "Attention merge output normalized row identities do not match the "
            "ordered source union."
        )

    measurement_values = {
        str(row.get("measurement_type", "")) for row in output_rows
    }
    if measurement_values != {identity["measurement_type"]}:
        raise ValueError(
            "Attention merge output measurement family mismatch: "
            f"expected={identity['measurement_type']!r}, "
            f"actual={sorted(measurement_values)!r}."
        )

    report = {
        field: merge_report.get(field)
        for field in (
            "base_row_count",
            "supplement_row_count",
            "merged_row_count",
            "key_column_count",
            "key_columns",
            "duplicate_identical_count",
            "supplement_duplicate_identical_count",
        )
    }
    expected_report = _expected_merge_report(
        source_rows=source_rows,
        fieldnames=merged_fieldnames,
        merged_row_count=len(output_rows),
    )
    for field, expected in expected_report.items():
        if report.get(field) != expected:
            raise ValueError(
                f"Attention merge report {field} mismatch: "
                f"expected={expected}, actual={report.get(field)!r}."
            )

    for source, rows in zip(normalized_sources, source_rows, strict=True):
        source["normalized_row_identity_digest"] = _normalized_csv_row_digest(
            rows,
            merged_fieldnames,
        )

    payload: dict[str, Any] = {
        "schema_version": ATTENTION_MERGE_PROVENANCE_SCHEMA,
        "row_identity_schema": ATTENTION_MERGE_ROW_IDENTITY_SCHEMA,
        "model": identity["model"],
        "device": identity["device"],
        "measurement_type": identity["measurement_type"],
        **{
            field: identity[field]
            for field in _ATTENTION_PROFILE_IDENTITY_FIELDS
        },
        "sources": normalized_sources,
        "merge": report,
        "output": {
            "csv_path": str(output_csv),
            "csv_sha256": _sha256_bytes(output_bytes),
            "row_count": len(output_rows),
            "fieldnames": output_fieldnames,
            "normalized_row_identity_digest": _normalized_csv_row_digest(
                output_rows,
                output_fieldnames,
            ),
            "alias_csv": str(alias_csv),
            "alias_csv_sha256": _sha256_bytes(alias_bytes),
        },
    }
    payload["config_sha256"] = _merge_config_digest(payload)
    return payload


def publish_attention_merge(
    *,
    output_csv: str | Path,
    alias_csv: str | Path,
    sidecar_path: str | Path,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and publish one deterministic two-source attention merge."""

    canonical = Path(output_csv)
    alias = Path(alias_csv)
    sidecar = Path(sidecar_path)
    output_targets = {
        "canonical": canonical,
        "alias": alias,
        "sidecar": sidecar,
    }
    _validate_publication_file_targets(
        context="Attention merge publication",
        targets=output_targets,
    )
    resolved_outputs = {path.resolve() for path in output_targets.values()}
    source_artifact_paths: set[Path] = set()
    source_sidecars: list[Path] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise TypeError(f"Attention merge source {index} must be a mapping.")
        source_artifact_paths.add(
            _attention_merge_source_path(source, index=index, field="csv_path")
        )
        source_sidecar = _attention_merge_source_path(
            source,
            index=index,
            field="sidecar_path",
        )
        source_artifact_paths.add(source_sidecar)
        source_sidecars.append(source_sidecar)
    if any(
        _paths_overlap(output, source)
        for output in resolved_outputs
        for source in source_artifact_paths
    ):
        raise ValueError(
            "Attention merge publication cannot overwrite a bound source artifact."
        )

    fieldnames, rows, merged_bytes, report = _prepare_attention_merge_output(
        sources=sources,
    )
    payload = _build_attention_merge_sidecar_payload(
        output_csv=canonical,
        alias_csv=alias,
        sources=sources,
        merge_report=report,
        output_fieldnames=fieldnames,
        output_rows=rows,
        output_bytes=merged_bytes,
        alias_bytes=merged_bytes,
    )
    sidecar_bytes = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    round_trip_payload = json.loads(sidecar_bytes.decode("utf-8"))
    if round_trip_payload != payload:
        raise ValueError("Attention merge sidecar JSON round-trip validation failed.")
    if round_trip_payload.get("config_sha256") != _merge_config_digest(
        round_trip_payload
    ):
        raise ValueError("Attention merge sidecar config_sha256 validation failed.")

    for source_sidecar in source_sidecars:
        source_artifact_paths.update(
            _attention_run_bound_artifact_paths(source_sidecar)
        )
    if any(
        _paths_overlap(output, source)
        for output in resolved_outputs
        for source in source_artifact_paths
    ):
        raise ValueError(
            "Attention merge publication cannot overwrite a bound source artifact."
        )

    canonical.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    alias.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(merged_bytes)
    sidecar.write_bytes(sidecar_bytes)
    alias.write_bytes(merged_bytes)
    return {
        "canonical": canonical,
        "alias": alias,
        "sidecar": sidecar,
        "report": report,
    }


def write_attention_merge_sidecar(
    *,
    output_csv: str | Path,
    alias_csv: str | Path,
    sidecar_path: str | Path,
    sources: Sequence[Mapping[str, Any]],
    merge_report: Mapping[str, Any],
) -> Path:
    """Write provenance for one deterministic two-source attention merge."""

    sidecar = Path(sidecar_path)
    payload = _build_attention_merge_sidecar_payload(
        output_csv=Path(output_csv),
        alias_csv=Path(alias_csv),
        sources=sources,
        merge_report=merge_report,
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sidecar


def validate_attention_merge_sidecar(
    *,
    sidecar_path: str | Path,
) -> None:
    """Fail fast if a merge sidecar or any bound artifact has drifted."""

    sidecar = Path(sidecar_path)
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Attention merge provenance sidecar does not exist: {sidecar}"
        )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Invalid attention merge provenance sidecar: {sidecar}"
        ) from exc
    if payload.get("schema_version") != ATTENTION_MERGE_PROVENANCE_SCHEMA:
        raise ValueError(
            "Unsupported attention merge provenance schema_version: "
            f"{payload.get('schema_version')!r}."
        )
    if payload.get("row_identity_schema") != ATTENTION_MERGE_ROW_IDENTITY_SCHEMA:
        raise ValueError(
            "Unsupported attention merge row_identity_schema: "
            f"{payload.get('row_identity_schema')!r}."
        )
    output = payload.get("output")
    sources = payload.get("sources")
    merge_report = payload.get("merge")
    if (
        not isinstance(output, Mapping)
        or not isinstance(sources, list)
        or not isinstance(merge_report, Mapping)
    ):
        raise ValueError("Attention merge provenance structure is invalid.")
    rebuilt = _build_attention_merge_sidecar_payload(
        output_csv=Path(str(output.get("csv_path", ""))),
        alias_csv=Path(str(output.get("alias_csv", ""))),
        sources=[
            {
                "label": source.get("label"),
                "csv_path": source.get("csv_path"),
                "sidecar_path": source.get("sidecar_path"),
            }
            for source in sources
            if isinstance(source, Mapping)
        ],
        merge_report=merge_report,
    )
    if rebuilt != payload:
        raise ValueError(
            "Attention merge provenance metadata or bound artifacts have drifted."
        )
