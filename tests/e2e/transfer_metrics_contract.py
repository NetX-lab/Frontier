"""Small, fail-closed validators for transfer and request metric artifacts.

The validators intentionally operate on persisted runtime artifacts instead of
replaying simulator internals.  They are used by direct case verification and
keep the evidence boundary explicit:

* missing identity is ``INSUFFICIENT_EVIDENCE``;
* contradictory identity or timing is ``FAIL``;
* complete, internally independent evidence is ``PASS``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def _result(status: str, errors: Iterable[str] = (), **details: Any) -> dict[str, Any]:
    return {
        "status": status,
        "errors": list(errors),
        **details,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def _finite_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} is not finite: {value!r}")
    return number


def _non_negative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative int: {value!r}")
    return value


_PERCENTILES = (50, 90, 95, 99)
_STATISTIC_FIELDS = (
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p50",
    "p90",
    "p95",
    "p99",
)
_TPOT_NO_DATA_ERROR = (
    "No TPOT data available (all requests may have num_decode_tokens=1)"
)


def _linear_percentile(values: list[float], percentile: int) -> float:
    """Return the NumPy default (linear) percentile without trusting output data."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _recomputed_statistics(values: list[float]) -> dict[str, float | int | str]:
    if not values:
        raise ValueError("statistics require at least one value")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    result: dict[str, float | int | str] = {
        "count": len(values),
        "mean": mean,
        "median": _linear_percentile(values, 50),
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "unit": "ms",
    }
    for percentile in _PERCENTILES:
        result[f"p{percentile}"] = _linear_percentile(values, percentile)
    return result


def _close_metric(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6)


def _validate_recomputed_statistics(
    system_metrics: Mapping[str, Any],
    key: str,
    values: list[float],
    *,
    missing: list[str],
    failures: list[str],
    allow_no_data_error: bool = False,
) -> dict[str, float | int | str] | None:
    """Compare one persisted aggregate section with independently recomputed values."""

    section = system_metrics.get(key)
    if not isinstance(section, Mapping):
        missing.append(f"system metrics missing aggregate section: {key}")
        return None
    if not values:
        if not allow_no_data_error:
            failures.append(
                f"{key}: aggregate contains no data although request evidence is present"
            )
            return None
        error = section.get("error")
        if not isinstance(error, str) or not error:
            missing.append(f"{key}: missing documented no-data error object")
        elif error != _TPOT_NO_DATA_ERROR:
            failures.append(
                f"{key}: unexpected no-data error object: {error!r}"
            )
        return {"count": 0, "unit": "ms", "error": error or ""}

    expected = _recomputed_statistics(values)
    if "unit" not in section:
        missing.append(f"{key}: missing unit")
    elif section["unit"] != "ms":
        failures.append(
            f"{key}.unit mismatch: expected='ms' actual={section['unit']!r}"
        )

    for field in _STATISTIC_FIELDS:
        if field not in section:
            missing.append(f"{key}: missing field {field}")
            continue
        expected_value = expected[field]
        actual_value = section[field]
        if field == "count":
            if type(actual_value) is not int:
                failures.append(
                    f"{key}.count must be an int: actual={actual_value!r}"
                )
            elif actual_value != expected_value:
                failures.append(
                    f"{key}.count mismatch: expected={expected_value} actual={actual_value}"
                )
            continue
        try:
            actual_number = _finite_number(actual_value, f"{key}.{field}")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        expected_number = float(expected_value)
        if not _close_metric(actual_number, expected_number):
            failures.append(
                f"{key}.{field} mismatch: expected={expected_number} actual={actual_number}"
            )
    return expected


def _validate_strict_system_aggregates(
    system_metrics: Mapping[str, Any],
    completion_rows: Mapping[str, Mapping[str, Any]],
    arrival_rows: Iterable[Mapping[str, Any]],
    csv_rows: list[Mapping[str, Any]],
    *,
    expected_request_count: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Independently recompute system aggregates for release validation."""

    missing: list[str] = []
    failures: list[str] = []
    arrivals_by_id: dict[str, float] = {}
    for index, row in enumerate(arrival_rows):
        request_id = row.get("request_id")
        if request_id is None:
            missing.append(f"request_arrival[{index}] is missing request_id")
            continue
        key = str(request_id)
        if key in arrivals_by_id:
            failures.append(f"duplicate request_arrival row for request ID={key}")
            continue
        try:
            arrivals_by_id[key] = _finite_number(
                row["arrived_at"], f"request_arrival[{index}].arrived_at"
            )
        except (KeyError, TypeError, ValueError) as exc:
            missing.append(f"request_arrival[{index}] missing valid arrived_at: {exc}")

    if len(arrivals_by_id) != expected_request_count:
        failures.append(
            "request arrival cardinality mismatch: "
            f"expected={expected_request_count} actual={len(arrivals_by_id)}"
        )
    if set(arrivals_by_id) != set(completion_rows):
        failures.append(
            "request arrival/completion ID mismatch: "
            f"arrivals={sorted(arrivals_by_id)} completions={sorted(completion_rows)}"
        )

    ttft_values: list[float] = []
    tpot_values: list[float] = []
    e2e_values: list[float] = []
    completion_times: list[float] = []
    total_tokens = 0
    total_decode_tokens = 0
    for request_id, truth in completion_rows.items():
        try:
            arrived = _finite_number(truth["arrived_at"], f"{request_id}.arrived_at")
            prefill_completed = _finite_number(
                truth["prefill_completed_at"],
                f"{request_id}.prefill_completed_at",
            )
            first_decode_completed = _finite_number(
                truth["first_decode_token_completed_at"],
                f"{request_id}.first_decode_token_completed_at",
            )
            completed = _finite_number(truth["completed_at"], f"{request_id}.completed_at")
            num_prefill_tokens = _non_negative_int(
                truth["num_prefill_tokens"], f"{request_id}.num_prefill_tokens"
            )
            num_decode_tokens = _non_negative_int(
                truth["num_decode_tokens"], f"{request_id}.num_decode_tokens"
            )
            csv_row = next(
                (row for row in csv_rows if str(row.get("Request Id")) == request_id),
                None,
            )
            if csv_row is None:
                failures.append(f"{request_id}: missing CSV row for aggregate token totals")
                continue
            csv_num_tokens = _non_negative_int(
                int(csv_row["request_num_tokens"]),
                f"{request_id}.request_num_tokens",
            )
            csv_num_decode = _non_negative_int(
                int(csv_row["request_num_decode_tokens"]),
                f"{request_id}.request_num_decode_tokens",
            )
            if csv_num_tokens != num_prefill_tokens + num_decode_tokens:
                failures.append(
                    f"{request_id}: aggregate token total mismatch: "
                    f"csv={csv_num_tokens} truth={num_prefill_tokens + num_decode_tokens}"
                )
            if csv_num_decode != num_decode_tokens:
                failures.append(
                    f"{request_id}: aggregate decode-token mismatch: "
                    f"csv={csv_num_decode} truth={num_decode_tokens}"
                )
            total_tokens += csv_num_tokens
            total_decode_tokens += csv_num_decode
            ttft_values.append((prefill_completed - arrived) * 1000.0)
            e2e_values.append((completed - arrived) * 1000.0)
            if num_decode_tokens > 1:
                tpot_values.append(
                    (completed - first_decode_completed)
                    * 1000.0
                    / (num_decode_tokens - 1)
                )
            completion_times.append(completed)
            if request_id in arrivals_by_id and not _close_metric(
                arrivals_by_id[request_id], arrived
            ):
                failures.append(
                    f"{request_id}: arrival timestamp mismatch: "
                    f"event={arrivals_by_id[request_id]} completion={arrived}"
                )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(f"{request_id}: aggregate input is invalid: {exc}")

    aggregate_details: dict[str, Any] = {
        "recomputed_ttft_statistics": _recomputed_statistics(ttft_values)
        if ttft_values
        else None,
        "recomputed_tpot_statistics": _recomputed_statistics(tpot_values)
        if tpot_values
        else {"count": 0, "unit": "ms", "error": _TPOT_NO_DATA_ERROR},
        "recomputed_request_e2e_time_statistics": _recomputed_statistics(e2e_values)
        if e2e_values
        else None,
    }
    _validate_recomputed_statistics(
        system_metrics,
        "ttft_statistics",
        ttft_values,
        missing=missing,
        failures=failures,
    )
    _validate_recomputed_statistics(
        system_metrics,
        "tpot_statistics",
        tpot_values,
        missing=missing,
        failures=failures,
        allow_no_data_error=True,
    )
    _validate_recomputed_statistics(
        system_metrics,
        "request_e2e_time_statistics",
        e2e_values,
        missing=missing,
        failures=failures,
    )

    throughput = system_metrics.get("throughput_metrics")
    if not isinstance(throughput, Mapping):
        missing.append("system metrics missing aggregate section: throughput_metrics")
    elif not completion_times or not arrivals_by_id:
        failures.append("throughput_metrics cannot be recomputed without request times")
    else:
        duration_s = max(completion_times) - min(arrivals_by_id.values())
        expected_throughput: dict[str, float | int] = {
            "total_duration_ms": duration_s * 1000.0,
            "total_duration_seconds": duration_s,
            "requests_per_second": expected_request_count / duration_s
            if duration_s > 0
            else 0.0,
            "total_tokens_processed": total_tokens,
            "total_decode_tokens_generated": total_decode_tokens,
            "tokens_per_second": total_tokens / duration_s
            if duration_s > 0
            else 0.0,
            "decode_tokens_per_second": total_decode_tokens / duration_s
            if duration_s > 0
            else 0.0,
        }
        aggregate_details["recomputed_throughput_metrics"] = expected_throughput
        for field, expected_value in expected_throughput.items():
            if field not in throughput:
                missing.append(f"throughput_metrics: missing field {field}")
                continue
            if field in {
                "total_tokens_processed",
                "total_decode_tokens_generated",
            }:
                if type(throughput[field]) is not int:
                    failures.append(
                        f"throughput_metrics.{field} must be an int: "
                        f"actual={throughput[field]!r}"
                    )
                elif throughput[field] != expected_value:
                    failures.append(
                        f"throughput_metrics.{field} mismatch: "
                        f"expected={expected_value} actual={throughput[field]}"
                    )
                continue
            try:
                actual_value = _finite_number(
                    throughput[field], f"throughput_metrics.{field}"
                )
            except ValueError as exc:
                failures.append(str(exc))
                continue
            if not _close_metric(actual_value, float(expected_value)):
                failures.append(
                    f"throughput_metrics.{field} mismatch: "
                    f"expected={float(expected_value)} actual={actual_value}"
                )

    return missing, failures, aggregate_details


def _request_key(row: Mapping[str, Any]) -> tuple[int, ...]:
    request_ids = row.get("request_ids")
    if type(request_ids) is not list or not request_ids:
        raise ValueError("request_ids must be a non-empty list")
    normalized = tuple(_non_negative_int(int(request_id), "request_id") for request_id in request_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"request_ids contain duplicates: {normalized!r}")
    return normalized


def _request_identity_key(
    row: Mapping[str, Any],
) -> tuple[tuple[int, int], ...]:
    request_ids = _request_key(row)
    runtime_epochs = row.get("request_runtime_epochs")
    if (
        type(runtime_epochs) is not list
        or len(runtime_epochs) != len(request_ids)
        or any(type(value) is not int or value < 0 for value in runtime_epochs)
    ):
        raise ValueError(
            "request_runtime_epochs must align with request_ids"
        )
    return tuple(zip(request_ids, runtime_epochs))


def _request_runtime_iteration_key(
    request_ids: Any,
    runtime_epochs: Any,
    iteration_ids: Any,
    field_name: str,
) -> tuple[tuple[int, int, int], ...]:
    if type(request_ids) is not list or not request_ids:
        raise ValueError(f"{field_name}.request_ids must be a non-empty list")
    normalized_request_ids = tuple(
        _non_negative_int(int(request_id), f"{field_name}.request_id")
        for request_id in request_ids
    )
    if len(set(normalized_request_ids)) != len(normalized_request_ids):
        raise ValueError(
            f"{field_name}.request_ids contain duplicates: "
            f"{normalized_request_ids!r}"
        )
    if (
        type(runtime_epochs) is not list
        or len(runtime_epochs) != len(normalized_request_ids)
        or any(type(value) is not int or value < 0 for value in runtime_epochs)
    ):
        raise ValueError(
            f"{field_name}.request_runtime_epochs must align with request_ids"
        )
    if (
        type(iteration_ids) is not list
        or len(iteration_ids) != len(normalized_request_ids)
        or any(type(value) is not int or value < 0 for value in iteration_ids)
    ):
        raise ValueError(
            f"{field_name}.iteration_ids must align with request_ids"
        )
    return tuple(
        zip(normalized_request_ids, runtime_epochs, iteration_ids)
    )


def validate_transfer_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_target_by_transfer_id: Mapping[str, int] | None = None,
    required_transfer_kinds: set[str] | None = None,
) -> dict[str, Any]:
    """Validate the persisted transfer lineage rows."""

    errors: list[str] = []
    normalized_rows = [dict(row) for row in rows]
    if not normalized_rows:
        return _result("INSUFFICIENT_EVIDENCE", ["transfer ledger is empty"])

    required_transfer_kinds = set(required_transfer_kinds or ())
    unknown_required_kinds = required_transfer_kinds - {"kv_cache", "m2n"}
    if unknown_required_kinds:
        errors.append(
            "unknown required transfer kinds="
            f"{sorted(unknown_required_kinds)}"
        )
    seen_ids: set[str] = set()
    directions = {
        ("PREFILL", "DECODE"),
        ("PREFILL", "DECODE_ATTN"),
        ("DECODE_ATTN", "DECODE_FFN"),
        ("DECODE_FFN", "DECODE_ATTN"),
    }
    completed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(normalized_rows):
        prefix = f"row[{index}]"
        transfer_id = row.get("transfer_id")
        if transfer_id is None:
            errors.append(f"{prefix}: missing transfer_id")
            continue
        transfer_id = str(transfer_id)
        if transfer_id in seen_ids:
            errors.append(f"{prefix}: duplicate transfer_id={transfer_id}")
        seen_ids.add(transfer_id)

        required = (
            "transfer_kind",
            "request_ids",
            "request_runtime_epochs",
            "batch_id",
            "source_cluster",
            "target_cluster",
            "source_replica_id",
            "source_replica_local_id",
            "target_replica_id",
            "target_replica_local_id",
            "target_bound",
            "bytes",
            "start_ts_s",
            "end_ts_s",
            "duration_ms",
            "status",
        )
        missing = [field for field in required if field not in row]
        if missing:
            errors.append(f"{prefix}: missing fields={missing}")
            continue
        try:
            request_identity_key = _request_identity_key(row)
            _non_negative_int(int(row["batch_id"]), f"{prefix}.batch_id")
            source_cluster = str(row["source_cluster"])
            target_cluster = str(row["target_cluster"])
            if (source_cluster, target_cluster) not in directions:
                raise ValueError(
                    f"invalid direction={source_cluster}->{target_cluster}"
                )
            source_replica = row["source_replica_id"]
            target_replica = row["target_replica_id"]
            if source_replica is None or target_replica is None:
                raise ValueError("source/target Replica identity is incomplete")
            _non_negative_int(int(source_replica), f"{prefix}.source_replica_id")
            _non_negative_int(int(target_replica), f"{prefix}.target_replica_id")
            bytes_value = _non_negative_int(int(row["bytes"]), f"{prefix}.bytes")
            start = _finite_number(row["start_ts_s"], f"{prefix}.start_ts_s")
            end = _finite_number(row["end_ts_s"], f"{prefix}.end_ts_s")
            duration_ms = _finite_number(row["duration_ms"], f"{prefix}.duration_ms")
            if end < start - 1e-12:
                raise ValueError(f"end_ts_s precedes start_ts_s: {start} > {end}")
            expected_duration_ms = (end - start) * 1000.0
            if abs(expected_duration_ms - duration_ms) > 1e-6:
                raise ValueError(
                    f"duration mismatch: expected={expected_duration_ms} actual={duration_ms}"
                )
            if row["status"] != "completed":
                raise ValueError(f"transfer status is not completed: {row['status']!r}")
            if row["target_bound"] is not True:
                raise ValueError(
                    f"transfer target was not scheduler-bound: "
                    f"{row['target_bound']!r}"
                )

            if row["transfer_kind"] == "m2n":
                layer_id = row.get("layer_id")
                afd_stage_idx = row.get("afd_stage_idx")
                iteration_ids = row.get("iteration_ids")
                batch_global_id = row.get("batch_global_id")
                if (
                    type(layer_id) is not int
                    or layer_id < 0
                    or type(afd_stage_idx) is not int
                    or afd_stage_idx < 0
                    or type(batch_global_id) is not int
                    or batch_global_id < 0
                    or type(iteration_ids) is not list
                    or len(iteration_ids) != len(request_identity_key)
                    or any(type(value) is not int or value < 0 for value in iteration_ids)
                ):
                    raise ValueError(
                        "m2n layer/stage/iteration identity is incomplete"
                    )
                attention_owner_replica_id = row.get(
                    "attention_owner_replica_id"
                )
                attention_owner_replica_local_id = row.get(
                    "attention_owner_replica_local_id"
                )
                _non_negative_int(
                    attention_owner_replica_id,
                    f"{prefix}.attention_owner_replica_id",
                )
                if attention_owner_replica_local_id is not None:
                    _non_negative_int(
                        attention_owner_replica_local_id,
                        f"{prefix}.attention_owner_replica_local_id",
                    )
            elif row["transfer_kind"] != "kv_cache":
                raise ValueError(f"unknown transfer_kind={row['transfer_kind']!r}")
            if expected_target_by_transfer_id is not None:
                expected_target = expected_target_by_transfer_id.get(transfer_id)
                if expected_target is not None and int(target_replica) != int(expected_target):
                    raise ValueError(
                        f"target Replica mismatch: expected={expected_target} actual={target_replica}"
                    )
            completed_rows.append(row)
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in completed_rows:
        if row.get("transfer_kind") != "m2n":
            continue
        grouped.setdefault(
            (
                _request_identity_key(row),
                row.get("batch_global_id"),
                row.get("layer_id"),
                row.get("afd_stage_idx"),
                tuple(row.get("iteration_ids", [])),
            ),
            [],
        ).append(row)
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["start_ts_s"]))
        directions_seen = [f"{row['source_cluster']}->{row['target_cluster']}" for row in ordered]
        a2f = [row for row in ordered if row["source_cluster"] == "DECODE_ATTN"]
        f2a = [row for row in ordered if row["source_cluster"] == "DECODE_FFN"]
        if len(a2f) != 1 or len(f2a) != 1 or len(ordered) != 2:
            errors.append(
                "m2n identity must contain exactly one A->F and one F->A "
                f"transfer: identity={key!r}, directions={directions_seen!r}"
            )
            continue

        attn_to_ffn = a2f[0]
        ffn_to_attn = f2a[0]
        if attn_to_ffn.get("pipeline_stage") != "attn_to_ffn":
            errors.append(
                f"A->F pipeline_stage mismatch for identity={key!r}: "
                f"{attn_to_ffn.get('pipeline_stage')!r}"
            )
        if ffn_to_attn.get("pipeline_stage") != "ffn_to_attn":
            errors.append(
                f"F->A pipeline_stage mismatch for identity={key!r}: "
                f"{ffn_to_attn.get('pipeline_stage')!r}"
            )

        if (
            attn_to_ffn["source_replica_id"]
            != ffn_to_attn["target_replica_id"]
        ):
            errors.append(
                "A->F source Replica must equal F->A target Replica "
                f"for identity={key!r}"
            )
        if attn_to_ffn["target_replica_id"] != ffn_to_attn["source_replica_id"]:
            errors.append(
                "A->F target Replica must equal F->A source Replica "
                f"for identity={key!r}"
            )
        if int(attn_to_ffn["bytes"]) != int(ffn_to_attn["bytes"]):
            errors.append(
                "A->F and F->A byte counts must match "
                f"for identity={key!r}: "
                f"a2f={attn_to_ffn['bytes']}, f2a={ffn_to_attn['bytes']}"
            )
        if (
            attn_to_ffn.get("attention_owner_replica_id")
            != attn_to_ffn["source_replica_id"]
        ):
            errors.append(
                f"A->F Attention owner Replica mismatch for identity={key!r}"
            )
        if (
            ffn_to_attn.get("attention_owner_replica_id")
            != ffn_to_attn["target_replica_id"]
        ):
            errors.append(
                "F->A transfer must retain the original Attention owner "
                f"Replica for identity={key!r}"
            )
        if (
            attn_to_ffn.get("attention_owner_replica_id")
            != ffn_to_attn.get("attention_owner_replica_id")
            or attn_to_ffn.get("attention_owner_replica_local_id")
            != ffn_to_attn.get("attention_owner_replica_local_id")
        ):
            errors.append(
                "A->F and F->A Attention owner identity must match "
                f"for identity={key!r}"
            )
        if float(attn_to_ffn["end_ts_s"]) > float(ffn_to_attn["start_ts_s"]) + 1e-12:
            errors.append(
                f"m2n direction ordering is reversed/overlapping for identity={key!r}"
            )

    kind_counts = {
        kind: sum(1 for row in completed_rows if row["transfer_kind"] == kind)
        for kind in ("kv_cache", "m2n")
    }
    for required_kind in sorted(required_transfer_kinds):
        if kind_counts.get(required_kind, 0) == 0:
            errors.append(
                f"required transfer kind is missing: {required_kind}"
            )
    direction_counts = {
        direction: sum(
            1
            for row in completed_rows
            if f"{row['source_cluster']}->{row['target_cluster']}" == direction
        )
        for direction in (
            "PREFILL->DECODE",
            "PREFILL->DECODE_ATTN",
            "DECODE_ATTN->DECODE_FFN",
            "DECODE_FFN->DECODE_ATTN",
        )
    }
    details = {
        "transfer_count": len(normalized_rows),
        "kv_transfer_count": kind_counts["kv_cache"],
        "m2n_transfer_count": kind_counts["m2n"],
        "total_transfer_bytes": sum(int(row["bytes"]) for row in completed_rows),
        "total_duration_ms": sum(
            float(row["duration_ms"]) for row in completed_rows
        ),
        "direction_counts": direction_counts,
    }
    if errors:
        return _result("FAIL", errors, **details)
    return _result("PASS", **details)


def validate_kv_stage_alignment(
    stage_rows: Iterable[Mapping[str, Any]],
    transfer_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join each KV transfer to its observed source and target stages."""

    stages = [dict(row) for row in stage_rows]
    kv_rows = [
        dict(row)
        for row in transfer_rows
        if row.get("transfer_kind") == "kv_cache"
    ]
    if not stages or not kv_rows:
        return _result(
            "INSUFFICIENT_EVIDENCE",
            ["KV stage validation requires stage rows and KV transfers"],
        )

    required_stage_fields = (
        "cluster_type",
        "request_ids",
        "request_runtime_epochs",
        "iteration_ids",
        "replica_id",
        "replica_local_id",
        "stage_id",
        "stage_start_ts",
        "stage_end_ts",
    )
    relevant_clusters = {
        str(row.get(field))
        for row in kv_rows
        for field in ("source_cluster", "target_cluster")
    }
    relevant_stages = [
        row for row in stages if str(row.get("cluster_type")) in relevant_clusters
    ]
    for index, row in enumerate(relevant_stages):
        missing = [field for field in required_stage_fields if field not in row]
        if missing:
            return _result(
                "INSUFFICIENT_EVIDENCE",
                [f"KV stage[{index}] is missing identity fields={missing}"],
            )

    errors: list[str] = []
    matched_target_stage_count = 0
    for index, transfer in enumerate(kv_rows):
        prefix = f"KV transfer[{index}]"
        try:
            transfer_identity = set(
                _request_runtime_iteration_key(
                    transfer.get("request_ids"),
                    transfer.get("request_runtime_epochs"),
                    transfer.get("iteration_ids"),
                    prefix,
                )
            )
            source_cluster = str(transfer["source_cluster"])
            target_cluster = str(transfer["target_cluster"])
            source_replica_id = _non_negative_int(
                int(transfer["source_replica_id"]),
                f"{prefix}.source_replica_id",
            )
            target_replica_id = _non_negative_int(
                int(transfer["target_replica_id"]),
                f"{prefix}.target_replica_id",
            )
            source_replica_local_id = transfer["source_replica_local_id"]
            if source_replica_local_id is not None:
                source_replica_local_id = _non_negative_int(
                    int(source_replica_local_id),
                    f"{prefix}.source_replica_local_id",
                )
            target_replica_local_id = transfer["target_replica_local_id"]
            if target_replica_local_id is not None:
                target_replica_local_id = _non_negative_int(
                    int(target_replica_local_id),
                    f"{prefix}.target_replica_local_id",
                )
            start_time = _finite_number(
                transfer["start_ts_s"],
                f"{prefix}.start_ts_s",
            )
            end_time = _finite_number(
                transfer["end_ts_s"],
                f"{prefix}.end_ts_s",
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix} identity is invalid: {exc}")
            continue

        source_candidates: list[tuple[float, int]] = []
        target_candidates: list[
            tuple[float, int, int | None]
        ] = []
        for stage_index, stage in enumerate(relevant_stages):
            stage_prefix = f"stage[{stage_index}]"
            try:
                stage_identity = set(
                    _request_runtime_iteration_key(
                        stage["request_ids"],
                        stage["request_runtime_epochs"],
                        stage["iteration_ids"],
                        stage_prefix,
                    )
                )
                stage_replica_id = _non_negative_int(
                    int(stage["replica_id"]),
                    f"{stage_prefix}.replica_id",
                )
                stage_id = _non_negative_int(
                    int(stage["stage_id"]),
                    f"{stage_prefix}.stage_id",
                )
                stage_replica_local_id = stage["replica_local_id"]
                if stage_replica_local_id is not None:
                    stage_replica_local_id = _non_negative_int(
                        int(stage_replica_local_id),
                        f"{stage_prefix}.replica_local_id",
                    )
                stage_start = _finite_number(
                    stage["stage_start_ts"],
                    f"{stage_prefix}.stage_start_ts",
                )
                stage_end = _finite_number(
                    stage["stage_end_ts"],
                    f"{stage_prefix}.stage_end_ts",
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{stage_prefix} identity is invalid: {exc}")
                continue
            if not transfer_identity.issubset(stage_identity):
                continue
            if (
                stage.get("cluster_type") == source_cluster
                and stage_replica_id == source_replica_id
                and stage_replica_local_id == source_replica_local_id
            ):
                source_candidates.append((stage_end, stage_id))
            if (
                stage.get("cluster_type") == target_cluster
                and stage_id == 0
            ):
                target_candidates.append(
                    (
                        stage_start,
                        stage_replica_id,
                        stage_replica_local_id,
                    )
                )

        if not source_candidates:
            errors.append(
                f"{prefix} has no source stage matching request/runtime-epoch/"
                "iteration and Replica"
            )
        else:
            source_terminal_stage_id = max(
                stage_id for _, stage_id in source_candidates
            )
            latest_source_end = max(
                stage_end for stage_end, _ in source_candidates
            )
            latest_source_stage_ids = {
                stage_id
                for stage_end, stage_id in source_candidates
                if abs(stage_end - latest_source_end) <= 1e-9
            }
            if abs(latest_source_end - start_time) > 1e-9:
                errors.append(
                    f"{prefix} start does not equal the latest compatible "
                    "source stage completion: "
                    f"transfer_start={start_time}, "
                    f"latest_source_end={latest_source_end}"
                )
            if latest_source_stage_ids != {source_terminal_stage_id}:
                errors.append(
                    f"{prefix} latest source completion is not exclusively "
                    "from the terminal pipeline stage: "
                    f"terminal_stage_id={source_terminal_stage_id}, "
                    f"latest_stage_ids={sorted(latest_source_stage_ids)}"
                )
        if not target_candidates:
            errors.append(
                f"{prefix} has no target entry stage matching "
                "request/runtime-epoch/iteration"
            )
        else:
            earliest_target_start = min(
                stage_start
                for stage_start, _, _ in target_candidates
            )
            target_timing_valid = earliest_target_start >= end_time - 1e-12
            if not target_timing_valid:
                errors.append(
                    f"{prefix} earliest compatible target entry stage starts "
                    "before transfer end: "
                    f"stage_start={earliest_target_start}, "
                    f"transfer_end={end_time}"
                )
            earliest_target_identities = {
                (stage_replica_id, stage_replica_local_id)
                for stage_start, stage_replica_id, stage_replica_local_id
                in target_candidates
                if abs(stage_start - earliest_target_start) <= 1e-9
            }
            if earliest_target_identities != {
                (target_replica_id, target_replica_local_id)
            }:
                errors.append(
                    f"{prefix} target Replica does not match the earliest "
                    "compatible target stage: "
                    f"ledger={(target_replica_id, target_replica_local_id)!r} "
                    f"stage={sorted(earliest_target_identities, key=repr)!r}"
                )
            elif target_timing_valid:
                matched_target_stage_count += 1

    details = {
        "kv_transfer_count": len(kv_rows),
        "matched_target_stage_count": matched_target_stage_count,
    }
    if errors:
        return _result("FAIL", errors, **details)
    return _result("PASS", **details)


def validate_stage_transfer_alignment(
    stage_rows: Iterable[Mapping[str, Any]],
    transfer_rows: Iterable[Mapping[str, Any]],
    *,
    expected_layer_protocols: Mapping[int, str],
    expected_moe_ep_size: int | None = None,
) -> dict[str, Any]:
    """Join stage-grain lineage to the persisted cross-role transfer rows."""

    stages = [dict(row) for row in stage_rows]
    transfers = [dict(row) for row in transfer_rows]
    if not stages or not transfers:
        return _result(
            "INSUFFICIENT_EVIDENCE",
            ["stage and transfer ledgers are both required"],
        )

    errors: list[str] = []
    normalized_protocols: dict[int, str] = {}
    for raw_layer_id, raw_protocol in expected_layer_protocols.items():
        try:
            layer_id = _non_negative_int(
                raw_layer_id,
                "expected_layer_protocols.layer_id",
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        protocol = str(raw_protocol)
        if protocol not in {"dense", "moe"}:
            errors.append(
                f"expected layer {layer_id} has invalid protocol={protocol!r}"
            )
            continue
        normalized_protocols[layer_id] = protocol
    if not normalized_protocols:
        return _result(
            "INSUFFICIENT_EVIDENCE",
            errors or ["expected_layer_protocols is empty"],
        )
    has_moe_protocol = any(
        protocol == "moe" for protocol in normalized_protocols.values()
    )
    if has_moe_protocol and expected_moe_ep_size is None:
        return _result(
            "INSUFFICIENT_EVIDENCE",
            ["MoE stage validation requires expected_moe_ep_size"],
        )
    if expected_moe_ep_size is not None and (
        type(expected_moe_ep_size) is not int or expected_moe_ep_size <= 0
    ):
        errors.append(
            "expected_moe_ep_size must be a positive int, got "
            f"{expected_moe_ep_size!r}"
        )

    attn_stages = [
        row
        for row in stages
        if row.get("cluster_type") == "DECODE_ATTN"
        and row.get("execution_scope") == "FULL_STAGE_WORLD"
    ]
    ffn_stages = [
        row
        for row in stages
        if row.get("cluster_type") == "DECODE_FFN"
        and row.get("execution_scope") in {"FULL_STAGE_WORLD", "EP_WAVE_LANE"}
    ]
    if not attn_stages or not ffn_stages:
        return _result(
            "INSUFFICIENT_EVIDENCE",
            ["PD-AF stage ledger lacks DECODE_ATTN/DECODE_FFN identity"],
        )

    observed_attn_layers = {
        row.get("layer_id")
        for row in attn_stages
        if type(row.get("layer_id")) is int
    }
    observed_ffn_layers = {
        row.get("layer_id")
        for row in ffn_stages
        if type(row.get("layer_id")) is int
    }
    expected_layers = set(normalized_protocols)
    for layer_id in sorted(expected_layers):
        if layer_id not in observed_attn_layers:
            errors.append(
                f"expected layer {layer_id} has no DECODE_ATTN stage"
            )
        if layer_id not in observed_ffn_layers:
            errors.append(
                f"expected layer {layer_id} has no DECODE_FFN stage"
            )
    unexpected_layers = (
        observed_attn_layers | observed_ffn_layers
    ) - expected_layers
    if unexpected_layers:
        errors.append(
            f"stage ledger contains unexpected layers={sorted(unexpected_layers)}"
        )
    for row in ffn_stages:
        layer_id = row.get("layer_id")
        protocol = normalized_protocols.get(layer_id)
        if protocol is None:
            continue
        expected_scope = (
            "FULL_STAGE_WORLD" if protocol == "dense" else "EP_WAVE_LANE"
        )
        if row.get("execution_scope") != expected_scope:
            errors.append(
                f"layer {layer_id} protocol={protocol} requires "
                f"DECODE_FFN scope={expected_scope}, got "
                f"{row.get('execution_scope')!r}"
            )
        expected_operation_kind = (
            "full_stage" if protocol == "dense" else "ep_ffn"
        )
        if row.get("operation_kind") != expected_operation_kind:
            errors.append(
                f"layer {layer_id} protocol={protocol} requires "
                f"DECODE_FFN operation_kind={expected_operation_kind}, got "
                f"{row.get('operation_kind')!r}"
            )

    required_stage_fields = (
        "batch_id",
        "request_ids",
        "request_runtime_epochs",
        "iteration_ids",
        "layer_id",
        "afd_stage_idx",
        "replica_id",
        "replica_local_id",
        "stage_start_ts",
        "stage_end_ts",
    )
    for index, row in enumerate(attn_stages + ffn_stages):
        missing = [field for field in required_stage_fields if field not in row]
        if missing:
            return _result(
                "INSUFFICIENT_EVIDENCE",
                [f"PD-AF stage[{index}] is missing identity fields={missing}"],
            )
    required_ffn_fields = (
        "stage_id",
        "operation_id",
        "schedule_epoch",
        "source_request_runtime_epochs",
        "source_batch_arrival_times",
        "source_group_ready_ts",
    )
    for index, row in enumerate(ffn_stages):
        missing = [field for field in required_ffn_fields if field not in row]
        if missing:
            return _result(
                "INSUFFICIENT_EVIDENCE",
                [f"DECODE_FFN stage[{index}] is missing queue fields={missing}"],
            )

    attn_by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in attn_stages:
        try:
            batch_id = _non_negative_int(int(row["batch_id"]), "stage.batch_id")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"DECODE_ATTN stage identity is invalid: {exc}")
            continue
        attn_by_batch.setdefault(batch_id, []).append(row)

    canonical_request_ids: set[str] = set()
    request_identity_by_stage: dict[
        tuple[int, int, int],
        set[tuple[int, int, int]],
    ] = {}
    for row in attn_stages:
        try:
            batch_id = _non_negative_int(int(row["batch_id"]), "stage.batch_id")
            layer_id = _non_negative_int(
                int(row["layer_id"]), "stage.layer_id"
            )
            afd_stage_idx = _non_negative_int(
                int(row["afd_stage_idx"]), "stage.afd_stage_idx"
            )
            request_identity = _request_runtime_iteration_key(
                row.get("request_ids"),
                row.get("request_runtime_epochs"),
                row.get("iteration_ids"),
                "DECODE_ATTN stage",
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"DECODE_ATTN stage request identity is invalid: {exc}")
            continue
        canonical_request_ids.update(
            str(request_id) for request_id, _, _ in request_identity
        )
        request_identity_by_stage.setdefault(
            (batch_id, layer_id, afd_stage_idx),
            set(),
        ).update(
            request_identity
        )

    parsed_attn: dict[
        tuple[int, tuple[tuple[int, int, int], ...], int, int],
        list[dict[str, Any]],
    ] = {}
    for index, row in enumerate(attn_stages):
        prefix = f"DECODE_ATTN stage[{index}]"
        try:
            batch_id = _non_negative_int(int(row["batch_id"]), f"{prefix}.batch_id")
            request_identity = _request_runtime_iteration_key(
                row["request_ids"],
                row["request_runtime_epochs"],
                row["iteration_ids"],
                prefix,
            )
            normalized_request_ids = tuple(
                str(request_id) for request_id, _, _ in request_identity
            )
            layer_id = _non_negative_int(int(row["layer_id"]), f"{prefix}.layer_id")
            afd_stage_idx = _non_negative_int(
                int(row["afd_stage_idx"]), f"{prefix}.afd_stage_idx"
            )
            replica_id = _non_negative_int(
                int(row["replica_id"]), f"{prefix}.replica_id"
            )
            if row["replica_local_id"] is not None:
                raise ValueError("DECODE_ATTN stage must not carry an EP lane identity")
            stage_start = _finite_number(
                row["stage_start_ts"], f"{prefix}.stage_start_ts"
            )
            stage_end = _finite_number(
                row["stage_end_ts"], f"{prefix}.stage_end_ts"
            )
            if stage_end < stage_start - 1e-12:
                raise ValueError("stage_end_ts precedes stage_start_ts")
            parsed = dict(row)
            parsed["_normalized_request_ids"] = normalized_request_ids
            parsed["_normalized_request_identity"] = request_identity
            parsed["_normalized_replica_id"] = replica_id
            parsed["_normalized_stage_start"] = stage_start
            parsed["_normalized_stage_end"] = stage_end
            key = (
                batch_id,
                request_identity,
                layer_id,
                afd_stage_idx,
            )
            parsed_attn.setdefault(key, []).append(parsed)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix} identity is invalid: {exc}")

    parsed_ffn: list[dict[str, Any]] = []
    for index, row in enumerate(ffn_stages):
        prefix = f"DECODE_FFN stage[{index}]"
        try:
            layer_id = _non_negative_int(int(row["layer_id"]), f"{prefix}.layer_id")
            protocol = normalized_protocols[layer_id]
            source_request_ids = row.get("source_request_ids")
            source_request_runtime_epochs = row.get(
                "source_request_runtime_epochs"
            )
            source_batch_ids = row.get("source_batch_ids")
            source_batch_arrival_times = row.get("source_batch_arrival_times")
            if type(source_request_ids) is not list or not source_request_ids:
                return _result(
                    "INSUFFICIENT_EVIDENCE",
                    [f"{prefix} is missing source_request_ids"],
                )
            if type(source_batch_ids) is not list or not source_batch_ids:
                return _result(
                    "INSUFFICIENT_EVIDENCE",
                    [f"{prefix} is missing source_batch_ids"],
                )
            if (
                type(source_batch_arrival_times) is not list
                or len(source_batch_arrival_times) != len(source_batch_ids)
            ):
                raise ValueError(
                    "source_batch_arrival_times must align with source_batch_ids"
                )
            normalized_source_batch_arrival_times = tuple(
                _finite_number(
                    arrival_time,
                    f"{prefix}.source_batch_arrival_times[{arrival_index}]",
                )
                for arrival_index, arrival_time in enumerate(
                    source_batch_arrival_times
                )
            )
            iteration_ids = row["iteration_ids"]
            source_request_identity = _request_runtime_iteration_key(
                source_request_ids,
                source_request_runtime_epochs,
                iteration_ids,
                f"{prefix}.source",
            )
            normalized_source_request_ids = tuple(
                str(request_id)
                for request_id, _, _ in source_request_identity
            )
            _request_identity_key(row)
            afd_stage_idx = _non_negative_int(
                int(row["afd_stage_idx"]), f"{prefix}.afd_stage_idx"
            )
            replica_id = _non_negative_int(
                int(row["replica_id"]), f"{prefix}.replica_id"
            )
            stage_id = _non_negative_int(
                int(row["stage_id"]), f"{prefix}.stage_id"
            )
            operation_id = _non_negative_int(
                int(row["operation_id"]), f"{prefix}.operation_id"
            )
            schedule_epoch = _non_negative_int(
                int(row["schedule_epoch"]), f"{prefix}.schedule_epoch"
            )
            if protocol == "dense":
                if row["replica_local_id"] is not None:
                    raise ValueError(
                        "dense DECODE_FFN stage must not carry an EP lane identity"
                    )
                replica_local_id = None
            else:
                replica_local_id = _non_negative_int(
                    int(row["replica_local_id"]),
                    f"{prefix}.replica_local_id",
                )
            stage_start = _finite_number(
                row["stage_start_ts"], f"{prefix}.stage_start_ts"
            )
            stage_end = _finite_number(
                row["stage_end_ts"], f"{prefix}.stage_end_ts"
            )
            if stage_end < stage_start - 1e-12:
                raise ValueError("stage_end_ts precedes stage_start_ts")
            completion_end = stage_end
            if protocol == "moe":
                missing_completion_fields = [
                    field
                    for field in (
                        "stage_completion_observed_ts",
                        "stage_completion_observed_source",
                    )
                    if field not in row
                ]
                if missing_completion_fields:
                    return _result(
                        "INSUFFICIENT_EVIDENCE",
                        [
                            f"{prefix} is missing observed completion fields="
                            f"{missing_completion_fields}"
                        ],
                    )
                completion_end = _finite_number(
                    row["stage_completion_observed_ts"],
                    f"{prefix}.stage_completion_observed_ts",
                )
                if completion_end < stage_start - 1e-12:
                    raise ValueError(
                        "stage_completion_observed_ts precedes stage_start_ts"
                    )
                if (
                    row["stage_completion_observed_source"]
                    != "ep_alltoall_combine_collective"
                ):
                    raise ValueError(
                        "MoE DECODE_FFN stage completion source must be "
                        "ep_alltoall_combine_collective"
                    )
            source_group_ready = _finite_number(
                row["source_group_ready_ts"],
                f"{prefix}.source_group_ready_ts",
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix} identity is invalid: {exc}")
            continue

        expected_source_request_identity: set[tuple[int, int, int]] = set()
        normalized_source_batch_ids: list[int] = []
        for source_batch_id in source_batch_ids:
            try:
                normalized_batch_id = _non_negative_int(
                    int(source_batch_id), f"{prefix}.source_batch_id"
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"{prefix} has invalid source_batch_id: {exc}")
                continue
            normalized_source_batch_ids.append(normalized_batch_id)
            expected_source_request_identity.update(
                request_identity_by_stage.get(
                    (normalized_batch_id, layer_id, afd_stage_idx),
                    set(),
                )
            )
        if len(normalized_source_batch_ids) != len(source_batch_ids):
            continue
        if len(set(normalized_source_batch_ids)) != len(
            normalized_source_batch_ids
        ):
            errors.append(f"{prefix} source_batch_ids contain duplicates")
            continue
        if (
            set(source_request_identity)
            != expected_source_request_identity
        ):
            errors.append(
                f"{prefix} source request identity does not match source "
                "batch request/runtime-epoch/iteration lineage"
            )
        for normalized_batch_id in normalized_source_batch_ids:
            if (
                normalized_batch_id,
                layer_id,
                afd_stage_idx,
            ) not in request_identity_by_stage:
                errors.append(
                    f"{prefix} source_batch_id={normalized_batch_id} has no "
                    "matching DECODE_ATTN layer/stage identity"
                )
        parsed = dict(row)
        parsed["_normalized_source_request_ids"] = normalized_source_request_ids
        parsed["_normalized_source_request_identity"] = (
            source_request_identity
        )
        parsed["_normalized_source_batch_ids"] = tuple(normalized_source_batch_ids)
        parsed["_normalized_source_batch_arrival_times"] = (
            normalized_source_batch_arrival_times
        )
        parsed["_normalized_layer_id"] = layer_id
        parsed["_normalized_afd_stage_idx"] = afd_stage_idx
        parsed["_normalized_protocol"] = protocol
        parsed["_normalized_replica_id"] = replica_id
        parsed["_normalized_replica_local_id"] = replica_local_id
        parsed["_normalized_stage_id"] = stage_id
        parsed["_normalized_operation_id"] = operation_id
        parsed["_normalized_schedule_epoch"] = schedule_epoch
        parsed["_normalized_source_group_ready"] = source_group_ready
        parsed["_normalized_stage_start"] = stage_start
        parsed["_normalized_stage_end"] = stage_end
        parsed["_normalized_completion_end"] = completion_end
        parsed["_operation_key"] = (
            replica_id,
            stage_id,
            operation_id,
            schedule_epoch,
            layer_id,
            afd_stage_idx,
            tuple(normalized_source_batch_ids),
            source_request_identity,
            protocol,
        )
        parsed_ffn.append(parsed)

    m2n_rows = [row for row in transfers if row.get("transfer_kind") == "m2n"]
    if not m2n_rows:
        return _result(
            "INSUFFICIENT_EVIDENCE",
            ["transfer ledger has no m2n rows"],
        )
    observed_transfer_layers = {
        row.get("layer_id")
        for row in m2n_rows
        if type(row.get("layer_id")) is int
    }
    missing_transfer_layers = expected_layers - observed_transfer_layers
    if missing_transfer_layers:
        errors.append(
            "expected layers have no M2N transfer chain="
            f"{sorted(missing_transfer_layers)}"
        )
    unexpected_transfer_layers = observed_transfer_layers - expected_layers
    if unexpected_transfer_layers:
        errors.append(
            "M2N ledger contains unexpected layers="
            f"{sorted(unexpected_transfer_layers)}"
        )

    required_transfer_identity = (
        "batch_id",
        "request_ids",
        "request_runtime_epochs",
        "iteration_ids",
        "layer_id",
        "afd_stage_idx",
        "source_replica_id",
        "source_replica_local_id",
        "target_replica_id",
        "target_replica_local_id",
        "start_ts_s",
        "end_ts_s",
    )
    for index, row in enumerate(m2n_rows):
        missing = [
            field for field in required_transfer_identity if field not in row
        ]
        if missing:
            return _result(
                "INSUFFICIENT_EVIDENCE",
                [f"m2n transfer[{index}] is missing identity fields={missing}"],
            )

    for row in m2n_rows:
        batch_id = row["batch_id"]
        try:
            normalized_batch_id = _non_negative_int(int(batch_id), "m2n.batch_id")
            normalized_layer_id = _non_negative_int(
                int(row["layer_id"]), "m2n.layer_id"
            )
            normalized_afd_stage_idx = _non_negative_int(
                int(row["afd_stage_idx"]), "m2n.afd_stage_idx"
            )
            request_identity = _request_runtime_iteration_key(
                row.get("request_ids"),
                row.get("request_runtime_epochs"),
                row.get("iteration_ids"),
                "m2n",
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"m2n batch identity is invalid: {exc}")
            continue
        expected_request_identity = request_identity_by_stage.get(
            (
                normalized_batch_id,
                normalized_layer_id,
                normalized_afd_stage_idx,
            ),
            set(),
        )
        if set(request_identity) != expected_request_identity:
            errors.append(
                "m2n request lineage does not match source stage "
                "request/runtime-epoch/iteration identity: "
                f"transfer_id={row.get('transfer_id')!r}"
            )
        if normalized_batch_id not in attn_by_batch:
            errors.append(
                f"m2n transfer_id={row.get('transfer_id')!r} has no "
                "DECODE_ATTN stage batch"
            )

    transfer_groups: dict[
        tuple[int, tuple[tuple[int, int, int], ...], int, int],
        list[dict[str, Any]],
    ] = {}
    for index, row in enumerate(m2n_rows):
        prefix = f"m2n transfer[{index}]"
        try:
            request_identity = _request_runtime_iteration_key(
                row["request_ids"],
                row["request_runtime_epochs"],
                row["iteration_ids"],
                prefix,
            )
            key = (
                _non_negative_int(int(row["batch_id"]), f"{prefix}.batch_id"),
                request_identity,
                _non_negative_int(int(row["layer_id"]), f"{prefix}.layer_id"),
                _non_negative_int(
                    int(row["afd_stage_idx"]), f"{prefix}.afd_stage_idx"
                ),
            )
            source_replica_id = _non_negative_int(
                int(row["source_replica_id"]), f"{prefix}.source_replica_id"
            )
            target_replica_id = _non_negative_int(
                int(row["target_replica_id"]), f"{prefix}.target_replica_id"
            )
            source_replica_local_id = row["source_replica_local_id"]
            if source_replica_local_id is not None:
                source_replica_local_id = _non_negative_int(
                    int(source_replica_local_id),
                    f"{prefix}.source_replica_local_id",
                )
            target_replica_local_id = row["target_replica_local_id"]
            if target_replica_local_id is not None:
                target_replica_local_id = _non_negative_int(
                    int(target_replica_local_id),
                    f"{prefix}.target_replica_local_id",
                )
            parsed = dict(row)
            parsed["_normalized_source_replica_id"] = source_replica_id
            parsed["_normalized_target_replica_id"] = target_replica_id
            parsed["_normalized_source_replica_local_id"] = (
                source_replica_local_id
            )
            parsed["_normalized_target_replica_local_id"] = (
                target_replica_local_id
            )
            transfer_groups.setdefault(key, []).append(parsed)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{prefix} identity is invalid: {exc}")

    used_attn_keys: set[
        tuple[int, tuple[tuple[int, int, int], ...], int, int]
    ] = set()
    used_ffn_indices: set[int] = set()
    transfer_ready_by_operation: dict[tuple[Any, ...], float] = {}
    for key, group in transfer_groups.items():
        a2f_rows = [
            row
            for row in group
            if row.get("source_cluster") == "DECODE_ATTN"
            and row.get("target_cluster") == "DECODE_FFN"
        ]
        f2a_rows = [
            row
            for row in group
            if row.get("source_cluster") == "DECODE_FFN"
            and row.get("target_cluster") == "DECODE_ATTN"
        ]
        if len(group) != 2 or len(a2f_rows) != 1 or len(f2a_rows) != 1:
            errors.append(
                "stage alignment requires exactly one A->F and one F->A "
                f"transfer for identity={key!r}"
            )
            continue
        attn_matches = parsed_attn.get(key, [])
        if len(attn_matches) != 1:
            errors.append(
                f"identity={key!r} must match exactly one DECODE_ATTN stage, "
                f"found={len(attn_matches)}"
            )
            continue
        used_attn_keys.add(key)
        attn_stage = attn_matches[0]
        source_batch_id, request_identity, layer_id, afd_stage_idx = key
        ffn_matches = [
            (index, row)
            for index, row in enumerate(parsed_ffn)
            if source_batch_id in row["_normalized_source_batch_ids"]
            and set(request_identity).issubset(
                row["_normalized_source_request_identity"]
            )
            and row["_normalized_layer_id"] == layer_id
            and row["_normalized_afd_stage_idx"] == afd_stage_idx
        ]
        if not ffn_matches:
            errors.append(
                f"identity={key!r} has no matching DECODE_FFN stage lanes"
            )
            continue
        used_ffn_indices.update(index for index, _ in ffn_matches)

        a2f = a2f_rows[0]
        f2a = f2a_rows[0]
        attn_replica_id = attn_stage["_normalized_replica_id"]
        attn_replica_local_id = attn_stage["replica_local_id"]
        ffn_replica_ids = {
            row["_normalized_replica_id"] for _, row in ffn_matches
        }
        latest_a2f_end: float | None = None
        if len(ffn_replica_ids) != 1:
            errors.append(
                f"identity={key!r} spans multiple DECODE_FFN Replicas: "
                f"{sorted(ffn_replica_ids)}"
            )
        else:
            ffn_replica_id = next(iter(ffn_replica_ids))
            if (
                a2f["_normalized_target_replica_id"] != ffn_replica_id
                or f2a["_normalized_source_replica_id"] != ffn_replica_id
            ):
                errors.append(
                    f"identity={key!r} transfer/DECODE_FFN Replica mismatch"
                )
            source_batch_ids = set().union(
                *(
                    set(row["_normalized_source_batch_ids"])
                    for _, row in ffn_matches
                )
            )
            aggregate_a2f_rows = [
                (candidate_key, candidate)
                for candidate_key, candidate_group in transfer_groups.items()
                if candidate_key[0] in source_batch_ids
                and candidate_key[2] == layer_id
                and candidate_key[3] == afd_stage_idx
                for candidate in candidate_group
                if candidate.get("source_cluster") == "DECODE_ATTN"
                and candidate.get("target_cluster") == "DECODE_FFN"
                and candidate["_normalized_target_replica_id"] == ffn_replica_id
            ]
            if not aggregate_a2f_rows:
                errors.append(
                    f"identity={key!r} has no A->F transfer to "
                    f"DECODE_FFN Replica={ffn_replica_id}"
                )
            else:
                try:
                    latest_a2f_end = max(
                        _finite_number(
                            candidate["end_ts_s"],
                            f"identity={candidate_key!r}.a2f_end_ts_s",
                        )
                        for candidate_key, candidate in aggregate_a2f_rows
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(
                        f"identity={key!r} has invalid aggregate A->F timing: {exc}"
                    )
        if (
            a2f["_normalized_source_replica_id"] != attn_replica_id
            or f2a["_normalized_target_replica_id"] != attn_replica_id
        ):
            errors.append(
                f"identity={key!r} transfer/DECODE_ATTN Replica mismatch"
            )
        if (
            a2f["_normalized_source_replica_local_id"]
            != attn_replica_local_id
            or f2a["_normalized_target_replica_local_id"]
            != attn_replica_local_id
        ):
            errors.append(
                f"identity={key!r} attention local Replica identity mismatch"
            )
        ffn_lane_ids = {
            row["_normalized_replica_local_id"] for _, row in ffn_matches
        }
        if f2a["_normalized_source_replica_local_id"] not in ffn_lane_ids:
            errors.append(
                f"identity={key!r} F->A source EP lane is not a stage participant"
            )

        attn_end = attn_stage["_normalized_stage_end"]
        a2f_start = _finite_number(
            a2f["start_ts_s"], f"identity={key!r}.a2f_start_ts_s"
        )
        a2f_end = _finite_number(
            a2f["end_ts_s"], f"identity={key!r}.a2f_end_ts_s"
        )
        f2a_start = _finite_number(
            f2a["start_ts_s"], f"identity={key!r}.f2a_start_ts_s"
        )
        latest_ffn_end = max(
            row["_normalized_completion_end"] for _, row in ffn_matches
        )
        if abs(attn_end - a2f_start) > 1e-9:
            errors.append(
                f"identity={key!r} DECODE_ATTN end does not equal A->F start"
            )
        for _, ffn_row in ffn_matches:
            source_index = ffn_row["_normalized_source_batch_ids"].index(
                source_batch_id
            )
            emitted_source_arrival = ffn_row[
                "_normalized_source_batch_arrival_times"
            ][source_index]
            if abs(emitted_source_arrival - a2f_end) > 1e-9:
                errors.append(
                    f"identity={key!r} source arrival does not equal A->F end"
                )
        operation_keys = {
            row["_operation_key"] for _, row in ffn_matches
        }
        if len(operation_keys) != 1:
            errors.append(
                f"identity={key!r} matches multiple DECODE_FFN operations"
            )
        elif latest_a2f_end is not None:
            operation_key = next(iter(operation_keys))
            existing_ready = transfer_ready_by_operation.get(operation_key)
            if (
                existing_ready is not None
                and abs(existing_ready - latest_a2f_end) > 1e-9
            ):
                errors.append(
                    f"identity={key!r} disagrees on aggregate A->F readiness"
                )
            transfer_ready_by_operation[operation_key] = latest_a2f_end
        if abs(latest_ffn_end - f2a_start) > 1e-9:
            errors.append(
                f"identity={key!r} DECODE_FFN end does not equal F->A start"
            )

    ffn_operations: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in parsed_ffn:
        ffn_operations.setdefault(row["_operation_key"], []).append(row)

    operations_by_resource: dict[
        tuple[int, int],
        list[tuple[tuple[Any, ...], float, float, float]],
    ] = {}
    for operation_key, operation_rows in ffn_operations.items():
        operation_protocol = operation_rows[0]["_normalized_protocol"]
        operation_lane_ids = {
            row["_normalized_replica_local_id"] for row in operation_rows
        }
        if operation_protocol == "dense":
            if len(operation_rows) != 1 or operation_lane_ids != {None}:
                errors.append(
                    f"DECODE_FFN operation={operation_key!r} dense protocol "
                    "must have exactly one lane-free stage row"
                )
        else:
            expected_lane_ids = set(range(int(expected_moe_ep_size)))
            if (
                len(operation_rows) != int(expected_moe_ep_size)
                or operation_lane_ids != expected_lane_ids
            ):
                errors.append(
                    f"DECODE_FFN operation={operation_key!r} MoE lanes mismatch: "
                    f"expected={sorted(expected_lane_ids)} "
                    f"actual={sorted(operation_lane_ids)}"
                )
        start_times = [
            row["_normalized_stage_start"] for row in operation_rows
        ]
        completion_end_times = [
            row["_normalized_completion_end"] for row in operation_rows
        ]
        emitted_ready_times = [
            row["_normalized_source_group_ready"] for row in operation_rows
        ]
        emitted_source_arrival_times = [
            row["_normalized_source_batch_arrival_times"]
            for row in operation_rows
        ]
        if max(start_times) - min(start_times) > 1e-9:
            errors.append(
                f"DECODE_FFN operation={operation_key!r} lanes do not share one start"
            )
        if max(emitted_ready_times) - min(emitted_ready_times) > 1e-9:
            errors.append(
                f"DECODE_FFN operation={operation_key!r} lanes disagree on readiness"
            )
        if (
            operation_protocol == "moe"
            and max(completion_end_times) - min(completion_end_times) > 1e-9
        ):
            errors.append(
                f"DECODE_FFN operation={operation_key!r} lanes disagree on "
                "observed completion"
            )
        reference_source_arrivals = emitted_source_arrival_times[0]
        if any(
            any(
                abs(actual - expected) > 1e-9
                for actual, expected in zip(
                    source_arrivals,
                    reference_source_arrivals,
                )
            )
            for source_arrivals in emitted_source_arrival_times[1:]
        ):
            errors.append(
                f"DECODE_FFN operation={operation_key!r} lanes disagree on "
                "source arrivals"
            )
        transfer_ready = transfer_ready_by_operation.get(operation_key)
        if transfer_ready is None:
            errors.append(
                f"DECODE_FFN operation={operation_key!r} has no transfer-derived readiness"
            )
            continue
        emitted_ready = emitted_ready_times[0]
        if abs(emitted_ready - max(reference_source_arrivals)) > 1e-9:
            errors.append(
                f"DECODE_FFN operation={operation_key!r} source_group_ready_ts "
                "does not equal max source arrival"
            )
        if abs(emitted_ready - transfer_ready) > 1e-9:
            errors.append(
                f"DECODE_FFN operation={operation_key!r} source_group_ready_ts "
                "does not equal latest A->F end"
            )
        resource_key = (
            operation_rows[0]["_normalized_replica_id"],
            operation_rows[0]["_normalized_stage_id"],
        )
        operations_by_resource.setdefault(resource_key, []).append(
            (
                operation_key,
                min(start_times),
                max(completion_end_times),
                transfer_ready,
            )
        )

    queued_operation_count = 0
    max_queue_delay_ms = 0.0
    for resource_key, operations in operations_by_resource.items():
        previous_operation_end: float | None = None
        for operation_key, stage_start, stage_end, ready_time in sorted(
            operations,
            key=lambda item: (item[1], item[2], repr(item[0])),
        ):
            expected_start = (
                ready_time
                if previous_operation_end is None
                else max(ready_time, previous_operation_end)
            )
            if abs(stage_start - expected_start) > 1e-9:
                errors.append(
                    f"DECODE_FFN operation={operation_key!r} on resource="
                    f"{resource_key!r} starts at {stage_start}, expected "
                    f"max(ready={ready_time}, previous_end="
                    f"{previous_operation_end})={expected_start}"
                )
            queue_delay_ms = max(0.0, (stage_start - ready_time) * 1e3)
            if queue_delay_ms > 1e-6:
                queued_operation_count += 1
            max_queue_delay_ms = max(max_queue_delay_ms, queue_delay_ms)
            previous_operation_end = (
                stage_end
                if previous_operation_end is None
                else max(previous_operation_end, stage_end)
            )

    if len(used_attn_keys) != len(attn_stages):
        errors.append(
            "not every DECODE_ATTN stage has one exact transfer identity: "
            f"matched={len(used_attn_keys)} stages={len(attn_stages)}"
        )
    if len(used_ffn_indices) != len(ffn_stages):
        errors.append(
            "not every DECODE_FFN stage row has one exact transfer identity: "
            f"matched={len(used_ffn_indices)} stages={len(ffn_stages)}"
        )

    if errors:
        return _result(
            "FAIL",
            errors,
            stage_count=len(stages),
            decode_attn_stage_count=len(attn_stages),
            decode_ffn_stage_count=len(ffn_stages),
            decode_ffn_operation_count=len(ffn_operations),
            queued_operation_count=queued_operation_count,
            max_queue_delay_ms=max_queue_delay_ms,
            m2n_count=len(m2n_rows),
        )
    return _result(
        "PASS",
        stage_count=len(stages),
        decode_attn_stage_count=len(attn_stages),
        decode_ffn_stage_count=len(ffn_stages),
        decode_ffn_operation_count=len(ffn_operations),
        queued_operation_count=queued_operation_count,
        max_queue_delay_ms=max_queue_delay_ms,
        m2n_count=len(m2n_rows),
        source_request_ids=sorted(canonical_request_ids),
    )


def validate_transfer_ledger(
    path: str | Path,
    *,
    expected_target_by_transfer_id: Mapping[str, int] | None = None,
    required_transfer_kinds: set[str] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return _result("INSUFFICIENT_EVIDENCE", [f"missing transfer ledger: {path}"])
    try:
        rows = _read_jsonl(path)
    except ValueError as exc:
        return _result("FAIL", [str(exc)])
    return validate_transfer_rows(
        rows,
        expected_target_by_transfer_id=expected_target_by_transfer_id,
        required_transfer_kinds=required_transfer_kinds,
    )


def validate_request_metrics(
    metrics_dir: str | Path,
    *,
    expected_request_count: int | None = None,
    transfer_rows: Iterable[Mapping[str, Any]] | None = None,
    required_transfer_kinds: set[str] | None = None,
) -> dict[str, Any]:
    """Validate request cardinality, units, and timing identities."""

    metrics_dir = Path(metrics_dir)
    normalized_transfer_rows = (
        None
        if transfer_rows is None
        else [dict(row) for row in transfer_rows]
    )
    required_transfer_kinds = set(required_transfer_kinds or ())
    csv_path = metrics_dir / "request_metrics.csv"
    system_path = metrics_dir / "system_metrics.json"
    ground_truth_path = metrics_dir / "metrics_ground_truth.jsonl"
    missing = [
        str(path)
        for path in (csv_path, system_path, ground_truth_path)
        if not path.is_file()
    ]
    if missing:
        return _result("INSUFFICIENT_EVIDENCE", [f"missing request evidence: {missing}"])

    errors: list[str] = []
    unknown_required_kinds = required_transfer_kinds - {"kv_cache", "m2n"}
    if unknown_required_kinds:
        errors.append(
            "unknown required transfer kinds="
            f"{sorted(unknown_required_kinds)}"
        )
    observed_transfer_kinds = {
        str(row.get("transfer_kind"))
        for row in normalized_transfer_rows or ()
    }
    for required_kind in sorted(required_transfer_kinds):
        if required_kind not in observed_transfer_kinds:
            errors.append(
                f"required transfer kind is missing: {required_kind}"
            )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_ids = [row.get("Request Id", "") for row in csv_rows]
    if not csv_rows or any(not value for value in csv_ids):
        errors.append("request_metrics.csv has no complete request rows")
    if len(set(csv_ids)) != len(csv_ids):
        errors.append("request_metrics.csv contains duplicate Request Id values")

    try:
        system_metrics = json.loads(system_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return _result("FAIL", [f"invalid system_metrics.json: {exc}"])
    if not isinstance(system_metrics, Mapping):
        return _result("FAIL", ["system_metrics.json must contain an object"])
    metadata = system_metrics.get("simulation_metadata")
    if not isinstance(metadata, Mapping) or "completed_requests" not in metadata:
        return _result(
            "INSUFFICIENT_EVIDENCE",
            ["system metrics is missing simulation_metadata.completed_requests"],
        )
    try:
        completed_requests = _non_negative_int(
            metadata["completed_requests"],
            "simulation_metadata.completed_requests",
        )
    except ValueError as exc:
        return _result("FAIL", [str(exc)])
    if completed_requests != len(csv_rows):
        errors.append(
            f"completed request cardinality mismatch: system={completed_requests} csv={len(csv_rows)}"
        )
    if expected_request_count is not None and len(csv_rows) != expected_request_count:
        errors.append(
            f"expected request cardinality mismatch: expected={expected_request_count} actual={len(csv_rows)}"
        )

    try:
        truth_rows = _read_jsonl(ground_truth_path)
    except ValueError as exc:
        return _result("FAIL", [str(exc)])
    completion_events = [
        row for row in truth_rows if row.get("event_type") == "request_completion"
    ]
    completion_counts: dict[str, int] = {}
    completion_rows: dict[str, dict[str, Any]] = {}
    for row in completion_events:
        if "request_id" not in row:
            errors.append("request_completion row is missing request_id")
            continue
        request_id = str(row["request_id"])
        completion_counts[request_id] = completion_counts.get(request_id, 0) + 1
        completion_rows.setdefault(request_id, row)
    duplicate_completion_ids = sorted(
        request_id
        for request_id, count in completion_counts.items()
        if count != 1
    )
    if duplicate_completion_ids:
        errors.append(
            "duplicate request_completion rows for request IDs="
            f"{duplicate_completion_ids}"
        )
    if len(completion_events) != len(csv_rows):
        errors.append(
            "request completion cardinality mismatch: "
            f"truth={len(completion_events)} csv={len(csv_rows)}"
        )
    if set(completion_rows) != set(csv_ids):
        errors.append(
            f"ground-truth/request-metrics ID mismatch: truth={sorted(completion_rows)} csv={sorted(csv_ids)}"
        )

    csv_by_id = {str(row["Request Id"]): row for row in csv_rows}
    observed_request_metrics: dict[str, dict[str, float | int]] = {}
    for request_id, truth in completion_rows.items():
        try:
            arrived = _finite_number(truth["arrived_at"], f"{request_id}.arrived_at")
            prefill_completed = _finite_number(
                truth["prefill_completed_at"], f"{request_id}.prefill_completed_at"
            )
            first_decode_completed = _finite_number(
                truth["first_decode_token_completed_at"],
                f"{request_id}.first_decode_token_completed_at",
            )
            completed = _finite_number(truth["completed_at"], f"{request_id}.completed_at")
            e2e_s = _finite_number(truth["request_e2e_time_s"], f"{request_id}.request_e2e_time_s")
            ttft_s = _finite_number(truth["ttft_s"], f"{request_id}.ttft_s")
            tpot_s = _finite_number(truth["tpot_s"], f"{request_id}.tpot_s")
            num_prefill_tokens = _non_negative_int(
                truth["num_prefill_tokens"], f"{request_id}.num_prefill_tokens"
            )
            num_decode_tokens = _non_negative_int(
                truth["num_decode_tokens"], f"{request_id}.num_decode_tokens"
            )
            if num_prefill_tokens == 0 or num_decode_tokens == 0:
                raise ValueError("prefill/decode token counts must both be positive")
            if not (
                arrived <= prefill_completed + 1e-12
                and prefill_completed <= first_decode_completed + 1e-12
                and first_decode_completed <= completed + 1e-12
            ):
                raise ValueError("request timing order is invalid")
            if abs(e2e_s - (completed - arrived)) > 1e-9:
                errors.append(f"{request_id}: E2E decomposition mismatch")
            if abs(ttft_s - (prefill_completed - arrived)) > 1e-9:
                errors.append(f"{request_id}: TTFT decomposition mismatch")
            expected_tpot_s = (
                (completed - first_decode_completed) / (num_decode_tokens - 1)
                if num_decode_tokens > 1
                else 0.0
            )
            if abs(tpot_s - expected_tpot_s) > 1e-9:
                errors.append(
                    f"{request_id}: TPOT decomposition mismatch: "
                    f"expected_s={expected_tpot_s} actual_s={tpot_s}"
                )
            csv_row = csv_by_id.get(request_id)
            if csv_row is None:
                continue
            for csv_name, truth_name in (
                ("request_e2e_time", "request_e2e_time_s"),
                ("ttft", "ttft_s"),
                ("tpot", "tpot_s"),
            ):
                if csv_name not in csv_row or truth_name not in truth:
                    errors.append(f"{request_id}: missing {csv_name}/{truth_name}")
                    continue
                csv_ms = _finite_number(csv_row[csv_name], f"{request_id}.{csv_name}")
                truth_ms = _finite_number(truth[truth_name], f"{request_id}.{truth_name}") * 1000.0
                if abs(csv_ms - truth_ms) > 1e-6:
                    errors.append(
                        f"{request_id}: unit/value mismatch for {csv_name}: csv_ms={csv_ms} truth_ms={truth_ms}"
                    )
            csv_num_prefill = _non_negative_int(
                int(csv_row["request_num_prefill_tokens"]),
                f"{request_id}.request_num_prefill_tokens",
            )
            csv_num_decode = _non_negative_int(
                int(csv_row["request_num_decode_tokens"]),
                f"{request_id}.request_num_decode_tokens",
            )
            csv_num_total = _non_negative_int(
                int(csv_row["request_num_tokens"]),
                f"{request_id}.request_num_tokens",
            )
            if (
                csv_num_prefill != num_prefill_tokens
                or csv_num_decode != num_decode_tokens
                or csv_num_total != num_prefill_tokens + num_decode_tokens
            ):
                errors.append(
                    f"{request_id}: request token cardinality mismatch"
                )

            decode_e2e_ms = _finite_number(
                csv_row["decode_e2e_time"], f"{request_id}.decode_e2e_time"
            )
            expected_decode_e2e_ms = (completed - prefill_completed) * 1000.0
            if abs(decode_e2e_ms - expected_decode_e2e_ms) > 1e-6:
                errors.append(
                    f"{request_id}: decode E2E decomposition mismatch: "
                    f"expected_ms={expected_decode_e2e_ms} actual_ms={decode_e2e_ms}"
                )

            tpot_ms = _finite_number(csv_row["tpot"], f"{request_id}.tpot")
            tpot_computation_ms = _finite_number(
                csv_row["tpot_computation"], f"{request_id}.tpot_computation"
            )
            tpot_transfer_ms = _finite_number(
                csv_row["tpot_transfer"], f"{request_id}.tpot_transfer"
            )
            if abs(tpot_ms - (tpot_computation_ms + tpot_transfer_ms)) > 1e-6:
                errors.append(
                    f"{request_id}: TPOT computation/transfer decomposition mismatch"
                )

            csv_kv_ms = _finite_number(
                csv_row["transfer_kv_cache"], f"{request_id}.transfer_kv_cache"
            )
            csv_m2n_total_ms = _finite_number(
                csv_row["transfer_m2n_total"], f"{request_id}.transfer_m2n_total"
            )
            csv_m2n_a2f_ms = _finite_number(
                csv_row["transfer_m2n_attn_to_ffn"],
                f"{request_id}.transfer_m2n_attn_to_ffn",
            )
            csv_m2n_f2a_ms = _finite_number(
                csv_row["transfer_m2n_ffn_to_attn"],
                f"{request_id}.transfer_m2n_ffn_to_attn",
            )
            if abs(csv_m2n_total_ms - (csv_m2n_a2f_ms + csv_m2n_f2a_ms)) > 1e-6:
                errors.append(
                    f"{request_id}: M2N direction decomposition mismatch"
                )

            if normalized_transfer_rows is not None:
                request_transfers = [
                    row
                    for row in normalized_transfer_rows
                    if request_id in {
                        str(value) for value in row.get("request_ids", ())
                    }
                ]
                request_transfer_kinds = {
                    str(row.get("transfer_kind"))
                    for row in request_transfers
                }
                for required_kind in sorted(required_transfer_kinds):
                    if required_kind not in request_transfer_kinds:
                        errors.append(
                            f"{request_id}: required transfer kind is missing: "
                            f"{required_kind}"
                        )
                kv_ms = sum(
                    _finite_number(
                        row["duration_ms"],
                        f"{request_id}.{row.get('transfer_id')}.duration_ms",
                    )
                    for row in request_transfers
                    if row.get("transfer_kind") == "kv_cache"
                )
                m2n_a2f_ms = sum(
                    _finite_number(
                        row["duration_ms"],
                        f"{request_id}.{row.get('transfer_id')}.duration_ms",
                    )
                    for row in request_transfers
                    if row.get("transfer_kind") == "m2n"
                    and row.get("source_cluster") == "DECODE_ATTN"
                )
                m2n_f2a_ms = sum(
                    _finite_number(
                        row["duration_ms"],
                        f"{request_id}.{row.get('transfer_id')}.duration_ms",
                    )
                    for row in request_transfers
                    if row.get("transfer_kind") == "m2n"
                    and row.get("source_cluster") == "DECODE_FFN"
                )
                m2n_total_ms = m2n_a2f_ms + m2n_f2a_ms
                for label, actual, expected in (
                    ("KV", csv_kv_ms, kv_ms),
                    ("M2N total", csv_m2n_total_ms, m2n_total_ms),
                    ("M2N A->F", csv_m2n_a2f_ms, m2n_a2f_ms),
                    ("M2N F->A", csv_m2n_f2a_ms, m2n_f2a_ms),
                ):
                    if abs(actual - expected) > 1e-6:
                        errors.append(
                            f"{request_id}: {label} request/ledger mismatch: "
                            f"csv_ms={actual} ledger_ms={expected}"
                        )
                expected_tpot_transfer_ms = (
                    m2n_total_ms / (num_decode_tokens - 1)
                    if num_decode_tokens > 1
                    else 0.0
                )
                if abs(tpot_transfer_ms - expected_tpot_transfer_ms) > 1e-6:
                    errors.append(
                        f"{request_id}: TPOT transfer/ledger mismatch: "
                        f"csv_ms={tpot_transfer_ms} "
                        f"ledger_ms={expected_tpot_transfer_ms}"
                    )

            observed_request_metrics[request_id] = {
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "e2e_ms": e2e_s * 1000.0,
                "ttft_ms": ttft_s * 1000.0,
                "tpot_ms": tpot_s * 1000.0,
                "kv_transfer_ms": csv_kv_ms,
                "m2n_transfer_ms": csv_m2n_total_ms,
            }
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{request_id}: {exc}")

    if normalized_transfer_rows is not None:
        transfer_statistics = (
            ("kv_cache", "kv_cache_transfer_statistics"),
            ("m2n", "m2n_transfer_statistics"),
        )
        for transfer_kind, statistics_key in transfer_statistics:
            rows = [
                row
                for row in normalized_transfer_rows
                if row.get("transfer_kind") == transfer_kind
            ]
            if not rows:
                continue
            statistics = system_metrics.get(statistics_key)
            if not isinstance(statistics, Mapping):
                errors.append(f"system metrics missing {statistics_key}")
                continue
            expected_count = len(rows)
            expected_bytes = sum(int(row["bytes"]) for row in rows)
            expected_duration_ms = sum(
                _finite_number(
                    row["duration_ms"], f"{row.get('transfer_id')}.duration_ms"
                )
                for row in rows
            )
            if int(statistics.get("total_transfers", -1)) != expected_count:
                errors.append(
                    f"{transfer_kind}: system/ledger transfer count mismatch"
                )
            if (
                int(statistics.get("total_data_transferred_bytes", -1))
                != expected_bytes
            ):
                errors.append(
                    f"{transfer_kind}: system/ledger transfer byte mismatch"
                )
            actual_duration_ms = _finite_number(
                statistics.get("total_transfer_time_ms"),
                f"{statistics_key}.total_transfer_time_ms",
            )
            if abs(actual_duration_ms - expected_duration_ms) > 1e-6:
                errors.append(
                    f"{transfer_kind}: system/ledger transfer duration mismatch"
                )
            if transfer_kind == "m2n":
                expected_a2f = sum(
                    1 for row in rows if row.get("source_cluster") == "DECODE_ATTN"
                )
                expected_f2a = sum(
                    1 for row in rows if row.get("source_cluster") == "DECODE_FFN"
                )
                if int(statistics.get("attn_to_ffn_transfers", -1)) != expected_a2f:
                    errors.append("m2n: A->F system/ledger count mismatch")
                if int(statistics.get("ffn_to_attn_transfers", -1)) != expected_f2a:
                    errors.append("m2n: F->A system/ledger count mismatch")

    aggregate_missing: list[str] = []
    aggregate_details: dict[str, Any] = {}
    if expected_request_count is not None:
        aggregate_missing, aggregate_failures, aggregate_details = (
            _validate_strict_system_aggregates(
                system_metrics,
                completion_rows,
                (
                    row
                    for row in truth_rows
                    if row.get("event_type") == "request_arrival"
                ),
                csv_rows,
                expected_request_count=expected_request_count,
            )
        )
        errors.extend(aggregate_failures)

    details = {
        "request_count": len(csv_rows),
        "request_metrics": observed_request_metrics,
        **aggregate_details,
    }
    if errors:
        return _result("FAIL", [*errors, *aggregate_missing], **details)
    if aggregate_missing:
        return _result("INSUFFICIENT_EVIDENCE", aggregate_missing, **details)
    return _result("PASS", **details)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-ledger", type=Path)
    parser.add_argument("--stage-ledger", type=Path)
    parser.add_argument("--metrics-dir", type=Path)
    parser.add_argument("--expected-request-count", type=int)
    parser.add_argument(
        "--required-transfer-kind",
        action="append",
        choices=("kv_cache", "m2n"),
        default=[],
    )
    parser.add_argument("--expected-layer-protocols-json")
    parser.add_argument("--expected-moe-ep-size", type=int)
    args = parser.parse_args()
    if args.transfer_ledger is None and args.metrics_dir is None and args.stage_ledger is None:
        parser.error("provide --transfer-ledger, --stage-ledger, and/or --metrics-dir")
    expected_layer_protocols = None
    if args.expected_layer_protocols_json is not None:
        try:
            protocol_values = json.loads(args.expected_layer_protocols_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--expected-layer-protocols-json is invalid JSON: {exc}")
        if type(protocol_values) is not list or not protocol_values:
            parser.error(
                "--expected-layer-protocols-json must be a non-empty JSON array"
            )
        expected_layer_protocols = dict(enumerate(protocol_values))
    required_transfer_kinds = set(args.required_transfer_kind)
    if required_transfer_kinds and args.transfer_ledger is None:
        parser.error("--required-transfer-kind requires --transfer-ledger")
    if args.expected_request_count is not None and args.metrics_dir is None:
        parser.error("--expected-request-count requires --metrics-dir")
    if expected_layer_protocols is not None and args.transfer_ledger is None:
        parser.error(
            "--expected-layer-protocols-json requires --transfer-ledger"
        )
    if (
        args.stage_ledger is None
        and (
            "kv_cache" in required_transfer_kinds
            or expected_layer_protocols is not None
        )
    ):
        parser.error(
            "--stage-ledger is required for KV or expected layer-protocol "
            "validation"
        )
    has_moe_protocol = (
        expected_layer_protocols is not None
        and "moe" in expected_layer_protocols.values()
    )
    if args.expected_moe_ep_size is not None and not has_moe_protocol:
        parser.error(
            "--expected-moe-ep-size requires a MoE entry in "
            "--expected-layer-protocols-json"
        )
    if (
        has_moe_protocol
        and args.expected_moe_ep_size is None
    ):
        parser.error(
            "MoE --stage-ledger validation requires --expected-moe-ep-size"
        )
    if (
        args.stage_ledger is not None
        and expected_layer_protocols is None
        and "kv_cache" not in required_transfer_kinds
    ):
        parser.error(
            "--stage-ledger requires --expected-layer-protocols-json and/or "
            "--required-transfer-kind kv_cache"
        )
    results = {}
    transfer_rows = None
    if args.transfer_ledger is not None:
        results["transfer"] = validate_transfer_ledger(
            args.transfer_ledger,
            required_transfer_kinds=required_transfer_kinds,
        )
        transfer_path = Path(args.transfer_ledger)
        if transfer_path.is_file():
            try:
                transfer_rows = _read_jsonl(transfer_path)
            except ValueError as exc:
                results["transfer"] = _result("FAIL", [str(exc)])
    if args.metrics_dir is not None:
        results["request_metrics"] = validate_request_metrics(
            args.metrics_dir,
            expected_request_count=args.expected_request_count,
            transfer_rows=transfer_rows,
            required_transfer_kinds=required_transfer_kinds,
        )
    if args.stage_ledger is not None:
        stage_path = Path(args.stage_ledger)
        if not stage_path.is_file():
            if "kv_cache" in required_transfer_kinds:
                results["kv_stage"] = _result(
                    "INSUFFICIENT_EVIDENCE",
                    [f"missing stage ledger: {stage_path}"],
                )
            if expected_layer_protocols is not None:
                results["stage_transfer"] = _result(
                    "INSUFFICIENT_EVIDENCE",
                    [f"missing stage ledger: {stage_path}"],
                )
        elif transfer_rows is None:
            if "kv_cache" in required_transfer_kinds:
                results["kv_stage"] = _result(
                    "INSUFFICIENT_EVIDENCE",
                    ["KV stage validation requires --transfer-ledger"],
                )
            if expected_layer_protocols is not None:
                results["stage_transfer"] = _result(
                    "INSUFFICIENT_EVIDENCE",
                    ["stage-transfer validation requires --transfer-ledger"],
                )
        else:
            try:
                stage_rows = _read_jsonl(stage_path)
                if any(
                    row.get("transfer_kind") == "kv_cache"
                    for row in transfer_rows
                ):
                    results["kv_stage"] = validate_kv_stage_alignment(
                        stage_rows,
                        transfer_rows,
                    )
                if expected_layer_protocols is not None:
                    results["stage_transfer"] = validate_stage_transfer_alignment(
                        stage_rows,
                        transfer_rows,
                        expected_layer_protocols=expected_layer_protocols,
                        expected_moe_ep_size=args.expected_moe_ep_size,
                    )
            except ValueError as exc:
                result_key = (
                    "stage_transfer"
                    if expected_layer_protocols is not None
                    else "kv_stage"
                )
                results[result_key] = _result("FAIL", [str(exc)])
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(result["status"] == "PASS" for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
