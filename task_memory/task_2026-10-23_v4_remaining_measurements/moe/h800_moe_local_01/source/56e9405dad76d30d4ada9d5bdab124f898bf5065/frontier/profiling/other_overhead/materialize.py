"""Materialize PP stage-boundary overhead CSVs from raw traces."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from frontier.profiling.other_overhead.backends.vllm_pp_stage_boundary_mapping import (
    _pairing_key,
    build_pp_producer_send_path_row,
    build_pp_receiver_head_row,
    build_pp_stage_boundary_overhead_row,
    canonicalize_pp_stage_trace_model_name,
    normalize_vllm_pp_stage_trace_record,
    pair_adjacent_vllm_pp_stage_trace_records,
)
from frontier.profiling.other_overhead.schema import (
    PP_PRODUCER_SEND_PATH_AUDIT_COLUMNS,
    PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS,
    PP_PRODUCER_SEND_PATH_TARGET_COLUMNS,
    PP_RECEIVER_HEAD_AUDIT_COLUMNS,
    PP_RECEIVER_HEAD_IDENTITY_COLUMNS,
    PP_RECEIVER_HEAD_TARGET_COLUMNS,
    PP_STAGE_BOUNDARY_IDENTITY_COLUMNS,
    PP_STAGE_BOUNDARY_NUMERIC_COLUMNS,
)
from frontier.profiling.other_overhead.validation import (
    validate_pp_producer_send_path_dataframe,
    validate_pp_receiver_head_dataframe,
    validate_pp_stage_boundary_dataframe,
)


def _load_jsonl_records(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL path does not exist: {jsonl_path}")
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"JSONL path is empty: {jsonl_path}")
    return rows


def _localize_synced_stage_model_path(
    raw_record: dict[str, Any],
    *,
    raw_trace_jsonl: Path,
) -> dict[str, Any]:
    raw_model_name = str(raw_record.get("model_name", "")).strip()
    if not raw_model_name:
        return raw_record

    if canonicalize_pp_stage_trace_model_name(raw_model_name) != raw_model_name:
        return raw_record

    candidate_path = Path(raw_model_name)
    if candidate_path.exists():
        return raw_record

    local_candidate = raw_trace_jsonl.parent / candidate_path.name
    if not local_candidate.exists():
        return raw_record

    localized_record = dict(raw_record)
    localized_record["model_name"] = str(local_candidate)
    return localized_record


def _normalize_stage_records(
    *,
    raw_trace_jsonl: Path,
    raw_stage_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        normalize_vllm_pp_stage_trace_record(
            _localize_synced_stage_model_path(
                raw_record,
                raw_trace_jsonl=raw_trace_jsonl,
            )
        )
        for raw_record in raw_stage_records
    ]


def _frontier_wire_lookup_key(model_name: str, total_tokens: int) -> tuple[str, int]:
    return (canonicalize_pp_stage_trace_model_name(model_name), int(total_tokens))


def _record_sort_ts(value: Any) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _dedupe_normalized_stage_records(
    normalized_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduped: dict[tuple[tuple[Any, ...], int], dict[str, Any]] = {}
    for record in normalized_records:
        pair_key = (_pairing_key(record), int(record["pp_rank"]))
        existing = deduped.get(pair_key)
        if existing is None:
            deduped[pair_key] = record
            continue

        existing_order_key = (
            _record_sort_ts(existing.get("timestamp")),
            _record_sort_ts(existing.get("recv_end_ts")),
            _record_sort_ts(existing.get("preprocess_end_ts")),
            _record_sort_ts(existing.get("forward_end_ts")),
            _record_sort_ts(existing.get("send_end_ts")),
        )
        candidate_order_key = (
            _record_sort_ts(record.get("timestamp")),
            _record_sort_ts(record.get("recv_end_ts")),
            _record_sort_ts(record.get("preprocess_end_ts")),
            _record_sort_ts(record.get("forward_end_ts")),
            _record_sort_ts(record.get("send_end_ts")),
        )
        if candidate_order_key >= existing_order_key:
            deduped[pair_key] = record

    return list(deduped.values())


def _decode_activation_bytes_per_token_by_layout(
    adjacent_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[tuple[str, int, int, int], float]:
    bytes_per_token_by_layout: dict[tuple[str, int, int, int], float] = {}
    for _, consumer in adjacent_pairs:
        num_prefill_tokens = int(consumer["num_prefill_tokens"])
        num_decode_tokens = int(consumer["num_decode_tokens"])
        activation_bytes_per_rank = int(consumer["activation_bytes_per_rank"])
        if (
            num_prefill_tokens != 0
            or num_decode_tokens <= 0
            or activation_bytes_per_rank <= 0
        ):
            continue

        layout_key = (
            str(consumer["model_name"]),
            int(consumer["tensor_parallel_degree"]),
            int(consumer["pp_world_size"]),
            int(consumer["pp_rank"]),
        )
        bytes_per_token = activation_bytes_per_rank / float(num_decode_tokens)
        existing = bytes_per_token_by_layout.get(layout_key)
        if existing is None or bytes_per_token < existing:
            bytes_per_token_by_layout[layout_key] = bytes_per_token
    return bytes_per_token_by_layout


def _is_mtp_lookahead_decode_consumer_active_pair(
    consumer: dict[str, Any],
    *,
    decode_activation_bytes_per_token_by_layout: dict[tuple[str, int, int, int], float],
) -> bool:
    num_prefill_tokens = int(consumer["num_prefill_tokens"])
    num_decode_tokens = int(consumer["num_decode_tokens"])
    if num_prefill_tokens != 0 or num_decode_tokens <= 0:
        return False

    layout_key = (
        str(consumer["model_name"]),
        int(consumer["tensor_parallel_degree"]),
        int(consumer["pp_world_size"]),
        int(consumer["pp_rank"]),
    )
    baseline_bytes_per_token = decode_activation_bytes_per_token_by_layout.get(
        layout_key
    )
    if baseline_bytes_per_token is None:
        return False

    expected_activation_bytes = baseline_bytes_per_token * float(num_decode_tokens)
    return float(consumer["activation_bytes_per_rank"]) > expected_activation_bytes


def build_frontier_pp_wire_mean_lookup(
    frontier_op_trace_records: list[dict[str, Any]],
) -> dict[tuple[str, int], float]:
    durations_by_key: dict[tuple[str, int], list[float]] = {}
    for row in frontier_op_trace_records:
        if row.get("name") != "pipeline_parallel_send_recv":
            continue
        meta = row.get("meta", {})
        model_name = str(meta.get("model_name", "")).strip()
        if not model_name:
            continue
        total_tokens = int(
            meta.get(
                "effective_total_tokens_transfer",
                meta.get(
                    "effective_total_tokens_rounded",
                    meta.get("total_tokens", 0),
                ),
            )
        )
        key = _frontier_wire_lookup_key(model_name, total_tokens)
        durations_by_key.setdefault(key, []).append(float(row["duration_ms"]))
    if not durations_by_key:
        raise ValueError("No pipeline_parallel_send_recv rows found in Frontier op trace.")
    return {key: mean(values) for key, values in durations_by_key.items()}


def materialize_pp_stage_boundary_rows(
    *,
    raw_trace_jsonl: Path,
    frontier_op_trace_jsonl: Path,
) -> list[dict[str, Any]]:
    raw_stage_records = _load_jsonl_records(raw_trace_jsonl)
    frontier_op_trace_records = _load_jsonl_records(frontier_op_trace_jsonl)

    normalized_stage_records = _normalize_stage_records(
        raw_trace_jsonl=raw_trace_jsonl,
        raw_stage_records=raw_stage_records,
    )
    normalized_stage_records = _dedupe_normalized_stage_records(
        normalized_stage_records
    )
    adjacent_pairs = pair_adjacent_vllm_pp_stage_trace_records(normalized_stage_records)
    if not adjacent_pairs:
        raise ValueError(
            "No adjacent PP stage trace pairs were produced from the raw trace."
        )

    wire_lookup = build_frontier_pp_wire_mean_lookup(frontier_op_trace_records)
    rows: list[dict[str, Any]] = []
    for producer, consumer in adjacent_pairs:
        total_tokens = int(consumer["num_prefill_tokens"]) + int(
            consumer["num_decode_tokens"]
        )
        lookup_key = _frontier_wire_lookup_key(producer["model_name"], total_tokens)
        wire_ms = wire_lookup.get(lookup_key)
        if wire_ms is None:
            raise ValueError(
                "Missing Frontier PP wire mean for PP stage-boundary row: "
                f"model_name={producer['model_name']}, total_tokens={total_tokens}."
            )
        rows.append(
            build_pp_stage_boundary_overhead_row(
                producer,
                consumer,
                existing_pp_send_recv_wire_ms=wire_ms,
            )
        )
    return rows


def aggregate_pp_stage_boundary_rows(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    if not rows:
        raise ValueError("PP stage-boundary rows must not be empty before aggregation.")

    raw_df = pd.DataFrame(rows)
    measurement_columns = [
        column
        for column in PP_STAGE_BOUNDARY_NUMERIC_COLUMNS
        if column not in PP_STAGE_BOUNDARY_IDENTITY_COLUMNS
    ]
    aggregated = (
        raw_df.groupby(list(PP_STAGE_BOUNDARY_IDENTITY_COLUMNS), as_index=False, sort=False)[
            measurement_columns
        ]
        .mean()
    )
    return aggregated


def materialize_pp_stage_boundary_csv(
    *,
    raw_trace_jsonl: Path,
    frontier_op_trace_jsonl: Path,
    output_csv: Path,
) -> pd.DataFrame:
    rows = materialize_pp_stage_boundary_rows(
        raw_trace_jsonl=raw_trace_jsonl,
        frontier_op_trace_jsonl=frontier_op_trace_jsonl,
    )
    aggregated = aggregate_pp_stage_boundary_rows(rows)
    validated = validate_pp_stage_boundary_dataframe(aggregated)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_csv, index=False)
    return validated


def materialize_pp_receiver_head_rows(
    *,
    raw_trace_jsonl: Path,
) -> list[dict[str, Any]]:
    raw_stage_records = _load_jsonl_records(raw_trace_jsonl)
    normalized_stage_records = _normalize_stage_records(
        raw_trace_jsonl=raw_trace_jsonl,
        raw_stage_records=raw_stage_records,
    )
    normalized_stage_records = _dedupe_normalized_stage_records(
        normalized_stage_records
    )
    adjacent_pairs = pair_adjacent_vllm_pp_stage_trace_records(normalized_stage_records)
    if not adjacent_pairs:
        raise ValueError(
            "No adjacent PP stage trace pairs were produced from the raw trace."
        )

    previous_forward_end_by_consumer_rank: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    sorted_pairs = sorted(
        adjacent_pairs,
        key=lambda pair: (
            int(pair[1]["pp_rank"]),
            float(pair[1]["timestamp"]),
            int(pair[1]["batch_id"]),
        ),
    )
    for producer, consumer in sorted_pairs:
        consumer_rank = int(consumer["pp_rank"])
        previous_consumer_forward_end_ts_same_stage = (
            previous_forward_end_by_consumer_rank.get(consumer_rank)
        )
        current_consumer_forward_end_ts = float(consumer["forward_end_ts"])
        is_decode_only_pair = (
            int(consumer["num_decode_tokens"]) > 0
            and int(consumer["num_prefill_tokens"]) == 0
        )
        if not is_decode_only_pair or consumer_rank <= 0:
            previous_forward_end_by_consumer_rank[consumer_rank] = (
                current_consumer_forward_end_ts
            )
            continue

        row = build_pp_receiver_head_row(
            producer,
            consumer,
            previous_consumer_forward_end_ts_same_stage=(
                previous_consumer_forward_end_ts_same_stage
            ),
        )
        previous_forward_end_by_consumer_rank[consumer_rank] = (
            current_consumer_forward_end_ts
        )
        rows.append(row)

    if not rows:
        raise ValueError(
            "No decode PP receiver-head rows were produced from the raw trace."
        )
    return rows


def materialize_pp_prefill_consumer_active_rows(
    *,
    raw_trace_jsonl: Path,
) -> list[dict[str, Any]]:
    raw_stage_records = _load_jsonl_records(raw_trace_jsonl)
    normalized_stage_records = _normalize_stage_records(
        raw_trace_jsonl=raw_trace_jsonl,
        raw_stage_records=raw_stage_records,
    )
    normalized_stage_records = _dedupe_normalized_stage_records(
        normalized_stage_records
    )
    adjacent_pairs = pair_adjacent_vllm_pp_stage_trace_records(normalized_stage_records)
    if not adjacent_pairs:
        raise ValueError(
            "No adjacent PP stage trace pairs were produced from the raw trace."
        )
    decode_activation_bytes_per_token_by_layout = (
        _decode_activation_bytes_per_token_by_layout(adjacent_pairs)
    )

    previous_forward_end_by_consumer_rank: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    sorted_pairs = sorted(
        adjacent_pairs,
        key=lambda pair: (
            int(pair[1]["pp_rank"]),
            float(pair[1]["timestamp"]),
            int(pair[1]["batch_id"]),
        ),
    )
    for producer, consumer in sorted_pairs:
        consumer_rank = int(consumer["pp_rank"])
        previous_consumer_forward_end_ts_same_stage = (
            previous_forward_end_by_consumer_rank.get(consumer_rank)
        )
        current_consumer_forward_end_ts = float(consumer["forward_end_ts"])
        is_prefill_only_pair = (
            int(consumer["num_prefill_tokens"]) > 0
            and int(consumer["num_decode_tokens"]) == 0
        )
        is_mtp_lookahead_decode_pair = (
            _is_mtp_lookahead_decode_consumer_active_pair(
                consumer,
                decode_activation_bytes_per_token_by_layout=(
                    decode_activation_bytes_per_token_by_layout
                ),
            )
        )
        if (
            (not is_prefill_only_pair and not is_mtp_lookahead_decode_pair)
            or consumer_rank <= 0
        ):
            previous_forward_end_by_consumer_rank[consumer_rank] = (
                current_consumer_forward_end_ts
            )
            continue

        row = build_pp_receiver_head_row(
            producer,
            consumer,
            previous_consumer_forward_end_ts_same_stage=(
                previous_consumer_forward_end_ts_same_stage
            ),
        )
        previous_forward_end_by_consumer_rank[consumer_rank] = (
            current_consumer_forward_end_ts
        )
        if is_mtp_lookahead_decode_pair:
            row["other_overhead_source"] = (
                "vllm_mtp_lookahead_consumer_active_replay"
            )
        else:
            row["other_overhead_source"] = "vllm_prefill_consumer_active_replay"
        rows.append(row)

    if not rows:
        raise ValueError(
            "No prefill PP consumer-active rows were produced from the raw trace."
        )
    return rows


def materialize_pp_producer_send_path_rows(
    *,
    raw_trace_jsonl: Path,
    frontier_op_trace_jsonl: Path,
) -> list[dict[str, Any]]:
    raw_stage_records = _load_jsonl_records(raw_trace_jsonl)
    frontier_op_trace_records = _load_jsonl_records(frontier_op_trace_jsonl)
    normalized_stage_records = _normalize_stage_records(
        raw_trace_jsonl=raw_trace_jsonl,
        raw_stage_records=raw_stage_records,
    )
    normalized_stage_records = _dedupe_normalized_stage_records(
        normalized_stage_records
    )
    wire_lookup = build_frontier_pp_wire_mean_lookup(frontier_op_trace_records)

    rows: list[dict[str, Any]] = []
    for producer in normalized_stage_records:
        if bool(producer["is_last_rank"]):
            continue
        if int(producer["num_prefill_tokens"]) <= 0:
            continue
        if int(producer["num_decode_tokens"]) != 0:
            continue

        total_tokens = int(producer["num_prefill_tokens"]) + int(
            producer["num_decode_tokens"]
        )
        lookup_key = _frontier_wire_lookup_key(producer["model_name"], total_tokens)
        wire_ms = wire_lookup.get(lookup_key)
        if wire_ms is None:
            raise ValueError(
                "Missing Frontier PP wire mean for PP producer send-path row: "
                f"model_name={producer['model_name']}, total_tokens={total_tokens}."
            )
        rows.append(
            build_pp_producer_send_path_row(
                producer,
                existing_pp_send_recv_wire_ms=wire_ms,
            )
        )

    if not rows:
        raise ValueError(
            "No prefill PP producer send-path rows were produced from the raw trace."
        )
    return rows


def aggregate_pp_receiver_head_rows(
    rows: list[dict[str, Any]],
    *,
    profiling_precision: str,
) -> pd.DataFrame:
    if not rows:
        raise ValueError("PP receiver-head rows must not be empty before aggregation.")

    raw_df = pd.DataFrame(rows)
    source_agg = (
        {"other_overhead_source": ("other_overhead_source", "first")}
        if "other_overhead_source" in raw_df.columns
        else {}
    )
    aggregated = (
        raw_df.groupby(list(PP_RECEIVER_HEAD_IDENTITY_COLUMNS), as_index=False, sort=False)
        .agg(
            producer_pp_rank=("producer_pp_rank", "first"),
            sample_count=("producer_pp_rank", "size"),
            **source_agg,
            **{
                column: (column, "mean")
                for column in (*PP_RECEIVER_HEAD_TARGET_COLUMNS, *PP_RECEIVER_HEAD_AUDIT_COLUMNS)
            },
        )
    )
    aggregated["profiling_precision"] = str(profiling_precision)
    if "other_overhead_source" not in aggregated.columns:
        aggregated["other_overhead_source"] = "vllm_receiver_head_replay"
    return aggregated


def aggregate_pp_prefill_consumer_active_rows(
    rows: list[dict[str, Any]],
    *,
    profiling_precision: str,
) -> pd.DataFrame:
    aggregated = aggregate_pp_receiver_head_rows(
        rows,
        profiling_precision=profiling_precision,
    )
    if (aggregated["other_overhead_source"] == "vllm_receiver_head_replay").any():
        aggregated.loc[
            aggregated["other_overhead_source"] == "vllm_receiver_head_replay",
            "other_overhead_source",
        ] = "vllm_prefill_consumer_active_replay"
    return aggregated


def aggregate_pp_producer_send_path_rows(
    rows: list[dict[str, Any]],
    *,
    profiling_precision: str,
) -> pd.DataFrame:
    if not rows:
        raise ValueError(
            "PP producer send-path rows must not be empty before aggregation."
        )

    raw_df = pd.DataFrame(rows)
    aggregated = (
        raw_df.groupby(
            list(PP_PRODUCER_SEND_PATH_IDENTITY_COLUMNS), as_index=False, sort=False
        )
        .agg(
            consumer_pp_rank=("consumer_pp_rank", "first"),
            sample_count=("consumer_pp_rank", "size"),
            **{
                column: (column, "mean")
                for column in (
                    *PP_PRODUCER_SEND_PATH_TARGET_COLUMNS,
                    *PP_PRODUCER_SEND_PATH_AUDIT_COLUMNS,
                )
            },
        )
    )
    aggregated["profiling_precision"] = str(profiling_precision)
    aggregated["other_overhead_source"] = "vllm_producer_send_path_replay"
    return aggregated


def materialize_pp_receiver_head_csv(
    *,
    raw_trace_jsonl: Path,
    output_csv: Path,
    profiling_precision: str,
) -> pd.DataFrame:
    rows = materialize_pp_receiver_head_rows(raw_trace_jsonl=raw_trace_jsonl)
    aggregated = aggregate_pp_receiver_head_rows(
        rows,
        profiling_precision=profiling_precision,
    )
    validated = validate_pp_receiver_head_dataframe(aggregated)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_csv, index=False)
    return validated


def materialize_pp_prefill_consumer_active_csv(
    *,
    raw_trace_jsonl: Path,
    output_csv: Path,
    profiling_precision: str,
) -> pd.DataFrame:
    rows = materialize_pp_prefill_consumer_active_rows(raw_trace_jsonl=raw_trace_jsonl)
    aggregated = aggregate_pp_prefill_consumer_active_rows(
        rows,
        profiling_precision=profiling_precision,
    )
    validated = validate_pp_receiver_head_dataframe(aggregated)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_csv, index=False)
    return validated


def materialize_pp_producer_send_path_csv(
    *,
    raw_trace_jsonl: Path,
    frontier_op_trace_jsonl: Path,
    output_csv: Path,
    profiling_precision: str,
) -> pd.DataFrame:
    rows = materialize_pp_producer_send_path_rows(
        raw_trace_jsonl=raw_trace_jsonl,
        frontier_op_trace_jsonl=frontier_op_trace_jsonl,
    )
    aggregated = aggregate_pp_producer_send_path_rows(
        rows,
        profiling_precision=profiling_precision,
    )
    validated = validate_pp_producer_send_path_dataframe(aggregated)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_csv, index=False)
    return validated
