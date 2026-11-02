"""Import measured vLLM MLA scope rows into Frontier attention profiling schema."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import (
    get_profiling_metric_names,
    validate_attention_profiling_dataframe,
)
from frontier.types import MeasurementType

REQUIRED_RUNTIME_META_FIELDS: tuple[str, ...] = (
    "attention_backend",
    "use_mla",
    "runtime_num_kv_heads",
    "runtime_head_size",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "qk_head_dim",
    "v_head_dim",
    "block_size",
    "kv_cache_dtype",
    "calculate_kv_scales",
    "attn_module_sliding_window",
    "alibi_slopes",
    "logits_soft_cap",
    "attn_type",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_actual_tokens",
)

REQUIRED_DYNAMIC_ROW_FIELDS: tuple[str, ...] = (
    "batch_size",
    "batch_num_tokens",
    "batch_num_prefill_tokens",
    "batch_num_decode_tokens",
    "batch_request_num_tokens",
)

STRUCTURAL_RUNTIME_META_FIELDS: tuple[str, ...] = (
    "attention_backend",
    "use_mla",
    "runtime_num_kv_heads",
    "runtime_head_size",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "qk_head_dim",
    "v_head_dim",
    "block_size",
    "kv_cache_dtype",
    "calculate_kv_scales",
    "attn_module_sliding_window",
    "alibi_slopes",
    "logits_soft_cap",
    "attn_type",
)

DYNAMIC_RUNTIME_META_FIELDS: tuple[str, ...] = (
    "max_seqlen_q",
    "max_seqlen_k",
    "num_actual_tokens",
)

PROFILE_DYNAMIC_SCALAR_FIELDS: tuple[str, ...] = (
    "batch_size",
    "batch_num_tokens",
    "batch_num_prefill_tokens",
    "batch_num_decode_tokens",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_actual_tokens",
)


@dataclass(frozen=True)
class _MlaScopeStats:
    scope: str
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    std_ms: float


def _required_mla_scopes() -> tuple[str, ...]:
    return get_profiling_metric_names(LATENT_MLA_ATTENTION_FAMILY)


def load_vllm_mla_rows(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from a vLLM Frontier CUDA-event op log."""

    if not path.exists():
        raise ValueError(f"vLLM MLA CUDA op log not found: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSON object at {path}:{line_number}")
        rows.append(payload)

    if not rows:
        raise ValueError(f"vLLM MLA CUDA op log is empty: {path}")
    return rows


def _rows_by_scope(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_scope: dict[str, list[dict[str, Any]]] = {}
    required_scopes = _required_mla_scopes()
    for row in rows:
        op_name = row.get("op_name")
        if isinstance(op_name, str) and op_name in required_scopes:
            by_scope.setdefault(op_name, []).append(row)
    return by_scope


def _require_meta(row: dict[str, Any], scope: str) -> dict[str, Any]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Missing runtime meta for {scope}: {row}")

    missing = [field for field in REQUIRED_RUNTIME_META_FIELDS if field not in meta]
    if missing:
        raise ValueError(f"Missing runtime meta fields for {scope}: {missing}")
    return meta


def _validate_flashinfer_mla_meta(meta: dict[str, Any], scope: str) -> None:
    contract = LATENT_MLA_ATTENTION_FAMILY.runtime_meta_contract
    if contract is None:
        raise ValueError("Latent MLA attention family must declare runtime meta contract")
    if meta["attention_backend"] != "FLASHINFER_MLA":
        raise ValueError(
            f"Unexpected attention_backend for {scope}: {meta['attention_backend']}"
        )
    if meta["use_mla"] is not True:
        raise ValueError(f"use_mla must be true for {scope}")
    if int(meta["runtime_num_kv_heads"]) != contract.expected_runtime_num_kv_heads:
        raise ValueError(
            f"Unexpected runtime_num_kv_heads for {scope}: "
            f"{meta['runtime_num_kv_heads']}"
        )
    if (
        contract.expected_n_q_head is not None
        and "n_q_head" in meta
        and int(meta["n_q_head"]) != contract.expected_n_q_head
    ):
        raise ValueError(f"Unexpected n_q_head for {scope}: {meta['n_q_head']}")
    if int(meta["qk_head_dim"]) != (
        int(meta["qk_nope_head_dim"]) + int(meta["qk_rope_head_dim"])
    ):
        raise ValueError(f"Inconsistent qk_head_dim for {scope}: {meta['qk_head_dim']}")
    if int(meta["runtime_head_size"]) != (
        int(meta["kv_lora_rank"]) + int(meta["qk_rope_head_dim"])
    ):
        raise ValueError(
            f"Inconsistent runtime_head_size for {scope}: {meta['runtime_head_size']}"
        )
    if int(meta["block_size"]) not in set(contract.supported_block_sizes):
        raise ValueError(
            f"Unsupported FlashInfer MLA block_size for {scope}: {meta['block_size']}"
        )
    if meta["attn_module_sliding_window"] is not None:
        raise ValueError(f"FlashInfer MLA sliding window must be disabled for {scope}")
    if meta["alibi_slopes"] is not None:
        raise ValueError(f"FlashInfer MLA ALiBi must be disabled for {scope}")
    if meta["logits_soft_cap"] is not None:
        raise ValueError(f"FlashInfer MLA logits soft cap must be disabled for {scope}")
    if str(meta["attn_type"]).lower() != "decoder":
        raise ValueError(f"FlashInfer MLA only supports decoder attention for {scope}")


def _structural_meta_signature(meta: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple((field, meta[field]) for field in STRUCTURAL_RUNTIME_META_FIELDS)


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _dynamic_profile_shape_signature(
    row: dict[str, Any], meta: dict[str, Any], scope: str
) -> tuple[tuple[str, Any], ...]:
    missing = [field for field in REQUIRED_DYNAMIC_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Missing dynamic profile shape fields for {scope}: {missing}")

    row_signature = tuple(
        (field, _normalize_signature_value(row[field]))
        for field in REQUIRED_DYNAMIC_ROW_FIELDS
    )
    meta_signature = tuple((field, meta[field]) for field in DYNAMIC_RUNTIME_META_FIELDS)
    return row_signature + meta_signature


def _profile_dynamic_scalar_signature(
    row: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (field, int(row[field]))
        for field in REQUIRED_DYNAMIC_ROW_FIELDS
        if field != "batch_request_num_tokens"
    ) + tuple((field, int(meta[field])) for field in DYNAMIC_RUNTIME_META_FIELDS)


def _profile_row_dynamic_scalar_signature(
    row: pd.Series,
) -> tuple[tuple[str, Any], ...]:
    return tuple((field, int(row[field])) for field in PROFILE_DYNAMIC_SCALAR_FIELDS)


def _batch_level_dynamic_signature(
    row: dict[str, Any],
    meta: dict[str, Any],
    scope: str,
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (field, _normalize_signature_value(row[field]))
        for field in REQUIRED_DYNAMIC_ROW_FIELDS
    ) + (("max_seqlen_k", meta["max_seqlen_k"]),)


def _validate_dynamic_profile_row(
    row: dict[str, Any],
    meta: dict[str, Any],
    scope: str,
) -> None:
    batch_size = int(row["batch_size"])
    batch_num_tokens = int(row["batch_num_tokens"])
    batch_num_prefill_tokens = int(row["batch_num_prefill_tokens"])
    batch_num_decode_tokens = int(row["batch_num_decode_tokens"])
    request_token_counts = row["batch_request_num_tokens"]
    if not isinstance(request_token_counts, (list, tuple)):
        raise ValueError(
            "Inconsistent dynamic profile shape for "
            f"{scope}: batch_request_num_tokens must be a sequence."
        )
    normalized_request_token_counts = [int(value) for value in request_token_counts]
    if len(normalized_request_token_counts) != batch_size:
        raise ValueError(
            "Inconsistent dynamic profile shape for "
            f"{scope}: batch_size={batch_size}, "
            "batch_request_num_tokens length="
            f"{len(normalized_request_token_counts)}."
        )
    if sum(normalized_request_token_counts) != batch_num_tokens:
        raise ValueError(
            "Inconsistent dynamic profile shape for "
            f"{scope}: sum(batch_request_num_tokens)="
            f"{sum(normalized_request_token_counts)}, "
            f"batch_num_tokens={batch_num_tokens}."
        )
    if batch_num_prefill_tokens + batch_num_decode_tokens != batch_num_tokens:
        raise ValueError(
            "MLA import dynamic token metadata is inconsistent: "
            "num_actual_tokens is op-scoped, but "
            "batch_num_prefill_tokens + batch_num_decode_tokens must match "
            f"batch_num_tokens={batch_num_tokens}; "
            f"batch_num_prefill_tokens={batch_num_prefill_tokens}, "
            f"batch_num_decode_tokens={batch_num_decode_tokens}."
        )

    num_actual_tokens = int(meta["num_actual_tokens"])
    if num_actual_tokens <= 0 or num_actual_tokens > batch_num_tokens:
        raise ValueError(
            "MLA import requires op-scoped num_actual_tokens to be in "
            f"(0, batch_num_tokens]; num_actual_tokens={num_actual_tokens}, "
            f"batch_num_tokens={batch_num_tokens}."
        )


def _group_validated_scope_rows(
    by_scope: dict[str, list[dict[str, Any]]],
) -> dict[tuple[tuple[str, Any], ...], dict[str, list[dict[str, Any]]]]:
    grouped: dict[tuple[tuple[str, Any], ...], dict[str, list[dict[str, Any]]]] = {}
    for scope in _required_mla_scopes():
        for row in by_scope[scope]:
            meta = _require_meta(row, scope)
            signature = _dynamic_profile_shape_signature(row, meta, scope)
            grouped.setdefault(signature, {}).setdefault(scope, []).append(row)
    return grouped


def _scope_stats(scope: str, rows: list[dict[str, Any]]) -> _MlaScopeStats:
    times: list[float] = []
    for row in rows:
        if "cuda_time_ms" not in row:
            raise ValueError(f"Missing cuda_time_ms for {scope}: {row}")
        times.append(float(row["cuda_time_ms"]))
    if any(time_ms < 0.0 for time_ms in times):
        raise ValueError(f"Negative CUDA timing found for {scope}: {times}")

    return _MlaScopeStats(
        scope=scope,
        count=len(times),
        min_ms=min(times),
        max_ms=max(times),
        mean_ms=statistics.fmean(times),
        median_ms=statistics.median(times),
        std_ms=statistics.pstdev(times),
    )


def _validated_scope_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_scope = _rows_by_scope(rows)
    required_scopes = _required_mla_scopes()
    missing = [scope for scope in required_scopes if not by_scope.get(scope)]
    if missing:
        raise ValueError(f"Missing required MLA attention scopes: {missing}")

    reference_signature: tuple[tuple[str, Any], ...] | None = None
    batch_level_signatures: dict[Any, tuple[tuple[str, Any], ...]] = {}
    for scope in required_scopes:
        for row in by_scope[scope]:
            meta = _require_meta(row, scope)
            _validate_flashinfer_mla_meta(meta, scope)
            _dynamic_profile_shape_signature(row, meta, scope)
            _validate_dynamic_profile_row(row, meta, scope)
            signature = _structural_meta_signature(meta)
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise ValueError(
                    f"Inconsistent runtime meta for {scope}: "
                        f"{dict(signature)} vs {dict(reference_signature)}"
                    )
            batch_id = row.get("batch_id")
            batch_level_signature = _batch_level_dynamic_signature(row, meta, scope)
            if batch_id not in batch_level_signatures:
                batch_level_signatures[batch_id] = batch_level_signature
            elif batch_level_signature != batch_level_signatures[batch_id]:
                raise ValueError(
                    "Inconsistent dynamic profile shape for "
                    f"batch_id={batch_id}, scope={scope}: "
                    f"{dict(batch_level_signature)} vs "
                    f"{dict(batch_level_signatures[batch_id])}"
                )

    return by_scope


def _validate_mla_dynamic_token_count_consistency(
    row: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    batch_num_tokens = int(row["batch_num_tokens"])
    num_actual_tokens = int(meta["num_actual_tokens"])
    if num_actual_tokens <= 0 or num_actual_tokens > batch_num_tokens:
        raise ValueError(
            "MLA import requires op-scoped num_actual_tokens to be in "
            "(0, batch_num_tokens] for exact-row runtime lookup: "
            f"num_actual_tokens={num_actual_tokens}, "
            f"batch_num_tokens={batch_num_tokens}."
        )


def _build_profile_row_base(
    *,
    representative_row: dict[str, Any],
    representative_meta: dict[str, Any],
    model_name: str,
    model_arch: str,
    precision: str,
    quant_signature: str,
    measurement_type: MeasurementType,
    num_tensor_parallel_workers: int,
    max_model_len: int,
) -> dict[str, Any]:
    contract = LATENT_MLA_ATTENTION_FAMILY.runtime_meta_contract
    if contract is None or contract.expected_n_q_head is None:
        raise ValueError(
            "Latent MLA attention family must declare expected_n_q_head"
        )
    max_seqlen_k = int(representative_meta["max_seqlen_k"])
    return {
        "model_name": model_name,
        "model_arch": model_arch,
        "precision": precision,
        "quant_signature": quant_signature,
        "measurement_type": measurement_type.value,
        "attention_backend": str(representative_meta["attention_backend"]),
        "n_q_head": contract.expected_n_q_head,
        "n_kv_head": int(representative_meta["runtime_num_kv_heads"]),
        "head_size": int(representative_meta["runtime_head_size"]),
        "qk_nope_head_dim": int(representative_meta["qk_nope_head_dim"]),
        "qk_rope_head_dim": int(representative_meta["qk_rope_head_dim"]),
        "qk_head_dim": int(representative_meta["qk_head_dim"]),
        "kv_lora_rank": int(representative_meta["kv_lora_rank"]),
        "v_head_dim": int(representative_meta["v_head_dim"]),
        "block_size": int(representative_meta["block_size"]),
        "num_tensor_parallel_workers": int(num_tensor_parallel_workers),
        "max_model_len": int(max_model_len),
        "batch_size": int(representative_row["batch_size"]),
        "batch_num_tokens": int(representative_row["batch_num_tokens"]),
        "batch_num_prefill_tokens": int(
            representative_row["batch_num_prefill_tokens"]
        ),
        "batch_num_decode_tokens": int(
            representative_row["batch_num_decode_tokens"]
        ),
        "max_seqlen_q": int(representative_meta["max_seqlen_q"]),
        "max_seqlen_k": max_seqlen_k,
        "num_actual_tokens": int(representative_meta["num_actual_tokens"]),
        "is_prefill": int(representative_row["batch_num_prefill_tokens"]) > 0,
        "max_seq_len": max_seqlen_k,
        "is_mla_profile_import": True,
    }


def _write_scope_stats(
    profile_row: dict[str, Any],
    scope: str,
    rows: list[dict[str, Any]] | None,
) -> None:
    prefix = f"time_stats.{scope}"
    if not rows:
        for suffix in ("min", "max", "mean", "median", "std", "count"):
            profile_row[f"{prefix}.{suffix}"] = math.nan
        return

    stats = _scope_stats(scope, rows)
    profile_row[f"{prefix}.min"] = stats.min_ms
    profile_row[f"{prefix}.max"] = stats.max_ms
    profile_row[f"{prefix}.mean"] = stats.mean_ms
    profile_row[f"{prefix}.median"] = stats.median_ms
    profile_row[f"{prefix}.std"] = stats.std_ms
    profile_row[f"{prefix}.count"] = stats.count


def build_frontier_mla_profile_dataframe(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    model_arch: str,
    precision: str,
    quant_signature: str,
    measurement_type: str | MeasurementType,
    num_tensor_parallel_workers: int,
    max_model_len: int,
) -> pd.DataFrame:
    """Build a Frontier-compatible attention profiling DataFrame from vLLM MLA rows."""

    if not rows:
        raise ValueError("Cannot import empty vLLM MLA row set")

    by_scope = _validated_scope_rows(rows)
    grouped_rows = _group_validated_scope_rows(by_scope)
    normalized_measurement_type = (
        measurement_type
        if isinstance(measurement_type, MeasurementType)
        else MeasurementType.from_string(measurement_type)
    )

    profile_rows: list[dict[str, Any]] = []
    for scope_rows_by_name in grouped_rows.values():
        representative_scope = next(iter(scope_rows_by_name))
        representative_row = scope_rows_by_name[representative_scope][0]
        representative_meta = _require_meta(representative_row, representative_scope)
        _validate_mla_dynamic_token_count_consistency(
            representative_row,
            representative_meta,
        )
        profile_row = _build_profile_row_base(
            representative_row=representative_row,
            representative_meta=representative_meta,
            model_name=model_name,
            model_arch=model_arch,
            precision=precision,
            quant_signature=quant_signature,
            measurement_type=normalized_measurement_type,
            num_tensor_parallel_workers=num_tensor_parallel_workers,
            max_model_len=max_model_len,
        )
        for scope in _required_mla_scopes():
            _write_scope_stats(profile_row, scope, scope_rows_by_name.get(scope))
        profile_rows.append(profile_row)

    df = pd.DataFrame(profile_rows)
    df.attrs["dynamic_signature_fields"] = (
        *REQUIRED_DYNAMIC_ROW_FIELDS,
        *DYNAMIC_RUNTIME_META_FIELDS,
    )
    df.attrs["batch_request_num_tokens_by_signature"] = tuple(
        sorted(
            {dict(signature)["batch_request_num_tokens"] for signature in grouped_rows},
            key=lambda token_vector: (len(token_vector), token_vector),
        )
    )
    validate_attention_profiling_dataframe(
        df,
        LATENT_MLA_ATTENTION_FAMILY,
        measurement_type=normalized_measurement_type,
    )
    return df


def build_mla_profile_groundtruth_comparison(
    rows: list[dict[str, Any]],
    frontier_profile_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare imported Frontier MLA medians against vLLM CUDA-op rows."""

    required_scopes = _required_mla_scopes()
    by_scope = _validated_scope_rows(rows)
    grouped_rows = _group_validated_scope_rows(by_scope)
    grouped_rows_by_scalar_signature: dict[
        tuple[tuple[str, Any], ...],
        dict[str, list[dict[str, Any]]],
    ] = {}
    for scope_rows_by_name in grouped_rows.values():
        representative_scope = next(iter(scope_rows_by_name))
        representative_row = scope_rows_by_name[representative_scope][0]
        representative_meta = _require_meta(representative_row, representative_scope)
        scalar_signature = _profile_dynamic_scalar_signature(
            representative_row,
            representative_meta,
        )
        if scalar_signature in grouped_rows_by_scalar_signature:
            raise ValueError(
                "Ambiguous MLA profile comparison signature without "
                "batch_request_num_tokens: "
                f"{dict(scalar_signature)}"
            )
        grouped_rows_by_scalar_signature[scalar_signature] = scope_rows_by_name

    sparse_profile = len(frontier_profile_df) != 1 or any(
        frontier_profile_df[f"time_stats.{scope}.median"].isna().any()
        for scope in required_scopes
        if f"time_stats.{scope}.median" in frontier_profile_df.columns
    )
    comparison_rows: list[dict[str, Any]] = []

    for profile_row_index, (_, frontier_row) in enumerate(
        frontier_profile_df.iterrows()
    ):
        profile_signature = _profile_row_dynamic_scalar_signature(frontier_row)
        if profile_signature not in grouped_rows_by_scalar_signature:
            raise ValueError(
                "No vLLM MLA rows match Frontier profiling row "
                f"{profile_row_index}: {dict(profile_signature)}"
            )
        scope_rows_by_name = grouped_rows_by_scalar_signature[profile_signature]

        for scope in required_scopes:
            frontier_median_raw = frontier_row[f"time_stats.{scope}.median"]
            if pd.isna(frontier_median_raw):
                continue
            if scope not in scope_rows_by_name:
                raise ValueError(
                    "Frontier MLA profile row has timing for an unobserved scope: "
                    f"profile_row_index={profile_row_index}, scope={scope}"
                )
            stats = _scope_stats(scope, scope_rows_by_name[scope])
            frontier_median_ms = float(frontier_median_raw)
            absolute_error_ms = abs(frontier_median_ms - stats.median_ms)
            if stats.median_ms == 0.0 and absolute_error_ms != 0.0:
                raise ValueError(
                    f"Cannot compute relative error for {scope}: zero vLLM median "
                    f"with nonzero Frontier median {frontier_median_ms}."
                )
            relative_error_pct = (
                0.0
                if stats.median_ms == 0.0
                else absolute_error_ms / stats.median_ms * 100.0
            )
            comparison_row = {
                "scope": scope,
                "vllm_cuda_time_ms": stats.median_ms,
                "frontier_profile_median_ms": frontier_median_ms,
                "absolute_error_ms": absolute_error_ms,
                "relative_error_pct": relative_error_pct,
                "vllm_sample_count": stats.count,
            }
            if sparse_profile:
                comparison_row.update(
                    {
                        "profile_row_index": profile_row_index,
                        "batch_size": int(frontier_row["batch_size"]),
                        "batch_num_tokens": int(frontier_row["batch_num_tokens"]),
                        "batch_num_prefill_tokens": int(
                            frontier_row["batch_num_prefill_tokens"]
                        ),
                        "batch_num_decode_tokens": int(
                            frontier_row["batch_num_decode_tokens"]
                        ),
                        "max_seqlen_q": int(frontier_row["max_seqlen_q"]),
                        "max_seqlen_k": int(frontier_row["max_seqlen_k"]),
                        "num_actual_tokens": int(frontier_row["num_actual_tokens"]),
                        "is_prefill": bool(frontier_row["is_prefill"]),
                        "max_seq_len": int(frontier_row["max_seq_len"]),
                    }
                )
            comparison_rows.append(comparison_row)

    if not sparse_profile:
        comparison_rows.sort(
            key=lambda record: required_scopes.index(str(record["scope"]))
        )

    return pd.DataFrame(comparison_rows)


def load_vllm_mla_profile_dataframe(
    path: Path,
    *,
    model_name: str,
    model_arch: str,
    precision: str,
    quant_signature: str,
    measurement_type: str | MeasurementType,
    num_tensor_parallel_workers: int,
    max_model_len: int,
) -> pd.DataFrame:
    """Load a vLLM MLA JSONL op log and convert it to Frontier profiling schema."""

    return build_frontier_mla_profile_dataframe(
        load_vllm_mla_rows(path),
        model_name=model_name,
        model_arch=model_arch,
        precision=precision,
        quant_signature=quant_signature,
        measurement_type=measurement_type,
        num_tensor_parallel_workers=num_tensor_parallel_workers,
        max_model_len=max_model_len,
    )
