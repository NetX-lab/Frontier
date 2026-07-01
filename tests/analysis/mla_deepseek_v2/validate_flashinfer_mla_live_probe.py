#!/usr/bin/env python3
"""Validate DeepSeek-V2 MLA FlashInfer live-probe CUDA op rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY  # noqa: E402
from frontier.attention.profiling_mapping import get_profiling_metric_names  # noqa: E402


REQUIRED_SCOPES = get_profiling_metric_names(LATENT_MLA_ATTENTION_FAMILY)

SIDECAR_REQUIRED_COLUMNS = (
    "scope",
    "vllm_cuda_time_ms",
    "frontier_profile_median_ms",
    "absolute_error_ms",
    "relative_error_pct",
    "vllm_sample_count",
)

SIDECAR_ROW_AWARE_DYNAMIC_COLUMNS = (
    "profile_row_index",
    "batch_size",
    "batch_num_tokens",
    "batch_num_prefill_tokens",
    "batch_num_decode_tokens",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_actual_tokens",
)

SIDECAR_ROW_AWARE_SIGNATURE_COLUMNS = tuple(
    column
    for column in SIDECAR_ROW_AWARE_DYNAMIC_COLUMNS
    if column != "profile_row_index"
)


@dataclass(frozen=True)
class MlaImportSidecarValidationSummary:
    scope_count: int
    sample_count_sum: int
    max_absolute_error_ms: float
    max_relative_error_pct: float
    decode_vllm_ms: float
    decode_frontier_ms: float


def _load_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError(f"CUDA op log not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"CUDA op log is empty: {path}")
    return rows


def _rows_by_required_scope(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_scope: dict[str, list[dict[str, object]]] = {scope: [] for scope in REQUIRED_SCOPES}
    for row in rows:
        scope = row.get("op_name")
        if scope in by_scope:
            by_scope[str(scope)].append(row)
    return by_scope


def _scope_cuda_median_ms(scope: str, rows: list[dict[str, object]]) -> float:
    timings = []
    for row in rows:
        if "cuda_time_ms" not in row:
            raise ValueError(f"Missing cuda_time_ms for {scope}: {row}")
        timing = float(row["cuda_time_ms"])
        if timing < 0.0:
            raise ValueError(f"Negative cuda_time_ms for {scope}: {timing}")
        timings.append(timing)
    if not timings:
        raise ValueError(f"Missing required MLA scopes: ['{scope}']")
    return statistics.median(timings)


def _row_matches_sidecar_signature(
    row: dict[str, object],
    record: dict[str, object],
) -> bool:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return False
    row_values = {
        "batch_size": row.get("batch_size"),
        "batch_num_tokens": row.get("batch_num_tokens"),
        "batch_num_prefill_tokens": row.get("batch_num_prefill_tokens"),
        "batch_num_decode_tokens": row.get("batch_num_decode_tokens"),
        "max_seqlen_q": meta.get("max_seqlen_q"),
        "max_seqlen_k": meta.get("max_seqlen_k"),
        "num_actual_tokens": meta.get("num_actual_tokens"),
    }
    return all(
        int(row_values[column]) == int(record[column])
        for column in row_values
    )


def _row_dynamic_signature(scope: str, row: dict[str, object]) -> tuple[int, ...]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Missing runtime meta for {scope}: {row}")
    row_values = {
        "batch_size": row.get("batch_size"),
        "batch_num_tokens": row.get("batch_num_tokens"),
        "batch_num_prefill_tokens": row.get("batch_num_prefill_tokens"),
        "batch_num_decode_tokens": row.get("batch_num_decode_tokens"),
        "max_seqlen_q": meta.get("max_seqlen_q"),
        "max_seqlen_k": meta.get("max_seqlen_k"),
        "num_actual_tokens": meta.get("num_actual_tokens"),
    }
    missing = [
        column
        for column in SIDECAR_ROW_AWARE_SIGNATURE_COLUMNS
        if row_values[column] is None
    ]
    if missing:
        raise ValueError(
            f"Missing row-aware vLLM dynamic fields for {scope}: {missing}"
        )
    return tuple(
        int(row_values[column]) for column in SIDECAR_ROW_AWARE_SIGNATURE_COLUMNS
    )


def _source_requires_row_aware_sidecar(
    by_scope: dict[str, list[dict[str, object]]],
) -> bool:
    for scope, scope_rows in by_scope.items():
        signatures = {
            _row_dynamic_signature(scope, row)
            for row in scope_rows
        }
        if len(signatures) > 1:
            return True
    return False


def _validate_sidecar_error_record(
    *,
    record: dict[str, object],
    scope: str,
    expected_vllm_median_ms: float,
    expected_sample_count: int,
    max_absolute_error_ms: float,
    max_relative_error_pct: float,
    tolerance: float,
) -> tuple[float, float, float, int]:
    sidecar_vllm_median_ms = _finite_sidecar_float(
        record, "vllm_cuda_time_ms", scope
    )
    if abs(sidecar_vllm_median_ms - expected_vllm_median_ms) > tolerance:
        raise ValueError(
            f"vLLM median mismatch for {scope}: sidecar={sidecar_vllm_median_ms}, "
            f"cuda_op_log={expected_vllm_median_ms}."
        )

    frontier_median_ms = _finite_sidecar_float(
        record, "frontier_profile_median_ms", scope
    )
    absolute_error_ms = _finite_sidecar_float(record, "absolute_error_ms", scope)
    relative_error_pct = _finite_sidecar_float(record, "relative_error_pct", scope)
    recomputed_abs_error = abs(frontier_median_ms - sidecar_vllm_median_ms)
    if abs(absolute_error_ms - recomputed_abs_error) > tolerance:
        raise ValueError(
            f"absolute_error_ms mismatch for {scope}: sidecar={absolute_error_ms}, "
            f"recomputed={recomputed_abs_error}."
        )
    if sidecar_vllm_median_ms == 0.0:
        recomputed_relative_error_pct = (
            0.0 if recomputed_abs_error == 0.0 else float("inf")
        )
    else:
        recomputed_relative_error_pct = (
            recomputed_abs_error / sidecar_vllm_median_ms * 100.0
        )
    if abs(relative_error_pct - recomputed_relative_error_pct) > tolerance:
        raise ValueError(
            f"relative_error_pct mismatch for {scope}: sidecar={relative_error_pct}, "
            f"recomputed={recomputed_relative_error_pct}."
        )
    if (
        absolute_error_ms > max_absolute_error_ms + tolerance
        or relative_error_pct > max_relative_error_pct + tolerance
    ):
        raise ValueError(
            f"Frontier MLA import sidecar error for {scope} exceeds allowed threshold: "
            f"absolute_error_ms={absolute_error_ms} "
            f"(limit {max_absolute_error_ms}), relative_error_pct={relative_error_pct} "
            f"(limit {max_relative_error_pct})."
        )

    sample_count = int(record["vllm_sample_count"])
    if sample_count != expected_sample_count:
        raise ValueError(
            f"vLLM sample count mismatch for {scope}: sidecar={sample_count}, "
            f"cuda_op_log={expected_sample_count}."
        )
    return sidecar_vllm_median_ms, frontier_median_ms, absolute_error_ms, sample_count


def _finite_sidecar_float(record: dict[str, object], column: str, scope: str) -> float:
    value = float(record[column])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column} for {scope}: {record[column]}")
    return value


def validate_frontier_import_sidecar_against_vllm_log(
    *,
    cuda_op_log: Path,
    sidecar_csv: Path,
    max_absolute_error_ms: float,
    max_relative_error_pct: float,
) -> MlaImportSidecarValidationSummary:
    """Validate a Frontier MLA import sidecar against its source vLLM rows."""

    if max_absolute_error_ms < 0.0:
        raise ValueError(f"max_absolute_error_ms must be nonnegative: {max_absolute_error_ms}")
    if max_relative_error_pct < 0.0:
        raise ValueError(f"max_relative_error_pct must be nonnegative: {max_relative_error_pct}")
    if not sidecar_csv.exists():
        raise ValueError(f"Frontier MLA import sidecar not found: {sidecar_csv}")

    rows = _load_rows(cuda_op_log)
    by_scope = _rows_by_required_scope(rows)
    missing = [scope for scope, scope_rows in by_scope.items() if not scope_rows]
    if missing:
        raise ValueError(f"Missing required MLA scopes: {missing}")

    sidecar = pd.read_csv(sidecar_csv)
    missing_columns = [
        column for column in SIDECAR_REQUIRED_COLUMNS if column not in sidecar.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing Frontier sidecar columns: {missing_columns}")
    missing_row_aware_columns = [
        column
        for column in SIDECAR_ROW_AWARE_DYNAMIC_COLUMNS
        if column not in sidecar.columns
    ]
    present_row_aware_columns = [
        column
        for column in SIDECAR_ROW_AWARE_DYNAMIC_COLUMNS
        if column in sidecar.columns
    ]
    source_requires_row_aware = _source_requires_row_aware_sidecar(by_scope)
    if present_row_aware_columns and missing_row_aware_columns:
        raise ValueError(
            "Incomplete row-aware dynamic sidecar columns: "
            f"missing {missing_row_aware_columns}."
        )
    if source_requires_row_aware and missing_row_aware_columns:
        raise ValueError(
            "Missing row-aware dynamic sidecar columns for multi-signature "
            f"vLLM MLA log: {missing_row_aware_columns}."
        )
    row_aware_sidecar = not missing_row_aware_columns
    if not row_aware_sidecar and len(sidecar) != len(REQUIRED_SCOPES):
        raise ValueError(
            "Frontier MLA import sidecar must contain exactly "
            f"{len(REQUIRED_SCOPES)} rows; got {len(sidecar)}."
        )

    sidecar_scopes = tuple(str(scope) for scope in sidecar["scope"])
    if not row_aware_sidecar and sidecar_scopes != REQUIRED_SCOPES:
        raise ValueError(
            "Frontier MLA import sidecar scope order mismatch: "
            f"expected {REQUIRED_SCOPES}, got {sidecar_scopes}."
        )

    max_observed_abs_error = 0.0
    max_observed_relative_error = 0.0
    sample_count_sum = 0
    decode_vllm_ms = 0.0
    decode_frontier_ms = 0.0
    tolerance = 1e-12
    consumed_signatures: set[tuple[object, ...]] = set()
    for record in sidecar.to_dict("records"):
        scope = str(record["scope"])
        if row_aware_sidecar:
            matching_rows = [
                row
                for row in by_scope[scope]
                if _row_matches_sidecar_signature(row, record)
            ]
            if not matching_rows:
                raise ValueError(
                    "No vLLM MLA row matches row-aware sidecar signature for "
                    f"{scope}: {record}"
                )
            sidecar_signature = (
                scope,
                *(
                    int(record[column])
                    for column in SIDECAR_ROW_AWARE_DYNAMIC_COLUMNS
                    if column != "profile_row_index"
                ),
            )
            if sidecar_signature in consumed_signatures:
                raise ValueError(
                    "Duplicate row-aware Frontier MLA sidecar signature: "
                    f"{sidecar_signature}"
                )
            consumed_signatures.add(sidecar_signature)
            expected_vllm_median_ms = _scope_cuda_median_ms(scope, matching_rows)
            expected_sample_count = len(matching_rows)
        else:
            expected_vllm_median_ms = _scope_cuda_median_ms(scope, by_scope[scope])
            expected_sample_count = len(by_scope[scope])
        (
            sidecar_vllm_median_ms,
            frontier_median_ms,
            absolute_error_ms,
            sample_count,
        ) = _validate_sidecar_error_record(
            record=record,
            scope=scope,
            expected_vllm_median_ms=expected_vllm_median_ms,
            expected_sample_count=expected_sample_count,
            max_absolute_error_ms=max_absolute_error_ms,
            max_relative_error_pct=max_relative_error_pct,
            tolerance=tolerance,
        )

        sample_count_sum += sample_count
        max_observed_abs_error = max(max_observed_abs_error, absolute_error_ms)
        max_observed_relative_error = max(
            max_observed_relative_error,
            _finite_sidecar_float(record, "relative_error_pct", scope),
        )
        if scope == "attn_mla_decode":
            decode_vllm_ms = max(decode_vllm_ms, sidecar_vllm_median_ms)
            decode_frontier_ms = max(decode_frontier_ms, frontier_median_ms)

    if row_aware_sidecar:
        observed_required_rows = sum(len(scope_rows) for scope_rows in by_scope.values())
        if sample_count_sum != observed_required_rows:
            raise ValueError(
                "Row-aware Frontier MLA sidecar does not conserve vLLM samples: "
                f"sidecar_sample_count_sum={sample_count_sum}, "
                f"cuda_op_log_rows={observed_required_rows}."
            )

    return MlaImportSidecarValidationSummary(
        scope_count=len(sidecar),
        sample_count_sum=sample_count_sum,
        max_absolute_error_ms=max_observed_abs_error,
        max_relative_error_pct=max_observed_relative_error,
        decode_vllm_ms=decode_vllm_ms,
        decode_frontier_ms=decode_frontier_ms,
    )


def _validate_meta(
    meta: dict[str, object],
    *,
    scope: str,
    expected_backend: str,
    expected_runtime_head_size: int,
    expected_runtime_kv_heads: int,
    expected_block_size: int,
) -> None:
    required = [
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
    ]
    missing = [field for field in required if field not in meta]
    if missing:
        raise ValueError(f"Missing runtime meta fields for {scope}: {missing}")
    if meta["attention_backend"] != expected_backend:
        raise ValueError(f"Unexpected attention_backend for {scope}: {meta['attention_backend']}")
    if meta["use_mla"] is not True:
        raise ValueError(f"use_mla must be true for {scope}")
    if int(meta["runtime_num_kv_heads"]) != expected_runtime_kv_heads:
        raise ValueError(
            f"Unexpected runtime_num_kv_heads for {scope}: {meta['runtime_num_kv_heads']}"
        )
    if int(meta["runtime_head_size"]) != expected_runtime_head_size:
        raise ValueError(
            f"Unexpected runtime_head_size for {scope}: {meta['runtime_head_size']}"
        )
    if int(meta["block_size"]) != expected_block_size:
        raise ValueError(f"Unexpected block_size for {scope}: {meta['block_size']}")
    if meta["attn_module_sliding_window"] is not None:
        raise ValueError(f"Unexpected sliding window for {scope}: {meta['attn_module_sliding_window']}")
    if meta["alibi_slopes"] is not None:
        raise ValueError(f"Unexpected alibi_slopes for {scope}: {meta['alibi_slopes']}")
    if meta["logits_soft_cap"] is not None:
        raise ValueError(f"Unexpected logits_soft_cap for {scope}: {meta['logits_soft_cap']}")
    if str(meta["attn_type"]).lower() != "decoder":
        raise ValueError(f"Unexpected attn_type for {scope}: {meta['attn_type']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-op-log", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--expected-backend", required=True)
    parser.add_argument("--expected-runtime-head-size", type=int, required=True)
    parser.add_argument("--expected-runtime-kv-heads", type=int, required=True)
    parser.add_argument("--expected-block-size", type=int, required=True)
    parser.add_argument("--frontier-import-sidecar", type=Path)
    parser.add_argument("--max-absolute-error-ms", type=float, default=0.0)
    parser.add_argument("--max-relative-error-pct", type=float, default=0.0)
    args = parser.parse_args()

    rows = _load_rows(args.cuda_op_log)
    run_log = args.run_log.read_text(encoding="utf-8") if args.run_log.exists() else ""
    normalized_run_log = run_log.lower().replace("_", " ")
    normalized_expected_backend = args.expected_backend.lower().replace("_", " ")
    if normalized_expected_backend not in normalized_run_log:
        raise ValueError(f"Run log does not mention expected backend {args.expected_backend}")

    by_scope: dict[str, list[dict[str, object]]] = {scope: [] for scope in REQUIRED_SCOPES}
    for row in rows:
        scope = row.get("op_name")
        if scope in by_scope:
            by_scope[str(scope)].append(row)

    missing = [scope for scope, scope_rows in by_scope.items() if not scope_rows]
    if missing:
        raise ValueError(f"Missing required MLA scopes: {missing}")

    for scope, scope_rows in by_scope.items():
        for row in scope_rows:
            meta = row.get("meta")
            if not isinstance(meta, dict):
                raise ValueError(f"Missing runtime meta for {scope}")
            _validate_meta(
                meta,
                scope=scope,
                expected_backend=args.expected_backend,
                expected_runtime_head_size=args.expected_runtime_head_size,
                expected_runtime_kv_heads=args.expected_runtime_kv_heads,
                expected_block_size=args.expected_block_size,
            )
        total_cuda_ms = sum(float(row.get("cuda_time_ms", 0.0)) for row in scope_rows)
        print(f"{scope}: rows={len(scope_rows)} total_cuda_time_ms={total_cuda_ms:.6f}")

    if args.frontier_import_sidecar is not None:
        summary = validate_frontier_import_sidecar_against_vllm_log(
            cuda_op_log=args.cuda_op_log,
            sidecar_csv=args.frontier_import_sidecar,
            max_absolute_error_ms=args.max_absolute_error_ms,
            max_relative_error_pct=args.max_relative_error_pct,
        )
        print(
            "Frontier MLA import sidecar validation passed: "
            f"scope_count={summary.scope_count} "
            f"sample_count_sum={summary.sample_count_sum} "
            f"decode_vllm_ms={summary.decode_vllm_ms:.6f} "
            f"decode_frontier_ms={summary.decode_frontier_ms:.6f} "
            f"max_absolute_error_ms={summary.max_absolute_error_ms:.6f} "
            f"max_relative_error_pct={summary.max_relative_error_pct:.6f}"
        )

    print("FlashInfer MLA live probe validation passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        import sys

        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
