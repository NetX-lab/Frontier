"""Run and validate one model from the frozen H200 profiling manifest."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from frontier.operators.typed_contracts import (
    parse_typed_operator_contract_column,
)

from frontier.moe_gating_runtime import (
    MOE_GATING_RUNTIME_CONTEXT_COLUMN,
    MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN,
    get_moe_gating_runtime_context_metadata,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from tests.performance.profiling.validate_h200_six_model_non_dummy_e2e import (
    SUPPORTED_MODELS,
    build_model_contract,
)


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen H200 manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_ready_for_h200_collection":
        raise ValueError(
            "H200 manifest is not frozen for collection: "
            f"status={manifest.get('status')!r}"
        )
    if manifest.get("scope", {}).get("device") != "h200":
        raise ValueError("H200 manifest device must equal 'h200'.")
    if manifest.get("counting_contract", {}).get("total_logical_rows") != 10_656:
        raise ValueError("H200 manifest logical-row total must equal 10656.")
    return manifest


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_command(
    *,
    label: str,
    command: list[str],
    log_path: Path,
    timeout_seconds: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"label={label}\n")
        log_file.write(f"start={started_iso}\n")
        log_file.write(f"command={_command_text(command)}\n")
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = 124
            timed_out = True
            log_file.write(
                f"\nERROR: command exceeded timeout_seconds={timeout_seconds}.\n"
            )
    ended_at = time.time()
    record = {
        "label": label,
        "command": command,
        "command_text": _command_text(command),
        "log_path": str(log_path),
        "start": started_iso,
        "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(ended_at - started_at, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
    }
    records.append(record)
    if exit_code != 0:
        raise RuntimeError(
            f"Profiling command {label!r} failed with exit_code={exit_code}; "
            f"see {log_path}."
        )
    return record


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        if value in (0, 1):
            return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}.")


def _normalize_int_list(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected list-like value, got {value!r}.")
    return tuple(int(item) for item in parsed)


def _require_exact_values(
    frame: pd.DataFrame,
    *,
    path: Path,
    column: str,
    expected: Iterable[Any],
) -> None:
    if column not in frame.columns:
        raise ValueError(f"{path.name} is missing required column {column!r}.")
    expected_values = set(expected)
    observed_values = set(frame[column].dropna().tolist())
    if observed_values != expected_values:
        raise ValueError(
            f"{path.name} column {column!r} has values {sorted(observed_values)!r}; "
            f"expected {sorted(expected_values)!r}."
        )


def _require_constant(
    frame: pd.DataFrame,
    *,
    path: Path,
    column: str,
    expected: Any,
) -> None:
    if column not in frame.columns:
        raise ValueError(f"{path.name} is missing required column {column!r}.")
    series = frame[column]
    if expected is None:
        if not series.isna().all():
            raise ValueError(f"{path.name} column {column!r} must be empty.")
        return
    if isinstance(expected, bool):
        observed = {_normalize_bool(value) for value in series}
    elif isinstance(expected, int):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(
                f"{path.name} column {column!r} contains non-numeric values."
            )
        observed = {int(value) for value in numeric}
    else:
        observed = {str(value).strip() for value in series}
        expected = str(expected)
    if observed != {expected}:
        raise ValueError(
            f"{path.name} column {column!r} has values {observed!r}; "
            f"expected only {expected!r}."
        )


def _require_finite_positive(
    frame: pd.DataFrame,
    *,
    path: Path,
    column: str,
    mask: pd.Series | None = None,
) -> float:
    if column not in frame.columns:
        raise ValueError(f"{path.name} is missing timing column {column!r}.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if mask is not None:
        values = values[mask]
    if values.empty:
        raise ValueError(f"{path.name} timing slice {column!r} is empty.")
    valid = values.map(
        lambda value: pd.notna(value)
        and math.isfinite(float(value))
        and float(value) > 0
    )
    if not bool(valid.all()):
        bad_values = values[~valid].head(10).tolist()
        raise ValueError(
            f"{path.name} timing column {column!r} contains non-finite or "
            f"non-positive values: {bad_values!r}."
        )
    return float(values.min())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected profiling CSV is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Profiling CSV is empty: {path}")
    frame = pd.read_csv(path)
    # Validate optional typed metadata without replacing the JSON strings used
    # by the existing duplicate and coverage checks.
    parse_typed_operator_contract_column(frame)
    if frame.empty:
        raise ValueError(f"Profiling CSV has no rows: {path}")
    return frame


def _validate_common_metadata(
    frame: pd.DataFrame,
    *,
    path: Path,
    contract: Any,
    measurement_type: str,
) -> None:
    expected = {
        "measurement_type": measurement_type,
        "profiling_precision": contract.profiling_precision,
        "quant_signature": contract.quant_signature,
        "model_arch": contract.model_arch,
        "model_architecture_profile": contract.model_architecture_profile,
    }
    for column, value in expected.items():
        _require_constant(frame, path=path, column=column, expected=value)


def _standard_attention_required_keys(
    manifest: dict[str, Any],
) -> set[tuple[int, int, int, int, bool]]:
    return {
        (
            tp_size,
            int(row["prefill_chunk_size"]),
            int(row["kv_cache_size"]),
            int(row["batch_size"]),
            bool(row["is_prefill"]),
        )
        for tp_size in manifest["scope"]["tp_sizes"]
        for row in manifest["scope"]["attention"]["standard_workloads"]
    }


def _true_mixed_attention_required_keys(
    manifest: dict[str, Any],
) -> set[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    return {
        (
            tp_size,
            tuple(int(value) for value in row["prefill_seq_lens"]),
            tuple(int(value) for value in row["prefill_kv_cache_sizes"]),
            tuple(int(value) for value in row["decode_kv_cache_sizes"]),
        )
        for tp_size in manifest["scope"]["tp_sizes"]
        for row in manifest["scope"]["attention"]["true_mixed_workloads"]
    }


def _validate_standard_attention(
    frame: pd.DataFrame,
    *,
    path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    required_keys = _standard_attention_required_keys(manifest)
    observed_keys = {
        (
            int(row.num_tensor_parallel_workers),
            int(row.prefill_chunk_size),
            int(row.kv_cache_size),
            int(row.batch_size),
            _normalize_bool(row.is_prefill),
        )
        for row in frame.itertuples(index=False)
    }
    if len(frame) != len(observed_keys):
        raise ValueError(f"{path.name} contains duplicate standard attention keys.")
    if observed_keys != required_keys:
        raise ValueError(
            f"{path.name} standard attention coverage mismatch: "
            f"missing={len(required_keys - observed_keys)}, "
            f"extra={len(observed_keys - required_keys)}."
        )

    is_prefill = frame["is_prefill"].map(_normalize_bool)
    prefill_min = _require_finite_positive(
        frame,
        path=path,
        column="time_stats.attn_prefill.median",
        mask=is_prefill,
    )
    decode_min = _require_finite_positive(
        frame,
        path=path,
        column="time_stats.attn_decode.median",
        mask=~is_prefill,
    )
    return {
        "row_count": len(frame),
        "prefill_row_count": int(is_prefill.sum()),
        "decode_row_count": int((~is_prefill).sum()),
        "minimum_positive_prefill_ms": prefill_min,
        "minimum_positive_decode_ms": decode_min,
    }


def _validate_true_mixed_attention(
    frame: pd.DataFrame,
    *,
    path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    required_keys = _true_mixed_attention_required_keys(manifest)
    observed_keys = {
        (
            int(row.num_tensor_parallel_workers),
            _normalize_int_list(row.prefill_seq_lens),
            _normalize_int_list(row.prefill_kv_cache_sizes),
            _normalize_int_list(row.decode_kv_cache_sizes),
        )
        for row in frame.itertuples(index=False)
    }
    if len(frame) != len(observed_keys):
        raise ValueError(f"{path.name} contains duplicate true-mixed attention keys.")
    if observed_keys != required_keys:
        raise ValueError(
            f"{path.name} true-mixed attention coverage mismatch: "
            f"missing={len(required_keys - observed_keys)}, "
            f"extra={len(observed_keys - required_keys)}."
        )
    prefill_min = _require_finite_positive(
        frame,
        path=path,
        column="time_stats.attn_prefill.median",
    )
    decode_min = _require_finite_positive(
        frame,
        path=path,
        column="time_stats.attn_decode.median",
    )
    return {
        "row_count": len(frame),
        "minimum_positive_prefill_ms": prefill_min,
        "minimum_positive_decode_ms": decode_min,
    }


def _validate_attention_files(
    *,
    accepted_dir: Path,
    manifest: dict[str, Any],
    contract: Any,
    profile_method: str,
    measurement_type: str,
) -> dict[str, Any]:
    suffix = "_kernel_only" if profile_method == "record_function" else ""
    standard_path = accepted_dir / f"attention{suffix}.csv"
    true_mixed_path = accepted_dir / f"attention_true_mixed{suffix}.csv"
    combined_path = accepted_dir / f"attention_combined{suffix}.csv"
    standard = _read_csv(standard_path)
    true_mixed = _read_csv(true_mixed_path)
    combined = _read_csv(combined_path)
    for path, frame in (
        (standard_path, standard),
        (true_mixed_path, true_mixed),
        (combined_path, combined),
    ):
        _validate_common_metadata(
            frame,
            path=path,
            contract=contract,
            measurement_type=measurement_type,
        )
        _require_exact_values(
            frame,
            path=path,
            column="num_tensor_parallel_workers",
            expected=manifest["scope"]["tp_sizes"],
        )
        _require_constant(
            frame,
            path=path,
            column="n_embd",
            expected=contract.embedding_dim,
        )
        _require_constant(
            frame,
            path=path,
            column="n_q_head",
            expected=contract.num_q_heads,
        )
        _require_constant(
            frame,
            path=path,
            column="n_kv_head",
            expected=contract.num_kv_heads,
        )
        _require_constant(frame, path=path, column="block_size", expected=16)

    standard_report = _validate_standard_attention(
        standard,
        path=standard_path,
        manifest=manifest,
    )
    true_mixed_report = _validate_true_mixed_attention(
        true_mixed,
        path=true_mixed_path,
        manifest=manifest,
    )
    if "is_true_mixed_batch" not in combined.columns:
        raise ValueError(
            f"{combined_path.name} is missing 'is_true_mixed_batch'."
        )
    is_true_mixed = combined["is_true_mixed_batch"].map(_normalize_bool)
    combined_standard = combined[~is_true_mixed].copy()
    combined_true_mixed = combined[is_true_mixed].copy()
    _validate_standard_attention(
        combined_standard,
        path=combined_path,
        manifest=manifest,
    )
    _validate_true_mixed_attention(
        combined_true_mixed,
        path=combined_path,
        manifest=manifest,
    )
    expected_combined_rows = len(standard) + len(true_mixed)
    if len(combined) != expected_combined_rows:
        raise ValueError(
            f"{combined_path.name} has {len(combined)} rows; "
            f"expected {expected_combined_rows}."
        )
    return {
        "standard": standard_report,
        "true_mixed": true_mixed_report,
        "combined_row_count": len(combined),
    }


def _validate_linear_file(
    *,
    accepted_dir: Path,
    manifest: dict[str, Any],
    contract: Any,
    model_config: ModelConfig,
    profile_method: str,
    measurement_type: str,
) -> dict[str, Any]:
    filename = (
        "linear_op_kernel_only.csv"
        if profile_method == "record_function"
        else "linear_op.csv"
    )
    path = accepted_dir / filename
    frame = _read_csv(path)
    _validate_common_metadata(
        frame,
        path=path,
        contract=contract,
        measurement_type=measurement_type,
    )
    required_keys = {
        (tp_size, token_count)
        for tp_size in manifest["scope"]["tp_sizes"]
        for token_count in manifest["scope"]["linear_op"]["num_tokens"]
    }
    observed_keys = {
        (int(row.num_tensor_parallel_workers), int(row.num_tokens))
        for row in frame.itertuples(index=False)
    }
    if len(frame) != len(observed_keys):
        raise ValueError(f"{path.name} contains duplicate linear-op keys.")
    if observed_keys != required_keys:
        raise ValueError(
            f"{path.name} linear-op coverage mismatch: "
            f"missing={len(required_keys - observed_keys)}, "
            f"extra={len(observed_keys - required_keys)}."
        )
    contract_target_columns = set(contract.linear_target_columns)
    minima: dict[str, float] = {}
    for tp_size in manifest["scope"]["tp_sizes"]:
        plan = build_profiling_plan(
            model_config=model_config,
            tp_size=tp_size,
            attn_tp=manifest["scope"]["tp_sizes"],
            ffn_tp=manifest["scope"]["tp_sizes"],
            is_moe=contract.is_moe and not contract.is_mixed_layer_moe,
        )
        tp_mask = (
            pd.to_numeric(
                frame["num_tensor_parallel_workers"],
                errors="coerce",
            )
            == tp_size
        )
        replicated_ops = set(plan["replicated_ops"])
        applicable_ops = [
            op_name
            for op_name in plan["enabled_ops"]
            if f"time_stats.{op_name}.median" in contract_target_columns
            and (tp_size == 1 or op_name not in replicated_ops)
        ]
        for op_name in applicable_ops:
            column = f"time_stats.{op_name}.median"
            minimum = _require_finite_positive(
                frame,
                path=path,
                column=column,
                mask=tp_mask,
            )
            minima[f"tp{tp_size}:{op_name}"] = minimum
    return {
        "row_count": len(frame),
        "validated_target_slices": len(minima),
        "minimum_positive_target_ms": min(minima.values()),
    }


def _validate_moe_file(
    *,
    accepted_dir: Path,
    manifest: dict[str, Any],
    contract: Any,
    model_config: ModelConfig,
    profile_method: str,
    measurement_type: str,
) -> dict[str, Any]:
    filename = (
        "moe_kernel_only.csv"
        if profile_method == "record_function"
        else "moe.csv"
    )
    path = accepted_dir / filename
    frame = _read_csv(path)
    _validate_common_metadata(
        frame,
        path=path,
        contract=contract,
        measurement_type=measurement_type,
    )
    contexts = manifest["models"][contract.model_name][
        "gating_runtime_contexts"
    ]
    required_keys = {
        (
            tp_size,
            ep_size,
            token_count,
            load_distribution,
            seed,
            manifest["scope"]["moe"]["routing_runtime_paths"][0],
            context,
        )
        for tp_size in manifest["scope"]["tp_sizes"]
        for ep_size in manifest["scope"]["moe"]["expert_parallel_sizes"]
        for token_count in manifest["scope"]["moe"]["num_tokens"]
        for load_distribution in manifest["scope"]["moe"]["load_distributions"]
        for seed in manifest["scope"]["moe"]["sample_seeds"]
        for context in contexts
    }
    if frame["seed"].isna().any():
        raise ValueError(f"{path.name} contains missing MoE sample seeds.")
    observed_keys = {
        (
            int(row.num_tensor_parallel_workers),
            int(row.expert_parallel_size),
            int(row.num_tokens),
            str(row.load_distribution),
            int(row.seed),
            str(row.routing_runtime_path),
            str(row.gating_runtime_context),
        )
        for row in frame.itertuples(index=False)
    }
    if len(frame) != len(observed_keys):
        raise ValueError(f"{path.name} contains duplicate MoE feature keys.")
    if observed_keys != required_keys:
        raise ValueError(
            f"{path.name} MoE coverage mismatch: "
            f"missing={len(required_keys - observed_keys)}, "
            f"extra={len(observed_keys - required_keys)}."
        )
    if any(
        context not in {"direct", "prefill_warmed"}
        for context in frame[MOE_GATING_RUNTIME_CONTEXT_COLUMN].astype(str)
    ):
        raise ValueError(f"{path.name} contains a non-canonical gating context.")
    for context in contexts:
        expected_metadata = get_moe_gating_runtime_context_metadata(context)
        context_rows = frame[
            frame[MOE_GATING_RUNTIME_CONTEXT_COLUMN].astype(str) == context
        ]
        _require_constant(
            context_rows,
            path=path,
            column=MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN,
            expected=expected_metadata[MOE_GATING_RUNTIME_CONTEXT_IMPL_COLUMN],
        )
    _require_constant(
        frame,
        path=path,
        column="num_experts",
        expected=int(model_config.num_experts),
    )
    _require_constant(
        frame,
        path=path,
        column="router_topk",
        expected=int(model_config.num_experts_per_tok),
    )
    _require_constant(
        frame,
        path=path,
        column="routing_runtime_path",
        expected=manifest["scope"]["moe"]["routing_runtime_paths"][0],
    )
    minima = {
        column: _require_finite_positive(frame, path=path, column=column)
        for column in contract.moe_target_columns
    }
    return {
        "row_count": len(frame),
        "context_row_counts": {
            context: int(
                (
                    frame[MOE_GATING_RUNTIME_CONTEXT_COLUMN].astype(str)
                    == context
                ).sum()
            )
            for context in contexts
        },
        "minimum_positive_target_ms": min(minima.values()),
    }


def _copy_csv(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite accepted CSV: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = _read_csv(source)
    frame.to_csv(destination, index=False)


def _merge_moe_contexts(
    *,
    sources: list[Path],
    destination: Path,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite accepted CSV: {destination}")
    frames = [_read_csv(path) for path in sources]
    merged = pd.concat(frames, ignore_index=True)
    feature_columns = [
        column for column in merged.columns if not column.startswith("time_stats.")
    ]
    duplicate_mask = merged.duplicated(subset=feature_columns, keep=False)
    if duplicate_mask.any():
        raise ValueError(
            f"MoE context merge found {int(duplicate_mask.sum())} duplicate rows."
        )
    merged = merged.sort_values(feature_columns, kind="stable").reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)


def _build_attention_command(
    *,
    manifest: dict[str, Any],
    model: str,
    method: str,
    output_dir: Path,
    num_gpus: int,
) -> list[str]:
    scope = manifest["scope"]
    attention = scope["attention"]
    standard_workloads = attention["standard_workloads"]
    decode_batch_sizes = sorted(
        {
            int(row["batch_size"])
            for row in standard_workloads
            if not row["is_prefill"]
        }
    )
    decode_kv_sizes = sorted(
        {
            int(row["kv_cache_size"])
            for row in standard_workloads
            if not row["is_prefill"]
        }
    )
    true_mixed = attention["true_mixed_workloads"]
    prefill_batch_sizes = sorted(
        {len(row["prefill_seq_lens"]) for row in true_mixed}
    )
    prefill_chunk_sizes = sorted(
        {int(row["prefill_seq_lens"][0]) for row in true_mixed}
    )
    true_mixed_decode_batch_sizes = sorted(
        {len(row["decode_kv_cache_sizes"]) for row in true_mixed}
    )
    true_mixed_decode_kv_sizes = sorted(
        {int(row["decode_kv_cache_sizes"][0]) for row in true_mixed}
    )
    return [
        sys.executable,
        "-m",
        "frontier.profiling.attention.main",
        "--disable_ray",
        "--num_gpus",
        str(num_gpus),
        "--output_dir",
        str(output_dir),
        "--device",
        "h200",
        "--models",
        model,
        "--num_tensor_parallel_workers",
        *[str(value) for value in scope["tp_sizes"]],
        "--max_model_len",
        str(attention["max_model_len"]),
        "--max_seq_len",
        str(attention["max_seq_len"]),
        "--min_batch_size",
        str(attention["min_batch_size"]),
        "--max_batch_size",
        str(attention["max_batch_size"]),
        "--batch_size_list",
        *[str(value) for value in decode_batch_sizes],
        "--decode_kv_cache_size_list",
        *[str(value) for value in decode_kv_sizes],
        "--enable_chunked_prefill_grid_search",
        "--enable_true_mixed",
        "--true_mixed_prefill_batch_sizes",
        *[str(value) for value in prefill_batch_sizes],
        "--true_mixed_prefill_chunk_sizes",
        *[str(value) for value in prefill_chunk_sizes],
        "--true_mixed_decode_batch_sizes",
        *[str(value) for value in true_mixed_decode_batch_sizes],
        "--true_mixed_decode_kv_cache_sizes",
        *[str(value) for value in true_mixed_decode_kv_sizes],
        "--true_mixed_prefill_kv_cache_size",
        "0",
        "--profile_method",
        method,
        "--yes",
    ]


def _build_linear_command(
    *,
    manifest: dict[str, Any],
    model: str,
    contract: Any,
    method: str,
    output_dir: Path,
    num_gpus: int,
) -> list[str]:
    tp_sizes = [str(value) for value in manifest["scope"]["tp_sizes"]]
    command = [
        sys.executable,
        "-m",
        "frontier.profiling.linear_op.main",
        "--disable_ray",
        "--num_gpus",
        str(num_gpus),
        "--output_dir",
        str(output_dir),
        "--device",
        "h200",
        "--models",
        model,
        "--num_tensor_parallel_workers",
        *tp_sizes,
        "--attn_tp",
        *tp_sizes,
        "--ffn_tp",
        *tp_sizes,
        "--max_tokens",
        str(manifest["scope"]["linear_op"]["max_tokens"]),
        "--num_tokens_list",
        *[
            str(value)
            for value in manifest["scope"]["linear_op"]["num_tokens"]
        ],
        "--profile_method",
        method,
        "--yes",
    ]
    if contract.is_moe and not contract.is_mixed_layer_moe:
        command.append("--is_moe")
    return command


def _build_moe_command(
    *,
    manifest: dict[str, Any],
    model: str,
    method: str,
    context: str,
    output_dir: Path,
    num_gpus: int,
) -> list[str]:
    scope = manifest["scope"]
    return [
        sys.executable,
        "-m",
        "frontier.profiling.moe.main",
        "--disable_ray",
        "--num_gpus",
        str(num_gpus),
        "--output_dir",
        str(output_dir),
        "--device",
        "h200",
        "--models",
        model,
        "--num_tensor_parallel_workers",
        *[str(value) for value in scope["tp_sizes"]],
        "--expert_parallel_sizes",
        *[str(value) for value in scope["moe"]["expert_parallel_sizes"]],
        "--max_tokens",
        str(scope["moe"]["max_tokens"]),
        "--num_tokens_list",
        *[str(value) for value in scope["moe"]["num_tokens"]],
        "--enable_load_imbalance",
        "--load_distributions",
        *scope["moe"]["load_distributions"],
        "--num_samples_per_distribution",
        str(len(scope["moe"]["sample_seeds"])),
        "--routing_runtime_path",
        scope["moe"]["routing_runtime_paths"][0],
        "--gating_runtime_context",
        context,
        "--profile_method",
        method,
        "--yes",
    ]


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model {args.model!r}; expected one of {SUPPORTED_MODELS}."
        )
    if args.model not in manifest["scope"]["models"]:
        raise ValueError(f"Model {args.model!r} is absent from the frozen manifest.")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != manifest["source_commit"]:
        raise ValueError(
            f"Manifest source_commit={manifest['source_commit']} does not match "
            f"current HEAD={current_commit}."
        )

    model_root = args.output_root.resolve() / args.model
    if model_root.exists():
        raise FileExistsError(
            f"Refusing to reuse existing model output directory: {model_root}"
        )
    raw_root = model_root / "raw"
    log_root = model_root / "logs"
    accepted_dir = model_root / "accepted" / args.model
    status_path = model_root / "status.json"
    model_root.mkdir(parents=True)

    contract = build_model_contract(args.model)
    model_config = ModelConfig.from_model_name(args.model)
    command_records: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    try:
        for method in manifest["scope"]["profile_methods"]:
            attention_output = raw_root / "attention" / method
            _run_command(
                label=f"attention_{method}",
                command=_build_attention_command(
                    manifest=manifest,
                    model=args.model,
                    method=method,
                    output_dir=attention_output,
                    num_gpus=args.num_gpus,
                ),
                log_path=log_root / f"attention_{method}.log",
                timeout_seconds=args.command_timeout_seconds,
                records=command_records,
            )

            linear_output = raw_root / "linear_op" / method
            _run_command(
                label=f"linear_op_{method}",
                command=_build_linear_command(
                    manifest=manifest,
                    model=args.model,
                    contract=contract,
                    method=method,
                    output_dir=linear_output,
                    num_gpus=args.num_gpus,
                ),
                log_path=log_root / f"linear_op_{method}.log",
                timeout_seconds=args.command_timeout_seconds,
                records=command_records,
            )

            if contract.is_moe:
                for context in manifest["models"][args.model][
                    "gating_runtime_contexts"
                ]:
                    moe_output = raw_root / "moe" / method / context
                    _run_command(
                        label=f"moe_{method}_{context}",
                        command=_build_moe_command(
                            manifest=manifest,
                            model=args.model,
                            method=method,
                            context=context,
                            output_dir=moe_output,
                            num_gpus=args.num_gpus,
                        ),
                        log_path=(
                            log_root / f"moe_{method}_{context}.log"
                        ),
                        timeout_seconds=args.command_timeout_seconds,
                        records=command_records,
                    )

        accepted_dir.mkdir(parents=True)
        for method in manifest["scope"]["profile_methods"]:
            attention_source_dir = (
                raw_root
                / "attention"
                / method
                / "compute"
                / "h200"
                / args.model
            )
            for filename in (
                "attention.csv",
                "attention_true_mixed.csv",
                "attention_combined.csv",
            ):
                if method == "record_function":
                    filename = filename.replace(".csv", "_kernel_only.csv")
                _copy_csv(
                    attention_source_dir / filename,
                    accepted_dir / filename,
                )

            linear_filename = (
                "linear_op_kernel_only.csv"
                if method == "record_function"
                else "linear_op.csv"
            )
            linear_source = (
                raw_root
                / "linear_op"
                / method
                / "compute"
                / "h200"
                / args.model
                / linear_filename
            )
            _copy_csv(linear_source, accepted_dir / linear_filename)

            if contract.is_moe:
                moe_filename = (
                    "moe_kernel_only.csv"
                    if method == "record_function"
                    else "moe.csv"
                )
                moe_sources = [
                    raw_root
                    / "moe"
                    / method
                    / context
                    / "compute"
                    / "h200"
                    / args.model
                    / moe_filename
                    for context in manifest["models"][args.model][
                        "gating_runtime_contexts"
                    ]
                ]
                _merge_moe_contexts(
                    sources=moe_sources,
                    destination=accepted_dir / moe_filename,
                )

        validation: dict[str, Any] = {
            "status": "PASS",
            "model": args.model,
            "source_commit": current_commit,
            "files": {},
        }
        for method, measurement_type in manifest["scope"][
            "profile_methods"
        ].items():
            validation["files"][f"attention:{method}"] = (
                _validate_attention_files(
                    accepted_dir=accepted_dir,
                    manifest=manifest,
                    contract=contract,
                    profile_method=method,
                    measurement_type=measurement_type,
                )
            )
            validation["files"][f"linear_op:{method}"] = _validate_linear_file(
                accepted_dir=accepted_dir,
                manifest=manifest,
                contract=contract,
                model_config=model_config,
                profile_method=method,
                measurement_type=measurement_type,
            )
            if contract.is_moe:
                validation["files"][f"moe:{method}"] = _validate_moe_file(
                    accepted_dir=accepted_dir,
                    manifest=manifest,
                    contract=contract,
                    model_config=model_config,
                    profile_method=method,
                    measurement_type=measurement_type,
                )
        validation["accepted_dir"] = str(accepted_dir)
        validation["accepted_csv_count"] = len(list(accepted_dir.glob("*.csv")))
        validation["total_physical_rows"] = sum(
            len(pd.read_csv(path)) for path in accepted_dir.glob("*.csv")
        )
        _write_json(model_root / "validation.json", validation)

        status = {
            "status": "PASS",
            "model": args.model,
            "source_commit": current_commit,
            "manifest": str(manifest_path),
            "started_at": started_at,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "num_gpus": args.num_gpus,
            "commands": command_records,
            "accepted_dir": str(accepted_dir),
            "validation_report": str(model_root / "validation.json"),
        }
        _write_json(status_path, status)
        return status
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "model": args.model,
            "source_commit": current_commit,
            "manifest": str(manifest_path),
            "started_at": started_at,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "num_gpus": args.num_gpus,
            "commands": command_records,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(status_path, failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one model from the frozen H200 manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--command-timeout-seconds", type=int, default=7_200)
    args = parser.parse_args()
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be positive.")
    if args.command_timeout_seconds <= 0:
        raise ValueError("--command-timeout-seconds must be positive.")
    return args


def main() -> None:
    args = parse_args()
    status = run_model(args)
    print(
        json.dumps(
            {
                "status": status["status"],
                "model": status["model"],
                "accepted_dir": status["accepted_dir"],
                "validation_report": status["validation_report"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
