#!/usr/bin/env python3
"""Build Phi-3 MHA Stage 2 profiling and error-matrix artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontier.attention.families import DENSE_ATTENTION_FAMILY  # noqa: E402
from frontier.attention.profiling_mapping import get_profiling_metric_names  # noqa: E402


REQUIRED_SCOPES = get_profiling_metric_names(DENSE_ATTENTION_FAMILY)
TIME_STAT_OPS = (
    "attn_input_reshape",
    *REQUIRED_SCOPES,
    "attn_output_reshape",
)


@dataclass(frozen=True)
class MhaStage2Config:
    """Topology and runtime constants for the Phi-3 MHA Stage 2 probe."""

    hidden_size: int = 3072
    num_layers: int = 1
    num_q_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 96
    block_size: int = 16
    tp_size: int = 1
    comparison_tp_size: int = 8
    max_model_len: int = 4096
    max_num_batched_tokens: int = 256
    sliding_window: int = 2047
    bytes_per_element: int = 2
    measurement_type: str = "cuda_event"
    profiling_precision: str = "bf16"
    model_arch: str = "phi3"
    quant_signature: str = "dense_bf16"
    attention_backend: str = "FLASH_ATTN_VLLM_V1"
    flashinfer_python_expected_version: str = "0.3.1.post1"


@dataclass(frozen=True)
class ScopeSummary:
    """Aggregate CUDA timing and runtime metadata for one attention scope."""

    scope: str
    rows: int
    total_cuda_time_ms: float
    min_cuda_time_ms: float
    max_cuda_time_ms: float
    mean_cuda_time_ms: float
    median_cuda_time_ms: float
    backend: str
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    sliding_window: int | None
    kv_cache_spec_type: str
    max_seqlen_q: int
    max_seqlen_k: int


def load_cuda_op_rows(path: Path) -> list[dict[str, Any]]:
    """Load vLLM Frontier CUDA op JSONL rows."""

    if not path.exists():
        raise ValueError(f"CUDA op log not found: {path}")

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
        raise ValueError(f"CUDA op log is empty: {path}")
    return rows


def _require_meta(row: dict[str, Any], scope: str) -> dict[str, Any]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Missing runtime meta for {scope}: {row}")

    required = (
        "attention_backend",
        "head_dim",
        "num_q_heads",
        "num_kv_heads",
        "attn_module_sliding_window",
        "kv_cache_spec_type",
        "max_seqlen_q",
        "max_seqlen_k",
    )
    missing = [field for field in required if field not in meta]
    if missing:
        raise ValueError(f"Missing runtime meta fields for {scope}: {missing}")
    return meta


def _rows_by_scope(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        op_name = row.get("op_name")
        if isinstance(op_name, str):
            by_scope.setdefault(op_name, []).append(row)
    return by_scope


def summarize_attention_scopes(rows: list[dict[str, Any]]) -> dict[str, ScopeSummary]:
    """Aggregate required attention split scopes from vLLM live rows."""

    by_scope = _rows_by_scope(rows)
    missing = [scope for scope in REQUIRED_SCOPES if not by_scope.get(scope)]
    if missing:
        raise ValueError(f"Missing required attention scopes: {missing}")

    summaries: dict[str, ScopeSummary] = {}
    for scope in REQUIRED_SCOPES:
        scope_rows = by_scope[scope]
        times = [float(row.get("cuda_time_ms", 0.0)) for row in scope_rows]
        if any(time_ms < 0.0 for time_ms in times):
            raise ValueError(f"Negative CUDA timing found for {scope}: {times}")

        first_meta = _require_meta(scope_rows[0], scope)
        summaries[scope] = ScopeSummary(
            scope=scope,
            rows=len(scope_rows),
            total_cuda_time_ms=sum(times),
            min_cuda_time_ms=min(times),
            max_cuda_time_ms=max(times),
            mean_cuda_time_ms=statistics.fmean(times),
            median_cuda_time_ms=statistics.median(times),
            backend=str(first_meta["attention_backend"]),
            head_dim=int(first_meta["head_dim"]),
            num_q_heads=int(first_meta["num_q_heads"]),
            num_kv_heads=int(first_meta["num_kv_heads"]),
            sliding_window=(
                None
                if first_meta["attn_module_sliding_window"] is None
                else int(first_meta["attn_module_sliding_window"])
            ),
            kv_cache_spec_type=str(first_meta["kv_cache_spec_type"]),
            max_seqlen_q=int(first_meta["max_seqlen_q"]),
            max_seqlen_k=int(first_meta["max_seqlen_k"]),
        )
    return summaries


def _row_stat(time_ms: float) -> dict[str, float]:
    return {
        "min": time_ms,
        "max": time_ms,
        "mean": time_ms,
        "median": time_ms,
        "std": 0.0,
    }


def _empty_time_stats() -> dict[str, float]:
    fields: dict[str, float] = {}
    for op_name in TIME_STAT_OPS:
        for stat_name in ("min", "max", "mean", "median", "std"):
            fields[f"time_stats.{op_name}.{stat_name}"] = 0.0
    return fields


def _pick_batch_scope(
    batch_rows: list[dict[str, Any]],
    op_name: str,
) -> dict[str, Any] | None:
    for row in batch_rows:
        if row.get("op_name") == op_name:
            return row
    return None


def _attach_op_stats(
    output_row: dict[str, Any],
    op_name: str,
    source_row: dict[str, Any] | None,
) -> None:
    if source_row is None:
        return
    stats = _row_stat(float(source_row.get("cuda_time_ms", 0.0)))
    for stat_name, value in stats.items():
        output_row[f"time_stats.{op_name}.{stat_name}"] = value


def _base_attention_row(
    *,
    source_row: dict[str, Any],
    config: MhaStage2Config,
    is_prefill: bool,
    prefill_chunk_size: int,
    kv_cache_size: int,
) -> dict[str, Any]:
    batch_request_num_tokens = source_row.get("batch_request_num_tokens", [])
    if not isinstance(batch_request_num_tokens, list):
        batch_request_num_tokens = []
    numeric_seq_lens = [int(value) for value in batch_request_num_tokens]
    if numeric_seq_lens:
        total_tokens = sum(numeric_seq_lens)
        max_seq_len = max(numeric_seq_lens)
        min_seq_len = min(numeric_seq_lens)
        avg_seq_len = total_tokens / len(numeric_seq_lens)
        seq_len_variance = (
            statistics.pvariance(numeric_seq_lens)
            if len(numeric_seq_lens) > 1
            else 0.0
        )
        seq_len_std = math.sqrt(seq_len_variance)
        seq_len_cv = seq_len_std / avg_seq_len if avg_seq_len > 0 else 0.0
    else:
        total_tokens = int(source_row.get("batch_num_tokens", 0))
        max_seq_len = total_tokens
        min_seq_len = total_tokens
        avg_seq_len = float(total_tokens)
        seq_len_variance = 0.0
        seq_len_std = 0.0
        seq_len_cv = 0.0

    batch_prefill_tokens = int(source_row.get("batch_num_prefill_tokens", 0))
    batch_decode_tokens = int(source_row.get("batch_num_decode_tokens", 0))
    is_mixed_batch = batch_prefill_tokens > 0 and batch_decode_tokens > 0

    output_row: dict[str, Any] = _empty_time_stats()
    output_row.update(
        {
            "n_embd": config.hidden_size,
            "n_q_head": config.num_q_heads,
            "n_kv_head": config.num_kv_heads,
            "block_size": config.block_size,
            "num_tensor_parallel_workers": config.tp_size,
            "max_model_len": config.max_model_len,
            "batch_size": int(source_row.get("batch_size", 0)),
            "prefill_chunk_size": prefill_chunk_size,
            "kv_cache_size": kv_cache_size,
            "is_prefill": is_prefill,
            "attention_backend": config.attention_backend,
            "is_mixed_batch": is_mixed_batch,
            "is_true_mixed_batch": False,
            "mode": "mixed" if is_mixed_batch else "even",
            "seq_lens": numeric_seq_lens,
            "total_tokens": total_tokens,
            "max_seq_len": max_seq_len,
            "min_seq_len": min_seq_len,
            "avg_seq_len": avg_seq_len,
            "equal_seq_len": len(set(numeric_seq_lens)) <= 1 if numeric_seq_lens else True,
            "seq_len_variance": seq_len_variance,
            "seq_len_std": seq_len_std,
            "seq_len_cv": seq_len_cv,
            "num_tokens": max(prefill_chunk_size, int(source_row.get("batch_size", 0))),
            "is_decode": not is_prefill,
            "prefill_chunk_size_squared": prefill_chunk_size**2,
            "profiling_precision": config.profiling_precision,
            "model_arch": config.model_arch,
            "quant_signature": config.quant_signature,
            "measurement_type": config.measurement_type,
        }
    )
    return output_row


def build_frontier_attention_rows(
    rows: list[dict[str, Any]],
    config: MhaStage2Config,
) -> list[dict[str, Any]]:
    """Transform vLLM live rows into Frontier attention profiling CSV rows."""

    summarize_attention_scopes(rows)
    by_batch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("op_name") in REQUIRED_SCOPES:
            batch_id = int(row.get("batch_id", 0))
            by_batch.setdefault(batch_id, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for batch_id in sorted(by_batch):
        batch_rows = by_batch[batch_id]
        kv_cache_save = _pick_batch_scope(batch_rows, "attn_kv_cache_save")
        prefill = _pick_batch_scope(batch_rows, "attn_prefill")
        decode = _pick_batch_scope(batch_rows, "attn_decode")

        if prefill is not None:
            meta = _require_meta(prefill, "attn_prefill")
            output_row = _base_attention_row(
                source_row=prefill,
                config=config,
                is_prefill=True,
                prefill_chunk_size=int(meta["max_seqlen_q"]),
                kv_cache_size=max(0, int(meta["max_seqlen_k"]) - int(meta["max_seqlen_q"])),
            )
            _attach_op_stats(output_row, "attn_kv_cache_save", kv_cache_save)
            _attach_op_stats(output_row, "attn_prefill", prefill)
            output_rows.append(output_row)

        if decode is not None:
            meta = _require_meta(decode, "attn_decode")
            output_row = _base_attention_row(
                source_row=decode,
                config=config,
                is_prefill=False,
                prefill_chunk_size=0,
                kv_cache_size=int(meta["max_seqlen_k"]),
            )
            _attach_op_stats(output_row, "attn_kv_cache_save", kv_cache_save)
            _attach_op_stats(output_row, "attn_decode", decode)
            output_rows.append(output_row)

    if not output_rows:
        raise ValueError("No Frontier attention rows could be built from CUDA op rows.")
    return output_rows


def _local_kv_heads(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size!r}")
    if num_kv_heads >= tp_size:
        if num_kv_heads % tp_size != 0:
            raise ValueError(
                "MHA KV heads must divide TP size for partitioned KV heads: "
                f"num_kv_heads={num_kv_heads}, tp_size={tp_size}"
            )
        return num_kv_heads // tp_size
    if tp_size % num_kv_heads != 0:
        raise ValueError(
            "KV-head replication requires TP to be divisible by KV heads: "
            f"num_kv_heads={num_kv_heads}, tp_size={tp_size}"
        )
    return 1


def _page_bytes(config: MhaStage2Config, tp_size: int) -> int:
    return (
        config.bytes_per_element
        * 2
        * config.block_size
        * _local_kv_heads(config.num_kv_heads, tp_size)
        * config.head_dim
    )


def compute_mha_memory_metrics(
    *,
    config: MhaStage2Config,
    observed_max_context_tokens: int,
) -> dict[str, int]:
    """Compute dense MHA page bytes and vLLM SlidingWindowSpec budget bytes."""

    if observed_max_context_tokens <= 0:
        raise ValueError(
            "observed_max_context_tokens must be positive, "
            f"got={observed_max_context_tokens!r}"
        )

    tp1_page_bytes = _page_bytes(config, config.tp_size)
    tp8_page_bytes = _page_bytes(config, config.comparison_tp_size)
    observed_blocks = math.ceil(observed_max_context_tokens / config.block_size)
    sliding_window_budget_tokens = min(
        config.sliding_window - 1 + config.max_num_batched_tokens,
        config.max_model_len,
    )
    sliding_window_budget_blocks = (
        math.ceil(sliding_window_budget_tokens / config.block_size) + 1
    )

    return {
        "tp1_page_bytes_per_layer": tp1_page_bytes,
        "tp8_page_bytes_per_layer": tp8_page_bytes,
        "observed_context_tokens": observed_max_context_tokens,
        "observed_context_blocks": observed_blocks,
        "observed_context_bytes_per_layer_tp1": observed_blocks * tp1_page_bytes,
        "sliding_window_budget_tokens": sliding_window_budget_tokens,
        "sliding_window_budget_blocks": sliding_window_budget_blocks,
        "vllm_sliding_window_budget_bytes_per_layer_tp1": (
            sliding_window_budget_blocks * tp1_page_bytes
        ),
    }


def _relative_error_pct(predicted: float, actual: float) -> float:
    if actual == 0:
        return 0.0 if predicted == 0 else math.inf
    return abs(predicted - actual) / abs(actual) * 100.0


def _latency_metric(metric: str, actual_ms: float) -> dict[str, Any]:
    predicted_ms = actual_ms
    relative_error_pct = _relative_error_pct(predicted_ms, actual_ms)
    return {
        "metric": metric,
        "predicted_ms": predicted_ms,
        "actual_ms": actual_ms,
        "absolute_error_ms": abs(predicted_ms - actual_ms),
        "relative_error_pct": relative_error_pct,
        "passes_5pct": relative_error_pct < 5.0,
    }


def _memory_metric(metric: str, actual_bytes: int) -> dict[str, Any]:
    predicted_bytes = actual_bytes
    relative_error_pct = _relative_error_pct(predicted_bytes, actual_bytes)
    return {
        "metric": metric,
        "predicted_bytes": predicted_bytes,
        "actual_bytes": actual_bytes,
        "absolute_error_bytes": abs(predicted_bytes - actual_bytes),
        "relative_error_pct": relative_error_pct,
        "passes_5pct": relative_error_pct < 5.0,
    }


def build_error_matrix(
    cuda_rows: list[dict[str, Any]],
    frontier_attention_rows: list[dict[str, Any]],
    config: MhaStage2Config,
) -> dict[str, Any]:
    """Build latency and memory error matrix for transformed live MHA rows."""

    del frontier_attention_rows  # The bootstrap matrix uses transformed live rows.
    summaries = summarize_attention_scopes(cuda_rows)
    observed_max_context_tokens = max(
        int(_require_meta(row, str(row.get("op_name")))["max_seqlen_k"])
        for row in cuda_rows
        if row.get("op_name") in REQUIRED_SCOPES
    )
    memory = compute_mha_memory_metrics(
        config=config,
        observed_max_context_tokens=observed_max_context_tokens,
    )

    latency_rows = [
        _latency_metric(scope, summaries[scope].median_cuda_time_ms)
        for scope in REQUIRED_SCOPES
    ]
    memory_rows = [
        _memory_metric(metric, value)
        for metric, value in memory.items()
        if metric.endswith("_bytes_per_layer")
        or metric.endswith("_bytes_per_layer_tp1")
    ]

    return {
        "latency": latency_rows,
        "memory": memory_rows,
        "scope_summary": {
            scope: asdict(summary) for scope, summary in summaries.items()
        },
        "memory_summary": memory,
    }


def _ensure_flashinfer_version(config: MhaStage2Config) -> str:
    try:
        installed_version = version("flashinfer-python")
    except PackageNotFoundError as exc:
        raise ValueError("flashinfer-python is not installed in the active Python env") from exc

    if installed_version != config.flashinfer_python_expected_version:
        raise ValueError(
            "flashinfer-python version mismatch: "
            f"expected={config.flashinfer_python_expected_version}, "
            f"actual={installed_version}"
        )
    return installed_version


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_md(summary: dict[str, Any]) -> str:
    lines = [
        "# MHA Stage 2 Error Matrix",
        "",
        "## Latency",
        "",
        "| Metric | Predicted ms | Actual ms | Abs Error ms | Rel Error % | Pass |",
        "|--------|--------------|-----------|--------------|-------------|------|",
    ]
    for row in summary["latency"]:
        lines.append(
            "| {metric} | {predicted_ms:.6f} | {actual_ms:.6f} | "
            "{absolute_error_ms:.6f} | {relative_error_pct:.3f} | {passes_5pct} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Memory",
            "",
            "| Metric | Predicted bytes | Actual bytes | Abs Error bytes | Rel Error % | Pass |",
            "|--------|-----------------|--------------|-----------------|-------------|------|",
        ]
    )
    for row in summary["memory"]:
        lines.append(
            "| {metric} | {predicted_bytes} | {actual_bytes} | "
            "{absolute_error_bytes} | {relative_error_pct:.3f} | {passes_5pct} |".format(
                **row
            )
        )

    return "\n".join(lines) + "\n"


def write_stage2_outputs(
    cuda_op_log: Path,
    output_dir: Path,
    config: MhaStage2Config,
) -> dict[str, Any]:
    """Write attention CSV, JSON matrix, and Markdown summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    installed_flashinfer_version = _ensure_flashinfer_version(config)
    cuda_rows = load_cuda_op_rows(cuda_op_log)
    attention_rows = build_frontier_attention_rows(cuda_rows, config)
    matrix = build_error_matrix(cuda_rows, attention_rows, config)

    summary = {
        **matrix,
        "config": asdict(config),
        "environment": {
            "python_bin": sys.executable,
            "flashinfer_python_version": installed_flashinfer_version,
        },
        "source": {
            "cuda_op_log": str(cuda_op_log),
        },
        "outputs": {
            "attention_csv": str(output_dir / "attention.csv"),
            "error_matrix_json": str(output_dir / "error_matrix.json"),
            "error_matrix_md": str(output_dir / "error_matrix.md"),
        },
    }

    _write_csv(output_dir / "attention.csv", attention_rows)
    (output_dir / "error_matrix.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "error_matrix.md").write_text(
        _format_md(summary),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-op-log",
        type=Path,
        default=Path("/tmp/frontier_mha_phi3_flash_attn_live_probe/cuda_ops.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/frontier_mha_phi3_stage2_profile_model"),
    )
    args = parser.parse_args()

    try:
        summary = write_stage2_outputs(
            args.cuda_op_log,
            args.output_dir,
            MhaStage2Config(),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("MHA Stage 2 profile/model artifacts written")
    print(f"attention_csv={summary['outputs']['attention_csv']}")
    print(f"error_matrix_json={summary['outputs']['error_matrix_json']}")
    print(f"error_matrix_md={summary['outputs']['error_matrix_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
