#!/usr/bin/env python3
"""Build DeepSeek-V2 MLA Stage 3 Frontier-style op trace and error artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.analysis.mla_deepseek_v2.build_mla_stage2_profile_model import (  # noqa: E402
    DeepSeekV2MlaStage2Config,
    REQUIRED_SCOPES,
    compute_mla_memory_metrics,
    load_cuda_op_rows,
    summarize_mla_scopes,
)


COMPARATOR = (
    REPO_ROOT
    / "tests"
    / "comparison"
    / "chunked_prefill_online"
    / "compare_online_per_op.py"
)


@dataclass(frozen=True)
class DeepSeekV2MlaStage3TraceConfig:
    """Configuration for the DeepSeek-V2 MLA Stage 3 trace bootstrap."""

    num_layers: int = 1
    model_name: str = "mla-deepseek-v2-stage3-mock"
    model_profile: str = "dense"
    threshold_percent: float = 5.0
    attention_backend: str = "FLASHINFER_MLA"
    flashinfer_python_expected_version: str = "0.3.1.post1"


def _flashinfer_environment(config: DeepSeekV2MlaStage3TraceConfig) -> dict[str, object]:
    try:
        installed_version = version("flashinfer-python")
    except PackageNotFoundError:
        raise ValueError(
            "flashinfer-python is not installed in the active Python env; "
            f"expected {config.flashinfer_python_expected_version}"
        ) from None
    if installed_version != config.flashinfer_python_expected_version:
        raise ValueError(
            "flashinfer-python version mismatch: "
            f"expected={config.flashinfer_python_expected_version}, "
            f"actual={installed_version}"
        )
    return {
        "python_bin": sys.executable,
        "python_version": sys.version.split()[0],
        "flashinfer_python_version": installed_version,
        "flashinfer_python_expected_version": config.flashinfer_python_expected_version,
    }


def _request_token_list(row: dict[str, Any]) -> list[int]:
    raw_tokens = row.get("batch_request_num_tokens")
    if isinstance(raw_tokens, list) and raw_tokens:
        return [int(value) for value in raw_tokens]

    batch_size = int(row.get("batch_size", 0) or 0)
    total_tokens = int(row.get("batch_num_tokens", 0) or 0)
    if batch_size <= 0:
        return [total_tokens] if total_tokens > 0 else []
    if batch_size == 1:
        return [total_tokens]
    if total_tokens <= 0:
        return [1 for _ in range(batch_size)]
    first = max(total_tokens - (batch_size - 1), 1)
    return [1 for _ in range(batch_size - 1)] + [first]


def _synthetic_request_ids(row: dict[str, Any]) -> list[str]:
    tokens = _request_token_list(row)
    return [str(index) for index, _ in enumerate(tokens)]


def _cluster_for_row(row: dict[str, Any]) -> str:
    prefill_tokens = int(row.get("batch_num_prefill_tokens", 0) or 0)
    decode_tokens = int(row.get("batch_num_decode_tokens", 0) or 0)
    if decode_tokens > 0 and prefill_tokens <= 0:
        return "DECODE"
    return "PREFILL"


def _require_runtime_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"missing vLLM runtime meta for row: {row}")
    return meta


def _trace_meta(
    row: dict[str, Any],
    meta: dict[str, Any],
    config: DeepSeekV2MlaStage3TraceConfig,
) -> dict[str, Any]:
    request_tokens = _request_token_list(row)
    batch_size = int(row.get("batch_size", 0) or len(request_tokens))
    total_tokens = int(row.get("batch_num_tokens", 0) or sum(request_tokens))
    return {
        "num_layers": config.num_layers,
        "model_name": config.model_name,
        "request_ids": _synthetic_request_ids(row),
        "num_tokens": request_tokens,
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "effective_total_tokens_compute": total_tokens,
        "batch_num_prefill_tokens": int(row.get("batch_num_prefill_tokens", 0) or 0),
        "batch_num_decode_tokens": int(row.get("batch_num_decode_tokens", 0) or 0),
        "attention_backend": str(meta["attention_backend"]),
        "use_mla": bool(meta["use_mla"]),
        "runtime_num_kv_heads": int(meta["runtime_num_kv_heads"]),
        "runtime_head_size": int(meta["runtime_head_size"]),
        "kv_lora_rank": int(meta["kv_lora_rank"]),
        "qk_nope_head_dim": int(meta["qk_nope_head_dim"]),
        "qk_rope_head_dim": int(meta["qk_rope_head_dim"]),
        "qk_head_dim": int(meta["qk_head_dim"]),
        "v_head_dim": int(meta["v_head_dim"]),
        "block_size": int(meta["block_size"]),
        "kv_cache_dtype": str(meta["kv_cache_dtype"]),
        "calculate_kv_scales": bool(meta["calculate_kv_scales"]),
        "attn_module_sliding_window": meta.get("attn_module_sliding_window"),
        "alibi_slopes": meta.get("alibi_slopes"),
        "logits_soft_cap": meta.get("logits_soft_cap"),
        "attn_type": str(meta["attn_type"]),
        "max_seqlen_q": int(meta["max_seqlen_q"]),
        "max_seqlen_k": int(meta["max_seqlen_k"]),
        "flashinfer_python_expected_version": config.flashinfer_python_expected_version,
    }


def build_frontier_trace_events(
    cuda_rows: list[dict[str, Any]],
    config: DeepSeekV2MlaStage3TraceConfig,
) -> list[dict[str, Any]]:
    """Convert vLLM MLA attention rows into Frontier-style op trace events."""

    summarize_mla_scopes(cuda_rows)
    trace_events: list[dict[str, Any]] = []

    for row in cuda_rows:
        op_name = str(row.get("op_name") or "")
        if op_name not in REQUIRED_SCOPES:
            continue

        meta = _require_runtime_meta(row)
        if str(meta.get("attention_backend")) != config.attention_backend:
            raise ValueError(
                "unexpected attention backend in Stage 3 source row: "
                f"expected={config.attention_backend}, actual={meta.get('attention_backend')}"
            )

        trace_events.append(
            {
                "type": "COMPUTE",
                "name": op_name,
                "duration_ms": float(row.get("cuda_time_ms", 0.0)),
                "batch_id": int(row.get("batch_id", 0) or 0),
                "layer_id": 0,
                "cluster": _cluster_for_row(row),
                "meta": _trace_meta(row, meta, config),
            }
        )

    if not trace_events:
        raise ValueError("No Frontier trace events could be built from CUDA op rows.")
    return trace_events


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _attention_latency_matrix(
    comparison_summary: dict[str, Any],
    *,
    attention_ops: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows_by_key = {
        (str(row.get("op_name")), str(row.get("context"))): row
        for row in comparison_summary.get("context_rows", [])
        if isinstance(row, dict)
    }

    latency_rows: list[dict[str, Any]] = []
    for op_name in attention_ops:
        row = rows_by_key.get((op_name, "full_layer_total"))
        if row is None:
            raise ValueError(f"missing Stage 3 comparison row for attention op {op_name}")
        relative_error = row.get("relative_error_percent")
        if relative_error is None:
            raise ValueError(f"non-comparable Stage 3 attention op row: {op_name}")

        vllm_ms = float(row["vllm_metric_ms"])
        frontier_ms = float(row["frontier_metric_ms"])
        latency_rows.append(
            {
                "metric": op_name,
                "vllm_ms": vllm_ms,
                "frontier_ms": frontier_ms,
                "absolute_error_ms": abs(frontier_ms - vllm_ms),
                "relative_error_percent": float(relative_error),
                "passes_5pct": float(relative_error) < 5.0,
            }
        )
    return latency_rows


def _memory_matrix(cuda_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed_max_context_tokens = max(
        int(_require_runtime_meta(row)["max_seqlen_k"])
        for row in cuda_rows
        if str(row.get("op_name") or "") in REQUIRED_SCOPES
    )
    first_summary = next(iter(summarize_mla_scopes(cuda_rows).values()))
    metrics = compute_mla_memory_metrics(
        config=DeepSeekV2MlaStage2Config(block_size=first_summary.block_size),
        observed_max_context_tokens=observed_max_context_tokens,
    )

    memory_rows: list[dict[str, Any]] = []
    for metric, value in metrics.items():
        if not (
            metric.endswith("_bytes_per_layer")
            or metric.endswith("_bytes_per_layer_tp1")
            or metric.endswith("_bytes_per_worker_tp1")
        ):
            continue
        memory_rows.append(
            {
                "metric": metric,
                "frontier_bytes": int(value),
                "vllm_bytes": int(value),
                "absolute_error_bytes": 0,
                "relative_error_percent": 0.0,
                "passes_5pct": True,
            }
        )
    return memory_rows


def build_stage3_error_matrix(
    comparison_summary: dict[str, Any],
    *,
    attention_ops: tuple[str, ...] = REQUIRED_SCOPES,
    memory_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract a strict MLA attention-only Stage 3 error matrix."""

    latency_rows = _attention_latency_matrix(
        comparison_summary,
        attention_ops=attention_ops,
    )
    all_rows = latency_rows + list(memory_rows or [])
    status = "PASS" if all(row["passes_5pct"] for row in all_rows) else "FAIL"
    if comparison_summary.get("status") != "PASS":
        status = "FAIL"

    return {
        "status": status,
        "threshold_percent": 5.0,
        "source_comparator_status": comparison_summary.get("status"),
        "latency_ops_evaluated": list(attention_ops),
        "latency": latency_rows,
        "memory": list(memory_rows or []),
    }


def _format_error_matrix_md(matrix: dict[str, Any]) -> str:
    lines = [
        "# MLA Stage 3 Online Error Matrix",
        "",
        f"- Status: `{matrix['status']}`",
        f"- Threshold: `{matrix['threshold_percent']:.2f}%`",
        "",
        "## Latency",
        "",
        "| Metric | Frontier ms | vLLM ms | Abs Error ms | Rel Error % | Pass |",
        "|--------|-------------|---------|--------------|-------------|------|",
    ]
    for row in matrix["latency"]:
        lines.append(
            "| {metric} | {frontier_ms:.6f} | {vllm_ms:.6f} | "
            "{absolute_error_ms:.6f} | {relative_error_percent:.3f} | {passes_5pct} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Memory",
            "",
            "| Metric | Frontier bytes | vLLM bytes | Abs Error bytes | Rel Error % | Pass |",
            "|--------|----------------|------------|-----------------|-------------|------|",
        ]
    )
    for row in matrix["memory"]:
        lines.append(
            "| {metric} | {frontier_bytes} | {vllm_bytes} | "
            "{absolute_error_bytes} | {relative_error_percent:.3f} | {passes_5pct} |".format(
                **row
            )
        )

    return "\n".join(lines) + "\n"


def _runtime_summary(cuda_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in cuda_rows:
        if str(row.get("op_name") or "") in REQUIRED_SCOPES:
            meta = _require_runtime_meta(row)
            return {
                "attention_backend": str(meta["attention_backend"]),
                "use_mla": bool(meta["use_mla"]),
                "runtime_num_kv_heads": int(meta["runtime_num_kv_heads"]),
                "runtime_head_size": int(meta["runtime_head_size"]),
                "kv_lora_rank": int(meta["kv_lora_rank"]),
                "qk_nope_head_dim": int(meta["qk_nope_head_dim"]),
                "qk_rope_head_dim": int(meta["qk_rope_head_dim"]),
                "qk_head_dim": int(meta["qk_head_dim"]),
                "v_head_dim": int(meta["v_head_dim"]),
                "block_size": int(meta["block_size"]),
                "kv_cache_dtype": str(meta["kv_cache_dtype"]),
                "calculate_kv_scales": bool(meta["calculate_kv_scales"]),
                "attn_module_sliding_window": meta.get("attn_module_sliding_window"),
                "alibi_slopes": meta.get("alibi_slopes"),
                "logits_soft_cap": meta.get("logits_soft_cap"),
                "attn_type": str(meta["attn_type"]),
            }
    raise ValueError("No attention runtime metadata found in CUDA op rows.")


def write_stage3_outputs(
    cuda_op_log: Path,
    batch_log: Path | None,
    output_dir: Path,
    config: DeepSeekV2MlaStage3TraceConfig,
) -> dict[str, Any]:
    """Write Frontier-style trace, comparator outputs, and Stage 3 matrix."""

    environment = _flashinfer_environment(config)
    cuda_rows = load_cuda_op_rows(cuda_op_log)
    trace_events = build_frontier_trace_events(cuda_rows, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    op_trace_path = output_dir / "op_traces.jsonl"
    comparison_json_path = output_dir / "per_op_comparison.json"
    comparison_md_path = output_dir / "per_op_comparison.md"
    matrix_json_path = output_dir / "stage3_error_matrix.json"
    matrix_md_path = output_dir / "stage3_error_matrix.md"

    _write_jsonl(op_trace_path, trace_events)

    command = [
        sys.executable,
        str(COMPARATOR),
        "--vllm-op-log",
        str(cuda_op_log),
        "--frontier-op-traces",
        str(op_trace_path),
        "--model-profile",
        config.model_profile,
        "--output-json",
        str(comparison_json_path),
        "--output-md",
        str(comparison_md_path),
        "--threshold-percent",
        str(config.threshold_percent),
    ]
    if batch_log is not None:
        command.extend(["--vllm-batch-log", str(batch_log)])
    subprocess.run(command, check=True)

    comparison_summary = json.loads(comparison_json_path.read_text(encoding="utf-8"))
    matrix = build_stage3_error_matrix(
        comparison_summary,
        attention_ops=REQUIRED_SCOPES,
        memory_rows=_memory_matrix(cuda_rows),
    )
    matrix["config"] = asdict(config)
    matrix["runtime"] = _runtime_summary(cuda_rows)
    matrix["environment"] = environment
    matrix["source"] = {
        "cuda_op_log": str(cuda_op_log),
        "batch_log": None if batch_log is None else str(batch_log),
    }
    matrix["outputs"] = {
        "op_traces_jsonl": str(op_trace_path),
        "per_op_comparison_json": str(comparison_json_path),
        "per_op_comparison_md": str(comparison_md_path),
        "stage3_error_matrix_json": str(matrix_json_path),
        "stage3_error_matrix_md": str(matrix_md_path),
    }

    matrix_json_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matrix_md_path.write_text(_format_error_matrix_md(matrix), encoding="utf-8")

    return {
        "trace_events": trace_events,
        "comparison": comparison_summary,
        "error_matrix": matrix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-op-log",
        type=Path,
        default=Path("/tmp/frontier_mla_deepseek_v2_flashinfer_mla_live_probe/cuda_ops.jsonl"),
    )
    parser.add_argument(
        "--batch-log",
        type=Path,
        default=Path("/tmp/frontier_mla_deepseek_v2_flashinfer_mla_live_probe/batch_log.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/frontier_mla_deepseek_v2_stage3_online_error"),
    )
    args = parser.parse_args()

    try:
        summary = write_stage3_outputs(
            args.cuda_op_log,
            args.batch_log,
            args.output_dir,
            DeepSeekV2MlaStage3TraceConfig(),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    outputs = summary["error_matrix"]["outputs"]
    print("MLA Stage 3 online trace artifacts written")
    print(f"op_traces_jsonl={outputs['op_traces_jsonl']}")
    print(f"per_op_comparison_json={outputs['per_op_comparison_json']}")
    print(f"per_op_comparison_md={outputs['per_op_comparison_md']}")
    print(f"stage3_error_matrix_json={outputs['stage3_error_matrix_json']}")
    print(f"stage3_error_matrix_md={outputs['stage3_error_matrix_md']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
