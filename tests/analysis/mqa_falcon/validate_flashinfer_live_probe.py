#!/usr/bin/env python3
"""Validate Falcon MQA FlashInfer live-probe CUDA op output."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frontier.attention.families import DENSE_ATTENTION_FAMILY  # noqa: E402
from frontier.attention.profiling_mapping import get_profiling_metric_names  # noqa: E402


REQUIRED_SCOPES = get_profiling_metric_names(DENSE_ATTENTION_FAMILY)
REQUIRED_META_FIELDS = (
    "attention_backend",
    "head_dim",
    "num_q_heads",
    "num_kv_heads",
    "kv_cache_dtype",
    "calculate_kv_scales",
    "attn_module_sliding_window",
    "flashinfer_window_left",
    "kv_cache_spec_type",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_actual_tokens",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"CUDA op log not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSON object at {path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"CUDA op log is empty: {path}")
    return rows


def _require_run_log(run_log: Path, needles: Iterable[str]) -> None:
    if not run_log.exists():
        raise ValueError(f"Run log not found: {run_log}")
    text = run_log.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise ValueError(f"Required runtime contract not found in run log: {missing}")


def _scope_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        op_name = row.get("op_name")
        if isinstance(op_name, str):
            by_scope[op_name].append(row)
    return by_scope


def _validate_scope_presence(by_scope: dict[str, list[dict[str, Any]]]) -> None:
    missing = [scope for scope in REQUIRED_SCOPES if not by_scope.get(scope)]
    if missing:
        raise ValueError(f"Missing required CUDA scopes: {missing}")


def _validate_row_values(
    by_scope: dict[str, list[dict[str, Any]]],
    expected_backend: str,
    expected_head_dim: int,
    expected_q_heads: int,
    expected_kv_heads: int,
) -> None:
    for scope in REQUIRED_SCOPES:
        for row in by_scope[scope]:
            cuda_time_ms = float(row.get("cuda_time_ms", 0.0))
            if cuda_time_ms < 0.0:
                raise ValueError(f"Negative CUDA time for {scope}: {cuda_time_ms}")
            count = int(row.get("count", 0))
            if count <= 0:
                raise ValueError(f"Non-positive count for {scope}: {count}")
            meta = row.get("meta")
            if not isinstance(meta, dict):
                raise ValueError(f"Missing runtime meta for {scope}: {row}")
            missing_fields = [field for field in REQUIRED_META_FIELDS if field not in meta]
            if missing_fields:
                raise ValueError(f"Missing runtime meta fields for {scope}: {missing_fields}")
            if meta["attention_backend"] != expected_backend:
                raise ValueError(
                    f"Unexpected backend for {scope}: {meta['attention_backend']!r}"
                )
            if int(meta["head_dim"]) != expected_head_dim:
                raise ValueError(f"Unexpected head_dim for {scope}: {meta['head_dim']!r}")
            if int(meta["num_q_heads"]) != expected_q_heads:
                raise ValueError(
                    f"Unexpected num_q_heads for {scope}: {meta['num_q_heads']!r}"
                )
            if int(meta["num_kv_heads"]) != expected_kv_heads:
                raise ValueError(
                    f"Unexpected num_kv_heads for {scope}: {meta['num_kv_heads']!r}"
                )
            if scope == "attn_decode" and int(meta["max_seqlen_q"]) != 1:
                raise ValueError(
                    f"Decode scope must have max_seqlen_q=1, got {meta['max_seqlen_q']!r}"
                )
            if scope == "attn_prefill" and int(meta["max_seqlen_q"]) <= 1:
                raise ValueError(
                    f"Prefill scope must have max_seqlen_q>1, got {meta['max_seqlen_q']!r}"
                )


def _print_summary(by_scope: dict[str, list[dict[str, Any]]]) -> None:
    print("FlashInfer MQA live probe validation passed.")
    for scope in REQUIRED_SCOPES:
        rows = by_scope[scope]
        total_ms = sum(float(row.get("cuda_time_ms", 0.0)) for row in rows)
        min_ms = min(float(row.get("cuda_time_ms", 0.0)) for row in rows)
        max_ms = max(float(row.get("cuda_time_ms", 0.0)) for row in rows)
        first_meta = rows[0]["meta"]
        print(
            f"{scope}: rows={len(rows)} total_cuda_time_ms={total_ms:.6f} "
            f"min_cuda_time_ms={min_ms:.6f} max_cuda_time_ms={max_ms:.6f} "
            f"backend={first_meta['attention_backend']} head_dim={first_meta['head_dim']} "
            f"num_q_heads={first_meta['num_q_heads']} num_kv_heads={first_meta['num_kv_heads']} "
            f"kv_cache_dtype={first_meta['kv_cache_dtype']} "
            f"calculate_kv_scales={first_meta['calculate_kv_scales']} "
            f"attn_module_sliding_window={first_meta['attn_module_sliding_window']} "
            f"flashinfer_window_left={first_meta['flashinfer_window_left']} "
            f"kv_cache_spec_type={first_meta['kv_cache_spec_type']} "
            f"max_seqlen_q={first_meta['max_seqlen_q']} "
            f"max_seqlen_k={first_meta['max_seqlen_k']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-op-log", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--expected-backend", type=str, default="FLASHINFER_VLLM_V1")
    parser.add_argument("--expected-head-dim", type=int, default=64)
    parser.add_argument("--expected-q-heads", type=int, default=71)
    parser.add_argument("--expected-kv-heads", type=int, default=1)
    parser.add_argument("--expected-tp", type=int, required=True)
    parser.add_argument(
        "--expected-chunked-prefill-enabled",
        type=str,
        choices=["True", "False"],
        required=True,
    )
    args = parser.parse_args()

    try:
        rows = _load_jsonl(args.cuda_op_log)
        by_scope = _scope_rows(rows)
        _validate_scope_presence(by_scope)
        _validate_row_values(
            by_scope,
            args.expected_backend,
            args.expected_head_dim,
            args.expected_q_heads,
            args.expected_kv_heads,
        )
        _require_run_log(
            args.run_log,
            (
                "Initializing a V1 LLM engine",
                "FlashInfer",
                f"tensor_parallel_size={args.expected_tp}",
                f"chunked_prefill_enabled={args.expected_chunked_prefill_enabled}",
            ),
        )
        _print_summary(by_scope)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
