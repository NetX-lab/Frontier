#!/usr/bin/env python3
"""Freeze the canonical SharedGPT calibration workload from the pinned upstream trace.

The source is ``dataset/sharegpt_req300_rate10_llama.jsonl`` from
``casys-kaist/LLMServingSim`` at tag ``v1.0.0``. The script verifies its SHA-256,
takes the first ``--num-requests`` requests in trace order, and retimes their
arrivals to ``--qps``: the trace's own inter-arrival gaps are multiplied by one
constant so that the selected requests arrive at exactly the target mean rate.
The relative arrival pattern of the trace is kept, and no random seed is involved.

Outputs, all consumed by existing readers:

``trace.csv``
    ``arrived_at, num_prefill_tokens, num_decode_tokens`` for Frontier's
    ``TraceReplayRequestGenerator``.
``request_ids.json``
    The row list read by ``tests/comparison/dp_placement_pp/vllm_replay.py`` and
    the native P/D harness.
``warmup_pool.csv``
    The remaining trace requests, disjoint from the formal set, for native warmups.
``manifest.json``
    Source identity, selection and retiming rules, and the length-distribution
    report of the selected set against the full trace.
``SHA256SUMS``
    Checksums of the four files above.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

SOURCE = {
    "repository": "https://github.com/casys-kaist/LLMServingSim",
    "tag": "v1.0.0",
    "commit": "04bdeb8b6fb3609e8c6ce561fba18ef1b0e37479",
    "path": "dataset/sharegpt_req300_rate10_llama.jsonl",
    "url": "https://raw.githubusercontent.com/casys-kaist/LLMServingSim/v1.0.0/dataset/sharegpt_req300_rate10_llama.jsonl",
    "sha256": "fad744ff3b25bff1e164cc210af4f29176e8d57d1be14c4f981a10bc32ddd61a",
    "license": "MIT",
    "tokenizer": "Llama (per the upstream dataset README)",
}
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
OUTPUT_FILES = ("trace.csv", "request_ids.json", "warmup_pool.csv", "manifest.json")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[int]) -> dict:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
        "quantiles": {f"p{round(q * 100)}": round(quantile(values, q), 3) for q in QUANTILES},
    }


def ks_statistic(sample: list[int], reference: list[int]) -> float:
    points = sorted(set(sample) | set(reference))
    sample_sorted, reference_sorted = sorted(sample), sorted(reference)

    def cdf(ordered: list[int], x: int) -> float:
        low, high = 0, len(ordered)
        while low < high:
            middle = (low + high) // 2
            if ordered[middle] <= x:
                low = middle + 1
            else:
                high = middle
        return low / len(ordered)

    return round(max(abs(cdf(sample_sorted, x) - cdf(reference_sorted, x)) for x in points), 4)


def length_report(rows: list[dict], full: list[dict]) -> dict:
    report = {}
    for field, label in (("input_toks", "prompt_tokens"), ("output_toks", "output_tokens")):
        selected = [row[field] for row in rows]
        everything = [row[field] for row in full]
        report[label] = {
            "selected": describe(selected),
            "full_trace": describe(everything),
            "relative_mean_deviation": round(
                statistics.mean(selected) / statistics.mean(everything) - 1.0, 4
            ),
            "ks_statistic_vs_full_trace": ks_statistic(selected, everything),
        }
    return report


def build(source: Path, output_dir: Path, num_requests: int, qps: float, namespace: str) -> dict:
    digest = sha256_of(source)
    if digest != SOURCE["sha256"]:
        raise ValueError(f"{source} has SHA-256 {digest}, expected {SOURCE['sha256']}")
    full = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if num_requests >= len(full):
        raise ValueError(f"--num-requests {num_requests} leaves no warmup pool in {len(full)} requests")
    arrivals_s = [row["arrival_time_ns"] / 1e9 for row in full]
    if any(later < earlier for earlier, later in zip(arrivals_s, arrivals_s[1:])):
        raise ValueError("source arrivals are not in trace order")

    selected, warmup = full[:num_requests], full[num_requests:]
    origin_s = arrivals_s[0]
    source_span_s = arrivals_s[num_requests - 1] - origin_s
    target_span_s = (num_requests - 1) / qps
    scale = target_span_s / source_span_s

    request_rows = []
    for index, row in enumerate(selected):
        request_rows.append({
            "arrived_at": round((arrivals_s[index] - origin_s) * scale, 6),
            "frontier_request_id": index,
            "num_decode_tokens": row["output_toks"],
            "num_prefill_tokens": row["input_toks"],
            "request_id": f"{namespace}-r{index:03d}",
            "role": "formal",
            "source_index": index,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        for row in request_rows:
            writer.writerow([row["arrived_at"], row["num_prefill_tokens"], row["num_decode_tokens"]])
    (output_dir / "request_ids.json").write_text(json.dumps({
        "formal_request_ids": [row["request_id"] for row in request_rows],
        "qualification_only": False,
        "rows": request_rows,
        "sizing_request_ids": [],
        "warmup_request_ids": [],
    }, indent=1, sort_keys=True) + "\n")
    with (output_dir / "warmup_pool.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["request_id", "source_index", "num_prefill_tokens", "num_decode_tokens"])
        for offset, row in enumerate(warmup):
            index = num_requests + offset
            writer.writerow([f"{namespace}-w{index:03d}", index, row["input_toks"], row["output_toks"]])

    realized_span_s = request_rows[-1]["arrived_at"]
    gaps = [later["arrived_at"] - earlier["arrived_at"] for earlier, later in zip(request_rows, request_rows[1:])]
    manifest = {
        "schema": "frontier.canonical_sharegpt_workload.v1",
        "namespace": namespace,
        "source": SOURCE,
        "selection": {
            "rule": f"first {num_requests} requests in trace order",
            "num_formal_requests": num_requests,
            "source_indices": [0, num_requests - 1],
            "warmup_pool": f"source indices {num_requests}..{len(full) - 1} ({len(warmup)} requests), disjoint from the formal set",
        },
        "arrivals": {
            "rule": (
                "arrived_at = (t_i - t_0) * scale, with t_i the source arrival_time_ns in seconds; "
                "scale makes the formal requests arrive at exactly the target mean rate "
                "(num_formal_requests - 1) / span; the raw arrival_time_ns is not replayed"
            ),
            "target_qps": qps,
            "source_mean_qps_formal_set": round((num_requests - 1) / source_span_s, 4),
            "source_mean_qps_full_trace": round((len(full) - 1) / (arrivals_s[-1] - origin_s), 4),
            "scale": round(scale, 9),
            "realized_span_s": realized_span_s,
            "realized_mean_qps": round((num_requests - 1) / realized_span_s, 6),
            "inter_arrival_cv": round(statistics.pstdev(gaps) / statistics.mean(gaps), 4),
            "seed": None,
        },
        "lengths": {
            "source": "token counts from the trace (input_toks, output_toks); the trace carries no text, so lengths are not re-tokenized and are identical for every model",
            "edge_cases": {
                "prompt_tokens_eq_1": sum(1 for row in selected if row["input_toks"] == 1),
                "prompt_tokens_lt_16": sum(1 for row in selected if row["input_toks"] < 16),
                "output_tokens_eq_1": sum(1 for row in selected if row["output_toks"] == 1),
                "max_total_tokens": max(row["input_toks"] + row["output_toks"] for row in selected),
            },
            "distribution": length_report(selected, full),
            "head_prefix_means": {
                f"head{size}": {
                    "prompt_tokens": round(statistics.mean(row["input_toks"] for row in full[:size]), 3),
                    "output_tokens": round(statistics.mean(row["output_toks"] for row in full[:size]), 3),
                }
                for size in (64, 128, 256, len(full))
            },
        },
        "native_request_contract": (
            "one /v1/completions request per row at its arrived_at offset, deterministic prompt token ids "
            "derived from request_id, max_tokens = min_tokens = num_decode_tokens, ignore_eos"
        ),
        "metric_notes": {
            "tpot_population": "requests with num_decode_tokens == 1 have no inter-token interval and are excluded from the TPOT mean only; they stay in TTFT, THPT and E2E",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_of(output_dir / name)}  {name}\n" for name in OUTPUT_FILES)
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="local copy of the pinned upstream trace")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--qps", type=float, required=True)
    parser.add_argument("--namespace", required=True, help="request-id prefix")
    args = parser.parse_args(argv)
    manifest = build(args.source, args.output_dir, args.num_requests, args.qps, args.namespace)
    print(json.dumps({
        "realized_mean_qps": manifest["arrivals"]["realized_mean_qps"],
        "realized_span_s": manifest["arrivals"]["realized_span_s"],
        "scale": manifest["arrivals"]["scale"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
