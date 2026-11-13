#!/usr/bin/env python3
"""Freeze 128 requests of 6144/1024 tokens at one request per second.

The CSV uses TraceReplayRequestGenerator's existing three-column schema.
request_ids.json contains only formal rows; warmup_pool.csv is a separate
sixteen-request template, replayed before each clean repetition. The native
harness must qualify both phases' template IDs with the repetition number.
No model weights, tokenizer, network access, or random sampling is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

NAMESPACE = "fixed-6144-1024-req128-qps1"
NUM_REQUESTS = 128
PROMPT_TOKENS = 6144
OUTPUT_TOKENS = 1024
WARMUP_REQUESTS = 16
REPETITIONS = 3
OUTPUT_FILES = ("trace.csv", "request_ids.json", "warmup_pool.csv", "manifest.json")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output_dir: Path) -> dict:
    rows = [{
        "arrived_at": float(index),
        "frontier_request_id": index,
        "num_decode_tokens": OUTPUT_TOKENS,
        "num_prefill_tokens": PROMPT_TOKENS,
        "request_id": f"{NAMESPACE}-formal-r{index:03d}",
        "role": "formal",
    } for index in range(NUM_REQUESTS)]
    warmup = [{
        "request_id": f"{NAMESPACE}-warmup-r{index:03d}",
        "arrived_at": float(index),
        "num_prefill_tokens": PROMPT_TOKENS,
        "num_decode_tokens": OUTPUT_TOKENS,
    } for index in range(WARMUP_REQUESTS)]

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        fields = ["arrived_at", "num_prefill_tokens", "num_decode_tokens"]
        writer.writerow(fields)
        writer.writerows([row[field] for field in fields] for row in rows)
    (output_dir / "request_ids.json").write_text(json.dumps({
        "formal_request_ids": [row["request_id"] for row in rows],
        "qualification_only": False,
        "rows": rows,
        "sizing_request_ids": [],
        "warmup_request_ids": [row["request_id"] for row in warmup],
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "warmup_pool.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(warmup[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(warmup)

    manifest = {
        "schema": "frontier.canonical_fixed_workload.v1",
        "namespace": NAMESPACE,
        "source": {
            "kind": "synthetic_fixed_token_counts",
            "rule": "k = 1024 tokens; 6k prompt and 1k output per request",
            "sharedgpt": "deferred; historical frozen SharedGPT artifacts are unchanged",
        },
        "selection": {
            "num_formal_requests": NUM_REQUESTS,
            "rule": "all 128 deterministic formal rows, in row order, in every repetition",
        },
        "arrivals": {
            "rule": "arrived_at = row index in seconds; fixed spacing, not Poisson or retimed gaps",
            "target_qps": 1.0,
            "inter_arrival_s": 1.0,
            "first_arrival_s": 0.0,
            "last_arrival_s": 127.0,
            "realized_span_s": 127.0,
            "realized_mean_qps": 1.0,
        },
        "lengths": {
            "prompt_tokens_per_request": PROMPT_TOKENS,
            "output_tokens_per_request": OUTPUT_TOKENS,
            "total_tokens_per_request": PROMPT_TOKENS + OUTPUT_TOKENS,
            "formal_prompt_tokens": NUM_REQUESTS * PROMPT_TOKENS,
            "formal_output_tokens": NUM_REQUESTS * OUTPUT_TOKENS,
            "formal_total_tokens": NUM_REQUESTS * (PROMPT_TOKENS + OUTPUT_TOKENS),
        },
        "frontier_replay": {
            "loader": "TraceReplayRequestGenerator",
            "trace_file": "trace.csv",
            "prefill_scale_factor": 1.0,
            "decode_scale_factor": 1.0,
            "time_scale_factor": 1.0,
            "minimum_max_tokens": PROMPT_TOKENS + OUTPUT_TOKENS,
            "request_id_binding": "frontier_request_id is the row ordinal; the harness records the actual runtime ID mapping",
        },
        "native_request_contract": (
            "One /v1/completions request per row at its phase-relative arrived_at; "
            "construct exactly num_prefill_tokens deterministic valid prompt token IDs from the stable template request_id "
            "and the model vocabulary in the native harness; repeat-qualified transport IDs must not change prompt content. Do not tokenize text, download weights, "
            "truncate, or add BOS/EOS tokens. Use max_tokens = min_tokens = num_decode_tokens and "
            "ignore_eos; runtime evidence must confirm actual prompt and output counts."
        ),
        "repetitions": {
            "clean_repetitions": REPETITIONS,
            "repeat_indices": list(range(1, REPETITIONS + 1)),
            "runtime_request_id_rule": "{request_id}-repeat{repeat:02d}",
            "request_id_binding": "the harness qualifies every formal and warmup template ID with repeat=1,2,3 before dispatch",
            "sequence": "warmup -> both roles idle and request-owned KV released -> formal -> both roles idle and request-owned KV released; repeat three times",
        },
        "warmup": {
            "trace_file": "warmup_pool.csv",
            "num_requests_per_repetition": WARMUP_REQUESTS,
            "prompt_tokens_per_request": PROMPT_TOKENS,
            "output_tokens_per_request": OUTPUT_TOKENS,
            "target_qps": 1.0,
            "inter_arrival_s": 1.0,
            "first_arrival_s": 0.0,
            "last_arrival_s": 15.0,
            "origin": "separate warmup origin before each repetition; not the formal origin",
            "end": "last warmup output plus observed PREFILL/DECODE idle and all request-owned KV released before the first formal request",
            "included_in_metrics": False,
            "coverage_limit": "sixteen aggregate warmup completions do not prove all-lane CUDA Graph replay; runtime evidence must establish coverage on every required lane/rank",
        },
        "measurement": {
            "population": "formal rows only, separately for each clean repetition; never warmup IDs",
            "origin": "first formal request issue after warmup drain; independent clock origin per repetition",
            "arrival_end_s": 127.0,
            "window_start": "first formal client request issue",
            "window_end": "last formal client output token; not last arrival or the later idle/KV-release check",
            "ttft": "per-request client issue to first output token; internal prefill completion is diagnostic only",
            "tpot": "per-request (E2E - TTFT) / (actual output tokens - 1)",
            "e2e": "per-request client issue to last output token; not workload makespan",
            "thpt": "917504 formal prompt-plus-output tokens / the formal measurement window duration in seconds",
            "statistic": "mean for TTFT, TPOT and per-request E2E; p50/p95 diagnostic only",
        },
        "reproducibility": {
            "builder": "tests/integration/canonical_case/build_fixed_manifest.py",
            "builder_sha256": sha256_of(Path(__file__)),
            "seed": None,
            "rule": "fixed integers, ordered rows, UTF-8, LF, sorted JSON keys; no timestamps, absolute paths, or external inputs",
            "command": "python3 tests/integration/canonical_case/build_fixed_manifest.py --output-dir data/processed_traces/canonical_fixed_6144_1024_req128_qps1",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_of(output_dir / name)}  {name}\n" for name in OUTPUT_FILES),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args.output_dir)
    print(json.dumps({"lengths": manifest["lengths"], "arrivals": manifest["arrivals"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
