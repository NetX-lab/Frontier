"""Explain Frontier's first D step (KV transfer end -> first output token) with the step-cost table.

Usage: first_d_step.py <task directory>

For each case and DECODE depth, the first D step of each request is
first_decode_token_completed_at - transfer_kv_cache_request_end_ts, and its ratio to the
request's own TPOT is printed. The expected value comes from the DECODE entries of the case's
step-cost table, which price the k=1 step (the 1-token prefill after the KV load) from the
profile run's traced steps: at depth 1 the k=1 step starts at the step boundary that reports
the READ; at depth 2 it waits for the batch already in flight, one decode-only step.
Times are in ms.
"""
import json
import sys
from pathlib import Path

import pandas as pd

CASES = (
    # name, depth-1 run, depth-2 run
    ("native06", "h4a_p0_diagnostic/results/run_engine_step", "h4a_p0_diagnostic/results/run_engine_step_depth2"),
    ("p2_32", "h200_mainline/diagnostic/results/p2_32e", "h200_mainline/diagnostic/results/p2_32e2"),
    ("p2_64", "h200_mainline/diagnostic/results/p2_64e", "h200_mainline/diagnostic/results/p2_64e2"),
)


def first_steps(run):
    metrics = pd.read_csv(run / "request_metrics.csv").set_index("Request Id")
    rows = []
    for line in (run / "metrics_ground_truth.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["event_type"] != "request_completion":
            continue
        first = record["first_decode_token_completed_at"] * 1e3
        tpot = (record["completed_at"] * 1e3 - first) / (record["num_decode_tokens"] - 1)
        step = first - metrics.loc[record["request_id"], "transfer_kv_cache_request_end_ts"]
        rows.append({"first_step": step, "ratio": step / tpot})
    return pd.DataFrame(rows)


def decode_entries(run):
    entries = json.loads((run / "h4a5_attribution.json").read_text())["step_table"]["DECODE"]
    k1_alone = next(e for e in entries if e["num_prefill_tokens"] == 1 and e["num_decode_requests"]["max"] == 0)
    k1_mixed = next(e for e in entries if e["num_prefill_tokens"] == 1 and e["num_decode_requests"]["max"] > 0)
    decode_only = next(e for e in entries if e["num_prefill_requests"] == 0)
    return k1_alone["time_ms"], k1_mixed["time_ms"], decode_only["time_ms"]


def main() -> None:
    root = Path(sys.argv[1])
    for name, depth1, depth2 in CASES:
        k1_alone, k1_mixed, decode_only = decode_entries(root / depth1)
        print(f"== {name}: table k=1 alone {k1_alone:.2f}  k=1 with decodes {k1_mixed:.2f}  decode-only {decode_only:.2f}")
        for tag, run, expected in (("depth 1", depth1, k1_mixed), ("depth 2", depth2, decode_only + k1_mixed)):
            frame = first_steps(root / run)
            print(f"  {tag}: n {len(frame)}  first step mean {frame.first_step.mean():.2f}  "
                  f"p50 {frame.first_step.median():.2f}  min {frame.first_step.min():.2f}  "
                  f"max {frame.first_step.max():.2f}  expected {expected:.2f}  "
                  f"ratio to own TPOT p50 {frame.ratio.median():.3f}")


if __name__ == "__main__":
    main()
