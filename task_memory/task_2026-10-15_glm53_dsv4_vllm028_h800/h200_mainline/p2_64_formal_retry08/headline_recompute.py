#!/usr/bin/env python3
"""Recompute the four P2-64 generation08 headline metrics from the committed gate row CSVs.

The formal report quotes numbers produced by `compare.py`. This script reproduces them from the
row-level inputs only, so the acceptance verdict does not rest on the comparison tool's own summary.

Pooling follows the comparison contract. The three latency metrics pool as the mean of the
per-repeat means, which is also their overall mean because every repeat carries 128 requests.
Throughput pools as total tokens over total span, summed across repeats. Concatenating the repeats
into a single span would instead cover the idle gaps between them and understate throughput; the
per-repeat mean is reported alongside because it agrees to 0.0002% here.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

ROWS = (
    Path(__file__).resolve().parents[2]
    / "h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2/gate/rows"
)
RUN_TAG = "dsv4-h200-p2-64-formal-20261107-08"
REPEATS = ("repeat01", "repeat02", "repeat03")
METRICS = ("ttft", "tpot", "e2e", "thpt")
ERROR_LIMIT = 0.10


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, float]:
    tokens = sum(
        int(row["request_num_prefill_tokens"]) + int(row["request_num_decode_tokens"])
        for row in rows
    )
    span = max(float(row["completion_s"]) for row in rows) - min(
        float(row["arrival_s"]) for row in rows
    )
    return {
        "requests": len(rows),
        "ttft": statistics.fmean(float(row["ttft_ms"]) for row in rows),
        # The comparison contract excludes single-decode-token requests from TPOT.
        "tpot": statistics.fmean(
            float(row["tpot_ms"]) for row in rows if int(row["request_num_decode_tokens"]) > 1
        ),
        "e2e": statistics.fmean(float(row["request_e2e_time_ms"]) for row in rows),
        "thpt": tokens / span,
        "tokens": tokens,
        "span_s": span,
    }


def relative_errors(frontier: dict[str, float], vllm: dict[str, float]) -> dict[str, float]:
    return {name: abs(frontier[name] - vllm[name]) / abs(vllm[name]) for name in METRICS}


def main() -> int:
    frontier = summarize(read_rows(ROWS / "frontier/a10.csv"))
    per_repeat = {
        repeat: summarize(read_rows(ROWS / f"vllm/{RUN_TAG}-{repeat}.csv")) for repeat in REPEATS
    }
    pooled = {
        name: statistics.fmean(per_repeat[repeat][name] for repeat in REPEATS) for name in METRICS
    }
    pooled_thpt_mean_of_repeats = pooled["thpt"]
    pooled["thpt"] = sum(per_repeat[repeat]["tokens"] for repeat in REPEATS) / sum(
        per_repeat[repeat]["span_s"] for repeat in REPEATS
    )

    report = {
        "rows_root": str(ROWS),
        "error_limit": ERROR_LIMIT,
        "frontier": frontier,
        "vllm_per_repeat": per_repeat,
        "vllm_pooled": pooled,
        "vllm_pooled_thpt_mean_of_repeats": pooled_thpt_mean_of_repeats,
        "errors_per_repeat": {
            repeat: relative_errors(frontier, per_repeat[repeat]) for repeat in REPEATS
        },
        "errors_pooled": relative_errors(frontier, pooled),
    }
    worst = max(
        (value, f"{repeat}/{name}")
        for repeat, errors in report["errors_per_repeat"].items()
        for name, value in errors.items()
    )
    report["worst_per_repeat_error"] = {"value": worst[0], "where": worst[1]}
    report["verdict"] = "PASS" if worst[0] <= ERROR_LIMIT else "FAIL"

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    text = json.dumps(report, indent=1, sort_keys=True) + "\n"
    if out is not None:
        out.write_text(text)
    print(text, end="")
    print(f"VERDICT={report['verdict']} WORST={worst[0] * 100:.2f}% AT={worst[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
