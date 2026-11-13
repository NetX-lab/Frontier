"""Describe retained enqueue/sleep evidence without repairing timing or admitting costs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_result import iter_jsonl, queue_held_analysis, sha256


def diagnose(records) -> dict:
    """Keep every raw repeat and compare its CPU enqueue window with its GPU budget."""
    counts = Counter()
    unheld = []
    failures = []
    for record in records:
        if record["op"] == "cpu_all_reduce":
            counts["host_cells"] += 1
            continue
        counts["device_cells"] += 1
        quality = queue_held_analysis(record, "device", 0.01)
        counts["queue_" + quality["queue_gate"]] += 1
        identity = {key: record[key] for key in ("pass", "scenario", "call")}
        if quality["queue_gate"] != "PASS":
            failures.append({**identity, **quality})
        for rank in record["participating_ranks"]:
            sleep = record["per_rank_queue_sleep_us"][rank]
            launch_max = max(record["per_rank_launch_enqueue_us"][rank])
            for repeat, held in enumerate(record["per_rank_queue_held"][rank]):
                enqueue = record["per_rank_enqueue_us"][rank][repeat]
                counts["held_rank_repeats" if held else "unheld_rank_repeats"] += 1
                counts[("held" if held else "unheld") +
                       ("_enqueue_over_sleep" if enqueue > sleep else "_enqueue_within_sleep")] += 1
                if not held:
                    unheld.append({**identity, "rank": rank, "repeat": repeat,
                                   "enqueue_us": enqueue, "sleep_us": sleep,
                                   "launch_max_enqueue_us": launch_max,
                                   "enqueue_over_sleep": enqueue / sleep})
    return {"counts": dict(counts), "unheld_rank_repeats": unheld,
            "queue_failures": failures, "queue_threshold": 0.01,
            "threshold_comparison": "strictly less than",
            "advisory_only": True, "timing_changed": False, "cost_data_qualified": False,
            "cause": "UNPROVEN: CPU enqueue overruns are observed associations, not proof of GPU sleep duration, query delay or cross-rank start alignment.",
            "next_action": "Retain the timing path and threshold. Collect fresh native H200 queue, group, rail and transport evidence; resolve any failed cells before formal cost admission."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.directory / "collectives/measurements.jsonl"
    if not path.exists():
        path = path.with_suffix(".jsonl.gz")
    before = sha256(path)
    result = diagnose(iter_jsonl(path))
    result.update(input_path=str(path.resolve()), input_sha256=before,
                  input_unchanged=sha256(path) == before,
                  producer_sha256=sha256(Path(__file__)),
                  analyzer_sha256=sha256(Path(__file__).with_name("analyze_result.py")))
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "counts": result["counts"],
                      "cause": result["cause"], "input_unchanged": result["input_unchanged"]}))


if __name__ == "__main__":
    main()
