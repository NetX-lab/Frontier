"""Supplementary DIAGNOSTIC reading of a client-gate result against a subset of the native repetitions.

Usage: repetition_reading.py <repetitions, e.g. 2,3> <name>=<gate reading.json> ...

D-H4B-K1-REPRICE, next action 4: the gate verdict stays the one over all repetitions. This reading
pools the native numerators and denominators of the chosen repetitions, as the gate pools all of
them, and compares the Frontier value with the same relative-error limit. It separates the
first-contact READs of P2-64 repetition 1 from the rest of the target; it is not a gate verdict.
"""
import json
import sys
from pathlib import Path


def pooled(reading, repeats):
    selected = [value for key, value in reading["repetitions"].items() if int(key[-2:]) in repeats]
    if len(selected) != len(repeats):
        raise SystemExit(f"repetitions {repeats} not all in {sorted(reading['repetitions'])}")
    rows = []
    for index, metric in enumerate(reading["metrics"]):
        parts = [repetition["metrics"][index] for repetition in selected]
        if any(part["metric"] != metric["metric"] for part in parts):
            raise SystemExit(f"metric order differs between repetitions: {metric['metric']}")
        native = sum(p["vllm_numerator"] for p in parts) / sum(p["vllm_denominator"] for p in parts)
        error = abs(metric["frontier"] - native) / native
        rows.append((metric["metric"], native, metric["frontier"], metric["frontier"] - native, error,
                     "PASS" if error <= reading["relative_error_limit"] else "FAIL",
                     [p["status"] for p in parts]))
    return rows


def main(repeats, readings):
    for argument in readings:
        name, _, path = argument.partition("=")
        reading = json.loads(Path(path).read_text())
        print(f"== {name}: gate verdict over all repetitions {reading['verdict']}; "
              f"supplementary reading over repetitions {repeats}")
        for metric, native, frontier, absolute, error, status, statuses in pooled(reading, repeats):
            print(f"  {metric:22s} native {native:12.3f}  frontier {frontier:12.3f}  error {absolute:+10.3f} "
                  f"({error:6.2%})  {status}  per repetition {statuses}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or not all("=" in a for a in sys.argv[2:]):
        raise SystemExit(__doc__)
    main([int(r) for r in sys.argv[1].split(",")], sys.argv[2:])
