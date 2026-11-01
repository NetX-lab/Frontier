"""Retain the client harness reading beside a mechanism-evidenced warm reading.

Usage: e2e_warm_reading.py --rows ROWS --reading JSON --openers JSON --output NEW_JSON
Openers map every native repeat side to normalized formal request IDs. This tool
does not identify warmup events or choose exclusions from request latency.
"""

import argparse
import hashlib
import json
from pathlib import Path

from tests.comparison.e2e_gate.gate import (
    admission, client_reading, metric_row, metric_values, read_rows, verdict,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(rows_dir, reading_path, openers_path):
    raw = reading_path.read_text()
    harness = json.loads(raw)
    openers = json.loads(openers_path.read_text())
    sample, sides = read_rows(rows_dir)
    require(harness["gate"] == "client" and sample["cell_id"] == harness["cell_id"], "Not the bound client cell")
    native = [s for s in sides if s.startswith("vllm/")]
    [member] = [s for s in sides if s.startswith("frontier/")]
    require(set(openers) == set(native) == set(harness["repetitions"]), "Repeat evidence keys differ")
    for side, ids in openers.items():
        require(isinstance(ids, list) and len(ids) == len(set(ids)), f"Duplicate/invalid openers: {side}")
        require(set(ids) <= set(sample["formal_request_ids"]), f"Non-formal openers: {side}")
    checks = admission(sample, sides, harness["routing_status"])
    original = client_reading({"gate": "client"}, sample, sides, checks)
    require(not checks["blocked_status"] and not checks["client_metric_gaps"], "Unadmitted original rows")
    require(original["metrics"] == harness["metrics"] and original["repetitions"] == harness["repetitions"]
            and original["formal_windows"] == harness["formal_windows"]
            and original["verdict"] == harness["verdict"], "Original reading not reproduced exactly")
    metrics = [r["metric"] for r in original["metrics"]]
    count = len(sample["formal_request_ids"])
    pairs, repeats = {}, {}
    for side in native:
        kept = [rid for rid in sample["formal_request_ids"] if rid not in openers[side]]
        require(kept, f"Empty warm population: {side}")
        pairs[side] = {name: metric_values(sides[name].loc[~sides[name].index.isin(openers[side])])
                       for name in (side, member)}
        rows = [metric_row(m, member, pairs[side][side][m], pairs[side][member][m], count, None)
                for m in metrics]
        repeats[side] = dict(metrics=rows, verdict=verdict([r["status"] for r in rows]),
                             excluded_request_ids=openers[side], retained_request_count=len(kept))
    pooled = {}
    for name in ("vllm", "frontier"):
        pooled[name] = {m: tuple(map(sum, zip(*(pair[side if name == "vllm" else member][m]
                           for side, pair in pairs.items())))) for m in metrics}
    rows = [metric_row(m, member, pooled["vllm"][m], pooled["frontier"][m], count * len(native), None)
            for m in metrics]
    warm = dict(metrics=rows, repetitions=repeats, formal_request_repeat_count=count * len(native),
                retained_request_repeat_count=sum(r["retained_request_count"] for r in repeats.values()),
                verdict=verdict([verdict([r["status"] for r in rows]), *(r["verdict"] for r in repeats.values())]))
    # Identical pools retain the harness's ratio rounding, not a second rounding
    # after multiplying both terms by the repeat count. Totals still sum both sides.
    if not any(openers.values()):
        value_fields = ("vllm", "frontier", "absolute_error", "relative_error", "status")
        for row, source in zip(warm["metrics"], original["metrics"]):
            row.update({field: source[field] for field in value_fields})
        for side in native:
            warm["repetitions"][side].update(original["repetitions"][side])
        require(all(row[field] == source[field] for row, source in zip(warm["metrics"], harness["metrics"])
                    for field in value_fields) and all(
                warm["repetitions"][s]["metrics"] == harness["repetitions"][s]["metrics"] for s in native),
                "Empty-set reproduction failed")
    sources = [rows_dir / "sample.json", reading_path, openers_path, Path(__file__),
               *(rows_dir / f"{side}.csv" for side in sides)]
    return dict(decision="D-WARM-TARGET", classification="RETAINED_CLEAN_REFERENCE_WARM_READING",
                all_request=harness, all_request_verbatim=raw, reading_sha256=digest(reading_path),
                warm=warm, openers=openers, empty_set_reproduction="PASS_EXACT",
                method="Paired exclusion per repeat; PD-POOL sums latency and rate numerators/denominators on both sides. "
                       "Empty exclusions preserve the original ratio rounding with summed totals. No counterfactual replay.",
                sources=[dict(path=str(p.resolve()), sha256=digest(p)) for p in sources])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rows", "reading", "openers", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    document = collect(args.rows, args.reading, args.openers)
    with args.output.open("x") as stream:
        stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(f"PASS_EXACT original; warm={document['warm']['verdict']} retained={document['warm']['retained_request_repeat_count']}")


if __name__ == "__main__":
    main()
