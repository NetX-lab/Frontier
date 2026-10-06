#!/usr/bin/env python3
"""Build the request trace of a DP-placement calibration case.

The trace is the single workload source for both systems (plan §18.6): Frontier
reads the CSV directly and the vLLM replay client posts one completion per row
at the row's `arrived_at` offset. Every value comes from the case's workload
file, so a retuned case is a new input file, not a code change.

Segments, in time order:

``warmup``
    Evenly spaced requests. Excluded from every comparison. A workload lists
    either one group of them or several named `groups`, each with its own
    lengths and spacing, the first at time zero and every later one
    `gap_before_s` after the previous group's last request; a named group's
    request ids carry its name.
``burst`` (optional)
    Requests that arrive after an idle gap, in the order the workload lists
    their prompt kinds and `spacing_s` apart (together when it is absent),
    followed, when the burst names `probe_offset_s`, by one probe at that
    offset from the first of them. The probe's placement is the T2 witness,
    and its row is marked `probe`. A workload lists either one `burst` or
    several named `bursts`, each after its own idle gap (the burst's
    `idle_gap_s` where it names one, else the workload's); a named burst's
    request ids carry its name.
``steady`` (optional, after a burst)
    Staggered arrivals whose prompt and decode lengths cycle through the listed
    values. They supply the causal-join rows of T1.
``poisson`` (optional)
    Open-loop arrivals at `qps` requests per second, `gap_before_s` after the
    last earlier request: the first arrives at once and each later one after
    an exponential interval drawn from a generator seeded with `seed`, as a
    serving benchmark sends them. Every request has the listed lengths; a
    length given as an inclusive `[low, high]` range is drawn uniformly per
    request, prompt before decode, from a second generator seeded with
    `length_seed`, so the arrival times match the fixed-length workload with
    the same `seed`.
``sizing`` (optional)
    Isolated single prompts of increasing length, used only to measure the
    engine's iteration time before a pipeline-parallel run is sized.

Outputs: ``trace.csv`` (`arrived_at,num_prefill_tokens,num_decode_tokens`,
Frontier's trace format, one row per request in arrival order) and
``request_ids.json``, which maps each row to its request id, role and segment.
Frontier numbers trace rows from zero in file order, so the row index is the
join to its request ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from itertools import cycle, islice
from pathlib import Path


def build_rows(workload: dict) -> list[dict]:
    namespace = workload["request_id_namespace"]
    rows: list[dict] = []

    def add(segment: str, role: str, name: str, arrived_at: float, prefill: int, decode: int):
        rows.append({
            "request_id": f"{namespace}-{name}",
            "segment": segment,
            "role": role,
            "arrived_at": round(arrived_at, 6),
            "num_prefill_tokens": prefill,
            "num_decode_tokens": decode,
        })

    warmups = workload["warmups"]
    groups = warmups["groups"] if "groups" in warmups else [{"name": "", **warmups}]
    for group in groups:
        group_start = rows[-1]["arrived_at"] + group["gap_before_s"] if rows else 0.0
        prefix = f"{group['name']}-" if group["name"] else ""
        for index in range(group["count"]):
            add("warmup", "warmup", f"{prefix}w{index}", group_start + group["interval_s"] * index,
                group["num_prefill_tokens"], group["num_decode_tokens"])
    last_arrival = rows[-1]["arrived_at"] if rows else 0.0

    if "bursts" in workload:
        bursts = workload["bursts"]
    elif "burst" in workload:
        bursts = [{"name": "", **workload["burst"]}]
    else:
        bursts = []
    for burst in bursts:
        burst_start = last_arrival + burst.get("idle_gap_s", workload["idle_gap_s"])
        prefix = f"{burst['name']}-" if burst["name"] else ""
        first_row = len(rows)
        for index, kind in enumerate(burst["order"], start=1):
            add("burst", "formal", f"{prefix}b{index}",
                burst_start + burst.get("spacing_s", 0.0) * (index - 1),
                burst[f"{kind}_prefill_tokens"], burst["num_decode_tokens"])
        if "probe_offset_s" in burst:
            add("burst", "formal", f"{prefix}b{len(burst['order']) + 1}",
                burst_start + burst["probe_offset_s"],
                burst["probe_prefill_tokens"], burst["probe_decode_tokens"])
        for row in rows[first_row:]:
            row["burst"] = burst["name"]
            row["probe"] = False
        if "probe_offset_s" in burst:
            rows[-1]["probe"] = True
        last_arrival = rows[-1]["arrived_at"]

    steady = workload.get("steady")
    if steady is not None:
        steady_start = burst_start + steady["gap_after_burst_s"]
        lengths = zip(cycle(steady["num_prefill_tokens"]), cycle(steady["num_decode_tokens"]))
        for index, (prefill, decode) in enumerate(islice(lengths, steady["count"])):
            add("steady", "formal", f"s{index:02d}", steady_start + steady["interval_s"] * index,
                prefill, decode)

    poisson = workload.get("poisson")
    if poisson is not None:
        intervals = random.Random(poisson["seed"])
        prefill, decode = poisson["num_prefill_tokens"], poisson["num_decode_tokens"]
        lengths = random.Random(poisson["length_seed"]) if isinstance(prefill, list) or isinstance(decode, list) else None

        def draw(length):
            return lengths.randint(*length) if isinstance(length, list) else length

        arrived_at = rows[-1]["arrived_at"] + poisson["gap_before_s"]
        for index in range(poisson["count"]):
            add("poisson", "formal", f"p{index:03d}", arrived_at, draw(prefill), draw(decode))
            arrived_at += intervals.expovariate(poisson["qps"])

    sizing = workload.get("sizing")
    if sizing is not None:
        sizing_start = rows[-1]["arrived_at"] + sizing["gap_before_s"]
        for index, prefill in enumerate(sizing["num_prefill_tokens"]):
            add("sizing", "sizing", f"z{index}", sizing_start + sizing["interval_s"] * index,
                prefill, sizing["num_decode_tokens"])

    arrivals = [row["arrived_at"] for row in rows]
    if arrivals != sorted(arrivals):
        raise ValueError("workload segments overlap: arrivals are not in time order")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    workload = json.loads(args.workload.read_text())
    rows = build_rows(workload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        for row in rows:
            writer.writerow([row["arrived_at"], row["num_prefill_tokens"], row["num_decode_tokens"]])
    (args.output_dir / "request_ids.json").write_text(json.dumps({
        "workload": str(args.workload),
        "request_id_namespace": workload["request_id_namespace"],
        "request_id_encoding": (
            "x-request-id header <request_id>; vLLM completions name the engine "
            "request cmpl-<request_id>-0; Frontier numbers trace rows from 0"
        ),
        "warmup_request_ids": [row["request_id"] for row in rows if row["role"] == "warmup"],
        "formal_request_ids": [row["request_id"] for row in rows if row["role"] == "formal"],
        "sizing_request_ids": [row["request_id"] for row in rows if row["role"] == "sizing"],
        "rows": [{"frontier_request_id": index, **row} for index, row in enumerate(rows)],
    }, indent=1))
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
