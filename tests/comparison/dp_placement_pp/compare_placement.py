#!/usr/bin/env python3
"""Compare the vLLM DP placement chain with Frontier's (plan §18.5 G5, §18.6).

Inputs are the extractor's ``chain.json`` for one vLLM run and the
``run_frontier_case.py`` output directory for the same trace.

T1 replays the native history through Frontier's coordinator and frontend
model. Every coordinator receipt is fed to `VllmDPLoadBalancer.report` at its
recorded time, and every frontend route to `select`. Engine keys
`(wave, step)` become their rank, which keeps their order and equality. Each
native route is then compared with the replayed choice and with the counts the
replay saw. With the inputs matched, a difference can only come from count
calculation, key grouping, snapshot publication or frontend selection. When
the counts differ the row is labeled `snapshot publication` and carries both
publish times, because the native coordinator's clock also advances on
messages Frontier does not model and the publish-to-apply delay is not
modeled (semantic row S30).

T2 is the placement of each burst and its probe. A burst's row is accepted
only when the native trace shows the premise of plan §18.1 C4, that the
intended snapshot was applied at the frontend before the probe was routed:

- the burst was routed in trace order, all from one snapshot;
- every engine's count in the probe's snapshot came from an iteration of the
  burst that scheduled without applying an output;
- no engine applied an output between the burst's first route and the probe's.

Otherwise the burst is `SCENARIO_NOT_REACHED` with the failed checks named.
A qualified burst also records whether the completion-reporting control
placed the probe differently from the reference, which is what makes the
slice discriminating.

Outputs in ``--output-dir``: ``t1_replay_table.csv``, ``t2_placement_table.csv``,
``workflow_gap_table.csv`` and ``comparison_status.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from frontier.scheduler.request_load import RequestLoad  # noqa: E402
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer  # noqa: E402

FRONTIER_RUNS = ("vllm_load_balancing", "completion_reporting_control", "round_robin")


def replay_native_history(chain: dict, num_engines: int) -> list[dict]:
    """Feed native receipts and routes through Frontier's balancer, in time order."""

    receipts = chain["coordinator_receipts"]
    keys = sorted({(receipt["wave"], receipt["step"]) for receipt in receipts})
    key_rank = {key: rank for rank, key in enumerate(keys)}
    publish_time = {snapshot["snapshot"]: snapshot["publish_monotonic"] for snapshot in chain["snapshots"]}
    events = [(receipt["monotonic"], 0, receipt) for receipt in receipts] + [
        (route["route_monotonic"], 1, route) for route in chain["placement"]
    ]
    events.sort(key=lambda event: (event[0], event[1]))
    origin = events[0][0]
    balancer = VllmDPLoadBalancer(num_engines)
    rows = []
    for time, kind, record in events:
        if kind == 0:
            balancer.report(
                time - origin, record["engine"], key_rank[(record["wave"], record["step"])],
                RequestLoad(record["waiting"], record["running"]),
            )
            continue
        engine = balancer.select(time - origin)
        # `select` reserved against the chosen engine; undo it to show the
        # counts the choice read.
        seen = [[load.waiting, load.running] for load in balancer.frontend_counts]
        seen[engine][0] -= 1
        if engine == record["engine"] and seen == record["counts_used"]:
            status, cause = "MATCH", ""
        elif seen == record["counts_used"]:
            status, cause = "MISMATCH", "frontend selection"
        else:
            status, cause = "MISMATCH", "snapshot publication"
        native_publish = publish_time.get(record["snapshot"])
        rows.append({
            "request_id": record["request_id"],
            "segment": record["segment"],
            "time_s": round(time - origin, 6),
            "native_engine": record["engine"],
            "native_snapshot": record["snapshot"],
            "native_snapshot_publish_s": None if native_publish is None else round(native_publish - origin, 6),
            "native_counts": record["counts_used"],
            "frontier_engine": engine,
            "frontier_last_publish_s": balancer.last_publish_ms / 1000,
            "frontier_counts": seen,
            "status": status,
            "first_cause": cause,
        })
    return rows


def qualify_burst(chain: dict, burst: list[dict]) -> dict:
    placement = {row["request_id"]: row for row in chain["placement"]}
    routes = [placement[row["request_id"]] for row in burst]
    probe = routes[-1]
    burst_start = min(route["route_monotonic"] for route in routes[:-1])
    route_order = [row["request_id"] for row in sorted(routes, key=lambda row: row["route_seq"])]
    snapshots = {snapshot["snapshot"]: snapshot for snapshot in chain["snapshots"]}
    iterations = {(record["engine"], record["seq"]): record for record in chain["engine_iterations"]}
    probe_snapshot = snapshots.get(probe["snapshot"])
    provenance = probe_snapshot["provenance"] if probe_snapshot else []

    def scheduled_in_burst(source: dict | None) -> bool:
        if source is None:
            return False
        iteration = iterations[(source["engine"], source["iteration_seq"])]
        return iteration["monotonic"] > burst_start and iteration["branch"] == "scheduled_without_applying"

    applied = [
        record["monotonic"] for record in chain["engine_iterations"]
        if record["applied_output"] and record["monotonic"] > burst_start
    ]
    first_applied = min(applied) if applied else None
    checks = {
        "burst_routed_in_trace_order": route_order == [row["request_id"] for row in burst],
        "burst_routed_from_one_snapshot": len({route["snapshot"] for route in routes[:-1]}) == 1,
        "probe_snapshot_scheduled_in_burst_on_every_engine": bool(provenance)
        and all(scheduled_in_burst(source) for source in provenance),
        "no_output_applied_before_probe": first_applied is None or first_applied > probe["route_monotonic"],
    }
    return {
        "checks": checks,
        "qualified": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "route_order": route_order,
        "probe_request_id": probe["request_id"],
        "probe_snapshot": probe["snapshot"],
        "probe_snapshot_counts": probe["counts_used"],
        "probe_snapshot_provenance": provenance,
        "probe_route_after_burst_s": probe["route_monotonic"] - burst_start,
        "probe_snapshot_applied_after_burst_s": (
            probe_snapshot["applied_monotonic"] - burst_start
            if probe_snapshot and probe_snapshot["applied_monotonic"] is not None else None
        ),
        "first_applied_output_after_burst_s": None if first_applied is None else first_applied - burst_start,
    }


def burst_rows(chain: dict, burst: list[dict], frontier: dict, qualification: dict) -> list[dict]:
    placement = {row["request_id"]: row for row in chain["placement"]}
    rows = []
    for position, row in enumerate(burst, start=1):
        native = placement[row["request_id"]]
        fixed = frontier["vllm_load_balancing"]["placement_by_request_id"][row["request_id"]]
        if not qualification["qualified"]:
            status = "SCENARIO_NOT_REACHED"
        else:
            status = "MATCH" if fixed == native["engine"] else "MISMATCH"
        rows.append({
            "burst": row["burst"],
            "request_id": row["request_id"],
            "trace_position": position,
            "num_prefill_tokens": row["num_prefill_tokens"],
            "native_route_seq": native["route_seq"],
            "native_engine": native["engine"],
            "native_snapshot": native["snapshot"],
            "native_counts": native["counts_used"],
            **{f"frontier_{run}": frontier[run]["placement_by_request_id"][row["request_id"]] for run in FRONTIER_RUNS},
            "status": status,
        })
    return rows


def probe_selection(frontier_run: dict, request_id: str) -> dict | None:
    return next((s for s in frontier_run["selections"] if s["request_id"] == request_id), None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--vllm-chain", type=Path, required=True)
    parser.add_argument("--request-ids", type=Path, required=True)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    chain = json.loads(args.vllm_chain.read_text())
    request_ids = json.loads(args.request_ids.read_text())
    frontier = {
        run: json.loads((args.frontier_dir / run / "evidence.json").read_text())
        for run in FRONTIER_RUNS
    }
    bursts: dict[str, list[dict]] = {}
    for row in request_ids["rows"]:
        if row["segment"] == "burst":
            bursts.setdefault(row["burst"], []).append(row)
    num_engines = len(chain["placement"][0]["counts_used"])

    t1 = replay_native_history(chain, num_engines)
    t2: list[dict] = []
    t2_status = {}
    for name, burst in bursts.items():
        qualification = qualify_burst(chain, burst)
        rows = burst_rows(chain, burst, frontier, qualification)
        t2 += rows
        probe = rows[-1]
        t2_status[name] = {
            "qualification": qualification,
            "probe_status": probe["status"],
            "probe_native_engine": probe["native_engine"],
            "probe_frontier": {run: probe[f"frontier_{run}"] for run in FRONTIER_RUNS},
            "control_differs_from_native": probe["frontier_completion_reporting_control"] != probe["native_engine"],
            "probe_frontier_selection": {
                run: probe_selection(frontier[run], probe["request_id"])
                for run in ("vllm_load_balancing", "completion_reporting_control")
            },
        }
    gaps = [
        {"table": "T1", "request_id": row["request_id"], "first_cause": row["first_cause"],
         "native": row["native_engine"], "frontier": row["frontier_engine"],
         "anchor": "frontier/scheduler/utils/vllm_dp_load_balancer.py; vllm/v1/engine/coordinator.py"}
        for row in t1 if row["status"] != "MATCH"
    ] + [
        {"table": "T2", "request_id": row["request_id"], "first_cause": "see the burst's qualification",
         "native": row["native_engine"], "frontier": row["frontier_vllm_load_balancing"],
         "anchor": "tests/comparison/dp_placement_pp/compare_placement.py qualify_burst"}
        for row in t2 if row["status"] == "MISMATCH"
    ]
    formal = [row for row in t1 if row["segment"] in ("burst", "steady")]
    status = {
        "vllm_chain": str(args.vllm_chain),
        "frontier_dir": str(args.frontier_dir),
        "t1_routes": len(t1),
        "t1_formal_routes": len(formal),
        "t1_formal_match": sum(row["status"] == "MATCH" for row in formal),
        "t1_mismatch_by_cause": {
            cause: sum(row["first_cause"] == cause for row in t1 if row["status"] != "MATCH")
            for cause in sorted({row["first_cause"] for row in t1 if row["status"] != "MATCH"})
        },
        "t2_bursts": t2_status,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "t1_replay_table.csv", t1)
    write_csv(args.output_dir / "t2_placement_table.csv", t2)
    write_csv(args.output_dir / "workflow_gap_table.csv", gaps)
    (args.output_dir / "comparison_status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps({
        "t1_formal_routes": status["t1_formal_routes"],
        "t1_formal_match": status["t1_formal_match"],
        "t1_mismatch_by_cause": status["t1_mismatch_by_cause"],
        "t2": {name: {key: burst[key] for key in (
            "probe_status", "probe_native_engine", "probe_frontier", "control_differs_from_native")}
            | {"failed_checks": burst["qualification"]["failed_checks"]}
            for name, burst in t2_status.items()},
    }, indent=1))
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


if __name__ == "__main__":
    raise SystemExit(main())
