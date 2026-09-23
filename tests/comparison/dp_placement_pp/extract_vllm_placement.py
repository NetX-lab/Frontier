#!/usr/bin/env python3
"""Extract the DP placement chain from one instrumented vLLM run.

Reads the per-process placement records that the ground-truth checkout writes
(`vllm/v1/frontier_trace.py`, record kinds in the case manifest) and joins them
by their correlation ids, never by time:

- an engine's published iterations and the coordinator's receipts from that
  engine pair one to one in order (each engine reports over its own ordered
  channel), and each pair must carry the same `(engine, wave, step)`;
- a coordinator publication, the frontend's application of it and every
  routing decision made from it share `snapshot`;
- a routing decision and the engine iteration that admitted the request share
  the request id.

Within one process, `seq` orders the records. Each publication is traced back
to the engine reports its counts came from: the coordinator publishes either
its current per-engine counts or the counts it latched when a report with a
newer step arrived, and each count was set by the last report of that engine
before that point. That provenance is what qualifies the T2 slice (plan
§18.6): which engine iterations, of which branch, the frontend had seen when
it routed the probe.

Outputs in ``--output-dir``: ``placement.csv`` (one row per request),
``engine_iterations.csv``, ``snapshots.csv``, ``chain.json`` (the same rows
with list-valued fields intact, for the comparison), and
``extraction_status.json`` with every completeness check.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

ENGINE_REQUEST_ID = re.compile(r"^cmpl-(?P<request_id>.+)-0$")
ROLE_BY_KIND = {
    "engine_iteration": "engine",
    "coordinator_receive": "coordinator",
    "coordinator_publish": "coordinator",
    "frontend_snapshot": "frontend",
    "frontend_route": "frontend",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_processes(record_dir: Path) -> dict[str, list[list[dict]]]:
    processes: dict[str, list[list[dict]]] = {"engine": [], "coordinator": [], "frontend": []}
    for path in sorted(record_dir.glob("dp_placement_*.jsonl")):
        records = read_jsonl(path)
        roles = {ROLE_BY_KIND[record["kind"]] for record in records}
        if len(roles) != 1:
            raise ValueError(f"{path.name}: records of several roles {sorted(roles)}")
        seqs = [record["seq"] for record in records]
        if seqs != list(range(len(records))):
            raise ValueError(f"{path.name}: seq is not 0..n-1 in file order")
        processes[roles.pop()].append(records)
    return processes


def case_request_id(engine_request_id: str) -> str:
    match = ENGINE_REQUEST_ID.match(engine_request_id)
    if match is None:
        raise ValueError(f"engine request id {engine_request_id!r} is not cmpl-<id>-0")
    return match.group("request_id")


def trace_publications(coordinator: list[dict]) -> list[dict]:
    """Attach to each publication the engine reports its counts came from."""

    last_report: dict[int, dict] = {}
    latched_from: dict[int, dict] | None = None
    publications = []
    for record in coordinator:
        if record["kind"] == "coordinator_receive":
            if record["disposition"] == "latched_previous_step":
                latched_from = dict(last_report)
            last_report[record["engine"]] = record
        else:
            source = latched_from if record["counts_source"] == "latched_previous_step" else last_report
            if record["counts_source"] == "latched_previous_step":
                latched_from = None
            engines = range(len(record["counts"]))
            publications.append({
                **record,
                "provenance": [
                    None if engine not in source else {
                        key: source[engine].get(key) for key in ("engine", "wave", "step", "iteration_seq")
                    }
                    for engine in engines
                ],
            })
    return publications


def extract(run_dir: Path, request_ids: dict) -> tuple[dict, dict]:
    processes = load_processes(run_dir / "dp_placement")
    problems: list[str] = []
    if len(processes["coordinator"]) != 1 or len(processes["frontend"]) != 1:
        problems.append(
            f"expected one coordinator and one frontend file, found "
            f"{len(processes['coordinator'])} and {len(processes['frontend'])}"
        )
    coordinator = processes["coordinator"][0] if processes["coordinator"] else []
    frontend = processes["frontend"][0] if processes["frontend"] else []

    iterations = sorted(
        (record for records in processes["engine"] for record in records),
        key=lambda record: (record["engine"], record["seq"]),
    )
    iteration_by_seq = {(record["engine"], record["seq"]): record for record in iterations}
    admission = {}
    for record in iterations:
        for engine_request_id in record["scheduled_new_req_ids"]:
            request_id = case_request_id(engine_request_id)
            if request_id in admission:
                problems.append(f"{request_id} admitted twice")
            admission[request_id] = record

    receipts = [record for record in coordinator if record["kind"] == "coordinator_receive"]
    for engine in sorted({record["engine"] for record in iterations} | {r["engine"] for r in receipts}):
        published = [r for r in iterations if r["engine"] == engine and r["published"]]
        received = [r for r in receipts if r["engine"] == engine]
        if len(published) != len(received):
            problems.append(f"engine {engine} published {len(published)} reports, coordinator received {len(received)}")
        for iteration, receipt in zip(published, received):
            if [iteration[k] for k in ("wave", "step", "waiting", "running")] != [
                receipt[k] for k in ("wave", "step", "waiting", "running")
            ]:
                problems.append(f"engine {engine} iteration seq {iteration['seq']} and its receipt differ")
            receipt["iteration_seq"] = iteration["seq"]

    publications = trace_publications(coordinator)
    publication_by_id = {record["snapshot"]: record for record in publications}
    applied = {}
    for record in frontend:
        if record["kind"] != "frontend_snapshot":
            continue
        publication = publication_by_id.get(record["snapshot"])
        if publication is None:
            problems.append(f"frontend applied snapshot {record['snapshot']}, never published")
        elif publication["counts"] != record["counts"]:
            problems.append(f"snapshot {record['snapshot']} counts differ between coordinator and frontend")
        applied[record["snapshot"]] = record

    snapshots = []
    for publication in publications:
        provenance = [
            None if source is None or source["iteration_seq"] is None else {
                **source,
                "branch": iteration_by_seq[(source["engine"], source["iteration_seq"])]["branch"],
            }
            for source in publication["provenance"]
        ]
        frontend_record = applied.get(publication["snapshot"])
        snapshots.append({
            "snapshot": publication["snapshot"],
            "counts_waiting_running": publication["counts"],
            "counts_source": publication["counts_source"],
            "wave": publication["wave"],
            "publish_monotonic": publication["monotonic"],
            "applied_monotonic": None if frontend_record is None else frontend_record["monotonic"],
            "provenance": provenance,
        })

    routes = {}
    for record in frontend:
        if record["kind"] != "frontend_route":
            continue
        request_id = case_request_id(record["request_id"])
        if request_id in routes:
            problems.append(f"{request_id} routed twice")
        routes[request_id] = record
        if record["snapshot"] and record["snapshot"] not in applied:
            problems.append(f"{request_id} routed from snapshot {record['snapshot']}, never applied")

    client = {record["request_id"]: record for record in read_jsonl(run_dir / "client_requests.jsonl")}
    metrics = {
        case_request_id(record["request_id"]): record
        for record in read_jsonl(run_dir / "request_metrics.jsonl")
    }
    placement = []
    for row in request_ids["rows"]:
        request_id = row["request_id"]
        route, admitted, sent = routes.get(request_id), admission.get(request_id), client.get(request_id)
        if route is None or admitted is None or sent is None:
            problems.append(f"{request_id}: route={route is not None} admission={admitted is not None} client={sent is not None}")
            continue
        if route["engine"] != admitted["engine"]:
            problems.append(f"{request_id} routed to {route['engine']} but admitted by {admitted['engine']}")
        placement.append({
            "request_id": request_id,
            "segment": row["segment"],
            "role": row["role"],
            "arrived_at": row["arrived_at"],
            "dispatch_offset_s": sent["dispatch_offset_s"],
            "route_monotonic": route["monotonic"],
            "route_seq": route["seq"],
            "engine": route["engine"],
            "snapshot": route["snapshot"],
            "counts_used": route["counts"],
            "score": route["score"],
            "start_index": route["start_index"],
            "reservation": route["reservation"],
            "admitted_wave": admitted["wave"],
            "admitted_step": admitted["step"],
            "admitted_branch": admitted["branch"],
            "admitted_monotonic": admitted["monotonic"],
            "http_status": sent["http_status"],
            "ttft_ms": metrics.get(request_id, {}).get("ttft"),
            "model_execution_time_ms": metrics.get(request_id, {}).get("request_model_execution_time"),
        })

    status = {
        "run_dir": str(run_dir),
        "num_engine_files": len(processes["engine"]),
        "num_engine_iterations": len(iterations),
        "num_published_iterations": sum(1 for record in iterations if record["published"]),
        "num_coordinator_receipts": len(receipts),
        "num_publications": len(publications),
        "num_frontend_snapshots": len(applied),
        "num_routes": len(routes),
        "num_requests": len(request_ids["rows"]),
        "num_placements": len(placement),
        "out_of_order_receipts": sum(
            1 for record in coordinator
            if record["kind"] == "coordinator_receive" and record["disposition"] == "out_of_order"
        ),
        "request_metrics_complete": sorted(metrics) == sorted(row["request_id"] for row in request_ids["rows"]),
        "problems": problems,
        "status": "PASS" if not problems and len(placement) == len(request_ids["rows"]) else "FAIL",
    }
    chain = {"placement": placement, "engine_iterations": iterations, "snapshots": snapshots}
    return chain, status


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--request-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    chain, status = extract(args.run_dir, json.loads(args.request_ids.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "placement.csv", chain["placement"])
    write_csv(args.output_dir / "engine_iterations.csv", chain["engine_iterations"])
    write_csv(args.output_dir / "snapshots.csv", chain["snapshots"])
    (args.output_dir / "chain.json").write_text(json.dumps(chain, indent=1))
    (args.output_dir / "extraction_status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps({key: status[key] for key in ("status", "num_placements", "num_requests")}
                     | {"problems": status["problems"][:10]}))
    return 0 if status["status"] == "PASS" else 5


if __name__ == "__main__":
    raise SystemExit(main())
