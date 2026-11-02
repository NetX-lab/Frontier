"""Fill the UNKNOWN rows of the Q4 GPU-hours table from platform container clocks.

Inputs:
- `../assessment/gpu_hours.csv` (Q4, unchanged): 35 H200 jobs, 19 with UNKNOWN hours.
- `receipts/<job>.json` from `h200_mainline/terminal_receipt.py`, or
  `receipts/<job>.replica_only.json` when the parent job object is purged (HTTP 404)
  and only its replica objects remain.

For every receipt, container GPU-hours are recomputed from each terminated container's
startedAt/finishedAt and the job's per-replica GPU limit, and must equal the value the
receipt records; the replica count times the GPU limit must equal the table's `gpus`.

Usage: merge_gpu_hours.py  (writes gpu_hours_complete.csv and gpu_hours_complete.json)
"""

import csv
import datetime
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = HERE.parent / "assessment" / "gpu_hours.csv"
RECEIPTS = HERE / "receipts"


def seconds(container):
    clock = container["state"]["terminated"]
    start, finish = (datetime.datetime.strptime(clock[key], "%Y-%m-%dT%H:%M:%SZ")
                     for key in ("startedAt", "finishedAt"))
    return (finish - start).total_seconds()


def receipt_for(job):
    full, replica_only = RECEIPTS / f"{job}.json", RECEIPTS / f"{job}.replica_only.json"
    if full.exists():
        receipt = json.loads(full.read_text())
        task = receipt["tasks"]["default"]
        gpus_per_replica = int(task["resources"][0]["limits"]["nvidia.com/gpu"])
        return full, receipt, task["replicas"], gpus_per_replica, "platform terminal receipt"
    receipt = json.loads(replica_only.read_text())
    limits = {int(replica["gpu_limits"]["default"]) for replica in receipt["replicas"]}
    if len(limits) != 1:
        raise SystemExit(f"{job}: replicas disagree on the GPU limit {limits}")
    return (replica_only, receipt, len(receipt["replicas"]), limits.pop(),
            "replica objects; parent job object purged (HTTP 404)")


def main():
    rows = list(csv.DictReader(TABLE.open()))
    failures, merged = [], []
    for row in rows:
        if row["gpu_hours"] != "UNKNOWN":
            merged.append(dict(row, clock_source="Q4 assessment", clock_file="", clock_sha256=""))
            continue
        job = row["job_id"]
        path, receipt, replicas, gpus_per_replica, source = receipt_for(job)
        containers = [c for replica in receipt["replicas"] for c in replica["containers"]]
        hours = sum(seconds(c) for c in containers) * gpus_per_replica / 3600
        if receipt["creator"] != "i-fengyicheng" or receipt["charged_group"] != "step_main":
            failures.append(f"{job}: creator {receipt['creator']}, group {receipt['charged_group']}")
        if len(containers) != replicas or replicas * gpus_per_replica != int(row["gpus"]):
            failures.append(f"{job}: {replicas} replicas x {gpus_per_replica} GPUs, "
                            f"{len(containers)} containers, table gpus {row['gpus']}")
        if abs(hours - receipt["container_gpu_hours"]) > 1e-9:
            failures.append(f"{job}: recomputed {hours} differs from receipt {receipt['container_gpu_hours']}")
        merged.append(dict(row, gpu_hours=repr(hours), basis="container start-to-finish x GPU limit",
                           clock_source=source, clock_file=str(path.relative_to(HERE)),
                           clock_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if failures:
        raise SystemExit("Not merged:\n" + "\n".join(failures))

    with (HERE / "gpu_hours_complete.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    by_scope = defaultdict(float)
    for row in merged:
        by_scope[row["accounting_scope"]] += float(row["gpu_hours"])
    filled = [row for row in merged if row["clock_source"] != "Q4 assessment"]
    summary = {
        "jobs": len(merged),
        "filled_from_platform_clocks": len(filled),
        "filled_gpu_hours": sum(float(row["gpu_hours"]) for row in filled),
        "total_container_gpu_hours": sum(by_scope.values()),
        "by_accounting_scope": dict(by_scope),
        "basis": "container start-to-finish x GPU limit for every job; includes initialization, "
                 "compilation, warm-up and failed attempts; excludes queue time and pod time "
                 "before the container started (image pull)",
        "source_table": {"path": str(TABLE), "sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest()},
    }
    with (HERE / "gpu_hours_complete.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
