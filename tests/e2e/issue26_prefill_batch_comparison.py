"""Compare formal prefill batch membership using validated request identities."""

import argparse
import csv
import json
from pathlib import Path


def json_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compare(ledger, mapping, diagnostic, clients, output):
    with mapping.open() as stream:
        frontier_ids = {row["frontier_request_id"]: row["client_request_id"]
                        for row in csv.DictReader(stream)}
    formal_ids = list(frontier_ids.values())
    assert len(set(formal_ids)) == len(formal_ids)
    vllm_ids = {row["response_id"] + "-0": row["request_id"]
                for row in json_rows(clients)}
    validation = json.loads(diagnostic.read_text())
    assert validation["status"] == "PASS" and validation["mode"] == "batch"
    batches = {"frontier": [], "vllm": []}
    for row in json_rows(ledger):
        if not any(int(n) > 1 for n in row["request_num_tokens"]):
            continue
        batches["frontier"].append({
            "dp": row["replica_local_id"], "batch_id": row["batch_id"],
            "start_s": row["stage_start_ts"], "end_s": row["stage_end_ts"],
            "members": sorted((frontier_ids[str(rid)], int(n)) for rid, n in zip(
                row["request_ids"], row["request_num_tokens"])),
        })
    for worker in validation["workers"]:
        if worker["identity"][1] != 0:
            continue
        for row in worker["formal_prefill_batches"]:
            batches["vllm"].append({
                "dp": worker["identity"][0], "batch_id": row["batch_id"],
                "members": sorted((vllm_ids[rid], int(n)) for rid, n in zip(
                    row["request_ids"], row["request_num_tokens"])),
            })
    by_request = {}
    for side, records in batches.items():
        indexed = {}
        for record in records:
            for rid, tokens in record["members"]:
                if tokens > 1 and rid in formal_ids:
                    assert rid not in indexed, (side, rid)
                    indexed[rid] = record
        assert set(indexed) == set(formal_ids)
        by_request[side] = indexed
    comparisons = []
    for rid in formal_ids:
        f, v = (by_request[side][rid] for side in ("frontier", "vllm"))
        comparisons.append({"request_id": rid, "same_dp": f["dp"] == v["dp"],
                            "same_members": f["members"] == v["members"],
                            "frontier": f, "vllm": v})
    report = {
        "status": "COMPLETE", "formal_requests": len(formal_ids),
        "same_dp_count": sum(row["same_dp"] for row in comparisons),
        "same_members_count": sum(row["same_members"] for row in comparisons),
        "first_request": comparisons[0], "comparisons": comparisons,
        "sources": {"ledger": str(ledger), "mapping": str(mapping),
                    "diagnostic_validation": str(diagnostic), "clients": str(clients)},
        "limits": "vLLM batches come from an isolated diagnostic run. Clean and diagnostic arrival timing may differ. DP-local batch IDs do not identify a shared collective round; unlogged dummy rounds are excluded. Membership equality does not establish timing parity.",
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("comparisons", "sources")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ledger", "mapping", "diagnostic", "clients", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    compare(**vars(args))
