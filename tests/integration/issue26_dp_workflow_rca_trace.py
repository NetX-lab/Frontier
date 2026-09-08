"""Join same-run vLLM routing, enqueue, and prefill admission diagnostics."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze(run, command_path, output):
    clients = rows(run / "client.jsonl")
    expected = {f"pf4096_dc1024:{i}" for i in range(100)}
    formal = {row["response_id"] + "-0": row for row in clients
              if row["request_id"] in expected}
    assert len(clients) == 400 and len(formal) == 100
    assert {row["request_id"] for row in formal.values()} == expected
    assert all(row["prompt_tokens"] == 4096 and row["completion_tokens_observed"] == 1024
               for row in clients)
    route, enqueue, snapshots = {}, {}, []
    state_checks = 0
    for path in sorted(run.glob("server.dp_route*.pid*.jsonl")):
        state = None
        for row in rows(path):
            event = row["event"]
            if event == "snapshot_receive":
                snapshots.append(row)
                if row["counts_updated"]:
                    state = [list(pair) for pair in row["counts"]]
            elif event == "route":
                assert row["client_count"] == 1 and row["eng_start_index"] == 0
                assert row["explicit_dp_rank"] is None
                if state is not None:
                    assert state == row["counts"], (row, state)
                    state_checks += 1
                router = VllmDPLoadBalancer(len(row["counts"]))
                router.frontend_counts = [RequestLoad(*pair) for pair in row["counts"]]
                assert router.select(0.0) == row["selected_dp_rank"]
                state = [list(pair) for pair in router.frontend_counts]
                assert row["request_id"] not in route
                route[row["request_id"]] = row
            elif event == "enqueue":
                assert row["request_id"] not in enqueue
                enqueue[row["request_id"]] = row
    assert len(route) == len(enqueue) == 400
    assert set(route) == set(enqueue) == {row["response_id"] + "-0" for row in clients}

    schedules = {}
    for line_number, row in enumerate(rows(run / "server.scheduler.log"), 1):
        if row["event"] != "schedule":
            continue
        for request_id, tokens in row["num_scheduled_tokens"].items():
            if request_id in formal and tokens > 1:
                assert request_id not in schedules
                schedules[request_id] = {**row, "source_line": line_number}
    assert set(schedules) == set(formal)

    admissions = defaultdict(dict)
    for path in sorted(run.glob("server.batch.dp*.tp*.pp*.jsonl")):
        computed = Counter()
        for row in rows(path):
            worker = (row["dp_rank"], row["tp_rank"], row["pp_rank"])
            for request_id, tokens in zip(row["request_ids"], row["request_num_tokens"]):
                if request_id in formal and tokens > 1:
                    assert worker not in admissions[request_id]
                    admissions[request_id][worker] = {
                        "batch_id": row["batch_id"], "source": str(path),
                        "members": [{"request_id": rid, "tokens": n,
                                     "prior_scheduled_tokens": computed[rid]}
                                    for rid, n in zip(row["request_ids"], row["request_num_tokens"])],
                    }
            computed.update(dict(zip(row["request_ids"], row["request_num_tokens"])))
    joined = []
    for request_id, client in formal.items():
        r, q = route[request_id], enqueue[request_id]
        assert r["selected_dp_rank"] == q["dp_rank"]
        assert r["timestamp_monotonic_s"] <= q["timestamp_monotonic_s"]
        assert set(admissions[request_id]) == {(q["dp_rank"], tp, 0) for tp in range(4)}
        batch = admissions[request_id][(q["dp_rank"], 0, 0)]
        assert all(item["members"] == batch["members"] for item in admissions[request_id].values())
        scheduled = schedules[request_id]
        assert {item["request_id"]: item["tokens"] for item in batch["members"]} == scheduled["num_scheduled_tokens"]
        assert q["timestamp_monotonic_s"] <= scheduled["timestamp_monotonic"]
        joined.append({"request_id": request_id, "client_request_id": client["request_id"],
                       "route": r, "enqueue": q, "admission": batch,
                       "schedule": scheduled,
                       "enqueue_to_schedule_ms": 1000 * (scheduled["timestamp_monotonic"] - q["timestamp_monotonic_s"]),
                       "route_to_enqueue_ms": 1000 * (q["timestamp_monotonic_s"] - r["timestamp_monotonic_s"])})
    joined.sort(key=lambda row: row["route"]["timestamp_monotonic_s"])
    first_route = joined[0]["route"]["timestamp_monotonic_s"]
    earlier_snapshots = [row for row in snapshots if row["timestamp_monotonic_s"] < first_route]
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "PASS", "source_run": str(run), "formal_count": len(joined),
              "all_routes": len(route), "snapshot_state_checks": state_checks,
              "last_snapshot_before_formal": max(earlier_snapshots,
                  key=lambda row: row["timestamp_monotonic_s"], default=None),
              "requests": joined,
              "limits": "Instrumented same-run workflow evidence. Prior scheduled tokens are "
                        "cumulative batch-ledger work, not directly observed KV occupancy. "
                        "Route-input replay changes the arrival observation boundary only as "
                        "a diagnostic counterfactual; neither replay closes E2E acceptance."}
    (output / "same_run_join.json").write_text(json.dumps(result, indent=2) + "\n")
    for boundary in ("route", "enqueue"):
        ordered = sorted(joined, key=lambda row: row[boundary]["timestamp_monotonic_s"])
        origin = ordered[0][boundary]["timestamp_monotonic_s"]
        trace = output / f"{boundary}_arrivals.csv"
        mapping = output / f"{boundary}_mapping.csv"
        with trace.open("w") as stream, mapping.open("w") as map_stream:
            writer = csv.writer(stream)
            writer.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
            map_writer = csv.writer(map_stream)
            map_writer.writerow(["frontier_request_id", "client_request_id"])
            for index, row in enumerate(ordered):
                writer.writerow([row[boundary]["timestamp_monotonic_s"] - origin, 4096, 1024])
                map_writer.writerow([index, row["client_request_id"]])
        command = json.loads(command_path.read_text())
        command[command.index("--trace_request_generator_config_trace_file") + 1] = str(trace.resolve())
        (output / f"{boundary}_command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "formal_count", "all_routes", "snapshot_state_checks")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "command", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.command, args.output)
