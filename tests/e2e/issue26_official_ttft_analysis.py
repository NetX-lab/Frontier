"""Validate official request metrics and emit an observed engine-arrival trace."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def analyze(run, output, warmups):
    client = read_jsonl(run / "client.jsonl")
    server = read_jsonl(run / "server.request_metrics.jsonl")
    formal_ids = {f"pf4096_dc1024:{i}" for i in range(100)}
    expected = formal_ids | {
        f"warmup:pf4096_dc1024:r{replay}:{i}"
        for replay in range(warmups) for i in range(100)
    }
    assert len(client) == (warmups + 1) * 100 and {r["request_id"] for r in client} == expected
    assert len(server) == (warmups + 1) * 100
    by_server = {r["request_id"]: r for r in server}
    assert len(by_server) == 400
    assert {r["response_id"] + "-0" for r in client} == set(by_server)
    phases = [r for r in read_jsonl(run / "client.log") if "replay" in r]
    assert [r["replay"] for r in phases] == list(range(warmups + 1))
    assert all(r["completed_requests"] == 100 for r in phases)
    assert all(phases[i]["phase_end_monotonic_s"] <=
               phases[i + 1]["phase_start_monotonic_s"] for i in range(warmups))
    joined = []
    for row in client:
        actual = by_server[row["response_id"] + "-0"]
        assert row["prompt_tokens"] == actual["request_num_prefill_tokens"] == 4096
        assert row["completion_tokens_observed"] == actual["request_num_decode_tokens"] == 1024
        for field in ("ttft", "arrival_time", "completion_time", "queue_arrival_monotonic_s"):
            assert math.isfinite(actual[field]) and actual[field] > 0, (row["request_id"], field)
        assert actual["completion_time"] > actual["arrival_time"]
        if row["request_id"] in formal_ids:
            joined.append({"client_request_id": row["request_id"],
                           "server_request_id": actual["request_id"],
                           "server_ttft_ms": actual["ttft"],
                           "client_ttft_ms": row["client_ttft_ms"],
                           "server_arrival_wall_s": actual["arrival_time"],
                           "queue_arrival_monotonic_s": actual["queue_arrival_monotonic_s"]})
    joined.sort(key=lambda r: (r["queue_arrival_monotonic_s"], r["client_request_id"]))
    origin = joined[0]["queue_arrival_monotonic_s"]
    for index, row in enumerate(joined):
        row["frontier_request_id"] = index
        row["arrived_at"] = row["queue_arrival_monotonic_s"] - origin
    output.mkdir(parents=True, exist_ok=False)
    with (output / "request_id_map.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    with (output / "frontier_queue_arrivals.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arrived_at", "num_prefill_tokens", "num_decode_tokens"])
        writer.writerows((row["arrived_at"], 4096, 1024) for row in joined)
    report = {"status": "PASS", "eligible_formal_requests": 100,
              "warmup_requests_excluded": warmups * 100, "source_directory": str(run),
              "server_ttft_mean_ms": statistics.mean(r["server_ttft_ms"] for r in joined),
              "client_ttft_mean_ms": statistics.mean(r["client_ttft_ms"] for r in joined),
              "queue_arrival_span_s": joined[-1]["arrived_at"],
              "metric_definition": "official vLLM server first_token_latency",
              "trace_definition": "observed engine queue arrivals, normalized to first formal queue arrival",
              "frontier_id_assumption": "fresh simulator process; sequential Request IDs start at zero",
              "limitations": "This validates clean evidence, not Frontier parity. Wall and monotonic clocks are not subtracted. Missing critical-path work is not estimated from the total gap."}
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.warmups < 10:
        raise ValueError("--warmups must be at least 10")
    analyze(arguments.run, arguments.output, arguments.warmups)
