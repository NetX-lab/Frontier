"""Validate exact-case request joins and report endpoint-probe A/B evidence."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def analyze(root, request_count, warmups):
    expected = {f"pf4096_dc1024:{i}" for i in range(request_count)}
    all_expected = expected | {
        f"warmup:pf4096_dc1024:r{replay}:{i}"
        for replay in range(warmups) for i in range(request_count)
    }
    summary = {}
    formal_by_mode = {}
    for mode in ("baseline", "endpoints"):
        path = root / mode
        clients = rows(path / "client.jsonl")
        server_rows = rows(path / "server.request_metrics.jsonl")
        server = {row["request_id"]: row for row in server_rows}
        require(len(clients) == len(all_expected), f"{mode}: incomplete client rows")
        require({row["request_id"] for row in clients} == all_expected,
                f"{mode}: unexpected client IDs")
        require(len(server) == len(server_rows) == len(all_expected),
                f"{mode}: duplicate or incomplete server rows")
        phases = [row for row in rows(path / "client.log") if "replay" in row]
        require(len(phases) == warmups + 1, f"{mode}: missing replay barriers")
        for previous, current in zip(phases, phases[1:]):
            require(previous["phase_end_monotonic_s"] <= current["phase_start_monotonic_s"],
                    f"{mode}: warmup overlap")
        formal = []
        for client in clients:
            engine_id = client["response_id"] + "-0"
            row = server[engine_id]
            require(client["prompt_tokens"] == row["request_num_prefill_tokens"] == 4096,
                    f"{engine_id}: incorrect prompt length")
            require(client["completion_tokens_observed"] == row["request_num_decode_tokens"] == 1024,
                    f"{engine_id}: incorrect output length")
            require(row["queue_arrival_monotonic_s"] > 0, f"{engine_id}: missing queue arrival")
            if client["request_id"] in expected:
                formal.append((client, row))
        formal_by_mode[mode] = formal
        summary[mode] = {
            "complete_requests_including_warmups": len(clients),
            "formal_requests": len(formal),
            "client_ttft_mean_ms": statistics.mean(c["client_ttft_ms"] for c, _ in formal),
            "frontend_ttft_mean_ms": statistics.mean(s["ttft"] for _, s in formal),
        }

    endpoint_rows = defaultdict(list)
    files = sorted((root / "endpoints").glob("server.prefill.rank*.jsonl"))
    require(len(files) == 8, "Expected endpoint files from all eight ranks")
    anchor_widths = []
    for path in files:
        records = rows(path)
        require(records[0]["event"] == "clock_anchor", f"{path}: missing anchor")
        anchor_widths.append(records[0]["monotonic_after_s"] - records[0]["monotonic_before_s"])
        for record in records[1:]:
            require(record["event"] == "prefill_completed", f"{path}: unexpected record")
            for request_id in record["request_ids"]:
                endpoint_rows[request_id].append(record)
    canonical = []
    rank_spreads = []
    for client, server in formal_by_mode["endpoints"]:
        engine_id = server["request_id"]
        records = endpoint_rows[engine_id]
        ranks = {row["rank"] for row in records}
        require(len(records) == 4 and ranks in ({0, 1, 2, 3}, {4, 5, 6, 7}),
                f"{engine_id}: expected exactly one complete TP4 group, got {ranks}")
        completed = max(row["prefill_completed_monotonic_s"] for row in records)
        queue = server["queue_arrival_monotonic_s"]
        require(math.isfinite(completed) and completed >= queue,
                f"{engine_id}: invalid canonical ordering")
        for row in records:
            require(row["prefill_completed_monotonic_s"] - row["anchor_uncertainty_s"]
                    <= row["observed_after_sync_monotonic_s"], f"{engine_id}: future endpoint")
        ttft = (completed - queue) * 1000
        require(ttft <= client["client_ttft_ms"] + 0.1,
                f"{engine_id}: canonical TTFT exceeds client TTFT")
        canonical.append({"request_id": client["request_id"], "engine_request_id": engine_id,
                          "queue_arrival_monotonic_s": queue,
                          "prefill_completed_monotonic_s": completed, "canonical_ttft_ms": ttft})
        rank_spreads.append((completed - min(r["prefill_completed_monotonic_s"] for r in records)) * 1000)
    summary["endpoints"].update({
        "canonical_ttft_mean_ms": statistics.mean(r["canonical_ttft_ms"] for r in canonical),
        "max_rank_completion_spread_ms": max(rank_spreads),
        "max_initial_anchor_width_us": max(anchor_widths) * 1e6,
    })
    baseline = summary["baseline"]["frontend_ttft_mean_ms"]
    recorded = summary["endpoints"]["frontend_ttft_mean_ms"]
    summary["frontend_ttft_ab_change_percent"] = (recorded / baseline - 1) * 100
    summary["status"] = "PASS_REQUEST_JOINS_AND_ORDERING"
    summary["limitations"] = ["One sequential A/B pair does not isolate run-to-run variance.",
                               "Initial anchor brackets do not bound long-run clock drift.",
                               "This is endpoint integration validation, not simulator parity."]
    return summary, canonical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    result, requests = analyze(args.run_root, 100, 10)
    with (args.run_root / "endpoint_validation.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    with (args.run_root / "canonical_requests.json").open("x") as stream:
        json.dump(requests, stream, indent=2)
    print(json.dumps(result, indent=2))
