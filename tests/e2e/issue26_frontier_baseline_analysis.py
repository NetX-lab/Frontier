"""Validate fresh Frontier request artifacts and compare unadjusted server TTFT."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def indexed(rows, field):
    result = {int(row[field]): row for row in rows}
    assert len(rows) == len(result) == 100, (field, "Expected 100 unique requests")
    assert set(result) == set(range(100)), (field, "Unexpected request IDs")
    return result


def close(actual, expected, label, tolerance=1e-6):
    actual, expected = float(actual), float(expected)
    assert math.isfinite(actual) and math.isfinite(expected), (label, "Nonfinite value")
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance), (label, actual, expected)


def analyze(metrics, mapping, output):
    with mapping.open() as stream:
        mapped = indexed(list(csv.DictReader(stream)), "frontier_request_id")
    records = [json.loads(line) for line in
               (metrics / "metrics_ground_truth.jsonl").read_text().splitlines() if line]
    assert len(records) == 200 and {r["event_type"] for r in records} == {
        "request_arrival", "request_completion"}, "Unexpected request event coverage"
    arrivals = indexed([r for r in records if r["event_type"] == "request_arrival"], "request_id")
    completions = indexed([r for r in records if r["event_type"] == "request_completion"], "request_id")
    with (metrics / "request_metrics.csv").open() as stream:
        csv_rows = indexed(list(csv.DictReader(stream)), "Request Id")
    system = json.loads((metrics / "system_metrics.json").read_text())
    assert system["simulation_metadata"]["completed_requests"] == 100
    assert system["simulation_metadata"]["total_requests"] == 100
    compared = []
    for rid, row in sorted(completions.items()):
        source, arrival, exported = mapped[rid], arrivals[rid], csv_rows[rid]
        for name, expected in (("prefill", 4096), ("decode", 1024)):
            assert row[f"num_{name}_tokens"] == arrival[f"num_{name}_tokens"] == expected, rid
            assert float(exported[f"request_num_{name}_tokens"]) == expected, rid
        for event in (arrival, row):
            close(event["arrived_at"], source["arrived_at"], (rid, "arrival mapping"), 1e-9)
            close(event["arrived_at_ms"], 1000 * event["arrived_at"], (rid, "arrival units"))
        assert 0 <= row["arrived_at"] < row["prefill_completed_at"] <= row["completed_at"], rid
        for endpoint in ("prefill_completed_at", "completed_at"):
            close(row[endpoint + "_ms"], 1000 * row[endpoint], (rid, endpoint))
        for metric, endpoint, csv_name in (("ttft", "prefill_completed_at", "ttft"),
                                         ("request_e2e_time", "completed_at", "request_e2e_time")):
            close(row[metric + "_s"], row[endpoint] - row["arrived_at"], (rid, metric), 1e-9)
            close(row[metric + "_ms"], 1000 * row[metric + "_s"], (rid, metric + " units"))
            close(exported[csv_name], row[metric + "_ms"], (rid, "CSV " + metric))
        official = float(source["server_ttft_ms"])
        assert math.isfinite(official) and official > 0, (rid, "Official TTFT")
        gap = row["ttft_ms"] - official
        compared.append({"frontier_request_id": rid, "client_request_id": source["client_request_id"],
                         "server_request_id": source["server_request_id"], "arrived_at_s": row["arrived_at"],
                         "prefill_completed_at_s": row["prefill_completed_at"], "completed_at_s": row["completed_at"],
                         "frontier_ttft_ms": row["ttft_ms"], "official_server_ttft_ms": official,
                         "signed_gap_ms": gap, "absolute_error_ms": abs(gap),
                         "relative_error_percent": 100 * abs(gap) / official})
    predicted = statistics.mean(r["frontier_ttft_ms"] for r in compared)
    actual = statistics.mean(r["official_server_ttft_ms"] for r in compared)
    close(system["ttft_statistics"]["mean"], predicted, "System TTFT mean")
    assert system["ttft_statistics"]["unit"] == "ms"
    summary = {"artifact_validation": "PASS", "formal_requests": 100,
               "frontier_ttft_mean_ms": predicted, "official_server_ttft_mean_ms": actual,
               "mean_signed_gap_ms": predicted - actual, "absolute_error_of_means_ms": abs(predicted - actual),
               "relative_error_of_means_percent": 100 * abs(predicted - actual) / actual,
               "gap_definition": "Frontier mean minus official vLLM server mean",
               "d006_formal_gate": "NOT_EVALUATED_UNADJUSTED_BASELINE",
               "limitations": "Frontier queue-arrival-to-prefill-completion omits unaccounted official-server critical-path work. This baseline does not establish D006 parity or attribute the residual to CPU overhead.",
               "metrics_directory": str(metrics), "mapping_file": str(mapping)}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "request_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compared[0]))
        writer.writeheader()
        writer.writerows(compared)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ("metrics", "mapping", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    analyze(args.metrics, args.mapping, args.output)
