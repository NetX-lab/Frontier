"""Verify drained single-request replays and report their formal CUDA span."""

import argparse
import json
from pathlib import Path
from statistics import median


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def analyze(run, warmups):
    clients = list(read_rows(run / "client.jsonl"))
    expected = [f"warmup:pf4096_dc1024:r{r}:0" for r in range(warmups)]
    expected.append("pf4096_dc1024:0")
    assert [row["request_id"] for row in clients] == expected
    assert all(row["prompt_tokens"] == 4096 and
               row["completion_tokens_observed"] == 1024 for row in clients)
    logs = list(read_rows(run / "client.log"))
    phases = [row for row in logs if "replay" in row]
    assert [row["replay"] for row in phases] == list(range(warmups + 1))
    for index, (phase, client) in enumerate(zip(phases, clients)):
        assert phase["completed_requests"] == 1
        assert phase["phase_start_monotonic_s"] <= client["dispatch_monotonic_s"]
        assert client["dispatch_monotonic_s"] < phase["phase_end_monotonic_s"]
        if index:
            assert phases[index - 1]["phase_end_monotonic_s"] <= phase["phase_start_monotonic_s"]
            assert clients[index - 1]["client_completion_time_ns"] <= client["request_arrival_wall_time_ns"]
    assert logs[-1] == {"formal_requests": 1, "formal_unique_ids": 1}
    request = clients[-1]["response_id"] + "-0"
    selected, workers = [], set()
    for path in sorted(run.glob("server.batch.dp*.tp*.pp*.jsonl")):
        for row in read_rows(path):
            workers.add((row["dp_rank"], row["tp_rank"], row["pp_rank"]))
            if request in row["request_ids"] and row["batch_num_prefill_tokens"]:
                assert row["request_ids"] == [request]
                assert row["batch_size"] == 1 and row["request_num_tokens"] == [4096]
                assert row["batch_num_tokens"] == row["batch_num_prefill_tokens"] == 4096
                assert row["batch_num_decode_tokens"] == 0
                selected.append({**row, "source": str(path)})
    assert workers == {(dp, tp, 0) for dp in range(2) for tp in range(4)}
    assert len(selected) == 4 and {row["tp_rank"] for row in selected} == set(range(4))
    assert len({(row["dp_rank"], row["batch_id"]) for row in selected}) == 1
    values = sorted(row["batch_execution_time_ms"] for row in selected)
    manifest = json.loads((run / "mode_manifest.json").read_text())
    assert manifest["warmup_rounds"] == warmups
    return {
        "bounded_identity_and_drain": "PASS", "warmups": warmups,
        "client_rows": len(clients), "requests_per_replay": 1,
        "formal": clients[-1], "selected": selected,
        "median_ms": median(values), "rank_p90_ms": values[2] * .3 + values[3] * .7,
        "rank_max_ms": values[-1], "rank_spread_ms": values[-1] - values[0],
        "manifest_formal_requests": manifest["formal_requests"],
        "actual_formal_requests": 1,
        "limits": "Bounded diagnostic only. P90 is across four ranks, not repeated runs. Full standard-suite validation is separate.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--warmups", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run, args.warmups), indent=2))
