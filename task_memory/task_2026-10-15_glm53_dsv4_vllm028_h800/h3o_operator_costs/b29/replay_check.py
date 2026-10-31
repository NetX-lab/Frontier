"""Join the unobserved B29 short replays and check serving-path offsets."""
import csv
import json
import math
from pathlib import Path
import sys


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(root):
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["status"] == "FINISHED", receipt["status"]
    assert receipt["observer_enabled"] is False
    directory = Path(receipt["metrics_dir"])
    ground = rows(directory / "metrics_ground_truth.jsonl")
    arrivals = {row["request_id"]: row for row in ground if row["event_type"] == "request_arrival"}
    completed = {row["request_id"]: row for row in ground if row["event_type"] == "request_completion"}
    transfers = {row["request_ids"][0]: row for row in rows(directory / "frontier_transfer_ledger.jsonl")
                 if row["transfer_kind"] == "kv_cache"}
    with (directory / "request_metrics.csv").open() as stream:
        metrics = {int(row["Request Id"]): row for row in csv.DictReader(stream)}
    assert len(arrivals) == len(completed) == len(transfers) == len(metrics) == 8
    facts = {}
    for rid in sorted(completed):
        arrival, done, transfer = arrivals[rid], completed[rid], transfers[rid]
        # A10 serving TTFT ends at first DECODE output. Frontier CSV TTFT keeps
        # its existing canonical prefill-completion endpoint; verify both.
        first = done["first_decode_token_completed_at"]
        ttft = (first - done["arrived_at"]) * 1000
        canonical_ttft = (done["prefill_completed_at"] - done["arrived_at"]) * 1000
        assert math.isclose(float(metrics[rid]["ttft"]), canonical_ttft, rel_tol=0, abs_tol=1e-8)
        assert math.isclose(done["ttft_ms"], canonical_ttft, rel_tol=0, abs_tol=1e-8)
        facts[rid] = dict(
            ttft_ms=ttft, canonical_ttft_ms=canonical_ttft,
            pure_decode_first_token_s=done["decode_first_token_completed_at"], arrived_at_s=done["arrived_at"], arrival_s=arrival["time_s"],
            arrival_offset_ms=(arrival["time_s"] - arrival["arrived_at"]) * 1000,
            prefill_end_s=done["prefill_completed_at"], transfer_start_s=transfer["start_ts_s"],
            handoff_offset_ms=(transfer["start_ts_s"] - done["prefill_completed_at"]) * 1000,
            transfer_end_s=transfer["end_ts_s"], decode_first_token_s=first,
            first_decode_step_ms=(done["decode_first_token_completed_at"] - first) * 1000,
            decode_lane=transfer["target_replica_local_id"],
        )
    return facts


def main(scratch, output):
    results = []
    for case in ("p2_32", "p2_64"):
        baseline = run(scratch / f"{case}_baseline_native_env")
        delayed = run(scratch / f"{case}_delayed_native_env")
        for rid, new in delayed.items():
            old = baseline[rid]
            assert new["arrived_at_s"] == old["arrived_at_s"] == float(rid)
            for facts, ingress, handoff in ((old, 0.0, 0.0), (new, 10.3, 11.4)):
                assert math.isclose(facts["arrival_offset_ms"], ingress, rel_tol=0, abs_tol=1e-9)
                assert math.isclose(facts["handoff_offset_ms"], handoff, rel_tol=0, abs_tol=1e-9)
            delta = new["ttft_ms"] - old["ttft_ms"]
            step = max(old["first_decode_step_ms"], new["first_decode_step_ms"])
            components = dict(
                prefill_shift_ms=(new["prefill_end_s"] - old["prefill_end_s"]) * 1000,
                handoff_shift_ms=new["handoff_offset_ms"] - old["handoff_offset_ms"],
                receive_duration_shift_ms=((new["transfer_end_s"] - new["transfer_start_s"])
                                          - (old["transfer_end_s"] - old["transfer_start_s"])) * 1000,
                receive_to_first_token_shift_ms=((new["decode_first_token_s"] - new["transfer_end_s"])
                                                - (old["decode_first_token_s"] - old["transfer_end_s"])) * 1000,
            )
            assert math.isclose(sum(components.values()), delta, rel_tol=0, abs_tol=1e-9)
            results.append(dict(case=case, request=rid, baseline=old, delayed=new, ttft_delta_ms=delta,
                                residual_from_21_7_ms=delta - 21.7, decode_step_ms=step,
                                differs_by_more_than_one_step=abs(delta - 21.7) > step + 1e-9,
                                delta_components=components))
    output.write_text(json.dumps(results, indent=2) + "\n")
    print("case request TTFT baseline/default(ms) delta(ms) arrival offset(ms) handoff offset(ms) D-first baseline/default(s)")
    for row in results:
        old, new = row["baseline"], row["delayed"]
        print(f"{row['case']} {row['request']} {old['ttft_ms']:.6f}/{new['ttft_ms']:.6f} "
              f"{row['ttft_delta_ms']:.6f} {old['arrival_offset_ms']:.3f}/{new['arrival_offset_ms']:.3f} "
              f"{old['handoff_offset_ms']:.3f}/{new['handoff_offset_ms']:.3f} "
              f"{old['decode_first_token_s']:.9f}/{new['decode_first_token_s']:.9f}")
    print("Requests differing from 21.7 ms by more than their first decode step:",
          [(row["case"], row["request"]) for row in results if row["differs_by_more_than_one_step"]])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
