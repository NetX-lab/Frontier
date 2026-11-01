"""Check whether the NIXL wire-price change can move any READ completion in a replay.

With nixl_read_completion=engine_step, a DECODE lane posts a READ when one of
its engine steps starts and finds it at the first later step start at or after
the READ's wire end (vllm_v1_nixl_reads.NixlReadCompletion). In the run-ahead DP
busy loop a step starts when the host forms the forward (`formed_s`). So the
wire-price change moves a completion only if, after some step start s on the
target lane, another step starts within [s + old wire time, s + new wire time].
An idle wake at a READ's wire end would also place a step start exactly there.

This prices both wire times with the run's own link settings and the committed
predictor, then counts such step-start pairs inside every KV transfer window.

Usage: nixl_window_check.py <output json> <run root> [<run root> ...]
"""

import bisect
import collections
import gzip
import json
import sys
from pathlib import Path

from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.types import ClusterType

HERE = Path(__file__).resolve().parent
BOUNDARY_PROMPT = 6144


def transfer_times(config, prompt_tokens):
    """Old (structural) and new (wire) transfer ms for one request at the run's link settings."""
    row = next(row for row in json.loads((HERE / "nixl_boundaries.json").read_text())["rows"]
               if row["prompt"] == prompt_tokens)
    link = config["kv_cache_transfer_config"]
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig(
        network_bandwidth_gbps=link["network_bandwidth_gbps"],
        network_latency_ms=link["network_latency_ms"],
        enable_compression=link["enable_compression"],
        compression_ratio=link["compression_ratio"],
        kv_connector=link["kv_connector"],
        nixl_read_completion=link["nixl_read_completion"],
    ))

    def price(size):
        return predictor.get_transfer_time(ClusterType.PREFILL, ClusterType.DECODE, None, size)

    return dict(structural_bytes=row["structural_bytes"], wire_bytes=row["wire_bytes"],
                old_ms=price(row["structural_bytes"]), new_ms=price(row["wire_bytes"]))


def step_starts(run_root):
    """Sorted distinct step-start (host formation) times per DECODE lane."""
    lanes = collections.defaultdict(set)
    with gzip.open(run_root / "forwards.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["role"] == "decode":
                lanes[(row["replica"], row["dp"])].add(row["formed_s"])
    return {lane: sorted(times) for lane, times in lanes.items()}


def check_run(run_root):
    run_root = Path(run_root)
    receipt = json.loads((run_root / "receipt.json").read_text())
    metrics = Path(receipt["metrics_dir"])
    config = json.loads((metrics / "config.json").read_text())
    requests =[json.loads(line) for line in (metrics / "metrics_ground_truth.jsonl").read_text().splitlines()]
    prompts = sorted({row["num_prefill_tokens"] for row in requests if row["event_type"] == "request_arrival"})
    assert prompts == [BOUNDARY_PROMPT], prompts
    times = transfer_times(config, BOUNDARY_PROMPT)
    old_s, new_s = times["old_ms"] * 1e-3, times["new_ms"] * 1e-3
    starts = step_starts(run_root)
    ledger = [json.loads(line) for line in (metrics / "frontier_transfer_ledger.jsonl").read_text().splitlines()]
    kv = [row for row in ledger if row["transfer_kind"] == "kv_cache"]
    hits, positive_gaps, starts_checked = [], [], 0
    for row in kv:
        lane = starts[(row["target_replica_id"], row["target_replica_local_id"])]
        first = bisect.bisect_left(lane, row["start_ts_s"])
        last = bisect.bisect_right(lane, row["end_ts_s"])
        for s in lane[first:last]:
            starts_checked += 1
            low = bisect.bisect_left(lane, s + old_s - 1e-12)
            high = bisect.bisect_right(lane, s + new_s + 1e-12)
            hits.extend(dict(transfer_id=row["transfer_id"], post_s=s, step_start_s=t, gap_ms=(t - s) * 1e3)
                        for t in lane[low:high])
            later = bisect.bisect_right(lane, s)
            if later < len(lane):
                positive_gaps.append((lane[later] - s) * 1e3)
    return dict(
        run_root=str(run_root), source_commit=receipt["source_commit"],
        link=config["kv_cache_transfer_config"], transfer_ms=times,
        kv_transfers=len(kv), step_starts_checked=starts_checked,
        min_positive_step_start_gap_ms=min(positive_gaps),
        step_start_gaps_below_new_wire_ms=sum(gap <= times["new_ms"] for gap in positive_gaps),
        window_hits=hits,
        status="NO_COMPLETION_CAN_MOVE" if not hits else "COMPLETIONS_CAN_MOVE",
    )


def main(output, *run_roots):
    document = dict(classification="D-WARM-TWO-FIX_NIXL_STEP_WINDOW_CHECK",
                    runs=[check_run(root) for root in run_roots])
    Path(output).write_text(json.dumps(document, indent=1) + "\n")
    for run in document["runs"]:
        print(run["run_root"], run["transfer_ms"], "transfers", run["kv_transfers"],
              "starts", run["step_starts_checked"], "min gap ms", run["min_positive_step_start_gap_ms"],
              "gaps<=new", run["step_start_gaps_below_new_wire_ms"], "hits", len(run["window_hits"]), run["status"])


if __name__ == "__main__":
    main(*sys.argv[1:])
