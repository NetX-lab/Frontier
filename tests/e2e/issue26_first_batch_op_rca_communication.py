"""Summarize the bounded communication-only diagnostic without summing ranks."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    scopes = {"expert_parallel_allreduce", "attn_post_proj_tp_allreduce", "tensor_parallel_allreduce"}
    for batch_path in sorted(args.run.glob("server.batch.dp*.jsonl")):
        with batch_path.open() as stream:
            selected = [row for line in stream if (row := json.loads(line)).get("op_profile_selected")]
        assert len(selected) == 3
        assert [row["op_profile_selected_index"] for row in selected] == [0, 1, 2]
        op_path = batch_path.with_name(batch_path.name.replace("server.batch", "server.ops"))
        with op_path.open() as stream:
            ops = [json.loads(line) for line in stream]
        assert {row["batch_id"] for row in ops} == {row["batch_id"] for row in selected}
        assert {row["op_name"] for row in ops} == scopes
        for batch in selected:
            totals, counts = defaultdict(float), defaultdict(int)
            identities = set()
            for row in ops:
                if row["batch_id"] == batch["batch_id"]:
                    for key in ("dp_rank", "tp_rank", "pp_rank", "batch_size", "batch_num_tokens",
                                "batch_num_prefill_tokens", "batch_num_decode_tokens"):
                        assert row[key] == batch[key], (op_path, key, row[key], batch[key])
                    assert row["batch_request_num_tokens"] == batch["request_num_tokens"]
                    assert (row["timing_mode"], row["scope_mode"], row["aggregation_mode"]) == (
                        "cuda_event", "default", "per_scope")
                    assert row["count"] == 1 and math.isfinite(row["cuda_time_ms"]) and row["cuda_time_ms"] > 0
                    identity = (row["op_name"], row["scope_seq"])
                    assert identity not in identities, (op_path, batch["batch_id"], identity)
                    identities.add(identity)
                    totals[row["op_name"]] += row["cuda_time_ms"]
                    counts[row["op_name"]] += 1
            assert counts == {"expert_parallel_allreduce": 48,
                              "attn_post_proj_tp_allreduce": 48,
                              "tensor_parallel_allreduce": 1}
            assert identities == {(name, sequence) for name, count in counts.items()
                                  for sequence in range(count)}
            rows.append(dict(batch, scope_ms=dict(totals), scope_counts=dict(counts), source=str(op_path)))
    assert len(rows) == 24
    assert {(row["dp_rank"], row["tp_rank"], row["pp_rank"], row["op_profile_selected_index"])
            for row in rows} == {(dp, tp, 0, index) for dp in range(2) for tp in range(4) for index in range(3)}
    first = [row for row in rows if row["request_ids"] == ["cmpl-pf4096_dc1024:0-0"]
             and row["request_num_tokens"] == [4096]]
    assert len(first) == 4 and {row["tp_rank"] for row in first} == set(range(4))
    result = {"status": "PASS_COLLECTION_DIAGNOSTIC_ONLY", "rank_batches": len(rows),
              "first_formal_batch": first, "selected_batches": rows,
              "limits": "CUDA-event scope elapsed includes waits and submission gaps. Ranks are parallel participants, never additive. Reduced instrumentation is not a clean E2E metric."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "first": [
        {"tp_rank": row["tp_rank"], "batch_ms": row["batch_execution_time_ms"],
         "scope_ms": row["scope_ms"]} for row in first]}))


if __name__ == "__main__":
    main()
