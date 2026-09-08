"""Validate bounded first-request compute probes without adding parallel ranks."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

from issue26_first_batch_op_rca_vllm import read_rows, write_json


# Multipliers describe calls per decoder layer; zero means one model-level call.
GROUP_SCOPES = {
    "attention": {name: 1 for name in (
        "attn_pre_proj", "attn_rope", "attn_kv_cache_save", "attn_prefill",
        "row_parallel_gemm")},
    "moe": {"moe_gating": 2, "moe_shuffling": 1,
            "moe_grouped_gemm": 1, "moe_sum": 1},
    "detail": {"input_layernorm": 1, "post_attention_layernorm": 1,
               "embedding_compute": 0, "final_layernorm": 0,
               "attn_output_init": 1, "moe_grouped_gemm_w1": 1,
               "moe_activation": 1, "moe_grouped_gemm_w2": 1},
}


def first_request_rows(run, group, request_id, layers, tp_size):
    expected = {name: layers * multiplier if multiplier else 1
                for name, multiplier in GROUP_SCOPES[group].items()}
    first = []
    for path in sorted(run.glob("server.batch.dp*.jsonl")):
        selected = [row for row in read_rows(path) if row.get("op_profile_selected")]
        assert len(selected) == 1, (path, "expected one selected real batch")
        batch = selected[0]
        assert batch["op_profile_selected_index"] == 0, path
        ops_path = path.with_name(path.name.replace("server.batch", "server.ops"))
        ops = read_rows(ops_path)
        assert {row["batch_id"] for row in ops} == {batch["batch_id"]}, ops_path
        if batch["request_ids"] != [request_id]:
            continue
        assert batch["request_num_tokens"] == [4096], batch
        assert batch["batch_num_prefill_tokens"] == 4096, batch
        assert batch["batch_num_decode_tokens"] == 0, batch
        physical_tokens = [1, 1]
        physical_tokens[batch["dp_rank"]] = 4096
        assert batch["batch_dp_token_counts"] == physical_tokens, batch
        assert math.isfinite(batch["batch_execution_time_ms"]) and batch["batch_execution_time_ms"] > 0
        scopes, identities = defaultdict(list), set()
        for row in ops:
            for key in ("dp_rank", "tp_rank", "pp_rank", "batch_size",
                        "batch_num_tokens", "batch_num_prefill_tokens", "batch_num_decode_tokens"):
                assert row[key] == batch[key], (ops_path, key)
            assert row["batch_request_num_tokens"] == batch["request_num_tokens"], ops_path
            assert (row["timing_mode"], row["scope_mode"], row["aggregation_mode"]) == (
                "cuda_event", "default", "per_scope"), ops_path
            assert row["count"] == 1 and math.isfinite(row["cuda_time_ms"]) and row["cuda_time_ms"] > 0, row
            identity = (row["op_name"], row["scope_seq"])
            assert identity not in identities, (ops_path, identity)
            identities.add(identity)
            scopes[row["op_name"]].append(row)
        assert {name: len(rows) for name, rows in scopes.items()} == expected, (ops_path, expected)
        assert identities == {(name, sequence) for name, count in expected.items()
                              for sequence in range(count)}, ops_path
        totals, counts = defaultdict(float), defaultdict(int)
        for name, rows in scopes.items():
            for row in rows:
                logical_name = name
                if name == "moe_gating":
                    logical_name = ("moe_gating_linear" if row["scope_seq"] % 2 == 0
                                    else "moe_gating_routing_topk")
                totals[logical_name] += row["cuda_time_ms"]
                counts[logical_name] += 1
        pairs = []
        if group == "moe":
            sums = {row["scope_seq"]: row["cuda_time_ms"] for row in scopes["moe_sum"]}
            pairs = [{"layer_sequence": row["scope_seq"],
                      "grouped_gemm_ms": row["cuda_time_ms"],
                      "sum_ms": sums[row["scope_seq"]],
                      "complete_expert_compute_ms": row["cuda_time_ms"] + sums[row["scope_seq"]]}
                     for row in sorted(scopes["moe_grouped_gemm"], key=lambda row: row["scope_seq"])]
        first.append({**batch, "group": group, "source": str(ops_path),
                      "scope_totals_ms": dict(totals), "scope_counts": dict(counts),
                      "grouped_gemm_plus_sum_by_layer": pairs,
                      "grouped_gemm_plus_sum_ms": sum(row["complete_expert_compute_ms"] for row in pairs)
                      if pairs else None})
    assert len(first) == tp_size, (run, "missing/duplicate first-request rank", len(first))
    assert {(row["dp_rank"], row["tp_rank"], row["pp_rank"]) for row in first} == {
        (first[0]["dp_rank"], tp, 0) for tp in range(tp_size)}, run
    return first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="GROUP=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-id", default="cmpl-pf4096_dc1024:0-0")
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--tp-size", type=int, default=4)
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.run)
    assert len(runs) == len(args.run), "duplicate group"
    assert runs.keys() <= GROUP_SCOPES.keys(), runs
    ranks = [row for group, path in runs.items() for row in first_request_rows(
        Path(path), group, args.request_id, args.layers, args.tp_size)]
    rows = [{"group": rank["group"], "dp_rank": rank["dp_rank"],
             "tp_rank": rank["tp_rank"], "pp_rank": rank["pp_rank"],
             "batch_id": rank["batch_id"], "op_name": name, "vllm_ms": duration,
             "scope_count": rank["scope_counts"][name], "timing_family": "cuda_event",
             "source": rank["source"]}
            for rank in ranks for name, duration in rank["scope_totals_ms"].items()]
    result = {"status": "PASS_COLLECTION_DIAGNOSTIC_ONLY", "groups": list(runs),
              "first_request_id": args.request_id, "rank_results": ranks,
              "limits": "Groups are separate runs and ranks are parallel. No cross-run scope sum, rank sum, or parent/child addition is performed. GG+sum pairs use sibling scopes from one run/rank/layer; 1 peer token requires scheduler evidence to identify dummy work. Physical DP metadata does not measure the peer's operations. Numerical P/S/V and CUDA-span acceptance remain separate."}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "first_batch_compute.json", result)
    with (args.output / "first_batch_compute.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(result["status"], "groups", list(runs), "first-request rank groups", len(ranks))


if __name__ == "__main__":
    main()
