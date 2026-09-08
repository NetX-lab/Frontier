"""Export an isolated first-forward sensitivity dataset from complete event samples."""

import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_key = "random_forrest_execution_time_predictor_config_linear_op_input_file"
    source = Path(config[source_key])
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    original = [dict(row) for row in rows]
    receipt = json.loads((args.samples_dir / "receipt.json").read_text())
    samples = json.loads((args.samples_dir / "event_samples.json").read_text())
    assert receipt["status"] == "COMPLETE_DIAGNOSTIC", receipt["status"]
    assert receipt["dtype"] == "torch.bfloat16" and receipt["inference_mode"]
    assert config["replica_config_attn_tensor_parallel_size"] == receipt["tp"]
    assert receipt["tokens"] == 4096 and receipt["tp"] == 4
    assert config["decode_cuda_graph_mode"] == "none"
    assert config["vllm_v1_scheduler_config_batch_size_cap"] == 1024
    matched = [i for i, row in enumerate(rows)
               if int(row["num_tokens"]) == receipt["tokens"]
               and int(row["num_tensor_parallel_workers"]) == receipt["tp"]]
    assert len(matched) == 1, matched
    row = rows[matched[0]]
    assert row["measurement_type"] == "CUDA_EVENT"
    assert row["profiling_precision"] == "BF16" and row["quant_signature"] == "none"
    for field, key in (("n_embd", "hidden"), ("n_head", "q_heads"), ("n_kv_head", "kv_heads")):
        assert int(row[field]) == receipt[key]
    contracts = json.loads(row["typed_operator_contracts"])
    operations = ("attn_pre_proj", "attn_rope", "attn_post_proj")
    changes = {}
    for op in operations:
        assert contracts[op] == receipt["profiling_plan"]["typed_operator_contracts"][op]
        selected = [sample for sample in samples if sample["op"] == op
                    and sample["context"] == "prefill_hot" and sample["timer"] == "original"]
        assert sorted(sample["round"] for sample in selected) == [0, 1]
        assert all(sample["calls_per_iteration"] == 1 and len(sample["samples_ms"]) == 20
                   for sample in selected)
        values = [value for sample in selected for value in sample["samples_ms"]]
        assert len(values) == 40 and all(value > 0 for value in values)
        stats = dict(min=min(values), max=max(values), mean=statistics.mean(values),
                     median=statistics.median(values), std=statistics.pstdev(values), count=len(values))
        changes[op] = dict(previous_median_ms=float(row[f"time_stats.{op}.median"]),
                           samples_ms=values, stats=stats)
        for stat, value in stats.items():
            row[f"time_stats.{op}.{stat}"] = str(value)
    changed_cells = [(i, col) for i, (before, after) in enumerate(zip(original, rows))
                     for col in columns if before[col] != after[col]]
    assert all(i == matched[0] and any(col.startswith(f"time_stats.{op}.") for op in operations)
               for i, col in changed_cells)
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = args.output / "linear_op_candidate.csv"
    with candidate.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    config[source_key] = str(candidate.resolve())
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    evidence = dict(status="DIAGNOSTIC_CANDIDATE_ONLY", source_config=str(args.config.resolve()),
                    source_csv=str(source), source_samples=str(args.samples_dir.resolve()),
                    source_receipt=receipt, row_count=len(rows), matched_csv_line=matched[0] + 2,
                    aggregation="All 40 event-only samples per op, pooled median; population std.",
                    context="prefill_hot", timer="original", changed_cells=changed_cells,
                    operations=changes, typed_operator_contracts=contracts,
                    cache_policy="Reuse current-task cache; normal dataframe content keys select changed models.",
                    limits="Only the bounded first forward is supported; no general runtime context selector.")
    (args.output / "export_receipt.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": evidence["status"], "changed_cells": len(changed_cells),
                      "medians_ms": {op: data["stats"]["median"] for op, data in changes.items()}}))


if __name__ == "__main__":
    main()
