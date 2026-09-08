"""Extract the first completed prefill's operator and EP accounting evidence."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re


OP = re.compile(r"\[OP-TRACE\]\[MONOLITHIC\]\[(\w+)\]\[(\w+)\] batch_id=(\d+), layer_id=(\d+), predicted_time_ms=([\d.]+)")
FIELD = re.compile(r"(\w+)=([^,]+)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ops = {}
    ep = []
    waves = []
    prefix = []
    last_moe_batch = None
    bytes_read = 0
    with args.log.open() as stream:
        for number, line in enumerate(stream, 1):
            prefix.append(line)
            bytes_read += len(line)
            if bytes_read > 32 * 1024 * 1024:
                raise ValueError("First-prefill trace exceeded its bounded read budget")
            match = OP.search(line)
            if match:
                family, name, batch, layer, value = match.groups()
                key = (family, name, int(batch), int(layer))
                if key in ops:
                    assert ops[key]["predicted_ms"] == float(value), key
                    ops[key]["occurrences"] += 1
                else:
                    ops[key] = {"family": family, "op_name": name, "batch_id": int(batch),
                                "layer_id": int(layer), "predicted_ms": float(value),
                                "source_line": number, "occurrences": 1}
                if family == "MOE":
                    last_moe_batch = int(batch)
            if "[EP-WORKLOAD]" in line:
                fields = dict(FIELD.findall(line))
                fields.update(source_line=number, predictor_batch_id=last_moe_batch)
                ep.append(fields)
            if "[EP-WAVE-END]" in line:
                waves.append(dict(FIELD.findall(line)))
            if "[TOKEN-ROLLOUT] req=0 " in line and "(first decode token after prefill)" in line:
                break
        else:
            raise ValueError("First request prefill completion marker was not observed")
    (args.output / "frontier_first_prefill.log").write_text("".join(prefix))
    layers = {int(row["layer_id"]) for row in ep}
    assert len(layers) == 48 and len(ep) == 48 * 8, (len(layers), len(ep))
    ledger = []
    for row in ops.values():
        if row["batch_id"] == 0 and row["family"] in {"ATTENTION", "COMM"}:
            ledger.append(dict(row, lane_id=0, source_kind="attention"))
    for lane in ep:
        batch, layer = lane["predictor_batch_id"], int(lane["layer_id"])
        for key, row in ops.items():
            if key[0] == "MOE" and key[2:] == (batch, layer):
                ledger.append(dict(row, lane_id=int(lane["ep_id"]), source_kind="expert",
                                   ep_source_line=lane["source_line"]))
    grouped = defaultdict(list)
    for row in ledger:
        grouped[(row["op_name"], row["layer_id"])].append(row)
    reduced = []
    for (name, layer), rows in sorted(grouped.items()):
        values = [row["predicted_ms"] for row in rows]
        assert min(values) == max(values), (name, layer, values)
        reduced.append({"op_name": name, "layer_id": layer, "predicted_ms": max(values),
                        "lane_count": len(rows), "source_lines": [row["source_line"] for row in rows]})
    totals = defaultdict(float)
    for row in reduced:
        totals[row["op_name"]] += row["predicted_ms"]
    # Aliases and child scopes are retained in rows but excluded from accounting.
    excluded = {"moe_gating", "add", "add_attn_residual", "add_ffn_residual",
                "moe_tp_allreduce", "share_expert_tp_allreduce"}
    op_total = sum(value for name, value in totals.items() if name not in excluded)
    comm_total = sum(float(row["dispatch_ms"]) + float(row["combine_ms"])
                     + float(row["post_combine_ms"]) for row in ep if row["ep_id"] == "0")
    assert len(waves) == 48
    endpoint_ms = float(waves[-1]["wave_end_time_s"]) * 1000
    rounding_budget_ms = (len(reduced) + len(layers) * 3) * 0.5e-6
    assert abs(op_total + comm_total - endpoint_ms) <= rounding_budget_ms
    summary = {"endpoint_ms": endpoint_ms, "log_rounding_budget_ms": rounding_budget_ms,
               "status": "PASS_FRONTIER_FIRST_PREFILL_ACCOUNTING", "source": str(args.log),
               "layer_count": len(layers), "ep_lane_rows": len(ep),
               "op_totals_ms": dict(totals), "excluded_from_sum": sorted(excluded),
               "operator_sum_ms": op_total, "ep_communication_ms": comm_total,
               "total_predicted_ms": op_total + comm_total,
               "limits": "Current first-request diagnostic run; equal EP lane predictions reduced once. Rounded log values. No complete-request E2E claim."}
    (args.output / "frontier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "frontier_operator_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in ledger))
    (args.output / "frontier_op_reduced.json").write_text(json.dumps(reduced, indent=2) + "\n")
    (args.output / "frontier_ep_rows.json").write_text(json.dumps(ep, indent=2) + "\n")
    with (args.output / "frontier_op_totals.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["op_name", "predicted_ms", "included_in_accounting"])
        writer.writerows((name, value, name not in excluded) for name, value in sorted(totals.items()))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
