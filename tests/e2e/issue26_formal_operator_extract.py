"""Extract the first three formal operator batches from validated H200 logs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


def phase(row):
    if row["batch_num_prefill_tokens"] and row["batch_num_decode_tokens"]:
        return "mixed"
    return "prefill" if row["batch_num_prefill_tokens"] else "decode"


def extract(run, validation_path, contract, output):
    started = time.monotonic()
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "PASS" and validation["mode"] == "ops"
    assert Path(validation["source"]).resolve() == run.resolve()
    clients = [json.loads(line) for line in (run / "client.jsonl").read_text().splitlines()]
    formal = {row["response_id"] + "-0": row["request_id"] for row in clients
              if not row["request_id"].startswith("warmup:")}
    assert len(formal) == 100 and contract.is_file()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"case_id": "pf4096_dc1024", "generation": run.parent.parent.name,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(run),
               "validation": str(validation_path), "scope_contract": str(contract),
               "client_source": str(run / "client.jsonl"), "python": sys.version,
               "command": [sys.executable, *sys.argv], "workers": [],
               "limits": "Worker-local formal samples only; no cross-DP round alignment, Frontier batch comparability, or latency gap claim. Retained scope sums cover only recorded forward intervals and are not CPU overhead."}
    nested = {"expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine"}
    for path in sorted(run.glob("server.batch.dp*.tp*.pp*.jsonl")):
        batches = {}
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if set(row["request_ids"]) & formal.keys():
                    batches[row["batch_id"]] = row
                if len(batches) == 3:
                    break
        assert len(batches) == 3 and {phase(row) for row in batches.values()} == {"prefill", "mixed", "decode"}
        selected = {key: {**row, "local_phase": phase(row), "client_request_ids": [formal[r] for r in row["request_ids"]],
                          "scope_sum_ms": Counter(), "scope_counts": Counter()} for key, row in batches.items()}
        source = path.with_name(path.name.replace("server.batch.", "server.ops."))
        raw_path = output / source.name
        prefixes = tuple(b'{"batch_id": ' + str(key).encode() + b',' for key in batches)
        last_prefix = prefixes[-1]
        seen_last, scanned, parsed, byte_count = False, 0, 0, 0
        sequences = {key: {} for key in batches}
        print(f"Extracting {source.name}: batches {list(batches)}", flush=True)
        with source.open("rb") as stream, raw_path.open("xb") as saved:
            for raw in stream:
                scanned += 1
                byte_count += len(raw)
                if seen_last and not raw.startswith(last_prefix):
                    break
                if not raw.startswith(prefixes):
                    continue
                row = json.loads(raw)
                key, op, seq = row["batch_id"], row["op_name"], row["scope_seq"]
                seen_last = key == max(batches)
                assert all(row[field] == batches[key][field] for field in ("dp_rank", "tp_rank", "pp_rank"))
                group = ("moe_gating_routing_topk" if seq % 2 else "moe_gating_linear") if op == "moe_gating" else op
                if op in nested:
                    group += "_inner" if seq % 2 else "_outer"
                selected[key]["scope_sum_ms"][group] += row["cuda_time_ms"]
                selected[key]["scope_counts"][group] += 1
                sequences[key].setdefault(op, set()).add(seq)
                saved.write(raw)
                parsed += 1
        assert seen_last
        for key, row in selected.items():
            assert all(sequences[key][op] == set(range(96)) for op in nested | {"moe_gating"})
            row["retained_scope_sum_ms"] = sum(value for op, value in row["scope_sum_ms"].items() if op != "add" and not op.endswith("_inner"))
        receipt["workers"].append({"source_batches": str(path), "source_ops": str(source), "raw_selected_ops": str(raw_path),
                                   "scanned_lines": scanned, "scanned_bytes": byte_count, "parsed_rows": parsed, "batches": list(selected.values())})
        print(f"Selected {parsed} rows from {scanned} lines", flush=True)
    assert len(receipt["workers"]) == 8
    receipt.update(status="PASS", elapsed_seconds=time.monotonic() - started, completed_utc=datetime.now(timezone.utc).isoformat())
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "receipt": str(output / "receipt.json"), "elapsed_seconds": receipt["elapsed_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "validation", "contract", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    extract(args.run.resolve(), args.validation.resolve(), args.contract.resolve(), args.output.resolve())
