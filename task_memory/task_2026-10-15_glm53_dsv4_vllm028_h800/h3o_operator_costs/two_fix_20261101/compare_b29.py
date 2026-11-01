"""Compare the D-WARM-TWO-FIX replays with the B29 replays they repeat.

Only the NIXL wire price changed between B29 (a69a5213) and the replay source,
so this checks that the inputs are the same, measures how the KV transfers,
request latencies and P-steps moved, and lists every reading's metric beside
its B29 value.

Usage: compare_b29.py <output directory>
"""

import csv
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
H3O = HERE.parent
CASES = ("p2_32", "p2_64")
B29_RUN = "/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/{case}_warm_k19b29"
B29_GATE = H3O / "codex_resume_20261031/{case}_warm_k19b29/reading.json"
B29_READING = H3O / "warm_target_20261101/readings/B29_{case}_{kind}_reading.json"
NEW_READING = HERE / "readings/{case}_{kind}_reading.json"
LIMIT = 0.1
FORWARD_FIELDS = ("role", "replica", "dp", "stage_id", "batch_stage_id", "start_s", "end_s",
                  "composed_ms", "request_ids", "step", "mode")
REQUEST_FIELDS = ("ttft", "ttft_kv_transfer", "transfer_kv_cache", "tpot", "request_e2e_time")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = list(values)
    return dict(count=len(values), mean=statistics.fmean(values), min=min(values), max=max(values),
                zero=sum(value == 0 for value in values))


def normalized(value, root):
    if isinstance(value, dict):
        return {key: normalized(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized(item, root) for item in value]
    return value.replace(root, "<run_root>") if isinstance(value, str) else value


def input_identity(receipt):
    """Everything in a receipt that fixes the replay inputs, without run paths."""
    root = receipt["run_root"]
    return dict(
        input_sha256=receipt["input_sha256"],
        trace_sha256=receipt["trace"]["sha256"],
        tables=sorted((Path(t["path"]).name, t["sha256"], t["rows"], tuple(t["consumers"]))
                      for t in receipt["profiling_tables"]),
        dummy=receipt["a1"]["enable_dummy_mode"],
        cc_backend_types=receipt["a1"]["cc_backend_types"],
        kv_admission=receipt["kv_admission"],
        added_fields=receipt["added_fields"],
        override_diff=normalized(receipt["override_diff"], root),
    )


def forwards(path):
    with gzip.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    return {role: [tuple(json.dumps(row.get(key), sort_keys=True) for key in FORWARD_FIELDS)
                   for row in rows if row["role"] == role] for role in ("prefill", "decode")}


def forward_comparison(b29, new):
    result = {}
    for role in ("prefill", "decode"):
        old, cur = b29[role], new[role]
        first = next((i for i, (a, b) in enumerate(zip(old, cur)) if a != b), None)
        result[role] = dict(b29_rows=len(old), new_rows=len(cur), identical=old == cur,
                            first_difference_index=first)
    return result


def ledger(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return {row["transfer_id"]: row for row in rows if row["transfer_kind"] == "kv_cache"}


def ledger_comparison(b29, new):
    assert set(b29) == set(new), "KV transfer sets differ"
    ids = sorted(b29)
    return dict(
        transfers=len(ids),
        bytes_equal=all(b29[i]["bytes"] == new[i]["bytes"] for i in ids),
        bytes=sorted({new[i]["bytes"] for i in ids}),
        start_delta_ms=summary((new[i]["start_ts_s"] - b29[i]["start_ts_s"]) * 1e3 for i in ids),
        end_delta_ms=summary((new[i]["end_ts_s"] - b29[i]["end_ts_s"]) * 1e3 for i in ids),
        duration_delta_ms=summary(new[i]["duration_ms"] - b29[i]["duration_ms"] for i in ids),
    )


def request_metrics(path):
    with open(path, newline="") as stream:
        return {row["Request Id"]: {key: float(row[key]) for key in REQUEST_FIELDS}
                for row in csv.DictReader(stream)}


def request_comparison(b29, new):
    assert set(b29) == set(new), "Request sets differ"
    return {key: summary(new[r][key] - b29[r][key] for r in sorted(b29)) for key in REQUEST_FIELDS}


def metric_rows(case, reading, b29, new):
    """One row per (scope, metric), pairing the B29 and new frontier values."""
    rows = []
    for scope, old_metrics, new_metrics in scopes(b29, new):
        for old, cur in zip(old_metrics, new_metrics):
            assert old["metric"] == cur["metric"] and old["vllm"] == cur["vllm"], (case, reading, scope)
            rows.append(dict(
                case=case, reading=reading, scope=scope, metric=cur["metric"], native=cur["vllm"],
                b29_frontier=old["frontier"], new_frontier=cur["frontier"],
                frontier_delta=cur["frontier"] - old["frontier"],
                b29_signed_error=(old["frontier"] - old["vllm"]) / old["vllm"],
                new_signed_error=(cur["frontier"] - cur["vllm"]) / cur["vllm"],
                b29_status=old["status"], new_status=cur["status"],
                status_recomputed=("PASS" if abs(cur["frontier"] - cur["vllm"]) / cur["vllm"] <= LIMIT
                                   else "FAIL"),
            ))
    return rows


def scopes(b29, new):
    yield "pooled", b29["metrics"], new["metrics"]
    for side in b29["repetitions"]:
        yield side.rsplit("-", 1)[-1], b29["repetitions"][side]["metrics"], new["repetitions"][side]["metrics"]


def compare_case(case):
    record = json.loads((HERE / f"{case}_command.json").read_text())
    assert record["status"] == "FINISHED", (case, record["status"])
    b29_root, new_root = Path(B29_RUN.format(case=case)), Path(record["run_root"])
    b29_receipt = json.loads((b29_root / "receipt.json").read_text())
    new_receipt = json.loads((new_root / "receipt.json").read_text())
    b29_inputs, new_inputs = input_identity(b29_receipt), input_identity(new_receipt)
    differing = sorted(key for key in b29_inputs if b29_inputs[key] != new_inputs[key])
    b29_metrics, new_metrics = Path(b29_receipt["metrics_dir"]), Path(new_receipt["metrics_dir"])

    gate_b29 = json.loads(Path(str(B29_GATE).format(case=case)).read_text())
    gate_new = json.loads((Path(record["gate_root"]) / "reading.json").read_text())
    readings = {"all_request": (gate_b29, gate_new)}
    for kind in ("warm", "empty"):
        old = json.loads(Path(str(B29_READING).format(case=case, kind=kind)).read_text())
        cur = json.loads(Path(str(NEW_READING).format(case=case, kind=kind)).read_text())
        assert cur["empty_set_reproduction"] == "PASS_EXACT", (case, kind)
        assert json.loads(cur["all_request_verbatim"]) == gate_new, (case, kind)
        readings[kind] = (old["warm"], cur["warm"])
    rows = [row for kind, (old, cur) in readings.items() for row in metric_rows(case, kind, old, cur)]
    verdicts = {kind: dict(b29=old["verdict"], new=cur["verdict"]) for kind, (old, cur) in readings.items()}

    result = dict(
        source_commit=dict(b29=b29_receipt["source_commit"], new=new_receipt["source_commit"]),
        receipt_sha256=dict(b29=sha256(b29_root / "receipt.json"), new=sha256(new_root / "receipt.json")),
        run_wall_s=dict(b29=b29_receipt["run_wall_s"], new=new_receipt["run_wall_s"]),
        input_identity_equal=not differing, differing_input_fields=differing,
        stage_end_events=dict(b29=b29_receipt["a11"]["stage_end_events"], new=new_receipt["a11"]["stage_end_events"]),
        stale_stage_end_events=dict(b29=b29_receipt["a11"]["stale_stage_end_events"],
                                    new=new_receipt["a11"]["stale_stage_end_events"]),
        collective_calls=dict(b29=b29_receipt["a11"]["collectives"]["calls"],
                              new=new_receipt["a11"]["collectives"]["calls"]),
        collectives_identical=sha256(b29_root / "collectives.json") == sha256(new_root / "collectives.json"),
        forwards=forward_comparison(forwards(b29_root / "forwards.jsonl.gz"), forwards(new_root / "forwards.jsonl.gz")),
        kv_transfers=ledger_comparison(ledger(b29_metrics / "frontier_transfer_ledger.jsonl"),
                                       ledger(new_metrics / "frontier_transfer_ledger.jsonl")),
        requests=request_comparison(request_metrics(b29_metrics / "request_metrics.csv"),
                                    request_metrics(new_metrics / "request_metrics.csv")),
        verdicts=verdicts,
        status_recomputation_matches=all(row["new_status"] == row["status_recomputed"] for row in rows),
        sources=dict(
            gate=dict(b29=str(B29_GATE).format(case=case), new=str(Path(record["gate_root"]) / "reading.json")),
            readings={kind: dict(b29=str(B29_READING).format(case=case, kind=kind),
                                 new=str(NEW_READING).format(case=case, kind=kind)) for kind in ("warm", "empty")},
        ),
    )
    return result, rows


def main(output):
    output = Path(output)
    output.mkdir(exist_ok=True)
    cases, rows = {}, []
    for case in CASES:
        cases[case], case_rows = compare_case(case)
        rows += case_rows
    document = dict(classification="D-WARM-TWO-FIX_B29_COMPARISON", relative_error_limit=LIMIT, cases=cases)
    (output / "comparison.json").write_text(json.dumps(document, indent=1) + "\n")
    with open(output / "metrics.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for case, result in cases.items():
        print(case, "inputs equal:", result["input_identity_equal"], "verdicts:", result["verdicts"])
        print("  forwards:", result["forwards"])
        print("  kv:", result["kv_transfers"])
    for row in rows:
        print(f"{row['case']} {row['reading']:<11} {row['scope']:<8} {row['metric']:<22} "
              f"native={row['native']:.6f} b29={row['b29_frontier']:.6f} new={row['new_frontier']:.6f} "
              f"err {row['b29_signed_error']:+.6%} -> {row['new_signed_error']:+.6%} {row['new_status']}")


if __name__ == "__main__":
    main(sys.argv[1])
