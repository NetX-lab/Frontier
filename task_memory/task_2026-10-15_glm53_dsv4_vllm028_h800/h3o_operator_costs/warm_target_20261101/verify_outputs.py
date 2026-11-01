"""Check paired exclusions, PD-POOL totals, immutable sources, and report values."""

import csv
import hashlib
import json
import math
from pathlib import Path

from tests.comparison.e2e_gate.gate import metric_values, read_rows

OUT = Path(__file__).resolve().parent
READINGS = OUT / "readings"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


evidence = read(OUT / "openers.json")
for run in evidence["runs"].values():
    for source in run["sources"]:
        assert sha(Path(source["path"])) == source["sha256"]
checks = []
for path in sorted(READINGS.glob("*_warm_reading.json")):
    document = read(path)
    reading_path = next(Path(s["path"]) for s in document["sources"] if s["sha256"] == document["reading_sha256"])
    assert document["all_request_verbatim"].encode() == reading_path.read_bytes()
    for source in document["sources"]:
        assert sha(Path(source["path"])) == source["sha256"]
    rows_dir = next(Path(s["path"]).parent for s in document["sources"] if s["path"].endswith("sample.json"))
    sample, sides = read_rows(rows_dir)
    [member] = [s for s in sides if s.startswith("frontier/")]
    totals = {"vllm": {}, "frontier": {}}
    for side, group in document["warm"]["repetitions"].items():
        excluded = document["openers"][side]
        filtered = {s: frame.loc[~frame.index.isin(excluded)] for s, frame in sides.items() if s in (side, member)}
        assert set(filtered[side].index) == set(filtered[member].index)
        assert len(filtered[side]) + len(excluded) == 128
        for metric in group["metrics"]:
            m = metric["metric"]
            for name, frame in (("vllm", filtered[side]), ("frontier", filtered[member])):
                numerator, denominator, eligible = metric_values(frame)[m]
                assert metric[f"{name}_numerator"] == numerator and metric[f"{name}_denominator"] == denominator
                assert metric[name] == numerator / denominator
                totals[name].setdefault(m, []).append((numerator, denominator, eligible))
    for metric in document["warm"]["metrics"]:
        m = metric["metric"]
        for name in ("vllm", "frontier"):
            numerator, denominator, eligible = tuple(map(sum, zip(*totals[name][m])))
            assert metric[f"{name}_numerator"] == numerator and metric[f"{name}_denominator"] == denominator
            assert math.isclose(metric[name], numerator / denominator, rel_tol=1e-15)
        assert metric["sample_count"] == 384
        assert metric["eligible_count"] == document["warm"]["retained_request_repeat_count"]
        assert metric["excluded_count"] == 384 - metric["eligible_count"]
    assert document["warm"]["verdict"] == "PASS"
    checks.append(dict(path=str(path), sha256=sha(path), paired_exclusion="PASS", pooled_totals="PASS",
                       unchanged_sources="PASS", verbatim_harness_bytes="PASS"))
assert len(checks) == 6
with (READINGS / "metrics.csv").open() as stream:
    metrics = list(csv.DictReader(stream))
assert len(metrics) == 192
assert sum(row["status"] == "FAIL" for row in metrics if row["reading"] == "all_request") == 3
assert all(row["status"] == "PASS" for row in metrics if row["reading"] == "warm")
for row in metrics:
    native, frontier = float(row["native"]), float(row["frontier"])
    assert float(row["signed_error"]) == frontier - native
    assert float(row["absolute_error"]) == abs(frontier - native)
    assert float(row["signed_relative_error"]) == (frontier - native) / native
    assert float(row["absolute_relative_error"]) == abs(frontier - native) / native
    assert row["status"] == ("PASS" if abs(frontier - native) / native <= 0.1 else "FAIL")
results = dict(status="PASS", checks=checks, metric_rows=192, warm_metric_pass=96, original_metric_fail=3)
with (OUT / "output_verification.json").open("x") as stream:
    stream.write(json.dumps(results, indent=2) + "\n")

lines = ["# WARM-READING verification and metric report", "", "## Modification History", "",
         "| Date | Change |", "| --- | --- |", "| 2026-11-01 | Verified D-WARM-TARGET retained-reference readings. |", "",
         "Task ID: WARM-READING. CPU only; no new native or Frontier run, GPU operation, or commit.", "",
         "## Outcome", "",
         "All six warm readings pass all four metrics in all three native repeats and in the pooled reading (96/96 PASS). "
         "The unchanged harness has three metric FAILs: P2-64 repeat01 TTFT, one in each generation. "
         "Six empty-exclusion checks reproduce the original pooled/per-repeat values exactly, with unchanged signed error and verdict. "
         "The all-request JSON is retained as both a parsed object and byte-exact text with SHA-256.", "",
         "Authoritative outputs are under `readings/`. Root-level first-iteration reading JSONs/metrics.csv remain unchanged as history; "
         "their no-op Frontier totals used the equivalent harness representation. The final generation retains summed totals on both sides "
         "also for an empty exclusion set, while keeping the original binary-float ratio rounding (differences otherwise at about 1e-13).", "",
         "## Mechanism and identification", "",
         "Source: vLLM `pull_worker.py:65-77` starts the first remote-engine handshake before a KV read; "
         "`base_worker.py:686-708` logs compatibility once per TP worker during handshake; `base_worker.py:879-905` caches "
         "the remote engine and shares pending handshake futures; `base_worker.py:508-509` defaults engine TTL to 3600 s.", "",
         "`identify_openers.py` reads decode request ownership from `nodes/*/serving/rank0.json` "
         "(`inventory.request_owner` + `counters.request_ids`) and the remote prefill engine from proxy KV-transfer fields. "
         "For each decode lane, the nth compatibility line of each of the eight TP workers forms one cohort. "
         "The complete cohort's one-second timestamp bins must each overlap the same unique same-lane proxy handoff "
         "(prefill response -> decode first byte), and the request must be the pair's first request. "
         "Every formal event binds uniquely; no latency threshold selects any request.", "",
         "The source events log the same compatibility hash for all engines, not the remote engine ID. "
         "Therefore pair identity is supplied by the independently retained request-owner inventory and proxy record, "
         "not by the hash. All TP workers have four events per D lane for P2-64 and two for P2-32.", "",
         "Clock mapping: server civil timestamps are UTC (offset 0), anchored to the corresponding server-start log "
         "and `server_ready.json.ready_at_utc`; the latest start log precedes readiness by less than 5 s. "
         "`canonical_repetitions.json` records origin_wall/origin_monotonic per phase. Proxy wall-minus-monotonic offsets "
         "agree with those origins within 1 ms in every run. Exact anchors, observed ranges, phase windows, event bins, "
         "server log paths/line numbers, and request handoff endpoints are in `openers.json`. "
         "The server log resolution is 1 s; no sub-second handshake timestamp or duration is asserted.", "",
         "Warmup cohorts can overlap multiple warmup handoffs. Those candidate lists are preserved without asserting "
         "a unique request/remote-engine assignment. Their window classification is unambiguous, all warmups have drained "
         "before formal admission, and no warmup request is included in or removed from the formal population. "
         "The formal first-contact pair count independently agrees with the formal event count.", "",
         "### Event and opener table", "",
         "| Run | Repeat | Warmup events | Formal events | Formal opener IDs | Validation |", "| --- | --- | ---: | ---: | --- | --- |"]
for label, run in evidence["runs"].items():
    for repeat in sorted({w["repeat"] for w in run["windows"]}):
        events = [e for e in run["events"] if e["repeat"] == repeat]
        ids = [r.split("formal-")[1].split("-repeat")[0] for r in run["formal_openers"] if r.endswith(f"repeat{repeat:02d}")]
        lines.append(f"| {label} | {repeat:02d} | {sum(e['phase'] == 'warmup' for e in events)} | "
                     f"{sum(e['phase'] == 'formal' for e in events)} | {', '.join(ids) or 'none'} | {run['validation']['status']} |")
lines += ["", "### Unique formal bindings", "",
          "| Run | Request | P lane | D lane | Log interval UTC | Proxy handoff (ms) |", "| --- | --- | ---: | ---: | --- | ---: |"]
for label, run in evidence["runs"].items():
    for event in run["events"]:
        if event["bound_request_id"]:
            rid = event["bound_request_id"].split("formal-")[1]
            lines.append(f"| {label} | {rid} | {event['prefill_lane']} | {event['decode_lane']} | "
                         f"{event['interval_utc'][0]} to {event['interval_utc'][1]} | {event['proxy_handoff_ms']:.12g} |")
lines += ["", "## Commands and acceptance", "",
          "Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (exact version in receipts/verification.json); "
          "`PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH=$PWD:$PWD/tests/analysis`; "
          "`ulimit -v 31457280` inherited by all child commands. All commands ran at the assigned repository root.", "",
          "1. `python <evidence>/identify_openers.py --run LABEL=ARTIFACTS ... --output <evidence>/openers.json`: "
          "PASS; clean32=4 warmup/0 formal, clean64=11/5, lifecycle=11/5 and wave=13/3; "
          "exact lifecycle ground-truth opener reproduction on both instrumented runs.",
          "2. Twelve final `python tests/analysis/e2e_warm_reading.py --rows ROWS --reading ORIGINAL --openers EVIDENCE "
          "--output NEW_JSON` calls through `run_analysis.py`: PASS; six originals and six empty sets exactly reproduce values; "
          "P2-32 retains 384 request-repeat rows, P2-64 retains 379 (repeat01=123, repeats02/03=128).",
          "3. `python <evidence>/verify_outputs.py`: PASS; independent paired-row/counter checks, summed pooling totals on both sides, "
          "verbatim source bytes, SHA-256 of every source, CSV arithmetic/units/verdicts, 192 rows, 96 warm PASS / 3 all-request FAIL.", "",
          "Full commands, cwd, environment, UTC start/end, exit code, git HEAD, dirty paths and output SHA-256 are in "
          "`command_receipts.json`. The initial repository already had unrelated dirty requirements.md and untracked ".replace("untracked ", "untracked `" )
          + ".cortexkit/`, `.pi/`, and `remaining_measurements/`; those were not edited.", "",
          "B29 rows were found in the retained `h3o_b/stage4/codex_resume_20261031/p2_{32,64}_warm_k19b29/rows`; "
          "their reading hashes exactly match the task-retained B29 readings. No normalization or replay was needed.", "",
          "## Complete metric table", "",
          "CSV `readings/metrics.csv` contains exact Python float representations and both signed/absolute relative errors. "
          "Below: all-request and warm absolute values, signed relative error in %, and metric PASS/FAIL; "
          "absolute relative error is the magnitude of the signed column. Displayed values are rounded to 12 significant digits; "
          "JSON/CSV retain exact precision. No confidence intervals.", "",
          "| Generation | Case | Repeat | Metric (unit) | All native | All Frontier | All signed error % | All | "
          "Warm native | Warm Frontier | Warm signed error % | Warm |", "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
warm = {(r["generation"], r["case"], r["repeat"], r["metric"]): r for r in metrics if r["reading"] == "warm"}
for a in metrics:
    if a["reading"] != "all_request":
        continue
    w = warm[a["generation"], a["case"], a["repeat"], a["metric"]]
    repeat = a["repeat"] if a["repeat"] == "pooled" else a["repeat"].rsplit("-", 1)[1]
    lines.append(f"| {a['generation']} | {a['case']} | {repeat} | {a['metric']} ({a['unit']}) | "
                 f"{float(a['native']):.12g} | {float(a['frontier']):.12g} | {float(a['signed_relative_error'])*100:.12g} | {a['status']} | "
                 f"{float(w['native']):.12g} | {float(w['frontier']):.12g} | {float(w['signed_relative_error'])*100:.12g} | {w['status']} |")
lines += ["", "## Limits", "",
          "This is a paired request-population reading of retained runs, not a replay of the system after stronger warmup. "
          "It does not subtract handshake time from other requests, reconstruct counterfactual scheduling, assert exact "
          "handshake durations, or prove full-domain profiling/input qualification. Numerical warm PASS does not close "
          "the remaining formal admission requirements. Only three retained native repeats are reported, without CIs.", ""]
with (OUT / "test_report_2026-11-01_warm_reading.md").open("x") as stream:
    stream.write("\n".join(lines))
print("PASS paired exclusions, pooled totals, immutable source SHA-256, verbatim reading, 192 CSV rows; report written")
