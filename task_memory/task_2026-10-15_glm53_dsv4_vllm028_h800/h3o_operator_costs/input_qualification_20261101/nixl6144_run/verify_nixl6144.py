"""Verify the NIXL6144 H200 run against the package's Post-GPU acceptance list.

Reads the downloaded artifacts under /data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download
and writes verification.json, nixl_summary.csv, case_summary.csv,
case_pass_spread.csv and paired_6143_6144.csv next to this script.
"""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from frontier.profiling.collectives.nixl_kv_read_benchmark import (
    KVRegionLayout, read_cases, summarize_case)

RUN = Path(__file__).resolve().parent
DOWN = Path("/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download")
WORKSPACE = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028")
STAGE = RUN.parent / "packages/nixl6144/stage_nixl6144.json"
EVIDENCE = WORKSPACE / ("task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/"
                        "h3o_operator_costs/b30b/kv_layout_evidence.json")
LOCK = WORKSPACE / ("task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/"
                    "h3o_operator_costs/g5h/launch/receipts/"
                    "dsv4-h200-nixl6144-q1-20261102_lock.json")

checks = []


def check(name, passed, values, note=None):
    checks.append({"name": name, "pass": bool(passed), "values": values, **(
        {"note": note} if note else {})})
    return passed


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


download = json.loads((DOWN / "download.json").read_text())
plan = json.loads((DOWN / "C_nixl/nixl_plan.json").read_text())
stage = json.loads(STAGE.read_text())
config = plan["config"]
layout = KVRegionLayout.from_config(config["kv_cache"])
pairs = int(config["tensor_parallel_size"])
cases = read_cases(config["grid"], layout, pairs)
expected = {case.name: case for case in cases}
raw = [json.loads(line) for line in
       (DOWN / "C_nixl/nixl_reads.jsonl").read_text().splitlines() if line.strip()]
csv_rows = list(csv.DictReader((DOWN / "C_nixl/nixl_summary.csv").open()))

# --- 0. read-only download integrity -----------------------------------------
check("download_hashes", download["hashes_equal"] and not download["missing_remote"],
      {"files": download["files"], "bytes": download["bytes"],
       "hashes_equal": download["hashes_equal"],
       "missing_remote": download["missing_remote"],
       "remote_nodes": download["remote_nodes"]})

# --- 1. two distinct H200 nodes, 16 ranks, placement, phase exits ------------
node_dirs = sorted(p for p in (DOWN / "nodes").iterdir() if p.is_dir())
placements = {}
for node in node_dirs:
    placement = json.loads((node / "placement.json").read_text())
    placements[node.name] = placement
h200_lines = {host: [line for line in placement["inventory"] if "H200" in line]
              for host, placement in placements.items()}
uuids = {host: placement["allocated_uuids"] for host, placement in placements.items()}
visible = {host: (placement["visible_devices"] or "").split(",") for host, placement in placements.items()}
inventory_counts = {host: len(placement["inventory"]) for host, placement in placements.items()}
all_uuids = [uuid for values in uuids.values() for uuid in values]
placement_ok = (
    len(node_dirs) == 2
    and all(count == 8 for count in inventory_counts.values())
    and all(len(h200_lines[h]) == 8 for h in h200_lines)
    and all(len(uuids[h]) == 8 for h in uuids)
    and all(len(visible[h]) == 8 for h in visible)
    and len(set(all_uuids)) == 16
    and sorted(p["node_rank"] for p in placements.values()) == [0, 1]
)
check("two_distinct_h200_nodes_16_ranks", placement_ok, {
    "nodes": sorted(h for h in placements),
    "node_ranks": {h: p["node_rank"] for h, p in placements.items()},
    "inventory_gpus_per_node": inventory_counts,
    "h200_inventory_lines_per_node": {h: len(v) for h, v in h200_lines.items()},
    "allocated_uuids_per_node": {h: len(v) for h, v in uuids.items()},
    "unique_uuids": len(set(all_uuids)),
    "cuda_visible_devices_per_node": {h: len(v) for h, v in visible.items()},
    "census_returncodes": {h: p["census_returncode"] for h, p in placements.items()},
})

phase_rows = {}
node_results = {}
for node in node_dirs:
    result = json.loads((node / "result.json").read_text())
    node_results[node.name] = result
    for phase in result["phases"]:
        phase_rows.setdefault(node.name, []).append(
            {"phase": phase["phase"], "returncode": phase["returncode"],
             "visible_devices": phase.get("visible_devices"),
             "ucx_net_devices": phase.get("ucx_net_devices"),
             "exception": phase.get("exception"), "skipped": bool(phase.get("skipped"))})
exits_ok = (len(node_results) == 2
            and all(not r["failed"] for r in node_results.values())
            and all(row["returncode"] == 0 for rows in phase_rows.values() for row in rows))
check("phase_exits_zero", exits_ok, {
    "per_node": phase_rows,
    "failed_flags": {h: r["failed"] for h, r in node_results.items()},
})

# TP pairing: 8 P ranks and 8 D ranks; reader rank 8+p reads P rank p.
reader_ranks_ok = all(
    sorted(reader["rank"] for reader in row["readers"]) ==
    list(range(pairs, pairs + expected[row["case"]].active_pairs)) for row in raw)
check("tp_pair_placement", reader_ranks_ok and all(
    step["phase"] == "C_nixl" and "torch.distributed.run" in " ".join(step.get("command") or [])
    and "--nnodes" in (step.get("command") or []) and "--nproc-per-node" in (step.get("command") or [])
    for rows in node_results.values() for step in rows["phases"]),
    {"reader_rank_pairs": "8..15 read from P ranks 0..7",
     "reader_ranks_match": reader_ranks_ok,
     "phase_commands": {h: [step.get("command") for step in r["phases"]]
                        for h, r in node_results.items()}})

# --- 2. runtime checks (soft gate: report every mismatch) -------------------
runtime_reports = {}
for path in sorted((DOWN / "runtime_checks").glob("*.json")):
    runtime_reports[path.stem] = json.loads(path.read_text())
mismatches = [{"host": host, **row} for host, report in runtime_reports.items()
              for row in report["checks"] if not row["pass"]]
runtime_ok = (len(runtime_reports) == 2
              and all(report["returncode"] == 0 for report in runtime_reports.values())
              and not mismatches)
check("runtime_checks", runtime_ok, {
    "hosts": sorted(runtime_reports),
    "gate": {h: r["runtime_check_gate"] for h, r in runtime_reports.items()},
    "workspace_head": {h: r["workspace_head"] for h, r in runtime_reports.items()},
    "checks_passed": {h: f"{sum(row['pass'] for row in r['checks'])}/{len(r['checks'])}"
                      for h, r in runtime_reports.items()},
    "mismatches": mismatches,
}, note="Soft gate: mismatched pinned identities are reported, not enforced.")

# --- 3. plan/config identity ------------------------------------------------
plan_ok = (
    config["passes"] == 3 and int(config["shuffle_seed"]) == int(stage["phases"][0]["config"]["shuffle_seed"])
    and int(config["warmup"]) == 2 and int(config["repeats"]) == 30
    and int(config["tensor_parallel_size"]) == 8
    and layout.block_bytes == 1002240 and len(cases) == 30
    and bool((plan.get("environment") or {}).get("ucx_net_devices"))
)
check("plan_config", plan_ok, {
    "passes": config["passes"], "shuffle_seed": config["shuffle_seed"],
    "warmup": config["warmup"], "repeats": config["repeats"],
    "tensor_parallel_size": config["tensor_parallel_size"],
    "block_bytes": layout.block_bytes, "num_regions": layout.num_regions,
    "cases": len(cases), "ucx_net_devices": (plan.get("environment") or {}).get("ucx_net_devices"),
    "ucx_backend_params": (plan.get("environment") or {}).get("ucx_backend_params"),
})

# --- 4. complete 30-case x 3-pass matrix ------------------------------------
keys = Counter((row["case"], row["pass"]) for row in raw)
expected_keys = Counter((case.name, index) for case in cases for index in range(config["passes"]))
extra = [f"{case}@{index}" for (case, index) in keys if (case, index) not in expected_keys]
missing = [f"{case}@{index}" for (case, index) in expected_keys if (case, index) not in keys]
duplicates = [f"{case}@{index}" for (case, index), count in keys.items() if count > 1]
matrix_ok = len(raw) == 90 and not extra and not missing and not duplicates
check("case_pass_matrix", matrix_ok, {
    "raw_rows": len(raw), "expected_rows": len(expected_keys),
    "distinct_cases": len({row["case"] for row in raw}),
    "distinct_passes": sorted({row["pass"] for row in raw}),
    "missing": missing, "unexpected": extra, "duplicates": duplicates,
})

# --- 5. geometry, trials, notifications, data checks ------------------------
problems = []
trials = 0
notifications_received = 0
notifications_expected = 0
per_record = []
for row in raw:
    case = expected.get(row["case"])
    if case is None:
        problems.append({"case": row["case"], "problem": "unknown case"})
        continue
    expected_blocks = layout.request_blocks(case.prompt_tokens)
    expected_bytes = case.blocks_per_request * case.requests_in_flight * layout.block_bytes
    expected_notifications = (case.active_pairs * case.requests_in_flight
                             * (int(config["warmup"]) + int(config["repeats"])))
    row_problems = []
    if row["prompt_tokens"] != case.prompt_tokens:
        row_problems.append("prompt_tokens mismatch")
    if row["blocks_per_request"] != expected_blocks:
        row_problems.append(f"blocks {row['blocks_per_request']} != {expected_blocks}")
    if row["descriptors_per_request"] != expected_blocks * layout.num_regions:
        row_problems.append("descriptor count mismatch")
    if row["bytes_per_reader"] != expected_bytes:
        row_problems.append(f"bytes {row['bytes_per_reader']} != {expected_bytes}")
    if row["active_pairs"] != case.active_pairs or row["requests_in_flight"] != case.requests_in_flight:
        row_problems.append("case parameters mismatch")
    if row["notifications_expected"] != expected_notifications:
        row_problems.append("notifications_expected mismatch")
    if row["notifications_received"] != expected_notifications:
        row_problems.append("notifications_received mismatch")
    readers = row["readers"]
    if len(readers) != case.active_pairs:
        row_problems.append(f"readers {len(readers)} != {case.active_pairs}")
    if sorted(reader["rank"] for reader in readers) != list(
            range(pairs, pairs + case.active_pairs)):
        row_problems.append("reader ranks mismatch")
    row_trials = 0
    for reader in readers:
        if reader["data_verified"] is not True:
            row_problems.append(f"rank {reader['rank']} data_verified false")
        if len(reader["timings"]) != int(config["repeats"]):
            row_problems.append(f"rank {reader['rank']} timings {len(reader['timings'])}")
        row_trials += len(reader["timings"])
        for timing in reader["timings"]:
            if not all(math.isfinite(timing[key]) for key in ("prep_us", "post_us", "complete_us")):
                row_problems.append(f"rank {reader['rank']} non-finite timing")
                break
            if min(timing.values()) < 0 or timing["complete_us"] <= 0:
                row_problems.append(f"rank {reader['rank']} non-positive timing")
                break
    if row_problems:
        problems.append({"case": row["case"], "pass": row["pass"], "problems": row_problems})
    trials += row_trials
    notifications_received += row["notifications_received"]
    notifications_expected += row["notifications_expected"]
    per_record.append({"case": row["case"], "pass": row["pass"],
                       "readers": len(readers), "trials": row_trials,
                       "blocks_per_request": row["blocks_per_request"],
                       "bytes_per_reader": row["bytes_per_reader"],
                       "notifications": row["notifications_received"]})

blocks_by_tokens = defaultdict(set)
for row in raw:
    blocks_by_tokens[row["prompt_tokens"]].add(row["blocks_per_request"])
bytes_ok = all(row["bytes_per_reader"] == row["blocks_per_request"]
               * row["requests_in_flight"] * layout.block_bytes for row in raw)
check("geometry_and_trials", not problems and trials == 12150 and bytes_ok
      and blocks_by_tokens[6143] == {46} and blocks_by_tokens[6144] == {46}
      and blocks_by_tokens[6145] == {51}, {
          "reader_timing_trials": trials, "expected_reader_trials": 12150,
          "blocks_per_request_by_prompt": {k: sorted(v) for k, v in blocks_by_tokens.items()},
          "bytes_per_reader_formula": "blocks x in_flight x 1002240",
          "bytes_per_reader_matches": bytes_ok,
          "problems": problems[:20], "problem_records": len(problems),
      })
check("notifications", notifications_received == 80352 and notifications_expected == 80352, {
    "received": notifications_received, "expected": notifications_expected,
    "documented_expected": 80352,
    "per_request_total": int(config["warmup"]) + int(config["repeats"]),
})

# --- 6. CSV rebuilt from raw ------------------------------------------------
rebuilt = [summarize_case(row) for row in raw]
csv_mismatches = []
for index, (row, summary) in enumerate(zip(rebuilt, csv_rows)):
    for key, value in row.items():
        if summary.get(key) != str(value):
            csv_mismatches.append({"row": index, "column": key, "raw": str(value),
                                   "csv": summary.get(key)})
check("csv_matches_raw", len(csv_rows) == 90 and len(rebuilt) == 90 and not csv_mismatches, {
    "csv_rows": len(csv_rows), "rebuilt_rows": len(rebuilt),
    "mismatch_count": len(csv_mismatches), "mismatches": csv_mismatches[:10],
})

# --- 7. native registration stride evidence ---------------------------------
evidence = json.loads(EVIDENCE.read_text())
native_lines = [entry["text"] for entry in evidence["logs"]]
native_strides = sorted({part.split("block_stride=")[1].split(",")[0]
                         for part in native_lines if "block_stride=" in part})
lock = json.loads(LOCK.read_text())
lock_hash = lock["benchmark_sources"].get(str(EVIDENCE))
check("native_registration_stride", native_strides == ["1002240"]
      and bool(native_lines) and lock_hash == sha256(EVIDENCE)
      and layout.block_bytes == 1002240, {
          "native_log_lines": len(native_lines),
          "native_block_strides": native_strides,
          "evidence_sha256": sha256(EVIDENCE),
          "lock_sha256": lock_hash,
          "synthetic_region_sum_bytes": layout.block_bytes,
          "native_source": evidence["native_source"],
          "native_source_sha256_recorded": evidence["native_source_sha256"],
      }, note="Native stride comes from recorded vLLM server logs; the run's block "
              "selection is synthetic and is checked separately.")

# --- summary CSVs ----------------------------------------------------------
READER_STATS = ("complete_us",)


def nz(values):
    return [v for v in values if v is not None]


def load_trials(row):
    """Per-repeat critical-path completion across the active readers, in ms."""
    per_rank = [reader["timings"] for reader in row["readers"]]
    return [max(t["complete_us"] for t in repeat) / 1000.0 for repeat in zip(*per_rank)]


def load_reader_trials(row):
    """Every reader's own timing of that case and pass, in ms."""
    return [timing["complete_us"] / 1000.0
            for reader in row["readers"] for timing in reader["timings"]]


def p90(values):
    return sorted(values)[min(len(values) - 1, int(0.9 * len(values)))]


grouped = defaultdict(list)
for row in raw:
    grouped[(row["case"], row["pass"])].append(row)

case_pass_rows = []
case_all = defaultdict(list)
for (case, pass_index), rows in sorted(grouped.items()):
    assert len(rows) == 1
    row = rows[0]
    trials_ms = load_trials(row)
    reader_ms = load_reader_trials(row)
    case_all[case].extend(trials_ms)
    case_pass_rows.append({
        "case": case, "pass": pass_index,
        "prompt_tokens": row["prompt_tokens"],
        "blocks_per_request": row["blocks_per_request"],
        "active_pairs": row["active_pairs"],
        "requests_in_flight": row["requests_in_flight"],
        "reader_count": len(row["readers"]),
        "reader_trials": sum(len(r["timings"]) for r in row["readers"]),
        "critical_path_trials": len(trials_ms),
        "bytes_per_reader": row["bytes_per_reader"],
        "complete_median_ms": round(statistics.median(trials_ms), 6),
        "complete_p90_ms": round(p90(trials_ms), 6),
        "complete_max_ms": round(max(trials_ms), 6),
        "reader_complete_median_ms": round(statistics.median(reader_ms), 6),
        "reader_complete_p90_ms": round(p90(reader_ms), 6),
        "reader_complete_max_ms": round(max(reader_ms), 6),
        "notifications_received": row["notifications_received"],
        "data_verified": all(r["data_verified"] for r in row["readers"]),
    })
case_pass_rows.sort(key=lambda row: (row["prompt_tokens"], row["active_pairs"],
                                     row["requests_in_flight"], row["pass"]))
with (RUN / "case_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(case_pass_rows[0]))
    writer.writeheader()
    writer.writerows(case_pass_rows)

spread_rows = []
for case in sorted({row["case"] for row in case_pass_rows}):
    passes = [row for row in case_pass_rows if row["case"] == case]
    medians = [row["complete_median_ms"] for row in passes]
    spread_rows.append({
        "case": case,
        "prompt_tokens": passes[0]["prompt_tokens"],
        "active_pairs": passes[0]["active_pairs"],
        "requests_in_flight": passes[0]["requests_in_flight"],
        "pass_medians_ms": ";".join(f"{value:.6f}" for value in medians),
        "median_of_pass_medians_ms": round(statistics.median(medians), 6),
        "pass_median_spread_ms": round(max(medians) - min(medians), 6),
        "pass_median_spread_pct_of_median": round(
            100.0 * (max(medians) - min(medians)) / statistics.median(medians), 4),
    })
with (RUN / "case_pass_spread.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(spread_rows[0]))
    writer.writeheader()
    writer.writerows(spread_rows)

paired_rows = []
for active in sorted({row["active_pairs"] for row in case_pass_rows}):
    for in_flight in sorted({row["requests_in_flight"] for row in case_pass_rows}):
        values = {}
        for tokens in (6143, 6144):
            case = f"t{tokens}_pairs{active}_inflight{in_flight}"
            rows = [row for row in case_pass_rows if row["case"] == case]
            values[tokens] = {
                "pass_medians_ms": [row["complete_median_ms"] for row in rows],
                "trial_median_ms": round(statistics.median(case_all[case]), 6),
            }
        delta = values[6144]["trial_median_ms"] - values[6143]["trial_median_ms"]
        paired_rows.append({
            "active_pairs": active, "requests_in_flight": in_flight,
            "bytes_per_reader": 46 * in_flight * layout.block_bytes,
            "median_6143_ms": values[6143]["trial_median_ms"],
            "median_6144_ms": values[6144]["trial_median_ms"],
            "pass_medians_6143_ms": ";".join(f"{v:.6f}" for v in values[6143]["pass_medians_ms"]),
            "pass_medians_6144_ms": ";".join(f"{v:.6f}" for v in values[6144]["pass_medians_ms"]),
            "delta_ms_6144_minus_6143": round(delta, 6),
            "ratio_6144_over_6143": round(values[6144]["trial_median_ms"]
                                          / values[6143]["trial_median_ms"], 6),
        })
with (RUN / "paired_6143_6144.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
    writer.writeheader()
    writer.writerows(paired_rows)

(RUN / "nixl_summary.csv").write_bytes((DOWN / "C_nixl/nixl_summary.csv").read_bytes())

result = {
    "status": "PASS" if all(row["pass"] for row in checks) else "FAIL",
    "generated_utc": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(),
    "job": "exp-1102-030025-971155",
    "cloud_root": download["cloud_root"],
    "download": {k: download[k] for k in ("files", "bytes", "hashes_equal",
                                          "missing_remote", "remote_nodes")},
    "checks": checks,
    "counts": {"raw_records": len(raw), "csv_rows": len(csv_rows),
               "cases": len({row["case"] for row in raw}),
               "reader_trials": trials,
               "notifications_received": notifications_received,
               "notifications_expected": notifications_expected},
    "environment": plan.get("environment"),
    "limits": ("Standalone synthetic NIXL KV READ diagnostic on 2x8 H200. "
               "The prompt 6143/6144/6145 grid reproduces the packed-block geometry; "
               "it does not observe native serving request wire bytes."),
}
(RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "counts": result["counts"],
                  "failed_checks": [row["name"] for row in checks if not row["pass"]]}, indent=2))
