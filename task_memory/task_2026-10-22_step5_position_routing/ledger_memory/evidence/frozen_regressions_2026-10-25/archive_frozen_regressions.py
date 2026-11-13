"""Retain exact mandatory regression evidence after the original runs finish."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PRIMARY = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
TASK = Path("task_memory/task_2026-10-22_step5_position_routing")
DEST = PRIMARY / TASK / "ledger_memory/evidence/frozen_regressions_2026-10-25"
SOURCE = "1461573de388b72450189bc0c1b88b4ac32491f0"
SUITES = ("strict77", "unit", "supplement101", "metadata196", "catalogue53", "governance52")
started = time.perf_counter()
start_utc = datetime.now(timezone.utc).isoformat()
assert sys.dont_write_bytecode
assert not DEST.exists()


def record(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def ids(path):
    result = {}
    for row in ET.parse(path).getroot().iter("testcase"):
        key = (row.get("classname", ""), row.get("name", ""))
        assert key not in result, key
        result[key] = next((name for name in ("error", "failure", "skipped") if row.find(name) is not None), "pass")
    return result


unit_comparison = json.loads((ROOT / "unit_comparison.json").read_bytes())
assert unit_comparison["status"] == "PASS_ALL_RETAINED_UNIT_IDS_AND_STATES"
assert unit_comparison["candidate_counts"] == {"pass": 8028, "failure": 279, "error": 32, "skipped": 281}
assert unit_comparison["retained_ids"] == 8620
assert unit_comparison["failure_error_skip_sets_identical"]
assert unit_comparison["raw_candidate_exit"] == 1
matrix = json.loads((ROOT / "strict77_audit/postrun_receipt.json").read_bytes())
assert matrix["verdict"] == "PASS" and matrix["identical_cases"] == matrix["cases_compared"] == 77
assert matrix["baseline"]["artifacts_rehashed"] == matrix["candidate"]["artifacts_rehashed"] == 451
assert matrix["baseline"]["cache_filename_count"] == matrix["candidate"]["cache_filename_count"] == 519
summaries = {}
for suite in SUITES:
    execution = json.loads((ROOT / suite / "execution.json").read_bytes())
    assert execution["source_before"] == execution["source_after"]
    assert execution["source_before"]["head"] == SOURCE
    assert execution["source_before"]["status"] == execution["source_before"]["submodule_status"] == ""
    assert execution["native_binary_projection"] is False and execution["source_binary_present"] is False
    exits = [row["returncode"] for row in execution["executions"]]
    assert exits == ([0, 0] if suite == "strict77" else [1] if suite == "unit" else [0])
    summary = {"source": SOURCE, "raw_exits": exits, "duration_s": [row["duration_seconds"] for row in execution["executions"]], "versions": execution["versions"]}
    if suite != "strict77":
        summary["counts"] = dict(Counter(ids(Path(execution["junit"])).values()))
        if suite != "unit":
            comparison = json.loads((ROOT / suite / "comparison.json").read_bytes())
            assert comparison["status"] == "PASS_IDENTICAL_IDS_STATUSES_AND_ENVIRONMENT"
            assert not comparison["missing"] and not comparison["added"] and not comparison["changed"]
    summaries[suite] = summary

selected = []
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix in (".py", ".json") or p.name.endswith("_launcher.log") or p.name in ("unit_compare.log", "unit_compare_first_failure.log")):
        selected.append(p)
for suite in SUITES:
    selected.extend(p for p in (ROOT / suite).iterdir() if p.is_file())
selected.extend(p for p in (ROOT / "strict77_audit").iterdir() if p.is_file())
selected.extend(p for p in (ROOT / "strict77_output").iterdir() if p.is_file())
selected.extend(p for p in (ROOT / "strict77_output/candidate").iterdir() if p.is_file())
previous_id_receipt = PRIMARY / TASK / "post_key_regressions/evidence/environment_reconciliation/first_unit_legacy_comparison.json"
previous_id = json.loads(previous_id_receipt.read_bytes())
old_id = ["tests.unit.test_collective_sim_cc_backend", "test_collective_artifact_reuses_only_identical_current_text"]
current_ids = ids(ROOT / "unit/results.xml")
assert tuple(old_id) not in current_ids
historical_missing = str(previous_id)
assert old_id[1] in historical_missing
DEST.mkdir(parents=True)
files = []
for source in sorted(set(selected)):
    relative = source.relative_to(ROOT)
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    target.write_bytes(raw)
    assert target.read_bytes() == raw
    files.append({"source": record(source), "archive": record(target)})
# Keep the old raw verdict as a historical record; no alias or exemption is added.
receipt = {"task_id": "D29-LEDGER-FINAL-FROZEN-REGRESSIONS", "status": "PASS_MANDATORY_SOURCE_REGRESSIONS_ONLY",
           "source": SOURCE, "suites": summaries, "strict77_audit": matrix,
           "canonical_unit_comparison": unit_comparison, "archived_files": files,
           "previous_approved_old_id": {"id": old_id, "present_current": False,
                                        "raw_presence_gate": "FAIL", "historical_receipt": record(previous_id_receipt), "waived": False},
           "created_utc": start_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "duration_s": time.perf_counter() - started,
           "environment": {"python": sys.version, "executable": sys.executable, "configured_memory_max_bytes": 32212254720,
                           "effective_cgroup_peak": "not measured", "math_threads": 1, "timeout_s": 60},
           "limits": ["Exact raw suite outcomes remain separate and overlapping; counts are not summed.",
                      "Full original generated outputs and audited caches remain at their cited scratch paths.",
                      "No native/wholeCLI/RSS/worker/G3/E2E admission, new control invocation or source edit."]}
(DEST / "archive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"status": receipt["status"], "source": SOURCE, "archived_files": len(files),
                  "archived_bytes": sum(row["archive"]["bytes"] for row in files),
                  "suites": {name: {key: value for key,value in row.items() if key != "versions"} for name,row in summaries.items()},
                  "old_id_presence": receipt["previous_approved_old_id"]["raw_presence_gate"]}, indent=2))
