"""Compare full production-wave outputs and the focused baseline/candidate suites."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(sys.argv[1]).resolve()
DEST = Path(__file__).resolve().parent
REPO = DEST.parents[3]
frozen = json.loads((ROOT / "freeze.json").read_text())
reports = {label: json.loads((ROOT / label / "wave_result.json").read_text())
           for label in ("baseline", "candidate")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


comparisons = []


def same(left, right, label):
    assert left.read_bytes() == right.read_bytes(), label
    comparisons.append({"path": label, "bytes": left.stat().st_size, "sha256": sha(left)})


same(ROOT / "baseline/pricing_payload.json", ROOT / "candidate/pricing_payload.json", "pricing_payload.json")
left = ROOT / "baseline/accepted-output/baseline"
right = ROOT / "candidate/accepted-output/candidate"
same(left / "ledger.json", right / "ledger.json", "ledger.json")
diagnostic_headers = {}
for path in sorted(left.rglob("op_traces.jsonl")):
    relative = str(path.relative_to(left))
    if relative in ("late-json-failure/op_traces.jsonl", "profile/op_traces.jsonl"):
        header_left, prefix_left = path.read_bytes().split(b"\n", 1)
        header_right, prefix_right = (right / relative).read_bytes().split(b"\n", 1)
        expected = 3902 if relative.startswith("late-json-failure/") else 0
        assert prefix_left == prefix_right and len(prefix_left) == expected
        headers = [json.loads(header_left), json.loads(header_right)]
        timestamps = [header["meta"].pop("timestamp") for header in headers]
        assert headers[0] == headers[1]
        diagnostic_headers[relative] = {"metadata_timestamps": timestamps,
            "other_header_values_equal": True, "prefix_bytes": len(prefix_left),
            "prefix_sha256": hashlib.sha256(prefix_left).hexdigest()}
    else:
        same(path, right / relative, relative)
assert len(comparisons) == 19
for label, report in reports.items():
    counts = report["operation_counts"]
    assert [counts[key] for key in ("_get_on_demand_prediction", "_predict_step5_ffn_operator",
        "predict_collective_plan", "_step5_position_queries", "cache_key")] == [288, 144, 64, 16, 192]
    assert report["output"]["events"] == 208 and report["output"]["wrong_plan_key_negatives"] == 64
    assert report["output"]["late_json_error_restores_buffer_and_prefix_bytes"] == 3902
    assert report["main_run_successful_file_reads"] == 1
    assert report["source_unchanged"] and report["fixture_metadata_unchanged"]
    assert report["view_control"]["applied"] is False
    assert report["view_control"]["production_method_files_verified"]
    source = Path(frozen["sources"][label]["root"])
    assert git(source, "rev-parse", "HEAD") == frozen["sources"][label]["head"]
    assert not git(source, "diff", "HEAD", "--name-only", "--", "frontier", "tests", "data")
    for relative, expected in frozen["sources"][label]["hashes"].items():
        assert sha(source / relative) == expected, relative
assert reports["baseline"]["view_control"] == reports["candidate"]["view_control"]
assert reports["baseline"]["operation_counts"]["_resolve_cluster_model_contract"] == 384
assert reports["candidate"]["operation_counts"]["_resolve_cluster_model_contract"] == 0
assert sha(Path(frozen["protected_body"])) == frozen["protected_body_sha256"]
fixture = Path(frozen["fixture"])
metadata = {str(path.relative_to(fixture)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in fixture.rglob("*") if path.is_file()}
assert metadata == frozen["fixture_metadata"] and len(metadata) == 262
for path, expected in frozen["input_hashes"].items():
    assert sha(Path(path)) == expected, path


def suite(path):
    cases = {}
    for item in ET.parse(path).iter("testcase"):
        failure = item.find("failure")
        error = item.find("error")
        skipped = item.find("skipped")
        status = "FAIL" if failure is not None else "ERROR" if error is not None else "SKIP" if skipped is not None else "PASS"
        node = failure if failure is not None else error
        cases[item.attrib["classname"] + "::" + item.attrib["name"]] = {
            "status": status, "message": node.attrib.get("message") if node is not None else None}
    return cases


suites = {}
for prefix in ("focused", "governance"):
    base = suite(ROOT / f"{prefix}-baseline.xml")
    candidate = suite(ROOT / f"{prefix}-candidate.xml")
    changed = {name: [result, candidate.get(name)] for name, result in base.items()
               if candidate.get(name) != result}
    assert not changed, changed
    new = {name: value for name, value in candidate.items() if name not in base}
    assert all(value["status"] == "PASS" for value in new.values())
    suites[prefix] = {"baseline": base, "candidate": candidate, "changed_existing": changed, "new_cases": new}

owned = json.loads((REPO / "task_memory/task_2026-10-22_step5_position_routing/repairs/named_model_query_scope_amendment.json").read_text())


def numstat(revision):
    text = git(REPO, "diff", "--numstat", revision, "--", *owned["repair_paths"])
    return {path: {"added": int(added), "deleted": int(deleted), "gross": int(added) + int(deleted),
                   "net": int(added) - int(deleted)}
            for added, deleted, path in (line.split("\t") for line in text.splitlines())}


step = numstat(owned["step_base"])
cumulative = numstat("1aee0af11cd58691e64603b99eca0ac459a5b350")
caps = dict(zip(owned["repair_paths"], (120, 223, 353, 151)))
assert sum(row["gross"] for row in step.values()) <= 100
for path in owned["repair_paths"]:
    assert step[path]["gross"] <= owned["path_step_ceilings"][path]
    assert cumulative[path]["gross"] <= caps[path]
baseline_usage = dict(zip(owned["repair_paths"], (96, 195, 299, 117)))
delta = sum(cumulative[path]["gross"] - baseline_usage[path] for path in owned["repair_paths"])
assert set(git(REPO, "diff", "--name-only", owned["step_base"], "--", "frontier", "tests", "data").splitlines()) == set(owned["repair_paths"])
warm = {label: report["warm_median_s"] for label, report in reports.items()}
receipt = {"task_id": "D29-NAMED-MODEL-QUERY", "session_id": frozen["session_id"],
    "status": "COMPONENT_VERIFIED_PARENT_ACCEPTANCE_PENDING", "sources": frozen["sources"],
    "source_binding_count": 23, "fixture_file_count": len(metadata),
    "protected_body_sha256": frozen["protected_body_sha256"], "byte_comparisons": comparisons,
    "diagnostic_headers": diagnostic_headers, "candidate_install_removed": True,
    "production_view_verification": reports["candidate"]["view_control"],
    "operation_counts": {label: report["operation_counts"] for label, report in reports.items()},
    "warm_median_s": warm, "warm_change_percent": 100 * (warm["candidate"] / warm["baseline"] - 1),
    "suite_comparison": suites, "step_numstat": step, "cumulative_numstat": cumulative,
    "cumulative_delta_gross": delta,
    "incremental_union_from_accepted_9491": {"core": 4173 + delta, "pricing": 3954, "native": 1350,
        "source": 9477 + delta, "global": 9491 + delta,
        "qualification": "Incremental receipt over accepted union; final composed audit belongs to primary."},
    "environment": {"python": sys.version, "executable": sys.executable,
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")}},
    "limits": "No full CLI, full unit, strict77, GPU/native or final acceptance run. Existing Torch-dependent failure remains unchanged."}
(ROOT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
paths = ["freeze.json", "run.py", "fixture_wiring.py", "prepare.log", "baseline.log", "candidate.log",
    "verification.json", "baseline/wave_result.json", "candidate/wave_result.json",
    "red.log", "green.log", "final-green.log", "final-green.xml", "focused-baseline.log", "focused-candidate.log", "focused-baseline.xml",
    "focused-candidate.xml", "governance-baseline.log", "governance-candidate.log",
    "governance-baseline.xml", "governance-candidate.xml"]
manifest = {}
for relative in paths:
    path, target = ROOT / relative, DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    target.write_bytes(path.read_bytes())
    manifest[relative] = {"bytes": path.stat().st_size, "sha256": sha(path)}
(DEST / "manifest.json").write_text(json.dumps({"scratch_root": str(ROOT), "files": manifest}, indent=2) + "\n")
print(json.dumps({"status": receipt["status"], "byte_comparisons": len(comparisons),
    "warm_median_s": warm, "warm_change_percent": receipt["warm_change_percent"],
    "step_numstat": step, "cumulative_numstat": cumulative,
    "incremental_union": receipt["incremental_union_from_accepted_9491"]}, indent=2))
