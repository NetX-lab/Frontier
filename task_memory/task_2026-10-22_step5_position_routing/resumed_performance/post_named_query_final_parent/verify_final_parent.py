"""Audit completed receipts only; never launch a test or simulation."""

from collections import Counter
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time


OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
TASK = "task_memory/task_2026-10-22_step5_position_routing/"
OWNER = Path("/data/ycfeng/Frontier/.worktrees/d29-post-named-query-regressions")
OWNER_SCOPE = TASK + "post_named_query_regressions/"
EVIDENCE = OWNER / OWNER_SCOPE / "evidence"
SOURCE = Path("/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08")
PRIMARY = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
REVISION = "dbe19f531ff68d9edafc480418e8927c40eb1c60"
TREE = "6c43090097e92c6fc0c33aa7358e69c2cf092705"
PIN = "3dac53c544e546574cb6e8078cd14053d0916113"
BASE = "86b9923f8c29f1c58ba9f8df23a396c70a4d1caa"
OWNER_CODE = "0a42e56ef53769a85a999b652cd9d03973d1f727"
OWNER_FINAL = "39f0c1000a96c10f186cabe5f77dfc3058f86549"
OLD_ID = "tests.unit.test_collective_sim_cc_backend::test_collective_artifact_reuses_only_identical_current_text"
PARSER = OWNER / TASK / "resumed_regressions/evidence/scripts/compare_unit.py"
AUDITOR = Path("/data/ycfeng/tmp/step5_workflow_fidelity_20261022/audit_evidence.py")
CHECKS = {}
COMMANDS = []


def git(*args, repo=REPO):
    argv = ["git", "--no-optional-locks", "-C", str(repo), *args]
    COMMANDS.append(argv)
    return subprocess.run(argv, capture_output=True, check=True, timeout=60).stdout


def checked(path, expected=None, size=None):
    path = Path(path)
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            count += len(block)
    value = {"path": str(path), "sha256": digest.hexdigest(), "bytes": count}
    assert expected is None or value["sha256"] == expected, value
    assert size is None or count == size, value
    assert str(path) not in CHECKS or CHECKS[str(path)] == value, value
    CHECKS[str(path)] = value
    return value


def item_checked(item):
    return checked(item["path"], item["sha256"], item.get("bytes"))


def load(path):
    checked(path)
    return json.loads(Path(path).read_text())


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def state(repo):
    index = Path(git("rev-parse", "--git-path", "index", repo=repo).decode().strip())
    if not index.is_absolute():
        index = repo / index
    return {"head": git("rev-parse", "HEAD", repo=repo).decode().strip(),
            "index_path": str(index), "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}


def source_state():
    module = SOURCE / "frontier/cc_backend/backends/collective-sim"
    result = {"head": git("rev-parse", "HEAD", repo=SOURCE).decode().strip(),
              "tree": git("rev-parse", "HEAD^{tree}", repo=SOURCE).decode().strip(),
              "status": git("status", "--porcelain", repo=SOURCE).decode().strip(),
              "submodule_head": git("rev-parse", "HEAD", repo=module).decode().strip(),
              "submodule_status": git("status", "--porcelain", repo=module).decode().strip()}
    assert result == {"head": REVISION, "tree": TREE, "status": "", "submodule_head": PIN, "submodule_status": ""}
    assert not (module / "sim/datacenter/htsim_ndp").exists()
    return result


def environment():
    assert os.getuid() == 10250 and sys.dont_write_bytecode
    cgroup = Path("/proc/self/cgroup").read_text()
    unit = next(line.rsplit("/", 1)[-1] for line in cgroup.splitlines() if line.startswith("1:name=systemd:"))
    argv = ["systemctl", "show", unit, "-p", "MemoryMax", "-p", "MemoryLimit", "-p", "ControlGroup"]
    properties = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=60).stdout
    values = dict(line.split("=", 1) for line in properties.splitlines())
    assert int(values["MemoryMax"]) == 30 * 1024 ** 3, values
    return {"uid": os.getuid(), "python": sys.version, "executable": sys.executable,
            "bytecode_disabled": True, "MemoryMax_bytes": int(values["MemoryMax"]),
            "systemd_properties": values, "cgroup": cgroup, "timeout_seconds": 60,
            "argv": sys.argv, "stream_block_bytes": 1024 * 1024}


def source_proofs(added):
    result = []
    for identifier in added:
        module, name = identifier.split("::", 1)
        path = module.replace(".", "/") + ".py"
        function = name.split("[", 1)[0]
        old = ast.parse(git("show", f"46fe83ac:{path}").decode())
        current = ast.parse(git("show", f"{REVISION}:{path}").decode())
        assert function not in {n.name for n in ast.walk(old) if isinstance(n, ast.FunctionDef)}
        assert function in {n.name for n in ast.walk(current) if isinstance(n, ast.FunctionDef)}
        result.append({"id": identifier, "source_path": path, "function": function,
                       "absent_in_46fe_source": True, "present_in_frozen_source": True})
    return result


def receipts():
    manifest = load(EVIDENCE / "records_manifest.json")
    assert manifest["file_count"] == len(manifest["files"]) == 33
    assert state(OWNER)["head"] == OWNER_FINAL
    assert git("rev-parse", OWNER_FINAL + "^", repo=OWNER).decode().strip() == OWNER_CODE
    commit_changes = {}
    for commit, expected_count in [(OWNER_CODE, 34), (OWNER_FINAL, 5)]:
        paths = git("diff-tree", "--no-commit-id", "--name-only", "-r", commit, repo=OWNER).decode().splitlines()
        assert len(paths) == expected_count and all(p.startswith(OWNER_SCOPE) for p in paths)
        commit_changes[commit] = paths
    for item in manifest["files"]:
        path = OWNER / OWNER_SCOPE / item["path"]
        checked(path, item["sha256"], item["bytes"])
        assert path.read_bytes() == git("cat-file", "blob", f"{OWNER_FINAL}:{OWNER_SCOPE + item['path']}", repo=OWNER)
    for name in ["initial_archive_manifest.json", "short_completion_archive_manifest.json",
                 "final_archive_manifest.json", "strict77/archive_manifest.json"]:
        for item in load(EVIDENCE / name):
            for kind in ["source", "retained"]:
                checked(item[kind], item["sha256"], item["bytes"])
    prior = REPO / TASK / "resumed_performance/post_named_query_regression_acceptance.json"
    short_parent = load(prior)
    assert prior.read_bytes() == git("show", f"{BASE}:{prior.relative_to(REPO)}")
    assert short_parent["status"] == "PASS_SHORT_SUITES_ONLY" and len(short_parent["checks"]) == 27
    for path, item in short_parent["checks"].items():
        checked(path, item["sha256"], item["bytes"])
    checked(REPO / TASK / "resumed_performance/post_key_parent_checks.py")
    checked(REPO / TASK / "resumed_performance/verify_post_named_query_shorts.py")
    parser_binding = next(x["after"] for x in load(EVIDENCE / "source_final_receipt.json")["script_before_after"] if x["after"]["path"] == str(PARSER))
    item_checked(parser_binding)
    parse = runpy.run_path(str(PARSER), run_name="parent_receipt_parser")["parse"]
    parsed = {}

    def cases(item):
        item_checked(item)
        if item["path"] not in parsed:
            parsed[item["path"]] = parse(Path(item["path"]))
        return parsed[item["path"]][0]

    final = load(EVIDENCE / "final_verification.json")
    comparison = load(EVIDENCE / "unit/final_comparison_both_baselines.json")
    current = cases(comparison["fresh_xml"])
    item_checked(final["raw_outputs"]["unit_log"])
    counts = dict(Counter(x["status"] for x in current.values()))
    assert counts == final["raw_unit_counts"] == {"pass": 8028, "failure": 279, "error": 32, "skipped": 281}
    assert final["raw_unit_exit"] == comparison["raw_exit"] == 1 and final["raw_unit_all_pass"] is False
    execution = load(EVIDENCE / "unit/execution.json")
    assert [x["returncode"] for x in execution["executions"]] == [1]
    assert execution["source_before"] == execution["source_after"] == source_state()
    for item in execution["executions"]:
        checked(item["log"])
    transitions = {}
    for label, item in [("46fe", comparison["retained46fe_xml"]), ("79", comparison["retained79_xml"])]:
        old = cases(item)
        missing = sorted(old.keys() - current.keys())
        added = sorted(current.keys() - old.keys())
        changed = sorted(k for k in old.keys() & current.keys() if old[k]["status"] != current[k]["status"])
        deltas = {s: {"added": sorted(k for k in current if current[k]["status"] == s and (k not in old or old[k]["status"] != s)),
                      "removed": sorted(k for k in old if old[k]["status"] == s and (k not in current or current[k]["status"] != s))}
                  for s in ["failure", "error", "skipped"]}
        assert not changed and not any(v for row in deltas.values() for v in row.values())
        assert missing == ([] if label == "46fe" else [OLD_ID])
        assert len(added) == (8 if label == "46fe" else 11) and all(current[k]["status"] == "pass" for k in added)
        assert comparison[f"retained{label}_id_transitions"]["missing"] == missing
        assert comparison[f"retained{label}_id_transitions"]["new_ids"] == added
        transitions[label] = {"baseline_counts": dict(Counter(x["status"] for x in old.values())),
                              "retained_count": len(old.keys() & current.keys()), "missing": missing,
                              "new_ids_all_pass": added, "changed_statuses": changed, "status_deltas": deltas,
                              "raw_presence_gate": "PASS" if not missing else "FAIL_APPROVED_OLD_ID_MISSING"}
    assert comparison["raw_retained46fe_status_gate"] == "PASS"
    assert comparison["raw_retained79_exact_status_set_gate"] == final["raw79_exact_status_gate"] == "FAIL_RETAINED_ID_LOSS_OR_STATUS_CHANGE"
    assert comparison["legacy_gate_raw_exit"] == comparison["raw79_gate_exit_preserved"] == final["raw_legacy_gate_exit"] == 1
    assert comparison["no_raw_presence_waiver"] and final["no_presence_waiver_or_alias"]
    legacy = load(EVIDENCE / "unit/legacy_comparison.json")
    assert comparison["required_gate_verdict"] == "FAIL_OR_INCOMPLETE_REQUIRED_GATE"
    item_checked(comparison["legacy_report"])
    shorts, short_cases = {}, {}
    for suite, expected in [("supplement101", 101), ("metadata196", 197), ("catalogue53", 53), ("governance52", 52)]:
        receipt = load(EVIDENCE / suite / "comparison.json")
        selected = cases(receipt["candidate_xml"])
        launch = load(EVIDENCE / suite / "execution.json")
        assert dict(Counter(x["status"] for x in selected.values())) == {"pass": expected}
        assert receipt["raw_exit"] == 0 and all(x["returncode"] == 0 for x in launch["executions"])
        assert launch["source_before"] == launch["source_after"] == source_state()
        assert len(selected) == short_parent["independent_suites"][suite]["counts"]["pass"]
        short_cases[suite] = selected
        shorts[suite] = {"counts": {"pass": expected}, "raw_exit": 0, "versions": launch["versions"],
                         "full_unit_status_counts_for_present_ids": dict(Counter(current[k]["status"] for k in selected if k in current))}
        if suite in ["metadata196", "catalogue53"]:
            assert all(k in current and current[k]["status"] == "pass" for k in selected)
    locked = load(legacy["evidence"]["locked_ids"]["path"])
    item_checked(legacy["evidence"]["locked_ids"])
    reconciliation = []
    for status, field, expected in [("failure", "unit_failure_ids", 12), ("error", "unit_error_ids", 1)]:
        old = set(locked[field])
        actual = {k for k in current if current[k]["status"] == status}
        assert old <= actual
        added = sorted(actual - old)
        assert len(added) == expected
        for identifier in added:
            message = current[identifier]["text"]
            if status == "failure":
                assert "No module named 'aiohttp'" in message and short_cases["supplement101"][identifier]["status"] == "pass"
                covered = [identifier]
            else:
                covered = sorted(k for k in short_cases["supplement101"] if k.startswith(identifier.split("::")[-1] + "::"))
                assert "No module named 'torch'" in message and len(covered) == 16
            reconciliation.append({"id": identifier, "raw_status": status, "supplement_passing_ids": covered})
    governance = sorted(k for k in short_cases["governance52"] if k in current and current[k]["status"] == "failure" and "No module named 'torch'" in current[k]["text"])
    assert len(governance) == 5

    # Execute only the original evidence auditor, retaining every predicate.
    checked(AUDITOR, "8d985f0f964c5ec6b16b4a0c3963490db38de26b9e0ec664927479ca849b4cb9")
    sys.path.insert(0, str(SOURCE))
    auditor = runpy.run_path(str(AUDITOR), run_name="parent_strict_evidence_auditor")
    namespace = auditor["main"].__globals__
    scratch = Path(tempfile.mkdtemp(prefix="d29_post_named_parent_strict_", dir="/data/ycfeng/tmp"))
    namespace.update(ROOT=scratch, BASELINE=Path("/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate"),
                     CANDIDATE=Path("/data/ycfeng/tmp/d29_workflow_fidelity_20261024_final07/candidate"), REPO=SOURCE,
                     REVISION=REVISION, BASELINE_REVISION="713e6a81708e87b86cf91fd89f3b1f954bdcbead")
    namespace["digest"] = lambda path: checked(path)["sha256"]
    raw_comparison = EVIDENCE / "strict77/comparison.json"
    checked(raw_comparison, "680c535ee73b14fffb1ed7e0d622b547ad6e8ac341502f9e70fce2510e51a447")
    (OUT / "comparison.json").write_bytes(raw_comparison.read_bytes())
    (scratch / "comparison.json").write_bytes(raw_comparison.read_bytes())
    with contextlib.redirect_stdout(io.StringIO()):
        audit_exit = auditor["main"]()
    (OUT / "postrun_receipt.json").write_bytes((scratch / "postrun_receipt.json").read_bytes())
    strict = load(OUT / "postrun_receipt.json")
    owner_strict = load(EVIDENCE / "strict77/final07_receipt.json")
    assert strict == owner_strict["audit"] and audit_exit == 0 and strict["verdict"] == "PASS"
    assert strict["cases_compared"] == strict["identical_cases"] == 77
    assert all(strict[label]["artifacts_rehashed"] == 451 and strict[label]["cache_filename_count"] == 519 for label in ["baseline", "candidate"])
    strict_execution = load(EVIDENCE / "strict77/execution.json")
    assert [x["returncode"] for x in strict_execution["executions"]] == [0, 0]
    for item in strict_execution["executions"]:
        checked(item["log"])
    checked(Path(final["raw_outputs"]["strict_root"]) / "audit.log")
    write("receipt_verification.json", {"status": "PASS_RECEIPTS_AND_RAW_DELTAS", "owner_commits": commit_changes,
          "record_files_verified": 34, "raw_unit": {"counts": counts, "raw_exit": 1, "junit": comparison["fresh_xml"],
          "log": final["raw_outputs"]["unit_log"], "junit_suite_attributes": parsed[comparison["fresh_xml"]["path"]][1]},
          "transitions": transitions, "genuine_new_source_cases": source_proofs(transitions["46fe"]["new_ids_all_pass"]),
          "shorts": shorts, "short_parent_receipt": CHECKS[str(prior)], "counts_not_summed": True,
          "raw_legacy_verdict": final["legacy_gate_verdict"], "required_gate_verdict": comparison["required_gate_verdict"],
          "environment_only_reconciliation": reconciliation, "governance_raw_failures_passing_separately": governance,
          "strict77": strict, "fresh_evidence_auditor_exit": audit_exit, "evidence_auditor_scratch": str(scratch), "hash_checks": list(CHECKS.values())})


def bindings():
    guard = load(EVIDENCE / "source_final_receipt.json")
    assert guard["source_before"] == guard["source_after"] == source_state()
    assert not guard["source_file_changes"] and guard["canonical_binary_absence_unchanged"]
    item_checked(guard["source_inventory"])
    inventory = load(guard["source_inventory"]["path"])
    assert len(inventory) == 2755 and len({x["path"] for x in inventory}) == 2755
    tree = git("ls-tree", "-r", REVISION, repo=SOURCE).decode().splitlines()
    expected_paths = {line.split("\t", 1)[1] for line in tree if line.split()[1] == "blob"}
    assert expected_paths == {x["path"] for x in inventory}
    for item in inventory:
        checked(SOURCE / item["path"], item["sha256"], item["bytes"])
    for group, expected in [("input_before_after", 14), ("script_before_after", 12)]:
        assert len(guard[group]) == expected
        for item in guard[group]:
            assert item["before"] == item["after"]
            item_checked(item["after"])
    recipe = SOURCE / TASK / "runtime_model_view/parent/final_scope_guard.py"
    checked(recipe, "96e8596bb9ce0e20463bfb440d1f5d8f0c760781aa6d424d763557d360f22a95")
    source = recipe.read_text()
    original_output = "output = Path(__file__).with_name('acceptance.json')"
    assert source.count(original_output) == 1
    source = source.replace(original_output, f"output = Path({str(OUT / 'scope_guard_recomputed.json')!r})")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, str(recipe), "exec"), {"__file__": str(recipe), "__name__": "parent_scope_binding_audit"})
    scope = load(OUT / "scope_guard_recomputed.json")
    owner_scope = load(EVIDENCE / "scope_guard_after.json")
    for key in ["integrated_head", "integrated_tree", "binding_count", "non_query_binding_count", "all_binding_digest", "non_query_binding_digest", "totals", "violations"]:
        assert scope[key] == owner_scope[key], key
    assert scope["binding_count"] == 124 and scope["non_query_binding_count"] == 120 and not scope["violations"]
    assert scope["totals"]["source_actual"] == 9565 and scope["totals"]["global_actual"] == 9579
    write("binding_verification.json", {"status": "PASS_FINAL_SOURCE_INPUT_SCRIPT_SCOPE_BINDINGS",
          "source": source_state(), "source_files_rehashed": len(inventory), "source_total_bytes": sum(x["bytes"] for x in inventory),
          "inputs_rehashed": 14, "scripts_rehashed": 12, "scope": scope, "hash_checks": list(CHECKS.values())})


def acceptance():
    receipts_result = load(OUT / "receipt_verification.json")
    bindings_result = load(OUT / "binding_verification.json")
    assert receipts_result["status"] == "PASS_RECEIPTS_AND_RAW_DELTAS"
    assert bindings_result["status"] == "PASS_FINAL_SOURCE_INPUT_SCRIPT_SCOPE_BINDINGS"
    phases = [receipts_result, bindings_result]
    assert all(x["protected_states_before"] == x["protected_states_after"] for x in phases)
    checks = {item["path"]: item for phase in phases for item in phase["hash_checks"]}
    write("parent_acceptance.json", {"task_id": "D29-POST-NAMED-QUERY-FINAL-PARENT-AUDIT", "session_id": "ses_e95ac82fdffeSeMEUfR9SxvW03",
          "status": "ACCEPTED_NO_NEW_CODE_FAILURE_WITH_UNWAIVED_APPROVED_79_ID_REPLACEMENT",
          "source_revision": REVISION, "source_tree": TREE, "submodule_revision": PIN, "records_base": BASE,
          "owner_records": [OWNER_CODE, OWNER_FINAL], "unit": receipts_result["raw_unit"],
          "raw_79_presence_gate": "FAIL_RETAINED_ID_LOSS_OR_STATUS_CHANGE", "raw_legacy_gate_exit": 1,
          "raw_legacy_verdict": receipts_result["raw_legacy_verdict"], "raw_required_gate_verdict": receipts_result["required_gate_verdict"], "raw_presence_gate_waived": False,
          "transitions": receipts_result["transitions"], "new_query_cases": receipts_result["genuine_new_source_cases"],
          "strict77": {"identical": 77, "compared": 77, "artifacts_rehashed_per_side": 451, "cache_filenames_per_side": 519,
                       "fresh_evidence_auditor_exit": 0, "comparison_sha256": receipts_result["strict77"]["comparison_sha256"]},
          "independent_shorts": receipts_result["shorts"], "counts_not_summed": True,
          "environment_only_reconciliation": receipts_result["environment_only_reconciliation"],
          "governance_raw_failures_passing_separately": receipts_result["governance_raw_failures_passing_separately"],
          "source_files_rehashed": 2755, "guarded_inputs_rehashed": 14, "scripts_rehashed": 12,
          "scope_bindings": 124, "non_query_bindings": 120, "scope_totals": bindings_result["scope"]["totals"],
          "non_query_digest": bindings_result["scope"]["non_query_binding_digest"], "violations": [],
          "unique_file_hash_checks": len(checks), "phase_receipts": list(CHECKS.values()),
          "limits": ["No tests, probes, simulation, native rebuild, GPU/network/platform operation or production edit.",
                     "Raw canonical exit1, 279 failures/32 errors/281 skips and approved79 ID-loss FAIL remain visible; no aliases or suite sums.",
                     "Strict equality is bound to the retained original comparison and fresh unchanged evidence-auditor/hash checks; no large-trace or pandas CSV parsing.",
                     "Whole6144/1024/2400s CLI, native runtime, package, G3 and four-metric E2E acceptance remain outside this receipt."]})


if __name__ == "__main__":
    if sys.argv[1] == "receipts" and os.environ.get("D29_PARENT_EVIDENCE_NAMESPACE") != "strict77":
        launch_path = EVIDENCE / "strict77/execution.json"
        command = json.loads(launch_path.read_text())["executions"][0]["command"]
        start = command.index("bwrap")
        end = command.index("--", start)
        namespace = command[start:end] + ["--bind", str(OUT), str(OUT), "--setenv", "D29_PARENT_EVIDENCE_NAMESPACE", "strict77",
                                         "--", sys.executable, "-B", str(Path(__file__).resolve()), "receipts"]
        write("receipt_namespace_command.json", {"retained_execution": checked(launch_path), "command": namespace,
              "scope": "Original strict namespace; only this audit output subtree additionally writable; no suite is launched"})
        os.execvp(namespace[0], namespace)
    started = time.monotonic()
    env = environment()
    before = {"primary": state(PRIMARY), "owner": state(OWNER), "frozen_source": state(SOURCE)}
    phase = sys.argv[1]
    assert phase in {"receipts", "bindings", "acceptance"}
    {"receipts": receipts, "bindings": bindings, "acceptance": acceptance}[phase]()
    after = {"primary": state(PRIMARY), "owner": state(OWNER), "frozen_source": state(SOURCE)}
    assert before == after, (before, after)
    name = {"receipts": "receipt_verification.json", "bindings": "binding_verification.json", "acceptance": "parent_acceptance.json"}[phase]
    result = json.loads((OUT / name).read_text())
    result.update(environment=env, protected_states_before=before, protected_states_after=after,
                  git_commands=COMMANDS, duration_seconds=time.monotonic() - started)
    write(name, result)
    print(json.dumps({"phase": phase, "status": result["status"], "hash_files": len(CHECKS),
                      "duration_seconds": result["duration_seconds"], "receipt_sha256": checked(OUT / name)["sha256"]}, indent=2))
