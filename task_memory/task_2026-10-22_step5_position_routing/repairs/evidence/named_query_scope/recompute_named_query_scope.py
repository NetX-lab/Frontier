#!/usr/bin/env python3
"""Compose the authorized four-path delta without executing project code."""

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys


OUT = Path(__file__).resolve().parent
RECORDS = OUT.parents[4]
TASK = "task_memory/task_2026-10-22_step5_position_routing/"
EVIDENCE = TASK + "repairs/evidence/"
PRIMARY = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
CANDIDATE = Path("/data/ycfeng/Frontier/.worktrees/step5-d29-named-query")
SUBPATH = "frontier/cc_backend/backends/collective-sim"
SUBREPO = CANDIDATE / SUBPATH
SOURCE = "1737ce9c4a761d2c57ee617f5f3a78d7bcfd9a58"
SOURCE_RECORDS = "3d75c786781802ad2753cbdd81d989c9b694660c"
ACCEPTED = "46fe83ac55876c10f6dd8ff792705e87efd1905f"
ALLOCATION = "617d832bed8f52d6b240e4e33023ed3e5d06dacd"
STEP_BASE = "de2d818fc49613e5b7fc5c41b4e49691b2a36bd6"
ORIGINAL = "1aee0af11cd58691e64603b99eca0ac459a5b350"
KEY_SOURCE = "c4e06a6159ee3b1e63c50abaa346066ae8b6da31"
KEY_MERGE = "30fb6a309aa564777ed8f1f5e094118d65bd17d3"
KEY_AUDIT = "6a980a518c9404e121b6164669eb37400bdda58c"
SNAP_AUDIT = "d267db2d7d337dc237c85ea63f16cfa0290eb42d"
PIN = "3dac53c544e546574cb6e8078cd14053d0916113"
ORIGINAL_PIN = "d28fe917370c2d55230c3ea226e5660785b44aca"
AMENDMENT = TASK + "repairs/named_model_query_scope_amendment.json"
QUERY = [
    "frontier/execution_time_predictor/prediction_model_registry.py",
    "frontier/execution_time_predictor/moe_routing_workload.py",
    "tests/unit/test_prediction_measurement_bindings.py",
    "tests/unit/test_moe_routing_runtime.py",
]
CAPS = {"core_source": 4404, "pricing": 3956, "native": 1582}
COMMANDS = []
VIOLATIONS = []


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args, repo=RECORDS, expected=(0,)):
    cmd = ["git", "--no-optional-locks", "-C", str(repo), *args]
    COMMANDS.append({"argv": cmd, "timeout_seconds": 60})
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode not in expected:
        raise RuntimeError(f"{args}: exit {result.returncode}: {result.stderr.decode()}")
    return result.stdout


def require(value, label):
    if not value:
        VIOLATIONS.append(label)


def state(repo):
    index = Path(git("rev-parse", "--git-path", "index", repo=repo).decode().strip())
    if not index.is_absolute():
        index = repo / index
    return {"HEAD": git("rev-parse", "HEAD", repo=repo).decode().strip(),
            "index_path": str(index), "index_sha256": sha(index.read_bytes())}


def entries(rev, paths, repo=RECORDS):
    result = {}
    for line in git("ls-tree", rev, "--", *paths, repo=repo).decode().splitlines():
        meta, path = line.split("\t")
        mode, kind, oid = meta.split()
        result[path] = {"mode": mode, "type": kind, "oid": oid}
    require(set(result) == set(paths), f"missing_tree_entries:{rev}")
    return result


def numstats(a, b, paths=(), repo=RECORDS):
    result = {}
    args = ["diff", "--no-renames", "--numstat", a, b]
    if paths:
        args += ["--", *paths]
    for line in git(*args, repo=repo).decode().splitlines():
        added, deleted, path = line.split("\t")
        added, deleted = int(added), int(deleted)
        result[path] = {"added": added, "deleted": deleted,
                        "gross": added + deleted, "net": added - deleted}
    return {path: result.get(path, {"added": 0, "deleted": 0, "gross": 0, "net": 0})
            for path in paths} if paths else result


def names(a, b):
    return git("diff", "--no-renames", "--name-status", a, b, "--", ".",
               ":(exclude)task_memory").decode().splitlines()


def digest(bindings, exclude=()):
    return sha("\n".join(f"{p} {v}" for p, v in sorted(bindings.items())
                         if p not in exclude).encode())


def main():
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 ** 3, 30 * 1024 ** 3))
    primary_before, candidate_before = state(PRIMARY), state(CANDIDATE)
    require(candidate_before["HEAD"] == SOURCE_RECORDS, "candidate_HEAD_not_requested")
    prior = {}
    inputs = {}
    files = {
        "snapshot_manifest": (SNAP_AUDIT, "snapshot_scope/snapshot_blob_manifest.json"),
        "snapshot_receipt": (SNAP_AUDIT, "snapshot_scope/snapshot_baseline_receipt.json"),
        "key_receipt": (KEY_AUDIT, "plan_key_scope/plan_key_delta_receipt.json"),
        "key_guard": (KEY_AUDIT, "plan_key_scope/changed_file_guard.json"),
        "key_inventory": (KEY_AUDIT, "plan_key_scope/artifact_sha256.json"),
    }
    for name, (commit, suffix) in files.items():
        path = EVIDENCE + suffix
        raw = git("show", f"{commit}:{path}")
        require((RECORDS / path).read_bytes() == raw, f"prior_local_drift:{name}")
        require(git("show", f"{ACCEPTED}:{path}") == raw, f"prior_accepted_drift:{name}")
        inputs[name] = {"path": path, "commit": commit, "sha256": sha(raw), "bytes": len(raw)}
        prior[name] = json.loads(raw)
    manifest, snapshot, key = prior["snapshot_manifest"], prior["snapshot_receipt"], prior["key_receipt"]
    require(inputs["snapshot_manifest"]["sha256"] == "807fe446a7ec87c94b5b6e9f0b128ea6ffc41d7e90f11f80c03abfdf3ed6a4e0", "snapshot_manifest_hash")
    require(inputs["key_receipt"]["sha256"] == prior["key_inventory"]["files"]["plan_key_delta_receipt.json"]["sha256"], "key_receipt_hash")
    require(inputs["key_guard"]["sha256"] == prior["key_inventory"]["files"]["changed_file_guard.json"]["sha256"], "key_guard_hash")
    amendment_raw = git("show", f"{ALLOCATION}:{AMENDMENT}")
    amendment = json.loads(amendment_raw)
    require(amendment["step_base"] == STEP_BASE, "explicit_step_base")
    require(amendment["repair_paths"] == QUERY, "allocated_query_paths")
    require(amendment["owner_gross_ceilings"] == CAPS, "owner_caps_changed")
    git("merge-base", "--is-ancestor", STEP_BASE, ALLOCATION)
    for rev in [SOURCE, SOURCE_RECORDS]:
        require(git("show", f"{rev}:{AMENDMENT}") == amendment_raw, f"amendment_drift:{rev}")
    inputs["amendment"] = {"path": AMENDMENT, "commit": ALLOCATION, "sha256": sha(amendment_raw), "bytes": len(amendment_raw)}
    requirements = git("show", f"{primary_before['HEAD']}:{TASK}requirements.md")
    inputs["authority_requirements"] = {"path": TASK + "requirements.md", "commit": primary_before["HEAD"], "sha256": sha(requirements), "lines": "338-362"}

    expected_names = sorted("M\t" + path for path in QUERY)
    guards = []
    for a, b, expected in [(ACCEPTED, ALLOCATION, []), (ACCEPTED, STEP_BASE, []),
                           (ACCEPTED, primary_before["HEAD"], []), (KEY_SOURCE, KEY_MERGE, []),
                           (KEY_SOURCE, ACCEPTED, []), (ACCEPTED, SOURCE, expected_names),
                           (ALLOCATION, SOURCE, expected_names), (STEP_BASE, SOURCE, expected_names),
                           (SOURCE, SOURCE_RECORDS, [])]:
        actual = names(a, b)
        guards.append({"from": a, "to": b, "actual_non_task_memory": actual, "expected": expected, "pass": sorted(actual) == expected})
        require(sorted(actual) == expected, f"changed_file_guard:{a}:{b}")
    records_names = git("diff", "--no-renames", "--name-status", SOURCE, SOURCE_RECORDS).decode().splitlines()

    key_paths = {item["path"]: item for item in prior["key_guard"]["key_paths"]}
    super_paths = [x["path"] for x in manifest["paths"] if x["kind"] != "submodule_pinned"]
    evidence_paths = [x["path"] for x in manifest["fixed_non_source_evidence"]]
    trees = {rev: entries(rev, super_paths + evidence_paths)
             for rev in [ACCEPTED, ALLOCATION, STEP_BASE, SOURCE, SOURCE_RECORDS]}
    base_bindings, candidate_bindings, binding_rows = {}, {}, []
    for row in manifest["paths"]:
        path = row["path"]
        if row["kind"] == "submodule_pinned":
            relative = path[len(SUBPATH) + 1:]
            actual = entries(PIN, [relative], repo=SUBREPO)[relative]
            raw = git("cat-file", "blob", f"{PIN}:{relative}", repo=SUBREPO)
            label, expected = "submodule:" + path, row["snapshot"]
            require(all(actual[k] == expected[k] for k in ["mode", "type", "oid"]), f"submodule_binding:{path}")
            local_path = SUBREPO / relative
        else:
            label = path
            expected = copy.deepcopy(row["snapshot"])
            if path in key_paths:
                expected.update(oid=key_paths[path]["candidate_oid"], sha256=key_paths[path]["candidate_sha256"])
                expected["bytes"] = next(x["actual_total_bytes"] for x in key["paths"] if x["path"] == path)
            actual = trees[SOURCE][path]
            require(all(trees[ACCEPTED][path][k] == expected[k] for k in ["mode", "type", "oid"]), f"accepted_baseline_binding:{path}")
            for rev in [ALLOCATION, STEP_BASE]:
                require(trees[rev][path] == trees[ACCEPTED][path], f"base_binding_drift:{rev}:{path}")
            require(trees[SOURCE_RECORDS][path] == actual, f"record_binding_drift:{path}")
            if path not in QUERY:
                require(actual == trees[ACCEPTED][path], f"non_query_binding:{path}")
            raw = git("cat-file", "blob", f"{SOURCE}:{path}") if actual["type"] == "blob" else None
            local_path = CANDIDATE / path
        raw_sha = sha(raw) if raw is not None else None
        if path not in QUERY and raw is not None:
            require(raw_sha == expected["sha256"], f"bound_sha256:{path}")
        local_sha = sha(local_path.read_bytes()) if raw is not None else None
        require(local_sha == raw_sha, f"candidate_worktree_bytes:{path}")
        base_bindings[label], candidate_bindings[label] = expected["oid"], actual["oid"]
        binding_rows.append({"binding": label, "kind": row["kind"], "baseline": expected,
                             "candidate": actual, "candidate_sha256": raw_sha, "local_sha256": local_sha,
                             "bytes": len(raw) if raw is not None else None, "changed": actual["oid"] != expected["oid"]})
    for row in manifest["fixed_non_source_evidence"]:
        path = row["path"]
        actual = trees[SOURCE][path]
        raw = git("cat-file", "blob", f"{SOURCE}:{path}")
        for rev in trees:
            require(trees[rev][path]["oid"] == row["blob"], f"fixed_image_binding:{rev}:{path}")
            require(trees[rev][path] == trees[ACCEPTED][path], f"fixed_image_mode_type:{rev}:{path}")
        require(sha(raw) == row["sha256"], f"fixed_image_sha256:{path}")
        local_sha = sha((CANDIDATE / path).read_bytes())
        require(local_sha == row["sha256"], f"fixed_image_worktree:{path}")
        base_bindings[path], candidate_bindings[path] = row["blob"], actual["oid"]
        binding_rows.append({"binding": path, "kind": "fixed_image_evidence", "baseline_oid": row["blob"],
                             "candidate": actual, "candidate_sha256": sha(raw), "local_sha256": local_sha, "bytes": len(raw)})
    native_rows = []
    for row in manifest["fixed_native_source_pins"]:
        base, derived = Path(row["base_path"]).read_bytes(), Path(row["derived_path"]).read_bytes()
        require(sha(base) == row["base_sha256"], f"native_base_sha256:{row['path']}")
        require(sha(derived) == row["derived_sha256"], f"native_derived_sha256:{row['path']}")
        text = git("diff", "--no-index", "--numstat", row["base_path"], row["derived_path"], expected=(0, 1)).decode().strip()
        added, deleted, _ = text.split("\t")
        native_rows.append({**row, "fresh_base_sha256": sha(base), "fresh_derived_sha256": sha(derived),
                            "added": int(added), "deleted": int(deleted), "gross": int(added) + int(deleted), "net": int(added) - int(deleted)})
        label = "native:" + row["path"]
        base_bindings[label], candidate_bindings[label] = row["derived_sha256"], sha(derived)
        binding_rows.append({"binding": label, "kind": "native_derived", "baseline_sha256": row["derived_sha256"], "candidate_sha256": sha(derived)})
    require(len(base_bindings) == len(candidate_bindings) == 124, "binding_count_124")
    require(digest(base_bindings) == key["binding_guard"]["candidate_all_binding_digest"], "reconstructed_accepted_key_digest")
    require(digest(base_bindings, QUERY) == digest(candidate_bindings, QUERY), "120_non_query_digest")
    require(sum(x["changed"] for x in binding_rows if "changed" in x) == 4, "binding_changes_four_only")
    substat = numstats(ORIGINAL_PIN, PIN, repo=SUBREPO)
    require(len(substat) == 5 and sum(x["gross"] for x in substat.values()) == 333, "submodule_333_five_files")
    require(git("rev-parse", "HEAD", repo=SUBREPO).decode().strip() == PIN, "submodule_local_pin")
    require(sum(x["gross"] for x in native_rows) == 99 and len(native_rows) == 4, "native_99_four_files")
    nonsource = numstats(ORIGINAL, SOURCE, ["docs/general/module-size-exceptions.md", SUBPATH])
    require(nonsource["docs/general/module-size-exceptions.md"]["gross"] == 12 and nonsource[SUBPATH]["gross"] == 2, "doc12_gitlink2")
    require(trees[SOURCE][SUBPATH]["oid"] == PIN, "candidate_gitlink_pin")
    if VIOLATIONS:
        raise RuntimeError("Baseline cannot be carried: " + repr(VIOLATIONS))

    # Carry rows only after the complete byte-binding guard succeeds.
    ledger = copy.deepcopy(key["path_ledger"]["entries"])
    by_path = {x["path"]: x for x in ledger}
    baseline_owners = {owner: sum(x["actual"] for x in ledger if x["owner"] == owner) for owner in CAPS}
    require(baseline_owners == key["totals"]["owner_actual"], "accepted_ledger_owner_sum")
    require(sum(baseline_owners.values()) == 9477 and sum(baseline_owners.values()) + 14 == 9491, "accepted_ledger_source_global_sum")
    changes = amendment["path_ceiling_adjustments"]
    require(len(changes) == 8 and sum(x["delta"] for x in changes) == 0, "eight_zero_sum_transfers")
    require(sum(x["delta"] for x in changes if x["delta"] > 0) == 65, "transfer_65")
    transitions = []
    for change in changes:
        row = by_path[change["path"]]
        require(row["owner"] == "core_source", f"transfer_owner:{row['path']}")
        require(row["cap"] == change["before"], f"transfer_before:{row['path']}")
        require(change["before"] + change["delta"] == change["after"], f"transfer_arithmetic:{row['path']}")
        row["cap"] = change["after"]
        transitions.append({**change, "owner": row["owner"]})
    donors = [x["path"] for x in changes if x["delta"] < 0]
    cumulative = numstats(ORIGINAL, SOURCE, QUERY + donors)
    baseline_stats = numstats(ORIGINAL, ACCEPTED, QUERY + donors)
    step_stats = numstats(STEP_BASE, SOURCE, QUERY)
    baseline_actual = {p: x["actual"] for p, x in by_path.items()}
    for path, row in by_path.items():
        row["baseline_cap"] = next(x["cap"] for x in key["path_ledger"]["entries"] if x["path"] == path)
        row["baseline_actual"] = baseline_actual[path]
        if path in cumulative:
            require(baseline_stats[path]["gross"] == row["baseline_actual"], f"carried_numstat:{path}")
            row["actual"] = cumulative[path]["gross"]
            row["cumulative_numstat"] = cumulative[path]
            row["basis"] = "fresh_cumulative_git_numstat"
        else:
            row["basis"] = "accepted_key_ledger_after_unchanged_binding"
        row["headroom"] = row["cap"] - row["actual"]
        require(row["headroom"] >= 0, f"path_cap:{path}")
    owner_actual, owner_caps, owner_paths = {}, {}, {}
    for row in ledger:
        owner = row["owner"]
        owner_actual[owner] = owner_actual.get(owner, 0) + row["actual"]
        owner_caps[owner] = owner_caps.get(owner, 0) + row["cap"]
        owner_paths[owner] = owner_paths.get(owner, 0) + 1
    require(len(ledger) == len(by_path) == amendment["effective_unique_paths"] == 114, "effective_paths_114")
    require(owner_caps == CAPS, "composed_owner_caps")
    require(owner_paths == key["totals"]["owner_paths"], "owner_path_counts")
    require(all(owner_actual[o] <= CAPS[o] for o in CAPS), "owner_actual_caps")
    for path in donors:
        require(cumulative[path] == baseline_stats[path], f"donor_actual_drift:{path}")
    step = {name: sum(x[name] for x in step_stats.values()) for name in ["added", "deleted", "gross", "net"]}
    source_gross = sum(step_stats[p]["gross"] for p in QUERY[:2])
    test_gross = sum(step_stats[p]["gross"] for p in QUERY[2:])
    require(step["gross"] <= amendment["repair_delta_ceiling"], "step_gross_ceiling")
    require(source_gross <= amendment["source_delta_ceiling"], "source_step_ceiling")
    require(test_gross <= amendment["test_delta_ceiling"], "test_step_ceiling")
    for p in QUERY:
        require(step_stats[p]["gross"] <= amendment["path_step_ceilings"][p], f"path_step_ceiling:{p}")
    source_total = sum(owner_actual.values())
    global_total = source_total + sum(x["gross"] for x in nonsource.values())
    require(source_total <= 9942 and global_total <= 10000, "source_global_caps")
    observed = {"step_gross": step["gross"], "step_retained_net": step["net"],
                "cumulative_gross_growth": sum(cumulative[p]["gross"] - baseline_stats[p]["gross"] for p in QUERY),
                "owner_actual": owner_actual, "source_actual": source_total, "global_actual": global_total}
    hypotheses = {"step_gross": 96, "step_retained_net": 84, "cumulative_gross_growth": 88,
                  "owner_actual": {"core_source": 4261, "pricing": 3954, "native": 1350}, "source_actual": 9565, "global_actual": 9579}
    require(observed == hypotheses, "observed_vs_expected_totals")
    require([cumulative[p]["gross"] for p in QUERY] == [104, 196, 348, 147], "query_expected_cumulative")
    require([by_path[p]["cap"] for p in QUERY] == [120, 223, 353, 151], "query_expected_caps")
    primary_after, candidate_after = state(PRIMARY), state(CANDIDATE)
    require(primary_before == primary_after, "primary_HEAD_index_changed")
    require(candidate_before == candidate_after, "candidate_HEAD_index_changed")
    receipt = {
        "task_id": "D29-NAMED-QUERY-SCOPE-DELTA", "session_id": "ses_e95ac82fdffeSeMEUfR9SxvW03",
        "status": "PASS_NAMED_QUERY_ACCOUNTING_ONLY" if not VIOLATIONS else "FAIL",
        "audit_kind": "COMPOSED_114_PATH_ACCOUNTING_WITH_FOUR_PATH_DELTA",
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "bytecode_disabled": bool(sys.dont_write_bytecode), "wrapper": "timeout 60s python3 -B",
                        "address_space_limit_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
                        "cgroup_memory_max": "unavailable: system and user transient scopes both denied before checker execution",
                        "inspection_size": "Only named receipts and blobs; no memory-heavy reads or full scans"},
        "commits": {name: {"commit": rev, "tree": git("rev-parse", rev + "^{tree}").decode().strip()}
                    for name, rev in {"accepted_key": ACCEPTED, "key_owner_source": KEY_SOURCE, "key_merge": KEY_MERGE,
                                      "allocation": ALLOCATION, "step_base": STEP_BASE, "candidate_source": SOURCE,
                                      "candidate_records": SOURCE_RECORDS, "original": ORIGINAL}.items()},
        "input_bindings": inputs, "authority": {"requirements_lines_338_362": requirements.decode().splitlines()[337:362], "amendment": amendment},
        "changed_file_guards": guards, "source_to_records_name_status": records_names,
        "binding_guard": {"count": 124, "non_query_count": 120, "baseline_all_digest": digest(base_bindings),
                          "candidate_all_digest": digest(candidate_bindings), "baseline_non_query_digest": digest(base_bindings, QUERY),
                          "candidate_non_query_digest": digest(candidate_bindings, QUERY), "rows": binding_rows,
                          "algorithm": "SHA256 of sorted 'path oid' lines joined by LF without terminal LF; submodule:<superproject-path> uses pinned blob OID; native:<path> uses derived SHA256"},
        "overlay_sequence": [{"ordinal": i + 1, "file": x["file"], "decision_id": x["decision_id"],
                              "basis": "carried_from_accepted_snapshot_audit", "before": x["before"], "after": x["after"]}
                             for i, x in enumerate(snapshot["overlays"])] + [{"ordinal": 16, "file": AMENDMENT,
                             "decision_id": amendment["decision_id"], "basis": "applied_to_accepted_key_ledger",
                             "apply_after": amendment["apply_after"], "transitions": transitions,
                             "transferred": 65, "zero_sum": sum(x["delta"] for x in changes)}],
        "step": {"numstat": step, "source_gross": source_gross, "tests_gross": test_gross,
                 "gross_cap": amendment["repair_delta_ceiling"], "source_cap": amendment["source_delta_ceiling"], "tests_cap": amendment["test_delta_ceiling"],
                 "retained_growth_ceiling": amendment.get("retained_growth_ceiling"), "cumulative_gross_growth": observed["cumulative_gross_growth"],
                 "distinction": "96 step gross, 84 physical retained net, 88 cumulative budget-gross growth; no separate retained-growth cap is declared by this amendment",
                 "paths": [{"path": p, "owner": by_path[p]["owner"], "step": step_stats[p], "step_cap": amendment["path_step_ceilings"][p],
                            "baseline": baseline_stats[p], "cumulative": cumulative[p], "cap": by_path[p]["cap"], "headroom": by_path[p]["headroom"]} for p in QUERY]},
        "donors": [{"path": p, "before": baseline_stats[p], "candidate": cumulative[p], "cap_before": by_path[p]["baseline_cap"],
                    "cap_after": by_path[p]["cap"], "headroom": by_path[p]["headroom"]} for p in donors],
        "path_ledger": ledger, "allocated_paths": len(ledger), "changed_paths": key["allocated_vs_changed"]["changed_paths"],
        "totals": {"owner_paths": owner_paths, "owner_caps": owner_caps, "owner_actual": owner_actual,
                   "baseline_owner_actual": key["totals"]["owner_actual"], "baseline_source": key["totals"]["source_actual_after_key"],
                   "baseline_global": key["totals"]["global_actual_after_key"], "source_actual": source_total, "source_cap": 9942,
                   "source_headroom": 9942 - source_total, "global_actual": global_total, "global_cap": 10000, "global_headroom": 10000 - global_total},
        "non_source": {"doc_and_gitlink": nonsource, "submodule_pin": PIN, "submodule_original_pin": ORIGINAL_PIN,
                       "submodule_numstats": substat, "submodule_gross": sum(x["gross"] for x in substat.values()),
                       "native_rows": native_rows, "native_gross": sum(x["gross"] for x in native_rows),
                       "fixed_evidence_count": len(evidence_paths), "live_image_inspection": False, "native_rebuild": False},
        "preserved_states": {"primary_before": primary_before, "primary_after": primary_after,
                             "candidate_before": candidate_before, "candidate_after": candidate_after},
        "hypotheses": hypotheses, "observed": observed, "violations": VIOLATIONS, "verification_commands": COMMANDS,
        "limits": ["Accounting only; no tests, probes, project imports, CSV/pickle/trace/model reads, network, GPU, package operations or production edits.",
                   "Only four query paths and six donors receive fresh cumulative Git numstats; the other 104 accepted ledger rows are carried after the full binding guard.",
                   "Historical overlays/checkpoint diagnostics are bound to accepted audit hashes, not re-audited. Native image identity is pinned evidence, not a fresh registry/worker inspection.",
                   "No full-workflow/runtime/review verdict. Parent pytest/wave and final integrated-head freeze remain separate."]}
    (OUT / "named_query_scope_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    fieldnames = ["path", "owner", "kind", "baseline_cap", "cap", "baseline_actual", "actual", "headroom", "basis"]
    (OUT / "named_query_paths.tsv").write_text("\t".join(fieldnames) + "\n" + "".join("\t".join(str(row[f]) for f in fieldnames) + "\n" for row in ledger))
    guard = {"task_id": receipt["task_id"], "accepted": ACCEPTED, "candidate": SOURCE, "records": SOURCE_RECORDS,
             "expected_non_task_memory_name_status": expected_names, "baseline_all_digest": digest(base_bindings),
             "candidate_all_digest": digest(candidate_bindings), "non_query_count": 120,
             "non_query_digest": digest(candidate_bindings, QUERY), "query_bindings": [x for x in binding_rows if x["binding"] in QUERY]}
    (OUT / "changed_file_guard.json").write_text(json.dumps(guard, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "observed": observed, "non_query_digest": guard["non_query_digest"], "violations": VIOLATIONS}, indent=2))
    return 0 if not VIOLATIONS else 1


if __name__ == "__main__":
    sys.exit(main())
