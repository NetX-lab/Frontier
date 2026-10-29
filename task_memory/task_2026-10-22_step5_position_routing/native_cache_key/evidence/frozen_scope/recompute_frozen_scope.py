"""Compose the native-key successor using bound ledgers, not a full re-audit."""

from collections import Counter
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


OUT = Path(__file__).resolve().parent
OWN = OUT.parents[4]
TASK = "task_memory/task_2026-10-22_step5_position_routing/"
PRIMARY = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
CANDIDATE = Path("/data/ycfeng/Frontier/.worktrees/step5-d29-native-cache-key")
SOURCE = "376983d2ecc441d1e0a8cb8b943e65a2dcb217b5"
FIXED = "9870597776229d84af5eb58d49bf9b1756092999"
STEP_BASE = "18cc3e9e1008d796a08f3b5637c9b10311979735"
ORIGINAL = "1aee0af11cd58691e64603b99eca0ac459a5b350"
AUDIT = "de5eeab05a2f312da415a9e6557847ab583b4990"
SNAP_AUDIT = "d267db2d7d337dc237c85ea63f16cfa0290eb42d"
SUB = "frontier/cc_backend/backends/collective-sim"
PIN = "3dac53c544e546574cb6e8078cd14053d0916113"
OLD_PIN = "d28fe917370c2d55230c3ea226e5660785b44aca"
PATHS = ["frontier/cc_backend/backends/collective_sim_cc_backend.py", "tests/unit/test_collective_sim_cc_backend.py"]
DONOR = "frontier/cc_backend/collective_plan.py"
CAPS = {"core_source": 4404, "pricing": 3956, "native": 1582}
PRIOR_DIR = TASK + "repairs/evidence/named_query_scope/"
AMENDMENT = TASK + "repairs/native_collective_key_scope_amendment.json"
INPUTS, COMMANDS = {}, []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, repo=OWN, exits=(0,)):
    argv = ["git", "--no-optional-locks", "-C", str(repo), *args]
    result = subprocess.run(argv, capture_output=True, timeout=60)
    COMMANDS.append({"argv": argv, "returncode": result.returncode, "timeout_seconds": 60,
                     "stdout_sha256": sha(result.stdout), "stdout_bytes": len(result.stdout)})
    assert result.returncode in exits, (argv, result.returncode, result.stderr.decode())
    return result.stdout


def bound(path, commit=None):
    path = Path(path)
    raw = path.read_bytes()
    item = {"path": str(path), "sha256": sha(raw), "bytes": len(raw)}
    if commit:
        relative = str(path.relative_to(PRIMARY))
        assert raw == git("show", f"{commit}:{relative}")
        item.update(commit=commit, relative_path=relative)
    INPUTS[str(path)] = item
    return raw


def state(repo, clean=False):
    index = Path(git("rev-parse", "--git-path", "index", repo=repo).decode().strip())
    if not index.is_absolute():
        index = repo / index
    result = {"head": git("rev-parse", "HEAD", repo=repo).decode().strip(),
              "index_path": str(index), "index_sha256": sha(index.read_bytes())}
    if clean:
        result["status"] = git("status", "--porcelain=v1", "--untracked-files=all", repo=repo).decode().strip()
        assert not result["status"], result
    return result


def main():
    started = time.monotonic()
    assert os.getuid() == 10250 and sys.dont_write_bytecode
    cgroup = Path("/proc/self/cgroup").read_text()
    unit = next(row.rsplit("/", 1)[-1] for row in cgroup.splitlines() if row.startswith("1:name=systemd:"))
    properties = subprocess.run(["systemctl", "show", unit, "-p", "MemoryMax", "-p", "MemoryLimit"], capture_output=True, text=True, check=True, timeout=60).stdout
    memory = dict(row.split("=", 1) for row in properties.splitlines())
    assert int(memory["MemoryMax"]) == 30 * 1024 ** 3
    before = {"primary": state(PRIMARY), "candidate": state(CANDIDATE, clean=True),
              "submodule": state(CANDIDATE / SUB, clean=True)}
    assert before["candidate"]["head"] == SOURCE and before["submodule"]["head"] == PIN
    assert git("rev-parse", SOURCE + "^").decode().strip() == FIXED

    # Primary task documents are current authority, not a source snapshot.
    authority = {}
    for suffix in ["requirements.md", "plan.md", "native_cache_key/requirements.md", "native_cache_key/plan.md"]:
        path = PRIMARY / TASK / suffix
        bound(path)
        authority[suffix] = {**INPUTS[str(path)], "read_from": "current_primary_worktree", "primary_head_at_read": before["primary"]["head"]}
    inventory_raw = bound(PRIMARY / PRIOR_DIR / "artifact_sha256.json", AUDIT)
    inventory = json.loads(inventory_raw)
    for name, expected in inventory["files"].items():
        raw = bound(PRIMARY / PRIOR_DIR / name, AUDIT)
        assert sha(raw) == expected["sha256"] and len(raw) == expected["bytes"]
    prior = json.loads((PRIMARY / PRIOR_DIR / "named_query_scope_receipt.json").read_text())
    assert not prior["violations"] and prior["allocated_paths"] == 114
    accepted = prior["commits"]["candidate_source"]["commit"]
    ledger = copy.deepcopy(prior["path_ledger"])
    tsv = list(csv.DictReader((PRIMARY / PRIOR_DIR / "named_query_paths.tsv").open(), delimiter="\t"))
    assert len(tsv) == len(ledger) == len({x["path"] for x in ledger}) == 114
    by_path = {x["path"]: x for x in ledger}
    for row in tsv:
        item = by_path[row["path"]]
        assert row["owner"] == item["owner"] and row["kind"] == item["kind"]
        assert all(int(row[k]) == item[k] for k in ["cap", "actual", "headroom"])
    primitives = runpy.run_path(str(PRIMARY / PRIOR_DIR / "recompute_named_query_scope.py"), run_name="native_key_accounting_primitives")
    digest, entries, numstats = (primitives[k] for k in ["digest", "entries", "numstats"])
    snapshot_path = PRIMARY / TASK / "repairs/evidence/snapshot_scope/recompute_snapshot.py"
    bound(snapshot_path, SNAP_AUDIT)
    allocation = runpy.run_path(str(snapshot_path), run_name="native_key_allocation_primitives")["allocation"]
    manifest_path = PRIMARY / TASK / "repairs/evidence/snapshot_scope/snapshot_blob_manifest.json"
    manifest = json.loads(bound(manifest_path, SNAP_AUDIT))
    amendment = json.loads(bound(PRIMARY / AMENDMENT))
    assert git("show", f"{SOURCE}:{AMENDMENT}") == (PRIMARY / AMENDMENT).read_bytes()
    assert amendment["step_base"] == STEP_BASE and amendment["repair_paths"] == PATHS
    assert amendment["owner_gross_ceilings"] == CAPS and amendment["effective_unique_paths"] == 114

    guards = []
    expected = sorted("M\t" + p for p in PATHS)
    for a, b, wanted in [(accepted, STEP_BASE, []), (accepted, FIXED, []), (STEP_BASE, FIXED, []),
                          (accepted, SOURCE, expected), (STEP_BASE, SOURCE, expected), (FIXED, SOURCE, expected)]:
        rows = git("diff", "--no-renames", "--name-status", a, b, "--", ".", ":(exclude)task_memory").decode().splitlines()
        assert sorted(rows) == wanted, (a, b, rows)
        guards.append({"from": a, "to": b, "non_task_memory_name_status": rows, "expected": wanted, "pass": True})

    unprefixed = [x["binding"] for x in prior["binding_guard"]["rows"] if not x["binding"].startswith(("native:", "submodule:"))]
    trees = {rev: entries(rev, unprefixed, repo=OWN) for rev in [accepted, STEP_BASE, FIXED, SOURCE]}
    baseline, successor, binding_rows = {}, {}, []
    for row in prior["binding_guard"]["rows"]:
        label = row["binding"]
        if label.startswith("native:"):
            continue
        if label.startswith("submodule:"):
            path = label.removeprefix("submodule:")
            relative = path.removeprefix(SUB + "/")
            old = new = entries(PIN, [relative], repo=CANDIDATE / SUB)[relative]
            assert new == row["candidate"]
            raw = git("cat-file", "blob", new["oid"], repo=CANDIDATE / SUB)
            local = CANDIDATE / path
        else:
            old, new = trees[accepted][label], trees[SOURCE][label]
            assert old == row["candidate"]
            assert trees[STEP_BASE][label] == trees[FIXED][label] == old
            raw = git("cat-file", "blob", new["oid"]) if new["type"] == "blob" else None
            local = CANDIDATE / label
        if label not in PATHS:
            assert old == new, label
        if raw is not None:
            assert local.read_bytes() == raw, label
            if label not in PATHS:
                assert sha(raw) == row["candidate_sha256"], label
        baseline[label], successor[label] = old["oid"], new["oid"]
        binding_rows.append({"binding": label, "baseline": old, "candidate": new,
                             "sha256": sha(raw) if raw is not None else None,
                             "bytes": len(raw) if raw is not None else None, "changed": old != new})
    native_rows = []
    for pin in manifest["fixed_native_source_pins"]:
        for side in ["base", "derived"]:
            assert sha(bound(pin[side + "_path"])) == pin[side + "_sha256"]
        stat = git("diff", "--no-index", "--numstat", pin["base_path"], pin["derived_path"], exits=(0, 1)).decode().strip()
        added, deleted, _ = stat.split("\t")
        bill = {"added": int(added), "deleted": int(deleted), "gross": int(added) + int(deleted), "net": int(added) - int(deleted)}
        old_bill = next(x for x in prior["non_source"]["native_rows"] if x["path"] == pin["path"])
        assert all(bill[k] == old_bill[k] for k in bill)
        native_rows.append({**pin, **bill})
        label = "native:" + pin["path"]
        baseline[label] = successor[label] = pin["derived_sha256"]
        binding_rows.append({"binding": label, "baseline_sha256": pin["derived_sha256"], "candidate_sha256": pin["derived_sha256"], "changed": False})
    assert len(baseline) == len(successor) == len(binding_rows) == 124
    fixed_evidence_count = sum(x["kind"] == "fixed_image_evidence" for x in prior["binding_guard"]["rows"])
    assert fixed_evidence_count == len(manifest["fixed_non_source_evidence"]) == 8
    assert sum(x["changed"] for x in binding_rows) == 2
    assert digest(baseline) == prior["binding_guard"]["candidate_all_digest"]
    assert digest(baseline, PATHS) == digest(successor, PATHS)
    substat = numstats(OLD_PIN, PIN, repo=CANDIDATE / SUB)
    assert len(substat) == 5 and sum(x["gross"] for x in substat.values()) == 333
    assert {SUB + "/" + p for p in substat} == {x["path"] for x in ledger if x["kind"] == "submodule_pinned"}
    assert len(native_rows) == 4 and sum(x["gross"] for x in native_rows) == 99
    non_source = numstats(ORIGINAL, SOURCE, ["docs/general/module-size-exceptions.md", SUB], repo=OWN)
    assert non_source["docs/general/module-size-exceptions.md"]["gross"] == 12 and non_source[SUB]["gross"] == 2
    assert not primitives["VIOLATIONS"]

    # Replay allocations only. Historical actual rows are not re-diffed.
    scope_path = PRIMARY / TASK / "scope.json"
    scope = json.loads(bound(scope_path))
    approved_raw = git("show", f"{scope['approved_ledger_commit']}:{TASK}scope.json")
    assert sha(approved_raw) == scope["approved_ledger_sha256"] and json.loads(approved_raw)["rows"] == scope["rows"]
    grouped = {}
    for row in scope["rows"]:
        grouped.setdefault(row["path"], []).append(row)
    caps = {}
    # Preserve the original billing model's two explicit shared-path owners.
    pricing_shared = {"frontier/execution_time_predictor/moe_operator_times.py", "frontier/profiling/moe/fp8_block_experts.py"}
    for path, rows in grouped.items():
        owners = {x["owner"] for x in rows}
        owner = "native" if "native" in owners else "pricing" if path in pricing_shared else "core_source" if owners & {"core", "source"} else "pricing"
        caps[path] = {"owner": owner, "cap": sum(x["ceiling"] for x in rows)}
    assert allocation(caps) == {"paths": 107, "owner_paths": {"core_source": 44, "native": 19, "pricing": 44}, "owner_caps": CAPS, "source_cap": 9942}
    overlays = []
    for descriptor in prior["overlay_sequence"] + [{"file": AMENDMENT}]:
        path = PRIMARY / descriptor["file"]
        raw = bound(path, accepted if descriptor["file"] != AMENDMENT else None)
        item = json.loads(raw)
        before_caps = allocation(caps)
        if overlays and item.get("apply_after"):
            assert item["apply_after"] == Path(overlays[-1]["file"]).name
        transitions = []
        for row in item.get("new_paths", []):
            assert row["path"] not in caps
            caps[row["path"]] = {"owner": row["owner"], "cap": row["ceiling"]}
            transitions.append({"path": row["path"], "before": 0, "delta": row["ceiling"], "after": row["ceiling"], "owner": row["owner"], "new_path": True})
        for row in item.get("existing_path_ceiling_adjustments", []) + item.get("path_ceiling_adjustments", []):
            entry = caps[row["path"]]
            assert entry["cap"] == row["before"] and row["before"] + row["delta"] == row["after"] >= 0
            transitions.append({**row, "owner": entry["owner"], "new_path": False})
            entry["cap"] = row["after"]
        after_caps = allocation(caps)
        zero_sum = {owner: sum(x["delta"] for x in transitions if x["owner"] == owner) for owner in CAPS}
        assert after_caps["owner_caps"] == CAPS and after_caps["paths"] == item["effective_unique_paths"] and not any(zero_sum.values())
        overlays.append({"file": descriptor["file"], "decision_id": item["decision_id"], "before": before_caps, "after": after_caps,
                         "transitions": transitions, "zero_sum_by_owner": zero_sum, "step_caps_not_measurements": {k: item[k] for k in ["repair_delta_ceiling", "source_delta_ceiling", "test_delta_ceiling", "retained_growth_ceiling"] if k in item}})
        if descriptor["file"] != AMENDMENT:
            if "before" in descriptor:
                assert before_caps == descriptor["before"] and after_caps == descriptor["after"]
        else:
            assert before_caps["paths"] == 114
    assert len(overlays) == 17 and set(caps) == set(by_path)
    adjustments = amendment["path_ceiling_adjustments"]
    assert adjustments == [{"path": DONOR, "before": 140, "delta": -1, "after": 139}, {"path": PATHS[1], "before": 236, "delta": 1, "after": 237}]
    for path, row in by_path.items():
        expected = next((x["after"] for x in adjustments if x["path"] == path), row["cap"])
        assert caps[path] == {"owner": row["owner"], "cap": expected}

    cumulative = numstats(ORIGINAL, SOURCE, PATHS + [DONOR], repo=OWN)
    previous = numstats(ORIGINAL, accepted, PATHS + [DONOR], repo=OWN)
    step = numstats(FIXED, SOURCE, PATHS, repo=OWN)
    assert step == numstats(STEP_BASE, SOURCE, PATHS, repo=OWN)
    assert [(step[p]["added"], step[p]["deleted"] ) for p in PATHS] == [(1, 1), (1, 0)]
    assert sum(x["gross"] for x in step.values()) == amendment["repair_delta_ceiling"] == 3
    assert sum(x["net"] for x in step.values()) == amendment["retained_growth_ceiling"] == 1
    assert step[PATHS[0]]["gross"] == amendment["source_delta_ceiling"] == 2
    assert step[PATHS[1]]["gross"] == amendment["test_delta_ceiling"] == 1
    old_owners = Counter()
    for row in ledger:
        old_owners[row["owner"]] += row["actual"]
        if row["path"] in previous:
            assert previous[row["path"]]["gross"] == row["actual"]
        row["baseline_actual"], row["baseline_cap"] = row["actual"], row["cap"]
        row["cap"] = caps[row["path"]]["cap"]
        if row["path"] in PATHS:
            row["actual"] = cumulative[row["path"]]["gross"]
            row["basis"] = "fresh_two_path_cumulative_git_numstat"
        else:
            row["basis"] = "accepted_named_query_actual_after_unchanged_binding"
        row["headroom"] = row["cap"] - row["actual"]
        assert row["headroom"] >= 0
    assert cumulative[DONOR] == previous[DONOR] and by_path[DONOR]["actual"] == 139
    owner_actual = {owner: sum(x["actual"] for x in ledger if x["owner"] == owner) for owner in CAPS}
    assert dict(old_owners) == prior["totals"]["owner_actual"]
    assert [by_path[p]["actual"] for p in PATHS + [DONOR]] == [93, 237, 139]
    assert owner_actual == {"core_source": 4261, "pricing": 3955, "native": 1350}
    source_total, global_total = sum(owner_actual.values()), sum(owner_actual.values()) + sum(x["gross"] for x in non_source.values())
    assert source_total == 9566 <= 9942 and global_total == 9580 <= 10000
    assert sum(x["actual"] > 0 for x in ledger) == 113
    assert sum(cumulative[p]["gross"] - previous[p]["gross"] for p in PATHS) == 1
    owner_receipt_path = Path("/data/ycfeng/tmp/d29_native_cache_key_20261024.xBHOmb/scope_check.json")
    owner_receipt = json.loads(bound(owner_receipt_path))
    bound(owner_receipt_path.with_name("scope_check.py"))
    assert sha(owner_receipt_path.with_name("scope_check.py").read_bytes()) == owner_receipt["script_sha256"]
    assert owner_receipt["owner_actual"] == owner_actual and owner_receipt["owner_caps"] == CAPS
    assert owner_receipt["non_key_successor_digest"] == digest(successor, PATHS)
    after = {"primary": state(PRIMARY), "candidate": state(CANDIDATE, clean=True), "submodule": state(CANDIDATE / SUB, clean=True)}
    assert before == after and not primitives["VIOLATIONS"]
    receipt = {"task_id": "D29-NATIVE-KEY-SCOPE-DELTA", "session_id": "ses_e95ac82fdffeSeMEUfR9SxvW03", "verdict": "PASS_FROZEN_NATIVE_KEY_SCOPE_DELTA",
               "audit_kind": "BOUND_LEDGER_PLUS_EXACT_TWO_PATH_SUCCESSOR", "source_commit": SOURCE,
               "source_tree": git("rev-parse", SOURCE + "^{tree}").decode().strip(), "fixedpoint": FIXED, "step_base": STEP_BASE,
               "accepted_named_source": accepted, "original_billing_base": ORIGINAL, "authority": authority, "input_bindings": INPUTS,
               "changed_file_guards": guards, "bindings": {"count": 124, "unchanged_count": 122, "changed_count": 2,
               "baseline_all_digest": digest(baseline), "candidate_all_digest": digest(successor), "baseline_non_key_digest": digest(baseline, PATHS),
               "candidate_non_key_digest": digest(successor, PATHS), "algorithm": prior["binding_guard"]["algorithm"], "rows": binding_rows},
               "overlay_sequence": overlays, "step": {"paths": step, "gross": 3, "source_gross": 2, "test_gross": 1, "retained_net": 1, "cumulative_gross_increment": 1,
               "distinction": "Replacement costs2 step gross while cumulative backend gross remains93; the assertion adds1 cumulative gross. Never add step3 to prior9565."},
               "fresh_cumulative": cumulative, "fresh_accepted_cumulative": previous, "path_ledger": ledger, "allocated_paths": 114, "changed_paths": 113,
               "totals": {"baseline_owner_actual": dict(old_owners), "owner_actual": owner_actual, "owner_caps": CAPS, "owner_paths": allocation(caps)["owner_paths"],
               "source_actual": source_total, "source_cap": 9942, "source_headroom": 9942-source_total, "global_actual": global_total, "global_cap": 10000, "global_headroom": 10000-global_total},
               "non_source": {"doc_and_gitlink": non_source, "submodule_pin": PIN, "submodule_original": OLD_PIN, "submodule_paths": substat, "submodule_gross": 333,
               "native_rows": native_rows, "native_patch_gross": 99, "fixed_evidence_count": fixed_evidence_count, "native_rebuild": False, "live_image_inspection": False},
               "states_before": before, "states_after": after, "violations": [], "verification_commands": COMMANDS + primitives["COMMANDS"],
               "environment": {"uid": os.getuid(), "python": sys.version, "executable": sys.executable, "bytecode_disabled": True, "MemoryMax_bytes": int(memory["MemoryMax"]),
               "timeout_seconds": 60, "cgroup": cgroup, "fallback_used": False}, "duration_seconds": time.monotonic()-started,
               "limits": ["Accounting/bindings only, not implementation review, native admission, regression or CLI acceptance.",
               "No tests/waves/benchmarks, project imports, largeCSV/trace/model/pickle reads, network/GPU/platform operations or production edits.",
               "112 actual rows carried only after124-binding/exact two-path guard; only2 changed paths+unedited donor receive cumulative Git counts.",
               "All17 overlays replay allocations only; historical diagnostic/checkpoint actuals are not re-audited.",
               "Primary task working files are current authority; dirty primary source is neither required nor used as the candidate.",
               "Pinned native base/derived files and eight source/patch/image evidence bindings were freshly checked; no registry/image rebuild or runtime qualification.",
               "The earlier owner dirty-patch receipt is retained as historical corroboration, not the final frozen-target guard."]}
    (OUT / "frozen_scope_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    fields = ["path", "owner", "kind", "baseline_cap", "cap", "baseline_actual", "actual", "headroom", "basis"]
    (OUT / "frozen_scope_paths.tsv").write_text("\t".join(fields)+"\n"+"".join("\t".join(str(row[k]) for k in fields)+"\n" for row in ledger))
    summary = {key: receipt[key] for key in ["task_id", "verdict", "audit_kind", "source_commit", "source_tree", "fixedpoint", "step_base",
               "accepted_named_source", "allocated_paths", "changed_paths", "step", "totals", "environment", "violations", "limits"]}
    summary["bindings"] = {key: value for key, value in receipt["bindings"].items() if key != "rows"}
    summary["candidate_clean_before_after"] = not before["candidate"]["status"] and not after["candidate"]["status"]
    summary["overlay_count"] = len(overlays)
    summary["new_overlay"] = overlays[-1]
    summary["non_source_gross"] = {"submodule": 333, "native_patch": 99, "doc": 12, "gitlink": 2}
    summary["evidence"] = {name: {"sha256": sha((OUT / name).read_bytes()), "bytes": (OUT / name).stat().st_size}
                           for name in ["frozen_scope_receipt.json", "frozen_scope_paths.tsv"]}
    (OUT / "frozen_scope_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"verdict": receipt["verdict"], "source_tree": receipt["source_tree"], "owner_actual": owner_actual, "source_actual": source_total,
                      "global_actual": global_total, "non_key_digest": digest(successor, PATHS), "duration_seconds": receipt["duration_seconds"], "violations": []}, indent=2))


if __name__ == "__main__":
    main()
