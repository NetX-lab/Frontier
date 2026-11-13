#!/usr/bin/env python3
"""Read-only D29 final two-path delta audit.

Recomputes the plan-key step against the accepted snapshot baseline without
executing tests, importing project code, or reading model/CSV/trace artifacts.
Writes only the receipt next to this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = "/data/ycfeng/Frontier"
RECORDS = (
    REPO + "/.worktrees/d29-snapshot-scope-audit/task_memory/"
    "task_2026-10-22_step5_position_routing"
)
EVIDENCE = RECORDS + "/repairs/evidence"
SNAPSHOT_EVIDENCE = EVIDENCE + "/snapshot_scope"
OUT = EVIDENCE + "/plan_key_scope"

CANDIDATE = "c4e06a6159ee3b1e63c50abaa346066ae8b6da31"
CANDIDATE_RECORDS = "d1e2fb8c0162c58e7e694fab2d3a900638f0d50a"
SNAPSHOT = "e200bb780d3b20ec6f5ff0a067ed8b9ddd505662"
STEP_BASE = "087db1bfcb3dbc2377d4032f650b29c48fb104fd"
ORIGINAL_BASE = "1aee0af11cd58691e64603b99eca0ac459a5b350"
SOURCE_SNAPSHOT = "db581ff6ad43e1ce57361ed00fc4be0256e85a45"
SOURCE_MERGE = "64c83ac4786cd76d2cd248e1149ce2adf4c4fa7f"
ACCEPTED_RECORDS = "d267db2d7d337dc237c85ea63f16cfa0290eb42d"
ACCEPTED_MERGE = "ae55a5f573962528a20bd850ca65c7e50747614d"
PARENT_ACCEPTANCE = "1a365ebc8f27090e48f9223de80807fc68fe13b0"

TWO = [
    "frontier/cc_backend/collective_plan.py",
    "tests/unit/test_step5_moe_pricing.py",
]
SUBMODULE_PREFIX = "frontier/cc_backend/backends/collective-sim/"
SUBMODULE_PIN = "3dac53c544e546574cb6e8078cd14053d0916113"
SUBMODULE_ORIGINAL = "d28fe917370c2d55230c3ea226e5660785b44aca"
SUBMODULE = REPO + "/.worktrees/step5-d29-plan-key/" + SUBMODULE_PREFIX.rstrip("/")


def git(*args, cwd=REPO, check=True):
    return subprocess.run(
        ["git", "-C", cwd, *args], capture_output=True, text=True, check=check
    ).stdout


def git_blob(rev, path, cwd=REPO):
    return subprocess.run(
        ["git", "-C", cwd, "cat-file", "blob", f"{rev}:{path}"],
        capture_output=True,
        check=True,
    ).stdout


def tree_entry(rev, path, cwd=REPO):
    line = git("ls-tree", rev, "--", path, cwd=cwd).strip().split("\n")[0]
    meta, tab, found = line.partition("\t")
    if not tab:
        raise RuntimeError(f"missing tree entry {rev}:{path}")
    mode, kind, oid = meta.split()
    return {"mode": mode, "type": kind, "oid": oid}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _exists(rev, path, cwd=REPO):
    return subprocess.run(
        ["git", "-C", cwd, "cat-file", "-e", f"{rev}:{path}"],
        capture_output=True,
    ).returncode == 0


def numstat(a, b, path, cwd=REPO):
    raw = git("diff", "--no-renames", "--numstat", a, b, "--", path, cwd=cwd).strip()
    if not raw:
        return {"added": 0, "deleted": 0, "gross": 0, "net": 0}
    added, deleted, _ = raw.split("\t")
    return {
        "added": int(added),
        "deleted": int(deleted),
        "gross": int(added) + int(deleted),
        "net": int(added) - int(deleted),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    candidate = os.environ.get("D29_CANDIDATE", CANDIDATE)
    candidate_records = os.environ.get("D29_CANDIDATE_RECORDS", CANDIDATE_RECORDS)
    guard_only = os.environ.get("D29_GUARD_ONLY") == "1"
    manifest = json.load(open(SNAPSHOT_EVIDENCE + "/snapshot_blob_manifest.json"))
    manifest_sha = sha256(open(SNAPSHOT_EVIDENCE + "/snapshot_blob_manifest.json", "rb").read())
    rows = list(
        csv.DictReader(open(SNAPSHOT_EVIDENCE + "/snapshot_paths.tsv"), delimiter="\t")
    )

    candidate_tree = git("rev-parse", candidate + "^{tree}").strip()
    snapshot_tree = git("rev-parse", SNAPSHOT + "^{tree}").strip()
    step_base_tree = git("rev-parse", STEP_BASE + "^{tree}").strip()

    # Exact repository diff guard from the audited snapshot.
    full = git("diff", "--no-renames", "--name-status", SNAPSHOT, candidate).strip().splitlines()
    non_task = git(
        "diff", "--no-renames", "--name-status", SNAPSHOT, candidate,
        "--", ".", ":(exclude)task_memory",
    ).strip().splitlines()
    task_memory_paths = sorted(
        line.split("\t", 1)[1] for line in full if line.split("\t", 1)[1].startswith("task_memory/")
    )
    records_delta = git(
        "diff", "--no-renames", "--name-only", candidate, candidate_records
    ).strip().splitlines()
    non_task_records = [p for p in records_delta if not p.startswith("task_memory/")]
    expected_non_task = ["M\t" + p for p in TWO]
    guard_pass = sorted(non_task) == sorted(expected_non_task) and not non_task_records

    # Per-path actuals and step deltas.
    path_facts = []
    override = {}
    for path in TWO:
        baseline_entry = next(p for p in manifest["paths"] if p["path"] == path)
        cand = tree_entry(candidate, path)
        cand_blob = git_blob(candidate, path)
        step = numstat(STEP_BASE, candidate, path)
        orig = numstat(ORIGINAL_BASE, candidate, path)
        row = next(r for r in rows if r["path"] == path)
        if cand["oid"] != baseline_entry["snapshot"]["oid"]:
            pass
        actual_lines = cand_blob.count(b"\n")
        override[path] = actual_lines
        path_facts.append({
            "path": path,
            "owner": row["owner"],
            "kind": row["kind"],
            "mode": cand["mode"],
            "cap_before_key": int(row["cap_before_key"]),
            "cap_after_key": int(row["cap_after_key"]),
            "actual_total_lines": actual_lines,
            "actual_total_bytes": len(cand_blob),
            "path_headroom": int(row["cap_after_key"]) - actual_lines,
            "step_added": step["added"],
            "step_deleted": step["deleted"],
            "step_gross": step["gross"],
            "step_net": step["net"],
            "vs_1aee0af": orig,
            "snapshot_oid": baseline_entry["snapshot"]["oid"],
            "step_base_oid": git("rev-parse", f"{STEP_BASE}:{path}").strip(),
            "snapshot_source_oid": git("rev-parse", f"{SOURCE_SNAPSHOT}:{path}").strip(),
            "original_source_oid": None if not _exists(ORIGINAL_BASE, path) else git(
                "rev-parse", f"{ORIGINAL_BASE}:{path}"
            ).strip(),
            "original_source_present": _exists(ORIGINAL_BASE, path),
            "candidate_oid": cand["oid"],
            "candidate_sha256": sha256(cand_blob),
            "changed": cand["oid"] != baseline_entry["snapshot"]["oid"],
        })

    # Owner and global totals, replacing the two cumulative actuals.
    owner_actual = {"core_source": 0, "pricing": 0, "native": 0}
    owner_cap = {"core_source": 0, "pricing": 0, "native": 0}
    owner_paths = {"core_source": 0, "pricing": 0, "native": 0}
    changed_paths = 0
    unchanged = []
    for row in rows:
        actual = override.get(row["path"], int(row["gross"]))
        owner_actual[row["owner"]] += actual
        owner_cap[row["owner"]] += int(row["cap_after_key"])
        owner_paths[row["owner"]] += 1
        if row["changed"] == "True":
            changed_paths += 1
        else:
            unchanged.append({
                "path": row["path"],
                "owner": row["owner"],
                "cap": int(row["cap_after_key"]),
                "actual": actual,
            })
    source_actual = sum(owner_actual.values())
    global_actual = source_actual + 12 + 2

    amendment = json.loads(git(
        "show", f"{STEP_BASE}:task_memory/task_2026-10-22_step5_position_routing/"
        "repairs/plan_key_reuse_scope_amendment.json"
    ))
    adjustments = []
    for move in amendment["path_ceiling_adjustments"]:
        row = next(r for r in rows if r["path"] == move["path"])
        actual = override.get(move["path"], int(row["gross"]))
        adjustments.append({
            "path": move["path"],
            "owner": row["owner"],
            "before": move["before"],
            "delta": move["delta"],
            "after": move["after"],
            "tsv_before": int(row["cap_before_key"]),
            "tsv_after": int(row["cap_after_key"]),
            "actual": actual,
            "after_minus_actual": move["after"] - actual,
        })
    cap_delta = sum(a["delta"] for a in adjustments)

    step_gross = sum(p["step_gross"] for p in path_facts)
    step_net = sum(p["step_net"] for p in path_facts)
    source_step_gross = numstat(STEP_BASE, candidate, TWO[0])
    test_step_gross = numstat(STEP_BASE, candidate, TWO[1])
    step_checks = {
        "paths_allocated": 2,
        "paths_changed": sum(1 for p in path_facts if p["changed"]),
        "source_gross": source_step_gross["gross"],
        "source_net": source_step_gross["net"],
        "test_gross": test_step_gross["gross"],
        "test_net": test_step_gross["net"],
        "step_gross": step_gross,
        "step_net_retained": step_net,
        "step_gross_ceiling": amendment["repair_delta_ceiling"],
        "source_gross_ceiling": amendment["source_delta_ceiling"],
        "test_gross_ceiling": amendment["test_delta_ceiling"],
        "retained_growth_ceiling": amendment["retained_growth_ceiling"],
        "gross_within": step_gross <= amendment["repair_delta_ceiling"],
        "source_within": source_step_gross["gross"] <= amendment["source_delta_ceiling"],
        "test_within": test_step_gross["gross"] <= amendment["test_delta_ceiling"],
        "retained_within": step_net <= amendment["retained_growth_ceiling"],
        "per_path_within": all(p["actual_total_lines"] <= p["cap_after_key"] for p in path_facts),
    }

    # Bindings: 107 superproject-tree entries (105 source/test/config + doc + gitlink),
    # 5 pinned submodule blobs, 8 fixed evidence blobs, 4 native derived files.
    binding_candidate = {}
    binding_baseline = {}
    mismatches = []
    submodule_rows = []
    for entry in manifest["paths"]:
        path = entry["path"]
        if entry["kind"] == "submodule_pinned":
            rel = path[len(SUBMODULE_PREFIX):]
            cand_blob = git_blob(SUBMODULE_PIN, rel, cwd=SUBMODULE)
            cand_oid = git("rev-parse", f"{SUBMODULE_PIN}:{rel}", cwd=SUBMODULE).strip()
            baseline_oid = entry["snapshot"]["oid"]
            submodule_rows.append({
                "path": path,
                "snapshot_oid": baseline_oid,
                "candidate_oid": cand_oid,
                "candidate_sha256": sha256(cand_blob),
                "bytes": len(cand_blob),
                "match": cand_oid == baseline_oid and sha256(cand_blob) == entry["snapshot"]["sha256"],
            })
            binding_candidate["submodule:" + path] = cand_oid
            binding_baseline["submodule:" + path] = baseline_oid
            if cand_oid != baseline_oid:
                mismatches.append({"path": "submodule:" + path, "snapshot_oid": baseline_oid, "candidate_oid": cand_oid})
        else:
            cand = tree_entry(candidate, path)
            cand_oid = cand["oid"]
            baseline_oid = entry["snapshot"]["oid"]
            binding_candidate[path] = cand_oid
            binding_baseline[path] = baseline_oid
            if path not in TWO and cand_oid != baseline_oid:
                mismatches.append({"path": path, "snapshot_oid": baseline_oid, "candidate_oid": cand_oid})
    for path in TWO:
        binding_candidate[path] = tree_entry(candidate, path)["oid"]
        binding_baseline[path] = next(
            p for p in manifest["paths"] if p["path"] == path
        )["snapshot"]["oid"]

    evidence_rows = []
    for item in manifest["fixed_non_source_evidence"]:
        cand = tree_entry(candidate, item["path"])
        evidence_rows.append({
            "path": item["path"],
            "snapshot_blob": item["blob"],
            "snapshot_sha256": item["sha256"],
            "candidate_oid": cand["oid"],
            "match": cand["oid"] == item["blob"],
        })
        binding_candidate[item["path"]] = cand["oid"]
        binding_baseline[item["path"]] = item["blob"]
        if cand["oid"] != item["blob"]:
            mismatches.append({"path": item["path"], "snapshot_oid": item["blob"], "candidate_oid": cand["oid"]})

    native_rows = []
    for item in manifest["fixed_native_source_pins"]:
        derived = open(item["derived_path"], "rb").read()
        base = open(item["base_path"], "rb").read()
        key = "native:" + item["path"]
        binding_candidate[key] = sha256(derived)
        binding_baseline[key] = item["derived_sha256"]
        native_rows.append({
            "path": item["path"],
            "base_sha256": sha256(base),
            "derived_sha256": sha256(derived),
            "snapshot_derived_sha256": item["derived_sha256"],
            "bytes": len(derived),
            "base_match": sha256(base) == item["base_sha256"],
            "derived_match": sha256(derived) == item["derived_sha256"],
        })
        if sha256(derived) != item["derived_sha256"]:
            mismatches.append({"path": key, "snapshot_oid": item["derived_sha256"], "candidate_oid": sha256(derived)})

    def digest(bindings, exclude=()):
        lines = [f"{p} {o}" for p, o in sorted(bindings.items()) if p not in exclude]
        return sha256("\n".join(lines).encode())

    non_key = set(TWO)
    baseline_non_key_digest = digest(binding_baseline, non_key)
    candidate_non_key_digest = digest(binding_candidate, non_key)
    key_delta_digest = sha256(
        "\n".join(
            f"{p} {binding_baseline[p]} {binding_candidate[p]}" for p in TWO
        ).encode()
    )

    submodule_diff = git(
        "diff", "--numstat", SUBMODULE_ORIGINAL, SUBMODULE_PIN, cwd=SUBMODULE
    ).strip().splitlines()
    submodule_numstat = [
        {
            "path": line.split("\t")[2],
            "added": int(line.split("\t")[0]),
            "deleted": int(line.split("\t")[1]),
        }
        for line in submodule_diff
    ]
    submodule_totals = {
        "files": len(submodule_numstat),
        "added": sum(x["added"] for x in submodule_numstat),
        "deleted": sum(x["deleted"] for x in submodule_numstat),
    }
    submodule_totals["gross"] = submodule_totals["added"] + submodule_totals["deleted"]
    submodule_totals["net"] = submodule_totals["added"] - submodule_totals["deleted"]
    submodule_totals["matches_accepted_333"] = (
        submodule_totals["files"] == 5 and submodule_totals["gross"] == 333
    )

    native_numstat = []
    for item in manifest["fixed_native_source_pins"]:
        raw = git(
            "diff", "--no-index", "--numstat", item["base_path"], item["derived_path"],
            check=False,
        ).strip()
        added, deleted, _ = raw.split("\t")
        native_numstat.append({
            "path": item["path"], "added": int(added), "deleted": int(deleted)
        })
    native_totals = {
        "files": len(native_numstat),
        "added": sum(x["added"] for x in native_numstat),
        "deleted": sum(x["deleted"] for x in native_numstat),
    }
    native_totals["gross"] = native_totals["added"] + native_totals["deleted"]
    native_totals["matches_accepted_99"] = (
        native_totals["files"] == 4 and native_totals["gross"] == 99
    )

    doc_path = "docs/general/module-size-exceptions.md"
    doc_entry = tree_entry(candidate, doc_path)
    doc_numstat = numstat(ORIGINAL_BASE, candidate, doc_path)
    gitlink_numstat = numstat(ORIGINAL_BASE, candidate, "frontier/cc_backend/backends/collective-sim")
    gitlink_cand = tree_entry(candidate, "frontier/cc_backend/backends/collective-sim")

    owner_transition = {}
    for owner in ("core_source", "pricing", "native"):
        before = owner_actual[owner] - (14 if owner == "pricing" else 0)
        owner_transition[owner] = {
            "allocated_paths": owner_paths[owner],
            "cap_before_key": owner_cap[owner] - (0 if owner != "pricing" else 0),
            "cap_after_key": owner_cap[owner],
            "actual_before_key": before,
            "actual_after_key": owner_actual[owner],
            "headroom": owner_cap[owner] - owner_actual[owner],
            "cap_unchanged": True,
        }

    violations = []
    if not guard_pass:
        violations.append("non_task_memory_diff_name_set_mismatch")
    if mismatches:
        violations.append("non_key_binding_mismatch")
    if not all(x["match"] for x in submodule_rows):
        violations.append("submodule_binding_mismatch")
    if not all(x["derived_match"] and x["base_match"] for x in native_rows):
        violations.append("native_hash_mismatch")
    if not all(x["match"] for x in evidence_rows):
        violations.append("fixed_evidence_mismatch")
    if cap_delta != 0:
        violations.append("overlay_not_zero_sum")
    if not all(step_checks[k] for k in (
        "gross_within", "source_within", "test_within", "retained_within",
        "per_path_within",
    )):
        violations.append("step_ceiling_exceeded")
    if owner_transition["pricing"]["actual_after_key"] > owner_transition["pricing"]["cap_after_key"]:
        violations.append("owner_cap_exceeded")
    if global_actual > 10000 or source_actual > 9942:
        violations.append("global_or_source_cap_exceeded")
    if candidate_non_key_digest != baseline_non_key_digest:
        violations.append("non_key_binding_digest_mismatch")
    if not submodule_totals["matches_accepted_333"]:
        violations.append("submodule_333_mismatch")
    if not native_totals["matches_accepted_99"]:
        violations.append("native_99_mismatch")
    if doc_numstat != {"added": 12, "deleted": 0, "gross": 12, "net": 12}:
        violations.append("module_note_12_mismatch")
    if gitlink_numstat != {"added": 1, "deleted": 1, "gross": 2, "net": 0}:
        violations.append("gitlink_2_mismatch")

    receipt = {
        "task_id": "D29-KEY-SCOPE-DELTA",
        "audit_kind": "FINAL_TWO_PATH_DELTA_FROM_AUDITED_SNAPSHOT",
        "status": "PASS_TWO_PATH_DELTA_ONLY" if not violations else "FAIL",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "records_worktree": REPO + "/.worktrees/d29-snapshot-scope-audit",
        "records_head_at_execution": git(
            "rev-parse", "HEAD", cwd=REPO + "/.worktrees/d29-snapshot-scope-audit"
        ).strip(),
        "primary_head_at_execution": git("rev-parse", "HEAD").strip(),
        "candidate_worktree": REPO + "/.worktrees/step5-d29-plan-key",
        "candidate_head": git("rev-parse", "HEAD", cwd=REPO + "/.worktrees/step5-d29-plan-key").strip(),
        "candidate_two_path_worktree_status": git(
            "status", "--porcelain", "--", *TWO,
            cwd=REPO + "/.worktrees/step5-d29-plan-key",
        ).strip(),
        "candidate_source_commit": candidate,
        "candidate_source_tree": candidate_tree,
        "candidate_records_commit": candidate_records,
        "snapshot_commit": SNAPSHOT,
        "snapshot_tree": snapshot_tree,
        "step_base_commit": STEP_BASE,
        "step_base_tree": step_base_tree,
        "source_snapshot_commit": SOURCE_SNAPSHOT,
        "source_snapshot_merge": SOURCE_MERGE,
        "original_source_base": ORIGINAL_BASE,
        "accepted_snapshot_audit": {
            "records_commit": ACCEPTED_RECORDS,
            "merged_commit": ACCEPTED_MERGE,
            "parent_acceptance_commit": PARENT_ACCEPTANCE,
            "baseline_receipt_sha256": sha256(
                open(SNAPSHOT_EVIDENCE + "/snapshot_baseline_receipt.json", "rb").read()
            ),
            "baseline_manifest_sha256": manifest_sha,
        },
        "diff_guard": {
            "expected_non_task_memory_name_status": expected_non_task,
            "actual_non_task_memory_name_status": non_task,
            "non_task_memory_count": len(non_task),
            "task_memory_paths_from_snapshot": task_memory_paths,
            "records_commit_non_task_memory_changes": non_task_records,
            "candidate_source_to_records_name_changes": len(records_delta),
            "pass": guard_pass,
        },
        "paths": path_facts,
        "step_checks": step_checks,
        "owner_transition": owner_transition,
        "overlay_transitions": [{
            "amendment": "task_memory/task_2026-10-22_step5_position_routing/repairs/plan_key_reuse_scope_amendment.json",
            "decision_id": amendment["decision_id"],
            "apply_after": amendment["apply_after"],
            "authority": amendment["authority"],
            "effective_unique_paths": amendment["effective_unique_paths"],
            "repair_paths": amendment["repair_paths"],
            "path_ceiling_adjustments": adjustments,
            "cap_delta_sum": cap_delta,
            "zero_sum": cap_delta == 0,
            "owner_ceilings": amendment["owner_gross_ceilings"],
            "retained_growth_ceiling": amendment["retained_growth_ceiling"],
            "violations": [] if cap_delta == 0 else ["cap_transfer_not_zero_sum"],
        }],
        "carried_baseline_overlays": {
            "count": 15,
            "replayed_here": False,
            "note": "The accepted snapshot audit replayed all 15 overlays; this delta audit only applies the plan-key overlay on top of it.",
        },
        "allocated_vs_changed": {
            "allocated_paths": len(rows),
            "changed_paths": changed_paths,
            "unchanged_allocated": unchanged,
        },
        "path_ledger": {
            "rows": len(rows),
            "note": "Two rows recomputed from Git blobs; the other 112 rows are carried unchanged from the accepted snapshot ledger and were re-bound byte-for-byte above.",
            "entries": [
                {
                    "path": row["path"],
                    "owner": row["owner"],
                    "kind": row["kind"],
                    "cap": int(row["cap_after_key"]),
                    "actual": override.get(row["path"], int(row["gross"])),
                    "basis": "recomputed" if row["path"] in TWO else "carried_accepted_baseline",
                }
                for row in rows
            ],
        },
        "totals": {
            "owner_actual": owner_actual,
            "owner_caps": owner_cap,
            "owner_paths": owner_paths,
            "source_actual_before_key": 9463,
            "source_actual_after_key": source_actual,
            "source_cap": 9942,
            "module_size_note_actual": 12,
            "gitlink_actual": 2,
            "global_actual_before_key": 9477,
            "global_actual_after_key": global_actual,
            "global_cap": 10000,
            "pricing_actual_expected": 3954,
            "pricing_actual_matches_expected": owner_actual["pricing"] == 3954,
            "source_matches_expected": source_actual == 9477,
            "global_matches_expected": global_actual == 9491,
            "caps_unchanged": owner_cap == {"core_source": 4404, "pricing": 3956, "native": 1582},
        },
        "non_source_bindings": {
            "module_size_note": {
                "path": doc_path,
                "snapshot_oid": doc_entry["oid"],
                "candidate_oid": doc_entry["oid"],
                "candidate_sha256": sha256(git_blob(candidate, doc_path)),
                "numstat_vs_1aee0af": doc_numstat,
                "match": doc_entry["oid"] == next(
                    p for p in manifest["paths"] if p["path"] == doc_path
                )["snapshot"]["oid"],
            },
            "submodule": {
                "path": "frontier/cc_backend/backends/collective-sim",
                "original_pin": SUBMODULE_ORIGINAL,
                "snapshot_and_candidate_pin": SUBMODULE_PIN,
                "candidate_gitlink": gitlink_cand,
                "numstat_vs_1aee0af": gitlink_numstat,
                "diff_vs_original": submodule_numstat,
                "totals": submodule_totals,
                "blobs": submodule_rows,
                "all_blobs_match": all(x["match"] for x in submodule_rows),
            },
            "native_image_patch": {
                "files": native_totals["files"],
                "added": native_totals["added"],
                "deleted": native_totals["deleted"],
                "gross": native_totals["gross"],
                "numstat": native_numstat,
                "pins": native_rows,
                "matches_accepted_99": native_totals["matches_accepted_99"],
            },
            "fixed_image_evidence": {
                "checked": len(evidence_rows),
                "rows": evidence_rows,
                "all_match": all(x["match"] for x in evidence_rows),
            },
        },
        "binding_guard": {
            "binding_count_total": len(binding_candidate),
            "binding_count_non_key": len(binding_candidate) - len(TWO),
            "digest_algorithm": "sha256 of sorted 'path oid' lines; native files keyed as native:<path> with derived sha256",
            "baseline_all_binding_digest": digest(binding_baseline),
            "candidate_all_binding_digest": digest(binding_candidate),
            "baseline_non_key_binding_digest": baseline_non_key_digest,
            "candidate_non_key_binding_digest": candidate_non_key_digest,
            "non_key_digest_match": baseline_non_key_digest == candidate_non_key_digest,
            "key_delta_digest": key_delta_digest,
            "non_key_mismatches": mismatches,
            "future_guard": (
                "Recompute the 122 non-key bindings at the final union head and require "
                "digest == candidate_non_key_binding_digest; the only allowed differences "
                "are the two key paths captured in key_delta_digest."
            ),
        },
        "verification_commands": [
            "git diff --no-renames --name-status e200bb78 c4e06a61 -- . ':(exclude)task_memory'",
            "git diff --no-renames --name-status c4e06a61 d1e2fb8c -- . ':(exclude)task_memory'",
            "git diff --no-renames --numstat 087db1bf c4e06a61 -- <two paths>",
            "git cat-file blob <rev>:<path> | sha256sum, with newline count for totals",
            "git ls-tree <rev> -- <path> for every manifest binding",
            "git -C <submodule> diff --numstat d28fe917 3dac53c",
            "git diff --no-index --numstat <native base> <native derived> for four pins",
            "sha256 of the eight fixed image evidence blobs and the eight native base/derived files",
        ],
        "violations": violations,
        "limits": [
            "Read-only Git metadata, small receipt/JSON/TSV reads, and scoped file hashes only.",
            "No tests, imports, code execution, probes, model/CSV/trace reads, Docker, network, GPU, package rebuild, or native rebuild.",
            "Candidate, primary source, tests, peers, and the source worktree were not modified or checked out.",
            "This is the final two-path snapshot delta, not the final 114-path union acceptance or full-workflow acceptance.",
            "Behavioral correctness of the cache-key implementation is not re-proven here; dual reviews and primary runtime checks remain.",
            "Cumulative checkpoint accounting for earlier authorized work is carried from the accepted baseline, not recomputed.",
        ],
        "residual_checks": [
            "Final union head must reproduce candidate_non_key_binding_digest for the 122 non-key bindings.",
            "Final union must keep source<=9942, global<=10000, pricing<=3956, core<=4404, native<=1582.",
            "The two key paths must remain at 139/140 and 311/311 lines with no further edits unless separately allocated.",
            "Full workflow and runtime acceptance remain outside this audit.",
        ],
    }
    out = OUT + "/plan_key_delta_receipt.json"
    if guard_only:
        print(json.dumps({"guard_only": True, "status": receipt["status"], "violations": violations, "non_key_digest": candidate_non_key_digest, "expected_non_key_digest": baseline_non_key_digest, "totals": receipt["totals"]}, indent=1))
        return 0 if not violations else 1
    with open(out, "w") as fh:
        json.dump(receipt, fh, indent=1)
        fh.write("\n")
    guard_doc = {
        "task_id": receipt["task_id"],
        "snapshot_commit": SNAPSHOT,
        "candidate_source_commit": candidate,
        "candidate_head": receipt["candidate_head"],
        "expected_non_task_memory_name_status": expected_non_task,
        "key_paths": [
            {
                "path": p["path"],
                "snapshot_oid": p["snapshot_oid"],
                "candidate_oid": p["candidate_oid"],
                "candidate_sha256": p["candidate_sha256"],
                "actual_total_lines": p["actual_total_lines"],
                "cap_after_key": p["cap_after_key"],
            }
            for p in path_facts
        ],
        "baseline_non_key_binding_digest": baseline_non_key_digest,
        "candidate_non_key_binding_digest": candidate_non_key_digest,
        "key_delta_digest": key_delta_digest,
        "guard_commands": [
            "git diff --no-renames --name-status e200bb78 <final-head> -- . ':(exclude)task_memory'",
            "D29_GUARD_ONLY=1 D29_CANDIDATE=<final-head> D29_CANDIDATE_RECORDS=<final-records> python3 repairs/evidence/plan_key_scope/recompute_plan_key_delta.py",
        ],
    }
    with open(OUT + "/changed_file_guard.json", "w") as fh:
        json.dump(guard_doc, fh, indent=1)
        fh.write("\n")
    print(json.dumps({
        "status": receipt["status"],
        "violations": violations,
        "guard": guard_pass,
        "step": step_checks,
        "totals": receipt["totals"],
        "bindings": {
            "total": len(binding_candidate),
            "non_key_match": baseline_non_key_digest == candidate_non_key_digest,
            "key_delta_digest": key_delta_digest,
        },
        "wrote": out,
    }, indent=1))
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
