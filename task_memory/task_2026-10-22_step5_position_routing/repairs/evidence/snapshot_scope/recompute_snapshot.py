"""Recompute the frozen D29 snapshot budget without running product code."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
TASK = Path("task_memory/task_2026-10-22_step5_position_routing")
OUT = ROOT / TASK / "repairs/evidence/snapshot_scope"
SNAPSHOT = "e200bb780d3b20ec6f5ff0a067ed8b9ddd505662"
BASE = "1aee0af11cd58691e64603b99eca0ac459a5b350"
OLD = "79f1a3fd35d42b751cd1520c764ddac717b923d9"
SUB = "frontier/cc_backend/backends/collective-sim"
# Read Git objects only; do not operate on this checkout or its index.
SUB_DB = Path("/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06") / SUB / ".git"
CAPS = {"core_source": 4404, "pricing": 3956, "native": 1582}
OVERLAYS = [
    "registry_scope_amendment.json", "core_review_scope_amendment.json",
    "legacy_output_scope_amendment.json", "prepare_metadata_scope_amendment.json",
    "final_scope_reconciliation.json", "cli_placement_scope_amendment.json",
    "model_group_reuse_scope_amendment.json", "init_validation_scope_amendment.json",
    "collective_artifact_parse_scope_amendment.json", "trace_serialization_scope_amendment.json",
    "prepared_point_scope_amendment.json", "immutable_cost_fields_scope_amendment.json",
    "collective_artifact_bytes_scope_amendment.json", "run_input_snapshot_scope_amendment.json",
    "plan_key_reuse_scope_amendment.json",
]
COMMANDS = []
VIOLATIONS = []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args, db=None):
    command = ["git"] + ([f"--git-dir={db}"] if db else []) + list(args)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True, timeout=20)
    COMMANDS.append({"argv": command, "exit_code": result.returncode,
                     "stdout_sha256": sha(result.stdout), "stdout_bytes": len(result.stdout)})
    return result.stdout


def text(*args, db=None):
    return git(*args, db=db).decode().strip()


def binding(path):
    raw = git("show", f"{SNAPSHOT}:{path}")
    live = (ROOT / path).read_bytes()
    check(raw == live, "input_worktree_drift", str(path))
    return {"path": str(path), "blob": text("rev-parse", f"{SNAPSHOT}:{path}"),
            "sha256": sha(raw), "bytes": len(raw)}


def read_json(path):
    return json.loads(git("show", f"{SNAPSHOT}:{path}"))


def check(condition, category, detail):
    if not condition:
        VIOLATIONS.append({"category": category, "detail": detail})


def numstat(before, after, paths, db=None):
    raw = git("diff", "--numstat", "--no-renames", before, after, "--", *paths, db=db)
    rows = {}
    for line in raw.decode().splitlines():
        added, deleted, path = line.split("\t", 2)
        rows[path] = {"added": int(added), "deleted": int(deleted),
                      "gross": int(added) + int(deleted), "net": int(added) - int(deleted)}
    return rows


def totals(rows):
    return {"paths": len(rows), **{key: sum(row[key] for row in rows.values())
                                 for key in ("added", "deleted", "gross", "net")}}


def allocation(ledger):
    counts, caps = Counter(), Counter()
    for row in ledger.values():
        counts[row["owner"]] += 1
        caps[row["owner"]] += row["cap"]
    return {"paths": len(ledger), "owner_paths": dict(counts), "owner_caps": dict(caps),
            "source_cap": sum(caps.values())}


def blob(commit, path, db=None):
    listing = text("ls-tree", commit, "--", path, db=db)
    if not listing:
        return None
    mode, kind, oid = listing.split("\t")[0].split()
    result = {"mode": mode, "type": kind, "oid": oid}
    if kind == "blob":
        raw = git("cat-file", "blob", oid, db=db)
        result.update(sha256=sha(raw), bytes=len(raw))
    return result


def native_patch(raw, base_root, derived_root, manifest):
    lines = raw.decode().splitlines(keepends=True)
    rows, pins = {}, []
    pos = 0
    while pos < len(lines):
        assert lines[pos].startswith("--- a/native-image/")
        name = lines[pos].strip().removeprefix("--- a/native-image/")
        assert lines[pos + 1].strip() == "+++ b/native-image/" + name
        pos += 2
        base_path, derived_path = base_root / name, derived_root / name
        original = base_path.read_bytes()
        original_lines = original.decode().splitlines(keepends=True)
        produced, cursor, added, deleted = [], 0, 0, 0
        while pos < len(lines) and lines[pos].startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[pos])
            assert match
            old_start, old_count, new_start, new_count = (
                int(value) if value is not None else 1 for value in match.groups())
            target = old_start - 1
            assert target >= cursor
            produced.extend(original_lines[cursor:target])
            cursor = target
            assert len(produced) == new_start - 1
            old_used = new_used = 0
            pos += 1
            while pos < len(lines) and not lines[pos].startswith(("@@", "--- a/")):
                line = lines[pos]
                assert line[0] in " +-"
                if line[0] in " -":
                    assert original_lines[cursor] == line[1:]
                    cursor += 1
                    old_used += 1
                if line[0] in " +":
                    produced.append(line[1:])
                    new_used += 1
                added += line[0] == "+"
                deleted += line[0] == "-"
                pos += 1
            assert (old_used, new_used) == (old_count, new_count)
        produced.extend(original_lines[cursor:])
        rebuilt = "".join(produced).encode()
        derived = derived_path.read_bytes()
        check(sha(original) == manifest["base_source_sha256"][name], "native_base_hash", name)
        check(sha(rebuilt) == manifest["derived_source_sha256"][name], "native_patch_result_hash", name)
        check(rebuilt == derived, "native_installed_hash", name)
        rows["native-image/" + name] = {"added": added, "deleted": deleted,
                                      "gross": added + deleted, "net": added - deleted}
        pins.append({"path": name, "base_path": str(base_path), "derived_path": str(derived_path),
                     "base_sha256": sha(original), "derived_sha256": sha(derived),
                     "rebuilt_sha256": sha(rebuilt), "applied_in_memory_without_fuzz": True})
    return rows, pins


def main():
    records_head = text("rev-parse", "HEAD")
    check(not text("diff", "--name-only", SNAPSHOT, records_head, "--", ".", ":!task_memory/**"),
          "audit_worktree_source_head_drift", records_head)
    check(text("merge-base", BASE, SNAPSHOT) == BASE, "baseline_ancestry", BASE)
    scope_path = TASK / "scope.json"
    scope = read_json(scope_path)
    grouped = {}
    for row in scope["rows"]:
        grouped.setdefault(row["path"], []).append(row)
    ledger = {}
    pricing_shared = {"frontier/execution_time_predictor/moe_operator_times.py",
                      "frontier/profiling/moe/fp8_block_experts.py"}
    for path, rows in grouped.items():
        owners = {row["owner"] for row in rows}
        owner = ("native" if "native" in owners else "pricing" if path in pricing_shared
                 else "core_source" if owners & {"core", "source"} else "pricing")
        ledger[path] = {"owner": owner, "cap": sum(row["ceiling"] for row in rows),
                        "allocation_origin": "scope.json", "original_allowance_rows": rows}
    initial = allocation(ledger)
    check(initial == {"paths": 107, "owner_paths": {"core_source": 44, "native": 19, "pricing": 44},
                      "owner_caps": CAPS, "source_cap": 9942}, "initial_allocation", initial)
    inputs = [binding(scope_path), binding(TASK / "shared_scope.md"),
              binding(TASK / "requirements.md"), binding(TASK / "plan.md")]
    approved_raw = git("show", f"{scope['approved_ledger_commit']}:{scope_path}")
    approved_scope = json.loads(approved_raw)
    check(approved_scope["rows"] == scope["rows"], "original_approval_rows_drift", str(scope_path))
    check(sha(approved_raw) == scope["approved_ledger_sha256"], "original_approval_hash", sha(approved_raw))
    overlays = []
    after_snapshot_caps = None
    for filename in OVERLAYS:
        path = TASK / "repairs" / filename
        amendment = read_json(path)
        inputs.append(binding(path))
        before = allocation(ledger)
        transitions = []
        for row in amendment.get("new_paths", []):
            check(row["path"] not in ledger, "overlay_duplicate_new_path", row)
            ledger[row["path"]] = {"owner": row["owner"], "cap": row["ceiling"],
                                    "allocation_origin": filename, "original_allowance_rows": []}
            transitions.append({"path": row["path"], "owner": row["owner"], "before": 0,
                                "delta": row["ceiling"], "after": row["ceiling"], "new_path": True})
        adjustments = amendment.get("existing_path_ceiling_adjustments", []) + amendment.get("path_ceiling_adjustments", [])
        for row in adjustments:
            current = ledger[row["path"]]
            check(current["cap"] == row["before"], "overlay_before_mismatch", {"overlay": filename, **row, "actual_before": current["cap"]})
            check(row["before"] + row["delta"] == row["after"], "overlay_arithmetic", {"overlay": filename, **row})
            check(row["after"] >= 0, "negative_cap", row)
            transitions.append({**row, "owner": current["owner"], "new_path": False})
            current["cap"] = row["after"]
            current["allocation_origin"] = filename
        after = allocation(ledger)
        check(after["owner_caps"] == CAPS, "owner_cap_change", {"overlay": filename, "after": after})
        check(after["paths"] == amendment["effective_unique_paths"], "overlay_path_count", filename)
        if amendment.get("apply_after"):
            check(amendment["apply_after"] == OVERLAYS[len(overlays) - 1], "overlay_order", filename)
        transferred = {owner: sum(max(0, row["delta"]) for row in transitions if row["owner"] == owner)
                       for owner in CAPS}
        overlay = {"file": str(path), "decision_id": amendment["decision_id"], "before": before,
                   "after": after, "transitions": transitions, "transferred_lines_by_owner": transferred,
                   "zero_sum_by_owner": {owner: sum(row["delta"] for row in transitions if row["owner"] == owner) for owner in CAPS},
                   "step_caps_not_measurements": {key: amendment[key] for key in (
                       "repair_delta_ceiling", "source_delta_ceiling", "test_delta_ceiling",
                       "retained_growth_ceiling", "path_step_ceilings", "per_path_step_caps") if key in amendment}}
        overlays.append(overlay)
        if filename == "run_input_snapshot_scope_amendment.json":
            after_snapshot_caps = {path: row["cap"] for path, row in ledger.items()}
    nonrecords = numstat(BASE, SNAPSHOT, [".", ":!task_memory/**"])
    extras = {path: nonrecords.pop(path) for path in (SUB, "docs/general/module-size-exceptions.md")}
    baseline_pin = text("rev-parse", f"{BASE}:{SUB}")
    pin = text("rev-parse", f"{SNAPSHOT}:{SUB}")
    check(baseline_pin.startswith("d28fe917"), "submodule_baseline_pin", baseline_pin)
    check(pin == "3dac53c544e546574cb6e8078cd14053d0916113", "submodule_current_pin", pin)
    sub_rows = numstat(baseline_pin, pin, ["."], db=SUB_DB)
    sub_patch = git("diff", "--no-renames", baseline_pin, pin, "--", ".", db=SUB_DB)
    review_path = Path("task_memory/task_2026-10-22_step5_d29_native_balanced/native_source.patch")
    review_raw = git("show", f"{SNAPSHOT}:{review_path}")
    build_path = Path("/data/ycfeng/tmp/d29-native-build-context02/native.patch")
    build_raw = build_path.read_bytes()
    normalized = review_raw.replace(b"a/native-image/vllm/", b"a/vllm/").replace(b"b/native-image/vllm/", b"b/vllm/")
    check(normalized == build_raw, "native_patch_prefix_normalization", str(build_path))
    derivation_path = TASK / "image_publication/derivation_manifest.json"
    manifest = read_json(derivation_path)
    check(sha(build_raw) == manifest["patch_sha256"], "build_patch_hash", sha(build_raw))
    native_rows, native_pins = native_patch(review_raw,
        Path("/data/ycfeng/tmp/step5_profiling_20261020/image_src"),
        Path("/data/ycfeng/tmp/d29_published_sources01/source"), manifest)
    actual = {**nonrecords, **{SUB + "/" + path: row for path, row in sub_rows.items()}, **native_rows}
    check(set(actual) <= set(ledger), "unallocated_changed_paths", sorted(set(actual) - set(ledger)))
    rows, owner_actual, owner_changed, manifests = [], Counter(), Counter(), []
    for path, entry in sorted(ledger.items()):
        count = actual.get(path, {"added": 0, "deleted": 0, "gross": 0, "net": 0})
        kind = "submodule_pinned" if path.startswith(SUB + "/") else "native_image_patch" if path.startswith("native-image/") else "superproject"
        record = {"path": path, **entry, "cap_before_key_reservation": after_snapshot_caps[path],
                  "actual": count, "kind": kind, "changed": count["gross"] != 0,
                  "headroom": entry["cap"] - count["gross"]}
        check(record["headroom"] >= 0, "path_cap_exceeded", record)
        owner_actual[entry["owner"]] += count["gross"]
        owner_changed[entry["owner"]] += count["gross"] != 0
        if kind != "native_image_patch":
            subpath = path.removeprefix(SUB + "/") if kind == "submodule_pinned" else path
            db = SUB_DB if kind == "submodule_pinned" else None
            previous = blob(baseline_pin if db else BASE, subpath, db=db)
            current = blob(pin if db else SNAPSHOT, subpath, db=db)
            record["baseline_blob"], record["snapshot_blob"] = previous, current
            if kind == "superproject" and current:
                check(sha((ROOT / path).read_bytes()) == current["sha256"], "source_worktree_drift", path)
            manifests.append({"path": path, "kind": kind, "changed": record["changed"],
                              "baseline": previous, "snapshot": current})
        else:
            record["native_hash_binding"] = next(pin for pin in native_pins if "native-image/" + pin["path"] == path)
        rows.append(record)
    for path, count in extras.items():
        manifests.append({"path": path, "kind": "gitlink" if path == SUB else "module_size_note",
                          "changed": True, "baseline": blob(BASE, path), "snapshot": blob(SNAPSHOT, path), "actual": count})
    source_total = sum(owner_actual.values())
    global_total = source_total + sum(row["gross"] for row in extras.values())
    check(dict(owner_actual) == {"core_source": 4173, "pricing": 3940, "native": 1350}, "expected_owner_actual", dict(owner_actual))
    check(source_total == 9463 and global_total == 9477, "expected_global_actual", [source_total, global_total])
    steps = []
    for label, commit, base_commit, merge, cap, source_cap, test_cap in [
        ("model_group_reuse", "3763e5b8", "3763e5b8^", None, 105, 45, 60),
        ("loader_family", "ee7e66ee", "ee7e66ee^", None, 45, 6, 39),
        ("artifact_parse", "51e1f720", "51e1f720^", None, 50, 15, 35),
        ("trace_serialization", "390bb5a2", "390bb5a2^", "64881373", 20, 20, 0),
        ("prepared_point", "0be11762", "0be11762^", "eece87b2", 250, 180, 70),
        ("immutable_cost_fields", "75404143", "75404143^", "1b6dfd6a", 8, 8, 0),
        ("artifact_bytes", "2f2d9339", "2f2d9339^", None, 36, 26, 10),
        ("run_input_snapshot", "db581ff6", "2f2d9339", "64c83ac4", 340, 120, 220),
    ]:
        change = numstat(base_commit, commit, ["frontier", "tests"])
        source = {path: row for path, row in change.items() if path.startswith("frontier/")}
        tests = {path: row for path, row in change.items() if path.startswith("tests/")}
        expected_step_gross = {"model_group_reuse": 71, "loader_family": 38, "artifact_parse": 43,
                               "trace_serialization": 14, "prepared_point": 159,
                               "immutable_cost_fields": 8, "artifact_bytes": 32,
                               "run_input_snapshot": 212}
        check(totals(change)["gross"] == expected_step_gross[label], "expected_step_measurement", label)
        check(totals(change)["gross"] <= cap, "step_cap", label)
        check(totals(source)["gross"] <= source_cap and totals(tests)["gross"] <= test_cap, "step_split_cap", label)
        inherited = []
        for path in change:
            current_oid = text("rev-parse", f"{SNAPSHOT}:{path}")
            step_oid = text("rev-parse", f"{commit}:{path}")
            inherited.append({"path": path, "step_blob": step_oid, "snapshot_blob": current_oid,
                              "unchanged_since_step": current_oid == step_oid})
        steps.append({"label": label, "source_commit": text("rev-parse", commit),
                      "step_base": text("rev-parse", base_commit), "merge_commit": text("rev-parse", merge) if merge else None,
                      "cap": cap, "cap_source": source_cap, "cap_test": test_cap,
                      "actual": totals(change), "actual_source": totals(source), "actual_test": totals(tests),
                      "per_path": change, "blob_inheritance": inherited})
    snapshot_step = next(step for step in steps if step["label"] == "run_input_snapshot")
    snapshot_amendment = read_json(TASK / "repairs/run_input_snapshot_scope_amendment.json")
    check(set(snapshot_step["per_path"]) == set(snapshot_amendment["repair_paths"]),
          "snapshot_changed_path_set", sorted(snapshot_step["per_path"]))
    for path, count in snapshot_step["per_path"].items():
        check(count["gross"] <= snapshot_amendment["path_step_ceilings"][path], "snapshot_path_step_cap", path)
    check((snapshot_step["actual_source"]["gross"], snapshot_step["actual_test"]["gross"],
           snapshot_step["actual"]["net"]) == (73, 139, 86), "snapshot_step_split", snapshot_step["actual"])
    snapshots = {}
    for commit in [OLD, "64881373", "eece87b2", "1b6dfd6a", "2f2d9339", "db581ff6", SNAPSHOT]:
        counts = numstat(BASE, commit, ["frontier", "tests"])
        counts.pop(SUB)
        owner_totals = Counter()
        for path, count in counts.items():
            check(path in ledger, "checkpoint_unallocated", path)
            owner_totals[ledger[path]["owner"]] += count["gross"]
        owner_totals["pricing"] += totals(sub_rows)["gross"]
        owner_totals["native"] += totals(native_rows)["gross"]
        snapshots[text("rev-parse", commit)] = {"owner_actual": dict(owner_totals),
                                               "source_actual": sum(owner_totals.values()),
                                               "global_actual": sum(owner_totals.values()) + 14}
    historical = []
    for label, before, checkpoint, paths, cap in [
        ("registry", "d96b3279", "a5189e19", ["frontier/execution_time_predictor/prediction_model_registry.py",
            "tests/unit/test_prediction_measurement_bindings.py", "frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py",
            "frontier/execution_time_predictor/moe_routing_workload.py"], 360),
        ("affected_fixtures", "e49b26f8", "f03a1157", ["tests/unit/test_moe_routing_runtime.py",
            "tests/unit/test_moe_grouped_gemm_padded_features.py"], 80),
    ]:
        measured = totals(numstat(before, checkpoint, paths))
        cumulative = totals(numstat(before, SNAPSHOT, paths))
        check(measured["gross"] <= cap, "historical_checkpoint_cap", label)
        historical.append({"label": label, "base": text("rev-parse", before), "checkpoint": text("rev-parse", checkpoint),
                           "scope": "checkpoint_only", "cap": cap, "checkpoint_actual": measured,
                           "current_cumulative_informational_not_cap_measurement": cumulative})
    old_receipt = TASK / "repairs/evidence/performance_scope/d29_final_budget_audit_receipt06.json"
    old_report = TASK / "repairs/evidence/performance_scope/report_2026-10-23_d29_final_budget_audit06.md"
    budget_path = TASK / "repairs/evidence/plan_key_reuse/budget_before.json"
    prior = read_json(old_receipt)
    budget = read_json(budget_path)
    old_count_map = {row["path"]: row["measured"] for row in prior["rows"]}
    old_actual = numstat(BASE, OLD, ["frontier", "tests"])
    old_actual.pop(SUB)
    old_actual.update({SUB + "/" + path: row for path, row in sub_rows.items()})
    old_actual.update(native_rows)
    prior_mismatches = [path for path in old_count_map if old_count_map[path] != old_actual.get(path, {"gross": 0})["gross"]]
    check(not prior_mismatches, "prior_receipt_recomputation", prior_mismatches)
    check(budget["actual"] == dict(owner_actual) and budget["source_actual"] == source_total
          and budget["global_component"] == global_total, "allocation_reconciliation_comparison", str(budget_path))
    for path, selection in budget["selected"].items():
        check(actual[path]["gross"] == selection["actual"]
              and ledger[path]["cap"] == selection["new_cap"], "key_allocation_selected_path", path)
    image_paths = [derivation_path, TASK / "image_publication/native_installed_sources.json",
                   TASK / "image_publication/publication_receipt.json", TASK / "image_publication/installed_export_receipt.json",
                   TASK / "image_publication/registry_index.json", TASK / "image_publication/registry_manifests.json",
                   TASK / "image_publication/patch_identity.md", review_path]
    image_bindings = [binding(path) for path in image_paths]
    for entry in image_bindings:
        previous = blob(OLD, entry["path"])
        check(previous and previous["oid"] == entry["blob"], "image_evidence_drift_since_accepted_audit", entry["path"])
    installed = read_json(TASK / "image_publication/native_installed_sources.json")
    publication = read_json(TASK / "image_publication/publication_receipt.json")
    index_raw = git("show", f"{SNAPSHOT}:{TASK}/image_publication/registry_index.json")
    check(sha(index_raw) == publication["registry_index_sha256"], "registry_index_hash", sha(index_raw))
    check(installed["image"] == publication["final_image"], "installed_image_binding", installed["image"])
    installed_raw = git("show", f"{SNAPSHOT}:{TASK}/image_publication/native_installed_sources.json")
    check(sha(installed_raw) == publication["source_receipt_sha256"], "installed_source_receipt_hash", sha(installed_raw))
    check(sha(git("show", f"{SNAPSHOT}:{derivation_path}")) == publication["derivation_manifest_sha256"], "derivation_manifest_hash", str(derivation_path))
    for pin_row in native_pins:
        check(installed["sources"][pin_row["path"]] == pin_row["derived_sha256"], "installed_inventory_pin", pin_row["path"])
    check(sha(review_raw) == prior["facts"]["native_patch_sha256"], "previous_native_patch_hash", sha(review_raw))
    check(sha(sub_patch) == prior["facts"]["submodule_patch_sha256"], "previous_submodule_patch_hash", sha(sub_patch))
    differences = {commit: text("diff", "--name-only", "--no-renames", commit, SNAPSHOT,
                              "--", ".", ":!task_memory/**") for commit in ["64c83ac4", "db581ff6", "087db1bf"]}
    check(not any(differences.values()), "source_snapshot_drift", differences)
    input_paths = [old_receipt, old_report, budget_path]
    inputs.extend(binding(path) for path in input_paths)
    receipt = {
        "task_id": "D29-SNAPSHOT-SCOPE-AUDIT", "session_id": "ses_e95ac82fdffeSeMEUfR9SxvW03",
        "audit_kind": "SNAPSHOT_BASELINE_AUDIT", "status": "PASS" if not VIOLATIONS else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(), "records_worktree": str(ROOT),
        "records_head_at_execution": records_head,
        "source_commit": SNAPSHOT, "source_tree": text("rev-parse", f"{SNAPSHOT}^{{tree}}"),
        "baseline_commit": BASE, "baseline_tree": text("rev-parse", f"{BASE}^{{tree}}"),
        "source_subtrees": {path: text("rev-parse", f"{SNAPSHOT}:{path}") for path in ["frontier", "tests"]},
        "source_snapshot_equivalence": differences, "initial_allocation": initial,
        "original_approval_binding": {"commit": scope["approved_ledger_commit"],
            "actual_scope_sha256_at_commit": sha(approved_raw), "current_rows_identical": approved_scope["rows"] == scope["rows"],
            "declared_original_ledger_sha256": scope["approved_ledger_sha256"],
            "note": "The approved rows are preserved; the current scope file also contains later approval metadata."},
        "effective_allocation_after_key_reservation": allocation(ledger), "overlays": overlays,
        "owner_actual": {owner: {"allocated_paths": allocation(ledger)["owner_paths"][owner],
            "changed_paths": owner_changed[owner], "gross": owner_actual[owner], "cap": CAPS[owner],
            "headroom": CAPS[owner] - owner_actual[owner]} for owner in CAPS},
        "source_actual": source_total, "source_cap": 9942, "source_headroom": 9942 - source_total,
        "global_actual": global_total, "global_cap": 10000, "global_headroom": 10000 - global_total,
        "allocated_paths": len(ledger), "changed_paths": len(actual),
        "unchanged_allocated_paths": sorted(set(ledger) - set(actual)),
        "components": {"superproject_source_test_config": totals(nonrecords), "submodule": totals(sub_rows),
                       "native_image": totals(native_rows), "extras": extras},
        "submodule": {"baseline_commit": baseline_pin, "source_commit": pin,
                      "baseline_tree": text("rev-parse", f"{baseline_pin}^{{tree}}", db=SUB_DB),
                      "source_tree": text("rev-parse", f"{pin}^{{tree}}", db=SUB_DB),
                      "object_database": str(SUB_DB), "patch_sha256": sha(sub_patch), "per_path": sub_rows,
                      "checkout_or_index_access": False},
        "native_image": {"review_patch_sha256": sha(review_raw), "build_patch_sha256": sha(build_raw),
                         "build_patch_path": str(build_path), "normalized_patch_identical": normalized == build_raw,
                         "pins": native_pins, "per_path": native_rows, "evidence_bindings": image_bindings,
                         "published_image": publication["final_image"], "runtime_qualified": False},
        "source_step_receipts": steps, "cumulative_checkpoint_actuals": snapshots,
        "historical_checkpoint_caps": historical,
        "historical_core_review": {"cap": 230, "recorded_actual": 228,
            "status": "CARRIED_NOT_INDEPENDENTLY_RECOMPUTABLE", "reason": prior["step_receipts_not_recomputable"]["core_review"]["why"]},
        "prior_receipt_comparison": {"paths_recomputed": len(old_count_map), "mismatches": prior_mismatches},
        "budget_before_comparison": {"path": str(budget_path), "totals_match": budget["actual"] == dict(owner_actual)},
        "observations": [
            "The module-size note explicitly describes frozen 1be91bad, not this snapshot. Its 98/5/4 changed-Python inventory is historical; current counts are 101/5/4. Note bytes and its 12-line budget remain unchanged.",
            "Initial step receipt iteration used the documentation-only branch tips 9ae60ebc and b476907e and showed zero deltas. Source-path Git history identified actual implementation commits 390bb5a2 and 0be11762; this receipt uses their parent diffs.",
            "Cgroup wrappers were unavailable: systemd-run --user returned 'Job failed'; system scope returned 'Interactive authentication required'. The successful command uses timeout 60s and RLIMIT_AS=30 GiB instead."] ,
        "execution_wrapper": {"timeout_seconds": 60, "memory_limit_kind": "RLIMIT_AS", "memory_limit_bytes": 30 * 1024**3,
            "command": "timeout 60s python3 -B -c 'import resource, runpy; resource.setrlimit(resource.RLIMIT_AS, (30 * 1024**3, 30 * 1024**3)); runpy.run_path(\"task_memory/task_2026-10-22_step5_position_routing/repairs/evidence/snapshot_scope/recompute_snapshot.py\", run_name=\"__main__\")'"},
        "key_reservation_not_implementation": {"allocation_commit": "087db1bfcb3dbc2377d4032f650b29c48fb104fd",
            "step_cap": 17, "source_cap": 6, "test_cap": 11, "retained_growth_cap": 15,
            "included_actual_key_code": False, "implementation_worktree_inspected": False,
            "paths": ["frontier/cc_backend/collective_plan.py", "tests/unit/test_step5_moe_pricing.py"]},
        "rows": rows, "input_bindings": inputs, "violations": VIOLATIONS,
        "environment": {"python": sys.version, "python_executable": sys.executable, "git": text("--version")},
        "verification_commands": COMMANDS,
        "limits": ["No source edits or reallocation; writes only under this evidence directory.",
            "No tests/fullscans, Docker/network/GPU operations, cleanup, native package rebuild or image rebuild.",
            "No peer/superproject checkout or index mutation. Pinned submodule Git objects were read without checkout/index inspection.",
            "Native patch applied to exactly four existing pinned base files in memory; four exported derived files hashed directly.",
            "Registry/publication receipts and native source inventories remain historical evidence, not fresh registry/runtime qualification.",
            "No key implementation source or results inspected; this is not final key/full-workflow acceptance.",
            "Only scoped Git diffs/blob reads and named file hashes; no product module imported."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snapshot_baseline_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (OUT / "snapshot_blob_manifest.json").write_text(json.dumps({"source_commit": SNAPSHOT,
        "source_tree": receipt["source_tree"], "paths": manifests,
        "fixed_native_source_pins": native_pins, "fixed_non_source_evidence": image_bindings,
        "future_guard_command": ["git", "diff", "--name-only", "--no-renames", SNAPSHOT,
            "<final-key-commit>", "--", ".", ":!task_memory/**"],
        "expected_future_changed_paths": receipt["key_reservation_not_implementation"]["paths"],
        "residual": "Recount only those two paths after exact Git name-set validation; all other blob/gitlink/evidence bindings must remain fixed."}, indent=2) + "\n")
    header = "path\towner\tkind\tcap_before_key\tcap_after_key\tadded\tdeleted\tgross\tnet\theadroom\tchanged\n"
    (OUT / "snapshot_paths.tsv").write_text(header + "".join(
        f"{row['path']}\t{row['owner']}\t{row['kind']}\t{row['cap_before_key_reservation']}\t{row['cap']}\t"
        f"{row['actual']['added']}\t{row['actual']['deleted']}\t{row['actual']['gross']}\t{row['actual']['net']}\t"
        f"{row['headroom']}\t{row['changed']}\n" for row in rows))
    print(json.dumps({"status": receipt["status"], "allocated_paths": len(ledger), "changed_paths": len(actual),
                      "owner_actual": dict(owner_actual), "source_actual": source_total, "global_actual": global_total,
                      "steps": {step["label"]: {"gross": step["actual"]["gross"], "source": step["actual_source"]["gross"],
                          "test": step["actual_test"]["gross"], "net": step["actual"]["net"]} for step in steps},
                      "violations": VIOLATIONS}, indent=2))


if __name__ == "__main__":
    main()
