"""Recompute every approved D29 source path and binding for the ledger candidate."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import time

ROOT = Path("/data/ycfeng/Frontier/.worktrees/d29-ledger-final-1f6b4314")
PRIMARY = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
TASK = Path("task_memory/task_2026-10-22_step5_position_routing")
SOURCE = "1f6b431478e2351e3ca6819f707cccbc3e4336e2"
RECORDS = "97ae8d9b64f4728ba3d3c9af0647832e83ee0335"
BASE = "1aee0af11cd58691e64603b99eca0ac459a5b350"
ACCEPTED = "b02772872d0c99ceb5ebb31a8e041926f12f7676"
SUB = "frontier/cc_backend/backends/collective-sim"
PIN = "3dac53c544e546574cb6e8078cd14053d0916113"
OLD_PIN = "d28fe917370c2d55230c3ea226e5660785b44aca"
CAPS = {"core_source": 4404, "pricing": 3956, "native": 1582}
PATHS = ["frontier/metrics/metrics_store.py", "tests/integration/run_step5_measured_main.py", "tests/integration/test_step5_measured_main.py", "tests/performance/step5_actual06_iteration.py"]
OUT = Path(os.environ["TMPDIR"])
COMMANDS = []
INPUTS = []

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def git(*args, repo=ROOT):
    argv = ["git", "--no-optional-locks", "-C", str(repo), *map(str, args)]
    result = subprocess.run(argv, capture_output=True, timeout=60)
    COMMANDS.append({"argv": argv, "exit_code": result.returncode, "stdout_sha256": sha(result.stdout), "stdout_bytes": len(result.stdout)})
    assert result.returncode == 0, (argv, result.returncode, result.stderr.decode())
    return result.stdout

def text(*args, repo=ROOT):
    return git(*args, repo=repo).decode().strip()

def state(repo, clean=False):
    index = Path(text("rev-parse", "--git-path", "index", repo=repo))
    if not index.is_absolute():
        index = repo / index
    status = text("status", "--porcelain=v1", "--untracked-files=all", repo=repo)
    if clean:
        assert not status, (repo, status)
    return {"head": text("rev-parse", "HEAD", repo=repo), "status": status, "index_path": str(index), "index_sha256": sha(index.read_bytes())}

def bound(path, revision=RECORDS):
    raw = git("show", f"{revision}:{path}")
    assert raw == (ROOT / path).read_bytes(), path
    item = {"path": str(path), "commit": revision, "blob": text("rev-parse", f"{revision}:{path}"), "sha256": sha(raw), "bytes": len(raw)}
    INPUTS.append(item)
    return raw

def counts(before, after, repo=ROOT):
    raw = git("diff", "--no-renames", "--numstat", before, after, "--", ".", ":(exclude)task_memory", repo=repo)
    result = {}
    for line in raw.decode().splitlines():
        added, deleted, path = line.split("\t", 2)
        assert added != "-" and deleted != "-", path
        result[path] = {"added": int(added), "deleted": int(deleted), "gross": int(added) + int(deleted), "net": int(added) - int(deleted)}
    return result

def blob(revision, path, repo=ROOT, live=None):
    raw = git("ls-tree", revision, "--", path, repo=repo).decode().strip()
    if not raw:
        return None
    mode, kind, oid = raw.split("\t")[0].split()
    result = {"mode": mode, "type": kind, "oid": oid}
    if kind == "blob":
        content = git("cat-file", "blob", oid, repo=repo)
        result.update(sha256=sha(content), bytes=len(content))
        if live is not None:
            assert live.read_bytes() == content, live
    return result

started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
assert sys.dont_write_bytecode and os.getuid() == 10250
before = {"candidate": state(ROOT, True), "primary": state(PRIMARY), "submodule": state(ROOT / SUB, True)}
assert before["candidate"]["head"] == RECORDS
assert before["submodule"]["head"] == PIN
assert not text("diff", "--name-only", SOURCE, RECORDS, "--", ".", ":(exclude)task_memory")
assert text("merge-base", BASE, SOURCE) == BASE
changed = text("diff", "--no-renames", "--name-status", ACCEPTED, SOURCE, "--", ".", ":(exclude)task_memory").splitlines()
assert sorted(changed) == sorted("M\t" + p for p in PATHS), changed

primitives_path = TASK / "repairs/evidence/snapshot_scope/recompute_snapshot.py"
bound(primitives_path)
primitives = runpy.run_path(str(ROOT / primitives_path), run_name="ledger_composed_metadata_primitives")
sequence = primitives["OVERLAYS"] + ["named_model_query_scope_amendment.json", "native_collective_key_scope_amendment.json", "ep_lane_ledger_scope_amendment.json"]
scope = json.loads(bound(TASK / "scope.json"))
approved_raw = git("show", f"{scope['approved_ledger_commit']}:{TASK}/scope.json")
assert sha(approved_raw) == scope["approved_ledger_sha256"]
assert json.loads(approved_raw)["rows"] == scope["rows"]
ledger = {}
shared_pricing = {"frontier/execution_time_predictor/moe_operator_times.py", "frontier/profiling/moe/fp8_block_experts.py"}
for original in scope["rows"]:
    ledger.setdefault(original["path"], {"rows": []})["rows"].append(original)
for path, entry in ledger.items():
    owners = {r["owner"] for r in entry["rows"]}
    entry.update(owner="native" if "native" in owners else "pricing" if path in shared_pricing else "core_source" if owners & {"core", "source"} else "pricing", cap=sum(r["ceiling"] for r in entry["rows"]))
assert primitives["allocation"](ledger) == {"paths": 107, "owner_paths": {"core_source": 44, "native": 19, "pricing": 44}, "owner_caps": CAPS, "source_cap": 9942}
overlays = []
for filename in sequence:
    amendment = json.loads(bound(TASK / "repairs" / filename))
    before_allocation = primitives["allocation"](ledger)
    transitions = []
    for row in amendment.get("new_paths", []):
        assert row["path"] not in ledger, row
        ledger[row["path"]] = {"owner": row["owner"], "cap": row["ceiling"], "rows": []}
        transitions.append({"path": row["path"], "owner": row["owner"], "before": 0, "delta": row["ceiling"], "after": row["ceiling"]})
    for row in amendment.get("existing_path_ceiling_adjustments", []) + amendment.get("path_ceiling_adjustments", []):
        entry = ledger[row["path"]]
        assert entry["cap"] == row["before"], (filename, row, entry["cap"])
        assert row["before"] + row["delta"] == row["after"] >= 0
        entry["cap"] = row["after"]
        transitions.append({**row, "owner": entry["owner"]})
    after_allocation = primitives["allocation"](ledger)
    assert after_allocation["owner_caps"] == CAPS
    assert after_allocation["paths"] == amendment["effective_unique_paths"]
    if amendment.get("apply_after"):
        assert amendment["apply_after"] == sequence[len(overlays) - 1], filename
    zero_sum = {owner: sum(r["delta"] for r in transitions if r["owner"] == owner) for owner in CAPS}
    assert not any(zero_sum.values()), (filename, zero_sum)
    overlays.append({"file": filename, "decision_id": amendment["decision_id"], "before": before_allocation, "after": after_allocation, "zero_sum_by_owner": zero_sum, "transitions": transitions})
assert len(ledger) == 116

super_counts = counts(BASE, SOURCE)
extras = {path: super_counts.pop(path) for path in [SUB, "docs/general/module-size-exceptions.md"]}
assert extras[SUB]["gross"] == 2 and extras["docs/general/module-size-exceptions.md"]["gross"] == 12
assert text("rev-parse", f"{BASE}:{SUB}") == OLD_PIN
assert text("rev-parse", f"{SOURCE}:{SUB}") == PIN
sub_counts = counts(OLD_PIN, PIN, ROOT / SUB)
review_path = Path("task_memory/task_2026-10-22_step5_d29_native_balanced/native_source.patch")
review = bound(review_path)
manifest = json.loads(bound(TASK / "image_publication/derivation_manifest.json"))
build_path = Path("/data/ycfeng/tmp/d29-native-build-context02/native.patch")
build = build_path.read_bytes()
assert review.replace(b"a/native-image/vllm/", b"a/vllm/").replace(b"b/native-image/vllm/", b"b/vllm/") == build
assert sha(build) == manifest["patch_sha256"]
native_counts, native_pins = primitives["native_patch"](review, Path("/data/ycfeng/tmp/step5_profiling_20261020/image_src"), Path("/data/ycfeng/tmp/d29_published_sources01/source"), manifest)
assert not primitives["VIOLATIONS"], primitives["VIOLATIONS"]
actual = {**super_counts, **{SUB + "/" + path: row for path, row in sub_counts.items()}, **native_counts}
assert set(actual) <= set(ledger), sorted(set(actual) - set(ledger))

rows, bindings, owner_actual = [], [], Counter()
for path, entry in sorted(ledger.items()):
    count = actual.get(path, {"added": 0, "deleted": 0, "gross": 0, "net": 0})
    assert count["gross"] <= entry["cap"], (path, count, entry["cap"])
    kind = "submodule_pinned" if path.startswith(SUB + "/") else "native_image_patch" if path.startswith("native-image/") else "superproject"
    row = {"path": path, "owner": entry["owner"], "kind": kind, "cap": entry["cap"], "actual": count["gross"], "fresh_numstat": count, "headroom": entry["cap"] - count["gross"]}
    owner_actual[entry["owner"]] += count["gross"]
    if kind == "native_image_patch":
        pin_row = next(r for r in native_pins if "native-image/" + r["path"] == path)
        binding = {"binding": "native:" + path, **pin_row}
    else:
        is_sub = kind == "submodule_pinned"
        relative = path.removeprefix(SUB + "/") if is_sub else path
        repo = ROOT / SUB if is_sub else ROOT
        binding = {"binding": ("submodule:" if is_sub else "") + path, "baseline": blob(OLD_PIN if is_sub else BASE, relative, repo), "candidate": blob(PIN if is_sub else SOURCE, relative, repo, ROOT / path)}
    rows.append(row)
    bindings.append(binding)
for path in extras:
    bindings.append({"binding": path, "baseline": blob(BASE, path), "candidate": blob(SOURCE, path, live=ROOT / path)})
image_paths = [TASK / "image_publication" / name for name in ["derivation_manifest.json", "native_installed_sources.json", "publication_receipt.json", "installed_export_receipt.json", "registry_index.json", "registry_manifests.json", "patch_identity.md"]]
for path in image_paths:
    bound(path)
    assert git("show", f"{ACCEPTED}:{path}") == git("show", f"{SOURCE}:{path}"), path
    bindings.append({"binding": str(path), "candidate": blob(SOURCE, path, live=ROOT / path)})
bindings.append({"binding": str(review_path), "candidate": blob(SOURCE, review_path, live=ROOT / review_path)})
installed = json.loads((ROOT / TASK / "image_publication/native_installed_sources.json").read_bytes())
publication = json.loads((ROOT / TASK / "image_publication/publication_receipt.json").read_bytes())
assert installed["image"] == publication["final_image"]
for name, key in [("registry_index.json", "registry_index_sha256"), ("native_installed_sources.json", "source_receipt_sha256"), ("derivation_manifest.json", "derivation_manifest_sha256")]:
    assert sha((ROOT / TASK / "image_publication" / name).read_bytes()) == publication[key], name
for pin_row in native_pins:
    assert installed["sources"][pin_row["path"]] == pin_row["derived_sha256"]
assert len(bindings) == 126
assert dict(owner_actual) == {"core_source": 4377, "pricing": 3955, "native": 1350}
assert all(owner_actual[o] <= CAPS[o] for o in CAPS)
source_actual = sum(owner_actual.values())
global_actual = source_actual + sum(r["gross"] for r in extras.values())
assert source_actual == 9682 <= 9942 and global_actual == 9696 <= 10000

prior_path = TASK / "native_cache_key/evidence/frozen_scope/frozen_scope_receipt.json"
prior = json.loads(bound(prior_path))
previous_counts = counts(BASE, ACCEPTED)
previous_counts.pop(SUB)
previous_counts.pop("docs/general/module-size-exceptions.md")
previous_counts.update({SUB + "/" + p: r for p, r in sub_counts.items()})
previous_counts.update(native_counts)
assert len(prior["path_ledger"]) == 114
for old in prior["path_ledger"]:
    assert old["actual"] == previous_counts.get(old["path"], {"gross": 0})["gross"], old["path"]
    if old["path"] not in PATHS:
        assert next(r for r in rows if r["path"] == old["path"])["actual"] == old["actual"], old["path"]
step_counts = counts(ACCEPTED, SOURCE)
assert set(step_counts) == set(PATHS)
step_cap = json.loads((ROOT / TASK / "repairs/ep_lane_ledger_scope_amendment.json").read_bytes())["path_step_ceilings"]
assert all(step_counts[p]["gross"] <= step_cap[p] for p in PATHS)
step = {"gross": sum(r["gross"] for r in step_counts.values()), "net": sum(r["net"] for r in step_counts.values()), "cap": 135, "paths": step_counts, "cumulative_increment": source_actual - prior["totals"]["source_actual"]}
assert step["gross"] == 124 and step["net"] == 96 and step["cumulative_increment"] == 116
for row in overlays[-1]["transitions"]:
    if row["delta"] < 0:
        assert not text("diff", "--name-only", ACCEPTED, SOURCE, "--", row["path"]), row["path"]
after = {"candidate": state(ROOT, True), "primary": state(PRIMARY), "submodule": state(ROOT / SUB, True)}
assert before == after, (before, after)
receipt = {"task_id": "D29-LEDGER-COMPOSED-SCOPE-AUDIT", "status": "PASS_FULL_COMPOSED_SCOPE_BINDINGS", "audit_kind": "fresh_all_source_numstats_and_native_patch_reconstruction", "source_commit": SOURCE, "records_commit": RECORDS, "accepted_source_commit": ACCEPTED, "baseline_commit": BASE, "source_tree": text("rev-parse", SOURCE + "^{tree}"), "created_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "owner_actual": dict(owner_actual), "owner_caps": CAPS, "allocation": primitives["allocation"](ledger), "source_actual": source_actual, "global_actual": global_actual, "step": step, "changed_paths": len(actual), "bindings_count": len(bindings), "bindings": bindings, "path_ledger": rows, "overlays": overlays, "inputs": INPUTS, "components": {"superproject": super_counts, "submodule": sub_counts, "native": native_counts, "extras": extras}, "native_pins": native_pins, "published_image": publication["final_image"], "prior_actual_rows_independently_recomputed": 114, "states_before": before, "states_after": after, "violations": [], "commands": COMMANDS, "environment": {"python": sys.version, "executable": sys.executable, "uid": os.getuid(), "bytecode_disabled": sys.dont_write_bytecode, "configured_memory_max_bytes": 32212254720, "effective_cgroup_peak": "not measured", "process_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "timeout_seconds": 60, "math_threads": 1}, "duration_s": time.perf_counter() - started, "limitations": ["Source accounting and bindings only; no review, regression, native or fullCLI admission.", "All116 source paths recomputed; historical per-step caps outside this ledger step retain their original pinned receipts.", "No GPU/platform/cloud operation, pricing/required-shape invocation or project/model/CSV imports.", "Published-image identity is checked from retained records and exported files, not a fresh registry/runtime query."]}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
with (OUT / "paths.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["path", "owner", "kind", "cap", "actual", "headroom"], delimiter="\t", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({k: receipt[k] for k in ["status", "source_commit", "allocation", "owner_actual", "source_actual", "global_actual", "step", "bindings_count", "prior_actual_rows_independently_recomputed", "duration_s"]}, indent=2))
