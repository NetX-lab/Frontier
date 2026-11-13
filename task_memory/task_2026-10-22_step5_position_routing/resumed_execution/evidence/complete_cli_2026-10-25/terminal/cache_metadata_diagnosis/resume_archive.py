"""Archive retained D29 cache diagnostics and prepare the unapproved profile proposal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

started = time.perf_counter()
root = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
task = root / "task_memory/task_2026-10-22_step5_position_routing"
evidence = task / "resumed_execution/evidence"
complete = evidence / "complete_cli_2026-10-25"
terminal = complete / "terminal"
original_collection = json.loads((terminal / "collection_receipt.json").read_bytes())
for entry in original_collection["artifacts"]:
    contents = (root / entry["archive"]).read_bytes()
    assert len(contents) == entry["bytes"]
    assert hashlib.sha256(contents).hexdigest() == entry["sha256"]
assert original_collection["artifact_count"] == 68
assert original_collection["artifact_bytes"] == 15352973
assert original_collection["workflow_qualified"] is False
assert original_collection["original_source_after"] is None

files = {
    "binding_repro.py": "d29_main09_binding_repro_20261025.py",
    "binding_repro.stdout": "d29_main09_binding_repro_20261025.stdout",
    "binding_repro.stderr": "d29_main09_binding_repro_20261025.stderr",
    "binding_repro.execution.json": "d29_main09_binding_repro_20261025.execution.json",
    "cache_bundle_bytes.py": "d29_main09_cache_bundle_bytes_20261025.py",
    "cache_bundle_bytes.json": "d29_main09_cache_bundle_bytes_20261025.json",
    "cache_bundle_bytes.stderr": "d29_main09_cache_bundle_bytes_20261025.stderr",
    "cache_bundle_bytes.execution.json": "d29_main09_cache_bundle_bytes_20261025.execution.json",
}
archive = terminal / "cache_metadata_diagnosis"
assert not archive.exists(), str(archive)
contents = {name: (Path("/data/ycfeng/tmp") / original).read_bytes()
            for name, original in files.items()}
cache_result = json.loads(contents["cache_bundle_bytes.json"])
assert cache_result["status"] == "PASS_CHANGED_CACHE_BYTES_MATCH_FROZEN_BUNDLES"
assert cache_result["files"] == 14 and cache_result["bytes"] == 21025604
assert all(row["matches_frozen_bundle"] and row["metadata_unchanged_during_read"]
           for row in cache_result["rows"])
for extension in ("py", "json"):
    assert (terminal / ("post_run_binding_observation." + extension)).read_bytes() == (
        Path("/data/ycfeng/tmp") / ("d29_main09_post_run_binding_observation_20261025." + extension)
    ).read_bytes()
archive.mkdir()
rows = []
for name, original in files.items():
    target = archive / name
    target.write_bytes(contents[name])
    assert target.read_bytes() == contents[name]
    rows.append({"source": str(Path("/data/ycfeng/tmp") / original),
                 "archive": str(target.relative_to(root)), "bytes": len(contents[name]),
                 "sha256": hashlib.sha256(contents[name]).hexdigest()})

initial = complete / "profile_proposal_initial"
assert not initial.exists(), str(initial)
initial.mkdir()
command_path = complete / "proposed_profile_command.json"
bindings_path = complete / "proposed_profile_bindings.json"
for filename, path in [("command.json", command_path), ("bindings.json", bindings_path)]:
    (initial / filename).write_bytes(path.read_bytes())

bindings = json.loads(bindings_path.read_bytes())
original = json.loads((evidence / "execution_preparation_2026-10-25/joint_bindings.json").read_bytes())
for key in ("source_root", "source_commit", "collective_checkout", "source_hashes", "input_hashes",
            "fixture_root", "references", "python", "python_version", "math_thread_env", "scripts",
            "original_setup"):
    assert bindings[key] == original[key], key
changed = {name for name in original["fixture_metadata"]
           if original["fixture_metadata"][name] != bindings["fixture_metadata"][name]}
assert len(changed) == 14
bindings["task_id"] = "D29-MAIN09-PROFILE-FOLLOWUP-PROPOSAL"
bindings["state"] = "PROPOSED_ADDITIONAL_TWO_INSTRUMENTED_CALLS_NOT_APPROVED"
bindings["records_head_at_proposal"] = original_collection["records_head"]
bindings["proposal"]["profiling_calls"] = 1
bindings["proposal"]["instrumented_pricing_calls"] = 2
bindings["proposal"]["approval_status"] = "NOT_APPROVED"
bindings["limitations"].append(
    "One standard-library cProfile process instruments both pricing attempts; diagnostic timings do not close the complete-CLI gate."
)
active_path = evidence / "execution_preparation_2026-10-25/joint_profile_bindings.json"
assert not active_path.exists(), str(active_path)
active_path.write_text(json.dumps(bindings, indent=2) + "\n")

command = json.loads(command_path.read_bytes())
old_path = command["proposed_bindings"]
assert old_path == str(bindings_path)
command["argv"] = [str(active_path) if value == old_path else value for value in command["argv"]]
command["proposed_bindings"] = str(active_path)
command["proposed_bindings_sha256"] = hashlib.sha256(active_path.read_bytes()).hexdigest()
command["initial_proposal_preserved"] = str(initial.relative_to(root))
command["draft_directory_correction"] = (
    "joint_timing.py resolves setup, wave and script hashes beside the bindings file. The active proposal keeps new bindings in the existing execution_preparation directory; reviewed scripts remain unchanged."
)
command["profile_processes"] = 1
command["instrumented_pricing_calls"] = 2
command["static_preflight"] = "PENDING"
command["independent_review"] = {"Standards": "PENDING", "Spec": "PENDING"}
command["state"] = command["status"] = "PROPOSED_NOT_APPROVED"
command["new_ordinary_pricing_calls_executed"] = 0
command["last_prepared_utc"] = datetime.now(timezone.utc).isoformat()
command_path.write_text(json.dumps(command, indent=2) + "\n")
receipt = {"status": "PASS_RETAINED_DIAGNOSTIC_ARCHIVE_AND_PROPOSAL_PREPARATION_ONLY",
    "collected_utc": datetime.now(timezone.utc).isoformat(),
    "original_terminal_artifacts_verified": len(original_collection["artifacts"]),
    "original_terminal_artifact_bytes": original_collection["artifact_bytes"],
    "cache_diagnostic_artifacts": rows, "cache_diagnostic_artifact_count": len(rows),
    "cache_diagnostic_artifact_bytes": sum(row["bytes"] for row in rows),
    "original_post_run_observation_already_archived": True,
    "initial_proposal_preserved": str(initial.relative_to(root)),
    "active_proposed_bindings": str(active_path.relative_to(root)),
    "proposal_approved": False, "pricing_calls": 0, "workload_calls": 0,
    "required_shape_calls": 0, "platform_calls": 0, "model_loads": 0,
    "production_source_edits": 0, "cleanup_performed": False,
    "wall_s": time.perf_counter() - started}
(archive / "archive_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({key: value for key, value in receipt.items()
                  if key != "cache_diagnostic_artifacts"}))
