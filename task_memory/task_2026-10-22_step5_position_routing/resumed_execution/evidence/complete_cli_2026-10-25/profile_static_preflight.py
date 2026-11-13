"""Check the pending D29 cProfile command without model setup or pricing."""
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

started = time.perf_counter()
root = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
evidence = root / "task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence"
complete = evidence / "complete_cli_2026-10-25"
preparation = evidence / "execution_preparation_2026-10-25"
command = json.loads((complete / "proposed_profile_command.json").read_bytes())
active_path = Path(command["proposed_bindings"])
bindings = json.loads(active_path.read_bytes())
original = json.loads((preparation / "joint_bindings.json").read_bytes())
def require(condition, message):
    if not condition:
        raise AssertionError(message)

require(command["status"] == command["state"] == "PROPOSED_NOT_APPROVED", "Approval state")
require(command["new_ordinary_pricing_calls_executed"] == 0, "No execution is authorized")
require(active_path.parent == preparation, "Existing driver resolves siblings beside bindings")
require(hashlib.sha256(active_path.read_bytes()).hexdigest() == command["proposed_bindings_sha256"], "Bindings SHA")
require(sys.version.split()[0] == bindings["python_version"], "Python version")
require(str(Path(sys.executable).absolute()) == bindings["python"], "Interpreter path")
require(sys.dont_write_bytecode, "Python -B")
for name, value in bindings["math_thread_env"].items():
    require(os.environ.get(name) == value, name)
require(command["argv"][:9] == ["timeout", "60s", "sudo", "-n", "systemd-run", "--scope", "--quiet", "-p", "MemoryMax=30G"], "Finite process limits")
argv = command["argv"]
require(argv[argv.index("--bindings") + 1] == str(active_path), "Command bindings path")
require(argv[argv.index("--output-root") + 1] == command["output_root"], "Output path")
require(argv[argv.index("cProfile") - 1] == "-m", "Standard-library cProfile")
require(argv[argv.index("cProfile") + 1:argv.index("cProfile") + 4] == ["-o", command["profile"], str(preparation / "joint_timing.py")], "Profile command")
for key in ("output_root", "profile"):
    path = Path(command[key])
    require(path.is_relative_to(Path("/data/ycfeng/tmp")) and not path.exists(), "Fresh local output: " + key)
for key in ("new_ordinary_pricing_calls_proposed", "instrumented_pricing_calls"):
    require(command[key] == 2, key)
for key in ("new_required_shape_calls", "new_gpu_allocations", "new_rjob_submissions", "automatic_retries", "warmup_calls", "source_or_sdk_edits"):
    require(command[key] == 0, key)
require(command["existing_ordinary_joint_used_and_limit"] == [2, 2], "Exhausted joint calls")
require(command["required_shape_used_and_limit"] == [6, 6], "Exhausted shape calls")
require(bindings["proposal"]["labels"] == ["cold", "warm"] and bindings["proposal"]["production_pricing_calls"] == 2, "Exactly two pricing attempts")
require(bindings["proposal"]["profiling_calls"] == 1 and command["profile_processes"] == 1, "Exactly one profiling process")
require(bindings["proposal"]["approval_status"] == "NOT_APPROVED", "Pending finite decision")
for key in ("source_root", "source_commit", "collective_checkout", "source_hashes", "input_hashes", "fixture_root", "references", "python", "python_version", "math_thread_env", "scripts", "original_setup"):
    require(bindings[key] == original[key], "Preserved binding: " + key)
require(set(bindings["fixture_metadata"]) == set(original["fixture_metadata"]), "All metadata paths retained")
changes = {name for name in original["fixture_metadata"] if original["fixture_metadata"][name] != bindings["fixture_metadata"][name]}
observation = json.loads((complete / "terminal/post_run_binding_observation.json").read_bytes())
require(changes == {row["relative"] for row in observation["fixture_metadata_changes"]} and len(changes) == 14, "Only fourteen observed cache metadata changes")
for row in observation["fixture_metadata_changes"]:
    require(bindings["fixture_metadata"][row["relative"]] == row["actual_size_mtime_ns"], "Recorded new metadata")
cache_proof = Path(bindings["post_main09_cache_snapshot"]["verification"])
cache = json.loads(cache_proof.read_bytes())
require(cache["status"] == "PASS_CHANGED_CACHE_BYTES_MATCH_FROZEN_BUNDLES" and cache["files"] == 14 and cache["bytes"] == 21025604, "Exact retained cache proof")
require(all(row["matches_frozen_bundle"] and row["metadata_unchanged_during_read"] for row in cache["rows"]), "Retained cache SHA checks")
plan = json.loads((complete / "complete_cli_plan.json").read_bytes())
git_assets = plan["git"]
for filename, expected in git_assets["files"].items():
    require(hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, "Existing Git asset")
require(os.environ.get("GIT_CONFIG_GLOBAL") == git_assets["config"], "Task Git configuration")
configured_dirs = [line.split("=", 1)[1].strip() for line in Path(git_assets["config"]).read_text().splitlines() if line.strip().startswith("directory")]
require(configured_dirs == [str(root), str(root / "frontier/cc_backend/backends/collective-sim")], "Exactly two trusted repositories")
for filename in bindings["scripts"]:
    ast.parse((preparation / filename).read_bytes(), filename=filename)
spec = importlib.util.spec_from_file_location("d29_static_joint_driver", preparation / "joint_timing.py")
joint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(joint)
initial = json.loads((complete / "profile_proposal_initial/bindings.json").read_bytes())
try:
    joint.check_bindings(initial, complete)
except FileNotFoundError as error:
    require(Path(error.filename) == complete / "joint_setup.py", "Expected initial sibling-directory defect")
    initial_failure = {"type": type(error).__name__, "message": str(error), "before_setup_or_pricing": True}
else:
    raise AssertionError("Initial misplaced bindings were not rejected")
source = joint.check_bindings(bindings, active_path.parent)
references = joint.read_references(bindings)
require(len(references[1]) == 16 and len(references[2]) == 208, "Full references retained")
require(sum(len(row["moe"]) + len(row["comm"]) for row in json.loads(references[0])) == 208, "Pricing reference operators")
require(sum(len(row["proofs"]) for row in json.loads(references[0])) == 144, "Pricing reference proofs")
require(not any(name == "frontier" or name.startswith("frontier.") for name in sys.modules), "No production imports or model setup")
result = {"status": "PASS_STATIC_PROFILE_PROPOSAL_ONLY", "finished_utc": datetime.now(timezone.utc).isoformat(),
    "source": source, "active_bindings": str(active_path), "bindings_sha256": hashlib.sha256(active_path.read_bytes()).hexdigest(),
    "initial_directory_defect_reproduced": initial_failure, "source_paths": len(bindings["source_hashes"]),
    "input_paths": len(bindings["input_hashes"]), "fixture_metadata_paths": len(bindings["fixture_metadata"]),
    "changed_cache_metadata_paths": len(changes), "reviewed_script_hashes": len(bindings["scripts"]),
    "full_reference_ledger_rows": len(references[1]), "full_reference_operator_rows": len(references[2]), "full_reference_compute_proofs": 144,
    "pricing_calls": 0, "workload_calls": 0, "required_shape_calls": 0, "platform_calls": 0, "model_loads": 0,
    "cprofile_application_executions": 0, "configured_memory_max": "30G", "proposal_approved": False,
    "wall_s": time.perf_counter() - started}
(complete / "profile_static_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
