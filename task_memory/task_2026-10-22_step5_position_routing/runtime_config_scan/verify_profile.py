"""Inspect retained profiling evidence without rerunning simulator pricing."""

import argparse
import hashlib
import json
from pathlib import Path
import pstats
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--profile", type=Path, required=True)
parser.add_argument("--freeze", type=Path, required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--allowed-change", action="append", default=[])
args = parser.parse_args()
baseline = json.loads(args.freeze.read_text())["sources"]["baseline"]
changes = subprocess.check_output(
    ["git", "diff", "--name-only", baseline["head"], "HEAD"],
    cwd=args.source_root, text=True,
).splitlines()
source_changes = [path for path in changes if not path.startswith("task_memory/")]
assert set(source_changes) == set(args.allowed_change), source_changes
hashes = {
    path: hashlib.sha256((args.source_root / path).read_bytes()).hexdigest()
    for path in baseline["hashes"]
}
changed_bindings = [path for path, value in hashes.items() if value != baseline["hashes"][path]]
assert set(changed_bindings) == set(args.allowed_change), changed_bindings
stats = pstats.Stats(str(args.profile))
config_functions = [
    {"file":key[0], "line":key[1], "function":key[2], "calls":value[1],
     "self_s":value[2], "cumulative_s":value[3]}
    for key, value in stats.stats.items()
    if any(term in (key[0] + key[2]).lower()
           for term in ("model_config", "model_architect", "consisten"))
]
print(json.dumps({
    "status":"PASS_RETAINED_PROFILE_BINDING_ONLY",
    "baseline_revision":baseline["head"],
    "current_revision":subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source_root, text=True,
    ).strip(),
    "profile":str(args.profile), "profile_bytes":args.profile.stat().st_size,
    "profile_sha256":hashlib.sha256(args.profile.read_bytes()).hexdigest(),
    "binding_count":len(hashes), "unchanged_bindings":len(hashes)-len(changed_bindings),
    "changed_bindings":changed_bindings,
    "config_functions":sorted(config_functions, key=lambda row:row["cumulative_s"], reverse=True),
    "limits":"Retained instrumented sample only; no new wave, RSS or end-to-end timing.",
}, indent=2))
