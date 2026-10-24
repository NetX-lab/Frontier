"""Keep the original one-attempt launcher alive and retain its local receipt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path("/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028")
primary = root / "task_memory/task_2026-10-23_v4_remaining_measurements/profile01/scope_lifetime/package_primary"
seal_path = primary / "root_seal.json"
seal = json.loads(seal_path.read_text())
package = Path(seal["package"])
assert seal["status"] == "SEALED_READY_SINGLE_SCOPE_LIFETIME_PROBE"
assert hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest() == seal["manifest_sha256"]
assert not (package / "launcher_started.json").exists()
receipt_path = primary / "root_submission.json"
command = ["bash", "/data/ycfeng/tmp/glm53_dsv4/scope_prelaunch_20261025_01/with_personal_env_ready.sh", "/data/ycfeng/tmp/stepmind-env/bin/python", "-u", "-B", str(package / "launch.py"), "--launch"]
log = Path("/data/ycfeng/tmp/glm53_dsv4/scope_prelaunch_20261025_01/launcher_01.raw.log")
receipt = {"status": "LAUNCH_REQUESTED_NO_JOB_YET", "start_utc": datetime.now(timezone.utc).isoformat(), "command": command, "seal_sha256": hashlib.sha256(seal_path.read_bytes()).hexdigest(), "manifest_sha256": seal["manifest_sha256"], "executable_package": str(package), "log": str(log), "gpu_attempt_ceiling": 1, "gpu_submission_verified": False, "limits": "One authorized original launcher invocation; platform creation and runtime acceptance remain unverified until exact API and business checks."}
with receipt_path.open("x") as output:
    json.dump(receipt, output, indent=2)
    output.write("\n")
with log.open("x") as output:
    child = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT)
    receipt["launcher_pid"] = child.pid
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "launcher_pid": child.pid, "log": str(log)}), flush=True)
    result = child.wait()
receipt.update(launcher_exit=result, launcher_end_utc=datetime.now(timezone.utc).isoformat())
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"launcher_exit": result, "log": str(log)}), flush=True)
sys.exit(result)
