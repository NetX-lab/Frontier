"""Separate the copied CPython base prefix from its existing dependency prefix."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
PREFIX = Path("/data/ycfeng/envs/frontier-runtime-py310-d29-20261025")
BASE = Path("/data/ycfeng/envs/frontier-runtime-base-py310-d29-20261025")
COPY_RECEIPT = json.loads((HERE / "python_runtime_completed.json").read_text())
assert not BASE.exists()
assert not (PREFIX / "pyvenv.cfg").exists()
assert COPY_RECEIPT["destination"] == str(PREFIX)
started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
BASE.mkdir()
(BASE / "bin").mkdir()
(BASE / "lib").mkdir()
shutil.copy2(PREFIX / "bin/python3.10", BASE / "bin/python3.10")
(BASE / "lib/python3.10").symlink_to(PREFIX / "lib/python3.10", target_is_directory=True)
binary_sha256 = hashlib.sha256((BASE / "bin/python3.10").read_bytes()).hexdigest()
assert binary_sha256 == COPY_RECEIPT["source_python_sha256"]
base_probe_command = [str(BASE / "bin/python3.10"), "-B", "-S", "-c",
                      "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':list(sys.version_info[:3])}))"]
base_probe = subprocess.run(base_probe_command, text=True, capture_output=True, check=True)
base_state = json.loads(base_probe.stdout)
assert base_state == {"prefix": str(BASE), "base_prefix": str(BASE), "version": [3, 10, 6]}, base_state
configuration = f"home = {BASE / 'bin'}\ninclude-system-site-packages = false\nversion = 3.10.6\n"
with (PREFIX / "pyvenv.cfg").open("x") as stream:
    stream.write(configuration)
receipt = {
    "task_id": "D29-MOUNTED-CPYTHON-RUNTIME",
    "status": "PREFIX_CONFIGURED_IMPORT_VERIFICATION_PENDING",
    "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
    "wall_s": time.perf_counter() - started,
    "hypothesis_before_execution": "Debian site.py requires a distinct base prefix to add the existing site-packages directory; a copied binary plus a shared stdlib establishes that prefix without copying dependencies again.",
    "prefix": str(PREFIX), "base_prefix": str(BASE),
    "base_binary_sha256": binary_sha256,
    "shared_stdlib": {"path": str(BASE / "lib/python3.10"), "target": str(PREFIX / "lib/python3.10")},
    "base_probe_command": base_probe_command, "base_probe_state": base_state,
    "pyvenv_cfg": configuration,
    "pyvenv_cfg_sha256": hashlib.sha256(configuration.encode()).hexdigest(),
    "original_environment_modified": False,
    "existing_copied_files_modified": False,
    "dependency_installations": 0, "pricing_wave_calls": 0, "gpu_calls": 0,
    "limitations": ["This only configures the copied runtime; original strict local/image dependency probes must pass."]
}
with (HERE / "python_runtime_configured.json").open("x") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
