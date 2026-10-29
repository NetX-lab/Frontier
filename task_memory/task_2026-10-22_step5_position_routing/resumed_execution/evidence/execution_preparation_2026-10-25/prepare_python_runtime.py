"""Copy the existing CPython runtime and installed packages into a new local prefix."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import sysconfig
import time

HERE = Path(__file__).resolve().parent
DESTINATION = Path("/data/ycfeng/envs/frontier-runtime-py310-d29-20261025")
SOURCE_STDLIB = Path(sysconfig.get_path("stdlib"))
SOURCE_PACKAGES = Path(sysconfig.get_path("purelib"))
SOURCE_PYTHON = Path(sys.executable).resolve()
PACKAGE_NAMES = ("numpy", "pandas", "scipy", "scikit-learn", "plotly", "fasteners", "ddsketch", "pytest")
assert sys.version_info[:3] == (3, 10, 6)
assert SOURCE_STDLIB == Path("/usr/lib/python3.10")
assert SOURCE_PACKAGES == Path("/data/ycfeng/envs/frontier-py310/lib/python3.10/site-packages")
assert not DESTINATION.exists(), "The runtime destination must be new."
started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
DESTINATION.mkdir()
(DESTINATION / "bin").mkdir()
(DESTINATION / "lib").mkdir()
shutil.copy2(SOURCE_PYTHON, DESTINATION / "bin/python3.10")
shutil.copytree(SOURCE_STDLIB, DESTINATION / "lib/python3.10")
shutil.copytree(SOURCE_PACKAGES, DESTINATION / "lib/python3.10/site-packages")
for name in ("python", "python3"):
    (DESTINATION / "bin" / name).symlink_to("python3.10")
files = 0
bytes_copied = 0
inventory = hashlib.sha256()
for relative_root, source_root in (("bin", SOURCE_PYTHON.parent), ("lib/python3.10", SOURCE_STDLIB), ("lib/python3.10/site-packages", SOURCE_PACKAGES)):
    selected = [Path("python3.10")] if relative_root == "bin" else sorted(path.relative_to(source_root) for path in source_root.rglob("*") if path.is_file())
    for relative in selected:
        source = source_root / relative
        destination = DESTINATION / relative_root / relative
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        destination_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        assert source_digest == destination_digest, str(relative)
        size = source.stat().st_size
        files += 1
        bytes_copied += size
        inventory.update((str(Path(relative_root) / relative) + "\0" + source_digest + "\n").encode())
receipt = {
    "task_id": "D29-MOUNTED-CPYTHON-RUNTIME",
    "status": "COPIED_NOT_YET_IMPORT_VERIFIED",
    "started_utc": started_utc,
    "ended_utc": datetime.now(timezone.utc).isoformat(),
    "wall_s": time.perf_counter() - started,
    "source_python": str(SOURCE_PYTHON),
    "source_python_sha256": hashlib.sha256(SOURCE_PYTHON.read_bytes()).hexdigest(),
    "source_stdlib": str(SOURCE_STDLIB),
    "source_packages": str(SOURCE_PACKAGES),
    "destination": str(DESTINATION),
    "verified_copied_files": files,
    "verified_copied_bytes": bytes_copied,
    "inventory_sha256": inventory.hexdigest(),
    "package_versions": {name: metadata.version(name) for name in PACKAGE_NAMES},
    "source_version": sys.version,
    "original_environment_modified": False,
    "dependency_installations": 0,
    "limitations": ["No worker, pricing wave, full CLI or native GPU execution.", "The image supplies OS shared libraries; image imports still require verification."]
}
(HERE / "python_runtime_copy.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
