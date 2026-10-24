"""Finish the interrupted runtime copy and verify every copied source file."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
DESTINATION = Path("/data/ycfeng/envs/frontier-runtime-py310-d29-20261025")
SOURCES = ((Path("/usr/lib/python3.10"), DESTINATION / "lib/python3.10"),
           (Path("/data/ycfeng/envs/frontier-py310/lib/python3.10/site-packages"), DESTINATION / "lib/python3.10/site-packages"))
PACKAGE_NAMES = ("numpy", "pandas", "scipy", "scikit-learn", "plotly", "fasteners", "ddsketch", "pytest")
assert DESTINATION.is_dir()
started_utc = datetime.now(timezone.utc).isoformat()
started = time.perf_counter()
missing = []
truncated = []
for source_root, target_root in SOURCES:
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = source.relative_to(source_root)
        target = target_root / relative
        if not target.is_file():
            missing.append(str(target.relative_to(DESTINATION)))
        elif target.stat().st_size == source.stat().st_size:
            continue
        else:
            truncated.append(str(target.relative_to(DESTINATION)))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
print(json.dumps({"stage": "remaining_copy_complete", "new_files": len(missing), "partial_files_completed": len(truncated)}), flush=True)
for name in ("python", "python3"):
    alias = DESTINATION / "bin" / name
    assert not alias.exists() and not alias.is_symlink()
    alias.symlink_to("python3.10")
files = 1
bytes_verified = (DESTINATION / "bin/python3.10").stat().st_size
inventory = hashlib.sha256()
source_python = Path("/usr/bin/python3.10")
python_digest = hashlib.sha256(source_python.read_bytes()).hexdigest()
assert hashlib.sha256((DESTINATION / "bin/python3.10").read_bytes()).hexdigest() == python_digest
inventory.update(("bin/python3.10\0" + python_digest + "\n").encode())
for source_root, target_root in SOURCES:
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        target = target_root / source.relative_to(source_root)
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == source_digest, str(target)
        files += 1
        bytes_verified += source.stat().st_size
        inventory.update((str(target.relative_to(DESTINATION)) + "\0" + source_digest + "\n").encode())
receipt = {
    "task_id": "D29-MOUNTED-CPYTHON-RUNTIME", "status": "COMPLETE_BYTE_VERIFIED_NOT_YET_IMPORT_VERIFIED",
    "started_utc": started_utc, "ended_utc": datetime.now(timezone.utc).isoformat(),
    "wall_s": time.perf_counter() - started,
    "destination": str(DESTINATION), "source_python": str(source_python), "source_python_sha256": python_digest,
    "source_stdlib": str(SOURCES[0][0]), "source_packages": str(SOURCES[1][0]),
    "missing_files_completed": missing, "partial_files_completed": truncated,
    "verified_copied_files": files, "verified_copied_bytes": bytes_verified,
    "inventory_sha256": inventory.hexdigest(),
    "package_versions": {name: metadata.version(name) for name in PACKAGE_NAMES},
    "source_version": sys.version, "original_environment_modified": False,
    "dependency_installations": 0,
    "limitations": ["Original timed-out copy retained; only missing and incomplete task-owned destination files were completed.",
                    "No worker, pricing wave, full CLI or native GPU execution; image OS libraries still require import verification."]
}
(HERE / "python_runtime_completed.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({key: value for key, value in receipt.items() if key not in ("missing_files_completed", "partial_files_completed")}, indent=2))
