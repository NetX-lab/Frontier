"""Read completed NIXL6144 artifacts from the authorized personal cloud directory.

Runs a read-only tar stream over `ssh shai-ycfeng` (where the personal cloud
directory is mounted at /mnt/ws-jfs/ycfeng), then checks every extracted file
against the streamed SHA-256 manifest.
"""
import datetime
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

CLOUD_ROOT = "/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/p2_nixl6144_q1_20261102"
REMOTE_ROOT = CLOUD_ROOT.replace("/mnt/codesign-exp/ycfeng", "/mnt/ws-jfs/ycfeng", 1)
LOCAL = Path("/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102")
DEST = LOCAL / "download"

PROGRAM = r'''from pathlib import Path
import hashlib, io, json, sys, tarfile
root = Path(REMOTE_ROOT)
required = [root / "C_nixl" / name for name in
            ("nixl_plan.json", "nixl_reads.jsonl", "nixl_summary.csv")]
missing = [str(p) for p in required if not p.is_file()]
nodes = sorted((root / "nodes").iterdir()) if (root / "nodes").is_dir() else []
runtime = sorted((root / "runtime_checks").iterdir()) if (root / "runtime_checks").is_dir() else []
paths = [p for p in required if p.is_file()]
for node in nodes:
    paths.extend(p for p in sorted(node.rglob("*")) if p.is_file())
paths.extend(p for p in runtime if p.is_file())
summary = {"root": str(root), "missing": missing,
           "nodes": [n.name for n in nodes], "runtime_checks": [p.name for p in runtime],
           "files": len(paths)}
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
    manifest = []
    for path in paths:
        value = path.read_bytes()
        name = str(path.relative_to(root))
        info = tarfile.TarInfo(name)
        info.size = len(value)
        archive.addfile(info, io.BytesIO(value))
        manifest.append(dict(path=name, bytes=len(value),
                             sha256=hashlib.sha256(value).hexdigest()))
    for name, value in (("download_manifest.json",
                         (json.dumps(manifest, indent=2) + "\n").encode()),
                        ("download_remote_summary.json",
                         (json.dumps(summary, indent=2) + "\n").encode())):
        info = tarfile.TarInfo(name)
        info.size = len(value)
        archive.addfile(info, io.BytesIO(value))
'''.replace("REMOTE_ROOT", repr(REMOTE_ROOT))

command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "shai-ycfeng",
           'test "$(id -un)" = i-fengyicheng && python3 -u -']
DEST.mkdir(parents=True, exist_ok=True)
archive_path = LOCAL / "selected.tar.gz"
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
with archive_path.open("wb") as output:
    response = subprocess.run(command, input=PROGRAM.encode(), stdout=output,
                              stderr=subprocess.PIPE, timeout=180)
if response.returncode:
    raise RuntimeError(response.stderr.decode())
with tarfile.open(archive_path) as bundle:
    for member in bundle:
        target = DEST / member.name
        assert member.isfile() and target.resolve().is_relative_to(DEST.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        with bundle.extractfile(member) as source, target.open("wb") as output:
            output.write(source.read())
manifest = json.loads((DEST / "download_manifest.json").read_text())
for row in manifest:
    value = (DEST / row["path"]).read_bytes()
    assert len(value) == row["bytes"], row["path"]
    assert hashlib.sha256(value).hexdigest() == row["sha256"], row["path"]
remote = json.loads((DEST / "download_remote_summary.json").read_text())
record = {
    "started_utc": started,
    "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "cloud_root": CLOUD_ROOT,
    "remote_root": REMOTE_ROOT,
    "local_root": str(DEST),
    "transfer": "read-only tar stream over ssh shai-ycfeng",
    "files": len(manifest),
    "bytes": sum(row["bytes"] for row in manifest),
    "hashes_equal": True,
    "missing_remote": remote["missing"],
    "remote_nodes": remote["nodes"],
    "remote_runtime_checks": remote["runtime_checks"],
    "paths": [row["path"] for row in manifest],
}
(DEST / "download.json").write_text(json.dumps(record, indent=2) + "\n")
print("DOWNLOADED", len(manifest), "files", record["bytes"], "bytes; hashes match",
      "missing", remote["missing"], "nodes", remote["nodes"], flush=True)
