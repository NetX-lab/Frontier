#!/usr/bin/env python3
"""Delete exactly the approved cache directories, one exact path at a time.

Each entry is re-verified immediately before deletion (regular directory, no
symlink component, contents recursively .pkl/.file only, file count and byte
totals still matching the manifest snapshot). Deletion uses `rm -rf -- <path>`
with an argv list, so no shell glob expansion or broader scan can occur.
"""
import json
import os
import stat
import subprocess
import sys
import time

MANIFEST = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261103/"
    "cache_deletion_manifest.json"
)
ALLOWED_ROOT = "/data/ycfeng/tmp/glm53_dsv4"
ALLOWED_EXT = {".pkl", ".file"}


def measure(path):
    files = {"count": 0, "alloc": 0, "logical": 0, "exts": set()}
    for root, dirs, names in os.walk(path, followlinks=False):
        for name in dirs:
            if os.path.islink(os.path.join(root, name)):
                raise RuntimeError(f"symlink dir inside entry: {os.path.join(root, name)}")
        for name in names:
            full = os.path.join(root, name)
            st = os.lstat(full)
            if not stat.S_ISREG(st.st_mode):
                raise RuntimeError(f"non-regular file inside entry: {full}")
            ext = os.path.splitext(name)[1]
            if ext not in ALLOWED_EXT:
                raise RuntimeError(f"unexpected extension inside entry: {full}")
            files["exts"].add(ext)
            files["count"] += 1
            files["alloc"] += st.st_blocks * 512
            files["logical"] += st.st_size
    files["exts"] = sorted(files["exts"])
    return files


def symlink_components(path):
    bad = []
    current = "/"
    for part in path.strip("/").split("/"):
        current = os.path.join(current, part)
        try:
            st = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(st.st_mode):
            bad.append(current)
    return bad


def main():
    entries = json.load(open(MANIFEST))["entries"]
    validated = {r["path"]: r for r in json.load(open("validation_pre.json"))["entries"]}
    results = []
    started = time.time()

    for entry in entries:
        path = entry["path"]
        row = {"path": path, "expected_file_count": entry["file_count"],
               "expected_allocated_bytes": entry["allocated_file_bytes"],
               "expected_logical_bytes": entry["logical_file_bytes"]}
        if validated.get(path, {}).get("status") != "ok":
            row["outcome"] = "skipped"
            row["reason"] = "entry did not pass pre-deletion validation"
            results.append(row)
            continue
        try:
            if path != os.path.realpath(path):
                raise RuntimeError("realpath differs from manifest path")
            if not path.startswith(ALLOWED_ROOT + "/"):
                raise RuntimeError("outside allowed root")
            bad = symlink_components(path)
            if bad:
                raise RuntimeError(f"symlink components: {bad}")
            if not os.path.isdir(path) or os.path.islink(path):
                raise RuntimeError("missing or not a plain directory")
            observed = measure(path)
            row["pre_file_count"] = observed["count"]
            row["pre_allocated_bytes"] = observed["alloc"]
            row["pre_logical_bytes"] = observed["logical"]
            row["extensions"] = observed["exts"]
            mismatch = [
                key for key, want in (
                    ("count", entry["file_count"]),
                    ("alloc", entry["allocated_file_bytes"]),
                    ("logical", entry["logical_file_bytes"]),
                ) if observed[key] != want
            ]
            if mismatch:
                raise RuntimeError(f"changed since manifest snapshot: {mismatch}")
            cmd = ["rm", "-rf", "--", path]
            row["command"] = " ".join(cmd)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            row["rm_returncode"] = proc.returncode
            row["rm_stdout"] = proc.stdout[-2000:]
            row["rm_stderr"] = proc.stderr[-2000:]
            exists = os.path.lexists(path)
            row["exists_after"] = exists
            if proc.returncode != 0 or exists:
                raise RuntimeError(f"rm failed (rc={proc.returncode}) or path still exists")
            row["outcome"] = "deleted"
        except Exception as exc:  # noqa: BLE001 - per-entry isolation for reporting
            row["outcome"] = "skipped"
            row["reason"] = str(exc)
        row["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        results.append(row)
        print(row["outcome"], path, flush=True)

    deleted = [r for r in results if r["outcome"] == "deleted"]
    skipped = [r for r in results if r["outcome"] != "deleted"]
    summary = {
        "total_entries": len(entries),
        "deleted": len(deleted),
        "skipped": len(skipped),
        "deleted_file_count": sum(r["pre_file_count"] for r in deleted),
        "deleted_allocated_bytes": sum(r["pre_allocated_bytes"] for r in deleted),
        "deleted_logical_bytes": sum(r["pre_logical_bytes"] for r in deleted),
        "runtime_seconds": round(time.time() - started, 3),
    }
    json.dump({"summary": summary, "results": results},
              open(sys.argv[1], "w"), indent=2)
    print(json.dumps(summary, indent=2))
    for row in skipped:
        print("SKIP", row["path"], "->", row.get("reason"))


if __name__ == "__main__":
    main()
