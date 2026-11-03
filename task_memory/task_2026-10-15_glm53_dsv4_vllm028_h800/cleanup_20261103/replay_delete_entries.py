#!/usr/bin/env python3
"""Delete exactly the approved replay/fidelity directories, one exact path each.

Re-verifies immediately before deletion: exact manifest path, no symlink
component, plain directory, regular-file count/bytes still matching the
manifest, and the symlink set still exactly equal to the manifest map. Then
runs `rm -rf -- <path>` via an argv list (no shell, no glob). rm unlinks
symlink entries without following their targets.
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
    "replay_deletion_manifest.json"
)
ALLOWED_ROOT = "/data/ycfeng/tmp/glm53_dsv4"


def measure(path):
    """Return regular-file totals and the observed symlink map without following links."""
    files = {"regular": 0, "alloc": 0, "logical": 0, "symlinks": 0,
             "non_regular": 0}
    symlinks = {}
    for root, dirs, names in os.walk(path, followlinks=False):
        kept = []
        for name in dirs:
            full = os.path.join(root, name)
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                symlinks[full] = os.readlink(full)
                files["symlinks"] += 1
            else:
                kept.append(name)
        dirs[:] = kept
        for name in names:
            full = os.path.join(root, name)
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                symlinks[full] = os.readlink(full)
                files["symlinks"] += 1
                continue
            if not stat.S_ISREG(st.st_mode):
                files["non_regular"] += 1
                continue
            files["regular"] += 1
            files["alloc"] += st.st_blocks * 512
            files["logical"] += st.st_size
    return files, symlinks


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
    validated = {r["path"]: r for r in
                 json.load(open("replay_validation_pre.json"))["entries"]}
    results = []
    started = time.time()

    for entry in entries:
        path = entry["path"]
        row = {"path": path, "category": entry["category"],
               "expected_file_count": entry["file_count"],
               "expected_logical_file_bytes": entry["logical_file_bytes"],
               "expected_allocated_file_bytes": entry["allocated_file_bytes"],
               "expected_symlinks": len(entry["symlinks"])}
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
            files, symlinks = measure(path)
            row["pre_file_count"] = files["regular"]
            row["pre_logical_file_bytes"] = files["logical"]
            row["pre_allocated_file_bytes"] = files["alloc"]
            row["pre_symlink_count"] = files["symlinks"]
            mismatch = []
            if files["regular"] != entry["file_count"]:
                mismatch.append(f"regular {files['regular']} != {entry['file_count']}")
            if files["logical"] != entry["logical_file_bytes"]:
                mismatch.append(
                    f"logical {files['logical']} != {entry['logical_file_bytes']}")
            if files["alloc"] != entry["allocated_file_bytes"]:
                mismatch.append(
                    f"allocated {files['alloc']} != {entry['allocated_file_bytes']}")
            if files["non_regular"]:
                mismatch.append(f"non-regular files: {files['non_regular']}")
            expected_links = {s["path"]: s["target"] for s in entry["symlinks"]}
            if symlinks != expected_links:
                mismatch.append(
                    "symlink map differs: "
                    f"missing={len(set(expected_links) - set(symlinks))} "
                    f"unexpected={len(set(symlinks) - set(expected_links))} "
                    f"changed={sum(1 for p in expected_links if p in symlinks and symlinks[p] != expected_links[p])}")
            if mismatch:
                raise RuntimeError("changed since manifest snapshot: " + "; ".join(mismatch))
            outside = sorted(p for p, t in symlinks.items()
                             if not (t == path or t.startswith(path + "/")))
            row["symlinks_targeting_outside_entry"] = outside
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
        "deleted_symlink_count": sum(r["pre_symlink_count"] for r in deleted),
        "deleted_logical_file_bytes": sum(r["pre_logical_file_bytes"] for r in deleted),
        "deleted_allocated_file_bytes": sum(r["pre_allocated_file_bytes"] for r in deleted),
        "runtime_seconds": round(time.time() - started, 3),
    }
    json.dump({"summary": summary, "results": results},
              open(sys.argv[1], "w"), indent=2)
    print(json.dumps(summary, indent=2))
    for row in skipped:
        print("SKIP", row["path"], "->", row.get("reason"))


if __name__ == "__main__":
    main()
