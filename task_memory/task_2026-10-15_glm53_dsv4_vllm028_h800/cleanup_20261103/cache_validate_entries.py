#!/usr/bin/env python3
"""Pre-delete validation for the approved sklearn cache deletion manifest.

Checks, per manifest entry:
- absolute path, inside /data/ycfeng/tmp/glm53_dsv4
- no symlink component, no mount point boundary, no worktree boundary
- entries are non-overlapping (no ancestor/descendant relation)
- directory contents are recursively only .pkl and .file files
- observed file count and byte counts match the manifest snapshot
"""
import json
import os
import stat
import sys

MANIFEST = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261103/"
    "cache_deletion_manifest.json"
)
ALLOWED_ROOT = "/data/ycfeng/tmp/glm53_dsv4"
ALLOWED_EXT = {".pkl", ".file"}


def mount_points():
    points = []
    with open("/proc/self/mounts") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                points.append(parts[1].replace("\\040", " "))
    return sorted(points)


def symlink_components(path):
    """Return the list of path components that are symlinks."""
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


def walk_entry(path):
    files = {"count": 0, "alloc": 0, "logical": 0, "by_ext": {}}
    problems = {"non_file": [], "unexpected_ext": [], "symlinks": [], "subdirs": 0}
    for root, dirs, names in os.walk(path, followlinks=False):
        problems["subdirs"] += len(dirs)
        for name in dirs:
            full = os.path.join(root, name)
            if os.path.islink(full):
                problems["symlinks"].append(full)
        for name in names:
            full = os.path.join(root, name)
            try:
                st = os.lstat(full)
            except OSError as exc:
                problems["non_file"].append(f"{full}: {exc}")
                continue
            if stat.S_ISLNK(st.st_mode):
                problems["symlinks"].append(full)
                continue
            if not stat.S_ISREG(st.st_mode):
                problems["non_file"].append(f"{full}: mode={oct(st.st_mode)}")
                continue
            ext = os.path.splitext(name)[1]
            files["by_ext"][ext] = files["by_ext"].get(ext, 0) + 1
            if ext not in ALLOWED_EXT:
                problems["unexpected_ext"].append(full)
            files["count"] += 1
            files["alloc"] += st.st_blocks * 512
            files["logical"] += st.st_size
        if not dirs and not names:
            pass
    return files, problems


def main():
    manifest = json.load(open(MANIFEST))
    entries = manifest["entries"]
    mounts = mount_points()
    paths = [e["path"] for e in entries]

    report = {"manifest": MANIFEST, "entries": [], "global_errors": []}

    # Overlap and containment checks over the full list.
    for i, a in enumerate(paths):
        if not a.startswith(ALLOWED_ROOT + "/"):
            report["global_errors"].append(f"outside allowed root: {a}")
        for b in paths[i + 1:]:
            if a == b:
                report["global_errors"].append(f"duplicate path: {a}")
            elif a.startswith(b + "/") or b.startswith(a + "/"):
                report["global_errors"].append(f"overlapping paths: {a} <> {b}")

    for entry in entries:
        path = entry["path"]
        row = {
            "path": path,
            "expected": {
                "file_count": entry["file_count"],
                "allocated_file_bytes": entry["allocated_file_bytes"],
                "logical_file_bytes": entry["logical_file_bytes"],
            },
            "status": "ok",
            "skips": [],
        }
        if entry.get("action") != "remove_directory":
            row["status"] = "skip"
            row["skips"].append(f"unexpected action {entry.get('action')}")
            report["entries"].append(row)
            continue
        if ".." in path.split("/"):
            row["status"] = "skip"
            row["skips"].append("path contains ..")
            report["entries"].append(row)
            continue
        real = os.path.realpath(path)
        row["realpath"] = real
        if real != path:
            row["status"] = "skip"
            row["skips"].append(f"realpath differs: {real}")
            report["entries"].append(row)
            continue
        if real == ALLOWED_ROOT or real.startswith(ALLOWED_ROOT + "/") is False:
            row["status"] = "skip"
            row["skips"].append("realpath outside allowed root")
            report["entries"].append(row)
            continue
        if "/.worktrees/" in real or "/worktrees/" in real:
            row["status"] = "skip"
            row["skips"].append("path crosses a worktree boundary")
            report["entries"].append(row)
            continue

        bad_links = symlink_components(path)
        if bad_links:
            row["status"] = "skip"
            row["skips"].append(f"symlink components: {bad_links}")
            report["entries"].append(row)
            continue

        if not os.path.isdir(path) or os.path.islink(path):
            row["status"] = "skip"
            row["skips"].append("missing or not a directory")
            report["entries"].append(row)
            continue

        crossing_mounts = [m for m in mounts if m == path or m.startswith(path + "/")]
        row["nested_mounts"] = crossing_mounts
        if crossing_mounts:
            row["status"] = "skip"
            row["skips"].append(f"mount points inside entry: {crossing_mounts}")
            report["entries"].append(row)
            continue

        st = os.stat(path)
        row["st_dev"] = st.st_dev
        row["st_ino"] = st.st_ino
        parent = os.stat(os.path.dirname(path))
        row["parent_st_dev"] = parent.st_dev
        if parent.st_dev != st.st_dev:
            row["status"] = "skip"
            row["skips"].append("entry st_dev differs from parent (mount boundary)")
            report["entries"].append(row)
            continue

        files, problems = walk_entry(path)
        row["observed"] = files
        row["problems"] = {
            k: ([v[:10], len(v)] if isinstance(v, list) else v)
            for k, v in problems.items()
        }
        if problems["symlinks"]:
            row["skips"].append(f"symlinks inside entry: {problems['symlinks'][:5]}")
        if problems["non_file"]:
            row["skips"].append(f"non-regular files: {problems['non_file'][:5]}")
        if problems["unexpected_ext"]:
            row["skips"].append(f"unexpected extensions: {problems['unexpected_ext'][:5]}")
        if files["count"] != entry["file_count"]:
            row["skips"].append(
                f"file count mismatch: observed {files['count']} != manifest {entry['file_count']}"
            )
        if files["alloc"] != entry["allocated_file_bytes"]:
            row["skips"].append(
                f"allocated bytes mismatch: observed {files['alloc']} != manifest {entry['allocated_file_bytes']}"
            )
        if files["logical"] != entry["logical_file_bytes"]:
            row["skips"].append(
                f"logical bytes mismatch: observed {files['logical']} != manifest {entry['logical_file_bytes']}"
            )
        if set(files["by_ext"]) - ALLOWED_EXT:
            row["skips"].append(f"extension set: {sorted(files['by_ext'])}")
        if row["skips"]:
            row["status"] = "skip"
        report["entries"].append(row)

    ok = [r for r in report["entries"] if r["status"] == "ok"]
    skipped = [r for r in report["entries"] if r["status"] != "ok"]
    report["summary"] = {
        "total_entries": len(entries),
        "verified_ok": len(ok),
        "skipped": len(skipped),
        "verified_file_count": sum(r["observed"]["count"] for r in ok),
        "verified_allocated_bytes": sum(r["observed"]["alloc"] for r in ok),
        "verified_logical_bytes": sum(r["observed"]["logical"] for r in ok),
    }
    json.dump(report, open(sys.argv[1], "w"), indent=2)
    print(json.dumps(report["summary"], indent=2))
    if report["global_errors"]:
        print("GLOBAL ERRORS:")
        for err in report["global_errors"]:
            print(" ", err)
    for row in skipped:
        print("SKIP", row["path"])
        for reason in row["skips"]:
            print("   ", reason)


if __name__ == "__main__":
    main()
