#!/usr/bin/env python3
"""Pre-deletion validation for the approved replay/fidelity cleanup manifest.

Per entry:
- absolute manifest path inside /data/ycfeng/tmp/glm53_dsv4, realpath == manifest path
- every path component a plain directory (no symlink ancestors), entry a plain dir
- no mount point at or inside the entry; parent st_dev == entry st_dev
- arbitrary regular file types allowed; regular-file count / st_blocks*512 /
  logical bytes must equal the manifest snapshot
- symlinks must match the manifest set exactly (path -> target), are recorded,
  and their targets are never opened or traversed
- non-regular non-symlink files are reported as invalid
- hard-link observations (st_nlink > 1) are collected during the same traversal
"""
import json
import os
import stat
import sys

MANIFEST = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261103/"
    "replay_deletion_manifest.json"
)
ALLOWED_ROOT = "/data/ycfeng/tmp/glm53_dsv4"


def mount_points():
    points = []
    with open("/proc/self/mounts") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                points.append(parts[1].replace("\\040", " "))
    return points


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


def walk_entry(path):
    """Traverse without following any symlink; return counts and symlink map."""
    files = {"regular": 0, "alloc": 0, "logical": 0, "symlinks": 0,
             "link_to_dir": 0, "link_to_file": 0, "link_to_missing": 0,
             "link_absolute": 0, "link_inside_entry": 0, "non_regular": 0}
    symlinks = {}
    hardlinks = {"files_with_nlink_gt1": 0, "max_nlink": 0,
                 "inode_groups": {}}
    problems = {"non_regular": [], "symlink_targets_missing": []}
    for root, dirs, names in os.walk(path, followlinks=False):
        kept = []
        for name in dirs:
            full = os.path.join(root, name)
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                files["symlinks"] += 1
                target = os.readlink(full)
                symlinks[full] = target
                if os.path.isabs(target):
                    files["link_absolute"] += 1
                    if target == path or target.startswith(path + "/"):
                        files["link_inside_entry"] += 1
                if os.path.isdir(full):  # follows the link only to classify it
                    files["link_to_dir"] += 1
                elif os.path.lexists(full):
                    files["link_to_file"] += 1
                else:
                    files["link_to_missing"] += 1
                    problems["symlink_targets_missing"].append(full)
            else:
                kept.append(name)
        dirs[:] = kept
        for name in names:
            full = os.path.join(root, name)
            st = os.lstat(full)
            if stat.S_ISLNK(st.st_mode):
                files["symlinks"] += 1
                target = os.readlink(full)
                symlinks[full] = target
                if os.path.isabs(target):
                    files["link_absolute"] += 1
                    if target == path or target.startswith(path + "/"):
                        files["link_inside_entry"] += 1
                if os.path.isdir(full):
                    files["link_to_dir"] += 1
                elif os.path.lexists(full):
                    files["link_to_file"] += 1
                else:
                    files["link_to_missing"] += 1
                    problems["symlink_targets_missing"].append(full)
                continue
            if not stat.S_ISREG(st.st_mode):
                files["non_regular"] += 1
                problems["non_regular"].append(f"{full}: mode={oct(st.st_mode)}")
                continue
            files["regular"] += 1
            files["alloc"] += st.st_blocks * 512
            files["logical"] += st.st_size
            if st.st_nlink > 1:
                hardlinks["files_with_nlink_gt1"] += 1
                hardlinks["max_nlink"] = max(hardlinks["max_nlink"], st.st_nlink)
                key = (st.st_dev, st.st_ino)
                hardlinks["inode_groups"].setdefault(key, []).append(full)
    # Keep the hard-link report small: only groups with more than one member.
    groups = [
        {"dev": key[0], "ino": key[1], "nlink": len(members),
         "sample_paths": members[:3]}
        for key, members in hardlinks["inode_groups"].items() if len(members) > 1
    ]
    hardlinks.pop("inode_groups")
    hardlinks["multi_member_group_count"] = len(groups)
    hardlinks["multi_member_groups_sample"] = groups[:10]
    return files, symlinks, hardlinks, problems


def main():
    manifest = json.load(open(MANIFEST))
    entries = manifest["entries"]
    mounts = mount_points()
    paths = [e["path"] for e in entries]
    report = {"manifest": MANIFEST, "entries": [], "global_errors": []}

    for i, a in enumerate(paths):
        if not a.startswith(ALLOWED_ROOT + "/"):
            report["global_errors"].append(f"outside allowed root: {a}")
        if ".." in a.split("/"):
            report["global_errors"].append(f"path contains ..: {a}")
        for b in paths[i + 1:]:
            if a == b:
                report["global_errors"].append(f"duplicate path: {a}")
            elif a.startswith(b + "/") or b.startswith(a + "/"):
                report["global_errors"].append(f"overlapping paths: {a} <> {b}")

    for entry in entries:
        path = entry["path"]
        row = {
            "path": path,
            "category": entry["category"],
            "expected": {"file_count": entry["file_count"],
                         "logical_file_bytes": entry["logical_file_bytes"],
                         "allocated_file_bytes": entry["allocated_file_bytes"],
                         "symlinks": len(entry["symlinks"]),
                         "retained": len(entry["retained"]),
                         "expected_records": entry.get("expected_records")},
            "status": "ok",
            "skips": [],
        }
        if entry.get("action") != "remove_directory":
            row["status"] = "skip"
            row["skips"].append(f"unexpected action {entry.get('action')}")
            report["entries"].append(row)
            continue
        real = os.path.realpath(path)
        row["realpath"] = real
        if real != path:
            row["status"] = "skip"
            row["skips"].append(f"realpath differs: {real}")
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
            row["skips"].append("missing or not a plain directory")
            report["entries"].append(row)
            continue
        nested_mounts = [m for m in mounts if m == path or m.startswith(path + "/")]
        row["nested_mounts"] = nested_mounts
        if nested_mounts:
            row["status"] = "skip"
            row["skips"].append(f"mount points at/inside entry: {nested_mounts}")
            report["entries"].append(row)
            continue
        st = os.stat(path)
        parent = os.stat(os.path.dirname(path))
        row["st_dev"] = st.st_dev
        row["parent_st_dev"] = parent.st_dev
        if parent.st_dev != st.st_dev:
            row["status"] = "skip"
            row["skips"].append("entry st_dev differs from parent")
            report["entries"].append(row)
            continue

        files, symlinks, hardlinks, problems = walk_entry(path)
        row["observed"] = files
        row["hardlinks"] = hardlinks
        row["problems"] = {
            "non_regular": [problems["non_regular"][:10], len(problems["non_regular"])],
            "symlink_targets_missing": [problems["symlink_targets_missing"][:10],
                                        len(problems["symlink_targets_missing"])],
        }

        if problems["non_regular"]:
            row["skips"].append(f"non-regular files: {problems['non_regular'][:5]}")
        if problems["symlink_targets_missing"]:
            row["skips"].append(
                f"symlink targets missing: {problems['symlink_targets_missing'][:5]}")
        if files["regular"] != entry["file_count"]:
            row["skips"].append(
                f"regular file count {files['regular']} != manifest {entry['file_count']}")
        if files["alloc"] != entry["allocated_file_bytes"]:
            row["skips"].append(
                f"allocated bytes {files['alloc']} != manifest {entry['allocated_file_bytes']}")
        if files["logical"] != entry["logical_file_bytes"]:
            row["skips"].append(
                f"logical bytes {files['logical']} != manifest {entry['logical_file_bytes']}")
        if files["symlinks"] != len(entry["symlinks"]):
            row["skips"].append(
                f"symlink count {files['symlinks']} != manifest {len(entry['symlinks'])}")

        expected_links = {s["path"]: s["target"] for s in entry["symlinks"]}
        missing_links = sorted(set(expected_links) - set(symlinks))
        extra_links = sorted(set(symlinks) - set(expected_links))
        changed_targets = sorted(
            p for p in expected_links
            if p in symlinks and symlinks[p] != expected_links[p]
        )
        escaped = sorted(
            p for p, t in symlinks.items()
            if not (t == path or t.startswith(path + "/"))
        )
        row["symlink_check"] = {
            "missing_vs_manifest": missing_links[:10],
            "missing_count": len(missing_links),
            "unexpected": extra_links[:10],
            "unexpected_count": len(extra_links),
            "target_changed": changed_targets[:10],
            "target_changed_count": len(changed_targets),
            "target_outside_entry": escaped[:10],
            "target_outside_entry_count": len(escaped),
        }
        for key, label in (
            ("missing_vs_manifest", "symlinks missing vs manifest"),
            ("unexpected", "unexpected symlinks"),
            ("target_changed", "symlink targets changed vs manifest"),
        ):
            if row["symlink_check"][key]:
                row["skips"].append(f"{label}: {row['symlink_check'][key][:5]}")
        if row["symlink_check"]["target_outside_entry"]:
            # Manifest-listed fixture symlinks (e.g. an escape-to-/tmp placeholder
            # from a temp-root rejection test) are deleted as links only; rm -rf
            # unlinks the link entry and never follows it. Recorded, not a skip.
            row["symlink_check"]["outside_note"] = (
                "manifest-listed; unlinked as a link entry, target not followed")

        if row["skips"]:
            row["status"] = "skip"
        report["entries"].append(row)

    ok = [r for r in report["entries"] if r["status"] == "ok"]
    report["summary"] = {
        "total_entries": len(entries),
        "verified_ok": len(ok),
        "skipped": len(report["entries"]) - len(ok),
        "verified_regular_files": sum(r["observed"]["regular"] for r in ok),
        "verified_symlinks": sum(r["observed"]["symlinks"] for r in ok),
        "verified_allocated_bytes": sum(r["observed"]["alloc"] for r in ok),
        "verified_logical_bytes": sum(r["observed"]["logical"] for r in ok),
        "hardlinks_files_with_nlink_gt1": sum(
            r["hardlinks"]["files_with_nlink_gt1"] for r in ok),
        "hardlinks_multi_member_groups": sum(
            r["hardlinks"]["multi_member_group_count"] for r in ok),
    }
    json.dump(report, open(sys.argv[1], "w"), indent=2)
    print(json.dumps(report["summary"], indent=2))
    if report["global_errors"]:
        print("GLOBAL ERRORS:")
        for err in report["global_errors"]:
            print(" ", err)
    for row in report["entries"]:
        if row["status"] != "ok":
            print("SKIP", row["path"])
            for reason in row["skips"]:
                print("   ", reason)


if __name__ == "__main__":
    main()
