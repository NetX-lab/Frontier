#!/usr/bin/env python3
"""Hash a bounded list of small retained files next to the approved cache dirs.

The list is derived from the manifest entry parents only (never the cache dirs
themselves) and covers small config/JSON/CSV/command/summary artifacts plus the
active P2-64 package manifest. Used before and after deletion to prove the
retained artifacts did not change.
"""
import hashlib
import json
import os
import stat
import sys

MANIFEST = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261103/"
    "cache_deletion_manifest.json"
)
ACTIVE_PACKAGE = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/p2_64_formal_retry04/"
    "package_01"
)
ALLOWED_PARENT_EXT = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".sh"}
ALLOWED_CONFIG_EXT = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".sh", ".toml"}
MAX_BYTES = 1 << 20
MAX_FILES = 600


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect():
    entries = json.load(open(MANIFEST))["entries"]
    parents = sorted({os.path.dirname(e["path"]) for e in entries})
    cache_dirs = {e["path"] for e in entries}
    targets = []

    def add(path):
        if path in cache_dirs:
            return
        try:
            st = os.lstat(path)
        except OSError:
            return
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BYTES:
            return
        targets.append(path)

    for parent in parents:
        for name in sorted(os.listdir(parent)):
            full = os.path.join(parent, name)
            if os.path.splitext(name)[1] in ALLOWED_PARENT_EXT:
                add(full)
        config_dir = os.path.join(parent, "config")
        if os.path.isdir(config_dir):
            for root, _dirs, names in os.walk(config_dir):
                for name in sorted(names):
                    if os.path.splitext(name)[1] in ALLOWED_CONFIG_EXT:
                        add(os.path.join(root, name))

    for name in ("manifest.json", "config.json", "launcher_started.json"):
        add(os.path.join(ACTIVE_PACKAGE, name))

    return sorted(set(targets))[:MAX_FILES]


def main():
    targets = collect()
    rows = []
    for path in targets:
        try:
            st = os.stat(path)
            rows.append({"path": path, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                         "sha256": sha256(path)})
        except OSError as exc:
            rows.append({"path": path, "error": str(exc)})
    out = {"count": len(rows), "files": rows}
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print("hashed", len(rows), "files")


if __name__ == "__main__":
    main()
