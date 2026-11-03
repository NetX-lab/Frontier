#!/usr/bin/env python3
"""Hash the retained files of the replay cleanup manifest plus small neighbours.

Covers: every path listed in manifest.retained (deduplicated), small
config/summary/log/test-evidence files directly beside each entry, the
authoritative baseline manifest/results, and the active P2-64 package files.
"""
import hashlib
import json
import os
import stat
import sys

MANIFEST = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261103/"
    "replay_deletion_manifest.json"
)
ACTIVE_PACKAGE = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/p2_64_formal_retry04/"
    "package_01"
)
BASELINE = "/data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline"
PARENT_EXT = {".json", ".jsonl", ".csv", ".txt", ".log", ".sh", ".py", ".xml",
              ".out", ".md", ".yaml", ".yml"}
MAX_BYTES = 1 << 20


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(manifest):
    targets = set()

    def add(path):
        try:
            st = os.lstat(path)
        except OSError:
            return
        if stat.S_ISREG(st.st_mode) and st.st_size <= MAX_BYTES:
            targets.add(path)

    for entry in manifest["entries"]:
        for path in entry["retained"]:
            add(path)
        parent = os.path.dirname(entry["path"])
        try:
            names = os.listdir(parent)
        except OSError:
            names = []
        for name in sorted(names):
            if os.path.splitext(name)[1] in PARENT_EXT:
                add(os.path.join(parent, name))

    add(os.path.join(BASELINE, "manifest.json"))
    add(os.path.join(BASELINE, "results.jsonl"))
    for name in ("manifest.json", "config.json", "launcher_started.json"):
        add(os.path.join(ACTIVE_PACKAGE, name))
    return sorted(targets)


def main():
    manifest = json.load(open(MANIFEST))
    targets = collect(manifest)
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
