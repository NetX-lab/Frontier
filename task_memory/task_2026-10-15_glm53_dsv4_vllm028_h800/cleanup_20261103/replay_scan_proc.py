#!/usr/bin/env python3
"""Scan /proc for live references to any approved deletion directory.

Reads cwd, root, exe, fd/*, maps, and cmdline of every visible process and
reports any reference to an approved path (or a descendant). Processes we
cannot read are counted as permission-limited, not as in-use.
"""
import json
import os
import sys

MANIFEST = (
    "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/"
    "task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261103/"
    "cache_deletion_manifest.json"
)


def matches(target, path):
    if not path:
        return False
    return path == target or path.startswith(target + "/")


def main():
    targets = [e["path"] for e in json.load(open(MANIFEST))["entries"]]
    own_pid = os.getpid()
    hits = []
    unreadable = {"processes": [], "fd_dirs": 0, "maps": 0}
    scanned = 0

    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        if int(pid) == own_pid:
            continue
        base = f"/proc/{pid}"
        try:
            procs = os.listdir(base)
        except (PermissionError, FileNotFoundError):
            unreadable["processes"].append(int(pid))
            continue
        if "cmdline" not in procs:
            continue
        scanned += 1
        try:
            cmdline = open(f"{base}/cmdline", "rb").read().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            cmdline = ""
        for name in ("cwd", "root", "exe"):
            try:
                link = os.readlink(f"{base}/{name}")
            except (PermissionError, FileNotFoundError, OSError):
                continue
            for target in targets:
                if matches(target, link):
                    hits.append(
                        {"pid": int(pid), "kind": name, "path": link,
                         "target": target, "cmdline": cmdline[:400]}
                    )
        fd_dir = f"{base}/fd"
        try:
            fds = os.listdir(fd_dir)
        except (PermissionError, FileNotFoundError):
            unreadable["fd_dirs"] += 1
            fds = []
        for fd in fds:
            try:
                link = os.readlink(f"{fd_dir}/{fd}")
            except (PermissionError, FileNotFoundError, OSError):
                continue
            for target in targets:
                if matches(target, link):
                    hits.append(
                        {"pid": int(pid), "kind": f"fd/{fd}", "path": link,
                         "target": target, "cmdline": cmdline[:400]}
                    )
        try:
            maps = open(f"{base}/maps", "rb").read().decode("utf-8", "replace")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            unreadable["maps"] += 1
            maps = ""
        for line in maps.splitlines():
            path = line.rsplit(" ", 1)[-1].strip() if line.strip() else ""
            for target in targets:
                if matches(target, path):
                    hits.append(
                        {"pid": int(pid), "kind": "maps", "path": path,
                         "target": target, "cmdline": cmdline[:400]}
                    )
                    break
        for target in targets:
            if target in cmdline:
                hits.append(
                    {"pid": int(pid), "kind": "cmdline", "path": target,
                     "target": target, "cmdline": cmdline[:400]}
                )

    out = {
        "scanned_processes": scanned,
        "unreadable_process_count": len(unreadable["processes"]),
        "unreadable_fd_dirs": unreadable["fd_dirs"],
        "unreadable_maps": unreadable["maps"],
        "hits": hits,
    }
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "hits"}, indent=2))
    print("hits:", len(hits))
    for hit in hits[:50]:
        print(hit)


if __name__ == "__main__":
    main()
