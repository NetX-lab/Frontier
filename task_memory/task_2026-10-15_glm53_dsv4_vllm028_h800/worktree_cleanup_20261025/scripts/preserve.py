"""Preserve every record of the in-scope worktrees before removal (D-WORKTREE-CLEANUP).

1. Copy task_memory files that are absent from the mainline worktree to the same relative path.
2. Archive task_memory files whose content differs from the mainline copy under
   worktree_cleanup_20261025/divergent_records/<worktree>/<path>.
3. Record uncommitted state and unreachable HEADs as refs/archive/worktrees/<name>.
Writes preserve_manifest.json; never deletes anything.
"""
import collections, hashlib, json, os, shutil, subprocess, tempfile

MAIN = "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028"
TASK = "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800"
RECORD = os.path.join(MAIN, TASK, "worktree_cleanup_20261025")
rows = [r for r in json.load(open("/data/ycfeng/tmp/wt_cleanup/classified.json")) if r["class"] == "this_task"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a, cwd=MAIN, env=None):
    return subprocess.run(["git", *a], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


# Fresh scan: path -> {sha: [worktrees]} for files missing from main; divergent list per worktree.
missing = collections.defaultdict(lambda: collections.defaultdict(list))
divergent = []
for r in rows:
    ign = git("ls-files", "--others", "--ignored", "--exclude-standard", "-z", cwd=r["path"]).split("\0")
    tracked_mod = git("diff", "--name-only", "-z", "HEAD", "--", "task_memory", cwd=r["path"]).split("\0")
    for p in sorted({p for p in ign + tracked_mod if p.startswith("task_memory/")}):
        full = os.path.join(r["path"], p)
        if not os.path.isfile(full) or os.path.islink(full):
            continue
        m = os.path.join(MAIN, p)
        if not os.path.lexists(m):
            missing[p][sha(full)].append(r["name"])
        elif os.path.islink(m) or not os.path.isfile(m) or sha(m) != sha(full):
            divergent.append((r["name"], p))

copied, conflicts = [], []
for p, variants in sorted(missing.items()):
    if len(variants) > 1:
        conflicts.append(p)
        for names in variants.values():
            divergent.append((names[0], p))
        continue
    (digest, names), = variants.items()
    src = os.path.join(next(r["path"] for r in rows if r["name"] == names[0]), p)
    dst = os.path.join(MAIN, p)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    assert sha(dst) == digest, dst
    copied.append({"path": p, "sha256": digest, "source_worktrees": names})

archived = []
for name, p in sorted(set(divergent)):
    src = os.path.join(next(r["path"] for r in rows if r["name"] == name), p)
    dst = os.path.join(RECORD, "divergent_records", name, p)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    archived.append({"worktree": name, "path": p, "sha256": sha(dst), "archive": os.path.relpath(dst, MAIN)})

refs = []
for r in rows:
    dirty = git("status", "--porcelain", cwd=r["path"])
    if not dirty and r["reachable"]:
        continue
    commit = r["head"]
    if dirty:
        with tempfile.TemporaryDirectory(dir="/data/ycfeng/tmp") as d:
            env = dict(os.environ, GIT_INDEX_FILE=os.path.join(d, "index"))
            git("read-tree", "HEAD", cwd=r["path"], env=env)
            git("add", "-A", cwd=r["path"], env=env)
            tree = git("write-tree", cwd=r["path"], env=env)
        commit = git("commit-tree", tree, "-p", r["head"], "-m",
                     f"Archive uncommitted state of worktree {r['name']} before removal (D-WORKTREE-CLEANUP)\n\n"
                     f"Worktree path: {r['path']}\nBranch: {r['branch'] or '(detached)'}\nHEAD: {r['head']}")
    ref = f"refs/archive/worktrees/{r['name']}"
    git("update-ref", ref, commit)
    refs.append({"worktree": r["name"], "ref": ref, "commit": commit, "head": r["head"],
                 "reason": "uncommitted state" if dirty else "unreachable HEAD", "dirty_lines": len(dirty.splitlines())})

json.dump({"copied": copied, "conflicts": conflicts, "divergent_archived": archived, "archive_refs": refs},
          open("/data/ycfeng/tmp/wt_cleanup/preserve_manifest.json", "w"), indent=1)
print(f"copied={len(copied)} bytes={sum(os.path.getsize(os.path.join(MAIN, c['path'])) for c in copied)} "
      f"conflicts={len(conflicts)} divergent_archived={len(archived)} refs={len(refs)}")
