"""Restore worktrees that tracked tests bind by absolute path, from the branch and the preserved records."""
import hashlib, json, os, shutil, subprocess, sys
MAIN = "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028"
names = sys.argv[1:]
rows = {r["name"]: r for r in json.load(open("/data/ycfeng/tmp/wt_cleanup/classified.json"))}
man = json.load(open("/data/ycfeng/tmp/wt_cleanup/preserve_manifest.json"))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
for n in names:
    r = rows[n]
    tip = subprocess.check_output(["git", "rev-parse", r["branch"]], text=True).strip()
    assert tip == r["head"], (n, tip, r["head"])
    subprocess.run(["git", "worktree", "add", r["path"], r["branch"]], check=True, capture_output=True)
    files = [(os.path.join(MAIN, c["path"]), c["path"], c["sha256"]) for c in man["copied"] if n in c["source_worktrees"]]
    files += [(os.path.join(MAIN, d["archive"]), d["path"], d["sha256"]) for d in man["divergent_archived"] if d["worktree"] == n]
    for src, rel, digest in files:
        dst = os.path.join(r["path"], rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        assert sha(dst) == digest, dst
    st = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=r["path"], text=True)
    print(n, "restored files", len(files), "status clean" if not st else "DIRTY:\n" + st[:500])
