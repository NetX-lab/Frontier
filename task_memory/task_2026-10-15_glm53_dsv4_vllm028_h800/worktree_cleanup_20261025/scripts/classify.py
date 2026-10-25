import json, os, subprocess, hashlib, collections
rows = json.load(open("/data/ycfeng/tmp/wt_cleanup/inventory.json"))
MAIN = "/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028"
KEEP_NAMES = {"Frontier", "glm53-dsv4-vllm028", "frontier-calibration-skill", "remove-legacy-run-outputs", "stage-admission-ordering"}
def subject(h):
    return subprocess.run(["git", "log", "-1", "--format=%s", h], capture_output=True, text=True).stdout.strip()
def is_step5(r):
    n = os.path.basename(r["path"])
    if n.startswith(("step5", "verify-step5", "d29-", "d30-", "d5-ttft")) or "stepcode-codex-home" in r["path"]:
        return True
    tm = os.path.join(r["path"], "task_memory")
    if os.path.isdir(tm) and any("step5" in d for d in os.listdir(tm)):
        return True
    s = subject(r["head"]).lower()
    return "step5" in s or "mrv2" in s or "step-5" in s or "step pd" in s
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
out = []
for r in rows:
    n = os.path.basename(r["path"])
    if n in KEEP_NAMES: cls = "keep"
    elif is_step5(r): cls = "other_task"
    else: cls = "this_task"
    r["name"], r["class"] = n, cls
    if cls == "this_task":
        ign = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"], cwd=r["path"], capture_output=True, text=True).stdout.split("\0")
        ign = [p for p in ign if p]
        top = collections.Counter()
        size = collections.Counter()
        tm_missing, tm_diff = [], []
        for p in ign:
            full = os.path.join(r["path"], p)
            t = p.split("/")[0]
            top[t] += 1
            try: size[t] += os.lstat(full).st_size
            except OSError: pass
            if p.startswith("task_memory/") and os.path.isfile(full) and not os.path.islink(full):
                m = os.path.join(MAIN, p)
                if not os.path.exists(m): tm_missing.append(p)
                elif os.path.isfile(m) and os.path.getsize(m) == os.path.getsize(full) and sha(m) == sha(full): pass
                else: tm_diff.append(p)
        r["ignored_top"] = {k: [top[k], size[k]] for k in top}
        r["tm_missing"], r["tm_diff"] = tm_missing, tm_diff
        r["has_submodule"] = os.path.exists(os.path.join(r["path"], "frontier/cc_backend/backends/collective-sim/.git"))
    out.append(r)
json.dump(out, open("/data/ycfeng/tmp/wt_cleanup/classified.json", "w"), indent=1)
c = collections.Counter(r["class"] for r in out); print(c)
