import json, os, subprocess
rows = [r for r in json.load(open("/data/ycfeng/tmp/wt_cleanup/classified.json")) if r["class"] == "this_task"]
refs = {a["worktree"] for a in json.load(open("/data/ycfeng/tmp/wt_cleanup/preserve_manifest.json"))["archive_refs"]}
out = []
for r in rows:
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=r["path"], capture_output=True, text=True).stdout.strip()
    if dirty and r["name"] not in refs:
        out.append({**{k: r[k] for k in ("name", "path", "head", "branch")}, "result": "SKIPPED: dirty without archive ref"})
        continue
    cmd = ["git", "worktree", "remove"] + (["--force"] if dirty else []) + [r["path"]]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out.append({**{k: r[k] for k in ("name", "path", "head", "branch")}, "forced": bool(dirty),
                "result": "removed" if p.returncode == 0 and not os.path.exists(r["path"]) else f"FAILED: {p.stderr.strip()}"})
json.dump(out, open("/data/ycfeng/tmp/wt_cleanup/remove_manifest.json", "w"), indent=1)
from collections import Counter
print(Counter(o["result"].split(":")[0] for o in out))
for o in out:
    if o["result"] != "removed": print(o["name"], o["result"])
