import json, subprocess, os
FEAT = "feat/glm53-dsv4-vllm028"
FORK = subprocess.check_output(["git", "merge-base", "main", FEAT], text=True).strip()
def git(*a, cwd=None):
    r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()
out = subprocess.check_output(["git", "worktree", "list", "--porcelain"], text=True)
wts = []
cur = {}
for line in out.splitlines() + [""]:
    if not line:
        if cur: wts.append(cur); cur = {}
        continue
    k, _, v = line.partition(" ")
    cur[k] = v or True
rows = []
for w in wts:
    path = w["worktree"]; head = w.get("HEAD"); branch = w.get("branch", "").replace("refs/heads/", "")
    _, mb = git("merge-base", head, FEAT)
    on_feat_line = mb != FORK  # shares history with the feature branch beyond the fork point
    contained_feat = git("merge-base", "--is-ancestor", head, FEAT)[0] == 0
    contained_main = git("merge-base", "--is-ancestor", head, "main")[0] == 0
    _, refs = git("branch", "-a", "--contains", head, "--format=%(refname:short)")
    _, tags = git("tag", "--contains", head)
    reachable = bool(refs.strip() or tags.strip())
    _, status = git("status", "--porcelain", cwd=path)
    rows.append(dict(path=path, head=head, branch=branch, on_feat_line=on_feat_line, contained_feat=contained_feat,
                     contained_main=contained_main, reachable=reachable, dirty_lines=len(status.splitlines()) if status else 0,
                     dirty_sample=status.splitlines()[:5]))
json.dump(rows, open("/data/ycfeng/tmp/wt_cleanup/inventory.json", "w"), indent=1)
print(len(rows))
