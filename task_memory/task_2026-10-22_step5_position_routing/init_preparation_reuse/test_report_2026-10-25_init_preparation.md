## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 HKT | Close the recovered init diagnosis with source and retained receipts only. |

# Source and records verification

Environment: local2026-10-25 HKT / UTC2026-10-24, `zsh`, Python3.10.6 and `python3 -B`. No Frontier import. A first metadata command with `python -B` exited127 (alias absent), then succeeded with `python3 -B`. A production-HEAD guard later found concurrent primary advancement2e8->2ca and stopped before any write; explicit production2e8 blob checks and a separate current document snapshot resolved that metadata assumption.

Inspection used targeted `rg -n`, numbered `Path.read_text()` excerpts at [summary.md](summary.md)'s locations, and SHA256 of listed source/task/retained-receipt files. No referenced CSV/model/trace was opened. Recovery receipt/census:33 completed tools,1 rejection,0 controls. Thirteen actual candidate/baseline/primary file digests and thirteen production2e8 committed blobs match; the explicit13-path886-to146 committed diff is empty.

Exact final metadata verification command, run from the owned worktree:

```sh
python3 -B - <<'VERIFY'
from pathlib import Path
from hashlib import sha256
import json, subprocess
root = Path('/data/ycfeng/Frontier/.worktrees/d29-init-preparation-finish')
owned = root / 'task_memory/task_2026-10-22_step5_position_routing/init_preparation_reuse'
e = json.loads((owned / 'evidence.json').read_text())
for item in e['code_bindings']:
    for location in e['code_roots'].values():
        assert sha256((Path(location) / item['path']).read_bytes()).hexdigest() == item['sha256']
    blob = subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), 'show',
        e['accepted_production_code_commit'] + ':' + item['path']])
    assert sha256(blob).hexdigest() == item['sha256']
for item in e['authority_and_retained_evidence']:
    if 'git_commit' in item:
        blob = subprocess.check_output(['git', '--no-optional-locks', '-C', item['git_root'], 'show',
            item['git_commit'] + ':' + item['git_path']])
    else:
        blob = Path(item['path']).read_bytes()
    assert sha256(blob).hexdigest() == item['sha256']
assert e['consumed_controls']['new_total'] == e['consumed_controls']['recovered_total'] == 0
assert e['inferred_contribution']['modeled_removed_work_s'] == 126.58
assert e['inferred_contribution']['accepted_loader_savings_recounted_s'] == 0
assert e['verdict'] == 'ALLOC'
print('PASS:13 code bindings;14 retained bindings;zero controls;126.58s arithmetic')
VERIFY
```

Git ownership/whitespace commands:

```sh
git --no-optional-locks -C /data/ycfeng/Frontier/.worktrees/d29-init-preparation-finish diff --check
git --no-optional-locks -C /data/ycfeng/Frontier/.worktrees/d29-init-preparation-finish diff --cached --check
git --no-optional-locks -C /data/ycfeng/Frontier/.worktrees/d29-init-preparation-finish diff --cached --name-only
git --no-optional-locks -C /data/ycfeng/Frontier/.worktrees/d29-init-preparation-finish status --porcelain --untracked-files=normal
```

Require the five Markdown Modification History tables, exactly six owned paths, no non-owned changes, parent886 in the resulting commit, unchanged bound code and clean final records status. The owning agent returns the commit SHA rather than embedding it self-referentially.

Result: PASS for13 source bindings,14 retained bindings, zero-control/arithmetic checks, five Markdown history tables and six-path ownership. The staged whitespace check passed; publication is confined to those paths. The final commit/parent/status receipt is returned by the owning agent. No pytest/runtime verification ran. Current startup saving and peak-memory impact are unmeasured; future implementation equivalence remains a separate gate.
