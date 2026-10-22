# D29 parent design-budget and command-syntax verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Independently check the final three-seam core revision:33 paths/+2625/-341=2966. |
| 2026-10-22 | Compile the proposed post-build receipt command and nested payload without executing either. |
| 2026-10-22 | Preserve the failed authored-total check, corrected arithmetic, and source-inspection limits. |

## Scope and environment

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python` (project Python3.10.6).
This checks a proposed design table, not executable pricing or model fidelity.
It records the received, pre-reconciliation table. The author is subsequently
revising that design; the final table must be verified again and is not expected
to retain this historical total.
No pytest, GPU, profiling or platform operation was used for this verification.

## Exact command

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -c 'import json,re; from pathlib import Path; p=Path("task_memory/task_2026-10-22_step5_position_routing/pricing_design.md"); text=p.read_text().split("## 6.",1)[1].split("## 7.",1)[0]; rows=re.findall(r"^\| `([^`]+)`.*?\| (\d+)\s*/\s*(\d+) \|",text,re.M); paths=[r[0] for r in rows]; added=sum(int(r[1]) for r in rows); deleted=sum(int(r[2]) for r in rows); result={"rows":len(rows),"unique_paths":len(set(paths)),"added":added,"deleted":deleted,"gross":added+deleted,"new_paths":[p for p in paths if not Path(p).exists()]}; print(json.dumps(result,indent=2)); assert (len(rows),len(set(paths)),added,deleted)==(37,37,1802,185)'
```

The first invocation was exactly this command with the final expected tuple
`(37,37,1902,185)` instead of `(37,37,1802,185)`. It exited1 with
`AssertionError`; its printed result already showed1802 additions/185 deletions.
The document's Section D sum had an extra100 additions in its summary. No
individual file allowance needed changing. After correcting the document's
summary, the command above exited0.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Authored arithmetic | 37 paths, +1902/-185, 2087 gross | 37 unique paths, +1802/-185, 1987 gross | FAIL; documentation corrected |
| Corrected arithmetic | 37 unique paths, +1802/-185, 1987 gross | Same | PASS |
| New file declarations | Only the four declared new paths absent | Policy module, collective-plan module, pricing test, backend test | PASS |

Absent paths: `frontier/moe_routing_policy.py`,
`frontier/cc_backend/collective_plan.py`,
`tests/unit/test_step5_moe_pricing.py`,
`tests/unit/test_collective_sim_cc_backend.py`.

## Source observations and limits

Parent read locked native `pynccl.py:324-357,387-422` and
`cuda_communicator.py:847-915`, confirming unequal rooted NCCL calls and tensor
grouping. Parent also read existing `collective_sim` schema, traffic-matrix
generation and prediction composition. Detailed findings and source locations
are maintained in `pricing_design.md` sections1/3.2 and `review.md`.

The existing formal communication decision remains `collective_sim`. The
proposed measured-plan lookup requires an explicit modeling decision. A final
cross-owner scope union, a faithful simulation-plan extension, actual grouped
GPU duration/kernel counts and implemented pricing remain unverified.

## Post-build emitter syntax

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B -c 'import ast; from pathlib import Path; text=Path("task_memory/task_2026-10-22_step5_position_routing/profile_source_design.md").read_text(); body=text.split("<<\x27PY\x27\n",1)[1].split("\nPY\n```",1)[0]; tree=ast.parse(body); compile(tree,"receipt_emitter_host","exec"); node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="container_code" for t in n.targets)); inner=ast.literal_eval(node.value); compile(inner,"receipt_emitter_container","exec"); print({"host_lines":len(body.splitlines()),"container_lines":len(inner.splitlines()),"syntax":"PASS","executed_emitter":False})'
```

Expected: both code bodies compile; no execution of either body. Actual:
`host_lines=51`, nested `container_lines=29`, `syntax=PASS`,
`executed_emitter=False`, exit0. The51 host lines already include the29 payload
lines, so they are not additive. Docker/image/runtime provenance is not tested
by this syntax check.

## Core three-seam budget revision — PASS

Parent parses only the path tables in `design.md` section9, not the prose
subtotals, using Python3.10.6. Each existing path must exist; `(new)` paths are
explicitly declared. All33 paths are unique. Added ceilings sum to2625 and
deleted ceilings to341, giving2966 gross, matching the declaration. The receipt
with every parsed row is `/data/ycfeng/tmp/d29_core_budget_2966_parent01.json`.
This verifies planned arithmetic only; actual diffs, cross-owner union and
implementation/native behavior remain unverified.
