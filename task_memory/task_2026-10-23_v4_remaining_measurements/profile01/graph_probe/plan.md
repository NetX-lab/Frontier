# Callback payload correction plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Defined the serial CPU correction/verification path and root-only runtime handoff. |

Dependency: `A inspected frozen source/header/tests -> B safe CPU RED -> C fresh correction -> D CPU GREEN + inherited-suite reuse + source/ABI/matrix receipts -> E commit/clean handoff`.

| Step | Scope | Done condition |
| --- | --- | --- |
| A | Source and authority | Official descriptor indirection, union layout, frozen hashes and compatible loader inspected. |
| B | New unit file, at most160 lines | Actual old callback safely fails exact-handle assertions on allocated padded envelopes; no native CUPTI function. |
| C | Fresh probe copy, at most80 gross changed lines | Only envelope classes, descriptor decoding/null check and additive ABI reporting differ from frozen attempt3. |
| D | CPU verification and records | New controls pass; compatible original suite uses candidate; unchanged AST/query order/strict predicates are evidenced; source/worker/ABI/matrix receipts are explicit. |
| E | Commit and return | Requirements re-read; budgets and hashes final; verified artifacts committed; clean worktree; runtime limitations handed to root. |

Verification uses `/data/ycfeng/envs/oneshot-opd-py312/bin/python` (local3.12.13) and pytest9.1.1, with temporary logs under `/data/ycfeng/tmp`. Root's pinned image receipt reports actual worker3.12.3; local bytecode evidence is not a worker execution. Source/header facts and root receipts are the authority for static source/ABI binding.

Reuse the frozen `tests/run_cpu_checks.py:DiagnosticCopy` loader by setting its existing `HERE` global to this candidate directory. Run only compatible original suites, omitting their old naked-GraphData callback fixtures and exact old ABI equality assertion; the new controls cover the corrected envelope. Do not run the old runner's main block or write its outputs/packages.

No independent agent work is allowed in this slice. Root-owned reviews and runtime checks are non-dependent external work, not substeps executed here.
