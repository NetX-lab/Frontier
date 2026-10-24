# Scope/lifetime CPU-source completion

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archived accepted standalone CPU/source implementation, not GPU/native ownership. |

The standalone probe extends frozen CUPTI resource primitives with live callback13
scope/thread/time copying, five real range windows and complete original-handle/ID-to-clone/replay joins.
Sources: probe_scope_lifetime.py252lines; tests/unit/test_cupti_scope_lifetime.py179lines.
Tuple alias/duplicate/query controls and closed raw-log bindings are corrected; original failures preserved.
Both Standards/Spec reviews closed with no new findings, including full13-file repairdiff inspection.
Owner and merged worktrees each pass primary50-case execution under60seconds/30GiB with CUDAfalse,
25fixtureexports and unchanged selected inputs;26retained raw bindings independently rehashed.
Deliverables: primary_acceptance.json, review.md, root_cpu_replay_owner_worktree.json,
root_cpu_replay_merged_worktree.json and root_raw_binding_closure.json in this directory.
S1/S2 accepted only. S3/S4 package/image/SDK/GPU TLS/lifetime and any native integration,
ownership, timing/non-perturbation or serving/cost/reference admission remain unfinished.
