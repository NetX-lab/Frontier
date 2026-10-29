# Scope/lifetime standalone source and GPU evidence

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Added the accepted single-image S5GPU diagnostic and preserved standalone/native/control limits. |
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
S1/S2CPU-source remains accepted independently. S3mirrored package/fresh gates/root seal and
unique S4jobexp-1025-021228-212042 are complete;original worktree/absolute paths and18inputs stay frozen.
S4Succeeded/Completed/exit0/restart0/launcher0,personal1H800/codesign,allocation spent1/1.
S5results/primary_acceptance.json is ACCEPTED_SINGLE_IMAGE_SCOPE_LIFETIME_WITH_LIMITS.
PrimaryR2raw60checks/5full-ID joins/6unmatched/0.122845s;20execution bindings;independent73scalar
checks/17negative rejections close threeR1omissions. New79closed-byte bindings/49immutable review
artifacts/18original-and-archive inputs PASS in0.062230s,without another audit/probe/control/GPU run.
Results/test_report_2026-10-25_scope_lifetime_gpu.md records exact commands/expected/actual/PASS/limits.
Only the sealed1024-element ordinary pointwise live-TLS route's frozenP1–P6are qualified.
Thread equality is observation;control.verdict=UNVERIFIED stays and supports topology/output identity only.
Six unmatched-created causes UNKNOWN;host driverrelease UNRECORDED/API13000+CUPTI130001telemetry.
Native serving/compiled-breakable ownership,dependencies,timing/non-perturbation,cost/reference remain open.
The parent P0metrics pilot still failed without finalJSON;R5/R6/R7continue independently.
