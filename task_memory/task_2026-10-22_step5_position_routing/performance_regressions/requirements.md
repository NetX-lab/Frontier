# D29 performance final regressions requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record fresh immutable-source regression assignment and strict ownership. |

## [Original Request]

TaskID D29-PERFORMANCE-FINAL-REGRESSIONS. Continue previous regression work on
immutable source79f1a3fd35d42b751cd1520c764ddac717b923d9, tree
8c6db078e852d3b885a6f14bdf5e8007d9ecddb7, worktree
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`, submodule
3dac53c544e546574cb6e8078cd14053d0916113. Source and submodule must stay clean.

Fresh required suites: unchanged strict77 run/compare against the original
baseline; full tests/unit with continue-on-collection-errors; same-source101
Python3.12 environment supplement; metadata196 and catalogue53 selections. Resolve
actual counts from new XML, not prior expected counts. Compare exact historical
267FAIL/31ERROR IDs and accepted e41 result8015PASS/279FAIL/32ERROR/281SKIP;
preserve raw exits, additions/removals, case loss and PASS-to-SKIP. All new cases
must pass. Add affected checks for the seven changed paths only if full-unit or
supplement does not settle them. Existing py312 CPU extras are explicitly allowed
for typed governance; no installation or ambient import-path substitution.

Use new strict output `/data/ycfeng/tmp/d29_workflow_fidelity_20261023_final05`
and unit `/data/ycfeng/tmp/d29_unit_parent05{.xml,.log,...}` with existing-path
checks. Matrix jobs4; unit timeout1800s; existing single-thread BLAS commands.

Only this new primary documentation directory and new scratch are writable.
Preserve all old artifacts; no frozen-source edits, further delegation, commits,
full CLI/GPU/network operations or dependency installation. Primary owns main08
and archives these records. Do not stop at background launch. If a gate fails,
retain precise IDs/evidence and diagnose before proposing a scoped fix; never
skip, weaken a gate or silently classify a new code failure as environmental.

User-supplied prerequisites: all independent source/record reviews closed No
findings before freeze; component parents81/1oldSKIP,2,29,52,128PASS separately;
integrated actual point passes with5calls/88descriptions/20reads/1decode and
unchanged fullpayload c2aa250d..., canonical d2c141f.../3888b075.... These are
prerequisite evidence, not fresh regression results or a main08 acceptance claim.
