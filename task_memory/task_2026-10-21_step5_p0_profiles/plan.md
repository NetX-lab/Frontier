# D21-P0-PROFILES plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define producer-first implementation,existing consumer compatibility and bounded P0 handoff. |

Dependency:A -> B -> C -> D. No delegation/platform operations.

| Step | Deliverable | Acceptance |
| --- | --- | --- |
| A | Native no-metadata workload/callable | Meaningful RED->GREEN at real metadata/forward seam;0 metadata identity,no builder/slot mapping/history,no cache mutation,positive source scopes. Existing ordinary path unchanged. |
| B | Consumer-shaped observations | Explicit serving physical table/capacity inputs; actual per-replay root counts,not aggregate/statistical counts; independent P/D source rows; full schema loader validates actual producer-shaped rows without admission changes. |
| C | Bounded P0 package | Existing source packager and launcher,one isolated H800/codesign/non-preemptible per package; independent roles,exact prepared grid/counts and all remaining gaps. No old stage overwritten. |
| D | Regression/seal/handoff | Focused + affected baseline IDs and>=50 existing shape scenarios; committed source,hashed inventory,actual CPU dry-runs; parent-only commands,budget,limits and final task records. |

Use existing Python3.12 profiling environment/offline dependency site and Python3.10 consumer environment where needed. Raw logs/cache under `/data/ycfeng/tmp/step5_p0_profiles_20261021`; compact reports/evidence here. Full native admission remains parent-owned/pending.
