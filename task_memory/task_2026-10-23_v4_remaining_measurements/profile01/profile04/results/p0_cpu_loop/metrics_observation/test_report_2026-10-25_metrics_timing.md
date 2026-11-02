# P0 final-metrics timing observation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Archived the completed bounded observation and exact decode control for handoff; no optimization selected. |

Allocation: `D-V4-P0-AUDIT-METRICS-OBSERVATION` in the task requirements. Measured timestamps remain the original 2026-10-24 UTC values; this report is the later archive date.

## Commands, environment and passing conditions

- Exact original command and envelope: [p0_metrics_01.receipt.json](p0_metrics_01.receipt.json). Local Python 3.10.6, `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s`, unchanged loading-GC pipeline, one allocated P0 rank0 observation only.
- Observer source: [observer_source.txt](observer_source.txt); original `/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/observe_metrics.py`. Frozen auditor `7ed0ea45...`, reader `fe4c3f82...`; original metrics body/order/results preserved through AST decoration.
- Decode control command/result: [metrics_control_compare.json](metrics_control_compare.json). Expected exact non-clock payload, decompressed kernel rows and useful steps; actual `PASS_METRICS_TIMING_OUTPUT_EQUIVALENCE`, 142,567 rows and 61 useful steps. PASS.
- P0 acceptance would require a complete final audit JSON within the existing deadline; partial rows are not sufficient.

## Observed results

| Observation | Actual | Interpretation |
| --- | --- | --- |
| P0 command | Exit 124 at the original 60s limit | FAIL/INCOMPLETE for full output |
| Actual memory limit | 32,212,254,720 bytes (30 GiB) | Applied scope; initial usage is not peak |
| Kernel loop | Completed at 52.988s; 1,215,919 kernels | Processing progress, not final acceptance |
| Per-stream stage | Completed at 57.999s | 155 streams |
| Useful-step stage | Completed at 58.030s | 512 recorded steps, 39 useful |
| Completed metrics calls | 155 calls, 1.451824s total | Measured small-set cost only |
| Last metrics call | Started at 59.258s with 1,215,919 kernels; no return | Censored duration; only 0.742s deadline headroom |
| Final audit JSON | Absent | R5 remains 24/32 complete |

Full timing records are in [metrics_timing_summary.json](metrics_timing_summary.json) and [p0_metrics_01.obs.jsonl](p0_metrics_01.obs.jsonl). GC, sampled stack, stderr and cgroup records are archived alongside them. The unchanged-work baseline and all previous failures remain immutable; no GPU operation occurred.

## Unimplemented seam proposal

Read-only explorer `ses_e9155c100ffeoePhOANZjUVYXY` checked the frozen `metrics()` and existing union recurrences. Best proposed route: one global stable sort of `(ts, ts+dur)` and per-key end trackers using the unchanged union recurrence; count/sum must retain original trace order. Filtering a stable global order preserves each key's local order, but exact output equality must still be tested.

The explorer's approximately 30-40 new lines and estimated 4.5s saving are projections, not implemented or measured results. Hierarchy multiplicity is unmeasured, and this route alone may still miss the deadline. Sorting each key then calling old metrics can change sum accumulation order and is not selected. No correction or additional P0/all-eight-rank invocation has been allocated or executed.

Next session must inspect actual code/evidence, record a bounded correction and exact-control allocation, then verify all count/sum/union/hierarchy/leaf/per-stream fields and original input hashes before any newly allocated full-input validation. Do not raise timeout/memory, skip events, remap ownership or retry GPU. R5, full32 trace, dependency and cost/reference admission remain open.
