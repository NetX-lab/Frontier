# D22-TIMING plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define executable correction, regression and immutable-stage milestones. |

Dependency: A -> B -> C -> D. No nested delegation; stage sealing follows verified source commit.

| Step | Scope | Done condition |
| --- | --- | --- |
| A | Timing seam RED->GREEN | Independent outer-only normalized windows, injected internal event overhead retained only in diagnostics, matched counts/minimums and separate executions/events. |
| B | Producer/judge/resource wiring | Independent layer reference exported truthfully; operator/GPU-active rules unchanged; no quality retry; preemptible stage materializes correct resources in CPU test. |
| C | Affected regression | All original 851 baseline IDs still PASS; focused faults reject mismatched/short references and preserve strict 5% checks. |
| D | Sealed full D18 probe and handoff | Fresh stage; all hashes/inventory/dry-runs/shell/workload/source identity verified; original stage02 unchanged; exact parent launch/judge commands, budget and limitations recorded. |

Verification uses `/data/ycfeng/envs/oneshot-opd-py312/bin/python`, `VLLM_TARGET_DEVICE=cpu`, existing offline dependencies, no native/GPU/platform operation. Raw logs live under `/data/ycfeng/tmp/d22_timing_20261021`; compact evidence lives here. Historical FAIL receipts are not recalculated or relabelled.
