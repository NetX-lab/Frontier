# Independent Spec review of the frozen V4 group metrics correction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC | Reviewed the frozen staged correction and inspected owner and completed primary receipts; recorded scope and validation limits. |

**No findings.** No missing or partial source requirement, unasked behavior, or incorrect implementation was identified within this allocation.

Reviewer: independent Spec axis. Fixed point and observed HEAD: `4ae81052e8b7a82593b68c6740dadb07e9e1624b`; no subsequent commits. Review command: `git diff --cached 4ae81052 -- tests/analysis/v4_trace_audit_group_metrics.py tests/unit/test_v4_trace_audit_group_metrics.py`.

Read disk AGENTS.md, current task requirements/plan/progress, the nested profile04 records, allocation/control inputs, and frozen source/receipts. Authority is [D-V4-P0-AUDIT-METRICS-GROUP-CORRECTION](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/requirements.md:280), requirements lines 282–285. Other review axes were not consulted.

Frozen source (G) and tests (U) match both staged blobs and disk:

| Artifact | SHA-256 | Gross lines / cap |
| --- | --- | --- |
| [G: v4_trace_audit_group_metrics.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_group_metrics.py:26) | `4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251` | 80 / 220 |
| [U: test_v4_trace_audit_group_metrics.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_v4_trace_audit_group_metrics.py:71) | `d9747f5b8bc4371a0ed60c309632a367bb7931ef8c6677d370ef88c10f713827` | 194 / 200 |

Static requirement coverage:

| Exact spec reference | Source and test evidence | Assessment |
| --- | --- | --- |
| requirements.md:283, “Preserve original group count/sum order,duplicates” | G:31–41; U:71–128 | `len(members)` and the unchanged duration sum use original members; sorting preserves every repeated membership. Float-order negative witness distinguishes the two sums. |
| requirements.md:283, “one global stable(ts,ts+dur)order” and “sort each existing group by integer positions” | G:28–29,36; U:99–122 | One stable tuple-key global sort produces numeric positions; group sort keys are integers. Different tie orders affect only identical endpoints, preserving the union arithmetic. |
| requirements.md:283, “unchanged per-group union recurrence” | G:35–40; [frozen audit.py:28](/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py:28); U:149–152 | Initial state, addition/max recurrence, and division by 1000 match the frozen metrics function exactly. |
| requirements.md:283, “Avoid per-event mutable membership/state lists and eager default-list allocations” | G:28–41 | The order map contains integers and each group uses scalar union/end state; no per-event lists or eager default-list creation remain. |
| requirements.md:283, “Reuse the accepted adapter/stage/pipeline composition with exact pins” and “do not change GC/loader policy” | G:15–22,45–76; U:130–182; [accepted adapter:16](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_metrics.py:16) | Accepted substitutions/group construction and pipeline/main are reused. AST replacement is confined to the metrics definition plus two sparse observations. Frozen stage, observer, loading-GC pipeline and reader hashes match. |
| requirements.md:283, “all fields and raw-ID values”; requirements.md:285 exclusions | G:45–76; frozen audit.py:62–119,120–190; U:130–182 | Existing grouping, attribution, stream overlap/reversal, raw index, and result composition stay outside the replaced function. Python object identities are internal sort lookups; raw IDs are untouched. |
| requirements.md:284, exact unit/control evidence before affected execution | U:66–80,99–181,184–190; receipts below | Units use the actual frozen metrics function and stage transformer. Saved AST strings bind to source, grouping, recurrence, main and pipeline; expected/actual snapshots are copied. |

Observed receipt evidence: owner and primary each report 9 tests, zero failures/errors/skips, and 42 consistent serialized checks. Independently reading those JSON values confirmed every saved expected/actual pair equals and all status fields are PASS. Primary's eight source pins match disk; actual memory limit is 32,212,254,720 bytes.

Primary controls report canonical non-clock JSON exact, 361 decode / 240 prefill metric groups exact (including float-hex checks), and clean complete decompressed indices of 142,567 / 570,964 rows exact. Retained output/index hashes and kernel totals match the frozen control inputs. These are inspected primary receipts, not reviewer executions.

Receipt paths below are relative to this report directory, except the two absolute primary execution receipts:

| Inspected artifact | SHA-256 |
| --- | --- |
| `primary/allocation.json` | `ed949333e768673045ce606e50548dbf7d683eb3f090889c8401ecdc63e79c3d` |
| `primary/control_inputs.json` | `750d7b26404c0bd7356f66ef530c0fc6323260bd95d25427fec7e62589f0e72e` |
| `owner/source_freeze.json` | `9677fd82be5c5049371af4a35db3b8a3572dc475908b2ffe9eceb0bf8d51abd2` |
| `owner/unit_result.json` | `54ff3b0421108f5ba20aa0c6b33fd97f9fc41d7ed0e23c9d35a241f5c8522886` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_controls_20261025_01/primary_unit.receipt.json` | `df6347a47bb3b191798e4e0d6c0df340d8364eaf01528e09807fc3b6a13d101f` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_controls_20261025_01/exact_controls.receipt.json` | `8a5cbbb4f022b631871703074fd61541816429fa3931d60028f9cdb07f9afa21` |

Limits: static text/AST and scoped JSON reads only. No reviewed module or driver was imported/executed, no unit/full-trace/index comparison was run, and no network/GPU/Git mutation or delegation occurred. Only this report was written. Raw traces and complete indices were not independently reread.

The 0.726559-second receipt elapsed value measures artifact comparison. P0 completion/speed, fresh pilot acceptance, R5/full32, native ownership/dependencies, and cost/reference admission remain unverified. Requirements.md:285 still makes the remaining seven affected ranks conditional on primary acceptance of a complete fresh pilot. The scope self-check found no drift; downstream execution and acceptance remain with primary.
