# Group metrics owner delivery

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 +08:00 | Completed owner source review and delivery from retained evidence; no verification rerun. |

Authority: `requirements.md` D-V4-P0-AUDIT-METRICS-GROUP-CORRECTION, current `plan.md`/`progress.md`, and [primary allocation](../primary/allocation.json). Task filenames follow the recorded +08:00 task date.
Scope: only the two source/test paths below and `metrics_correction02/owner/`; source/test caps 220/200 gross lines, maintained owner Markdown cap 100 total.
Measured cause: [closed diagnosis](../../metrics_tail/primary/diagnostic_analysis.json) retained exit124, batch_start55.267s, 1,215,919 events/240 groups/5,331,778 memberships; construction open >=3.352355s, two gen2 scans 2.181248s; sort/union not reached.

Source and tests frozen: **2026-10-25T04:25:45.684380+08:00**, equivalent to **2026-10-24T20:25:45.684380+00:00**. [Freeze receipt](source_freeze.json) and current bytes match the executed unit source hashes.

| Repository-relative path | Gross lines / cap | SHA256 |
| --- | --- | --- |
| `tests/analysis/v4_trace_audit_group_metrics.py` | 80 / 220 | `4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251` |
| `tests/unit/test_v4_trace_audit_group_metrics.py` | 194 / 200 | `d9747f5b8bc4371a0ed60c309632a367bb7931ef8c6677d370ef88c10f713827` |

Exact previously executed commands, archived here only:
```text
timeout 60s /usr/bin/python3 -B task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/metrics_correction02/owner/source_check.py
timeout 60s /usr/bin/python3 -B task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/metrics_correction02/owner/run_unit.py
```
Cwd: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; executable `/usr/bin/python3`, Python3.10.6/GCC11.3.0, standard library, no conda; retained platform Linux5.10.25/glibc2.35.
Static compile: PASS, 0.005339080933481455s, source execution0. Owner unit: expected9/actual9 tests PASS, failures0/errors0/skips0, 42 evidence rows PASS; elapsed0.2728235418908298s, unittest log0.247s.
Owner unit allowance **used1/1**; owner complete pipeline/trace controls/affected/GPU invocations all0. Final closeout reads receipts and writes documentation only.

| Check | Expected | Retained actual | Result |
| --- | --- | --- | --- |
| Frozen metrics and input identity | Exact count/sum/union and unchanged inputs/group objects | Empty, overlapping, tied, repeated memberships/identities and heterogeneous keys all exact | PASS |
| Original float accumulation | Total sum `0x1.3a92a30553262p-11`; reverse sum `0x1.3a92a30553261p-11`; both union `0x1.3a92a30553262p-11` | Same hex; reordered-sum negative witness true | PASS |
| Stable order and duplicate members | Global `[1,2,0,3]`; groups `[[1,2,0,3],[1,2,2]]`; integer keys | Exact copied values; sort calls3 = global1 + groups2 | PASS |
| Frozen recurrence and telemetry | Original union update/end update AST equal; stripped function equals source | Equal; one `DEBUG-V4-METRICS-GROUP-02:start/end` pair, memberships from group lengths | PASS |
| Grouping and substitution boundary | Every AST statement outside metrics, original grouping expressions/calls unchanged | Equal accepted/corrected module closure | PASS |
| Pipeline and entry | Accepted `pipeline_tree`, original single composition boundary and entry main unchanged | Equal; loader/GC observer policies inherited | PASS |
| Pins and receipt serialization | Six original source pins; copied expected/actual remain equal after list mutation | Six pins exact; all42 serialized rows consistent | PASS |

Evidence: [static receipt](source_check_result.json), [unit receipt](unit_result.json), [raw log](unit.log), [full AST closure](ast_closure.json), [freeze](source_freeze.json); retained executors `source_check.py`/`run_unit.py`. Unit receipt SHA256 `54ff3b0421108f5ba20aa0c6b33fd97f9fc41d7ed0e23c9d35a241f5c8522886`.
Owner review: Hooke, source/composition phase; inspected the actual two sources, accepted120-line adapter, frozen auditor/stage/loading-GC/observer sources and retained unit/AST evidence. Completed with no in-scope findings: one global stable order, identity-to-integer positions, original-order sums, all duplicates and exact recurrence; no per-event mutable lists/default-list allocations or raw-ID/output edits.
Owner checkpoint: [check-progress](check_progress.md), branchA, D1-D7/S1-S6 CLEAR. Markdown budget: 43 report + 57 checkpoint = 100/100; source/test bytes stay frozen. [Delivery inventory](delivery_manifest.json) records exact paths/hashes/gross lines separately.
Pending primary: its new unit, two complete exact controls, independent Spec/Standards reviews, source/AST and command receipts, fresh P0 pilot and seven conditional ranks, then R5 acceptance. These are not owner completion claims.
Limits: tiny fixtures/AST establish only their recorded exactness; full P0 improvement remains unmeasured, R5 remains24/32 in current shared records. No ownership/dependency/timing/GPU/cost/reference admission; old timer/tiny/control receipts and all frozen originals preserved.
