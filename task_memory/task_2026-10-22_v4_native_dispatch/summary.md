# W128-V4-NATIVE-DISPATCH source/CPU handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Primary R1 affected18PASS and merged116PASS; independent targeted APPROVE closes the source review, while routing/freeze/package/native remain pending. |
| 2026-10-22 | Archived completed source/CPU slice and root-owned remaining acceptance. |
| 2026-10-22 | Added reviewed balanced-routing prerequisite and deferred CPU-overhead work. |
| 2026-10-22 | Recorded R1 P2 correction, targeted verification and updated cumulative bounds. |

## Outcome

Implemented actual classifier inputs/return, dispatch inputs/descriptor/CPU return,
and post-all-reduce/pre-padding4-by-DP vector observation in the existing collector.
Execution-local association feeds existing bounded aggregate counters. Native
results/exceptions are unchanged; observation errors and row loss remain soft,
explicitly UNVERIFIED. Replay-only failed forwards now mark an association error
even when an aggregate row already contains a prior successful call.

The versioned R3 successor executes pinned CPU classifier/DP functions using those
independent operands. It preserves the other seven predicates and all frozen R2
reader/origin/shadow guards. Historical native05 is not retrofitted or admitted.

## Maintained code/test file boundaries

| File | Physical added/deleted lines | Ownership |
| --- | --- | --- |
| `tests/comparison/calibration/native_full_structure.py` | +93/-0 | Existing collector and V2 execution wrapper only |
| `tests/unit/test_native_v2_serving_inventory.py` | +233/-1 | Existing CPU fixtures and new operand/fault controls |
| `tests/unit/test_native_reader_binding.py` | +14/-11 | Authority fixture paths and configurable successor coverage |
| `task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py` | +136/-0 | New maintained versioned checker logic |

**Physical total: four code/test files,476 added+12 deleted=488 changed lines.**
R1 alone is two code/test files +6/-2=8 changed lines. About five later builder pin
lines would yield493 lines/five files; root must count the actual final delta.
No full checker or builder copy is hidden in task artifacts. Task records are
separate documentation, copied from root and updated locally as requested.

R3 reads frozen R2 from the explicit primary authority path and verifies SHA256
`29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a` before execution.
It modifies three unique source regions: source pins, hooks/oracle creation, and
the local-only dispatch branch. `successor_r2_effective.diff` reports **+4/-9**
effective lines (1,377 lines total; SHA256
`e50bd5ca4c7c536322e790d10d3a669ab501e17d87adecb5d7f3153edbf98ab4`).
The136-line maintained R3 adds the oracle and replacement logic separately; the
small effective R2 delta does not conceal these physical additions. Root must bind
both files/identities during freeze; copying R2 would add1,382 physical lines and
needs renewed scope review under the approved counting rule.

## Deliverables and validation

- `design.md`: source/association rationale and file ownership.
- `requirements.md`, `plan.md`, `progress.md`: root decisions preserved locally.
- `test_report_2026-10-22_native_dispatch_cpu.md`: exact commands, failure causes,
 72 final observer/native-oracle controls PASS;43 reader controls PASS.
- `review.md`: check-progress self-check, not independent code review.
- `successor_r2_effective.diff`: complete effective frozen-R2 changes.
- `source_record.diff`: full maintained source and task-record diff before commit.
- `r1_minimal.diff`: targeted R1 code/test delta against delivered `598c3a5`.

R1 rejects source-impossible classifier inputs(num_reqs1,num_tokens2,max_query1)
even when descriptor/DP vector agree. Existing parameter control observed RED
(DID NOT RAISE), then GREEN:18 affected controls PASS, including non-graph AST
equality. No native function or legitimate None/zero encoding changed. Original
reviewer recheck remains pending; see appended test report for exact evidence.

`git diff --check` passes. Module sizes954/611/136 are below2,000 lines.
Source reads verified existing generation05 has67 pins, worker/utils already pinned;
gpu/dp_utils, gpu/states and NIXL pull_scheduler add three, for70 actual dependencies.
Existing builder was inspected but not copied/changed/executed. Root owns those
small generation06 pin/checker changes after final code acceptance.

## Pending and limits

Independent review, root affected/full package checks, freeze, builder updates,
image/source startup checks and the single authorized fresh32H800 run are pending.
Root must incorporate reviewed balanced-routing binding before freeze/package/launch;
this dispatch slice contains no routing implementation and does not treat
uniform_random as balanced. Native05 retains its historical old-routing identity.
No GPU/platform/auth/SSH/cloud/build/package action occurred in this slice.

CPU fixtures are not native-run qualification. Actual row cardinality, observer
overhead, device completion and full four-metric calibration remain UNVERIFIED.
CPU-overhead analysis, measurements and suites are deferred until workflow/GPU
alignment, including op gap, is complete; no such work was added to this slice.
The existing4096-row limit remains; the fresh run must report retained row totals,
dropped rows and full counter reconciliation. No auxiliary verdict may stop healthy
native execution. Unrelated GLM/runtime/workload/metric/gate/native source untouched.
