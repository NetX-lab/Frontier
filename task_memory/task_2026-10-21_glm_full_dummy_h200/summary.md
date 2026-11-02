# W128-FULL completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Archived the source/CPU-qualified CUDA-event producer and explicit remaining Q11 blockers. |

## Outcome

Delivered an isolated native whole-model FULL dummy **CUDA-event** producer,
with parameterized capture widths, actual-query-one/zero-context metadata,
invalid KV slots, native padded request geometry, and both GLM index variants.
Persistent external CUDA event nodes attribute scope wall time on replay without
requiring Python hooks to execute. The single reduced-model GPU substitutes only
the native DP maximum-width descriptor decision; it does not measure collectives.

This is a partial Q11 delivery, not an end-to-end coverage closure: no GPU
measurement, KERNEL_ONLY attribution, consumer admission, TTFT/TPOT result or
formal H200 acceptance is claimed.

## Deliverables

- `frontier/profiling/attention/dsa_full_dummy.py`: new CPU-plan / native-GPU-run
  entry point; isolated `dsa_full_dummy.v1` JSON output with runtime identity.
- `tests/unit/test_dsa_full_dummy_profiling.py`: CPU mocks, failure-path checks,
  and optional pinned-native-git-object contract check.
- `task_memory/task_2026-10-21_glm_full_dummy_h200/notes.md`: exact schema,
  coverage mapping, native source references, launch requirements and bounded
  kernel-only/consumer follow-up.
- `task_memory/task_2026-10-21_glm_full_dummy_h200/test_report_2026-10-21_glm_full_dummy.md`
  and `focused.xml`: exact checks and observed results.
- Requirements, plan and progress remain local to this worktree. Parent task
  records and existing source files are unchanged.

## Observed verification

150 focused tests PASS, zero failures/errors/skips; Python 3.10.6 on the CPU
host. The native contract test reads commit
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. Offline plan succeeds with widths
2/4/8 and explicitly false GPU/kernel-only/admission flags. No simulator changes
mean no numerical matrix repeat was required in this slice.

## Open work

1. Primary freezes the actual native sizes/limits and qualifies whole-model
   external-event replay on one H200 with the approved locked runtime.
2. A profiling follow-up must qualify graph-node-to-scope kernel attribution;
   event intervals must never be relabeled as summed kernel durations.
3. Primary reviews the measured schema/coverage and a narrow explicit consumer
   admission change. Current `DSACoverageError` remains unchanged until then.

No GPU/StepMind operation, publish, push, merge, or destructive action occurred.
