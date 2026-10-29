# Interval metrics source notes

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Implemented the bounded interval representation correction; the sole owner unit passed with retained evidence. |

The sole production change replaces `batched_metrics` after loading the accepted group adapter pinned to `4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251`.
The adapter keeps one stable global event order, separate starts/ends columns, and an identity-to-position map. Each group retains the original count and duration sum expression and member order.
The original events list uses `range(len(ordered))`; other groups sort integer positions, including every repeated member. Repeated global identities retain every occurrence in the full-list path; identical endpoints at the mapped position preserve subgroup interval values.
Both frozen union recurrence statements remain exact AST matches. A copied composed AST must match the accepted instrumented metrics definition before replacement. The original sparse start/end observation statements are copied unchanged.
Accepted composition, substitutions, grouping, pipeline callable, loading/GC source and entry mechanism are retained. No accepted source is edited.

Tests load the actual new production adapter, execute its composed metrics definition in isolation, and extract only the original `metrics` and accepted `batched_metrics` function ASTs. Old outer diagnostic/control/test harnesses are never imported or executed.
Cases cover floathex equality, observable duration sum order, empty/mixed keys, ties, reversed groups, repeated memberships/global identities, input values/identities, exact recurrence/representation ASTs, unchanged AST outside metrics, sparse observations and rejected source mismatches.
The direct unittest entry copies all five cgroup-v1 sensor expressions from the pinned `observe_gc.py`, including mount-root handling. It writes exclusive invocation, cgroup, assertion and result evidence under `owner/unit_*`; assertion values are deep-copied.

Static source/test compilation passed without importing either module or running tests. The frozen execution allocation records exact source/test/dependency hashes, command, original Python and selected environment before the sole owner invocation.
Owner unit PASS: 9 tests, 90 retained assertions, exit 0; 1/1 spent. Unit evidence took 0.596984 seconds; enclosing command took 0.678841 seconds. Actual child memory limit: 32212254720 bytes (30 GiB). Source/test gross lines: 81/243 against 160/260; this document is below 80 lines.
Primary retains its separate unit, synthetic scale comparison, two complete controls, independent source reviews and final acceptance. No raw trace, control/performance run, GPU operation, main task-record edit or commit is part of this owner delivery.
Primary can import the new test module under a non-main name and use its own unittest runner and evidence location; the owner direct entry intentionally refuses to overwrite a spent owner's artifacts.
Performance benefit is a hypothesis supported by retained sort/endpoint measurements, not an observed speedup. Allocator corruption origin remains UNKNOWN; unit results cannot establish an allocator cure or R5 admission.

`unit_execution_allocation.json` retains the exact reviewed command, all 18 source/test/dependency hashes, Python 3.10.6 and the selected environment. `unit_receipt.json` binds stdout/stderr, cgroup and deep-copied assertions/results; all frozen files stayed unchanged. `unit_final_verification.json` records the retained-evidence checks.
