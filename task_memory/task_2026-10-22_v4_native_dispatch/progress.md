# V4 native dispatch evidence progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded approval and began implementation assignment. |
| 2026-10-22 | W128 completed host-operand source/CPU slice; review/freeze/build remain root-owned. |

- completed: native05 RCA and bounded proposal; explicit user approval recorded in requirements.
- completed: W128 source/ownership inspection, collector and successor implementation, CPU controls.
- completed: completion records and verified source slice ready for commit/root handoff.
- pending: root/reviewer acceptance; no further code writes before review feedback.
- pending: independent review, primary checks, final successor freeze, fresh package/image/pins, root submission and actual native acceptance.
- No new GPU job submitted. Native05 remains UNVERIFIED. Approved resource cap is32 non-preemptible H800/codesign,60minutes/32GPUh.

## W128-V4-NATIVE-DISPATCH

- Actual native seams: imported classifier and dispatch in gpu/model_runner;
  module-local dp_utils.dist proxy, CPU snapshot after actual all-reduce and before
  row0 padding. Existing V2 execution thread-local scope and aggregate counters
  own the association. New observations never decide native execution.
- Reason: local q1 shape cannot identify prefill and local counts omit DP peers.
  Expected: distinct independent classifier/DP evidence with unchanged native
  return/exception behavior. Result: pinned native CPU bodies reproduce q1
  PIECEWISE/FULL, peer-driven PIECEWISE, unequal FULL padding; raw observer evidence
  feeds the successor successfully.
- CPU commands/results: see `test_report_2026-10-22_native_dispatch_cpu.md`.
  Final affected controls 72 PASS; reader binding/origin/shadow suite 43 PASS.
  AST equality: only native_rank_graph_evidence differs from R2; seven other
  predicates and reader/shadow guards unchanged.
- Iterations: initial bare `python` unavailable (explicit py312 used); initial
  tensor carrier failed because existing fixture replaces sys.modules['torch']
  (numpy CPU carrier adopted; real torch used only by native-body oracle controls).
  Reader fixtures lacked local frozen versions (tests now reference authority
  paths); generation05 expected earlier rank-file failure now correctly fails on
  missing dp_utils source pin. No gate relaxed; frozen files unchanged.
- Bounds: no duplicate checker copy; hash-verified frozen R2 is an external
  dependency. Four maintained code/test files, below500 lines; report exact
  physical and effective-source totals in summary. Existing builder not copied
  or changed, avoiding premature generation06/package writes.
- Remaining: independent review, root affected/full package checks, authority
  dependency binding, final writer freeze, actual source/image pins and fresh run.
  Cardinality, host overhead and device completion remain UNVERIFIED.
