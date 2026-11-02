# V4 native dispatch evidence progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Primary accepted R1 after independent APPROVE and18 affected checks; merged-state116 correctness checks passed before commit. |
| 2026-10-22 | Recorded approval and began implementation assignment. |
| 2026-10-22 | W128 completed host-operand source/CPU slice; review/freeze/build remain root-owned. |
| 2026-10-22 | Recorded root balanced-routing prerequisite and CPU-overhead deferral after completed dispatch slice. |
| 2026-10-22 | R1 fixed reviewer P2 classifier query bound with observed RED/GREEN evidence. |

- completed: native05 RCA and bounded proposal; explicit user approval recorded in requirements.
- completed: W128 source/ownership inspection, collector and successor implementation, CPU controls.
- completed: completion records and verified source slice ready for commit/root handoff.
- completed: independent R1 APPROVE and primary source acceptance;18 affected checks and116 merged checks passed. Root inspected the actual collector and minimal R1 diff.
- pending: balanced binding, final successor freeze, fresh package/image/pins, root submission and actual native acceptance.
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

## Follow-up steering after source/CPU handoff

- completed: source/CPU slice committed as `64db6ad31d3f9170dc47c29e7baf6edcd0f45811`;
  72 correctness controls and43 reader controls PASS. No code changes after this
  steering and no further CPU suite or overhead work added.
- pending/root-owned: reviewed balanced-routing module/hook/source pins before
  freeze/package/launch. This observer slice does not implement routing or use
  uniform_random as a substitute for balanced. Native05 remains old-routing
  historical UNVERIFIED evidence.
- deferred: CPU-overhead analysis/measurement/suites until workflow/GPU alignment,
  including op gap, is complete. Actual overhead remains unknown, not a requested
  current-phase measurement or acceptance blocker for this correctness slice.

## W128-V4-NATIVE-DISPATCH-R1

- completed: reproduced old R3 acceptance of impossible classifier inputs
  (num_reqs1,num_tokens2,max_query_len1,has_prefillTrue) with matching PIECEWISE2
  descriptor and DP vector. Existing `query_bound` parameter failed RED with
  `DID NOT RAISE` (exit1,1.87s). No fixture/framework introduced.
- completed: added one inequality to R3; source gpu/model_runner1055–1058 derives
  len/sum/max from the same scheduled map, so sum cannot exceed len*max. Only
  invalid real operands are rejected; native functions and None/zero DP encoding
  are unchanged. No clamping or native behavior change.
- completed: affected controls18 PASS/27 deselected,2.92s, including13 negative
  operand cases, four positive source-backed dispatch cases and non-graph AST
  equality. Exact commands and retained raw log/XML paths are in the test report.
- bounds: cumulative four code/test files +476/-12=488 changed lines. R1 alone
  changes two files +6/-2=8 lines. About five later builder lines would yield493
  lines/five files; final builder delta remains root-owned and must be counted.
- pending: original reviewer recheck. Root's separately reported115 PASS does not
  close the former query-bound gap. No further tests, routing, freeze/build,
  CPU-overhead, native or platform operation performed.

## Primary R1 acceptance

- Independent targeted reviewer returned APPROVE, originalP2 closed; review remained static/receipt-based.
- Primary reran affected controls:18PASS/27deselected,2.91s; merged actual source:116PASS/0failures/errors/skips,49.16s. Logs/XML: `/data/ycfeng/tmp/glm53_dsv4/root_native_dispatch_r1_20261022.*` and `root_native_dispatch_merged_20261022.*`.
- Pre-merge root requirements/plan/progress were preserved at `/data/ycfeng/tmp/glm53_dsv4/root_dispatch_records_before_merge_20261022/`; incoming records retain the approved budget, balanced prerequisite and deferred CPU-overhead order. Main requirements remain the raw decision authority.
- This is source/CPU acceptance. No freeze, package or GPU submission follows until the separate balanced execution identity is resolved and bound.
