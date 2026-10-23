# D29 legacy output and fixture repair summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive two authorized slices, actual output evidence, tests and exact budget. |
| 2026-10-23 | Correct authority:four-path source delivered,two proposed paths await YC uncommitted. |

TaskID D29-LEGACY-OUTPUT-REPAIR. Worktree
`/data/ycfeng/Frontier/.worktrees/step5-d29-legacy-output`;branch
`feat/step5-d29-legacy-output`;immutable source
`9974acdb88f5c47166fee28dec564b7cc782ca1a`.

## Outcome

Original four-path repair:metadata emission now reuses existing MoE applicability
and registered profiling/source inventory,so legacy runs no longer gain Step-only
prepare. Actually inventoried prepare keeps original FP8 rather than default
BF16. Ledger omits absent collective_proofs while preserving all actual nonempty
plan/cost proofs. No timing,precision lookup,defaults,admission or comparator changed.

Uncommitted proposed fixture amendment:explicit global MoE catalogue assertions include
all five operators;local stubs expose operators/profiling_target and registry
interfaces;V4 expected inventory uses actual applicability and explicitly excludes
Step-only preparation. No assertions removed/skips added/production fallback.
Root corrected its premature authority before these files were staged/committed;
YC scope approval pending. Both edits are preserved isolated/uncommitted with no
further changes. The two paths and tests below are observations,not acceptance.

RED→GREEN:focused11 initially7 FAIL→11 PASS;six fixture contracts6 FAIL→6 PASS.
Final approved whole affected suites194 PASS;proposed test files53 PASS ran before
authority correction,0 SKIP/FAIL/ERROR. Actual Llama2 CSV cmp against original baseline
returns0(byte-identical). Full strict77/global unit review remains parent-owned.

## Deliverables

- requirements.md:raw requests,root amendments and superseded/latest accounting.
- plan.md:two independent slices,separate commit boundaries and parent gates.
- progress.md:observed commands/results and checkpoint.
- test_report_2026-10-23_legacy_output.md:exact environments,RED/GREEN,CSV compare,
  rawJUnit/errors,constructor prerequisite and scope.
- summary.md:this English completion archive and exact path ledger.

## Exact scope and budget

Approved source commit b60dfddd changes only the original four paths.
English task records archive b32ad29d;both immutable commits are ready for parent
integration. Proposed test files remain outside commits and awaiting approval.

| Path | Step added | Step deleted | Delta gross | Retained gross | Ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| frontier/config/quantization_manager.py |8|0|8|10|29|
| frontier/metrics/ep_wave_metrics.py |3|1|4|22|60|
| tests/unit/test_comm_operator_families.py |22|0|22|48|50|
| tests/unit/test_ep_wave_trace_context.py |3|0|3|49|65|
| tests/unit/test_moe_share_expert_operator_families.py (PENDING,UNCOMMITTED) |12|2|14|14|proposed50|
| tests/unit/test_dsv4_precision_metadata.py (PENDING,UNCOMMITTED) |2|0|2|2|proposed10|

Approved legacy four-path step+36/-1=37<=80,retained increment35.
Approved PRICING3805+35=3840/3956,remaining116;43 changed/44 approved owner paths,
approved effective109 global paths unchanged. Proposed two-path delta+14/-2=16
is excluded and uncommitted. Conditional total3856/3956,remaining100 applies only
if YC approves the expansion;no current implicit111-path authority.
The replaced ledger line was already in retained D29 diff,so37 churn is not
added blindly to owner3805(accurate approved3840,not3842).

Independent sub checkout3dac53c544e546574cb6e8078cd14053d0916113 is clean;
gitlink unchanged,no sub commit/build/network or other worktree modification.
Parent integrates b60dfddd preserving concurrent work;proposed edits await approval.

## Remaining acceptance

Parent strict77 rerun on original baseline,full-unit gate and independent review
not executed here. Existing dependency-bound aiohttp/torch suites remain parent
cross-env responsibilities. No GPU/native qualification or global D29 acceptance
is claimed. Approved owner source scope complete;proposed fixture admission and
commit are paused pending YC approval and are not delivered as completed scope.
