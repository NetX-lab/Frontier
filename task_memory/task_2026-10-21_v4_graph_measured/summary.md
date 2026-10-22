# W128-V4-GRAPH-MEASURED (A)

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Archived CPU implementation621b198, measured scope evidence,686 affected checks and strict77 IDENTICAL; primary acceptance pending. |
| 2026-10-21 | Applied H800-priority steering without runtime edits:688 tests,102 scopes/14 negatives; H200 data cannot qualify H800. |
| 2026-10-21 | Implemented independent-review corrections with CPU red/green,208 affected passes and102 scope/68 typed-negative receipts; review strict pending. |
| 2026-10-21 | Completed correction282cb11 sole clean strict77 IDENTICAL77/77; only task records changed after measurement. |
| 2026-10-21 | Reopened final P2:only known Unknown typed operator error conversion;two real-reader red/green cases and9/102/72 checks PASS, final strict pending. |
| 2026-10-21 | Final P2e3ce166 clean77 IDENTICAL77/77; all requested repairs complete with exact prior receipts retained. |

## Overview

Branch/worktree `v4-graph-measured` starts at published
`32aef6fe9e4ea31b8bdec50cbbc53401b6339cc1`. This slice replaces the active-batch
blanket graph rejection with native-descriptor compute geometry, and adds exact
isolated measured consumption through the existing wide schema and typed
linear/MoE readers. It does not admit a complete native V4 serving forward.

## Deliverables

- `frontier/attention/deepseek_v4.py`: typed coverage errors, native Batch compute
  width, separate actual request/context/phase snapshots, FULL capture-specific
  CSA/HCA branches, and explicit unqualified PIECEWISE branch errors.
- `frontier/execution_time_predictor/deepseek_v4.py`: graph-aware SP source/local
  rows, actual-state serving context, isolated scope receipts with exact geometry,
  no extrapolation, composite ownership, native backend and runtime identity.
- Existing predictor/shared dataframe dispatch delegates to the model module.
  The optional MoE round-trip parser preserves exact serialized features only for
  this consumer; legacy callers keep their old default. GLM algorithms, metrics,
  CC payload construction, B and H5 files are unchanged.
- `tests/unit/test_dsv4_graph_measured.py`: graph/source/CC geometry and measured
  positive mathematics/typed-negative regression checks. Three existing tests
  retain their node IDs while expecting the newly specific typed errors.
- `tests/integration/check_dsv4_measured_scopes.py`: reproducible CPU checks on
  the selected qualified local attention/shared/MARLIN tables.
- Task-local `real_scope_receipts.json`, report, progress and self-check.

## Observed validation

Affected tests:688 PASS/8 SKIP. Real rows:102 exact positive isolated scope
comparisons and14 typed negatives PASS. Both measurement methods retain their
identities and remain separate. Source9f58dbe compute and source5e051e4 attention
are not relabelled or rewritten. The MARLIN matching lane is synthetic, not
native routing evidence. Full sweep:5581 PASS/108 FAIL/72 SKIP/20 ERROR;
the only historical baseline-passing regression was the new CPU-helper allowlist
and passed on the686-test rerun. Zero unresolved regressions among historical
passing IDs present in the sweep; unrelated missing-dependency failures and64
absent historical IDs remain explicitly recorded. The sole clean strict77 on
`621b198476c677ae32464387149402c1c4952c9c` passed:77 executed/77 compared/77 identical,
zero failures/missing evidence/definition or predictor-cache filename differences.
Manifest `source_dirty=false`, clean cache and matching source/driver/harness
revision. `strict77_receipt.json` records original paths. The detached measured
checkout is retained clean. Subsequent changes are task records and H800 CPU-only
regression checks; runtime and strict case definitions remain identical.

Review diff: `/data/ycfeng/tmp/v4_graph_measured_final.diff`.

H800 TP8/DP2/EP16/P16D16 source geometry passes the existing interfaces without
H200 capacity constants; the test uses explicit illustrative descriptors, not an
H800 capture/capacity observation. Both actual H200 table methods typed-reject on
H800. No H800 qualified timing/calibration or gate waiver follows. No solely old
H20064 execution/package work existed in this slice. Root owns the68f891b common
workload, GPU/platform actions and integration with published B2a84c58/2ffd2cb.

## Open scope and handoff

- `read_isolated_scope(...)` returns one qualified isolated measurement receipt,
  including physicalTP1/targetTP8 and `serving_admission=false`. It never turns
  overlapping child scopes into a complete serving-layer prediction.
- Still missing: native routing/assignments and overlap; mHC/endpoint/zeroing/CPU
  measurements; decode/first-tail, FULL dummy and PIECEWISE capture equivalence;
  compiled/replay/physicalTP8/phase parity; full canonical E2E/four metric accuracy.
- Missing coverage raises `DeepSeekV4CoverageError` at component consumers;
  unsupported split/AFD/speculative configurations and complete-family/standalone
  training gates remain closed. No fallback or public runtime flag was added.
- No GPU/platform/auth/SSH/cloud/network/Docker/delegation, push or merge.
   Primary owns acceptance and integration with B/H5.

## Review correction

The primary reopened A following independent review. This supersedes the earlier
runtime-freeze statement: the V4 predictor now requires an explicit complete
isolated layout, checks source/cache/head consistency, selects no mixed-layout
medians, wraps only known reader coverage errors with their original causes, and
uses graph compute width for the layer0 pre-SP embedding all-reduce. The new
fixture `tests/fixtures/deepseek_v4_attention_layouts.json` is source5e051e4's
reduced H200 P6144 scope, not universal capacity or H800 evidence.

CPU red reproduction: layout5 failures, source-consistency2 failures, known
reader8 failures with1 unchanged unexpected-error control, and embedding1 failure.
Final affected suite:208 PASS/8 SKIP. Actual inputs:102 exact scopes/68 typed
negatives plus two exact mixed-layout controls PASS. No broad suite was repeated.
The test report records exact commands, raw artifact paths, native source and the
one-line B assertion recommendation. B files and timers were not edited.

Source correction `282cb110800b5641e9ab21d51ecf77dba601e233` is complete. Its sole
fresh clean strict77 passed:77 executed/compared/IDENTICAL, zero cache/provenance,
definition, missing-evidence or failure differences. Manifest source_dirty=false,
clean cache and matched source/driver/harness revision; retained checkout is clean.
`strict77_receipt.json.review_correction` preserves the selected result and original
paths; the original621b198 strict result is explicitly historical. Only task
records change afterward. Diff: `/data/ycfeng/tmp/v4_graph_measured_review.diff`.

All R1–R5 work is complete; primary acceptance, B assertion adaptation,
integration and timer audits are still root-owned. Native serving,
compiled/physical-TP/phase equivalence and H800 timings remain unqualified.

## Final P2 follow-up

Fresh static review found one omitted known validator error. The final runtime
repair adds only `Unknown typed operator ` to the V4 coverage conversion list.
Two real shared/MARLIN cases first leaked plain ValueError and then passed with
DeepSeekV4CoverageError retaining that exact original cause. See
`unknown_operator_red.json` and `unknown_operator_green.json`; the report preserves
the exact stdin reproduction and raw log paths. Existing unexpected-ValueError
propagation remains unchanged (9 focused controls PASS). The existing real-data
checker passes102 exact scopes/72 typed negatives/2 mixed-layout controls, adding
only four negative cases across the two measurement methods.

Final source `e3ce1660ccb58579980540b1eadce356c75da3aa` passed its sole final
clean77:77 executed/compared/IDENTICAL, zero cache/provenance, missing-evidence,
definition or failure differences. Manifest source/cache are clean and source,
driver and harness revisions match. Retained measured checkout is clean.
`strict77_receipt.json.p2_unknown_operator` records this run; all prior receipt
fields, including282cb11, are preserved exactly and not relabelled as P2 evidence.
Only records change afterward. Diff: `/data/ycfeng/tmp/v4_graph_measured_p2.diff`.

U1–U3 are complete. Root owns independent acceptance and A+B assertions. No
B/timers, native paths, consumption scopes or qualification limits changed.
