# W128-V4-GRAPH-MEASURED (A)

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Archived implemented CPU capabilities and measured-scope evidence; final regression status pending. |

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

Affected tests:686 PASS/8 SKIP. Real rows:102 exact positive isolated scope
comparisons and12 typed negatives PASS. Both measurement methods retain their
identities and remain separate. Source9f58dbe compute and source5e051e4 attention
are not relabelled or rewritten. The MARLIN matching lane is synthetic, not
native routing evidence. Full sweep:5581 PASS/108 FAIL/72 SKIP/20 ERROR;
the only historical baseline-passing regression was the new CPU-helper allowlist
and passed on the686-test rerun. Zero unresolved regressions among historical
passing IDs present in the sweep; unrelated missing-dependency failures and64
absent historical IDs remain explicitly recorded. Clean strict77 is pending.

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
