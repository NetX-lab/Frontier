# D29 registry mixed-view repair archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive verified lossless mixed views and separate root-review handoff. |

TaskID: D29-REGISTRY-VIEW-REPAIR. Branch `feat/step5-registry-bindings`,
base source8f96408c and record archiveb6714757.

## Deliverables and interface

- `frontier/execution_time_predictor/prediction_model_registry.py`: three lines
  restrict bare-name family/cluster projection to the non-native domain through
  the existing resolver. Native hash enumeration, public lookup and bundle paths
  remain unchanged; no fallback or parallel registry was added.
- `tests/unit/test_prediction_measurement_bindings.py`:26 added lines,200 total;
  actual fitted native gate plus matching legacy/non-native typed same-name model,
  both insertion orders, real public family/cluster views and strict public get.

Private `_get_family_model(..., include_native=True)` retains normal native
selection and ambiguity checks. Only `_models_view_for_family` requests
`include_native=False` for the already-enumerated bare-name projection. Public
`get_model(..., native_binding=...)` has no interface or behavior change. Global
typed serialization, cache hashes and both caller files remain unchanged.

## Observed validation and budget

- Final four-case RED:4 failures,0 errors/skips,36.54s, exact mixed-domain view
  ambiguity. Tight GREEN:4 passed,35.85s.
- Affected acceptance:217 passed,0 failures/errors/skips,213.85s; includes
  original213 checks and all native bundle/cache/legacy controls.
- Incremental29/58gross; whole repair331/360 vsd96b3279; callers35/60 unchanged.
- Revised per-path usage vs1aee0af: registry96/120, test200/200,
  predictor113/118 and routing136/145. Whitespace check passes.

Exact commands/environment are in
`test_report_2026-10-23_registry_view_repair.md`; machine receipts and review diff
are in `evidence/`. These records are archived with the fix through gitadd-f.
Root-owned submodule is preserved and excluded.

## Remaining dependencies

Root independently re-reviews and integrates the separate fix commit. Actual-main,
strict77, global/native GPU gates and the pre-existing independent consumer/public
binder issue remain root-owned and were not expanded into this repair.
