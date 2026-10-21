# D23-MRV2-INPUTS completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archive the completed CPU producer-to-fit/cache/lookup change and its limits. |

## Outcome

Truthful Step `attention_workload`, `mrv2_dp_dummy`, and `isolated_no_metadata` producer rows now reach the existing production loader/shared trainer/cache/root-count/exact cost path without cross-kind same-shape aliasing. Kind-bound physical provenance remains validated, not disabled. No scheduler/native qualification or accuracy claim follows.

Branch `feat/step5-mrv2-inputs`, worktree `/data/ycfeng/Frontier/.worktrees/step5-mrv2-inputs`, base `7fa73f9`. Verified source commit `ba484b0d9e1534f5f8e742127addee5cafe62560`. Task/session `D23-MRV2-INPUTS` / `ses_ea08efc2cffe6Z1Rc7srkVdebh`.

## Deliverables in requirement order

1. Shared domain: `frontier/attention/step5_inputs.py:18` contains one label/code/scope/metadata classification table. The existing `MRv2DummyDescriptor` moved here at line52 and retains import compatibility through `frontier/profiling/step5/inputs.py`. Line99 validates kind/physical scope, metadata, MRv2 runner, request ownership, PAD_SLOT_ID/cache semantics and exact descriptor geometry. Legal source-observed FULL/null with uniform_token_count1 is accepted while raw null remains unchanged.
2. Producer and feature schema: `frontier/profiling/step5/rows.py` uses shared classification/provenance; `frontier/attention/profiling_mapping.py:38` includes `native_workload_code`. Real query/context/physical padding, native scope fusion, method policy, actual count and typed scope validations remain intact. Old Step caches naturally invalidate via schema/source identity; no migration fallback exists.
3. Existing consumption/cache: `frontier/execution_time_predictor/step5.py:134,231,301,366–380,409` carries kind feature, observed-only scope mapping, union identity and model domains. Existing manager/publisher/cache code was not changed. All three kinds can coexist in one measured frame and cache.
4. Existing cost lookup: `frontier/execution_time_predictor/step5_cost.py:66,120,162–164` labels the current no-metadata binder as isolated diagnostic, marks ordinary metadata-bearing request queries as request-owned, and validates exact query kind against cached scope. Scheduler/batch-workload geometry and role graph-policy mapping are unchanged.
5. Focused tests: `tests/unit/test_step5_mrv2_inputs.py:45,159,193,205` exercises actual build_row/rows_to_dataframe → real shared fit/cold/warm/bundle/duration-count lookup, same-numeric collisions, dual missing-kind failure, strict invalid fields, observed FULL/null, request-owned runtime isolation and all-three-kind cache. `tests/unit/test_step5_attention_input_union.py:167` verifies separately sourced observed kinds merge into existing union/cache. Existing source-bound native InputBatch/state/builder tests gained FULL1/FULL2 raw-null cases; measured/no-metadata fixtures now construct and validate truthful isolated producer rows.

## Observed verification

- Inherited producer baseline465 PASS/24.84s; initial measured baseline141 PASS/55.42s. No inherited failures in those executed commands.
- New real producer mixed-input RED:3 failures/2.12s at the old scalar physical_scope check; first GREEN43 PASS/12.96s.
- Final producer467 PASS/16.70s; broad measured/fit/replay/bundle/PD263 PASS/191.88s; final focused identity/union49 PASS/17.56s. All final XML files have0 failures/errors/skips. Focused overlaps the broad population.
- Actual local native descriptor check:13/13 groups and26 SWA/DSA CPU producer rows pass shared loader/fit/cold/warm exact duration/count checks, including5 raw-null FULL groups. Selected receipt `observed_descriptors.json`, executed body `observed_descriptor_command.md`. These use fixture costs, not native measured timings.
- `git diff --check` and scoped source-step Git cleanliness PASS. No GPU/auth/platform/cloud/SSH/push/further delegation or unrelated `.pi/.cortexkit` edits. Actual-main runner/fixture, scheduler, giant shared manager and graph-policy files remained untouched.

Exact commands, environments, raw log/XML paths and the resolved two obsolete isolated-rejection assertions are in `test_report_2026-10-22_mrv2_inputs.md`; requirements and phase self-check are `requirements.md` and `checkpoint.md`.

## Open/deferred dependencies

No assigned CPU subtask remains unfinished. Parent owns first-prediction all-lane scheduler rendezvous/native binding, role graph-family mapping and integrated actual-main acceptance. The existing one-slot DummyForwardBatch binder remains isolated diagnostic until parent replaces it. MRv2 cost regressions bind truthful producer features at the existing cost seam; they do not implement a scheduler-native binder.

Native admission remains separate: original startup192 runtime gate FAIL, reference-state/pressure UNVERIFIED and two HTTP502 are preserved. Reached context max7166 is an observed input, not7167. This work neither repairs nor promotes those native gates. Parent re-runs acceptance after integrating the source and task-record commits.
