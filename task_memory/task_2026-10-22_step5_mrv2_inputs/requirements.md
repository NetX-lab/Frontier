# D23-MRV2-INPUTS requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record the delegated CPU producer-to-consumer identity scope. |
| 2026-10-22 | Preserve source-observed FULL/null dispatch descriptors under explicit follow-up authority. |

## [Original Request]

Implement the independently authorized producer → typed fit/cache/lookup identity gap, CPU only, from primary commit `7fa73f9` (including corrected producer `82f4452`). Truthful `attention_workload`, `mrv2_dp_dummy`, and `isolated_no_metadata` rows must coexist without same-numeric-shape aliasing. Preserve per-kind physical scopes, source geometry, actual methods/counts/typed scopes, query/context/padding semantics, and observed-only coverage. Reject missing/unknown/mislabeled kind and malformed MRv2 descriptor/provenance fields. Reuse the existing descriptor validation/splitting and trainer/cache/root/count pipeline; no duplicate publisher or model registry, no old-cache fallback.

Authorized files: `frontier/execution_time_predictor/step5.py`, `frontier/attention/profiling_mapping.py`, `frontier/profiling/step5/{inputs,rows}.py`, a small neutral shared runtime/domain module if needed, and `step5_cost.py` only for row-feature/source-identity plumbing. Relevant unit measured/profile fixtures and focused regressions are authorized. No scheduler/batch geometry/graph-policy changes, actual-main runner/fixture edits, further delegation, GPU/auth/platform/cloud/SSH/push, or unrelated `.pi/.cortexkit` changes. Parent owns the material scheduler/native binding decision and final acceptance.

Read parent requirements D20/D21/D22 and current producer/integration records. CPU implementation is approved independently of native evidence/profile admission; no native qualification or accuracy claims. Keep the one-slot no-metadata binder explicitly isolated diagnostic. Create branch `feat/step5-mrv2-inputs` in `.worktrees/step5-mrv2-inputs`, commit verified steps, and leave a clean scoped tree.

## Acceptance

- Actual `build_row`/`rows_to_dataframe` NONE/FULL/PIECEWISE MRv2 rows reach shared fit, cold/warm cache and exact duration/count lookup.
- Same-shape request-owned and MRv2 dummy keys stay distinct; missing coverage fails in either direction.
- Kind-bound physical identity survives input union, model cache, bundle and cost lookup. Coverage/domains contain only observed kinds.
- Invalid runner, descriptor counts/phase/mode, request ownership, PAD_SLOT_ID/cache semantics and metadata reject before fit.
- Re-run the 465 producer baseline and affected measured fit/replay/bundle suites; preserve inherited failures and repair in-scope inconsistencies.

References: `../task_2026-10-15_step5_preview_h800/requirements.md` D20–D22; `../task_2026-10-15_step5_preview_h800/test_report_2026-10-22_native_integration.md`; `../task_2026-10-21_step5_measured_compute/plan.md`.

## [Original Request] Source-observed descriptor follow-up

Read only the primary's local `canonical_g3/native_judgment/{report.md,mrv2_descriptors.json,producer_descriptor_check.json}`. The old descriptor rejects 5/13 actual dummy groups: legal FULL has raw `max_query_len=null` and `uniform_token_count=1`. Accept that source-backed case, retain null, derive query lengths from actual InputBatch dimensions, and add observed-descriptor regression. All13 groups have metadata92/92 and block tables `[num_reqs,448]`; PIECEWISE raw null requests gives one actual request with `[16]`, `[24]`, or `[32]`; FULL2 gives `[1,1]`. P lanes also execute FULL1, not just NONE1. Reached context max7166, not7167. This does not authorize role graph-family/scheduler edits or promote G3: reference state/pressure UNVERIFIED, two HTTP502, original startup192 runtime gate FAIL remain unchanged.
