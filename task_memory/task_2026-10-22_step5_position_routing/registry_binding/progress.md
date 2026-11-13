# D29 registry binding progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Commit verified four-path repair as8f96408c; leave root-owned submodule unchanged. |
| 2026-10-23 | Complete 213-test acceptance run, standalone baseline RED and exact four-path budget. |
| 2026-10-23 | Actual dual-group RED/GREEN; native key/view/query implementation in progress. |
| 2026-10-23 | Complete read-only caller inspection and prepare real-pattern regression. |

TaskID: D29-REGISTRY-SEAM-IMPLEMENTATION.

- completed: branch/base verified; root-owned submodule modification preserved.
- completed: inspected contract keys, precision buckets, public get, family views,
  shared initialization/admission, registered bundles and both CORE consumers.
  Family views feed real admission; retaining a last-wins projection is incorrect.
- completed: persistent RED using actual source fixture, real trainer and typed
  contracts for Step groups3-87/88-90. The archived independent RED is
  `/data/ycfeng/tmp/d29-core-registry-collision03.log`.
- completed: native retention/query, caller selectors and enumeration.
- completed: corrected actual dual-binder RED (`d29_registry_actual_red02.log`)
  with baseline `_store_model_precision` loaded read-only from Git in memory:
  one retained FP32 gate rather than two. GREEN `d29_registry_green03.log`:
  1 passed in12.09s; native keys retain/select both layers3/88.
- completed: expanded binding/source/backend/group selection, ambiguity/missing-source,
  family and cluster enumeration, bundle reload, physical-row recovery, legacy and
  warm-cache checks. Final02: 213 passed, zero failures/errors/skips,171.16s.
- completed: standalone baseline-store RED03 on the final persistent fixture;
  one failure, zero errors,9.51s, retained gate count1 instead of2.
- completed: exact budgets against1aee0af: registry93/120, predictor113/138,
  routing136/145, new test174/180 lines. Incremental repair againstd96b3279:
  registry93 + predictor19 + routing16 + test174 =302/360; callers35/60.
- completed: verified code commit `8f96408cf6fbb19c45a918b2464aea3ddff29f1a`
  (`fix(step5): retain native measurement bindings in model registry`), four
  code/test files,287 additions/15 deletions. Test report and English summary
  remain local in this ignored task directory; root can archive them separately.
  Root-owned submodule is the only remaining Git modification and was not staged.
- pending (root-owned): integrate/review commit, archive local records as needed,
  then run actual-main/strict77 and global/native gates. This session is complete.

Quick iterations: green01 rejected nullable native context through the stricter
top-level provenance serializer. Reused existing `canonical_json` for nested native
context JSON invariants instead. green02 revealed the new fixture accidentally
trained only one group; fixture now calls the real trainer for every selected
binding. red01 is not the final dual-binder evidence; red02 above is decisive.

Inspection limit: current independent `_train_moe_models()` returns bare-name
estimators and the public independent predictor leaves `_model_manager=None`.
No existing independent registry binder was found. Its real outputs were
tested without claiming the whole independent public path is covered.

Final checks include actual independent `test_independent_selected_actual_fit_cache_binds_native_context`
and shared `test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers`,
plus the new actual two-group binder test. No independent public registry binder
was added; that limitation remains explicit, not a claim of whole-chain acceptance.

Native invariant fields accept existing JSON strings and equivalent mappings.
`canonical_json` handles nullable layouts only within native context/selected
metadata; original unbound measurement and typed/cache serialization remain intact.
Synthetic same-hash conflicting bindings are retained in buckets, and family views
fail instead of silently replacing their bundle labels. Real trainer hashes differ.

Bundle observed:30 artifacts, ID
`3f2f6c45abef019ff59bdf8f66f49fd0c23880515dfde8235dbf7ca5f9aaca7a`.
FP32 gate labels: `moe_gating_linear:75601934` (3-87) and
`moe_gating_linear:fd767cb1` (88-90). Publication preserves pickle bytes;
warm training preserves exact lookup/hash but may rewrite existing duration/count
pickle files as before. No no-write claim is made for warm training.

Cause established by existing independent investigation: both native estimators
fit and have distinct cache hashes, but the same (name, typed identity, precision)
registry key overwrites the first. No trainer/global identity/hash change is needed.
