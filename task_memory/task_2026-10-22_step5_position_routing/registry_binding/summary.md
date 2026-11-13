# D29 registry binding completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive verified four-path binding repair and root integration contract. |

TaskID: D29-REGISTRY-SEAM-IMPLEMENTATION. Branch `feat/step5-registry-bindings`,
base `d96b32797ce5298097caa31e9948aa34985018db`.
Verified code commit: `8f96408cf6fbb19c45a918b2464aea3ddff29f1a`, four files,
287 additions/15 deletions. Task records remain local under the ignored
`task_memory/task_2026-10-22_step5_position_routing/registry_binding/` directory;
root integration can archive them separately without increasing the source budget.

## Deliverables

- `frontier/execution_time_predictor/prediction_model_registry.py`: retain native
  variants inside existing contract and precision registries; expose explicit
  selectors; enumerate native hash labels in views and registered bundles.
- `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py`:
  forward actual native context/source/group selection; retain context/source
  validation after selection.
- `frontier/execution_time_predictor/moe_routing_workload.py`: enumerate retained
  grouped-GEMM physical rows for the existing routed group before point recovery.
- `tests/unit/test_prediction_measurement_bindings.py`:174-line persistent CPU
  regression using real Step model/contracts, actual shared trainer/binder and
  both physical routed groups, plus selection/bundle/legacy controls.

## Exact integration interface

`_get_family_model(..., native_binding=None)` and public
`get_model(name, precision=None, layer_contract=None, *, native_binding=None)`
accept a mapping with `native_context`, `native_source` and
`selected_native_metadata`. Individual fields accept JSON strings or mappings;
selected metadata may contain the existing JSON/mapping `native_binding`.
Queries require explicit source when native candidates exist; omitted group may
resolve only when unique. Unavailable matching binding returns no family model;
ambiguous requests raise `ValueError`, never choose insertion order.

Existing native storage key is
`(name, (typed_identity, precision.upper(), canonical_json(native_binding)))`.
Unbound legacy keys and semantics remain unchanged. Native context uses existing
`moe_native_binding_context`, excluding point rows/workspaces. No global typed
serializer, model hash, shared manager or independent trainer was changed.

Native family/cluster view labels use existing `name:model_hash`; all bindings
survive. Conflicting same-hash native artifacts fail instead of being discarded.
Registered bundle typed identities remain the original scalar typed marker,
not the internal compound key. Legacy bundle labels/content remain unchanged.

## Observed validation

- Baseline-store standalone RED: final real dual-group fixture retains1 gate,
  expected assertion failure;1 failed,0 errors,9.51s.
- Final CPU regression:213 passed,0 failures/errors/skips,171.16s.
- Actual layers3/88 resolve distinct FP32 gate hashes75601934/fd767cb1 under
  FP8/BF16 expert groups. Bundle ID
  `3f2f6c45abef019ff59bdf8f66f49fd0c23880515dfde8235dbf7ca5f9aaca7a`
  contains30 validated artifacts, including both gate and expert bindings.
- Exact gross against1aee0af: registry93/120, predictor113/138,
  routing136/145, test174/180. Incremental repair302/360; caller delta35/60.
- `git diff --check` passes. Root-owned collective-sim submodule gitlink is
  preserved and excluded from the repair commit.

Exact commands, environment, receipts and verification limits are in
`test_report_2026-10-23_registry_bindings.md`; selected machine evidence is in
`evidence/`. Raw logs/caches stay under `/data/ycfeng/tmp`.

## Open and root-owned work

Independent selected actual fit/cache tests pass, but the inspected independent
public predictor has no registry binder. This repair does not claim whole-chain
independent coverage. Warm training may rewrite pickle bytes as before; bundle
publication byte immutability and warm hash/exact-lookup stability were verified.

Root must integrate and independently review this commit, resume four-path
ownership only after integration authorization, then run actual-main/strict77
and GPU/global native gates. CPU backend consumer and native observed compute
producer belong to PRICING and were not implemented here. No extra agents,
peer edits/imports, platform/GPU/network or dependency operations occurred.
