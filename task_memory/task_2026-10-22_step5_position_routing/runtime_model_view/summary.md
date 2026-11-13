# D29 named-model query component delivery

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Accept and integrate ffb435ca after parent checks, both reviews and the9579 composed audit. |
| 2026-10-24 | Archive implementation and production verification; leave parent acceptance open. |

## Outcome

**Component accepted and integrated.** Source `ffb435ca` matches owner `1737ce9c`;
records `8ecb705e` preserve owner `3d75c786`. Composed audit `de5eeab0` is archived
at `c6b1f3b8`, with zero violations. Parent checks all124 final bindings and the
120 non-query digest in `parent/acceptance.json`. Complete CLI qualification
is still open.

Implemented approved D29-NAMED-MODEL-QUERY in isolated branch/worktree
`step5-d29-named-query` at
`/data/ycfeng/Frontier/.worktrees/step5-d29-named-query`. Task/session:
`D29-NAMED-MODEL-QUERY` / `ses_e93af1ff7ffe0YosowRNrc24Xa`.
Source commit1737ce9c4a761d2c57ee617f5f3a78d7bcfd9a58 follows focused checks and
precedes the production wave.

The private family view accepts a keyword-only logical model name; position
queries request `moe_grouped_gemm`. Unfiltered behavior and the cache-name guard
remain. No public API/config/module/cache is added. Only the approved unrelated
flattened-view error surface changes; current-model validation remains live.

## Deliverables

Source/test paths, and no other source ownership:

- `frontier/execution_time_predictor/prediction_model_registry.py`
- `frontier/execution_time_predictor/moe_routing_workload.py`
- `tests/unit/test_prediction_measurement_bindings.py`
- `tests/unit/test_moe_routing_runtime.py`

Records under `task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/`:
`plan.md`, `progress.md`, `test_report_2026-10-24_named_model_query.md`, this
`summary.md`, and `repair_evidence/` with production wiring, source/input freeze,
raw compact logs/XML/wave receipts, verification and manifest. Large full JSONLs
remain under `/data/ycfeng/tmp/d29_named_model_query01`.

## Observed validation

- RED10FAIL/2PASS; final selected GREEN12PASS/17deselected65.92s.
- Focused baseline117PASS/1SKIP165.09s versus candidate125PASS/1SKIP195.81s:
  all118 existing statuses unchanged; eight new tests pass.
- Registry governance14PASS/1FAIL on each side: unchanged missing Torch in the
  existing producer round-trip test, not hidden by dependency workarounds.
- Committed-source full208-event wave, no candidate method installation:
  warm pricing median0.122730910→0.088166638s (−28.1626%); contract derivations
  384→0; numerical/fresh-proof/collective/point/key calls288/144/64/16/192 unchanged.
- Nineteen whole-file byte comparisons, including15 complete208-event JSONLs,
  complete costs/proofs/keys/order and2309545B ledger; snapshot and64 wrong-key
  guards pass. The3902B late-error prefix matches separately from unfrozen metadata
  timestamps. One binding read and zero pricing reads;262 fixture metadata/input
  hashes and23 source bindings per side unchanged. Protected body SHA256 remains
  1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555.
- Measured step14/3/49/30=96gross (source17/tests79), retained84. Cumulative
  registry104/120, workload196/223, bindings348/353, runtime147/151. Incremental
  union: CORE4261/4404, PRICING3954/3956, native1350/1582, source9565/9942,
  global9579/10000. No donor changes. Independent composed audit and the parent
  final-head guard now confirm these totals and all114 caps.
- Parent final-source focused125PASS/1existingSKIP and full208-event replay with
  19 byte comparisons pass. Both independent review axes have No findings.

## Open acceptance and limits

Standards/Spec reviews, parent verification, composed audit and integration are
complete. Full-output/current-model whole-workflow validation,
final frozen regression/full-unit/strict77,6144/1024/2400s CLI, native/GPU/G3/E2E
remain open. No such runs or external actions were performed here. Output emission
was not optimized; observed medians37.372→42.533ms are not attributed to this
query change. Correct-pin optional runtime artifacts were retained locally with
clean submodule3dac53c and no source or native cache-key changes.
