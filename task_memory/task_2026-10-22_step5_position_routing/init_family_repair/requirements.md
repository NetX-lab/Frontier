# D29-INIT-FAMILY-RESOLUTION requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record the delegated loader correction and acceptance limits. |

## [Original Request]

Implement the measured minimum CORE loader correction: remove repeated fixed-model
family derivation while preserving every row's typed admission. Use a new isolated
worktree at `/data/ycfeng/Frontier/.worktrees/step5-d29-loader-family`, based on
`24f970cb` (source byte-identical to `e41abc07`), without overwriting an existing
worktree. Own only `frontier/execution_time_predictor/profiling_dataframe_loaders.py`,
`tests/unit/test_moe_grouped_gemm_padded_features.py`, and this `init_family_repair/`
directory. Source delta <=6, test <=39, total <=45 gross relative to `24f970cb`;
retained path caps 17/237. CORE4404, global10000 and 111 effective paths stay fixed.

Use the existing `resolve_runtime_attention_family` import and existing
`architecture_family_id` keyword; derive once before the MoE typed `.map` and keep
actual per-row JSON/schema/ownership validation before filtering. No cache,
validator/binder API changes, skip/default/fallback, dataset changes, peer edits or
merges. Establish meaningful deterministic RED at the actual `_load_moe_df` seam
before fix/GREEN. Verify one derivation per call, unchanged selected data and paired
duration/kernel-count binding, and rejection of corruption in an unselected row.
Reuse `_position_source_rows` and existing negative coverage; do not duplicate a
fixture writer. Run affected existing suites, AST/whitespace and gross/path checks;
commit verified code separately from records. No complete CLI, GPU/platform,
full-input scan or dependency installation. Scratch under `/data/ycfeng/tmp`;
diagnosis probes <=60s/30GiB. Submodule gitlink remains root-owned `3dac53c`.
Primary owns final review, integration and freeze.

## Accepted seam and evidence

The requester explicitly selects the existing `_load_moe_df` seam. Instrumentation
must wrap and execute real derivation, not substitute validator behavior. Existing
parent3000-row measurement is 2.837165373s ->0.274640171s, one derivation and equal
payload; it is not whole-loader/initialization/CLI qualification. Authority and
allocation remain in `../repairs/init_validation_scope_amendment.json`.
