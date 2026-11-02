# Test report: stage 1a assignment B6, graph-replay V4 attention rows with per-scope gaps (2026-10-28)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead review and verification of Codex assignment B6 (`4663821b`). |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B6_replay_rows.md](h3o_operator_costs/codex_assignment_B6_replay_rows.md).
- Codex report: [h3o_operator_costs/codex-implement-4663821b-c54e-481f-9d68-397701609b2c/implementation.md](h3o_operator_costs/codex-implement-4663821b-c54e-481f-9d68-397701609b2c/implementation.md) (outcome `implemented`).
- Base: `d2a226fe`.
- Changed files (all inside the assigned scope):
  - `frontier/profiling/attention/dsv4_replay_rows.py` (new, 212 lines): `build_dsv4_replay_rows`, schema `dsv4_attention_replay.v1`.
  - `frontier/profiling/attention/dsa_rows.py`: `request_load_columns` split from `batch_load_columns` (mixed first-tail batches); `_stat_columns` takes a column prefix, default `time_stats`.
  - `frontier/profiling/attention/dsv4_attention_plan.py`: `require_engine_layer_inventory` (ordered layers, `max(1, ratio)`, target TP), shared by the rows and A4.
  - `frontier/profiling/attention/dsv4_attention_main.py`: the graph-replay branch of `aggregate` writes `attention_graph_replay_kernel_only.csv` with `graph_mode` from the plan.
  - `frontier/profiling/kernel_gap/dsv4_replay_gaps.py`: `LayerCallTiming`, `attention_gaps_by_layer`, `layer_call_timing` (A4's exclusion order and window formula moved here).
  - `tests/comparison/calibration/dsv4_composition_check.py`: uses the shared helpers.
  - Tests: `tests/unit/test_dsv4_replay_rows.py` (new, 34 tests), `tests/unit/test_dsv4_replay_gaps.py`, `tests/unit/test_dsa_attention_profiling.py`.

## Lead review

- Row key: engine, case, variant, planner role, dispatch (`cg_mode`, `num_tokens`, `num_reqs`), request signature, variant metadata and the observed scope set, as assigned. Rows are ordered by (case, variant, planner first, first step).
- Validation before rows: plan policy `graph_replay`; one record per engine; engine mode equals the plan; layer inventory and realized KV-cache groups equal the plan; one routing identity; every row `KERNEL_ONLY`; every call scope is a family operator of its variant, none warmup-only, none repeated; each scope's `kernel_count` equals its owned kernel intervals; every planned layer has a call in every step. A call that is not host-ahead raises.
- Per scope: `time_stats.<s>.{min,max,mean,median,std,count,kernel_count}` and `gap_stats.<s>.{min,max,mean,median,std,count}` (the per-call sum of the scope's kernels' signed gaps; 0 when a scope has no kernel gap entry). Per row: `layer_window_stats.*`.
- Dummy rows have no request load or key-length columns. `swa_key_len`, `compressed_key_len` and `selected_compressed_key_len` derive from `max_key_len`; the per-request lists (`prefill_seq_lens`, `prefill_kv_cache_sizes`, `decode_kv_cache_sizes`) are present, so the entry 1 loader can rebuild per-request features from them.
- No unused imports in the changed files (AST check). No lead correction was needed.

## Checks

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `ulimit -v 31457280`. Lead scratch: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_review/`.

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Unit and regression suites | `pytest tests/unit/test_dsv4_replay_rows.py test_dsv4_composition_check.py test_dsv4_replay_gaps.py test_dsv4_graph_bridge_check.py test_dsv4_kernel_identity_check.py test_dsv4_composition.py test_dsv4_graph_replay_profiling.py test_dsv4_attention_profiling.py test_dsv4_kv_cache_groups.py test_dsa_attention_profiling.py test_dsa_extend_profiling.py test_dsa_full_dummy_profiling.py test_dsa_query_one_profiling.py test_dsa_predictor.py -q` | all pass | 661 passed, 1 skipped (`test_locked_native_source_contract` needs `FRONTIER_VLLM_BS_ROOT`) | PASS |
| Known failures | `pytest tests/unit/test_profiling_runtime_boundary.py tests/unit/test_kernel_gap_pricing.py -q` | only the two known failures | 2 failed (`test_runtime_predictors_use_only_allowlisted_profiling_helpers`, `test_every_collective_launches_one_kernel_by_default`), 22 passed; Codex's before/after logs of the first are identical apart from elapsed time | PASS (unchanged) |
| A4 unchanged | `python .../b6_baseline/a4_baseline.py b6_review/a4_report.json`; `cmp .../b6_baseline/a4_report_d2a226fe.json b6_review/a4_report.json` | PASS, byte-identical | `PASS 125`; `cmp` exit 0 | PASS |
| Real aggregate | `python b6_review/run_aggregate.py` (the bridge config's `plan_args`, `graph_mode_flags("FULL_AND_PIECEWISE", sizes)`, raw dir with symlinks to the package 03 raw files, `--source_revision 230239c8 --model_config_revision package_03`) | 505 rows; decode 400, dummy 50, first_tail 50, prefill 5; FULL 345, PIECEWISE 150, NONE 10; window counts sum to 1925 | 505; 400 / 50 / 50 / 5; 345 / 150 / 10; 1925; one schema, graph mode, source and config revision | PASS |
| Determinism | `cmp` of the lead CSV with Codex's CSV | identical | identical; sha256 `277626cd766865bd...` | PASS |
| Row composition of the medians | `python b6_review/check_row_composition.py` (scope median cost plus median signed gap, zero class gaps, against `layer_window_stats.median`) | median ≤ 5%, p90 ≤ 10% | median 2.1e-16, p90 0.77%, max 3.19%; 178 single-call rows, max error 2.2e-16; 0 rows above 5% | PASS |
| Cold imports | fresh interpreter: `import frontier.profiling.attention.dsv4_replay_rows`, `...dsv4_attention_main`; no torch, vllm or cuda modules | exit 0 | exit 0, PASS both | PASS |

Evidence copied to `h3o_operator_costs/evidence/p1h_bridge/package_03/`: `b6_run_aggregate.py`, `b6_check_row_composition.py`, `b6_row_composition.json`, `b6_rows_sha256.txt`. The 4.8 MB CSV stays in the scratch directory; the committed code rebuilds it from the package 03 raw records.

## Limits

- Multi-call rows compose to 0.77% at p90 and 3.19% at most, because the median of a sum differs from the sum of medians; single-call rows are exact. This is the row statistic, not a prediction; A6 (group-held-out accuracy) qualifies prediction at other shapes.
- The rows cover the package 03 cases only (one H200, TP1 engine of a TP8 target). The production grids G1h come later (stage 2).
- CPU only; no GPU run.
