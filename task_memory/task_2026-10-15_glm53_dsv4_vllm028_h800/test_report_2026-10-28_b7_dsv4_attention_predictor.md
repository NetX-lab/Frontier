# Test report: stage 1a assignment B7, DeepSeek-V4 replay attention predictor and per-scope gap composition (2026-10-28)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead review and verification of Codex assignment B7 (`54e3584a`). |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B7_dsv4_attention_predictor.md](h3o_operator_costs/codex_assignment_B7_dsv4_attention_predictor.md) (entry 1 and the entry 2 revision; entry 10's runtime wiring moved to B8).
- Codex report: [h3o_operator_costs/codex-implement-54e3584a-5247-4c57-ade2-078d463b06a3/implementation.md](h3o_operator_costs/codex-implement-54e3584a-5247-4c57-ade2-078d463b06a3/implementation.md) (outcome `implemented`).
- Base: `a1feee14`.
- Changed files (all inside the assigned scope):
  - `frontier/attention/dsv4_features.py` (new, 134 lines): the phase rule (decode, first_tail, prefill, dummy), the 22-scope feature table and `build_dsv4_features` with closed-form causal sums of the SWA, compressed and selected key pairs and of the compressed writes.
  - `frontier/execution_time_predictor/dsv4_predictor.py` (new, 280 lines): `load_dsv4_attention_dataframe` (KERNEL_ONLY replay schema `dsv4_attention_replay.v1` only), `train_dsv4_attention_models` (one time model and one signed-gap model per key) and `DeepSeekV4AttentionPredictor.scope_times`.
  - `frontier/execution_time_predictor/dsv4_composition.py`: `compose_attention_phase(scope_ms, scope_gap_ms, ...)` weighs each scope by its cost plus its own signed gap; `segment_class` stays public for callers that price a scope without a measured gap.
  - `frontier/execution_time_predictor/deepseek_v4.py`: `load_attention_dataframe` takes a `schema_version` keyword; `_validate_isolated_identity` became the shared `validate_measured_identity` (the isolated path selects `graph_mode="eager"` after it); the balanced-routing check became the shared `validate_balanced_routing`.
  - `frontier/attention/deepseek_v4.py`: `deepseek_v4_padded_q_heads` (FlashMLA's 64 and 128 padded query-head widths) shared by the workload builders and the replay loader.
  - `frontier/profiling/attention/dsv4_replay_rows.py` and `dsv4_attention_main.py`: `DSV4_REPLAY_OP_NAME` moved to the rows module so the loader and the aggregate filename share one constant.
  - `tests/comparison/calibration/dsv4_composition_check.py`: the new compose signature; the class-value diagnostic passes kernel count times class value per scope.
  - `tests/unit/test_profiling_runtime_boundary.py`: allowlist entries for `dsv4_composition.py` (the kernel-gap schema names, imported since `7003cf1d`) and `dsv4_predictor.py` (the replay rows schema).
  - Tests: `tests/unit/test_dsv4_features.py` (new), `tests/unit/test_dsv4_predictor.py` (new), `tests/unit/test_dsv4_composition.py`, `tests/unit/test_dsv4_replay_rows.py`.

## Lead review

- **Features.** The phase rule and the 22-scope table match the assignment. The closed forms were checked by hand: `_floor_sum(n, r)` is the sum of floor(k/r) for k = 1..n, and `_clipped_floor_sum` caps each term at the index top-k from k = r·topk onward. The tests compare every sum with a token loop over 103 batches per variant, including contexts across a ratio boundary and beyond r·topk and the sliding window.
- **Predictor keys.** (scope, variant, graph runtime mode, phase), plus the planner role for `dsv4_sparse_attention` only. This is the recorded deviation from design §3.6 (planner-role comparison in `evidence/p1h_bridge/package_03/b7_planner_role_gaps.*`).
- **Loader.** It admits KERNEL_ONLY replay rows only. Checks:
  - the measured identity and the balanced routing, through the helpers shared with the isolated path;
  - `engine_cudagraph_mode` equals `graph_mode`;
  - padded heads and kv dtypes, the variant-to-ratio map, booleans, one `multi_stream_gemm_token_threshold`, scope ownership, finite positive times and finite gaps of present scopes, and NaN for absent scopes.

  Every rejection names the file and the column.
- **Isolated path.** Its behavior is unchanged. It still rejects a non-eager table, now through the `graph_mode` selection after the identity check instead of inside it, with a different message. It is deleted in B8.
- **Lead corrections** (small, made directly):
  - removed an unused `as exc` in `dsv4_predictor._number`;
  - reflowed two over-long lines in `dsv4_composition.py` (module docstring, `compose_attention_phase` signature);
  - removed an unused `DeepSeekV4BatchFeatures` import in `tests/unit/test_dsv4_predictor.py`.

  After these, the AST unused-import check reports none in the changed files. The touched tests were rerun afterwards: 121 passed.
- **Module sizes.** All at or below 2,000 lines: `execution_time_predictor/deepseek_v4.py` 975, `attention/deepseek_v4.py` 471, `dsv4_predictor.py` 280.

## Checks

Environment:
- Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6).
- Variables: `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `ulimit -v 31457280`.
- Lead scratch: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b7_review/`; runner `run_regression.sh`.

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| New and regression suites | `pytest tests/unit/test_dsv4_features.py test_dsv4_predictor.py test_dsv4_composition.py test_dsv4_composition_check.py test_dsv4_replay_rows.py test_dsv4_replay_gaps.py test_dsv4_graph_bridge_check.py test_dsv4_kernel_identity_check.py test_dsv4_graph_replay_profiling.py test_dsv4_attention_profiling.py test_dsv4_kv_cache_groups.py test_dsv4_graph_measured.py test_dsv4_attention_workload.py test_dsv4_step_cost_input.py test_dsv4_structural_execution.py test_dsv4_source_execution.py test_dsv4_structural_pd.py test_dsa_predictor.py test_dsa_attention_profiling.py -q` | all pass | 883 passed, 75 skipped (all in `test_dsv4_graph_measured.py`: accepted prefill receipts are unavailable), equal to Codex's count | PASS |
| Known failures | `pytest tests/unit/test_profiling_runtime_boundary.py tests/unit/test_kernel_gap_pricing.py -q` | only the two known failures | 2 failed, 22 passed. The boundary test fails only on `deepseek_v4.py` (AST listing: one extra import, `frontier.profiling.attention.dsa_rows`; B8 removes it). `test_every_collective_launches_one_kernel_by_default` fails as before (`OperatorSpec` has no `kernel_count`) | PASS (unchanged) |
| A4 unchanged | `python .../b6_baseline/a4_baseline.py b7_review/a4_report.json`; `cmp` with `.../b6_baseline/a4_report_d2a226fe.json` | PASS, byte-identical | `PASS 125`; `cmp` exit 0 | PASS |
| Real table composition | `python h3o_operator_costs/evidence/p1h_bridge/package_03/b7_real_table_composition.py`. Inputs: B6 table sha256 `277626cd766865bd...`, exact-lookup fake fit, each row's own batch features, compared with `layer_window_stats.median` | median ≤ 5%, p90 ≤ 10% | 505 rows, 554 models (277 time, 277 gap; decode 160, dummy 234, prefill 80, first_tail 80); median 0.49%, p90 1.34%, max 6.67%; 2 rows above 5%; JSON identical to Codex's report | PASS |
| Cold imports | fresh interpreter per module: `dsv4_predictor`, `dsv4_features`, `dsv4_composition`; no torch, vllm or cuda modules | exit 0 | PASS for all three | PASS |

The two rows above 5% are both CSA, FULL, decode, context about 6,150:
- row 13 (`e0-decode-b1-p6144`, follower layer, 1 padded row): 5.75%;
- row 204 (`e0-decode-b4-p6144`, planner layer, 4 padded rows): 6.67%.

In both, the largest pooled-scope weight difference is `dsv4_compressor_fused_compress_store`: −5.42 µs and +8.42 µs. That scope pools both planner roles, and the fit averages rows that share one feature tuple. This names the largest contributor; the differences do not simply add through the fork/join schedule. No key or feature was tuned to meet the limits.

Evidence in `h3o_operator_costs/evidence/p1h_bridge/package_03/`: `b7_real_table_composition.py` (lead copy of Codex's scratch script, writing next to itself), `b7_real_table_composition.json`, `b7_real_table_composition.log`.

## Limits

- The real-table check is in-sample: the fit is an exact lookup trained on the same 505 rows. It shows that the keys, features and composition reproduce the measured windows. It does not show accuracy at unmeasured shapes; A6 (group-held-out accuracy) does that.
- The predictor is not yet called by the simulator. Admission, registration, training in the sklearn predictor, the trained composition path and the V4 gap selection come in B8 (entries 4, 5, 3 and the entry 10 wiring).
- Older evidence scripts call the pre-B7 signature `compose_attention_phase(scope_ms, kernel_counts, gap_ms, ...)`: `evidence/p1h_bridge/b2/a4_*.py` and `evidence/p1h_bridge/package_03/b6_check_row_composition.py`. They record what ran at their commits and are not rerun on B7. The frozen copies under `p1h_bridge/package_0*/` carry their own module copy.
- The rows cover the package 03 cases only: one H200, a TP1 engine of a TP8 target. CPU only; no GPU run.
