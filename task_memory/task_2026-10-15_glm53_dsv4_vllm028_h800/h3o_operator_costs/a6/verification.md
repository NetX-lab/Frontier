# B19 verification and review handoff — 2026-10-29

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Lead review addendum at the end: the collective held-out under D-A6-COLLECTIVE-UNIT, the regenerated `result.json` and `summary.md`, the frozen plan, and the diagnosis of the three failing scopes. It supersedes the collective and frozen-plan statements of the Codex handoff below. |
| 2026-10-29 | Created by B19 (Codex `70321f85`) as its verification and handoff record. |

Outcome: **partial**. Joint A5 admission and the attention A6 tool/result are implemented. Uniform collective held-out evaluation needs a unit/trainer decision, and the frozen package_03 plan is absent. The empirical attention A6 gate fails; this is the required audit result, not a reason to tune the model here. Claude retains review, acceptance, commits and task records.

Working root: `/data/ycfeng/Frontier/.worktrees/b19-v4-heldout`. Changes stay within the assigned predictor/test paths and this `a6/` directory. Private caches, test outputs and logs stay in `/data/ycfeng/tmp/codex-implement/70321f85-dbfa-4760-af88-66f6131e804b/b19`.

## Changes and scope

- `frontier/execution_time_predictor/dsv4_predictor.py:259,330`: fitted models now carry `_frontier_dsv4_workload_group_ranges`, keyed by measured `padded_rows`. Runtime first selects the row group, then checks other feature/geometry ranges from that group. Dummy groups admit measured row counts without geometry.
- `tests/unit/test_dsv4_predictor.py:137,180-233`: replaces the unmeasured-size interpolation expectation and tests cross-row geometry leakage, retained own-group values, interpolation inside a measured group and dummy coverage.
- `tests/analysis/v4_attention_heldout.py` (new): canonical complete-unit split, the real PD manager trainer, runtime scope selection/prediction, joint versus old admission, metrics, provenance and deterministic JSON/Markdown output.
- `tests/unit/test_v4_attention_heldout.py` (new): six fixture tests for splitting, equal geometries, rejection accounting, legally zero selection and admitted-row statistics.
- `a6/result.json`, `a6/summary.md`, `a6/families.md`, `a6/verification.md` (new): detailed audit, scope summary, family classification/collective prerequisite, and this verification record.

Estimator fitting, feature schema, model keys, exact lookup, cache names, composition, tables and runtime scope selection are preserved. Cache identity does not change: training reattaches coverage after loading or fitting each estimator. Both training paths and warm-cache exact identity pass.

The initial worktree was clean. Deleted code is limited to superseded global-bound metadata/checks and the old interpolation-test expectation (predictor diff: +26/-7; existing test diff: +51/-6). No files were deleted.

## Method and provenance

Complete units are `(graph_runtime_mode, phase, padded_rows, dispatch_num_requests, prefill_seq_lens, prefill_kv_cache_sizes, decode_kv_cache_sizes)`. Request-list JSON formatting is canonicalized. Case IDs and layer rows do not split equal geometries. sklearn `KFold` supplies the seeded outer split; no estimator, scorer, grid or inner CV is reimplemented.

Committed input: `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`.

- SHA-256: `34c4f3a86822994b126d195a54fce934e208fd3ba9dff2cc4fddb560c129321f`.
- Source commit: `3831c459862332092ce82d88a6bd2d78c6750ef5`; dirty paths: ` M frontier/execution_time_predictor/dsv4_predictor.py`, ` M tests/unit/test_dsv4_predictor.py`, `?? tests/analysis/v4_attention_heldout.py`, `?? tests/unit/test_v4_attention_heldout.py`.
- 4,945 layer rows, 989 cases and 989 units; every unit has five rows. Mode counts: PIECEWISE 3,225, FULL 1,710, NONE 10. Phase counts: decode 3,360, first_tail 1,515, dummy 65, prefill 5.
- K=5, input seed=1234; held-out units per fold: 198,198,198,198,197; held-out rows: 990,990,990,990,985. Training/test unit overlap is zero in every fold.
- Runtime trainer: `Simulator -> train_dsv4_for_manager -> ExecutionTimePredictionModelManager._train_single_model` (`simulator.py:155`, `dsv4_training.py:48`, `prediction_family_trainers.py:1535`). `h4a4_driver.build_config` loads the pinned P2-32 input after removing the diagnostic table override.
- Direct estimator/CV config fields: `num_estimators=[250]`, `max_depth=[8]`, `min_samples_split=[2]`, `k_fold_cv_splits=2`, `num_training_job_threads=2`. RF random_state=0 is the existing trainer setting. All cache-identity fields and values are listed in `result.json.trainer.config_fields` and `cache_identity_fields`; caller initialization reads `kernel_gap_input_file` for paired-count training.
- The A10 fields `decode_cuda_graph_mode`, `replica_config_network_device`, `analytical_kv_cache_transfer_config_nixl_read_completion`, decode `max_concurrent_batches`, and `cudagraph_capture_sizes` are absent from the trainer/cache config fields (recorded intersection: `[]`).
- Each fold caches under `scratch/fold_0` through `fold_4`; no run cache is used. PREFILL and DECODE fit/cache configurations match; the DECODE trainer context is used. Python 3.10.6; sklearn 1.7.2, pandas 2.3.3, numpy 2.2.6.

## A6 observations

Kernel error is `(predicted - measured) / measured`; gates apply to median absolute error <=5% and p90 <=10%. Signed median errors, signed-gap absolute errors in microseconds, combined-time relative errors and zero denominators are retained in JSON. Rejected rows are excluded from error statistics, never assigned a substitute.

16 scopes PASS, three FAIL, two are LEGALLY_ZERO and one is NOT_EVALUABLE. The per-scope table follows (kernel statistics cover evaluable rows only).

| Scope | Held out | Evaluable | Joint rejected | Old rejected | Median | p90 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dsv4_fused_qkv_projection | 4945 | 4825 | 120 | 35 | 1.306% | 5.220% | PASS |
| dsv4_fused_qkv_rmsnorm | 4945 | 4825 | 120 | 35 | 3.628% | 8.931% | PASS |
| dsv4_q_projection | 4945 | 4825 | 120 | 35 | 1.417% | 8.826% | PASS |
| dsv4_query_padding | 0 | 0 | 0 | 0 | n/a | n/a | LEGALLY_ZERO |
| dsv4_attention_output_zero | 0 | 0 | 0 | 0 | n/a | n/a | LEGALLY_ZERO |
| dsv4_output_inv_rope_quant | 4945 | 4825 | 120 | 35 | 3.078% | 9.238% | PASS |
| dsv4_grouped_output_projection | 4945 | 4825 | 120 | 35 | 0.652% | 1.796% | PASS |
| dsv4_output_projection | 4945 | 4825 | 120 | 35 | 1.048% | 2.735% | PASS |
| dsv4_compressor_projection | 3956 | 3860 | 96 | 28 | 1.257% | 3.396% | PASS |
| dsv4_index_weights_projection | 1978 | 1930 | 48 | 14 | 1.799% | 5.382% | PASS |
| dsv4_index_compressor_projection | 1978 | 1930 | 48 | 14 | 1.335% | 4.711% | PASS |
| dsv4_index_query_projection | 1976 | 1930 | 46 | 12 | 1.455% | 4.448% | PASS |
| dsv4_index_fused_query_prepare | 1976 | 1930 | 46 | 12 | 4.593% | 12.207% | FAIL |
| dsv4_index_short_context_fill | 2 | 0 | 2 | 2 | n/a | n/a | NOT_EVALUABLE |
| dsv4_fused_q_kv_prepare | 4945 | 4815 | 130 | 35 | 2.733% | 6.936% | PASS |
| dsv4_compressor_save_partial_states | 3956 | 3852 | 104 | 28 | 3.854% | 9.437% | PASS |
| dsv4_index_compressor_save_partial_states | 1978 | 1926 | 52 | 14 | 3.673% | 8.465% | PASS |
| dsv4_compressor_fused_compress_store | 3956 | 3846 | 110 | 28 | 2.968% | 11.518% | FAIL |
| dsv4_index_compressor_fused_compress_store | 1978 | 1926 | 52 | 14 | 3.159% | 17.476% | FAIL |
| dsv4_index_scores | 1974 | 1878 | 96 | 16 | 1.674% | 5.093% | PASS |
| dsv4_index_topk | 1976 | 1878 | 98 | 16 | 2.052% | 5.810% | PASS |
| dsv4_sparse_attention | 4945 | 4787 | 158 | 35 | 0.682% | 1.997% | PASS |

| Admission | Scope-row requests | Evaluable | Rejected | Exact lookup | Interpolated |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint | 67244 | 65438 | 1806 | 61746 | 3692 |
| per_column | 67244 | 66766 | 478 | 62280 | 4486 |

Joint rejects 1,328 more scope-row requests. Shared admission has 65,438 predictions and zero value changes; every per-scope and per-key shared statistic is identical. Count conservation, row partition and zero fold overlap were checked against the emitted JSON.

| Rejection column | Joint | Old per-column |
| --- | ---: | ---: |
| max_context_len | 152 | 0 |
| min_context_len | 382 | 0 |
| model_key | 130 | 130 |
| num_compressed_writes | 6 | 0 |
| num_query_tokens | 44 | 0 |
| num_sequences | 60 | 0 |
| padded_rows | 882 | 340 |
| sum_compressed_key_pairs | 116 | 8 |
| sum_selected_key_pairs | 34 | 0 |

11,876 legally absent scope contributions were checked against the real workload owner; runtime selection mismatches: **0 (PASS)**. Query padding and attention-output zero are absent from every admitted table row. Index short-context fill has only two measured rows, both in an eager dummy unit whose model key disappears from its training fold; its accuracy is NOT_EVALUABLE.

Per-key verdicts: 103 PASS, 17 FAIL, 157 NOT_EVALUABLE (277 keys). All failing evaluable keys follow. The worst q_projection key uses exact lookup on all 668 predictions, so its failure is not caused by interpolation. Passing pooled scopes do not erase failing keys.

| Scope | Variant | Mode | Phase | Evaluable | Median | p90 | Interpolated |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| dsv4_q_projection | csa | FULL | decode | 668 | 9.229% | 26.215% | 0 |
| dsv4_index_compressor_fused_compress_store | csa | FULL | decode | 668 | 11.078% | 25.535% | 12 |
| dsv4_compressor_fused_compress_store | csa | PIECEWISE | first_tail | 594 | 3.793% | 16.321% | 30 |
| dsv4_compressor_fused_compress_store | csa | PIECEWISE | decode | 664 | 4.518% | 16.288% | 46 |
| dsv4_compressor_fused_compress_store | csa | FULL | decode | 668 | 4.414% | 15.557% | 12 |
| dsv4_index_fused_query_prepare | csa | PIECEWISE | decode | 664 | 5.568% | 13.314% | 0 |
| dsv4_index_fused_query_prepare | csa | PIECEWISE | first_tail | 598 | 4.712% | 12.742% | 0 |
| dsv4_output_inv_rope_quant | csa | PIECEWISE | first_tail | 598 | 3.947% | 11.859% | 0 |
| dsv4_output_inv_rope_quant | uncompressed | PIECEWISE | decode | 332 | 5.003% | 11.478% | 0 |
| dsv4_output_inv_rope_quant | csa | PIECEWISE | decode | 664 | 3.835% | 10.985% | 0 |
| dsv4_compressor_fused_compress_store | hca | FULL | decode | 668 | 3.574% | 10.767% | 16 |
| dsv4_output_inv_rope_quant | hca | PIECEWISE | decode | 664 | 3.236% | 10.749% | 0 |
| dsv4_output_inv_rope_quant | uncompressed | PIECEWISE | first_tail | 299 | 4.936% | 10.635% | 0 |
| dsv4_compressor_save_partial_states | csa | PIECEWISE | decode | 664 | 3.965% | 10.601% | 0 |
| dsv4_output_inv_rope_quant | hca | PIECEWISE | first_tail | 598 | 3.338% | 10.503% | 0 |
| dsv4_index_fused_query_prepare | csa | FULL | decode | 668 | 3.838% | 10.405% | 0 |
| dsv4_compressor_save_partial_states | csa | PIECEWISE | first_tail | 594 | 4.171% | 10.316% | 0 |

Fold rejections are audit coverage losses, not evidence that a full-table runtime rejects the same coordinates. Examples are retained per key/column; no executed regression failed due to A5 admission.

## Executed checks

All commands run from the working root with:

```bash
ulimit -v 31457280
export PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export TMPDIR=/data/ycfeng/tmp/codex-implement/70321f85-dbfa-4760-af88-66f6131e804b/b19
```

**Focused A5/tool check — PASS: 53 passed in 8.41s.**

```bash
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_dsv4_predictor.py tests/unit/test_v4_attention_heldout.py \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/codex-implement/70321f85-dbfa-4760-af88-66f6131e804b/b19/pytest_final_focused2 \
  --tb=short --show-capture=no
```

The initial ordinary seven-module regression invocation gave 313 passed and three failed in 175.36s. One file was missing and two defaults attempted `outputs/metrics` on the read-only source mount.

**Output-routing check — PASS: two previously blocked tests pass in 181.37s.**

```bash
timeout 240s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/70321f85-dbfa-4760-af88-66f6131e804b/b19/run_output_redirected_checks.py
```

This harness runs the unchanged warm-cache/both-trainer test and EP source-ID expansion test. It wraps only `MetricsConfig.__init__` to replace the default `outputs/metrics` with `scratch/pytest_metrics`, then calls the original initializer (including validation/mkdir). No repository test or config code is patched on disk.

**Final required modules plus new tool tests — 321 PASS, one missing prerequisite.**

```bash
timeout 600s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/70321f85-dbfa-4760-af88-66f6131e804b/b19/run_required_suite.py
```

This uses the same output-only wrapper and invokes `pytest.main` with these unchanged modules:

```text
tests/unit/test_dsv4_predictor.py
tests/unit/test_v4_attention_coverage.py
tests/unit/test_dsv4_attention_runtime.py
tests/unit/test_dsv4_operator_consumer.py
tests/unit/test_dsv4_structural_pd.py
tests/unit/test_dsv4_structural_execution.py
tests/unit/test_dsv4_source_execution.py
tests/unit/test_v4_attention_heldout.py
-q -p no:cacheprovider --tb=short --show-capture=no
--basetemp=/data/ycfeng/tmp/codex-implement/70321f85-dbfa-4760-af88-66f6131e804b/b19/pytest_required_suite
```

Observed final pytest result:

```text
=========================== short test summary info ============================
FAILED tests/unit/test_v4_attention_coverage.py::test_package03_plan_json_frozen - FileNotFoundError: [Errno 2] No such file or directory: '/data/ycfeng/Frontier/.worktrees/b19-v4-heldout/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json'
1 failed, 321 passed, 3 warnings in 256.66s (0:04:16)
```

FAIL/NOT_AVAILABLE: `test_v4_attention_coverage.py::test_package03_plan_json_frozen` needs `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json`. `git show HEAD:<path>` confirmed it is absent from HEAD. No coordinate was priced in this failure; do not regenerate a supposedly frozen reference inside this assignment.

**Attention audit runs 1 and 2 — executed successfully; empirical A6 FAIL; reproducibility PASS.**

Both invocations use exactly:

```bash
timeout 3600s /data/ycfeng/envs/frontier-py310/bin/python \
  tests/analysis/v4_attention_heldout.py \
  --table data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv \
  --folds 5 --seed 1234 \
  --scratch-dir /data/ycfeng/tmp/codex-implement/70321f85-dbfa-4760-af88-66f6131e804b/b19 \
  --output-dir task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6
```

Run logs: `scratch/heldout_run1.log`, `scratch/heldout_run2.log`. `result.json` and `summary.md` are byte-for-byte identical between runs. Differing JSON fields: **none**; timestamps and wall times are intentionally omitted. Result bytes: 2575020. Result SHA-256: `86e1ffe7795f068d421abf6b6cd7c5e229c659856d492ff86c2c24b8ad8642f2`. First-run copies and comparison receipt are in scratch.

**Diff hygiene — PASS:** `timeout 10s git diff --check`.

**Line counts:** predictor 357, existing predictor test 327, tool 512, new tool test 149. No touched Python module exceeds 1,000 or 2,000 lines.

## Remaining work and review

Family classification and source lines are in `families.md` and `result.json.other_families`. Standalone operators, MoE rank classes, runner rows and unequal EP combine are exact only; non-attention gaps are measured segment/TP constants times exact activity counts. Attention signed gaps are included in this audit. Uniform measured collectives interpolate.

**NOT_RUN — uniform collective complete-unit audit.** Its H200 table lacks all seven assigned unit columns and its runtime uses log-log curves, not the PD sklearn trainer. This contradicts applying decision 2 unchanged. No replacement unit or estimator was chosen. Table SHA-256: `f4d52911770cc39939dddc0fa98ce30b530b1cd60ea2bd94070d293e200ca559`. Claude should define/authorize a collective unit using ordered DP/SP tokens and payload geometry across modes/repetitions, and an audit through the existing backend loader/predictor. Existing LOO diagnostics are not the requested complete-unit split.

Claude should restore the retained package_03 plan, rerun its frozen-plan test, and review the three failed scopes, 17 failed keys and 157 unevaluable keys before deciding further work. Features, keys, estimators and tables were not tuned to obtain a pass.

Inspect the row-group metadata/error messages, unchanged cache/model/lookup path, unit canonicalization and split, real workload scope selection, legacy-admission comparison, and denominator handling. Rerun the focused pytest command and the audit command; the scratch output harness is also available for the full required suite. `task_memory/*` is ignored by `.gitignore:171`; preserve these artifacts explicitly under the lead's commit/task-record workflow.

Procedural deviation: initial instruction discovery mistakenly ran `find ..` and listed AGENTS.md filenames in sibling worktrees. No sibling file contents were read; subsequent discovery and code inspection were confined to this worktree. Output-path redirection was limited to process-local test setup in approved scratch; it does not modify production behavior.

## Lead review addendum (2026-10-29)

The Codex text above is kept as written. These points supersede it.

1. **Collective held-out.** The lead set the unit (requirements item 36, D-A6-COLLECTIVE-UNIT) and changed `tests/analysis/v4_attention_heldout.py`: `collective_heldout` reads the existing `_Curve.describe` leave-one-out of each uniform curve of the P2 DECODE backend, built as `Cluster._initialize_cc_backend` builds it, and applies the A6 thresholds per curve. The `BLOCKED_UNIT_CONTRACT` fields are removed. New test `test_collective_points_are_left_out_one_at_a_time` in `tests/unit/test_v4_attention_heldout.py` (a power law passes, a bent interior point fails with the expected error, a two-point curve is not evaluable). Result: 7 of 8 curves FAIL, `reduce_scatterv, ep, 16` passes; the table is in [families.md](families.md) and `summary.md`.
2. **Regenerated outputs.** `result.json` (2,578,014 bytes, SHA-256 `02ec668557f5db166629dbbc04320917e3223f6ec66712a64be63e2d0dedff9d`) and `summary.md` (SHA-256 `38159c0cce00f0226f6cf7f9ac2e940567888825cc6f5cf0b7188367e765d929`) were written by the corrected tool (`/data/ycfeng/tmp/lead_b19/run_a6_warm.sh`, 529 s on the fold caches of the lead's cold run). Against the cold run of the unchanged Codex tool, only the top-level key `other_families` differs; the `attention` section is identical.
3. **Reproduction.** The lead's cold run of the Codex tool in a separate scratch directory (`/data/ycfeng/tmp/lead_b19/run_a6.sh`, 3,031 s) reproduced the Codex `result.json` and `summary.md` byte for byte.
4. **Frozen plan.** `FULL_AND_PIECEWISE.json` of package 03 was ignored by `.gitignore` and existed only in the main worktree. It is committed on main (`c4df84e3`), so the frozen-plan test runs after the merge.
5. **The three failing attention scopes.** For `index_fused_query_prepare`, `compressor_fused_compress_store` and `index_compressor_fused_compress_store`, 95-100% of the held-out predictions (1,930 of 1,930, 3,696 of 3,846 and 1,838 of 1,926) are exact lookups of a feature tuple already present in training, so the error is the spread between measurements with equal features, not a fit error. Signed medians are 0.0-0.4%. The two measured layer groups of one forward (`plans_tile_schedule` true and false, equal geometry) differ by a p90 absolute deviation from their median of 7.5%, 7.6% and 17.9% for these kernels of 2-20 µs. The compressor time has two levels, about 2-3 µs when no row completes a compression group and about 19-20 µs when one or more do; `num_compressed_writes` already separates them. Scripts: `/data/ycfeng/tmp/lead_b19/variance.py`, `boundary2.py`, `repeat_noise.py`, `repeat_noise2.py`. The verdicts stay FAIL; the thresholds are not changed.
6. **Line counts after the lead edit:** predictor 357, predictor test 327, tool 532, tool test 163.

