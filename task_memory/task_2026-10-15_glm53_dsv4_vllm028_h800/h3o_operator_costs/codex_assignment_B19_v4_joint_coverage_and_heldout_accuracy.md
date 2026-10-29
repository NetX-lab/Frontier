# Task: joint V4 attention coverage (A5) and group-held-out accuracy (A6) (stage 4 assignment B19, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created at the start of stage 4, in parallel with B18. Covers the A5 admission per workload group, the A6 held-out tool and its result on the committed H200 tables, and the classification of the other V4 predictor families as exact-only or interpolating. |

## Goal

Two design gates of [design.md](design.md) §1 are open for the V4 attention predictor (`frontier/execution_time_predictor/dsv4_predictor.py`):

| Gate | Design text | Current code |
| --- | --- | --- |
| A5 | "Coverage is admitted jointly per workload group (mode, planner role, phase, layer variant, rows), not by independent min/max bounds." Codex review round 1, F007: independent bounds "accept unmeasured joint phase/geometry combinations" (`codex-review-9c8b2435-…/review.md:177-181`) | `train_dsv4_attention_models` keys a model by (scope, variant, graph mode, phase) and, for `dsv4_sparse_attention` only, the planner flag (`dsv4_model_key`). `_frontier_dsv4_bounds` holds one (min, max) per feature and geometry column over all rows of the key. `scope_times` admits a query when every column lies inside its own range, so a `padded_rows` value between two measured sizes, or a context range measured only at another row count, is admitted. This is the B7 decision ([codex_assignment_B7_dsv4_attention_predictor.md](codex_assignment_B7_dsv4_attention_predictor.md) lines 202-209), which deviates from A5 |
| A6 | "Held out by complete workload group: per scope, median relative error ≤ 5% and p90 ≤ 10%. Scopes that are legally zero or no-op are checked for exact zero instead." Review F007: a random row holdout "may mask categorical merges and neighboring-context leakage" | No held-out evaluation exists |

The committed table (`data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`) has 4,945 rows from 989 G1h cases, 5 rows per case (CSA and HCA as planner and reuse layers, and the uncompressed layer). Phases: decode 3,360, first_tail 1,515, dummy 65, prefill 5. Modes: PIECEWISE 3,225, FULL 1,710, NONE 10. `padded_rows` takes the values 1, 2, 4, 8, 16, 24 and 6144. Many scopes use `padded_rows` as their only feature (`frontier/attention/dsv4_features.py:19-50`), so at a measured size their runtime answer is the exact lookup; the attention, indexer and compressor scopes also use per-request sums, which the runtime meets at unmeasured values.

B19 delivers:

1. Joint admission per workload group in `dsv4_predictor.py`, with tests.
2. The held-out tool `tests/analysis/v4_attention_heldout.py`, with a unit test.
3. The A6 result on the committed tables, under the training configuration of the A10 runs.
4. The classification of every other V4 predictor family that a P2 run reads.

## Approach and decisions

### Read first

- [design.md](design.md) §1 (A5, A6), §3.6 (the V4 key and features).
- `frontier/execution_time_predictor/dsv4_predictor.py`, `frontier/attention/dsv4_features.py`, `frontier/execution_time_predictor/dsv4_training.py` (both training paths: `train_dsv4_for_predictor` with `SklearnExecutionTimePredictor._train_model`, and `train_dsv4_for_manager` with the manager's `_train_single_model`), `sklearn_execution_time_predictor.py:3158-3260` (`_train_model`, its cache key `_get_model_hash`, the exact lookup) and the V4 registrations near `:3579` and `:4065`.
- `frontier/execution_time_predictor/deepseek_v4.py` (how the runtime selects the scopes of a layer and calls `scope_times`), `dsv4_operator_predictor.py`, `dsv4_runner.py`.
- `tests/unit/test_dsv4_predictor.py`, `tests/unit/test_v4_attention_coverage.py`.
- The H4a driver `task_memory/.../h4a_p0_diagnostic/h4a4_driver.py` (`CASES`, `build_config`).

### Decisions

1. **A5 admission per workload group.**
   - The admission unit is the A5 workload group: the model key (which already holds mode, phase, layer variant and, for the planner scope, the planner role) plus `padded_rows`. A query whose `padded_rows` is not a measured value of its key raises `DeepSeekV4CoverageError` naming the operator, the key and the value.
   - Inside its group, every other feature and geometry column must lie inside that group's own measured (min, max); the error names the column, the value, the range and the group.
   - Dummy keys keep `padded_rows` as their only column and admit measured sizes only.
   - Store the per-group ranges on the fitted model in place of `_frontier_dsv4_bounds` (rename the attribute after what it holds). The fitted models, the exact lookup and the cache names do not change; if the cache identity must change, say why.
   - Tests: a `padded_rows` between two measured sizes raises; a context range measured only at another row count raises; a query inside its own group is admitted and gives the same value as before; the existing collision tests stay green.
   - If the stricter admission rejects a coordinate that a committed test or a structural suite prices today, report the coordinate and its source; do not widen the admission to keep the test.
2. **Held-out unit.**
   - The held-out unit is a complete workload: all rows of every case that share the same (graph mode, phase, `padded_rows`, `dispatch_num_requests`, `prefill_seq_lens`, `prefill_kv_cache_sizes`, `decode_kv_cache_sizes`). Each case's five layer rows always stay together, and two cases with the same request geometry are never split between training and test, so a held-out coordinate never appears in training. Report the number of units and their sizes.
   - Split the units into K folds with a seeded shuffle; K and the seed are required command-line arguments. The lead runs K = 5 with the seed of the P2 input config (`seed`).
   - Training in each fold calls `train_dsv4_attention_models` on the training rows with the trainer of the path the A10 runs use (find which of the two paths a PD P2 run takes and name it), built from the P2-32 input config through `h4a4_driver.build_config` with the diagnostic table override removed (as the lead's probe does, `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/probe_driver.py`). List the predictor-config fields the trainer reads and confirm that the A10 operator-path fields (`decode_cuda_graph_mode`, `replica_config_network_device`, `analytical_kv_cache_transfer_config_nixl_read_completion`, the decode `max_concurrent_batches`, `cudagraph_capture_sizes`) are not among them. No second estimator, grid, scorer or k-fold implementation.
   - The trainer caches by model name and a data hash. Point every fold's cache at a scratch directory under `/data/ycfeng/tmp/codex-implement/<request-id>/b19/`; never at a run's cache directory.
   - Prediction for each held-out row goes through `DeepSeekV4AttentionPredictor.scope_times` built from the fold's models, with the row's features from the loader (`load_dsv4_attention_dataframe`), so admission and lookup are the runtime's.
3. **Metrics.**
   - Per scope and per model key, over held-out rows: the relative error of the kernel time `(predicted - measured) / measured`, with count, median of the absolute value, p90 of the absolute value and the signed median. The gate is A6 on the kernel time: median ≤ 5 % and p90 ≤ 10 % for every scope, also listed per key.
   - The signed gap is reported as absolute error in microseconds (median and p90 of the absolute value), and the scope's combined time plus gap as relative error. These two are reported, not gated, because the gap's denominator can be zero or negative.
   - Scopes that are absent from a row's `operator_scopes` are legally zero: check that the runtime scope selection for that row's variant, planner role, phase and mode does not ask for them, so their contribution is exactly zero. Report the count checked and any mismatch.
   - A held-out row that the admission of decision 1 rejects is "not evaluable": count it per key with the column that rejected it, and keep it out of the error statistics. Report the same count under the old per-column bounds, so the effect of decision 1 is visible.
   - Also report the errors without decision 1 (per-column bounds) for the rows both admit, to show that decision 1 changes admission only.
4. **Other V4 predictor families.** For every V4 model family that a P2 PD run trains or reads (operators in `dsv4_operator_predictor.py`, MoE rank classes, runner preparation, sampling and block rows, collectives through the measured backend, kernel gaps), state whether runtime prediction is exact lookup only (a typed error outside measured coordinates) or can interpolate, with the code line. Run the held-out evaluation of decision 2 for every family that can interpolate, with a unit defined the same way; families that are exact only are covered by A5 and need no held-out run.
5. **Outputs.**
   - The tool writes `result.json` (per scope and key: counts, statistics, not-evaluable counts, verdicts; the fold assignment by unit; the input table SHA-256; the source commit and dirty paths; the trainer path and its config fields) and a short `summary.md` table to an output directory given on the command line.
   - Run it on the committed table into `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6/` with K = 5 and the input seed. A second run with the same arguments gives the same `result.json` apart from timestamps; say which fields differ.

## Out of scope

- Any change to the tables, the features, the model key, the estimator, the trainer, the composition or the runtime scope selection.
- The scheduler and graph-mode classifier (B18 owns `vllm_v1_iteration_policy.py`), the A10 tools and runs, A7, A8, A9, A11.
- Any GPU, platform or network action.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b19-v4-heldout` (Git worktree on branch `b19-v4-heldout`, created by the lead from the commit that adds this assignment). B18 runs at the same time in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` on disjoint paths; the separate worktree keeps each run's imports apart. Do not read or write the other worktree. The lead merges the result.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md).
- **Writable paths:**
  - `frontier/execution_time_predictor/dsv4_predictor.py`;
  - `tests/unit/test_dsv4_predictor.py`, `tests/unit/test_v4_attention_coverage.py`, `tests/unit/test_v4_attention_heldout.py` (new);
  - `tests/analysis/v4_attention_heldout.py` (new);
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6/` (new).
  - Other test files only if decision 1 changes their expectations; list each with the reason. B18 owns `frontier/scheduler/`, `tests/unit/test_dsv4_dp_sync.py`, its new classifier and `a10/` test files and `h3o_operator_costs/a10/`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b19/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No moves. Delete only code that this change supersedes, and list each deletion.
  - Python 3.10. Keep every touched module at or below 2,000 lines and report the line counts of touched modules above 1,000 lines.
  - No magic literals in code paths: K, the seed and the gate thresholds of A6 are arguments or named constants that cite design A6. No guards or defaults for unreachable states, and no fallback that hides a missing row.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. Training five folds may take tens of minutes; use the trainer's own thread setting.
  - Do not search under `/data/ycfeng/tmp/glm53_dsv4/` recursively.
  - Search with `/usr/bin/grep` and quote `--include='*.py'`.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Run pytest from the repository root.

## Acceptance

1. **A5.** The tests of decision 1 pass; `test_dsv4_predictor.py`, `test_v4_attention_coverage.py`, `test_dsv4_attention_runtime.py`, `test_dsv4_operator_consumer.py`, `test_dsv4_structural_pd.py`, `test_dsv4_structural_execution.py` and `test_dsv4_source_execution.py` pass, or each failure is reported with its coordinate as in decision 1.
2. **Tool.** `test_v4_attention_heldout.py` on a small fixture: no held-out unit appears in its fold's training rows; equal request geometries land in one unit; a rejected row is counted as not evaluable; a legally zero scope is never asked for.
3. **A6 result.** The run of decision 5 with the per-scope and per-key table, the not-evaluable counts under both admissions, the legally-zero check, and the verdict per scope against median ≤ 5 % and p90 ≤ 10 %. A failing scope is a result to report, not a reason to change the features, the key or the estimator.
4. **Families.** The list of decision 4 with code lines, and the held-out result of every interpolating family.
5. **Records.** Line counts; the deleted code, if any.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- The results of Acceptance 1-5, with the commands, the expected and actual values, and the counts.
- The A6 table: per scope, the held-out count, median and p90 of the time error, the verdict, and the worst keys.
- Scratch paths, deviations and their reasons, and limits.
- If the code, the tables or the design contradict a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Continue with the parts that do not depend on it.
