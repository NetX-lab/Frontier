# Task: the V4 operator consumer on the H200 tables (stage 3 assignment B16, file entries 1, 3, 5, 6, 7 and 10, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: Codex assignment B16 of stage 3 ([design.md](design.md) §3.1-§3.6 and §6 entries 1, 3, 5, 6, 7, 10). |

## Goal

The stage 2 producers measured the H200 V4 operators. No simulator code reads those rows yet. A non-dummy V4 run raises `DeepSeekV4CoverageError` at `sklearn_moe_execution_time_predictor.py:108-114`, and dummy mode prices every V4 operator with `predictor._dummy_execution_time`.

B16 delivers, all on CPU:

1. **The H200 V4 tables** in `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/`, built by a promotion script from the accepted GPU outputs. Each input file is recorded with its SHA-256.
2. **One V4 training function** for the V4 attention, operator, MoE and kernel-count models. Both training paths call it: the independent path (`_train_models`) and the shared `ExecutionTimePredictionModelManager`.
3. **The trained V4 pricing path.** Every V4 operator except the four runner operators is priced from the trained models. The attention phase uses the composed stream schedule (`compose_attention_phase`). Host time is the explicit zero.
4. **The V4 kernel gaps** by segment class (§3.3), including the V4 EP lane path.

After B16, a non-dummy V4 run reads no `_dummy_execution_time` (gate A1). The four runner operators (`dsv4_input_preparation`, `kv_cache_block_zero`, `dsv4_block_table_write`, `dsv4_sampling`) raise the typed coverage error, naming the operator, until B17 adds their rows. B17 also opens the PREFILL and unified DECODE admission (entry 4).

## Approach and decisions

### Existing mechanism (read first)

- **The V4 attention predictor:** `frontier/execution_time_predictor/dsv4_predictor.py`.
  - The loader is `load_dsv4_attention_dataframe` (KERNEL_ONLY only). `dsv4_input_path` maps `attention_kernel_only.csv` to `dsv4_attention_replay_kernel_only.csv`.
  - The trainer is `train_dsv4_attention_models(df, train)`. It fits a time model and a gap model per `dsv4_model_key`.
  - `DeepSeekV4AttentionPredictor.scope_times` returns `DeepSeekV4ScopeTimes(cost_ms, gap_ms)`.
  - Its registrations in `sklearn_execution_time_predictor.py`:
    - line 1437, the loader;
    - line 3568, the trainer;
    - line 4050, `{"dsv4_attention": DeepSeekV4AttentionPredictor(self._models)}`;
    - line 7607, the attention branch.
- **The V4 pricing functions** in `frontier/execution_time_predictor/deepseek_v4.py`:
  - `predict_attention_operator_times`;
  - `predict_source_execution`, whose `local_scope` raises in non-dummy mode;
  - `predict_structural_execution`, which prices gating ×2 and grouped GEMM from `base`, sets the CPU fields at lines 407-421 to `base if include_stage_owned`, and calls `serial_attention_phase` at line 455;
  - `predict_step_cost_execution` (diagnostic only, ADR 0002, unchanged);
  - `serving_context`;
  - `read_isolated_scope` (lines 652-776) and `_read_isolated_component`.
- **The phase table** `timing_operator_times` (`deepseek_v4.py:17-63`). It has no phase yet for `dsv4_input_preparation`, `dsv4_block_table_write` or `dsv4_sampling` (B17).
- **The composition:** `frontier/execution_time_predictor/dsv4_composition.py`.
  - `compose_attention_phase(scope_ms, scope_gap_ms, *, variant, graph_runtime_mode, rows, multi_stream_gemm_token_threshold, indexer_short_context)` has no caller yet.
  - `serial_attention_phase` serves dummy mode and the step-cost path (design §6 implementation notes; keep those two on it).
  - `segment_class(mode, name)` classifies the attention scopes.
  - `predict_attention_operator_times` already sets `timing._dsv4_scope_gap_ms` and `timing._dsv4_composition_inputs`.
- **Consumers of `_dsv4_attention_schedule`:** `frontier/entities/execution_time.py:1138` (returns `elapsed_ms`) and `frontier/metrics/metrics_store.py:1184` (the trace offsets).
- **The V4 FFN paths:**
  - `sklearn_moe_execution_time_predictor.py:95-118` (`_predict_dsv4_structural_execution`) and `1380-1395`;
  - `frontier/scheduler/utils/expert_parallel.py:129-139`, the V4 EP lane path. It calls `_predict_dsv4_structural_execution` directly, not through `_predict_layer_with_kernel_gap`.
- **The two training paths:**
  - **Independent:** MONOLITHIC runs train through `_train_models` (`sklearn_execution_time_predictor.py:3834`, overridden in `moe_dataset_training.py:476` to add `_train_moe_models`).
  - **Shared manager:** PDD and GDN runs use `ExecutionTimePredictionModelManager` (`simulator.py:155-163`). It trains through `_train_all_required_models` (`shared_prediction_model_manager.py:366-520`) and `prediction_family_trainers.py`.
    - It has a DSA branch at lines 806-816, which calls `functools.partial(self._train_single_model, execution_time_predictor_config=..., training_context=...)`.
    - It has no V4 branch. It skips V4 only when a step-cost table is bound (line 382).
- **Measurement type:**
  - PREFILL gets the event type in both copies of `_is_kernel_only_measurement_enabled_for_cluster` (`sklearn_execution_time_predictor.py:880`, `shared_prediction_model_manager.py:224`) and in `_get_measurement_types_for_cluster` (line 246).
  - The selection methods are at `sklearn_execution_time_predictor.py:880-1005`: `_get_default_measurement_type_for_cluster`, `_should_enable_measurement_family`, `_select_measurement_type_for_batch` and `_activate_measurement_type_for_batch`.
- **Kernel gaps:** `frontier/execution_time_predictor/kernel_gap.py`.
  - `read_dsv4_kernel_gap_ms(path, tp)` exists and reads the four `DSV4_EXECUTION_MODES` rows.
  - `load_kernel_gap_ms` is not V4-aware.
  - `KernelGapPricing._predict_layer_with_kernel_gap` picks "eager" or "cuda_graph" from `_step_measurement_type` and calls `layer.add_kernel_gap(gap, counts.attention, counts.operators)`.
  - `KernelCountTraining._paired_with_kernel_count_model` pairs each time model with its `time_stats.<op>.kernel_count` model.
- **The V4 operator registry:**
  - `frontier/operators/families.py:314-353`, `DEEPSEEK_V4_MHC_FAMILY`, with each operator's TP mode;
  - `frontier/operators/deepseek_v4_axes.py`, `DEEPSEEK_V4_OPERATOR_AXES`: the row axis of each operator (`sp_local_rows`, `descriptor_rows` or `logits_rows`);
  - `frontier/operators/binding.py:262-300`, the mHC variants per layer.
- **The EP rank class:** the class function and the routing algebra in `frontier/moe_ep_workload.py` (B12). The consumer imports them; it does not reimplement them.
- **Profiling paths:** `measurement_input_paths.substitute_input_path` ({DEVICE}, {MODEL}, {NETWORK_DEVICE}).

### Inputs (accepted GPU outputs; read-only, copy before use)

| Table | Source | Accepted in |
| --- | --- | --- |
| V4 attention replay rows | G1h-02 raw records `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/` (3 engines, 989 cases). Plan arguments: `h3o_operator_costs/g1h/package_plan.json` (`plan_args`, `cudagraph_capture_sizes`, `runs`, the pilot cases file) | [../test_report_2026-10-28_b11_g1h_attention_grid.md](../test_report_2026-10-28_b11_g1h_attention_grid.md) and its G1h-02 acceptance in [../progress.md](../progress.md) |
| V4 kernel gaps | The same G1h-02 raw records through the existing reducer `python -m frontier.profiling.kernel_gap.dsv4_replay_gaps` | Values in µs: `cuda_graph` 0.49773414267747085, `cuda_graph_piecewise` 0.6249537992538311, `eager` 0.5446049595929223, `eager_preparation` 1.2208671118771937 |
| Standalone V4 operators (mHC ×4 variants, embedding, mHC head, final norm, LM head) | `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_01/download/G2H_OPERATORS_RECORD_FUNCTION/compute/h200/deepseek-ai/DeepSeek-V4-Flash/linear_op_kernel_only.csv` (36 rows) | [../test_report_2026-10-28_b13_v4_standalone_operators.md](../test_report_2026-10-28_b13_v4_standalone_operators.md) |
| Shared expert (×3 operators) | `.../g2h_01/download/G2H_SHARED_EXPERT_EP16_RECORD_FUNCTION/compute/h200/deepseek-ai/DeepSeek-V4-Flash/linear_op_kernel_only.csv` (5 rows, L {1, 2, 3, 4, 768}) | Same report |
| Routed MoE and gate | `/data/ycfeng/tmp/glm53_dsv4/h200/g3h_01/download/G3H_MOE_RECORD_FUNCTION/ep{16,32}/compute/h200/deepseek-ai/DeepSeek-V4-Flash/moe_kernel_only.csv` | [../test_report_2026-10-28_b12_g3h_moe_rank_classes.md](../test_report_2026-10-28_b12_g3h_moe_rank_classes.md) |

- The EP32 shared-expert rows (`G2H_SHARED_EXPERT_EP32_RECORD_FUNCTION`) have the same identity as the EP16 rows: the shared expert's contract does not depend on the EP size. They stay out of the table, and serve as a repeat measurement (Acceptance 4).
- The CUDA_EVENT tables (`G2H_*_CUDA_EVENT`, `G3H_MOE_CUDA_EVENT`) are for validation only (§3.1). They are not promoted, so no event-type training can pick them up.
- Do not write into the download directories. `_aggregate_command` writes `dsv4_aggregate_summary.json` into its `--raw_dir`, so run it on a copy in the scratch directory.

### Decisions

1. **Promotion script.**
   - A new script `h3o_operator_costs/h200_tables/promote_tables.py` builds the tables from the inputs above, using only the existing producers' code paths:
     - the V4 attention replay CSV from `dsv4_attention_main aggregate`, with `--source_revision a689ed05f462eb269b9dffe640da151e6614a5ef` (the G1h-02 source revision, `g1h/package_plan.json`) and the `--model_config_revision` that `g1h/package_02` recorded;
     - `kernel_gap.csv` from `dsv4_replay_gaps`;
     - `linear_op_kernel_only.csv` as the row union of the G2h operator rows and the EP16 shared-expert rows. Columns that only one input has stay empty in the other's rows;
     - `moe_kernel_only.csv` as the row union of the EP16 and EP32 G3h rows.
   - It writes `h3o_operator_costs/h200_tables/manifest.json` with each input path and SHA-256, each output path, SHA-256 and row count, and the source revisions.
   - Running it twice gives byte-identical outputs.
2. **One V4 training function** (in `dsv4_predictor.py`, or a new torch-free module beside it if `dsv4_predictor.py` would pass 2,000 lines).
   - It trains:
     - the attention models (the existing `train_dsv4_attention_models`);
     - the operator models: one per operator name of the V4 operator tables, keyed by (operator, row axis, selected TP) from the row's typed contract;
     - the MoE models: grouped GEMM keyed by (EP size, M, rank-class triple), and the gate operators keyed by L;
     - the kernel-count model paired with every time model (`_paired_with_kernel_count_model`).
   - It takes the caller's `train` callable, as the DSA branch does. The independent path passes `self._train_model`; the shared manager passes its `functools.partial(self._train_single_model, ...)`.
   - Both paths call this function. For V4 they skip the generic compute, FFN, residual and CPU-overhead trainers. The generic paths of other models are unchanged.
3. **Lookup rule for operators and MoE: measured coordinates only.**
   - Every target coordinate is a measured grid point by construction (C1 §4): L ∈ {1, 2, 3, 4, 768}, W ∈ {1, 2, 4, 8, 16, 24, 6144}, logits rows 1-24, and the (EP, M, class) set of the G3h plan.
   - So the operator and MoE predictors use exact lookup only. A coordinate that was not measured raises `DeepSeekV4CoverageError`, naming the operator and the missing coordinate (gates A5 and A7).
   - The attention predictor keeps its existing rule (exact, else the fitted model within its bounds).
   - Duplicate measurements of one coordinate: the gate is measured in many G3h rows at the same L. Report the trainer's existing reduction for duplicate feature rows and use it; do not add a second reduction. If it has none, stop that part and report.
4. **Loader validation.**
   - The validations of `read_isolated_scope` move into the new loader:
     - the typed contract through `profile.resolve_layer_contract`;
     - the MoE backend identity (`vllm_marlin_mxfp4`, `mxfp4`, MARLIN) and `float_precision="round_trip"`;
     - the linear backends (`FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, except `share_expert_act`);
     - `validate_measured_identity`;
     - a finite positive median, and an integer positive `kernel_count` for KERNEL_ONLY;
     - the balanced routing identity (`validate_balanced_routing`).
   - Its MONOLITHIC-only, NONE-only and prefill-only checks do not move. Those are admission rules, and the attention admission (entry 4, B17) owns them.
   - The loader drops the `graph_mode == eager` check for the operator rows: the standalone rows are measured eagerly and are priced in every mode. The segment gap (decision 7) supplies the difference between modes.
   - Then delete `read_isolated_scope` and `_read_isolated_component`. Move their tests in `tests/unit/test_dsv4_graph_measured.py` to the new loader.
5. **Rows per operator.**
   - `local_scope` and the structural path compute each operator's row count from its axis in `DEEPSEEK_V4_OPERATOR_AXES` and its TP from the contract. Keep the existing row values (summary of `predict_source_execution`). For example, the mHC operators and the shared expert use L, the embedding, mHC head and final norm use the descriptor rows W, and the LM head uses the selected logits rows.
   - The gate uses the lane's L. The grouped GEMM uses the lane's (EP, M, class), computed by the class function of `moe_ep_workload.py` from the lane workload.
   - `moe_shuffling` is not charged. Its kernels run inside the grouped-GEMM envelope (G3h report, line 37). Remove it from the V4 routed phase and its workloads; do not price it as zero.
   - Hash layers 0-2 charge no hash selection, as now.
6. **Composition in the trained path.** The trained path sets `timing._dsv4_attention_schedule` from `compose_attention_phase`, with the trained scope costs, the trained scope gaps and `timing._dsv4_composition_inputs`. Dummy mode and `predict_step_cost_execution` keep `serial_attention_phase`.
7. **Kernel gaps (entry 10).**
   - For V4, `load_kernel_gap_ms` reads the four V4 values with `read_dsv4_kernel_gap_ms` at the attention TP size. A non-dummy V4 run with no `kernel_gap.csv` raises a clear error, because the V4 pricing requires the gaps. Other models are unchanged.
   - Each V4 operator's gap is its kernel count × the gap value of its segment class. One table classifies every V4 operator, kept beside the phase table or `segment_class`:
     - forward operators inside the model forward take `cuda_graph` under FULL, `cuda_graph_piecewise` under PIECEWISE, and `eager` under NONE. The attention scopes use the existing `segment_class`;
     - work outside the model forward (LM head and logits gather now; preparation, zeroing, block-table writes and sampling in B17) takes `eager_preparation` in every mode.
   - The V4 attention phase adds no gap through `add_kernel_gap`: the composed elapsed already includes the scope gaps.
   - The V4 EP lane path in `expert_parallel.py` goes through `_predict_layer_with_kernel_gap`.
   - Collectives keep the existing kernel-count convention of `KernelGapPricing`. Report what it charges for the V4 collectives.
8. **Measurement type.** V4 operator tables are KERNEL_ONLY in every role and graph mode. One function decides this, keyed on the V4 attention family as the existing V4 dispatch is. Both measurement selectors and `_get_measurement_types_for_cluster` use it. The gap's segment class comes from the forward's graph runtime mode (decision 7), not from the measurement type.
9. **Host time (§3.4).**
   - In non-dummy V4 pricing, the CPU stage fields (`schedule_time`, `sampler_e2e_time`, `prepare_inputs_e2e_time`, `process_model_outputs_time`, `ray_comm_time`) are the explicit zero.
   - A V4 run with `skip_cpu_overhead_modeling=False` raises a typed configuration error at predictor construction. Use the configuration error type of the existing V4 guards. Report any existing configuration or test that the error affects.
   - Dummy mode keeps its current values.
10. **The runner operators** (`kv_cache_block_zero` today; preparation, block-table writes and sampling have no phase yet). In non-dummy mode, `kv_cache_block_zero` raises `DeepSeekV4CoverageError` naming the operator. B17 adds the rows and phases.
11. **Registration lines only** in `sklearn_execution_time_predictor.py` (8,626 lines) and `metrics_store.py` (5,744 lines). All logic goes into the V4 modules. Report the line counts of these two files before and after.

## Out of scope

- The runner operators' rows and phases, the PREFILL and unified DECODE admission, the V2 dummy workload (entry 4), `runtime_limitations` (entry 11), the module-size record (entry 12) and the A3 ledger sentinel. These are B17.
- End-to-end replays (A10) and the diagnostic cross-check (A11).
- Any GPU, platform or network action. Changes to producers, to `balanced_routing.py` or to the collective consumer (B15).
- The step-cost path, apart from what decision 6 names.
- The open items of [../progress.md](../progress.md): the embedding mask, the decode standalone durations, the NONE-step eager gaps, the graph/eager pooling, and the dummy-step unowned kernels.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Concurrent work:** the G2h-02 replay job and its acceptance (lead, through /ds4-exec). Do not edit `h3o_operator_costs/g2h_runner/` or `frontier/profiling/`.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md) (§1 gates A1, A5-A7, A9 and A12; §3; §4; §6 entries 1, 3, 5-7, 10 and the implementation notes), [c1_coverage.md](c1_coverage.md) (§4), and the test reports named in the inputs table.
- **Writable paths:**
  - `frontier/execution_time_predictor/`: `dsv4_predictor.py`, `deepseek_v4.py`, `dsv4_composition.py`, `kernel_gap.py`, `sklearn_moe_execution_time_predictor.py`, `moe_dataset_training.py`, `shared_prediction_model_manager.py`, `prediction_family_trainers.py`, new modules beside them; `sklearn_execution_time_predictor.py` (registration and routing lines only);
  - `frontier/scheduler/utils/expert_parallel.py`;
  - `frontier/metrics/metrics_store.py` (only if decision 6 needs it);
  - `frontier/profiling/kernel_gap/schema.py` (only constants, if decision 7 needs one);
  - `tests/unit/`;
  - `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/` (new, through the promotion script);
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/` (new).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b16/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No moves. Delete only `read_isolated_scope`, `_read_isolated_component` and code this change supersedes; list each deletion.
  - Python 3.10. Keep the touched modules at or below 2,000 lines, apart from the two files of decision 11.
  - No magic literals in code paths: the backend names, row axes, TP modes and gap values come from the profile, the operator registry and the tables. No guards or defaults for states the code cannot reach, and no fallback that hides a missing row. No second implementation of the class algebra, the composition or the row-axis rule.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
  - Search with `/usr/bin/grep` and quote `--include='*.py'`. Do not search recursively under `/data/ycfeng/tmp/glm53_dsv4/`.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`. Run pytest from the repository root.
  - The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Acceptance

1. **Tables.**
   - The promotion script runs twice with byte-identical outputs.
   - The row counts per table are reported.
   - The attention identities equal the 4,945 identities of `/data/ycfeng/tmp/b13r_accept/after/rows.json`.
   - The four gap values equal the G1h-02 values above to every digit.
   - Every output passes the new loader.
2. **A1, no dummy values.**
   - A test makes any read of `_dummy_execution_time` raise.
   - It builds a MONOLITHIC non-dummy V4 predictor on the promoted tables, with TP8, DP2 and EP16, and prices a decode forward (FULL) and a prefill forward (NONE, with no new cache blocks). The forwards are priced through the stage path, every layer and the endpoints included.
   - The test passes, and no step-cost table is bound.
   - Report the stage components of both forwards.
3. **Exact lookup.**
   - For every measured coordinate of every operator, gate and grouped-GEMM row, the prediction equals the table's KERNEL_ONLY median and the recorded kernel count.
   - An unmeasured coordinate (for example L = 5, or a lane vector outside the G3h plan) raises `DeepSeekV4CoverageError` naming the operator and the coordinate.
4. **Repeat and validation report** (reported, not a gate):
   - EP16 against EP32 shared-expert medians, per operator and L;
   - KERNEL_ONLY sum against CUDA_EVENT elapsed per operator, from the download paths.
5. **Rank class.**
   - For each G3h task, the lane binding of its lane vector selects that task's measured row, for every lane of the forward. The EP16 and EP32 widths are both covered.
   - `moe_shuffling` is charged nowhere in the V4 path.
6. **Composition and gaps.**
   - With sentinel costs, gaps and kernel counts:
     - the attention phase equals `compose_attention_phase` on both sides of the multi-stream threshold;
     - each other operator adds its kernel count × the gap of its segment class, under FULL, PIECEWISE and NONE;
     - the LM head and logits gather take `eager_preparation`;
     - the attention phase adds no `add_kernel_gap` part;
     - the EP lane path gets its gap through `_predict_layer_with_kernel_gap`.
   - A non-dummy V4 run without `kernel_gap.csv` raises.
7. **Both training paths.**
   - The independent path and the shared manager give identical models, predictions and kernel counts on the same tables.
   - For V4, the shared manager trains no generic compute, FFN, residual or CPU-overhead models. Show the trained model names on both paths.
8. **A9, cache.** A cold-cache run and a warm-cache run give identical predictions and cache identities.
9. **Host time.** The CPU stage fields of a non-dummy V4 forward are zero. `skip_cpu_overhead_modeling=False` raises the typed configuration error.
10. **Runner operators.** A prefill forward with new cache blocks raises `DeepSeekV4CoverageError` naming `kv_cache_block_zero`.
11. **Other paths unchanged (A12, partial).**
    - Run the existing tests that import the changed modules (find them with `grep -l`). Report the counts.
    - The known failure `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` is deselected.
    - The tests of `tests/integration/test_vllm_dp_placement_runtime.py` that fail before the change (the late-join failure) are listed with their status before and after.
    - `bash examples/architecture/run_dummy_smoke_matrix.sh` passes.
    - The lead runs the 77-case fidelity matrix after the commit.
12. **New unit tests** for decisions 1-10. Report the count.
13. **Hygiene:**
    - line counts of every touched module;
    - no unused imports (`/data/ycfeng/tmp/unused_imports.py`);
    - `git diff --check` is clean;
    - a cold import of `frontier.execution_time_predictor.dsv4_predictor` and of any new module loads no `torch`, `vllm` or `cuda` module.

## Expected return

- Report language: English.
- Changed, new and deleted paths, each with a summary.
- The results of Acceptance 1-13, with the commands, the expected and actual values, and the counts.
- The promotion manifest path and the table row counts.
- Scratch paths, deviations and their reasons, and limits.
- If the code or the tables contradict a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Examples:
  - a target coordinate lies off the measured grid;
  - the trainer has no duplicate-row reduction;
  - the shared manager cannot take the V4 training function without an interface change;
  - a typed contract in the tables disagrees with `resolve_layer_contract`.
