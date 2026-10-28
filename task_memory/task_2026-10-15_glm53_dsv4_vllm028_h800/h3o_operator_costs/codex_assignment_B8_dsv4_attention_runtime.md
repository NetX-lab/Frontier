# Task: DeepSeek-V4 attention at runtime — one V2 workload builder for real and dummy forwards, sklearn registration of the replay predictor, and the trained attention path (stage 1a assignment B8, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B8 of stage 1a (route entries 4, 5 and the attention part of 3). |

## Goal

B7 (`dd4c4317`) added the replay attention predictor (`frontier/execution_time_predictor/dsv4_predictor.py`), the batch features (`frontier/attention/dsv4_features.py`) and the per-scope-gap composition (`compose_attention_phase`). Nothing at runtime calls them yet. The runtime still uses two builders:

- `build_deepseek_v4_attention_workload` for real batches, which requires `has_attention_metadata=True` and raises on two PIECEWISE cases;
- `build_deepseek_v4_dummy_attention_workload` for dummy forwards, which is eager-only and one row, and prices two warmup-only scopes (`dsv4_query_padding`, `dsv4_attention_output_zero`) that the V2 runner never executes in a dummy forward.

B8 does three things:

1. **Entry 4: one V2 workload builder.** `build_deepseek_v4_attention_workload` accepts real `Batch`es and `DummyForwardBatch`es, in NONE, PIECEWISE and FULL, and selects exactly the scope set the V2 runner executes. The dummy builder and the `has_attention_metadata` flag are deleted.
2. **Entry 5: sklearn registration.** The sklearn predictor loads the replay table, trains the V4 scope models with its own `_train_model` (including one-row keys) and publishes a `DeepSeekV4AttentionPredictor`.
3. **Entry 3, attention part: the trained attention path.** One function in `frontier/execution_time_predictor/deepseek_v4.py` prices a V4 attention layer: in dummy mode with the synthetic duration, in trained mode with the predictor's per-scope cost and signed gap. The isolated-eager attention path and its helpers are deleted.

## Approach and decisions

### Evidence (lead)

- **Real table.** `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_review/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`, read-only. It has 505 rows, sha256 `277626cd766865bd...`, threshold 1024, kv dtype `fp8_ds_mla`, `device_name` "NVIDIA H200". The B7 assignment (`h3o_operator_costs/codex_assignment_B7_dsv4_attention_predictor.md`, section "Evidence") lists its single-valued columns and the meaning of the request-list columns.
- **V2 scope rule.** The lead checked the rule below against every row: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_design/rules.py` reports "rows 505 mismatch 0" for both the scope set (`operator_scopes`) and `indexer_short_context`. With q the scheduled tokens and c the computed tokens of each request, s = q + c, r the layer's compress ratio and T = `index_topk`:
  - **Base scopes (every variant):** the six projection scopes of `_projection_workloads`, `dsv4_fused_q_kv_prepare` and `dsv4_sparse_attention`.
  - **r > 1 (CSA and HCA):** add `dsv4_compressor_projection`, `dsv4_compressor_save_partial_states` and `dsv4_compressor_fused_compress_store`. **The store always runs**, also for HCA without a boundary crossing and in every mode.
  - **CSA:** add `dsv4_index_weights_projection` and the three `dsv4_index_compressor_*` scopes. Then:
    - if the mode is NONE and max(s) // r ≤ T: add `dsv4_index_short_context_fill`; `indexer_short_context` is True;
    - otherwise (always under FULL and PIECEWISE): add `dsv4_index_query_projection`, `dsv4_index_fused_query_prepare` and `dsv4_index_topk`, and add `dsv4_index_scores` if and only if some request has q ≤ 1 or Σ over the requests with q > 1 of s // r is positive. `indexer_short_context` is False.
  - `indexer_short_context` is False for HCA and uncompressed layers.
- **V2 dummy geometry** (design §3.7 item 4). With R = `compute_rows(batch)`:
  - FULL: R requests of (q = 1, c = 0);
  - PIECEWISE and NONE: one request (q = R, c = 0).

  NONE dummies have R = 1 (`_price_deferred_dummies` clears their graph metadata). The lead's census of the dummy rows (`b8_design/dummy_scopes.out`): CSA FULL 19 scopes for every R; CSA NONE R = 1, 16 (short fill); CSA PIECEWISE R = 2, 18 (no `dsv4_index_scores`); CSA PIECEWISE R = 1, 4 or 8, 19; HCA 11; uncompressed 8.
- **Planner role.** `plans_tile_schedule` of a layer is `deepseek_v4_tile_schedule_planners(tuple(spec.variant_id for spec in model.get_layer_attention_specs()))[layer]`. Full-model layers (`b8_design/planners.out`): CSA planner 2, followers 4..42 even; HCA planner 3, followers 5..41 odd; uncompressed 0 and 1, both planners.
- **Why the trained path is reachable only through `_predict_attention_layer_time` today.** In non-dummy mode the sklearn constructor stops V4 at `require_enabled_for_execution(structural=dummy or step_costs)`, and `predict_source_execution`'s `local_scope` raises for every non-attention scope. Tests reach the trained attention path the way `tests/unit/test_dsa_predictor.py:404-505` reaches DSA: construct in dummy mode, then set `_enable_dummy_mode = False`, the measurement type and the input file, and train. Full admission is later work (see "Not in scope").
- **Kernel gaps.** `kernel_gap.load_kernel_gap_ms` returns the eager and cuda_graph values for V4 when `kernel_gap.csv` exists, and `_train_model` then pairs every `time_stats.*` target with a kernel-count model. Attention gaps are only added through `_record_attention_kernels` / `_record_kernel_count`, which the V4 path never calls. So no gap is added to V4 attention twice. Leave `frontier/execution_time_predictor/kernel_gap.py` unchanged.

### Entry 4: unified builder in `frontier/attention/deepseek_v4.py`

- **Signature.** `build_deepseek_v4_attention_workload(model_config, replica_config, batch, global_layer_id, *, cluster_type, execution_mode, prefix_caching, decode_context_parallel_size, prefill_context_parallel_size)`. The `has_attention_metadata` parameter and its raise are removed.
- **Input batches.**
  - A `DummyForwardBatch` goes through `validate_deepseek_v4_dummy_forward` (unchanged: no requests, `num_tokens == [1]`, no spec or AFD metadata).
  - Any other batch goes through `_validate_source_batch` (unchanged).
  - `_validate_attention_context` and the execution-mode consistency check stay for both.
- **Geometry.** R = `compute_rows(batch)` for both kinds.
  - Real batch: the queries, contexts and decoding flags of the scheduling snapshot, as today.
  - Dummy: the V2 dummy geometry above.

  The padding geometry drives every shape, the scope rule, `request_intervals`, `new_compressed_rows` and the geometry fields of the sparse-attention call (`swa_logical_window_span_rows`, `compressed_history_rows`, `query_intervals`).
- **Valid-row fields.** Zero for a dummy: `actual_num_tokens`, `kv_cache_write_rows` and `actual_query_rows` are 0, and `invalid_padding_rows` is R. A dummy writes no KV page (its slot mappings are invalid).
- **Serving-request fields.** Empty for a dummy: `semantic_is_decoding`, `source_request_order`, `decode_request_indices` and `prefill_request_indices` are `()` (both at the top level and inside the sparse-attention call), and `num_decode_query_tokens` and `num_prefill_query_tokens` are 0. Real batches keep today's values.
- **Scope rule.** Implement the V2 rule above:
  - remove both PIECEWISE raises (HCA, and CSA short context);
  - short-context fill only under NONE;
  - `dsv4_index_scores` by the rule above (today the long branch always adds it);
  - remove the `full_graph` parameter of `_compressor_workloads`; the store is unconditional.
- **New outputs** in the returned dict: `indexer_short_context` (bool) and `plans_tile_schedule` (bool, the planner role of `global_layer_id`).
- **Deletions.** `build_deepseek_v4_dummy_attention_workload`. The two warmup-only scopes stay registered as static operators (families, manifest, quantization, metrics constants, composition groups, feature table); they are no longer emitted. The lead records their removal as future work.
- **`DummyForwardBatch` docstring** (`frontier/entities/batch.py:1353-1362`). Label its current metadata sentences as V1-runner facts, and add the V2 / DeepSeek-V4 facts of design §3.7 item 4: the dummy has the synchronized descriptor's rows and requests (FULL pads requests to the descriptor; PIECEWISE and NONE keep one request), each request's sequence length equals its query length, every row is padding, it has dummy attention metadata, and FULL re-stages the capture metadata. No code change in `batch.py`.

### Shared KV dtype alias in `frontier/attention/families.py`

- `families.py:143-147` resolves vLLM's `fp8` alias to `fp8_ds_mla` inline. Move it into one named public helper, for example `deepseek_v4_main_kv_cache_dtype(kv_cache_dtype)`, with the alias table as a named constant, and use it in the cache resolver (behavior unchanged).
- The replay loader compares `kv_cache_dtype` and `attention_kv_cache_dtype` with the helper's value of `replica_config.kv_cache_dtype`. Reason: a real V4 `ReplicaConfig` uses `kv_cache_dtype="fp8"` (`tests/unit/test_dsv4_structural_execution.py:147-168`), and the table records `fp8_ds_mla`. Today the loader rejects that pair.

### Entry 5: sklearn registration

- **Loader** `_load_attention_df` (`sklearn_execution_time_predictor.py:1436-1442`): the V4 branch calls `load_dsv4_attention_dataframe(file_path, self._model_config, self._replica_config, self._block_size, self._active_measurement_type)`.
- **Trainer** `_train_attention_layer_models` (`:3572`): add a V4 branch beside the DSA branch, `train_dsv4_attention_models(self._load_attention_df(self._attention_input_file), self._train_model)`.
- **Prediction** `_predict_for_attention_layer_models` (`:4038-4054`): add the V4 branch beside DSA, returning `{"dsv4_attention": DeepSeekV4AttentionPredictor(self._models)}`.
- **Profiling loader** (`profiling_dataframe_loaders.py:315-320`): delete the V4 branch. V4 then raises in the following `require_enabled_for_execution()`. The manager path for V4 comes with final admission.
- **One-row keys in `_train_model`** (`:3148-3245`). Many replay keys have one row, and `GridSearchCV` needs two. Follow the existing precedent in `prediction_family_trainers.py:1729-1747`: with one row, fit the estimator directly (`best_estimator = model.fit(X, y)`), log "trained on its single row" and skip `grid_search.score`. All later steps (hash, feature names, exact lookup, cache store, training-prediction data, kernel-count pairing) stay the same. This only changes a case that raises today.
- **Threshold on the models.** `train_dsv4_attention_models` attaches `_frontier_dsv4_multi_stream_gemm_token_threshold` (the single value of `multi_stream_gemm_token_threshold`, as `int`) to every model, after each `train(...)` call, as it does for `_frontier_dsv4_bounds`. Cached models get it again on every run.
  - `DeepSeekV4AttentionPredictor(models)` drops its keyword. It reads the attribute from its own models and raises `ValueError` unless they carry exactly one value. It still exposes `multi_stream_gemm_token_threshold`.
  - Reason: `_predict_for_attention_layer_models` builds the predictor from `self._models` only, and the threshold must come from the measured table.
  - Update `tests/unit/test_dsv4_predictor.py` accordingly, and add a test for a missing and for two different thresholds.
- **Delete** `_predict_dsv4_dummy_attention_time` (`:7583-7622`).
- **V4 branch of `_predict_attention_layer_time`** (`:7649-7661`): call the new entry-3 function with the `serving_context` keys `execution_mode`, `prefix_caching`, `decode_context_parallel_size` and `prefill_context_parallel_size`, and return `AttentionTime(operator_times=...)`.
- Keep `sklearn_execution_time_predictor.py` changes to these registration lines; the module is above 2,000 lines (`docs/general/module-size-exceptions.md`) and must not grow beyond them.

### Entry 3, attention part, in `frontier/execution_time_predictor/deepseek_v4.py`

- **New function**, for example `predict_attention_operator_times(predictor, batch, layer_id, cluster_type, *, execution_mode, prefix_caching, decode_context_parallel_size, prefill_context_parallel_size) -> AttentionOperatorTimes`.
  - It builds the workload with the unified builder.
  - **Dummy mode** (`predictor._enable_dummy_mode`): each scope costs `predictor._dummy_execution_time * len(calls)`, as today. Attach `_dsv4_workloads`.
  - **Trained mode:**
    - features: `build_dsv4_dummy_features(R)` for a `DummyForwardBatch`; otherwise `build_dsv4_features(batch.num_tokens, batch.num_context_tokens, batch.request_is_decoding, variant=..., compress_ratio=max(1, ratio), sliding_window=model.sliding_window, index_topk=model.index_topk, padded_rows=R)`, with R the workload's `num_tokens`;
    - prediction: `predictor._predictions["dsv4_attention"].scope_times(features, variant=..., graph_runtime_mode=..., plans_tile_schedule=workload["plans_tile_schedule"], scopes=tuple(workload["operators"]))`, as the DSA branch reads `self._predictions["dsa_attention"]`. No `dsv4_predictor` import is needed here;
    - return `AttentionOperatorTimes(cost_ms)` and attach `_dsv4_workloads`, `_dsv4_scope_gap_ms` (the gap map) and `_dsv4_composition_inputs`: a mapping with `variant`, `graph_runtime_mode`, `rows` (R), `multi_stream_gemm_token_threshold` (from the predictor) and `indexer_short_context`. These are exactly the keyword arguments of `compose_attention_phase`.
  - It records no kernel counts.
- **Callers.** `predict_source_execution` (replacing `predictor._predict_dsv4_dummy_attention_time(...)`) and sklearn's `_predict_attention_layer_time`.
- **Remove `has_attention_metadata` everywhere:** the `predict_source_execution` parameter and its two checks (`:101-108`; the batch validation stays), `serving_context` (`:521`), and the tests that pass it.
- **Moves to `dsv4_predictor.py`:**
  - `load_attention_dataframe`, made replay-only: it always selects `DSV4_REPLAY_ROW_SCHEMA_VERSION`, and the `schema_version` keyword goes. Its only caller is the replay loader.
  - `validate_balanced_routing`. It imports `BALANCED_ROUTING_SHA256` and `routing_columns` from `frontier.profiling.attention.dsv4_attention_rows`.
  - `validate_measured_identity` and `_select_rows` stay in `deepseek_v4.py`; the non-attention part of `read_isolated_scope` uses both.
- **`read_isolated_scope`:**
  - delete its attention branch (`:721-827`) and the `attention_layout` parameter;
  - the phase check (`:703-708`) becomes: non-attention scopes raise `DeepSeekV4CoverageError` when any request is decoding or has q == 1 (this is what it does today for non-attention scopes);
  - drop the `DSV4_SCOPE_NAMES` / `scope_operators` import, the `attention_layout` key of the result and the decode wording of `qualification`.
- **Delete** `build_isolated_prefill_layouts`.
- **Files to delete** (the B8 design authorizes these deletions; they serve only the isolated-eager attention path):
  - `frontier/execution_time_predictor/dsv4_attention_input.py`;
  - `tests/comparison/calibration/prepare_dsv4_attention_input.py`;
  - `tests/integration/check_dsv4_measured_scopes.py`;
  - `tests/fixtures/deepseek_v4_attention_layouts.json`.

  Delete only these four files. Remove the attention-isolated tests of `tests/unit/test_dsv4_graph_measured.py` (they cover the deleted branch and files: around lines 150, 187-260, 327-372, 410-415, 446-603, 628-817) and keep or adapt the non-attention tests (around 264-322).
- **Runtime import boundary** (`tests/unit/test_profiling_runtime_boundary.py`). Expected after B8, each with its one-line comment:
  - `deepseek_v4.py`: only `frontier.profiling.common.runtime_identity` (`dsa_rows`, `dsv4_attention_rows` and `dsv4_attention_plan` go with the deleted code);
  - `dsv4_predictor.py`: `frontier.profiling.attention.dsv4_replay_rows` and `frontier.profiling.attention.dsv4_attention_rows`.

  Set the entries to the exact AST imports of the final modules. The pre-existing failure on `dsa_rows` then passes.

### Tests to update

- `tests/unit/test_dsv4_attention_workload.py`:
  - remove `has_attention_metadata` from the `workload` helper (`:49`) and from the parametrized cases (`:186-197`);
  - the HCA store tests (`:110-145`) now expect the store on every HCA call;
  - replace the `_predict_dsv4_dummy_attention_time` helpers and tests (`:280-341`) with tests of the new entry-3 function in dummy mode (cost = dummy duration × calls per scope). Trained mode is covered by Acceptance 2.
- `tests/unit/test_dsv4_source_execution.py`: line 56 and the case `({"has_attention_metadata": False}, "metadata")` (`:203`), which is deleted.
- `tests/unit/test_dsv4_structural_execution.py`:
  - the boundary-validator store test (`:103`);
  - the context dicts (`:125`, `:236`, `:261`);
  - the dummy-builder tests (`:285`, `:309`, `:332`) are rewritten to the V2 contract: scope sets per variant and mode, valid-row and serving fields of a dummy, FULL and PIECEWISE dummies priced (no raise).
- `tests/integration/run_dsv4_structural.py`: `validate_attention_boundary` store rule (`:98`) becomes `store = ratio > 1`.
- New tests for the builder's scope rule (see Acceptance 1) and for the owner pattern (Acceptance 2), in the existing V4 test modules or one new `tests/unit/test_dsv4_attention_runtime.py`.

### Not in scope (the lead records these deferrals)

- PD admission (`serving_context` keeps its KV-connector raise): route entry 8.
- The manager path for V4, metadata registration, and the family's `execution_enabled`: final admission.
- Structural composition wiring: `_dsv4_attention_schedule` from `compose_attention_phase` with the attached inputs. Today the structural and step-cost paths keep `serial_attention_phase`. Entries 6-9.
- Entry 10 wiring (V4 gap selection in `kernel_gap.py`, `add_kernel_gap`) and the kernel-count pairing of V4 targets.
- Which measurement family V4 uses per step, and mixed decode/prefill batches in trained mode (the feature phase rule raises `DeepSeekV4CoverageError` for them).
- Removing the static warmup-only operators.
- Task records under `task_memory/`.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `dd4c4317`).
  - Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
  - The lead edits task records under `task_memory/`: do not write there.
- **Writable paths:**
  - `frontier/attention/`: `deepseek_v4.py`, `families.py` (the alias helper only);
  - `frontier/entities/`: `batch.py` (the `DummyForwardBatch` docstring only);
  - `frontier/execution_time_predictor/`: `deepseek_v4.py`, `dsv4_predictor.py`, `sklearn_execution_time_predictor.py` (registration lines only), `profiling_dataframe_loaders.py` (the V4 branch deletion only), and the deletion of `dsv4_attention_input.py`;
  - `tests/unit/`: the V4 test modules named above, `test_dsv4_predictor.py`, `test_profiling_runtime_boundary.py`, and one new test module if needed;
  - `tests/integration/`: `run_dsv4_structural.py` (store rule), and the deletion of `check_dsv4_measured_scopes.py`;
  - `tests/comparison/calibration/`: the deletion of `prepare_dsv4_attention_input.py`;
  - `tests/fixtures/`: the deletion of `deepseek_v4_attention_layouts.json`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b8/`. Structural-run outputs, before snapshots and check scripts go there.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git archive` and `git diff` are fine.
  - No GPU, network or package installs. No deletions or moves other than the four files above.
  - Python 3.10 syntax. No torch, vllm or cuda imports at module load.
  - Keep modules at or below 2,000 lines (except `sklearn_execution_time_predictor.py`, registration lines only).
  - No magic literals in code paths; no guards for states the code cannot reach; no second path for something the code already does. When a change supersedes a path, delete the old path in the same change.
- **Read-only data:**
  - the real table above;
  - `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_design/` (the lead's rule and census scripts);
  - the A4 inputs, through `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_baseline.py`;
  - `task_memory/.../h3o_operator_costs/evidence/p1h_bridge/package_03/` (B7 evidence: `b7_real_table_composition.py` and `.json`);
  - `task_memory/.../h3o_operator_costs/design.md` §3.7 (lines 116-158).
- **Environment:**
  - `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vllm);
  - `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`;
  - `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`;
  - `ulimit -v 31457280` for tests and analysis, `ulimit -v 62914560` for structural simulation runs;
  - a `timeout` on every command.

## Acceptance

1. **Scope sets match the replay table.** For all 505 rows of the real table, build a Frontier batch at layer `global_layer_ids[0]` (first element of the row's JSON list):
   - non-dummy rows: a `Batch` from the request lists (prefills as (q, c, not decoding), decodes as (1, c, decoding)) with `DecodeCudaGraphMetadata` for the row's `graph_runtime_mode` and `padded_rows` (no metadata for NONE);
   - dummy rows: a `DummyForwardBatch` with the FULL or PIECEWISE metadata at `padded_rows`, or none for NONE.

   Use the h200 replica of Acceptance 3. Required: the workload's scope set equals `operator_scopes`, `indexer_short_context` equals the row's, `plans_tile_schedule` equals the row's and `num_tokens` (R) equals `padded_rows`. Report the counts by phase and mode. Commit a unit test of the rule on synthetic batches covering each branch (CSA short and long under NONE, the `dsv4_index_scores` rule, HCA without a crossing, FULL, PIECEWISE R = 2 and R = 4, NONE dummies).
2. **Owner-pattern test** (follow `tests/unit/test_dsa_predictor.py:404-505`, without the shared-manager variant):
   - a real `SklearnExecutionTimePredictor` subclass with `DummyRegressor` and grid `{}`, a real V4 `ReplicaConfig` with `kv_cache_dtype="fp8"`, a synthetic replay CSV (built from the `tests/unit/test_dsv4_replay_rows.py` fixtures, as `test_dsv4_predictor.py` does) that includes at least one one-row key;
   - training twice from the cache leaves the cache files' mtimes, the model hashes and the estimator count unchanged;
   - `_predict_for_attention_layer_models` returns the V4 predictor with the table's threshold;
   - `_predict_attention_layer_time` on a FULL decode `Batch` and on FULL and NONE `DummyForwardBatch`es returns the predicted costs, and composing them with `_dsv4_scope_gap_ms` and `_dsv4_composition_inputs` through `compose_attention_phase` equals the composition of the row's own medians.
3. **Trained runtime path on the real table** (scratch script, reported, not committed). Owner as in 2, with exact lookup, an h200 V4 replica (TP8, attn_dp 8, EP64, `kv_cache_dtype="fp8"`, `FLASHMLA_SPARSE_DSV4`, balanced routing) and `_attention_input_file` set to `attention_kernel_only.csv` in the real table's directory. For each row, build the batch as in 1, call `_predict_attention_layer_time`, compose with the attached inputs and compare with `layer_window_stats.median`. Report the number of models, the median, p90 and maximum absolute relative error, and the rows above 5%. Expected: the same per-row values as `b7_real_table_composition.json` (median 0.49%, p90 1.34%, max 6.67%); limits median ≤ 5%, p90 ≤ 10%. If they differ, report the rows and the cause; do not tune keys or features.
4. **A4 unchanged:** `a4_baseline.py <scratch>/a4_report.json` is byte-identical (`cmp`) to `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_report_d2a226fe.json`.
5. **Runtime import boundary:** `tests/unit/test_profiling_runtime_boundary.py` passes. Report the scratch AST listing of every predictor module's profiling imports.
6. **Step-cost path unchanged:** `tests/unit/test_dsv4_step_cost_input.py` passes without edits.
7. **Dummy-mode changes listed.** Run `tests/integration/run_dsv4_structural.py --dp 4 --case both` twice: on a `git archive dd4c4317` snapshot in scratch (before) and on the working tree (after), each with its own `--output-root` in scratch. Both must exit 0. List every difference in per-scope call counts of real and idle lanes, and in the summary metrics. Expected: idle lanes HCA 9 → 11 calls, CSA 13 → 16, uncompressed 8 → 8; real eager HCA calls without a boundary crossing gain one store call. Explain any other difference.
8. **Regression suites, all passing:** every `tests/unit/test_dsv4_*.py`, `test_dp_dummy_forward_pricing.py`, `test_forward_sync_state.py`, `test_monolithic_mixed_forward_sync.py`, `test_dsa_predictor.py`, `test_dsa_attention_profiling.py`, `test_profiling_runtime_boundary.py`, and `test_kernel_gap_pricing.py` (only the known failure `test_every_collective_launches_one_kernel_by_default` remains). Report the counts.
9. **Hygiene:** line counts of the changed modules; no unused imports in changed files (`/data/ycfeng/tmp/unused_imports.py`); cold imports of `frontier.execution_time_predictor.deepseek_v4`, `frontier.execution_time_predictor.dsv4_predictor` and `frontier.attention.deepseek_v4` exit 0 without torch, vllm or cuda modules; `/usr/bin/grep -rn "has_attention_metadata\|build_deepseek_v4_dummy_attention_workload\|_predict_dsv4_dummy_attention_time\|build_isolated_prefill_layouts\|dsv4_attention_input" frontier tests` finds nothing.

## Expected return

- Report language: English.
- Changed and deleted paths, each with a summary.
- Executed checks with their output tails; the Acceptance 1 counts, the Acceptance 3 numbers and the Acceptance 7 difference list.
- Scratch paths, deviations and their reasons, and limits.
- If a decision above cannot be implemented as written (for example a reachable state the assignment did not foresee), stop that part and report the evidence and options instead of choosing a different design.
