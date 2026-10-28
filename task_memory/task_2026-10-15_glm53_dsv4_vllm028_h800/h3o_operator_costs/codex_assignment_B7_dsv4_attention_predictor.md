# Task: DeepSeek-V4 attention replay loader, trainer and predictor, with each scope weighed by its own signed gap (stage 1a assignment B7, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B7 of stage 1a. |

## Goal

B6 (`a1feee14`) writes `attention_graph_replay_kernel_only.csv`. For every DeepSeek-V4 attention scope, each row holds:

- the kernel-only cost statistics, `time_stats.<scope>.*`;
- the signed gap statistics, `gap_stats.<scope>.*`. This is the gap before the scope's first kernel plus the gaps between its kernels.

Each row also holds the measured layer window, `layer_window_stats.*`. Nothing at runtime reads this table yet.

The adopted remedy (b) prices every attention scope with its own measured signed gap. The predictor learns the gap the same way it learns the duration. The per-class gap values stay only for non-attention operators and for pre- and post-forward work.

B7 adds three things:

1. **Shared batch features.** A batch-feature builder that the loader and the runtime both use.
2. **New predictor module.** `frontier/execution_time_predictor/dsv4_predictor.py` provides a loader, a trainer and a predictor for the replay table. The predictor returns each scope's predicted cost and signed gap.
3. **Composition revision.** `compose_attention_phase` takes one gap per scope instead of kernel counts times class gaps.

The runtime wiring is not part of B7. It comes in the next assignment (B8):

- predictor registration and training;
- the PREFILL and DECODE admission;
- the trained composition;
- the kernel-gap reader.

## Approach and decisions

### Evidence (lead, package 03)

- **Real table.** `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_review/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`, read-only.
  - It has 505 rows: decode 400, dummy 50, first_tail 50, prefill 5.
  - By mode: FULL 345, PIECEWISE 150, NONE 10.
  - It is the lead's B6 aggregate: sha256 `277626cd766865bd...`, the same file Codex wrote in B6.
- **Single-valued identity and layout columns:**

  | Column | Value |
  | --- | --- |
  | `measurement_type` | `KERNEL_ONLY` |
  | `model_architecture_profile` | `deepseek_v4` |
  | `attention_family` | `deepseek_v4_attention` |
  | `attention_backend` | `FLASHMLA_SPARSE_DSV4` |
  | `kv_cache_dtype`, `attention_kv_cache_dtype` | `fp8_ds_mla` |
  | `padded_heads` | 64 |
  | `multi_stream_gemm_token_threshold` | 1024 |
  | `profiling_precision` | `BF16` |
  | `n_q_head_per_rank` | 8 |
  | `num_tensor_parallel_workers` | 8 |
  | `block_size` | 256 |
  | `routing_runtime_path` | `balanced_topk` |
  | `schema_version` | `dsv4_attention_replay.v1` |
  | `graph_mode`, `engine_cudagraph_mode` | `FULL_AND_PIECEWISE` |
  | `parallel_groups` | `{"engine_tensor_parallel":1,"target_tensor_parallel":8}` |
  | `device_name` | `NVIDIA H200` |
  | `source_revision` | `230239c8` |

  - `compress_ratio` is one of {4, 128, 1}: CSA, HCA, uncompressed.
  - There is no `input_file_sha256` column.
- **Request lists.** Each request list is a JSON list; non-dummy rows carry three of them:
  - `prefill_seq_lens`: the scheduled tokens of each prefill request, in request order;
  - `prefill_kv_cache_sizes`: the computed tokens of the same requests, in the same order;
  - `decode_kv_cache_sizes`: the computed tokens of each decode request, sorted. Each decode request schedules 1 token.

  These come from `frontier/profiling/attention/dsa_rows.py:request_load_columns`. A first-tail request is a prefill with one scheduled token. Dummy rows have no request or key-length columns; their layout is `padded_rows`.
- **Planner role.** The lead compared the two planner roles (`plans_tile_schedule`) on rows that share `padded_rows` and the request lists.
  - Script: `evidence/p1h_bridge/package_03/b7_planner_role_gaps.py`; output: `b7_planner_role_gaps.log`.
  - **Only `dsv4_sparse_attention` differs systematically**, because it owns the FlashMLA tile-schedule planner kernel (one more kernel on planner layers). On CSA and HCA, in FULL and PIECEWISE, for decode, first_tail and the FULL dummy:
    - its cost is +100% to +215% on planner layers;
    - its signed gap differs by +0.34 to +1.91 µs.

    The PIECEWISE dummy shows 0.9% and 1.5%: those rows have no attention metadata.
  - Every other scope:
    - paired median gap difference at most 0.34 µs;
    - median cost difference at most 6.2%;
    - no consistent sign.
  - **Decision:** the planner role is part of the model key only for `dsv4_sparse_attention`. All other scopes pool both roles.
  - Design §3.6 listed the planner role for every scope. The lead records this deviation in the route document.
- **Training-data handling.**
  - The sklearn `_train_model` drops, with a warning, rows that are NaN in the feature or target columns. The trainer must therefore pass only the rows where the scope is present.
  - Training with one row per key is a B8 concern.

### Batch features: new module `frontier/attention/dsv4_features.py`

Follow `frontier/attention/dsa_features.py`: a frozen features dataclass, a feature-column table and one builder that both fitting and lookup use.

- **Phase.** Phase names equal the profiler's phase labels: `decode`, `first_tail`, `prefill`, `dummy`. Define them as named constants in this module. Do not import `frontier.profiling` here.
  - For a batch of (query length q, context length c, is_decoding) per request:
    - **decode:** every request is decoding;
    - **first_tail:** every request has q == 1 and at least one is not decoding;
    - **prefill:** no request is decoding and every q > 1;
    - **otherwise:** raise `DeepSeekV4CoverageError` (from `frontier/attention/deepseek_v4.py`), naming the mix.
  - A decoding request with q != 1 raises `ValueError`. The same contract is in `deepseek_v4._validate_source_batch`.
- **Features of a non-dummy batch.** For request i with keys k = c + t for t = 1..q, W = `sliding_window`, r = the stored compress ratio and T = `index_topk`:

  | Column | Value |
  | --- | --- |
  | `num_sequences` | number of requests |
  | `num_query_tokens` | Σq |
  | `padded_rows` | input: the dispatch rows of the call |
  | `sum_swa_key_pairs` | Σ min(k, W) |
  | `sum_compressed_key_pairs` | Σ floor(k / r) for CSA and HCA; 0 for uncompressed |
  | `sum_selected_key_pairs` | Σ min(floor(k / r), T) for CSA; equal to `sum_compressed_key_pairs` for HCA; 0 for uncompressed |
  | `num_compressed_writes` | Σ (floor((c + q) / r) − floor(c / r)) for CSA and HCA; 0 for uncompressed |
  | geometry | `min_query_len`, `max_query_len`, `min_context_len`, `max_context_len`, `max_key_len` |

  - Use closed forms, not loops over tokens. With F(n) = Σ_{k=0..n} floor(k / r) = r·m(m−1)/2 + m(n − r·m + 1), where m = floor(n / r):
    - the compressed sum of a request is F(c + q) − F(c);
    - the selected sum uses min(floor(k / r), T) = floor(min(k, r·T) / r);
    - the SWA sum has the analogous closed form of Σ min(k, W).
  - The variant decides which terms apply; r is used only for CSA and HCA. A CSA or HCA variant with r ≤ 1 raises `ValueError`.
- **Features of a dummy batch.** A separate builder takes `padded_rows` and returns phase `dummy` with `{padded_rows}` only.
- **Feature-column table.** Map every one of the 22 CSA-family scope names (`frontier/attention/families.py`, `DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT`) to its feature columns. Write the table explicitly; do not derive it from `AttentionOperatorRole`, whose roles do not match these classes.

  | Class | Feature columns | Scopes |
  | --- | --- | --- |
  | rows | `padded_rows` | `fused_qkv_projection`, `fused_qkv_rmsnorm`, `q_projection`, `query_padding`, `attention_output_zero`, `output_inv_rope_quant`, `grouped_output_projection`, `output_projection`, `compressor_projection`, `index_weights_projection`, `index_compressor_projection`, `index_query_projection`, `index_fused_query_prepare`, `index_short_context_fill` |
  | cache write | `padded_rows`, `num_query_tokens` | `fused_q_kv_prepare`, `compressor_save_partial_states`, `index_compressor_save_partial_states` |
  | compress | `padded_rows`, `num_sequences`, `num_compressed_writes` | `compressor_fused_compress_store`, `index_compressor_fused_compress_store` |
  | index | `padded_rows`, `num_sequences`, `sum_compressed_key_pairs` | `index_scores`, `index_topk` |
  | attention | `padded_rows`, `num_sequences`, `sum_swa_key_pairs`, `sum_selected_key_pairs` | `sparse_attention` |

  - All scope names above carry the `dsv4_` prefix.
  - The dummy phase uses (`padded_rows`,) for every scope.
  - Give one function that returns a scope's feature columns for a phase.
  - A test checks that the table covers exactly the CSA family operator names. The HCA and uncompressed families are subsets of the CSA family.

### Padded query heads helper in `frontier/attention/deepseek_v4.py`

- `padded_heads = 64 if local_heads <= 64 else 128` appears in both `build_deepseek_v4_attention_workload` (line 287) and `build_deepseek_v4_dummy_attention_workload` (line 416).
- Add one helper, for example `deepseek_v4_padded_q_heads(local_heads)`, with named constants for the two FlashMLA padded widths.
- Use the helper at both sites and in the new loader. Workload output must not change.

### Shared admission helpers in `frontier/execution_time_predictor/deepseek_v4.py`

- **`load_attention_dataframe`** gains a `schema_version` keyword. Its existing callers keep the eager schema `DSV4_ROW_SCHEMA_VERSION`:
  - `profiling_dataframe_loaders.py:316`;
  - `sklearn_execution_time_predictor.py:1438`;
  - `read_isolated_scope`;
  - the monkeypatch in `tests/unit/test_dsv4_graph_measured.py:410-416`.
- **`_validate_isolated_identity(rows, predictor, measurement)`** (lines 938-965) becomes a public `validate_measured_identity(rows, model_config, replica_config, measurement)`.
  - It keeps all current checks except the `graph_mode == "eager"` expectation:
    - single-valued identity columns;
    - profile, measurement, quant and precision;
    - `engine_tensor_parallel == 1`;
    - device substring;
    - runtime fingerprint.
  - `read_isolated_scope` calls it, then selects `graph_mode == "eager"` with `_select_rows`. This keeps the isolated behavior: an ambiguous graph mode still raises in the shared check, and a non-eager table still finds no rows.
- **Balanced-routing check** (lines 757-768). Move it into one shared helper and use it from `read_isolated_scope` and from the new loader, under the same condition (the replica's `moe_gating_routing_runtime_path == "balanced_topk"`). Error text may keep its current wording.
- Behavior of the isolated path is otherwise unchanged. `tests/unit/test_dsv4_graph_measured.py` must pass unchanged.

### New module `frontier/execution_time_predictor/dsv4_predictor.py`

Follow `frontier/execution_time_predictor/dsa_predictor.py`: input path, loader, trainer, predictor.

- **Input path.** `dsv4_input_path(attention_file)` maps the configured `attention_kernel_only.csv` to `attention_graph_replay_kernel_only.csv` in the same directory. It returns the replay name unchanged and raises `ValueError` for any other name.
  - Move the op name `_REPLAY_OP_NAME` from `frontier/profiling/attention/dsv4_attention_main.py:65` to a public constant of `frontier/profiling/attention/dsv4_replay_rows.py`, imported by the main module.
  - The runtime builds the file name from that constant and the kernel-only suffix.
  - A test checks that the aggregate's written filename equals the name `dsv4_input_path` resolves.
- **Loader.** `load_dsv4_attention_dataframe(attention_file, model_config, replica_config, block_size, measurement_type)`.
  - **Measurement.** Only `KERNEL_ONLY` is admitted; any other measurement raises `DeepSeekV4CoverageError`.
  - **Read and select.** Read with `load_attention_dataframe(..., schema_version=DSV4_REPLAY_ROW_SCHEMA_VERSION, float_precision="round_trip")`, using `tp = replica_config.attn_tensor_parallel_size`. This selects the schema, measurement, family, profile, quant, precision, backend and model shape.
  - **Identity and routing.** Then:
    - `validate_measured_identity`;
    - require `graph_mode` single-valued and equal to the single `engine_cudagraph_mode`;
    - apply the balanced-routing helper under its condition.
  - **Layout (raise `DeepSeekV4CoverageError` naming the column):**
    - `padded_heads == deepseek_v4_padded_q_heads(model.num_q_heads // tp)`;
    - `kv_cache_dtype` and `attention_kv_cache_dtype` equal `replica_config.kv_cache_dtype`;
    - each row's `attention_variant` is a supported family variant. Its `compress_ratio` equals `max(1, ratio)` of that variant's layers in the model config. Derive the variant → ratio map from `model.get_layer_attention_specs()` and `model.compress_ratios`, and require one ratio per variant;
    - `plans_tile_schedule` is boolean (coerce strings as `dsa_predictor` does);
    - `multi_stream_gemm_token_threshold` is single-valued and a positive integer.
  - **Ownership, for each row:**
    - `operator_scopes` parses to `{s: [s]}`, its scopes all belong to the row variant's family;
    - each present scope has finite positive `time_stats.<s>.median`, finite `gap_stats.<s>.median` (signed) and a positive integer `time_stats.<s>.kernel_count`;
    - each family scope absent from `operator_scopes` has a NaN `time_stats.<s>.median` where the column exists.
  - **Features, for each row.** Rebuild the requests from the three lists:
    - prefills as (q, c, not decoding), paired by position;
    - decodes as (1, c, decoding).

    Build the features with `frontier/attention/dsv4_features.py`, using the model's `sliding_window`, `index_topk` and the row's `compress_ratio` and `padded_rows`.
    - Dummy rows (`phase == dummy`) use the dummy builder.
    - Require these to agree, raising `ValueError` otherwise:
      - the computed phase equals the row `phase`;
      - `num_sequences == batch_size`;
      - `num_query_tokens == total_tokens`;
      - `max_key_len` equals the row's `max_key_len`.
    - Add the feature columns to the frame. NaN is allowed only where a dummy row has no value.
  - **Errors.** Every rejection names the file and the column. An empty selection raises `DeepSeekV4CoverageError`.
- **Model key and names:**
  - Key: (scope, variant, `graph_runtime_mode`, phase), plus the planner role for the one tile-planner scope only.
  - The tile-planner scope is a named constant, for example `TILE_PLANNER_SCOPE = "dsv4_sparse_attention"`, with a one-line comment on why: it owns the FlashMLA tile-schedule planner kernel.
  - Two targets per key: `time` (`time_stats.<s>.median`) and `gap` (`gap_stats.<s>.median`).
  - One function builds the model name from the key and the target, as `dsa_predictor._model_name` does. Names must be deterministic and unique per (key, target); B8 uses them as cache names.
- **Trainer.** `train_dsv4_attention_models(df, train) -> dict`.
  - For every key and target, call `train(model_name=..., df=rows, feature_cols=list(columns), target_col=target)` on the rows of that key where the scope is present.
  - Attach `_frontier_dsv4_bounds` to each fitted model: {column: (min, max)} over the feature columns plus the geometry columns. Dummy keys use the feature columns only.
  - Keep `train` as the caller's fitting function, as in `train_dsa_models`.
- **Predictor.** `DeepSeekV4AttentionPredictor(models, *, multi_stream_gemm_token_threshold)`.
  - It keeps only its own models, raises `ValueError` when it has none, and exposes the threshold. B8 passes the threshold read from the loaded table, so composition uses the measured engine setting.
  - `scope_times(features, *, variant, graph_runtime_mode, plans_tile_schedule, scopes)` returns a frozen result with two maps, `cost_ms` and `gap_ms`, over exactly `scopes`. For each scope:
    1. **Model.** Find the model; a missing model raises `DeepSeekV4CoverageError` naming the key.
    2. **Feature schema.** `model._frontier_feature_names` must equal the table's columns; otherwise raise `ValueError`.
    3. **Bounds.** A feature or geometry value outside the bounds raises `DeepSeekV4CoverageError` naming the column, value and range.
    4. **Value.** Use the exact lookup (`model._frontier_exact_lookup`) when the feature tuple is present; otherwise use `model.predict`.
    5. **Result checks.** The cost must be finite and ≥ 0; the gap must be finite and may be negative. Anything else raises `ValueError`.
  - It records no kernel counts. Composition stays with the B8 caller, because the prefix and suffix scopes come from other predictors.

### Composition revision in `frontier/execution_time_predictor/dsv4_composition.py`

- New signature: `compose_attention_phase(scope_ms, scope_gap_ms, *, variant, graph_runtime_mode, rows, multi_stream_gemm_token_threshold, indexer_short_context)`.
  - The weight of scope n is `scope_ms[n] + scope_gap_ms[n]`.
  - Require `set(scope_gap_ms) == set(scope_ms)`, raising `ValueError` with "one gap per scope".
- `segment_class` stays public and unchanged. A4 and B8 use it for scopes without a measured gap (kernel count times class value).
- Update the docstrings:
  - each scope weighs its cost plus its own signed gap (the gap before its first kernel and between its kernels);
  - the caller supplies measured gaps for attention scopes and kernel count times the class value for other scopes.
- `serial_attention_phase`, `attention_scope_predecessors` and the stream program stay unchanged.
- **Callers to update:**
  - `tests/comparison/calibration/dsv4_composition_check.py:142-145`: `compose_attention_phase(scope_ms, {n: signed_gaps.get(n, 0.0) for n in scope_ms}, **options)`;
  - `dsv4_composition_check.py:152`: `compose_attention_phase(scope_ms, {n: counts[n] * gap_ms[segment_class(group.step_mode, n)] for n in scope_ms}, **options)`;
  - `tests/unit/test_dsv4_composition.py`:
    - its `compose` helper;
    - replace `test_gaps_follow_the_segment_class_and_are_charged_once` with one test that each scope adds its own signed gap, including a negative gap, and one test of the `segment_class` mapping (FULL, PIECEWISE for sparse and non-sparse scopes, NONE, and an unknown mode raising);
    - add a gap-key mismatch case to `test_composition_rejects_incomplete_inputs`;
  - `tests/unit/test_dsv4_replay_rows.py:137`.
  - The A4 report must stay byte-identical: (a + g) + count·0.0 equals a + g, and cost + (count·class gap) is the same arithmetic as before.
- Historical evidence scripts under `task_memory/` use the old signature. Leave them alone; they are records of their commits.

### Runtime import boundary (`tests/unit/test_profiling_runtime_boundary.py`)

- Add `RUNTIME_IMPORTS` entries, each with a one-line comment:
  - `dsv4_composition.py`: `frontier.profiling.kernel_gap.schema`;
  - `dsv4_predictor.py`: exactly the profiling modules it imports, for example `frontier.profiling.attention.dsv4_replay_rows` for the schema version and op name.
- The test asserts module by module and stops at the first mismatch. The pre-existing extra import of `frontier.profiling.attention.dsa_rows` in `deepseek_v4.py` (from `2f87828b`, used by `build_isolated_prefill_layouts`) stays. B8 removes it with the isolated path.
- Report a scratch AST check that lists every module's mismatch. Expected after B7: only `deepseek_v4.py` with the extra `dsa_rows` import.

### Not in scope

- **B8 work:**
  - predictor registration in `sklearn_execution_time_predictor.py` and real sklearn training (entry 5);
  - the PREFILL and DECODE admission and the scope-set prediction (entry 4);
  - the trained composition path and the deletion of `read_isolated_scope` (entry 3);
  - the kernel-gap reader wiring (entry 10);
  - any comparison of the engine settings (`max_model_len`, `max_num_batched_tokens`, `max_num_seqs`) with the serving config.
- Changes to the replay row format or the eager rows.
- Task records under `task_memory/`.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `a1feee14`).
  - Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
  - The lead edits task records under `task_memory/`: do not write there.
- **Writable paths:**
  - `frontier/attention/`: the new `dsv4_features.py`, and `deepseek_v4.py` for the padded-heads helper;
  - `frontier/execution_time_predictor/`: the new `dsv4_predictor.py`; `dsv4_composition.py`; `deepseek_v4.py` for the shared helpers. Change the callers in `profiling_dataframe_loaders.py` and `sklearn_execution_time_predictor.py` only if the `schema_version` keyword requires it;
  - `frontier/profiling/attention/`: `dsv4_replay_rows.py` and `dsv4_attention_main.py`, for the op-name constant move only;
  - `tests/comparison/calibration/`: `dsv4_composition_check.py` only;
  - `tests/unit/`:
    - the new `test_dsv4_predictor.py` and `test_dsv4_features.py`;
    - updates to `test_dsv4_composition.py`, `test_dsv4_replay_rows.py` and `test_profiling_runtime_boundary.py`;
    - updates to `test_dsv4_graph_measured.py` and `test_dsv4_attention_workload.py` only if a refactor requires them.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b7/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits.
  - No GPU, network, package installs, deletions or moves.
  - Python 3.10 syntax.
  - No torch, vllm or cuda imports at module load.
  - Keep modules at or below 2,000 lines.
  - No magic literals in code paths: name the phase labels, column prefixes, target names and the tile-planner scope.
  - Add no guards for states the code cannot reach, and no second path for something the code already does.
- **Read-only data:**
  - the real table above;
  - the A4 inputs, through `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_baseline.py`;
  - the lead evidence under `task_memory/.../evidence/p1h_bridge/package_03/`.
- **Environment:**
  - `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vllm);
  - `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`;
  - `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`;
  - `ulimit -v 31457280`;
  - a `timeout` on every command. One A4 pass takes about a minute.

## Acceptance

1. **New unit tests.**
   - **`tests/unit/test_dsv4_features.py`:**
     - the phase rule for each case, including the coverage error on a mix and the error on a decoding request with q != 1;
     - each closed-form sum equals a brute-force loop over random (q, c) in all three variants, including contexts across a ratio boundary and beyond r·T and W;
     - the dummy builder;
     - the table covers exactly the CSA family names.
   - **`tests/unit/test_dsv4_predictor.py`**, using the fixtures of `tests/unit/test_dsv4_replay_rows.py` (`plan`, `saved_records`) and the `_fit` fake style of `tests/unit/test_dsa_predictor.py`:
     - **Round trip:** records → `build_dsv4_replay_rows` → aggregate CSV through `dsv4_attention_main` → `load_dsv4_attention_dataframe` → `train_dsv4_attention_models` with the fake fit → `scope_times` on the features of each row's own batch. Composing those costs and gaps with `compose_attention_phase` equals composing the row's own `time_stats` and `gap_stats` medians (exact lookup).
     - **Keys:**
       - `dsv4_sparse_attention` keeps separate models per planner role;
       - another scope pools both roles;
       - distinct phases and modes give distinct models.
     - **Signed gaps:** a negative gap survives the round trip.
     - **Bounds and coverage:**
       - a value outside the bounds raises `DeepSeekV4CoverageError`;
       - a missing key raises it too;
       - a feature-schema mismatch raises `ValueError`.
     - **Loader rejections:**
       - a CUDA_EVENT measurement;
       - the eager schema;
       - wrong `padded_heads`;
       - a wrong kv dtype;
       - a wrong ratio for a variant;
       - a phase that disagrees with its request lists;
       - a missing gap column of a present scope;
       - a non-positive time;
       - non-single-valued `multi_stream_gemm_token_threshold`;
       - mismatched `graph_mode` and `engine_cudagraph_mode`;
       - a device mismatch.
     - **Input path:** the mapping, its error, and agreement with the aggregate's filename.
2. **Regression suites, all passing:**
   - `tests/unit/test_dsv4_composition.py`, `test_dsv4_composition_check.py`, `test_dsv4_replay_rows.py`, `test_dsv4_replay_gaps.py`, `test_dsv4_graph_bridge_check.py`, `test_dsv4_kernel_identity_check.py`;
   - `test_dsv4_graph_replay_profiling.py`, `test_dsv4_attention_profiling.py`, `test_dsv4_kv_cache_groups.py`, `test_dsv4_graph_measured.py`, `test_dsv4_attention_workload.py`;
   - `test_dsv4_step_cost_input.py`, `test_dsv4_structural_execution.py`, `test_dsv4_source_execution.py`, `test_dsv4_structural_pd.py`;
   - `test_dsa_predictor.py`, `test_dsa_attention_profiling.py`.

   Report the counts. Also run:
   - `tests/unit/test_profiling_runtime_boundary.py`: expected to fail only on `deepseek_v4.py`'s `dsa_rows` import. Report the assertion message and the scratch AST listing;
   - `tests/unit/test_kernel_gap_pricing.py`: the known failure `test_every_collective_launches_one_kernel_by_default` is unchanged.
3. **A4 unchanged:** `a4_baseline.py <scratch>/a4_report.json` is byte-identical (`cmp`) to `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_report_d2a226fe.json`.
4. **Real table** (scratch script, reported, not committed):
   - **Load.** Load the real table with:
     - model `BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")`;
     - a replica namespace like the `replica` fixture of `tests/unit/test_dsv4_attention_workload.py`, plus `device="h200"`, `kv_cache_dtype="fp8_ds_mla"` and `moe_gating_routing_runtime_path="balanced_topk"`;
     - block size 256, `MeasurementType.KERNEL_ONLY`.
   - **Train.** Train with the exact-lookup fake fit. Report the number of models by target and by phase.
   - **Compose each row.** For each of the 505 rows:
     - predict its scopes' costs and gaps from the row's own batch features;
     - compose with the row's variant, mode, `padded_rows`, the predictor's threshold and `indexer_short_context`;
     - compare with `layer_window_stats.median`.
   - **Report** the median, p90 and maximum absolute relative error, and the rows above 5% with their keys.

     The expected limits are median ≤ 5% and p90 ≤ 10%. Pooling the planner roles and averaging rows that share a feature tuple both add error over the B6 row composition (median 2.1e-16, p90 0.77%).

     If the limits fail, report the rows and which pooled scopes contribute. Do not change the key or the features to meet them.
5. **Imports:** cold imports of `frontier.execution_time_predictor.dsv4_predictor`, `frontier.attention.dsv4_features` and `frontier.execution_time_predictor.dsv4_composition` exit 0 without torch, vllm or cuda modules loaded.
6. **Size:** report the line counts of the changed modules; `frontier/execution_time_predictor/deepseek_v4.py` must stay at or below 2,000.

## Expected return

- Report language: English.
- Changed paths, each with a summary.
- Executed checks with their output tails.
- Scratch paths, deviations and their reasons, and limits.
- If the real-table composition or the boundary listing differs from "Acceptance", report the evidence. Do not tune the key or the features to reach the numbers.
