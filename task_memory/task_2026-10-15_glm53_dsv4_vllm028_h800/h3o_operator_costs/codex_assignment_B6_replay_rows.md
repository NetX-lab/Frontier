# Task: DeepSeek-V4 attention profiling rows from graph-replay records, with a per-scope signed-gap column (stage 1a assignment B6, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B6 of stage 1a. |

## Goal

Graph-replay raw records (`engine_policy=graph_replay`) cannot become profiling rows today: `build_dsv4_rows` refuses graph-replay plans (`require_scope_timing`), and the eager rows have no gap information.

The adopted remedy (b) prices each attention scope with its own measured signed device gaps:

- every attention scope's signed gaps (the gap before the scope's first kernel and between its kernels, from host-ahead samples) are recorded beside its kernel-only cost, at the same shape, as a separate gap column;
- the predictor (a later assignment) learns gaps the same way it learns durations;
- the per-class gap values stay for non-attention operators and for pre- and post-forward work.

The A4 check (`tests/comparison/calibration/dsv4_composition_check.py`) already composes each layer call from its own costs and gaps and passes on package 03 (125 groups). B6 turns the same per-call quantities into rows.

After the change:

- `python -m frontier.profiling.attention.dsv4_attention_main aggregate` with a graph-replay plan writes `attention_graph_replay_kernel_only.csv`;
- each row gives, per attention scope:
  - the kernel-only cost statistics (`time_stats.<scope>.*`, with `kernel_count`);
  - the signed gap statistics (`gap_stats.<scope>.*`);
- each row also gives the statistics of the measured layer window, the target A4 composes against (`layer_window_stats.*`);
- A4 and the row builder compute the per-call window and gaps with one shared function, and A4's report is unchanged.

## Approach and decisions

### Evidence (lead, package 03 analysis copy)

- Raw files: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e{0,1}.jsonl`.
  - Both engines have mode `FULL_AND_PIECEWISE`.
  - They hold 175 steps and 1925 layer calls (175 steps x 11 layers).
  - All step and call records are `KERNEL_ONLY`.
- Call scopes:
  - With the container scopes removed (`dsv4_graph_replay_runner.CONTAINER_SCOPES`), every call scope is a canonical family operator of its layer's variant (`frontier/attention/families.py`, `DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT`).
  - None of them is in `dsv4_attention_plan.WARMUP_ONLY_OPERATORS`.
- Kernel counts:
  - For every (case, step, variant, planner role) the kernel count of each scope is uniform, and so is the scope set.
  - The kernel counts differ between planner roles (for example, `dsv4_sparse_attention` has 3 or 4 kernels on CSA layers and 2 or 3 on HCA layers), so the planner role must be part of the row key.
  - For every call scope, the number of attention-owned `kernel` interval rows equals the call's `kernel_count`. Owned non-kernel activities are only 15 memsets.
- Scope sets differ only on CSA layers. Besides the common scopes, CSA layers run one of three sets:
  - `dsv4_index_query_projection`, `dsv4_index_fused_query_prepare`, `dsv4_index_scores`, `dsv4_index_topk`: 23 cases;
  - the same set without `dsv4_index_scores`: `e0-dummy-piecewise2` only;
  - `dsv4_index_short_context_fill` instead: `e0-dummy-none6144` only.

  Native cause of the second set: vLLM `model_executor/layers/sparse_attn_indexer.py:449-509` (native commit `2cf0a69`). When a prefill chunk has `local_total_seq_lens == 0` (2 rows at compress ratio 4 give 0 compressed keys), it only runs `topk_indices.fill_(-1)` and skips the scores and the prefill top-k. The short-context fill runs only eagerly, because the branch is disabled under capture (`models/deepseek_v4/attention.py:906-930`).

  So the rows record the scope set that was observed; they do not derive it from a rule. A later assignment predicts the scope set and must reproduce these sets.
- Dummy steps (`phase == "dummy"`):
  - They have empty `requests`.
  - They have the full native metadata: one prefill of `num_tokens` rows, `max_seq_len == num_tokens`.
  - Their dispatch is, for example, `{"cg_mode": "PIECEWISE", "num_tokens": 2, "num_reqs": null, "num_tokens_across_dp": [2, 2]}`.
- Other step shapes:
  - First-tail cases b3, b7 and b10 are mixed batches (one first-tail prefill plus decodes).
  - Decode cases have a different request signature in each of their 8 steps, because the contexts grow.
- Expected grouping, an independent lead count with the key below (`/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/expected_rows.py`, read-only):
  - **505 rows from 1925 layer calls**;
  - by phase: decode 400, dummy 50, first_tail 50, prefill 5;
  - by `graph_runtime_mode`: FULL 345, PIECEWISE 150, NONE 10.
- A4 baseline at HEAD `d2a226fe`:
  - report: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_report_d2a226fe.json` (verdict PASS, 125 groups);
  - written by `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/a4_baseline.py <output.json>`, which calls `check_composition` without writing into the analysis copy.

### New module `frontier/profiling/attention/dsv4_replay_rows.py`

- **Public entry:**
  - `DSV4_REPLAY_ROW_SCHEMA_VERSION = "dsv4_attention_replay.v1"`;
  - `build_dsv4_replay_rows(plan: Dsv4ProfilePlan, records: Sequence[ReplayRecords]) -> list[dict[str, Any]]`.
  - The module docstring follows `dsv4_attention_rows.py` in explaining what a row aggregates. All times are milliseconds.
- **Inputs:** it uses `build_layer_calls` and `activity_gaps` of `frontier/profiling/kernel_gap/dsv4_replay_gaps.py`. It does not use `index_records`, `require_scope_timing` or `expected_scopes`.
- **Validation (raise `ValueError` that names the source file and, where applicable, the case, step and layer):**
  - `plan.limits.engine_policy` is `graph_replay`.
  - Engine indices are unique across the records.
  - Every engine:
    - has `cudagraph_capture.cudagraph_mode == plan.limits.cudagraph_mode`;
    - has a layer inventory equal to the plan's (global layer id, variant, `max(1, compress_ratio)`) and the plan's target TP. This check moves into a shared function, see "Shared plan check" below;
    - passes `require_realized_kv_cache_groups(plan, engine)`.
  - All engines carry the same routing identity, as `build_dsv4_rows` requires.
  - Every step and call record has `measurement == KERNEL_ONLY`.
  - Every step has exactly one layer call per plan layer.
  - Every call scope is a family operator of the layer's variant and is not in `WARMUP_ONLY_OPERATORS`.
- **Per-call timing:**
  - from the shared function of "Shared per-call timing" below: per scope, the sum of its kernels' signed gaps in ms, and the layer window in ms;
  - a call with an exclusion reason (no owned kernel, no dependency end, not host-ahead, non-positive window) raises, naming source, case, step, layer and reason. This is the measurement contract of a held replay (the step hold makes the host run ahead), so a non-host-ahead call means the measurement is invalid, not that it should be skipped.
- **Row key:**
  - engine index, case id, variant, planner role (`plan.layer(id).plans_tile_schedule`);
  - the step dispatch (`cg_mode`, `num_tokens`, `num_reqs`);
  - the request signature (`dsa_rows._request_signature`; an empty tuple pair for a dummy);
  - the canonical variant metadata (`dsv4_attention_rows._variant_metadata` and `canonical_json`);
  - the frozen call scope set.
- **Columns:**
  - `measurement_type` (`KERNEL_ONLY`), `case_id`, `phase`, `attention_variant`, `compress_ratio` (`max(1, ratio)`), `plans_tile_schedule`;
  - `engine_cudagraph_mode` (engine capture mode), `graph_runtime_mode` (dispatch `cg_mode`), `padded_rows` (dispatch `num_tokens`), `dispatch_num_requests` (dispatch `num_reqs`, empty when null);
  - `indexer_short_context` (`LayerCall.short_context`);
  - `index_scores_executed`: true when `dsv4_index_scores` is in the scope set of a CSA row; false for other variants;
  - `dsv4_attention_rows._model_columns(plan, engine)`;
  - non-dummy rows only:
    - the request load columns (see the refactor below);
    - `_key_length_columns(plan, layer, max_key_len, index_scores_executed)` with the observed `index_scores_executed`;
    - dummy rows leave these columns empty, because their layout is carried by `padded_rows`, the phase and the metadata columns;
  - `<source>_metadata.<name>`, as in the eager rows;
  - `global_layer_ids`, `num_layers_sampled`, `num_steps_sampled`;
  - `operator_scopes`: canonical JSON `{scope: [scope]}` over the row's scopes in family-operator order. In graph replay each call scope is one family operator; the same format as the eager rows lets one reader handle both;
  - per scope in family-operator order:
    - `time_stats.<scope>.{min,max,mean,median,std,count,kernel_count}`, computed by `dsa_rows.scope_columns` from call `time_ms` and `kernel_count`;
    - `gap_stats.<scope>.{min,max,mean,median,std,count}` from the per-call scope gap sums;
  - `layer_window_stats.{min,max,mean,median,std,count}` from the per-call windows.
  - Use one statistics function for the three column families. Generalize `dsa_rows._stat_columns` by a column prefix if needed; DSA and eager V4 output must stay byte-identical.
- **Row order:** (case id, variant, not `plans_tile_schedule`, the smallest step index of the row).

### Request load refactor in `frontier/profiling/attention/dsa_rows.py`

- Split the load half of `batch_load_columns` into a public `request_load_columns(requests)`. It keeps the decode-token check and the load dict.
- `batch_load_columns` calls it and keeps its context half and its pure-prefill-or-pure-decode error.
- The replay rows use `request_load_columns` only, so mixed first-tail batches get load columns.
- DSA and eager V4 rows are unchanged.

### Shared per-call timing in `frontier/profiling/kernel_gap/dsv4_replay_gaps.py`

One function computes, for one layer call and its attention-owned `ActivityGap`s:

- the exclusion reason, or none;
- the window `(max kernel end - first kernel start + first kernel gap_us) * US_TO_MS`;
- per scope, the sum of `gap_us * US_TO_MS` over its kernels.

It follows the current A4 logic exactly (`dsv4_composition_check.py:136-159`, including the order of the exclusion checks). Exclusion order:

1. no owned kernel → `no_owned_kernel`;
2. any owned activity without a dependency end → `NO_DEPENDENCY`;
3. any owned activity not host-ahead → `HOST_BOUND`;
4. non-positive window → `non_positive_target`.

The rest of the change:

- The reason constants move to `dsv4_replay_gaps`.
- A helper that indexes the attention-owned gaps by (source, case, step, layer) may join them, if both callers need it.
- `check_composition` uses the shared function. Its report on package 03 must be byte-identical to the baseline.
- The row builder uses the same function and raises on any reason.

### Shared plan check

- Add a function to `frontier/profiling/attention/dsv4_attention_plan.py`, next to `require_realized_kv_cache_groups`, that requires an engine record's layer inventory (global layer id, variant, `max(1, compress_ratio)`) and target TP to equal the plan's.
- Both `dsv4_composition_check.planner_roles` and the row builder call it.
- Keep the docstring note about the ratio convention.
- The A4 error text may change; keep its tests meaningful.

### Aggregate command (`frontier/profiling/attention/dsv4_attention_main.py`)

Select by `plan.limits.engine_policy`:

- **eager_chunked:** unchanged.
- **graph_replay:**
  - read with `read_replay_records(raw_paths)`;
  - apply the same missing-engine check;
  - build the identity with:
    - `_engine_identity_fields([record.engine for record in records])`;
    - `schema_version=DSV4_REPLAY_ROW_SCHEMA_VERSION`;
    - `graph_mode=plan.limits.cudagraph_mode`;
    - the same `parallel_groups`;
  - write `build_profile_method_output_path(..., op_name="attention_graph_replay", profile_method=_PROFILE_METHOD_BY_MEASUREMENT[KERNEL_ONLY])`. Keep the op name as one module constant; the file is `attention_graph_replay_kernel_only.csv`;
  - write the summary JSON as now.
- Update the module docstring for the graph-replay output.

### Not in scope

- The predictor (entry 1), the gap reader (entry 10), the trained composition path (entry 3) and the scope-set prediction (entry 4).
- Changes to `compose_attention_phase`.
- Any change to the eager rows' format.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `d2a226fe`).
  - Untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`: leave them alone.
  - The lead edits task records under `task_memory/`: do not write there.
- **Writable paths:**
  - `frontier/profiling/attention/`: change only:
    - `dsv4_replay_rows.py` (new);
    - `dsa_rows.py`;
    - `dsv4_attention_plan.py`;
    - `dsv4_attention_main.py`;
  - `frontier/profiling/kernel_gap/`: `dsv4_replay_gaps.py` only;
  - `tests/comparison/calibration/`: `dsv4_composition_check.py` only;
  - `tests/unit/`:
    - a new `test_dsv4_replay_rows.py`;
    - updates to `test_dsv4_composition_check.py`, `test_dsv4_replay_gaps.py` and `test_dsa_attention_profiling.py` where the refactor requires them.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b6/`.
  - For the real-data aggregate, make a scratch raw directory with symlinks to the two raw files. The aggregate writes its summary JSON into the raw directory, and the analysis copy must stay unchanged.
- **Restrictions:**
  - No Git commands that change state; the lead commits.
  - No GPU, network, package installs, deletions or moves.
  - Python 3.10 syntax.
  - No torch, vllm or cuda imports at module load.
  - Keep modules at or below 2,000 lines.
  - No magic literals in code paths: name constants for column prefixes, the op name and the schema version.
- **Read-only data:**
  - The raw files above.
  - The bridge config `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge_config.json`, with `plan_args`, `runs` and `cudagraph_capture_sizes`. Build the plan flags with `dsv4_graph_bridge_check.graph_mode_flags("FULL_AND_PIECEWISE", sizes)`.
  - The lead scripts in `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b6_baseline/`.
- **Environment:**
  - `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vllm);
  - `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`;
  - `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`;
  - `ulimit -v 31457280`;
  - a `timeout` on every command. One pass over the package 03 records takes about a minute.

## Acceptance

1. **New unit tests** (`tests/unit/test_dsv4_replay_rows.py`, synthetic records in the style of `test_dsv4_replay_gaps.py` and `test_dsv4_composition_check.py`). They cover:
   - one row per key, separated by planner role, dispatch, request signature, metadata and scope set;
   - time, gap and window statistics on known values, including a negative signed gap;
   - for a single-call row, `compose_attention_phase` of (cost + gap) per scope, with zero class gaps, equals its window;
   - the dummy row (no request or key-length columns, `padded_rows` set);
   - mixed first-tail load columns;
   - observed `index_scores_executed` and `indexer_short_context`;
   - each validation error: wrong policy, wrong engine mode, layer mismatch, a CUDA_EVENT record, a missing layer call, a non-family or warmup-only scope, a non-host-ahead activity, mixed kernel counts;
   - the graph-replay aggregate writing `attention_graph_replay_kernel_only.csv` with identity columns `schema_version=dsv4_attention_replay.v1` and `graph_mode=FULL_AND_PIECEWISE`.
2. **Regression suites, all passing:**
   - `tests/unit/test_dsv4_composition_check.py`, `test_dsv4_replay_gaps.py`, `test_dsv4_graph_bridge_check.py`, `test_dsv4_kernel_identity_check.py`, `test_dsv4_composition.py`, `test_dsv4_graph_replay_profiling.py`, `test_dsv4_attention_profiling.py`, `test_dsv4_kv_cache_groups.py`;
   - `test_dsa_attention_profiling.py`, `test_dsa_extend_profiling.py`, `test_dsa_full_dummy_profiling.py`, `test_dsa_query_one_profiling.py`, `test_dsa_predictor.py`.

   Report the counts. The two known failures elsewhere are `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` and `test_profiling_runtime_boundary.py::test_runtime_predictors_use_only_allowlisted_profiling_helpers`. Run the second of them too: it checks which profiling helpers the runtime predictors import. Report whether its failure is unchanged.
3. **A4 unchanged:** `a4_baseline.py <scratch>/a4_report.json` after the change is byte-identical (`cmp`) to `a4_report_d2a226fe.json`.
4. **Real data:** `dsv4_attention_main aggregate` on the scratch raw directory, with:
   - plan args = the config's `plan_args` + `graph_mode_flags(...)`;
   - `--output_root <scratch>/profiling --device h200 --source_revision 230239c8 --model_config_revision package_03`.

   It writes 505 rows:
   - by phase: decode 400, dummy 50, first_tail 50, prefill 5;
   - by mode: FULL 345, PIECEWISE 150, NONE 10;
   - the sum of `layer_window_stats.count` over the rows is 1925 (one per layer call).
5. **Row composition** (scratch script, reported, not committed):
   - for every row, compose the medians: `compose_attention_phase({s: time_stats.s.median + gap_stats.s.median}, kernel counts, zero class gaps, variant=, graph_runtime_mode=, rows=padded_rows, multi_stream_gemm_token_threshold=, indexer_short_context=)`;
   - compare the result with `layer_window_stats.median`;
   - report the median, p90 and maximum absolute relative error over rows, and the rows above 5%;
   - expected: about 0 for single-call rows, and median at or below 5% and p90 at or below 10% over all rows (the A4 limits).

   If the limits fail, report the rows and their call spreads. Do not change the statistics to meet them.
6. **Imports:** cold imports of `frontier.profiling.attention.dsv4_replay_rows` and `frontier.profiling.attention.dsv4_attention_main` exit 0 without torch, vllm or cuda modules loaded.

## Expected return

- Report language: English.
- Changed paths, each with a summary.
- Executed checks with their output tails.
- Scratch paths, deviations and their reasons, and limits.
- If the real-data counts or the composition differ from "Acceptance", report the differing rows with their keys. Do not tune the key to reach the numbers; return the evidence.
