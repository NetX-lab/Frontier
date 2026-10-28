# Task: the V4 model-runner operators — owned step scopes in the graph-replay runner, direct block-operator calls on a native-layout engine, and the G2h runner package plan (stage 2 assignment B13r, file entry 15 part 2, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B13r of stage 2 ([design.md](design.md) §3.1, §3.3 and §6 entry 15; [c1_coverage.md](c1_coverage.md) §3.2, §3.3 and §4.2; G1h-02 result in [../test_report_2026-10-28_g1h_worker_import_fix.md](../test_report_2026-10-28_g1h_worker_import_fix.md)). |

## Goal

Design entry 15 lists four operators that run on objects of the V2 model runner, outside the model forward:

| Operator | Native work (pinned vLLM 0.28.0) |
| --- | --- |
| `kv_cache_block_zero` | `KVBlockZeroer.zero_block_ids` (`vllm/v1/worker/utils.py:211-232`), called in `update_requests` (`vllm/v1/worker/gpu/model_runner.py:1037-1038`): one `_zero_kv_blocks_kernel` launch, grid (block ids, segments, chunks per page), plus the host-to-device copy of the ids |
| `dsv4_block_table_write` | `BlockTables.apply_staged_writes` (`vllm/v1/worker/gpu/block_table.py:122`, called at `model_runner.py` in `execute_model` after `update_requests`): one fused `_apply_write_kernel` launch, one program per (group, row) write, plus the host-to-device copies of the contents (`vllm/v1/worker/gpu/buffer_utils.py:180-280`) |
| `dsv4_input_preparation` | the post-DP preparation of `execute_model` (`model_runner.py` about 1471-1600): `prepare_inputs`, `prepare_attn`, `model_state.preprocess_state`, `model_state.prepare_attn`, `model_state.prepare_inputs` |
| `dsv4_sampling` | `sample()` without `compute_logits` (`model_runner.py:1333-1365`: the logits-row gather `hidden_states[logits_indices]` and the sampler) and `postprocess_sampled` (`model_runner.py:1367-1398`: `post_update` and `model_state.postprocess_state`) |

B13 (request `2c58b7b1`) measures the standalone operators (mHC, shared expert, embedding, final norm, LM head shard). B13r measures these four, on 1 H200, on native runner objects. It delivers, all on CPU:

1. **Owned step scopes in the graph-replay runner** (`frontier/profiling/attention/dsv4_graph_replay_runner.py`). Every recorded step attributes the kernels of these four operators to their own scope, so a replay trace has no unowned eager work outside the model forward except what the report lists.
2. **Direct block-operator calls on a native-layout engine.** The zeroing and the block-table writes measured on the C1 grids, on an engine whose KV-cache layout equals the native TP8 rank.
3. **Registry.** `dsv4_block_table_write`, `dsv4_input_preparation` and `dsv4_sampling` registered through the unchanged registry; `kv_cache_block_zero` (existing, `KV_CACHE_FAMILY`) becomes a profiling target.
4. **G2h runner package plan.** A `g2h_runner/` package plan and builder for two phases (the replay with the new scopes, and the direct calls), the expected rows, a native kernel note per operator, and the GPU time.

The lead commits the code, builds the package from the commit and submits it through /ds4-exec, then compares the kernels with native.

## Approach and decisions

### Why two measurement paths (observed)

- **G1h-02 replay** (`exp-1029-020008-235508`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/`): the reduced engine has the 11 leading layers. Its KV-cache groups are the native 5 ([14, 6, 5, 10, 4] layers per group; engine records in `raw/dsv4_raw_e{1,2}.jsonl`). So the per-group work has native shapes: `_apply_write_kernel` grid (1), (2), (5), `_gather_block_tables_kernel` (5, b), `_compute_slot_mappings_kernel` (5, …), and the sampler kernels over the full vocabulary (`_gumbel_sample_kernel` (b, 127, 1)) (`traces/g1-e0-0000-full.json`, `traces/g1-e2-1020-multi_tail.json`).
- **The zeroing is not native there.** The zeroer's segment table has one segment per distinct KV tensor, outer index and virtual block (`utils.py:139-199`), so it grows with the layer count: the replay launches `_zero_kv_blocks_kernel` with grid (2520, 30, 10) and (5, 30, 10), while the native P2 runs use (n, 139, 10) (native dumps in `h3o_operator_costs/evidence/v4_operators/`).

Hence:

- **Path A (replay).** The new owned scopes run in the replay of the full G1h case list. These rows are the costs of `dsv4_input_preparation` and `dsv4_sampling`, and the validation rows of `dsv4_block_table_write`. The zeroing rows of path A record their segment count; they are diagnostic only.
- **Path B (direct calls).** One engine with the native layer count (43) and the native KV-cache layout. The runner's own `kv_block_zeroer` and `block_tables` are called on the C1 grids. These rows are the costs of `kv_cache_block_zero` and `dsv4_block_table_write`.

### Existing mechanism (read first)

- **Graph-replay runner** (`dsv4_graph_replay_runner.py`, 561 lines):
  - `_wrap_step` opens the step range at `execute_model`, runs the host-ahead hold (`torch.cuda._sleep`), and `_wrap_sample_tokens` closes the range after `sample_tokens`. So every kernel of these four operators is already inside a recorded step range, but unowned.
  - Scopes are `(layer, scope)` owners checked against `SCOPE_TREE`; `_scope_call` → `case_runner._require_state().timed_call` (CUDA events for CUDA_EVENT; `record_function` annotations for KERNEL_ONLY; `annotate=False` for containers, because kernel-only annotations must not nest — `dsa_vllm_runner.py:271-303`).
  - `graph_replay_calls` (`dsa_vllm_runner.py:617-658`) attributes graph and eager intervals to owners; owner `None` intervals are kept.
  - The bridge checks (`g1h/package_02/dsv4_graph_bridge_check.py`, `profile_dsv4_prefill.py`), the rows builder (`frontier/profiling/attention/dsv4_attention_rows.py`) and the gap tool (`frontier/profiling/kernel_gap/dsv4_replay_gaps.py`) read these records.
- **Engine builder:** `dsv4_attention_runner.build_dsv4_engine` → `case_runner.build_engine` with `build_dsv4_hf_overrides` (`dsv4_attention_plan.py:408-434`: the measured layers, the per-rank heads and `o_groups`; the indexer heads and MoE layers stay native).
- **Package mechanism:** `h3o_operator_costs/g1h/prepare_package.py` reuses `p1h_bridge/prepare_package.py` with its own `package_plan.json` (base package, image, `cuda_compat_path`, `cloud_mount`, `step_main`, H200, the case lists of the three engines).
- **Consumer (do not change):** `frontier/execution_time_predictor/deepseek_v4.py:30` (`pre_dp` phase owns `kv_cache_block_zero`), 320-335 (the zeroing workload: block ids per forward).

### Decisions

1. **Owned step scopes (path A).**
   - Add top-level step scopes (no layer parent) to the runner's scope tree: `kv_cache_block_zero`, `dsv4_block_table_write`, `dsv4_input_preparation`, `dsv4_sampling`, and the diagnostic `dsv4_compute_logits`.
   - Wrap the runner instance's own callables, as `_wrap_step` does: `kv_block_zeroer.zero_block_ids`; `block_tables.apply_staged_writes`; `prepare_inputs`, `prepare_attn`, `model_state.preprocess_state`, `model_state.prepare_attn`, `model_state.prepare_inputs` (all → `dsv4_input_preparation`, several calls per step); `sample` and `postprocess_sampled` (→ `dsv4_sampling`); `model.compute_logits` (→ `dsv4_compute_logits`). Check each attribute against the pinned source and report the file:line of each call site.
   - `compute_logits` runs inside `sample()`. At replay TP1 it is the full-vocabulary GEMM, not the native TP8 shard plus gather. Its kernels must not count in `dsv4_sampling`. Use the existing container mechanism (no nested kernel-only annotations). The sampler then sees full-vocabulary logits, as natively after the gather.
   - A step scope has no layer. Choose how its owner is identified in the records so that the attention rows, the bridge checks, the rows builder and `dsv4_replay_gaps.py` keep their results on the existing records. Say what you chose and where each reader handles it.
   - `dsv4_replay_gaps.py` now treats any interval with a scope as attention-owned (`ActivityGap.attention_owned`, line 134) and requires scope and layer to be both set or both unset (line 266). On the new records, step-scope intervals must stay in the classes they have today (outside work: `eager_preparation`, or the graph classes for dummy-graph work), so the four V4 gap values of a path A run are comparable with G1h-02. Report the four values of the path A run against G1h-02 as a lead check after the run (the lead runs it; add the command to the package README).
   - **Unowned report.** For each recorded step, classify every kernel and memcpy interval as attention scope, step scope, model forward (graph or eager, outside the attention scopes) or unowned. Add a summary per case kind (FULL, PIECEWISE, first tail, multi-tail, dummy, prefill) of the unowned names and grids to the run output. Do not attribute by kernel name.
2. **Direct calls (path B).**
   - **Engine.** The native layer count (43, from the model config), the per-rank attention dimensions of the replay engine, and the native KV-cache layout. The routed experts do not fit one H200 at 43 layers (about 3.2 GB per MoE layer). Reduce only the routed expert count, through the hf overrides, to a value that the grid file gives (no literal in code). Show from the source, with file:line, that the KV-cache specs, the KV-cache groups and the zeroer's segment table do not depend on the routed expert count, and that the dummy-weight finalizers and the balanced routing accept the reduced count (including the hash layers, `num_hash_layers`).
   - **Layout check.** The run records the zeroer's segment count, chunks per page and block size per segment, and the KV-cache groups. The builder's plan check states the expected values: segments and chunks from the native trace grid (139, 10), and the groups from the B4 rule for 43 layers. Compute the expected segment count on CPU from the native layout rule if you can, and say how; if it needs vLLM objects, say so and leave it to the run.
   - **Zeroing grid.** `zero_block_ids(ids)` with n = 1-32 and 2,520 distinct block ids (C1 §3.2 and §4.2). Say how the ids are chosen (for example consecutive free blocks) and why the cost does not depend on which ids.
   - **Block-table write grid.** Coordinates (w_c, r): w_c continuing (group, row) writes of one entry each, w_c = 0-32; r new requests, r = 0-4, each writing its full row with the native per-group entries for a 6,144-token first chunk (24, 96, 96, 1,536, 768; total 2,520); without (0, 0) (C1 §3.3 and §4.2). Stage them with `append_block_ids` as `add_requests` (`overwrite=True`) and `update_requests` (`overwrite=False`) do, then call `apply_staged_writes` once. Derive the per-group entries from the engine's group block sizes and the first-chunk length in the grid file, and check them against (24, 96, 96, 1,536, 768).
   - **Timing.** Both methods, as in the replay: CUDA events (`cuda_event`) and `record_function` traces (`record_function`, KERNEL_ONLY), with warm-up and repeats from the grid file, and the host-ahead hold before each measured call so that device time is not host-bound. Keep the traces. Record the kernel and the memcpy intervals separately; the consumer decides later what counts.
   - **Rows.** One row per (operator, coordinate, method): the coordinates, the layout fields of the engine, the repeat count and the time statistics, in the closest existing schema. Name the schema and where the consumer will read it.
   - **Module.** A new module under a new package `frontier/profiling/model_runner/` (for example `dsv4_block_operators.py`) with its own entry point; torch and vLLM imported only inside the functions that run on the worker. Reuse the replay runner's engine builder, hold calibration and trace helpers; no second engine builder.
3. **Grid file.** One JSON file for path B (`g2h_runner/block_operator_grid.json`): layer count, routed expert count, zeroing n list, (w_c, r) lists, first-chunk tokens, warm-up and repeat counts, and the expected layout values. The rows come from this file only.
4. **Registry.** Register `dsv4_block_table_write`, `dsv4_input_preparation` and `dsv4_sampling` through the unchanged registry, in the families that match their roles (`KV_CACHE_FAMILY` for the write is one option; say which and why), with `profiling_target=True` and `execution_enabled` unchanged (the consumer is later). Set `profiling_target=True` on `kv_cache_block_zero`. Find every user of these families and of `profiling_target`, and report each effect. A change that alters a GLM plan or another model's plan stops that part.
5. **Native kernel notes.** `g2h_runner/native_kernel_notes.md`: for each operator, the kernels that the pinned source launches (file:line) and the matching lines (name, grid, count per forward, median duration) of the native dumps in `h3o_operator_costs/evidence/v4_operators/` (`p2_32_decode_dp0.txt`, `p2_32_prefill_dp0.txt`, `p2_64_*`), and the same kernels in the G1h-02 traces named above. The table below is the lead's reading; correct it where the evidence disagrees.

   | Operator | Native kernels (P2 dumps) |
   | --- | --- |
   | `kv_cache_block_zero` | `_zero_kv_blocks_kernel` [1, 139, 10] 2.30 µs, [2, 139, 10] 3.33 µs, [2520, 139, 10] 2,907 µs |
   | `dsv4_block_table_write` | `_apply_write_kernel` grid 1 (4.8 µs), 2 (6.1 µs), 5 (7.0 µs decode, 4.51 µs prefill), plus HtoD copies |
   | `dsv4_input_preparation` | `_prepare_prefill_inputs_kernel`, `_prepare_pos_seq_lens_kernel`, `_combine_sampled_and_draft_tokens_kernel`, `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`, `_compute_swa_indices_and_lens_kernel`, `_compute_global_topk_indices_and_lens_kernel`, `_compressed_slot_mapping_kernel`, `indexSelectSmallIndex<int, int, …>` |
   | `dsv4_sampling` | the logits-row index kernels, `_bias_kernel`, `_gumbel_sample_kernel` [b, 127, 1], `_get_num_sampled_and_rejected_kernel`, `_post_update_kernel`, the argmax reduce, `scatter_gather` and `vectorized_gather` [2, 2, 1] |

6. **Path A case list.** The same three engines and the same 989 case ids as `g1h/package_02` (a G1h-02 repeat), so that the lead can also compare the attention rows of the two runs. Reuse the G1h plan arguments and case lists; do not regenerate the grid.

### Package plan

- Add `h3o_operator_costs/g2h_runner/` with `package_plan.json`, `block_operator_grid.json`, a builder (`prepare_package.py`, reusing the `p1h_bridge` builder as `g1h/prepare_package.py` does, plus the path B phase), and `expected_rows.json` (path B rows, and the path A case ids per engine).
- Phases: `G2H_RUNNER_REPLAY` (path A, the bridge over the three engines) and `G2H_BLOCK_OPERATORS` (path B, both methods), each with its own output directory.
- Settings: destination `package_01`; name `G2H_V4_RUNNER_OPERATORS`; job tag `v4-h200-g2h-runner-20261028-01`; output directory `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_01`; 1 H200, `step_main`, non-preemptible; the image, mount and `cuda_compat_path` of `g1h/package_plan.json`.
- `source_revision`: the current HEAD; the lead replaces it after committing B13r. Do not build the package.
- The builder's dry-run check requires the planned rows and case ids to equal `expected_rows.json`, and runs the import probe of the worker entry modules with `fasteners` and `sklearn` blocked.
- **GPU time.** Path A from G1h-02 (engine runs 2,711 s, job 3,055 s, 0.85 GPU-hours) plus the scope overhead; path B from the 43-layer engine start-up and the planned calls. Compare with the G1h-G4h cap of 7 (about 0.92 used, plus G2h from B13).

## Out of scope

- Any GPU, platform or network action. Building or submitting the package.
- The B13 operators and files (`frontier/profiling/linear_op/`, `fp8_block_linear.py`, `tensor_parallel_layers.py`, `deepseek_v4_axes.py`, `g2h/`), except reading them.
- The consumers (entries 1-12), the loaders and trainers, `execution_enabled`, and `frontier/execution_time_predictor/`.
- The communication operators and the logits gather; the output copy of `sample_tokens` (`AsyncOutput`, `model_runner.py` about 1780-1790) is reported as unowned, not priced.
- The G1h grid, its case construction and the G1h package files; the MoE and collective producers; H800 files.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree; B13 is committed before this request starts). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md) (§3.1, §3.3, §4, §6 entries 15 and 17), [c1_coverage.md](c1_coverage.md) (§3.2, §3.3, §4.2), [codex_assignment_B13_v4_operator_producer.md](codex_assignment_B13_v4_operator_producer.md) and its report, and [codex_assignment_B11_g1h_attention_windows.md](codex_assignment_B11_g1h_attention_windows.md) for the G1h case list.
- **Writable paths:**
  - `frontier/profiling/attention/` (the graph-replay runner, the rows builder and the engine builder, for the step scopes and the 43-layer override);
  - `frontier/profiling/kernel_gap/dsv4_replay_gaps.py` (reading the step scopes only);
  - `frontier/profiling/model_runner/` (new);
  - `frontier/operators/families.py` (the registry entries);
  - `tests/unit/`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/` (new).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b13r/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No deletions or moves.
  - Python 3.10. No magic literals in code paths: the grid, the expert count and the expected layout come from the grid file, the sizes from the model config. No guards for states the code cannot reach. No second implementation of a vLLM function and no copy of a runner method: call the runner's objects.
  - The profiler entry points must import without `fasteners` and `sklearn` (`tests/unit/test_profiling_runtime_boundary.py`).
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. No recursive grep or find over the vLLM tree or the trace directories; name the files. Read at most two G1h-02 trace files (they are 4-60 MB each).
  - Keep every module at or below 2,000 lines.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`.
  - Tests that need torch (CPU) use `PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages` with the same interpreter. The native registry checks also need `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.
  - The pinned vLLM source `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`) is read-only.

## Acceptance

1. **Existing records unchanged.** On the G1h-02 records (`/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e{0,1,2}.jsonl`), the rows builder, the bridge checks and `dsv4_replay_gaps.py` give the same outputs before and after the change (name the commands and compare the outputs byte for byte or field by field). The lead's reference for the gap tool: `kernel_gap.csv` values `cuda_graph` 0.49773414267747085, `cuda_graph_piecewise` 0.6249537992538311, `eager` 0.5446049595929223, `eager_preparation` 1.2208671118771937 µs, with sample counts 169,232 / 289,730 / 30,243 / 91,170 (`/data/ycfeng/tmp/g1h_fix/gaps_g1h02/`). A fixture record with step-scope intervals gives the same classes as the same record with those intervals unowned.
2. **Step scopes.** Unit tests with stand-in runner objects show: each wrapped callable opens its scope with no layer parent; `compute_logits` kernels are not in `dsv4_sampling`; a scope opened inside another fails as the existing `_scope_call` check does; the step range still opens at `execute_model` and closes after `sample_tokens`.
3. **Unowned report.** A fixture step with intervals of each class gives the expected classification and summary.
4. **Path B plan.** The grid `--dry_run` (or equivalent plan check) lists exactly the rows of `expected_rows.json`: 33 zeroing rows and 164 write rows per method, with the per-group entries checked against (24, 96, 96, 1,536, 768). Invalid grids (an n above the engine's free blocks, r above the request rows, an expert count below the routing top-k) fail with clear messages.
5. **Layout evidence** (decision 2): the file:line chain showing that the expert count does not reach the KV-cache layout, the expected segment count and how it was derived, and the dummy-weight and hash-layer findings.
6. **Registry.** The three new operators and `kv_cache_block_zero` are profiling targets; every other family and every non-V4 plan is unchanged (tests).
7. **Existing tests pass:** the attention and replay tests (`test_dsv4_*`, `test_*graph_replay*`, `test_*bridge*`, `test_v4_attention_coverage.py`), `test_profiling_runtime_boundary.py`, the operator and families tests, the kernel-gap tests, and any other test that imports the changed modules. Report tests that need torch or the native source and are skipped. The known failure `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` is deselected.
8. **New unit tests** for the scopes, the classification, the grid parser, the write staging plan and the registry. Report the count.
9. **G2h runner package plan** and the builder's dry-run check on the current HEAD (plan check and import probe only, no package written), with the phase list and the GPU time.
10. **Hygiene:** line counts; no unused imports (`/data/ycfeng/tmp/unused_imports.py`); a cold import of each changed or new entry module exits 0 and loads no `torch`, `vllm` or `cuda` module unless the existing module already does at load time.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- The results of Acceptance 1-10, with the commands, the expected and actual values, and the counts.
- The call sites wrapped (file:line in the pinned source), the owner identity chosen for step scopes, the expected path B layout, and the GPU-time estimate.
- Scratch paths, deviations and their reasons, and limits.
- If the native source or the runner contradicts a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Examples: the KV-cache layout depends on the expert count; the dummy finalizers or the hash routing reject a reduced expert count; a wrapped runner attribute is bound in a way that an instance wrapper cannot reach; `compute_logits` cannot be separated without nested kernel-only annotations.
