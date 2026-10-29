# Task: MoE rank classes, the V4 gate on SP-local rows and the G3h package plan (stage 2 assignment B12, file entry 16, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B12 of stage 2 (G3h, [c1_coverage.md](c1_coverage.md) §4.3; [design.md](design.md) §3.5 and §6 entry 16). |

## Goal

G3h measures the V4 routed MoE and the gate on 1 H200. The V4 replays of H200 P2-32 (DP2 / EP16, 16 local experts per rank) and P2-64 (DP4 / EP32, 8 local experts per rank) need:

- the routed path (`moe_shuffling`, `moe_grouped_gemm`) for every **rank class** of every gathered row count M of [c1_coverage.md](c1_coverage.md) §4.3, the zero class included;
- the gate (`moe_gating_linear`, `moe_gating_routing_topk`) on the rank's **SP-local rows L**, not on M.

The MoE producer measures only EP rank 0 today (`moe_wrapper.py:294-296`, `main.py:150`), and it runs the gate on M (`moe_wrapper.py:877`). The accepted H800 rows (`task_2026-10-23_v4_remaining_measurements/moe/`) have both limits.

B12 delivers, all on CPU:

1. **Rank selection.** The producer measures a chosen EP rank: its local expert population comes from that rank's expert window.
2. **Rank classes from the DP lane rows.** A new producer input gives the SP-local rows of every DP lane of a forward. The producer derives M, each rank's class and each rank's L, and measures one representative rank per distinct (class, L).
3. **Row identity.** Each row records the rank, the class, the gate rows and the balanced routing identity, so the consumer (entry 6, later) can select rows by class.
4. **G3h package plan.** A `g3h/` package plan and builder, with the expected rows and the GPU time.

The lead commits the code, builds the package from the commit and submits G3h through /ds4-exec.

## Approach and decisions

### Existing mechanism (read first)

- **Producer:** `frontier/profiling/moe/main.py` (`profile_model`, `_worker_profile_task`, `_dry_run`, `_validate_canonical_moe_result_df`) and `moe_wrapper.py`:
  - `_build_local_expert_map` and `_compute_local_expert_token_counts`: the local expert window is `[ep_rank × local, (ep_rank + 1) × local)`;
  - `_prepare_routing_inputs`: for EP > 1, routing is generated over the global M rows and all experts, and the rank executes only its window through the expert map, as native EP does;
  - `profile()`: gating on `num_tokens`, then shuffling and grouped GEMM on the same routing sample;
  - `_build_profile_result`: the row and its typed operator contracts.
- **Routing:**
  - `load_distribution.generate_expert_routing(..., "balanced")` calls `moe_impl.uniform_topk`. Global row i goes to experts (i × k + j) mod E.
  - The native strategy is `frontier/profiling/moe/balanced_routing.py` (`balanced_topk`). Its bytes are pinned by `BALANCED_ROUTING_SHA256` in `frontier/profiling/attention/dsv4_attention_rows.py:56`, so **do not edit that file**.
  - With per-rank offsets, `balanced_topk` gives the same ids over the gathered rows.
- **Frontier routing algebra:** `frontier/moe_ep_workload.py`, with `materialize_expert_token_counts` (Hamilton apportionment, expert-id tie order), `build_contiguous_expert_ownership` and `materialize_layer_ep_workload`. Design §3.7 and `tests/unit/test_balanced_routing.py` show that these equal native `balanced_topk` + `determine_expert_map` per expert and per EP rank, for the DP2 / EP16 and DP4 / EP32 widths.
- **Row contract:** every row of a canonical MoE CSV carries all four target columns (`_validate_canonical_moe_result_df`, `moe_dataset_training._validate_moe_columns`). Keep that contract.
- **The H800 package** (the model for G3h): `task_2026-10-23_v4_remaining_measurements/moe/`:
  - `build_package.py` and `profile_plan.json`;
  - results: `results/primary_acceptance.json` and `summary.md`. The single-H800 job ran 92 s for 2 methods × 4 rows, with backend `MXFP4_MARLIN` and 7 kernels per grouped-GEMM call.
- **The H200 package mechanism:** `h3o_operator_costs/p1h_bridge/prepare_package.py` and `package_plan.json`. They give the base package of the H200 launcher, the image, `cuda_compat_path`, `cloud_mount`, `step_main` and the H200 fields.
- **Native V4 MoE (pinned source):**
  - `vllm/models/deepseek_v4/nvidia/model.py:571-577`: `GateLinear`, `out_dtype=torch.float32`;
  - `vllm/model_executor/layers/fused_moe/expert_map_manager.py:22` (`determine_expert_map`);
  - placement strategy default `linear` (`vllm/config/parallel.py:178`).

  The H200 P2-32 census (`evidence/runtime_census/p2_32.txt:223-254`) shows `marlin_moe_wna16::Marlin`, `moe_align_block_size_kernel` and `moe_sum_vec_kernel` per layer.

### Decisions

1. **Rank input.**
   - The wrapper takes the EP rank as a constructor argument, checked against `0 <= rank < expert_parallel_size`.
   - `_get_profiling_ep_rank` is removed; the expert map uses the given rank.
   - `_worker_profile_task` and the Ray path pass the rank of the task. The existing grid path passes 0, so its rows and dry-run plan stay as they are (Acceptance 1).
2. **DP lane rows input.**
   - Add one producer argument, for example `--dp_lane_local_rows`, taking one comma-separated vector per forward. Each vector holds the SP-local rows L_d of each DP lane.
   - With it, require:
     - exactly one EP size;
     - `--num_tensor_parallel_workers 1`;
     - `--load_distributions balanced`;
     - `--routing_runtime_path balanced_topk`;
     - EP divisible by the vector length D.
   - Ranks per lane are EP / D. Ranks are DP-major: rank r belongs to lane floor(r / (EP / D)). M = (EP / D) × Σ L_d.
   - Any lane can own any of the rows (native DP load balancing places a request on any lane), so the planner takes every distinct permutation of each vector.
   - For each permutation and rank, it computes the rank's local count vector with the Frontier algebra of `moe_ep_workload.py`, not a second implementation. The pair is (class, L of the rank's lane).
   - The planner measures one task per distinct (EP, M, class, L), on the lowest rank that has it.
   - This input replaces `--num_tokens_list` for that invocation. Reject the two together.
3. **Class.**
   - The class is C1's triple:
     - the rows routed to the rank (the sum of its local counts);
     - the local experts with one extra row;
     - the base rows per expert, floor(M × k / E).
   - It is computed in one function in `frontier/moe_ep_workload.py` (or a new torch-free module beside it). The function returns each rank's triple for given (M, k, E, EP), so the entry 6 consumer can import the same function later.
   - The producer checks that the counts from its generated routing ids (`_compute_local_expert_token_counts`) equal the algebra's counts for the rank. On a mismatch it fails.
4. **Gate on L.**
   - In a lane-rows task, the gate is measured on L rows and the routed path on M rows, in the same row. The row keeps all four target columns.
   - Add a `gating_num_tokens` column to every row. It equals `num_tokens` on the grid path, so all rows have one schema.
   - Check how `profile()` uses `num_tokens` for the gate input and the routing sample. Change only the gate input size.
5. **Balanced identity ("balanced path signature").**
   - With `routing_runtime_path balanced_topk`:
     - the routed ids of the grouped-GEMM and shuffling sample come from `balanced_routing.balanced_topk` over the M rows with offset 0. These are the same ids as `uniform_topk`; use one function for both.
     - Each row records `routing_strategy_sha256`, the SHA-256 of `balanced_routing.py`. The producer fails unless it equals `BALANCED_ROUTING_SHA256`. Import the constant; do not copy it.
   - Name the columns the same way as the attention rows (`routing_columns` in `dsv4_attention_rows.py`) where the meaning is the same.
6. **Row identity columns.** Add `ep_rank` and the three class columns (plain names, for example `local_routed_rows`, `local_experts_with_extra_row`, `base_rows_per_expert`). Write them on every row. Rank 0 rows on the grid path get the triple too, computed by the same function.
7. **The zero class is measured.**
   - Confirm from the producer and pinned native source that a rank with no local rows runs the same kernel sequence as native: `moe_align_block_size`, the two Marlin GEMMs and `moe_sum`, with zero local rows.
   - If either side returns early for zero local rows, stop that part. Report the code lines and the options.
8. **Gate parity check.**
   - Compare the producer's gate (`MoEGatingNetwork`, its `linear_backend`, dtypes and output dtype) and its `balanced_topk` routing step with the native V4 `GateLinear` (`out_dtype=torch.float32`) and the `RoutingSimulatorRouter` path.
   - Report any difference with file:line evidence. Do not change the gate implementation in B12.

### G3h grid (C1 §4.3)

- **EP16, DP2, 16 local experts.** Lane vectors:
  - (1, 1), (2, 2) and (768, 768);
  - (768, 1), whose permutations are taken by the planner;
  - margins (3, 3) and (4, 4).
- **EP32, DP4, 8 local experts.** Lane vectors:
  - (1, 1, 1, 1), (2, 2, 2, 2);
  - (768, 1, 1, 1), (768, 768, 1, 1), (768, 768, 768, 1) and (768, 768, 768, 768);
  - margins (3, 3, 3, 3) and (4, 4, 4, 4).
  - (768, 768, 1, 1) gives M12304, the reachable but unobserved margin of C1.
- **Expected M and classes.** The planned (M, class) set must equal the C1 §4.3 table, observed rows and margins. L takes {1, 2, 768} plus the margins {3, 4}.
- **Methods:** `cuda_event` (`moe.csv`) and `record_function` (`moe_kernel_only.csv`), as the H800 package.

### Package plan

- Add a new directory `h3o_operator_costs/g3h/` with `package_plan.json` and a builder (for example `prepare_package.py`). The builder:
  - follows the H800 `moe/build_package.py` for the MoE producer commands, the per-method phases and the dry-run plan check;
  - uses the H200 fields and the launcher base of `p1h_bridge/package_plan.json`.
- Settings:
  - destination `package_01`;
  - name `G3H_MOE`;
  - job tag `v4-h200-g3h-moe-rank-classes-20261028-01`;
  - output directory `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_moe/g3h_01`;
  - `--device h200`.
- If the launcher base cannot run a plain `python -m frontier.profiling.moe.main` command on H200 as the H800 package did, report why. Name the base you chose and the reason.
- The builder requires a committed `source_revision`. Set it to the current HEAD, and say in the report that the lead replaces it after committing B12. Do not build the package.
- In the builder's dry-run check, the producer's planned tasks must equal the expected (EP, M, class, L, rank) list. Write that list into the package plan or a file beside it.
- **GPU time.** Estimate from the H800 job (92 s for 8 rows, including start-up) and the planned task count. Compare it with C1's G3h estimate of 2 GPU-hours, inside the G1h-G4h cap of 7.

## Out of scope

- Any GPU, platform or network action. Building or submitting the package.
- The consumer: the V4 FFN pricing (entry 6), the loader and the trainer. The shared expert and other V4 operators (entry 15). Kernel gaps (entry 17).
- H800 G3. The same producer and builder must work for H800 later, but write no H800 files now.
- Changes to `balanced_routing.py`, the attention producer, the census, the B10 files and the B11 files.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Concurrent work:** assignments B10 (`tests/analysis/v4_collective_kernels.py`, `tests/analysis/v4_runtime_census.py`, `tests/unit/test_v4_collective_kernels.py`, `evidence/collective_kernels/`) and B11 (`frontier/profiling/attention/`, `tests/analysis/` attention coverage files, `g1h/`) may be uncommitted or running. Do not edit their files.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md) (§3.5, §3.7, §6 entry 16), [c1_coverage.md](c1_coverage.md) (§4.3, §5).
- **Writable paths:**
  - `frontier/profiling/moe/` (except `balanced_routing.py`);
  - `frontier/moe_ep_workload.py` (additions only), or a new torch-free module beside it;
  - `tests/unit/`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/` (new).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b12/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No deletions or moves.
  - Python 3.10. Keep modules at or below 2,000 lines: `main.py` has 1,174, `moe_wrapper.py` 914 and `moe_ep_workload.py` 898.
  - No magic literals in code paths. The lane vectors come from the CLI and the package plan; E, k and the local width come from the model config. No guards for states the code cannot reach. No second routing or class implementation.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`.
  - Tests that need torch (CPU) use `PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages` with the same interpreter, as in progress.md (`test_balanced_routing.py`, 102 passed). The native registry checks also need `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.
  - The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Acceptance

1. **Grid path unchanged.** The producer's `--dry_run` with the H800 package arguments (`moe/profile_plan.json` `common_args`, both methods) gives plan JSON equal to `moe/h800_moe_local_01/plans/{cuda_event,record_function}.json`. If a plan field must change, list it and say why.
2. **Classes.**
   - The class function reproduces every class of the C1 §4.3 table, observed and margin, for EP16 and EP32.
   - For every permutation of every G3h lane vector, its per-rank counts equal native `balanced_topk` + `determine_expert_map` from the pinned source. Extend `test_balanced_routing.py` or add a test beside it.
3. **Producer routing (CPU torch).**
   - For a rank of each class, the local counts that `_prepare_routing_inputs` generates equal the algebra's counts.
   - The routed ids equal `balanced_topk` with offset 0.
   - The gate input has L rows and the routed sample M rows.
   - Test this on CPU tensors where the code allows. Name any part that needs CUDA and is therefore left untested.
4. **G3h plan.** The lane-rows `--dry_run` for EP16 and for EP32 lists exactly the expected (M, class, L, rank) tasks. Report the task counts per EP and in total.
5. **Existing tests pass:** `tests/unit/test_dsv4_moe_profiling.py`, `test_balanced_routing.py`, `test_moe_profiling_output_metadata.py`, `test_moe_load_distribution_contract.py`, `test_moe_routing_input_contract.py`, `test_moe_routing_runtime.py`, `test_moe_gating_constructor_boundary.py`, `test_moe_ep_workload_materializer.py`, and any other test that imports the changed modules (find them with grep). Report a test that needs torch or the native source and is skipped in an environment.
6. **New unit tests** for the rank input, the lane-rows planner, the permutations and deduplication, the rejected argument combinations, the row identity columns and the routing identity check. Report the count.
7. **Findings** for decisions 7 (zero class) and 8 (gate parity), with file:line evidence.
8. **GPU time** for G3h, as above.
9. **Hygiene:**
   - line counts;
   - no unused imports (`/data/ycfeng/tmp/unused_imports.py`);
   - a cold import of `frontier.profiling.moe.main` and of the class module exits 0 and loads no `torch`, `vllm` or `cuda` module, unless the existing module already imports torch at load time. In that case say so, and do not change it.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- The results of Acceptance 1-9, with the commands, the expected and actual values, and the counts.
- The expected G3h task list (EP, M, class, L, rank).
- Scratch paths, deviations and their reasons, and limits.
- If the native source or the producer contradicts a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Examples: native placement is not contiguous for these engines; a zero-row rank skips kernels; the gate runs on rows other than L.
