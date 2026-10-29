# Task: the V4 standalone operators in the linear-op producer, the shared-expert FP8 binding and the G2h package plan (stage 2 assignment B13, file entry 15 part 1, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B13 of stage 2 (G2h, [c1_coverage.md](c1_coverage.md) §3.2-3.5 and §4.2; [design.md](design.md) §3.1 and §6 entry 15). |

## Goal

G2h measures the V4 operators outside attention and the routed MoE on 1 H200, for the replays of H200 P2-32 (TP8 / DP2 / EP16) and P2-64 (TP8 / DP4 / EP32). Design entry 15 lists them. They split into two groups by what a measurement needs:

- **Standalone operators (this assignment).** Each is one call of a pinned vLLM function or layer on given rows, independent of the engine state:
  - the shared expert (`share_expert_up_proj`, `share_expert_act`, `share_expert_down_proj`);
  - the mHC scopes `dsv4_mhc_first_pre`, `dsv4_mhc_ffn_post_attn_pre`, `dsv4_mhc_attn_post_ffn_pre`, `dsv4_mhc_final_post`, `dsv4_mhc_head`;
  - `dsv4_final_norm`, `dsv4_embedding` (local vocab shard, without its all-reduce) and `dsv4_lm_head` (local vocab shard, without the logits gather).
- **Engine operators (assignment B13r, later).** `kv_cache_block_zero`, the block-table writes, the post-DP input preparation and the sampling run on objects of the V2 model runner (`KVBlockZeroer`, the block tables, the input batch and the sampler). They are measured through the graph-replay runner and are out of scope here.

B13 delivers, all on CPU:

1. **Shared-expert FP8 binding.** The producer builds the block-FP8 linear method from the checkpoint's own vLLM quantization config, so the V4 shared expert uses the native `DeepseekV4FP8Config` (e8m0 weight scales) instead of a plain `Fp8Config`.
2. **V4 operator plans.** The linear-op producer measures the operators above on the row grid of one JSON grid file, with typed operator contracts and the row axis of each operator.
3. **Registry.** The eight `DEEPSEEK_V4_MHC_FAMILY` operators become profiling targets.
4. **G2h package plan.** A `g2h/` package plan and builder, the expected rows, a native kernel note per operator, and the GPU time.

The lead commits the code, builds the package from the commit and submits G2h through /ds4-exec. The lead also compares the G2h kernels with native after the run.

## Approach and decisions

### Existing mechanism (read first)

- **Producer:** `frontier/profiling/linear_op/main.py`:
  - `_build_profiling_plans` (579-625) excludes V4 from the TP sweep (`tensor_parallel_sweep_exclusion`, `profiling_plan.py:326-338`) and requires `--sequence_parallel_moe_topology TP DP EP` for it;
  - `profile_model` (723-976) runs one `LinearOpWrapper.profile(num_tokens)` per plan and token count. The token list `num_tokens_to_profile` is shared by all plans today;
  - `_materialize_profile_result_rows`, `_restrict_to_plan_operators`, `_attach_linear_op_output_metadata`, `_dry_run` and `_plan_summary`.
- **Shared-expert plan:** `build_sequence_parallel_shared_expert_plan` (`profiling_plan.py:539-606`) gives TP=1 rows with the replicated typed contract that `resolve_layer_contract` selects for the topology. The deepseek_v4 profile sets `replicate_shared_expert_in_sequence_parallel=True` (`frontier/model_architectures.py`, about line 699). The act uses `SiluAndMulWithClamp` when `swiglu_limit` is set (`linear_op_impl.py:676-699`, `frontier/profiling/common/layers/activation.py`). W6 commit `4b19e35` added this path; it has never run on a GPU.
- **Non-layer typed contracts:** `_non_layer_contract` (used for the target-embedded MTP operators, `profiling_plan.py` about 296-314). Reuse it for the V4 endpoint and mHC operators.
- **Wrapper:** `frontier/profiling/linear_op/linear_op_wrapper.py` builds `GPTModel` (`linear_op_impl.py`), calls `build_fp8_linear_method_kernels` and times by CUDA events (`TimerStatsStore`, 3 warm-up and 20 active steps) or by `RecordFunctionTracer` (traces under `<output_dir>/profiler_traces/`).
- **FP8 binding:**
  - `frontier/profiling/common/parallel_utils/fp8_block_linear.py`: `Fp8LinearMethodKernel` builds `Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic", weight_block_size=...))` and records `backend_name = type(method.fp8_linear).__name__`.
  - `tensor_parallel_layers.py:380-440`: `build_fp8_linear_method_kernels` runs it inside `vllm_config_context(model_config=SimpleNamespace(dtype, hf_config=SimpleNamespace(model_type), ...))`. That `hf_config` has no `expert_dtype` and no quantization config.
- **Native V4 (pinned source `/data/ycfeng/tmp/opencode/vllm-v0.28.0`, revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`):**
  - quantization: `vllm/config/model.py:1200-1260` detects the method by calling `override_quantization_method` of the registered methods; `DeepseekV4FP8Config.override_quantization_method` (`vllm/models/deepseek_v4/quant_config.py:133-153`) returns `deepseek_v4_fp8` for `model_type == "deepseek_v4"`. Its `is_scale_e8m0` (line 81) is True when the hf config's `expert_dtype` is `fp4` (V4-Flash: `"expert_dtype": "fp4"`).
  - block-FP8 kernel selection: `vllm/model_executor/kernels/linear/__init__.py` (about 250-275 and 425-470; `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` first for CUDA) and `vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py:149-197` (rows below 32 use the FlashInfer swapAB GEMM, others DeepGEMM).
  - shared expert: `DeepseekV4MLP`, `vllm/models/deepseek_v4/nvidia/model.py:89-148` (`disable_tp=is_sequence_parallel`, `SiluAndMulWithClamp(swiglu_limit)`).
  - decoder layer: `model.py:898-990`. Layer 0 calls `mhc_pre_broadcast_tilelang` (x is 2-D after the embedding); every later attention side and every FFN side calls `mhc_fused_post_pre_tilelang` with `n_splits=1, tile_n=1`, with the attention (`hc_attn_*`, `attn_norm`) or FFN (`hc_ffn_*`, `ffn_norm`) parameters.
  - model forward: `model.py:1130-1215`. The embedding runs on all W rows before `sp_shard`; the layers and the final `mhc_post_tilelang` run on the L SP-local rows; `hc_head_fused_kernel_tilelang` and `self.norm` (`RMSNorm`) run on the W rows after `sp_all_gather(...)[:full_num_tokens]`.
  - mHC functions: `vllm/model_executor/kernels/mhc/tilelang.py` (`mhc_pre_broadcast_tilelang` :303, `mhc_post_tilelang` :410, `mhc_fused_post_pre_tilelang` :433, `hc_head_fused_kernel_tilelang` :720). The parameter shapes are in the decoder layer and model `__init__` (`model.py` about 840-896 and 1075-1100).
  - embedding: `VocabParallelEmbedding.forward` (`vllm/model_executor/layers/vocab_parallel_embedding.py:487-506`) calls `get_masked_input_and_mask` (line 168, decorated with `torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)`), the embedding method and `masked_fill_`, then the all-reduce.
  - LM head: `ParallelLMHead(vocab_size, hidden_size)` with no quantization config (`model.py:1478`); `compute_logits` → `LogitsProcessor` (`model.py:1485, 1514-1518`).
- **Consumer shapes (do not change):** `frontier/execution_time_predictor/deepseek_v4.py:200-285` declares each scope's input and output shapes: `dsv4_embedding` (W,) → (W, hidden); the mHC first and fused scopes and the shared expert on L rows with residual (L, hc_mult, hidden); `dsv4_mhc_final_post` on L; `dsv4_mhc_head` (W, hc_mult, hidden) → (W, hidden); `dsv4_final_norm` (W, hidden); `dsv4_lm_head` on the selected logits rows b with `weight_tp=tp`. The producer's shapes must equal these.
- **The H200 package mechanism:** `h3o_operator_costs/g3h/package_plan.json` and `g3h/prepare_package.py` (base package, image, `cuda_compat_path`, `cloud_mount`, `step_main`, H200 fields, the per-method phases and the dry-run plan check).

### Decisions

1. **FP8 binding from the checkpoint config.**
   - `Fp8LinearMethodKernel` takes the vLLM quantization config instance instead of building `Fp8Config` itself.
   - `build_fp8_linear_method_kernels` creates that instance from the model's checkpoint quantization config with vLLM's own detection: for every name in vLLM's `QUANTIZATION_METHODS`, call `get_quantization_config(name).override_quantization_method(checkpoint_dict, None, hf_config=...)`. If no method overrides, use the checkpoint's `quant_method`. If two methods return different overrides, fail and list them. Then call `from_config(checkpoint_dict)` of the resolved class. No method name is written as a literal in Frontier code.
   - The `vllm_config_context` hf config carries the checkpoint fields that the config reads (`model_type`, `expert_dtype`, `quantization_config`), taken from the Frontier model config, not from literals.
   - Each row records the resolved method name and the weight-scale dtype beside `backend_name`.
   - GLM-5.1 / GLM-5.3 (`quant_method` fp8, `model_type` glm_moe_dsa) must resolve to `fp8` with the same `Fp8Config` fields as today (Acceptance 1).
   - Check `create_weights` and `process_weights_after_loading` with `is_scale_e8m0` True: the dtype of `weight_scale_inv`, and whether `fill_(1.0)` is valid for it. Report what you find and the file:line evidence.
2. **One grid file for the V4 operators.** Add one producer argument, for example `--deepseek_v4_operator_grid PATH`. The JSON gives:
   - the attention TP size of the vocab shard;
   - the SP-local rows L (mHC first, both fused scopes, final post);
   - the descriptor rows W (embedding, mHC head, final norm);
   - the logits rows b (LM head).

   With it, require a V4 model (the architecture profile decides, not the model name) and reject `--num_tokens_list` and `--sequence_parallel_moe_topology` in the same invocation. The shared expert stays on its existing `--sequence_parallel_moe_topology` path, one invocation per topology (no second way to plan it).
3. **One plan per row axis.** The grid gives three plans, one per axis, each with its own token list. Give each plan its own `num_tokens` list in the plan, and make `profile_model` and `_dry_run` iterate a plan's own list. The existing plans keep the CLI token list, so their dry-run plans stay equal (Acceptance 1).
4. **Rows.**
   - A row's `num_tokens` is the count on the plan's axis. Each typed operator contract records the axis name (plain names, for example `sp_local_rows`, `descriptor_rows`, `logits_rows`), the input and output shapes, the tensor-parallel mode from the registry (REPLICATED, or ATTENTION_TP for the embedding and the LM head) and the selected TP size.
   - The rows use the existing row schema and output metadata, so the linear-op CSV validation accepts them.
   - Name the axis table once, in one torch-free place that the consumer can import later. Do not add a field to `OperatorSpec`.
5. **Native calls, not copies.**
   - mHC: call the four pinned tilelang functions with parameters shaped as in the native modules (sizes from the model config: `hidden_size`, `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`, `rms_norm_eps`). The fused scopes use `n_splits=1, tile_n=1` as native. Measure both fused names separately, with their own parameter sets.
   - Final norm: vLLM `RMSNorm(hidden_size, rms_norm_eps)` on its CUDA custom-op path (native launches `vllm::rms_norm_kernel`). Say how the producer selects that path.
   - Embedding: TP rank 0 of the grid's attention TP. Prefer binding `VocabParallelEmbedding.forward` to that rank's shard without a TP group, and stop before the all-reduce. If that is not possible, call the forward's own parts (`get_masked_input_and_mask` with the shard indices from vLLM's index function, the embedding method, `masked_fill_`) and report why. Token ids are uniform over the full vocabulary.
   - LM head: the unquantized GEMM that `ParallelLMHead`'s method applies, on the rank's vocab shard (b, hidden) × (hidden, vocab / TP).
   - Use the same timer scopes as the existing wrapper (CUDA events and `record_function` names equal to the operator names), and warm up before timing so JIT compilation (tilelang, Inductor) is not timed.
   - Keep `RecordFunctionTracer`'s traces under `profiler_traces/` for every row, so the lead can read kernel names and grids per operator scope.
6. **Registry.** Set `profiling_target=True` on the eight `DEEPSEEK_V4_MHC_FAMILY` operators. Keep `execution_enabled=False` (the consumer is later). Find every user of `profiling_target` and of the family's profiling names (`get_family_profiling_names`, `profiling_ops`, tests) and report each effect. A change that alters a GLM plan or another model's plan stops that part.
7. **Module placement.** Put the V4 operator builders and timers in a new module under `frontier/profiling/linear_op/` that imports torch and vLLM only inside functions that run on the GPU worker, as the existing producer does. Keep every module at or below 2,000 lines (`main.py` 1,130, `linear_op_impl.py` 1,137, `profiling_plan.py` 606, `linear_op_wrapper.py` 262, `tensor_parallel_layers.py` 1,106, `families.py` 943).
8. **Shared-expert parity check.** Compare the producer's shared expert with `DeepseekV4MLP` under SP (`disable_tp=True`, the gate-up width 2 × `moe_intermediate_size`, the clamp limit, the bf16 dtypes). Report any difference with file:line evidence.

### G2h grid (C1 §4.2, with the axis correction)

C1 §4.2 placed the mHC head and the final norm on L. The pinned forward runs them on W after the gather (`model.py:1197-1211`), and the native P2 kernels show the W grid (`hc_head_fuse_tilelang_kernel` and `vllm::rms_norm_kernel` grid 6144 in the 6144-token prefill). The grid below uses W for them.

| Plan | Operators | Rows |
| --- | --- | --- |
| SP-local rows L | `dsv4_mhc_first_pre`, `dsv4_mhc_ffn_post_attn_pre`, `dsv4_mhc_attn_post_ffn_pre`, `dsv4_mhc_final_post` | 1, 2, 3, 4, 768 |
| Descriptor rows W | `dsv4_embedding`, `dsv4_mhc_head`, `dsv4_final_norm` | 1, 2, 4, 8, 16, 24, 6144 |
| Logits rows b | `dsv4_lm_head` | 1 to 24 |
| Shared expert (existing path) | `share_expert_up_proj`, `share_expert_act`, `share_expert_down_proj` | L = 1, 2, 3, 4, 768, for topologies (8, 2, 16) and (8, 4, 32) |

The vocab shard uses attention TP 8. Methods: `cuda_event` (`linear_op.csv`) and `record_function` (`linear_op_kernel_only.csv`). The `record_function` rows are the costs (design §3.1); the `cuda_event` rows are for validation.

### Package plan

- Add `h3o_operator_costs/g2h/` with `package_plan.json`, `operator_grid.json` (the grid above), a builder (for example `prepare_package.py`) and `expected_rows.json`. The builder follows `g3h/prepare_package.py`.
- Phases: for each method, one grid invocation and one shared-expert invocation per topology, each with its own output directory, so no invocation overwrites another's CSV. Name the phases plainly (for example `G2H_OPERATORS_CUDA_EVENT`, `G2H_SHARED_EXPERT_EP16_RECORD_FUNCTION`).
- Settings: destination `package_01`; name `G2H_V4_OPERATORS`; job tag `v4-h200-g2h-operators-20261028-01`; output directory `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_operators/g2h_01`; `--device h200`.
- `source_revision`: set it to the current HEAD and say in the report that the lead replaces it after committing B13. Do not build the package.
- The builder's dry-run check requires each invocation's planned rows (plan, operator, axis, rows, TP) to equal `expected_rows.json`.
- Write `g2h/native_kernel_notes.md`: for each operator, the native kernels (name, grid, count per forward, median duration) from the dumps in `h3o_operator_costs/evidence/v4_operators/` (`p2_32_decode_dp0.txt`, `p2_32_prefill_dp0.txt`, `p2_64_*`; produced by `native_all_kernels.py`, which truncates names to 150 characters), with the dump line numbers. The table below is the lead's reading; correct it where the dumps disagree.

  | Operator | Native kernels |
  | --- | --- |
  | Shared expert, decode rows | `scale_1x128_kernel`, `fp8_gemm_kernel_swapAB<4096u,4096u…>`, `act_and_mul_kernel<…silu…>`, `scale_1x128_kernel`, `fp8_gemm_kernel_swapAB<4096u,2048u…>` |
  | Shared expert, L = 768 | `per_token_group_quant_8bit_kernel`, `sm90_fp8_gemm_1d2d_impl<…4096u,4096u…>`, act grid 768, `sm90_fp8_gemm_1d2d_impl<…4096u,2048u…>` |
  | mHC fused, decode | `mhc_fused_tilelang_kernel` grid (L, 12, 8) and `mhc_pre_big_fuse_with_norm_tilelang_kernel` grid L, 85 per forward |
  | mHC fused, L = 768 | `mhc_post_tilelang_kernel`, `mhc_pre_big_fuse_with_norm_tilelang_kernel` and a tf32 GEMM |
  | mHC first | `mhc_pre_big_fuse_broadcast_with_norm_tilelang_kernel` |
  | mHC final post | `mhc_post_tilelang_kernel` |
  | mHC head | `hc_head_fuse_tilelang_kernel` grid W |
  | Final norm | `vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>` grid W |
  | Embedding | `triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0`, `triton_poi_fused_bitwise_and_bitwise_not_bitwise_or_ge_lt_1`, `indexSelectSmallIndex<c10::BFloat16, long, …>` grid 32, `masked_fill_kernel` grid 8W |
  | LM head | `nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT` grid (2, 64, 1) at b = 2 |

  The full native traces (for names longer than 150 characters) are in `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces/` and `dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces/`. Read at most one decode and one prefill rank file of each, with `timeout` and the memory limit; say whether the native act kernel is the clamp variant.
- **GPU time.** Estimate from the planned row count and the start-up of each invocation (vLLM import, tilelang and Inductor compilation). Compare it with C1's G2h estimate of 1 GPU-hour, inside the G1h-G4h cap of 7 (about 0.07 used, plus G1h-02).

## Out of scope

- Any GPU, platform or network action. Building or submitting the package.
- The engine operators of B13r (`kv_cache_block_zero`, block-table writes, post-DP preparation, sampling), the graph-replay runner and its plans.
- The consumers (entries 1-12), the loaders and trainers, `execution_enabled`, and `frontier/execution_time_predictor/`.
- The communication operators (`dsv4_embedding_allreduce`, `dsv4_logits_gather`, the SP and EP collectives) and the copies around them.
- H800 G2. The producer and the builder must work for H800 later, but write no H800 files now.
- Changes to the attention producer, the MoE producer, `balanced_routing.py`, the census and the B10-B12 files.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `4f54d928`). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Concurrent work:** the G1h-02 GPU job runs from `g1h/package_02`; do not touch `g1h/`, `g3h/` or `g5h/`.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md) (§3.1, §6 entry 15), [c1_coverage.md](c1_coverage.md) (§3.2-3.5, §4.2, §5), and the W6 producer summary `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/w6_dsv4_producers_20261020/summary.md`.
- **Writable paths:**
  - `frontier/profiling/linear_op/`;
  - `frontier/profiling/common/parallel_utils/fp8_block_linear.py` and `tensor_parallel_layers.py` (the binding only);
  - `frontier/operators/families.py` (the eight `profiling_target` values) and one new torch-free module for the axis table if it does not fit in the linear-op package;
  - `tests/unit/`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h/` (new).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b13/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No deletions or moves.
  - Python 3.10. No magic literals in code paths: the rows come from the grid file, the sizes from the model config, the TP size from the grid. No guards for states the code cannot reach. No second implementation of a vLLM function.
  - The profiler entry points must import without `fasteners` and `sklearn` (`tests/unit/test_profiling_runtime_boundary.py`); the worker image has neither.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. No recursive grep or find over the vLLM tree or the trace directories; name the files.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true` and `VIDUR_DISABLE_WANDB=1`.
  - Tests that need torch (CPU) use `PYTHONPATH=$PWD:/usr/local/lib/python3.10/dist-packages` with the same interpreter. The native registry checks also need `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.
  - The pinned vLLM source is read-only.

## Acceptance

1. **Existing plans unchanged.** `--dry_run` gives plan JSON equal to HEAD for: the GLM linear-op arguments of the existing GLM packages or tests (name the ones you use), and the V4 `--sequence_parallel_moe_topology 8 2 16` and `8 4 32` invocations. If a field must change, list it and say why.
2. **FP8 binding.** Unit tests with stand-in vLLM classes show: V4 resolves to the override with `expert_dtype` from the model config; GLM resolves to `fp8` with the same `Fp8Config` arguments as HEAD; two different overrides fail. Name what needs the real vLLM and is left to the GPU run.
3. **V4 plans.** The grid `--dry_run` lists exactly the rows of `expected_rows.json`: three plans, their operators, axes, row lists and TP sizes. Rejected combinations (grid with `--num_tokens_list`, grid with a topology, grid with a non-V4 model) fail with clear messages.
4. **Shapes.** For each operator and a row count on each axis, the producer's input and output shapes equal the shapes of `deepseek_v4.py:200-285` for the matching batch (test both sides from one fixture).
5. **Registry.** The eight operators are profiling targets; every other family and every non-V4 plan is unchanged (tests).
6. **Existing tests pass:** the linear-op tests (find them with grep: `test_linear_op*`, `test_*share_expert*`, `test_*sequence_parallel*`, `test_profiling_runtime_boundary.py`, `test_operator*`, `test_*families*`), and any other test that imports the changed modules. Report tests that need torch or the native source and are skipped.
7. **New unit tests** for the binding, the grid parser, the per-plan token lists, the axis table, the typed contracts and the rejected combinations. Report the count.
8. **Findings** for decision 1 (scale dtype and `fill_`), decision 5 (how RMSNorm's CUDA path and the embedding shard are bound), decision 6 (users of `profiling_target`) and decision 8 (shared-expert parity), with file:line evidence.
9. **G2h package plan** and the builder's dry-run check on the current HEAD (plan check only, no package written), with the phase list and the GPU time.
10. **Hygiene:** line counts; no unused imports (`/data/ycfeng/tmp/unused_imports.py`); a cold import of `frontier.profiling.linear_op.main` and of the new modules exits 0 and loads no `torch`, `vllm` or `cuda` module unless the existing module already does at load time (say so, and do not change it).

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- The results of Acceptance 1-10, with the commands, the expected and actual values, and the counts.
- The expected G2h rows per phase and the GPU-time estimate.
- Scratch paths, deviations and their reasons, and limits.
- If the native source or the producer contradicts a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Examples: the embedding cannot run on a vocab shard without a TP group; the RMSNorm CUDA path cannot be selected outside a vLLM config; the quantization detection needs objects the producer cannot build.
