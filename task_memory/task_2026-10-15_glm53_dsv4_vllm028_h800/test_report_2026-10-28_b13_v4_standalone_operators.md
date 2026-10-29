# Test report: B13, the V4 standalone operator producer (design entry 15, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead review, corrections and rerun before the B13 commit. |
| 2026-10-28 | Added the G2h run on 1 H200: job identity, rows, bindings, kernel identity against native (PASS with the embedding mask difference) and the durations (reported, not gated). |

## Scope

- B13 (Codex `2c58b7b1`, outcome partial, [h3o_operator_costs/codex_assignment_B13_v4_operator_producer.md](h3o_operator_costs/codex_assignment_B13_v4_operator_producer.md)): the standalone part of design entry 15. It profiles the eight mHC operators, the embedding, the final norm and the LM-head shard through the linear-op producer, binds the V4 shared expert to the checkpoint FP8 method, and adds the G2h package plan (`h3o_operator_costs/g2h/`).
- The model-runner operators (`dsv4_input_preparation`, `dsv4_sampling`, `kv_cache_block_zero`, `dsv4_block_table_write`) are B13r, not B13.
- Codex report: `h3o_operator_costs/codex-implement-2c58b7b1-6140-4ac7-a551-034c5ec9188f/implementation.md`. Codex scratch: `/data/ycfeng/tmp/codex-implement/2c58b7b1-6140-4ac7-a551-034c5ec9188f/b13/`.

## Lead corrections

| Change | Reason |
| --- | --- |
| Removed the V4-only branch that passed two logical widths `[2048, 2048]` for the shared gate-up projection; every binding passes `[layer.weight.shape[0]]` again. | In the pinned block-FP8 path the logical widths feed only `validate_fp8_block_shape` and the scale-parameter shapes. With 128-wide blocks, `[4096]` and `[2048, 2048]` give the same checks and the same 32 scale rows, so the branch selected no different kernel. |
| Replaced the dispatch facade inside `LinearOpWrapper` with a module function `linear_op_wrapper_class(profiling_plan)`, used at the three construction sites in `main.py`. | The facade made `LinearOpWrapper.__new__` return a different class. The selector keeps one way to construct each producer and keeps control flow visible at the call site. |
| `resolve_checkpoint_quantization` now calls vLLM's own `ModelConfig._verify_quantization` on a namespace holding only the fields it reads, then `get_quantization_config(method).from_config(...)`. The tests were rewritten around stand-in vLLM modules. | Codex's version walked the override list itself and added a conflict check. That is a second implementation of a vLLM function; the pinned method (`vllm/config/model.py:1199-1300`) already takes the first matching override. |

## Decisions (lead, D-NO-INTERRUPT)

1. **GLM ignored layers.** Native `Fp8Config.from_config` keeps the checkpoint's 541 ignored-layer entries; HEAD's direct constructor had none. Accepted: the list is read only by `Fp8Config.get_quant_method` (`fp8.py:175-205`), and the producer constructs `Fp8LinearMethod` directly, so the leaf kernels are unaffected.
2. **Canonical typed-contract admission.** `typed_contracts.py:182` calls the execution binding, which rejects the mHC family while `execution_enabled=False`. Producer serialization works; canonical admission does not. Deferred to the consumer stage (entries 1-12), which turns on execution for the family. The sentinel test `test_canonical_admission_is_blocked_by_disabled_family` stays so the change is visible when it happens.

## Checks after the corrections

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. CPU torch runs add `/usr/local/lib/python3.10/dist-packages` and `CUDA_VISIBLE_DEVICES=""`; native-source checks set `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. All runs use `ulimit -v 31457280` and a timeout.

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Focused pytest: the 10 files of the Codex focused command (log `/data/ycfeng/tmp/g1h_fix/b13_review/focused.log`) | all pass; skips only for missing vLLM | 225 passed, 3 skipped | PASS |
| 2 | Broad pytest: 43 files (command `/data/ycfeng/tmp/g1h_fix/b13_review/broad_cmd.sh`, log `broad.log`), 3 known cases deselected | all pass | 1017 passed, 3 skipped, 3 deselected | PASS |
| 3 | New test files collected | 24 plan cases + quantization cases | 29 (24 + 5 after the rewrite) | PASS |
| 4 | `g2h/prepare_package.py --check-plan` from the repository root | 6 invocations equal to `expected_rows.json`; no package, no GPU or platform operation | 6 invocations; operators 36 CSV rows / 65 measurements per method; each shared-expert topology 5 / 15; `package_built=false`, `gpu_or_platform_operations=0` | PASS |
| 5 | Cold import of `linear_op.main`, `fp8_quantization`, `deepseek_v4_plan`, `deepseek_v4_operators`, `deepseek_v4_axes` | no torch, vLLM, CUDA, fasteners or sklearn module loaded | none loaded | PASS |
| 6 | `/data/ycfeng/tmp/unused_imports.py` on the changed Python files | no unused import except the existing registration import | only `tensor_parallel_layers.py:123` `vllm...fp8_utils  # noqa: F401` (present at HEAD, registers `torch.ops.vllm.apply_w8a8_block_fp8_linear`) | PASS |
| 7 | Line counts | every module at or below 2,000 | families 943, axes 19, fp8 block linear 111, tensor-parallel layers 1,125, main 1,159, wrapper 281, V4 plan 123, V4 operators 218, quantization 40 | PASS |
| 8 | `git diff --check` | clean | clean | PASS |

Deselected, unchanged from earlier reports: `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` (`OperatorSpec` has no `kernel_count`; fails at HEAD too) and two `test_profiling_governance_minimal_red.py` MoE cases (local Triton needs the missing `setuptools`). The 3 skips need vLLM, which the CPU environment lacks.

Codex's Acceptance 1 (existing plans equal HEAD for the GLM and V4 topology invocations plus 50 non-V4 planning scenarios) ran before the corrections. The corrections do not touch planning code; checks 1, 2 and 4 cover the planning paths again.

## Not verified on CPU (left to G2h on 1 H200)

- e8m0 scale creation and copy: the CPU torch 2.1.0 has no `float8_e8m0fnu`.
- Native kernel selection for the shared expert (FlashInfer/DeepGEMM hybrid), the RMSNorm `vllm_c` provider, the compiled embedding mask, and grid and kernel-name parity against `h3o_operator_costs/g2h/native_kernel_notes.md`.
- Observation from Codex: the native activation kernels end in `true,true,true,false`, the clamp variant (`csrc/libtorch_stable/activation_kernels.cu:98-102`).

## GPU allowance

G2h: 6 phases, 190 operator points, one H200 on step_main, non-preemptible, timeout 3,600 s. Codex planning estimate 0.23-0.60 GPU-hour, inside the one-hour G2h cap; G1h-G4h used about 0.92 of 7 GPU-hours before G2h.

## G2h run on 1 H200

**Job.** `exp-1029-035012-764213`, tag `v4-h200-g2h-operators-20261028-01`, submitted through Pi `g2h-submit-1028k` at 2026-10-28T19:50:12Z, Succeeded at 19:57:00Z. Source `7753e5a9` (package plan pinned in `2572af34`), package `h3o_operator_costs/g2h/package_01/` (manifest SHA-256 `d87fc6ef…be8e1`, 551 files), image `vllm-openai@sha256:2286e853…6635`. Worker runtime: Python 3.12.3, torch 2.13.0+cu130, vLLM 0.28.0, FlashInfer 0.6.16.post3; `nvidia-smi` reports `NVIDIA H200, 143771 MiB`, driver 570.124.06. Quota before submission: step_main H200 non-preemptible 240 of 512 in use.

**GPU time.** Job wall clock 407 s = 0.113 GPU-hours (the six phases ran 225.3 s; the rest is the engine start and the copies). G1h-G4h now about 1.03 of 7 GPU-hours.

**Artifacts.** Cloud `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_operators/g2h_01`; local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_01/download/` (94 files, 23,445,714 B, equal to the remote listing). The priced table is `G2H_OPERATORS_RECORD_FUNCTION/compute/h200/deepseek-ai/DeepSeek-V4-Flash/linear_op_kernel_only.csv` plus the two shared-expert `linear_op_kernel_only.csv` tables; the CUDA-event tables are validation rows. Pi report: `/data/ycfeng/tmp/ds4_runs/g2h-submit-1028k/output.md`. Trace summaries (lead): [h3o_operator_costs/evidence/g2h/](h3o_operator_costs/evidence/g2h/), made by `g2h_trace_kernels.py`. The script assigns each kernel to the innermost `gpu_user_annotation` range that contains it.

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 9 | Package and job fields | manifest SHA-256 and 551 file hashes equal; 1 H200, step_main, non-preemptible, timeout 3,600 s, source `7753e5a9` | all equal; 551 of 551 files, 0 mismatches | PASS |
| 10 | Worker identity | creator `i-fengyicheng`, non-preemptible, queue `step-main-default`, local worktree mount, pinned image, device H200 | all equal in the Running read at 19:54:27Z (see deviations) | PASS |
| 11 | Phases and rows | 6 phases succeeded; rows equal to `g2h/expected_rows.json` | 6 of 6 succeeded; rows 36 / 5 / 5 / 36 / 5 / 5; operator rows 5 `sp_local_rows` + 7 × 3 `descriptor_rows` + 24 `logits_rows`; no `Traceback` in any step log | PASS |
| 12 | Shared-expert binding | checkpoint method `deepseek_v4_fp8`, e8m0 weight scales, FlashInfer/DeepGEMM kernel | `linear_bindings` of all four projections in all four tables: `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, `quantization_method=deepseek_v4_fp8`, `weight_scale_dtype=torch.float8_e8m0fnu`, `processed_weight_scale_dtype=torch.float32`; log `DeepGEMM E8M0 enabled on current platform.` | PASS |
| 13 | Kernel identity against [g2h/native_kernel_notes.md](h3o_operator_costs/g2h/native_kernel_notes.md) | every scope launches the native kernel names and grids | equal for all 11 operators except the embedding mask fusion (table below) | PASS with one recorded difference |
| 14 | Kernels outside the scopes | harness work only | operator traces: 4 `CUDAFunctorOnSelf_add<float>` per trace, as in G3h. Shared-expert traces: also the existing MLP harness input work (index gather, a torch RMSNorm made of pow, mean, add and rsqrt, bf16 copies and fills). None is inside a timed scope, so none enters a row | PASS |
| 15 | EP16 against EP32 shared expert | equal kernels (the shared expert does not depend on the EP width) | equal kernel names, grids and launches per call | PASS |

**Kernel identity.** In this table, L is the SP-local row count and W is the descriptor row count.

| Operator | G2h kernels (equal to native unless stated) |
| --- | --- |
| `dsv4_mhc_first_pre` | TF32 prenorm GEMM + `mhc_pre_big_fuse_broadcast_with_norm_tilelang_kernel` |
| `dsv4_mhc_ffn_post_attn_pre`, `dsv4_mhc_attn_post_ffn_pre` | decode: `mhc_fused_tilelang_kernel` grid (L, 12, 8) + `mhc_pre_big_fuse_with_norm`; L=768: post, TF32 GEMM 16384, pre-norm |
| `dsv4_mhc_final_post`, `dsv4_mhc_head` | `mhc_post_tilelang_kernel`; `hc_head_fuse_tilelang_kernel` |
| `dsv4_final_norm` | `vllm::rms_norm_kernel<c10::BFloat16, 8, 2, true>` |
| `dsv4_lm_head` | `nvjet_sm90_tst_128x8/128x16/128x24`, grid [2, 64, 1] |
| `share_expert_up_proj`, `share_expert_down_proj`, `share_expert_act` | decode: `scale_1x128_kernel` grid 1056 + `deep_gemm::fp8_gemm_kernel_swapAB` (4096 × 4096 and 4096 × 2048, grid 132) + the clamp `act_and_mul`; L=768: `per_token_group_quant_8bit` grids 1536 and 768 + `sm90_fp8_gemm_1d2d_impl` |
| `dsv4_embedding` | `indexSelectSmallIndex` grid 32 (W ≤ 16) or `vectorized_gather_kernel` (W = 24, 6144), masked fill grid 8W. **Difference:** the input mask is two Inductor kernels at W = 1, as native, but one fused kernel (`triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0`) at W ≥ 2, where native launches two at every W |

The RMSNorm warning in both operator logs (`op.py:406`, "execution fallback to native is possible") names the IR provider called `native`; the trace shows the same `vllm::rms_norm_kernel` as the native final norm.

**Embedding mask difference (open item O-EMB-MASK).** Observation: the producer calls vLLM's own `get_masked_input_and_mask`, compiled by its decorator `@torch.compile(dynamic=True)` (`vocab_parallel_embedding.py:168`). The fused kernel costs 0.83-0.94 µs in decode and 3.04 µs at W=6144; the native pair costs about 3.0-3.2 µs and 5.8 µs. The difference is about 2.2-2.8 µs once per forward. Inference, not verified: in the native run the function is traced inside the whole-model compiled graph, so Inductor splits the pointwise work differently; when the function is compiled alone, W=1 gets its own specialized graph, which gives the native pair. Decision (lead, D-NO-INTERRUPT): the row is kept and the difference is recorded; the stage 3 composition check covers it, because a fix would mean compiling the mask inside the native model graph.

**Durations (reported, not gated).** As in G5h and G3h, only kernel identity is gated here. Median µs per launch, G2h against native:

| Operator | Point | G2h | Native |
| --- | --- | --- | --- |
| `dsv4_mhc_first_pre` GEMM / pre-big-fuse | L=768 | 11.97 / 5.82 | 12.03 / 6.83 |
| `dsv4_mhc_ffn_post_attn_pre` post / GEMM / pre-norm | L=768 | 15.28 / 13.18 / 12.22 | 14.43 / 13.82 / 12.51 |
| `dsv4_mhc_head` | W=6144 | 174.43 | 174.75 |
| `dsv4_final_norm` | W=6144 | 29.01 | 27.94 |
| `dsv4_embedding` gather / masked fill | W=6144 | 17.68 / 59.57 | 18.75 / 59.50 |
| shared expert up / down / act | L=768 | 19.89 / 12.56 / 3.90 | 19.90 / 12.51 / 4.32 |
| `mhc_fused_tilelang_kernel` | decode | 2.46 | 5.57 |
| `mhc_pre_big_fuse_with_norm` | decode | 4.77 | 5.15 |
| `dsv4_mhc_final_post` | decode | 2.24 | 3.55 |
| `dsv4_mhc_head` | decode | 5.34 | 9.92-12.53 |
| `dsv4_mhc_first_pre` GEMM | decode | 2.69 | 3.28 |
| `dsv4_final_norm` | decode | 1.70 | 2.05 |
| shared expert up / down / act | decode | 7.3-7.6 / 4.96 / 1.5-1.66 | 8.90-9.06 / 5.95 / 2.34 |
| `dsv4_lm_head` | decode | 32.7-34.4 | 34.3 |

Prefill points agree within about 10%. Decode points measured alone are 17-130% shorter than the same kernels inside the native FULL CUDA graph. Per decode forward, the mHC scopes alone account for about 300 µs of difference (43 layers, two fused scopes each). Inference, not verified: inside the graph, programmatic dependent launch (PDL; the worker log reports it enabled for DeepGEMM) starts a kernel before its predecessor ends, so its trace duration includes the wait. The full model also runs with a colder L2 cache than a loop over one operator. Which of the two values is the right per-operator cost is a stage 3 question: the composition check compares the sum of priced operators with the native forward wall time. This is recorded with the NONE-step eager gap observation.

The CUDA-event tables (medians 37-224 µs for scopes whose kernels sum to 2-175 µs) include the host launch time of the multi-kernel scopes. They validate the run and are not priced.

**Deviations.**

- Pi reused `status_2.json` and `status_3.json` and overwrote the Starting and Running reads, against the rule not to overwrite. The job was unaffected. The Running read is kept only in the Pi report (`output.md`, section "Deviations and failures").
- The launcher's exit handler logged `Job Stop` at 19:57:03.7Z, 3.7 s after the job finished and after all six phases were copied. No job was stopped by hand.
- The shared-expert phase banner prints `share_expert_* : BF16` while the recorded bindings are FP8 with e8m0 scales. The banner is not the binding record. This is a cosmetic issue and is deferred.
