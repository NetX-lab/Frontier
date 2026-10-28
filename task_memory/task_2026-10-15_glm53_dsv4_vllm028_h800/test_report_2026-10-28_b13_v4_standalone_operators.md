# Test report: B13, the V4 standalone operator producer (design entry 15, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead review, corrections and rerun before the B13 commit. |

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
