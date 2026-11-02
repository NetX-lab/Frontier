## 1. Assignment and outcome

B13 implements the standalone part of design entry 15: V4 shared-expert FP8 binding; mHC, embedding, final norm, and LM-head profiling through the linear-op producer; profiling registry targets; and the CPU-checked G2h package plan.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Writes were confined to `frontier/profiling/linear_op/`, the two authorized FP8 binding files, `frontier/operators/families.py`, one new operator-axis module, `tests/unit/`, and the new `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h/` directory. Scratch is `/data/ycfeng/tmp/codex-implement/2c58b7b1-6140-4ac7-a551-034c5ec9188f/b13/`.

The implementation preserves the specified W/L/b axes, TP8 vocabulary shard, existing shared-expert topology path, disabled execution flag, native function calls, existing timers, and kernel-only versus CUDA-event distinction. It uses checkpoint and grid fields for model sizes and rows. The native mHC post multiplier remains the pinned implementation's constant 2.0, documented at `nvidia/model.py:853`.

Applicable root AGENTS.md and the named requirements, plan, issues, design, C1, and W6 artifacts were inspected. No applicable nested CLAUDE.md was found. Source records were treated as evidence without broadening B13 into the engine operators or consumers.

Outcome is **partial**. Implementation and independent CPU checks are finished, but two acceptance conflicts remain:

1. Canonical typed-contract admission calls an execution binding that rejects this family while `execution_enabled=False`. Producer serialization succeeds; canonical admission does not.
2. Native checkpoint `Fp8Config.from_config` retains GLM's 541 ignored-layer entries, whereas HEAD's direct constructor omitted them. The serialized-FP8, dynamic-activation, and block-size fields match HEAD, but complete constructor-field parity does not.

No Git state changes, dependency installation, delegation, GPU operation, network action, package build, submission, or publication was performed.

Initial HEAD was `4f54d9281989df78ae8f40034f510008f370a4c0`. During verification, Claude's documentation-only commit advanced HEAD to `e8eb3c97f689f47b090753e525730f90817c113f`. Its changes do not touch assigned code or tests. G2h `source_revision` now names that current HEAD; Claude must replace it after committing B13.

## 2. Changes

Paths below are relative to the working root. Nineteen files were changed or added.

| File | Change and purpose |
| --- | --- |
| `frontier/operators/families.py:313` | Changed the eight `DEEPSEEK_V4_MHC_FAMILY` operators to profiling targets. Execution, predictor, and E2E trace admission remain disabled. No other family fields changed. |
| **New** `frontier/operators/deepseek_v4_axes.py:5` | Immutable, torch-free axis table for the eight standalone and three shared-expert scopes. No field was added to `OperatorSpec`. |
| **New** `frontier/profiling/linear_op/fp8_quantization.py:6` | Builds the checkpoint HF context with `model_type`, `expert_dtype`, and quantization fields. `resolve_checkpoint_quantization:24` checks every registered override, rejects conflicting resolutions, and calls the resolved class's `from_config`. Native imports are lazy. |
| `frontier/profiling/common/parallel_utils/fp8_block_linear.py:46` | `Fp8LinearMethodKernel` accepts the resolved native configuration instance. Unit scales are initialized by converting FP32 ones and copying them. Records backend, quantization method, and scale dtype before and after native weight processing. |
| `frontier/profiling/common/parallel_utils/tensor_parallel_layers.py:391` | Binds checkpoint-driven FP8 methods under the checkpoint HF context. V4 shared gate-up uses two logical widths matching native merged projection semantics. `linear_fp8_bindings_by_operator:474` exposes binding metadata. |
| **New** `frontier/profiling/linear_op/deepseek_v4_plan.py:16` | Strict grid parser, three plans with independent row lists, row contracts, principal shapes, and native state shapes. Architecture checks use the existing `ModelConfig` resolver method. |
| **New** `frontier/profiling/linear_op/deepseek_v4_operators.py:14` | Worker-only native mHC, local embedding, local unquantized LM-head, and RMSNorm calls. `DeepseekV4OperatorWrapper:139` warms up, checks output shape, times each scope, and preserves traces and typed metadata. |
| `frontier/profiling/linear_op/linear_op_wrapper.py:55` | Delegates grid plans to the standalone wrapper. Adds FP8 binding metadata and V4 shared-expert row shapes/axes. Existing shared-expert construction remains on the topology path. |
| `frontier/profiling/linear_op/main.py:284` | Adds `--deepseek_v4_operator_grid`; rejects incompatible token-list/topology combinations. `_build_profiling_plans:581`, `_dry_run:672`, and `profile_model:749` use each plan's row list. Legacy dry-run summaries retain their previous fields and values. Removed two unused imports. |
| **New** `tests/unit/test_linear_op_checkpoint_quantization.py` | Eight CPU cases for override resolution, expert dtype, GLM fields, conflicting/equal overrides, lazy registry import, and kernel configuration/scale metadata. |
| **New** `tests/unit/test_dsv4_standalone_operator_plans.py` | Twenty-four cases for grid parsing, axes, contracts, consumer shape parity, incompatible arguments, independent iteration, package planning, and the canonical-admission blocker. Parser helpers use dry-run mode to avoid default output writes. |
| `tests/unit/test_dsv4_operator_manifest.py:54` | Updates manifest expectations for the eight profiling targets while preserving frozen execution and generic MoE/shared-family behavior. |
| `tests/unit/test_dsv4_source_execution.py:298` | Updates the structural-admission assertion for profiling targets; execution, predictor, and trace assertions remain closed. |
| `tests/unit/test_profiling_vllm_parallel_state.py:48` | Updates the native check to the checkpoint-model binding interface. |
| **New** `g2h/package_plan.json` | Six separate phases, H200 settings, current source revision, output paths, and GPU-time allowance. |
| **New** `g2h/operator_grid.json` | The approved L, W, and b grids with attention TP8. |
| **New** `g2h/expected_rows.json` | Exact plan/operator/axis/row/TP expectations for every phase. |
| **New** `g2h/prepare_package.py:68` | CPU dry-run equality check for all invocations. Future build path follows the accepted G3h mechanism and snapshots committed source; it was not executed. |
| **New** `g2h/native_kernel_notes.md` | Per-operator native names, grids, counts, median durations, dump lines, full-trace clamp evidence, and attribution limitations. |

Pre-existing work was preserved. `.cortexkit/`, `.pi/`, and `remaining_measurements/` were untouched. The initially modified G1h worker-import report was also untouched; Claude's concurrent commit subsequently included it. No G1h, G3h, G5h, consumer, attention-producer, MoE-producer, or task-record edits were made by this executor.

The five G2h files are ignored by the repository's current ignore rules. They exist on disk but do not appear in ordinary `git status`; Claude must include them explicitly when committing.

## 3. Verification

All commands used the required virtual-memory limit, `ulimit -v 31457280`, and an outer timeout. Subprocess checks also had bounded timeouts. The interpreter was `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6.

Minimal environment: `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, and `VIDUR_DISABLE_WANDB=1`. CPU torch checks added `/usr/local/lib/python3.10/dist-packages` to that path and set `CUDA_VISIBLE_DEVICES=""`. Native-source checks received `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. Pytest used `-p no:cacheprovider` and private scratch `--basetemp` paths. No dependency was installed.

### Acceptance 1–10

| Acceptance | Result | Evidence |
| --- | --- | --- |
| 1. Existing plans | **PASS** | Saved HEAD JSON equals candidate JSON for the GLM test arguments and V4 EP16/EP32 topology invocations. Fifty additional non-V4 planning scenarios equal HEAD. No legacy dry-run JSON field changed. |
| 2. FP8 binding | **PARTIAL** | Stand-in override and kernel tests pass. V4 receives checkpoint expert dtype; GLM resolves to `fp8`; conflicting overrides fail. GLM ignored-layer field parity differs as described below. Real vLLM/e8m0 execution remains untested. |
| 3. V4 plans | **PASS** | Exactly three grid plans, 36 CSV rows and 65 operator measurements per grid phase. All six builder invocations match `expected_rows.json`. All three prohibited combinations fail clearly. |
| 4. Shapes | **PASS** | All eleven scopes are compared to the actual consumer from common batch fixtures at W=24/b=3 and W=6144/b=1. LM-head `output_shape=None` is preserved; its concrete shard shape is recorded separately. |
| 5. Registry | **PASS** | Exactly eight flags changed. Execution stays false. All eleven other registered families equal HEAD, including callable identities normalized by qualified name. Non-V4 plans are unchanged. |
| 6. Existing tests | **PARTIAL** | Final available-environment run: 1020 passed, 3 skipped, 3 deselected. The known failure and missing prerequisites are listed below; the full unfiltered selection is not claimed to pass. |
| 7. New tests | **PASS** | 32 new cases: 24 plan/contract cases plus 8 quantization cases. All pass with CPU torch. Minimal environment: 31 passed, 1 torch-dependent skip. |
| 8. Findings | **DONE, with runtime limits** | Native source, registry effects, shared-expert parity, RMSNorm provider, embedding fallback, and full-trace clamp evidence are recorded below and in the kernel notes. e8m0 `fill_` behavior cannot be confirmed in this CPU environment. |
| 9. Package plan | **PASS, plan only** | Six expected phases pass on current source; `package_01` does not exist. Source revision is current HEAD. No build or GPU run occurred. |
| 10. Hygiene | **PASS with documented scanner exceptions** | Cold imports load none of torch/vLLM/CUDA/fasteners/sklearn. All source modules are below 2000 lines. `git diff --check` passes. Unused-import scanner finds only two pre-existing, intentional vLLM registration imports. |

### Executed checks and commands

For the commands below, `R` denotes the working root, `S` the scratch directory stated above, and `PY` the interpreter stated above.

- Baselines and matrix: `timeout 120s env PYTHONPATH="$R" PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 "$PY" "$S/regression.py"`. **PASS**. The script reads HEAD source with bounded `git show`, compares serialized JSON, and checks family declarations. Artifacts: `regression.log`, `plan_regression_50.json`, and the three `*_head.json` / `*_candidate.json` pairs.
- Package check: `timeout 30s env PYTHONPATH="$R" PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 "$PY" "$R/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h/prepare_package.py" --check-plan`. **PASS**, six invocations, `package_built=false`, zero GPU/platform operations. Final output: `package_check_final.json`.
- Focused CPU run: `timeout 180s env` with the CPU environment above, followed by `$PY -m pytest -q -ra -p no:cacheprovider --basetemp "$S/pytest_focused_final" tests/unit/test_linear_op_checkpoint_quantization.py tests/unit/test_dsv4_standalone_operator_plans.py tests/unit/test_model_architecture_registry.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_operator_manifest.py tests/unit/test_linear_op_profiling_output_metadata.py tests/unit/test_glm_linear_op_profiling.py tests/unit/test_dsv4_linear_op_profiling.py tests/unit/test_profiling_vllm_parallel_state.py tests/unit/test_profiling_runtime_boundary.py`. **PASS: 216 passed, 3 skipped**, 11.91 seconds. Log: `focused_tests_cpu_final.log`.
- Broad CPU run: a `timeout 600s` Python runner launched pytest with a 540-second subprocess timeout. Exact expanded argv is retained in `final_pytest_command.json`; selected paths are in `required_test_paths.txt`. It selected the linear-op, shared-expert, family, operator, runtime-boundary, and changed-module importer tests. `test_native_profiling_model_type_policy.py` was excluded at collection because of the prerequisite below. Pytest arguments were `-q -ra -p no:cacheprovider --basetemp "$S/pytest_regression_final" -k 'not test_every_collective_launches_one_kernel_by_default and not test_moe_wrapper_typed_result_round_trips_through_manager and not test_moe_wrapper_initializes_runtime_with_routed_width'`. **PASS for this selection: 1020 passed, 3 skipped, 3 deselected**, 34.59 seconds. Log: `required_tests_cpu_final.log`.
- Cold import: `timeout 30s env PYTHONPATH="$R" PYTHONDONTWRITEBYTECODE=1 "$PY" -` with a meta-path blocker for torch, vLLM, CUDA, fasteners, and sklearn, importing `frontier.profiling.linear_op.main`, the three new linear-op modules, and `frontier.operators.deepseek_v4_axes`. **PASS**, exit 0, no blocked modules loaded. Artifact: `cold_import_final.json`.
- Import hygiene: `timeout 30s "$PY" /data/ycfeng/tmp/unused_imports.py` followed by all changed Python paths. Results are retained in `unused_imports_final.log`. New modules/tests and main/wrapper have no unused imports. The scanner reports `vllm` for the existing `# noqa: F401` registration imports in `fp8_block_linear.py:29` and `tensor_parallel_layers.py:123`; these imports register operations and were preserved.
- Source hygiene: `timeout 30s git diff --check` and bounded line counts. **PASS**. Final production line counts: families 943; axes 19; FP8 block binding 111; tensor-parallel layers 1130; main 1159; wrapper 281; V4 plans 123; V4 worker 218; quantization helper 49. Existing `linear_op_impl.py` is 1137 and `profiling_plan.py` is 606. Package builder is 207 lines. No module exceeds 2000 lines.

The GLM baseline uses the arguments exercised by `test_glm_linear_op_profiling.py`: model GLM-5.3, TP sizes 1/8, max tokens 8192, rows 1/256/8192, and topology 8/4/32. The V4 baselines use rows 1/2/3/4/768 and topologies 8/2/16 and 8/4/32. Shared CLI ordering remains descending: `[768,4,3,2,1]`. Grid plans preserve file order.

The additional matrix covers ten existing dense/MoE configurations, five token/TP settings each. It checks planning fidelity, not a simulation or GPU timing matrix. Early matrix attempts used an insufficient default max-token limit and unsupported generic shared topology; those harness inputs were corrected before the final 50-case PASS. An initial in-memory comparison also needed JSON normalization to match serialized tuple/list representations.

### Failures, corrections, and unavailable prerequisites

An initial broad CPU run returned **4 failed, 1019 passed, 3 skipped**. One failure was introduced by two new raw architecture-resolver call sites. Those were replaced with the existing `ModelConfig.get_model_architecture_profile()` method; the focused and final runs pass the call-site gate.

The remaining three tests were explicitly deselected in the final run:

- `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default`: **FAIL**, `OperatorSpec` has no `kernel_count`. The same attribute error was reproduced from HEAD's communication declarations in `regression.py`. It is unrelated to the eight profiling flags and outside B13's behavior scope.
- `test_profiling_governance_minimal_red.py::test_moe_wrapper_typed_result_round_trips_through_manager` and `::test_moe_wrapper_initializes_runtime_with_routed_width`: **FAIL due to prerequisite**, local Triton imports `setuptools`, which is absent. No installation was attempted.

`test_native_profiling_model_type_policy.py` also fails collection: without the system package path it lacks torch; with CPU torch it reaches Triton's missing `setuptools`. This file is not claimed to pass.

The final skips are the native shared-expert checks in `test_dsv4_linear_op_profiling.py` and `test_glm_linear_op_profiling.py`, plus the native parallel-state test in `test_profiling_vllm_parallel_state.py`; vLLM is not installed in the CPU environment. CPU torch is 2.1.0+cu121 and lacks `float8_e8m0fnu`.

The minimal-environment broad run completed with **888 passed, 5 skipped, 6 failed**: the existing kernel-count failure and five tests importing torch without a skip guard. Its log is `required_tests_minimal_final.log`. The separate runtime-boundary and cold-import checks pass; missing torch for actual worker construction is an environment limitation, not a claim that worker execution is available.

### Native and registry findings

**Quantization and scales.** Pinned `vllm/models/deepseek_v4/quant_config.py:79-83` selects e8m0 when checkpoint expert dtype is fp4. `fp8.py:254` reads that property and `fp8.py:341-350` creates `weight_scale_inv` as `torch.float8_e8m0fnu`. `fp8_utils.py:1264-1305` leaves non-FP32 scales uninitialized; it does not demonstrate that `fill_(1.0)` is supported. This environment cannot test that operation because its torch lacks the dtype. The producer now converts FP32 ones and copies the representable unit scales, but the worker must verify e8m0 conversion/copy support.

Native `Fp8LinearMethod.process_weights_after_loading` delegates to the selected kernel (`fp8.py:368-423`). The FlashInfer/DeepGEMM hybrid delegates processing to DeepGEMM (`scaled_mm/flashinfer.py:179-182`). DeepGEMM's helper explicitly accepts checkpoint e8m0/uint8, upcasts it, skips requantization, and converts the required layout (`fp8_utils.py:1087-1112`; `scaled_mm/deep_gemm.py:84-110`). Consequently the producer records both pre-processing and processed scale dtypes rather than assuming they remain equal.

GLM native `Fp8Config.from_config` reads ignored layers (`fp8.py:156-175`); HEAD's direct constructor supplied no list, whose default is empty (`fp8.py:109-110`). The checkpoint has 541 entries. The tests expose this difference. GLM-5.1 has no checked-in configuration here; its stand-in case reuses the GLM-5.3 quantization fields and is not an independent GLM-5.1 checkpoint check.

**RMSNorm.** The producer constructs native RMSNorm and calls `forward_cuda` with IR provider priority `vllm_c`. `layernorm.py:10` registers the providers, and `layernorm.py:74-114` routes the CUDA call through IR in the default non-batch-invariant mode. `vllm/kernels/vllm_c.py:23-43` invokes `_C.rms_norm`. This selects the native custom-op provider rather than an Inductor fallback. Actual kernel identity remains a GPU check.

**Embedding and LM head.** The native embedding constructor reads the TP group (`vocab_parallel_embedding.py:279-282`), and its forward performs an all-reduce for TP>1 (`:487-506`). Therefore the permitted fallback uses native `_get_indices`, native unquantized weight creation, native compiled masking, native embedding method, and `masked_fill_`, stopping before reduction. It measures rank 0 of the grid's TP shard and uniform full-vocabulary token IDs. LM head uses the same unquantized method's `apply`, without logits gather. The checked-in vocab is 129280, yielding local TP8 width 16160. The consumer deliberately leaves LM-head principal output shape unspecified; the producer preserves that and adds `local_output_shape`.

**Shared-expert parity.** Native `DeepseekV4MLP` uses merged gate-up `[intermediate_size]*2`, row-parallel down projection, SP `disable_tp=True`, no bias, and `SiluAndMulWithClamp` when configured (`nvidia/model.py:89-148`). Frontier's existing shared module uses TP1 column/row layers with the same dimensions, no reduction, and the same clamp custom op (`linear_op_impl.py:629-699`; common activation `:37-67`). The new FP8 binding supplies logical widths `[2048,2048]`, matching native merged semantics. The checkpoint supplies hidden 4096, intermediate 2048, bf16, and clamp limit 10. Structural projection/activation parity is supported; real backend and numeric parity are not yet GPU-tested. Synthetic weights and inputs are deliberate standalone profiling inputs, not checkpoint-value parity.

**Profiling-target users.** The search results are retained in `profiling_users.txt` in the parent scratch directory, with changed-module test inventory in `b13/importing_tests.txt`. Effects were inspected as follows:

- `operators/spec.py:332-364` filters/order-checks profiling operators; `operators/families.py:925-942` now returns/caches the eight MHC profiling names. Family execution filtering is unchanged.
- `operators/binding.py:264-329` carries the new flags into the frozen V4 layer manifest. Generic MoE/shared views still explicitly suppress their targets. Updated manifest/source tests cover this effect.
- `operators/typed_contracts.py:182` calls `bind_operator_query`; disabled execution raises before profiling admission. The new blocker test reproduces this independently of serialization.
- `config/quantization_manager.py:32-56,87-96,146-147` uses enabled attention families or unchanged MoE/shared families; its V4 precision-name set already iterates all operators. No GLM/non-V4 name-set change occurs.
- `linear_op/profiling_plan.py:31-58,239-242,279-281` uses unchanged FFN/shared/memory families and explicit producer membership. Generic V4 sweep remains excluded. The new grid builder alone selects MHC profiling operators.
- `attention/profiling_mapping.py:81-93` selects attention-family metrics, so the MHC flag change has no effect.
- `training/moe_trainer.py:40-94`, `profiling/moe/main.py:521-524`, `profiling/moe/moe_wrapper.py:864-866`, `prediction_model_identity.py:35-52`, `moe_operator_times.py:32-35`, and `moe_dataset_training.py:137-140` select only the unchanged MoE family.
- `prediction_family_trainers.py:365-366,559`, `sklearn_execution_time_predictor.py:2877,2889`, and `layer_contract_resolution.py:189-195,228-243` use unchanged FFN/shared layer-contract families. No consumer activation was added.
- `profiling/experimental/sglang/gdn_trace.py:199` uses the unchanged GDN family. Family, precision, manifest, static-cache, governance, and operator-parity tests cover the relevant generic users.

**Native dumps.** `native_kernel_notes.md` includes all eleven operators with line-level references in the four requested dumps. Corrections include the first-pre TF32 GEMM, prefill `vectorized_gather_kernel`, W-axis head/norm grids, and the 86-launch prefill post aggregate. The fused aggregate is 42 later attention-side calls plus 43 FFN-side calls; aggregate scale counts also include attention and are qualified rather than assigned wholly to shared projections.

Exactly four full rank traces were read: one decode and one prefill TP0/DP0 trace from each P2 run, under timeout and memory limits. Retained full names and decompressed line references are in `native_full_names.json`. All activation names end in `true,true,true,false`; `csrc/libtorch_stable/activation_kernels.cu:98-102` identifies the third boolean as `HAS_CLAMP`. Thus the native activation is the clamp variant. No additional rank files were read during continuation.

## 4. G2h rows, GPU time, and remaining decisions

| Phase | CSV rows | Operator measurements |
| --- | ---: | ---: |
| G2H_OPERATORS_CUDA_EVENT | 36 | 65 |
| G2H_SHARED_EXPERT_EP16_CUDA_EVENT | 5 | 15 |
| G2H_SHARED_EXPERT_EP32_CUDA_EVENT | 5 | 15 |
| G2H_OPERATORS_RECORD_FUNCTION | 36 | 65 |
| G2H_SHARED_EXPERT_EP16_RECORD_FUNCTION | 5 | 15 |
| G2H_SHARED_EXPERT_EP32_RECORD_FUNCTION | 5 | 15 |
| Total | 92 | 190 |

SP-local plan: four mHC operators on `[1,2,3,4,768]`, selected TP1. Descriptor plan: embedding, head, and final norm on `[1,2,4,8,16,24,6144]`; embedding selected TP8, head/norm selected TP1. Logits plan: LM head on rows 1 through 24, selected TP8. Each existing shared-expert topology invocation has three operators on the five L points, selected TP1 with its replicated topology contract.

Every phase has its own worker-local and cloud output directory. CUDA-event phases write `linear_op.csv`; record-function phases write `linear_op_kernel_only.csv`. Grid traces remain below `profiler_traces/<operator>_<rows>/profiler_traces/`; existing shared traces use distinct UUID filenames below their phase's `profiler_traces/`. Raw timer annotations retain the existing `vidur_` prefix; the tracer returns canonical operator names.

Planning estimate: six cold imports/JIT startups at 120–300 seconds each, plus 120–360 seconds for timing/trace work, gives **840–2160 seconds, approximately 0.23–0.60 GPU-hour** on one H200. There are 190 operator points, 3800 active calls and 570 warmup calls. This is an allowance, not measured runtime. Retain C1's one-hour G2h cap.

Claude's concurrent G1h-02 report now records 0.85 GPU-hours and approximately 0.92 GPU-hours total prior usage (`test_report_2026-10-28_g1h_worker_import_fix.md:80`). The package plan was updated accordingly. Adding this G2h estimate yields approximately 1.15–1.52 of the seven-hour cap, before other work. B13r remains separate; no budget allocation for it is implied.

Decisions and prerequisites for Claude:

1. **Canonical profiling admission:** authorize the smallest profiling-only ownership validation change in `typed_contracts.py` and/or binding, or handle it in the forthcoming V4 loader. Recommendation: preserve execution guards and separate profiling ownership validation from execution admission. Those files are outside this execution scope.
2. **GLM field parity:** decide whether to accept native checkpoint ignored-layer retention or revise the requirement. Recommendation: retain checkpoint fields and explicitly accept the documented constructor-field difference. Observation: fields differ. Inference: because this producer directly constructs `Fp8LinearMethod`, the ignored-layer filter appears unlikely to alter these leaf kernels, but GPU evidence is unavailable.
3. **Native checks:** provide a compatible CPU/native environment with vLLM and setuptools to rerun the excluded prerequisite-dependent checks. Actual e8m0 scale conversion, native kernel selection, RMSNorm provider, embedding compilation, and kernel/grid parity require the authorized worker run.
4. **Known kernel-count failure:** route the existing communication `kernel_count` issue to its owner. It was not repaired or hidden as an accepted B13 result.
5. **Commit/package handoff:** explicitly include the ignored G2h files, commit the assigned changes, replace `source_revision` with that commit, and build only then. The current pinned revision does not include uncommitted B13 code.

## 5. Review handoff

Review the checkpoint detection and HF context, V4 merged gate-up widths, native mHC signatures/dtypes, embedding forward-part fallback, RMSNorm IR provider, per-plan iteration, and additive row metadata. Inspect the contract-admission blocker test rather than treating successful serialization as loader acceptance.

Rerun the focused CPU command and G2h `--check-plan` after any integration changes. The exact broad-test argv, baseline JSON pairs, 50-scenario evidence, cold-import receipt, line counts, unused-import results, package-check receipt, and native full-name extraction are retained under the scratch path above.

No package was built, no GPU timing was measured, no native worker image was validated, and Claude has not accepted the changes. Final acceptance, commits, package construction, submission, and post-run native comparison remain with Claude.
