## 1. Assignment and outcome

Implemented the CPU portion of B12, stage 2 assignment B12 / file entry 16. Final review and acceptance remain with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Edits were confined to `frontier/profiling/moe/` excluding frozen `balanced_routing.py`, additions to `frontier/moe_ep_workload.py`, `tests/unit/`, and the new `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/` directory. Scratch storage is `/data/ycfeng/tmp/codex-implement/68caa2f4-8400-407e-a297-e9e5e66ce13e/b12`.

The implementation follows the assignment, applicable AGENTS.md guidance, task requirements/plan/issues, design §3.5 and §3.7 / entry 16, C1 §4.3 and §5, the accepted H800 MoE package, and the P1h bridge package mechanism. It preserves the decisions to use contiguous native expert ownership, the existing Frontier routing algebra, every distinct lane permutation, the lowest representative rank for each `(EP, M, class, L)`, all four canonical targets in every row, and the existing gate implementation.

The new lane input requires positive SP-local row counts, exactly one EP size, physical TP1, balanced load distribution and `balanced_topk`. EP must be divisible by the number of DP lanes. It replaces the token grid; `--extra_num_tokens` is also rejected because it would introduce an additional token-grid input. Model expert count, top-k and local expert width come from model configuration.

No package was built. No GPU, network, installation, submission, deployment, Git mutation, task-record update or delegation was performed.

## 2. Changed files

| File | Change and purpose |
| --- | --- |
| [frontier/moe_ep_workload.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/moe_ep_workload.py:886) | Additions only: `materialize_balanced_ep_workload` and `materialize_balanced_rank_classes` reuse the existing Hamilton counts and contiguous ownership. The latter returns each rank's `(local routed rows, experts with an extra row, base rows per expert)` triple, including zero classes. |
| [frontier/profiling/moe/main.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/main.py:696) | Adds `--dp_lane_local_rows`; shares rank planning between dry run and execution; propagates task rank and gate rows through sequential, multiprocessing and Ray paths. Grid tasks select rank 0. Lane dry runs expose detailed `rank_tasks`; grid dry-run JSON remains unchanged. Removes two existing unused imports. |
| [frontier/profiling/moe/moe_wrapper.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_wrapper.py:303) | Validates constructor EP rank and uses its expert window; removes `_get_profiling_ep_rank`. Checks generated balanced local counts against the shared workload algebra. `profile()` passes L to gating while retaining M for shuffling/grouped GEMM and reusing one routing sample. Every result receives `ep_rank`, three class columns and `gating_num_tokens`. Balanced rows also receive the pinned strategy hash. Removes an existing unused import. |
| [frontier/profiling/moe/moe_impl.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_impl.py:81) | `uniform_topk` delegates to frozen `balanced_topk` with offset 0, preserving its historical int32 default IDs and third return value. The gate implementation is unchanged. |
| [frontier/profiling/moe/moe_rank_plan.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_rank_plan.py:80) — new | Torch-free lane parser, argument validation, distinct multiset permutations, `MoERankTask`, and planner. Deduplicates `(EP, M, class, L)` and selects the lowest rank, retaining a lane-placement witness. |
| [frontier/profiling/moe/moe_profile_identity.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_profile_identity.py:7) — new | Hashes frozen `balanced_routing.py`, imports attention's authoritative `BALANCED_ROUTING_SHA256`, and fails on mismatch. Uses the common `routing_strategy_sha256` column name. |
| [tests/unit/test_moe_rank_plan.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_moe_rank_plan.py:21) — new | 25 cases for exact G3h classes/tasks, permutations, deduplication, lowest rank, invalid arguments, dry runs and builder plan checks. |
| [tests/unit/test_moe_rank_routing.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_moe_rank_routing.py:36) — new | 46 cases for rank validation, all 33 representative routing samples on CPU, gate/routed row plumbing, row identity, count/hash failures, worker propagation and sequential/Ray scheduling. |
| [tests/unit/test_balanced_routing.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_balanced_routing.py:349) | Adds 26 distinct G3h placement cases. Executes the native registry, SP metadata and expert-map functions, comparing every rank's counts/classes and concatenated native IDs with offset-0 gathered IDs. |
| [tests/unit/test_moe_routing_input_contract.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_moe_routing_input_contract.py:7) | Updates its constructor-bypassing fixture with rank/runtime fields required by the wrapper; removes an unused torch import. |
| [tests/unit/test_profiling_governance_minimal_red.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_profiling_governance_minimal_red.py:995) | Updates the constructor-bypassing row fixture with rank/runtime fields. Existing file length stays 2,071 lines. |
| [tests/unit/test_dsv4_moe_profiling.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_moe_profiling.py:66) | Makes the existing CUDA/ROCm boundary test explicitly mock the modular vLLM API it assumes, so absent installed vLLM does not select a different test branch. Removes two unused imports. |
| [tests/unit/test_measurement_family_selector.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_measurement_family_selector.py:68) | Repairs existing scheduler/request fixtures with `_uses_output_placeholders`, `num_processed_tokens` and `num_prefill_tokens`, required by current scheduler behavior. Scheduler source is unchanged. |
| [g3h/package_plan.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/package_plan.json) — new | Records assigned H200 settings, both methods, lane vectors, native/source pins, launcher base, expected tasks and GPU estimate. Explicitly qualifies the existing gate/backend limitation. |
| [g3h/expected_tasks.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/expected_tasks.json) — new | Freezes the complete ordered 33-task `(EP, M, class, L, rank)` list. |
| [g3h/prepare_package.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/prepare_package.py:61) — new | Checks all four CPU dry-run plans against the expected task list. The normal builder requires committed full revisions and snapshots source/native bytes, inherits the launcher, and emits per-method phases with separate EP output directories and manifests. Only `--check-plan` was executed. |

The fixture repairs were local verification deviations. Three selector failures reproduced with the unchanged HEAD version of `moe_ep_workload.py`, establishing that they were stale fixtures rather than a B12 workload regression. After fixture repair, all 37 selector tests passed. No unrelated production behavior was changed.

Concurrent attention, census and collective-analysis changes were present, together with task-record changes and untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`. They were left untouched. The frozen balanced strategy, attention producer, B10/B11 files, consumer, trainer and H800 package files were not edited.

## 3. Verification and acceptance results

All shell checks used `ulimit -v 31457280` and a timeout. The interpreter was `/data/ycfeng/envs/frontier-py310/bin/python` (the supplied Python 3.10 environment). Bytecode and pytest cache output were disabled; temporary output went to B12 scratch.

For CPU torch checks, the environment was:

```sh
B12_PY=/data/ycfeng/envs/frontier-py310/bin/python
B12_SCRATCH=/data/ycfeng/tmp/codex-implement/68caa2f4-8400-407e-a297-e9e5e66ce13e/b12
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69
export TMPDIR="$B12_SCRATCH"
ulimit -v 31457280
```

The repeated environment prefixes of the executed commands are factored into that setup here. `/usr/lib/python3/dist-packages` supplies already-installed setuptools required by installed Triton; nothing was installed.

The final required-suite command was:

```sh
timeout 75s "$B12_PY" -m pytest -q -ra -p no:cacheprovider \
  --basetemp="$B12_SCRATCH/pytest-required-final" \
  tests/unit/test_dsv4_moe_profiling.py \
  tests/unit/test_balanced_routing.py \
  tests/unit/test_moe_profiling_output_metadata.py \
  tests/unit/test_moe_load_distribution_contract.py \
  tests/unit/test_moe_routing_input_contract.py \
  tests/unit/test_moe_routing_runtime.py \
  tests/unit/test_moe_gating_constructor_boundary.py \
  tests/unit/test_moe_ep_workload_materializer.py \
  tests/unit/test_moe_rank_plan.py \
  tests/unit/test_moe_rank_routing.py
```

Observed: **291 passed, 2 skipped in 34.59 s**, exit 0. The two skips are `test_dsv4_moe_profiling.py` checks requiring installed vLLM. Native-source registry checks executed successfully.

The broader import/reference regression command was:

```sh
timeout 270s "$B12_PY" "$B12_SCRATCH/run_importing_tests.py"
```

The script discovers referencing unit modules with `rg`, includes the required files, and runs pytest with an internal 240-second timeout. Observed: **1,531 passed, 24 skipped in 42.45 s**, exit 0, across **55 modules**. Five skips require installed vLLM; 19 cover explicitly retired PD-AF contracts. The exact expanded pytest argv, selected files and results are retained in `importing_command_final.txt`, `importing_tests_final.json` and `importing_tests_final.log`. This broad run preceded removal of three unused test imports; the required suite was rerun after that final cleanup.

For the supported minimal-environment subset, PYTHONPATH omitted both system package directories:

```sh
timeout 45s env PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis" \
  "$B12_PY" -m pytest -q -ra -p no:cacheprovider \
  --basetemp="$B12_SCRATCH/pytest-minimal-supported" \
  tests/unit/test_moe_rank_plan.py \
  tests/unit/test_moe_profiling_output_metadata.py \
  tests/unit/test_moe_routing_runtime.py \
  tests/unit/test_moe_ep_workload_materializer.py \
  tests/unit/test_balanced_routing.py
```

Observed: **74 passed, 1 skipped in 4.77 s**, exit 0. Balanced routing skips because torch is absent. Exploratory minimal-environment collections that included the existing unconditional torch imports in the load-distribution or gating-constructor tests failed with `ModuleNotFoundError: torch`; those modules passed in the CPU torch environment above.

### Acceptance 1 — grid compatibility: PASS

Both commands below used the H800 package's exact common arguments:

```sh
timeout 120s "$B12_PY" -m frontier.profiling.moe.main \
  --models deepseek-ai/DeepSeek-V4-Flash --device h800 --num_gpus 1 \
  --disable_ray --num_tensor_parallel_workers 1 --expert_parallel_sizes 16 \
  --num_tokens_list 12288 6152 32 16 --max_tokens 16384 \
  --load_distributions balanced --num_samples_per_distribution 1 \
  --routing_runtime_path balanced_topk --yes --profile_method cuda_event --dry_run
```

The second command substituted `record_function`. Parsed JSON exactly equaled `moe/h800_moe_local_01/plans/{cuda_event,record_function}.json`: four steps per method, **no changed plan fields**. Existing import diagnostics preceding JSON were excluded from parsing.

### Acceptance 2 — classes and native counts: PASS

The shared class function reproduces all C1 observed and margin classes: **11 distinct `(M, class)` pairs for EP16** and **14 for EP32**. The 26 added native placement cases cover every distinct permutation of every G3h lane vector: 7 EP16 placements and 19 EP32 placements. Every rank's counts equal native balanced routing plus native linear expert mapping.

Both native source roots resolve to revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`. The tested `determine_expert_map`, routing registry classes, `_compute_sp_num_tokens` and `DPMetadata` ASTs were also compared with `/data/ycfeng/tmp/opencode/vllm-v0.28.0` and found identical.

### Acceptance 3 — CPU producer routing: PASS within CPU scope

All **33 representative tasks** generate the algebra's local count vector and the selected rank's contiguous expert window. Gathered IDs equal frozen `balanced_topk(offset=0)`; balanced weights are float32. Profile plumbing tests confirm that gating receives L, routed paths receive M, and shuffling/grouped GEMM share the same routing sample.

The profile plumbing test substitutes the CUDA gate/timer and routed execution methods. Actual gate allocation/execution, CUDA timers, Marlin repacking, GEMMs, reduction and CUDA kernel sequence/timing were **NOT_RUN**, as assigned. CPU routing tests bypass the GPU constructor and use CPU tensors.

### Acceptance 4 — G3h plans: PASS

Executed:

```sh
timeout 120s "$B12_PY" task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/prepare_package.py --check-plan
```

Observed: four plans, **13 EP16 tasks and 20 EP32 tasks per method**, `package_built=false`, zero GPU/platform operations. Each ordered task list equals `expected_tasks.json` exactly.

Class below means `(local_routed_rows, local_experts_with_extra_row, base_rows_per_expert)`.

| EP | M | Class | L | Rank |
| --- | --- | --- | --- | --- |
| 16 | 16 | (0, 0, 0) | 1 | 6 |
| 16 | 16 | (16, 16, 0) | 1 | 0 |
| 16 | 32 | (0, 0, 0) | 2 | 12 |
| 16 | 32 | (16, 16, 0) | 2 | 0 |
| 16 | 48 | (16, 0, 1) | 3 | 2 |
| 16 | 48 | (32, 16, 1) | 3 | 0 |
| 16 | 64 | (16, 0, 1) | 4 | 8 |
| 16 | 64 | (32, 16, 1) | 4 | 0 |
| 16 | 6152 | (2304, 0, 144) | 1 | 3 |
| 16 | 6152 | (2304, 0, 144) | 768 | 3 |
| 16 | 6152 | (2320, 16, 144) | 1 | 0 |
| 16 | 6152 | (2320, 16, 144) | 768 | 0 |
| 16 | 12288 | (4608, 0, 288) | 768 | 0 |
| 32 | 32 | (0, 0, 0) | 1 | 24 |
| 32 | 32 | (8, 8, 0) | 1 | 0 |
| 32 | 64 | (8, 0, 1) | 2 | 16 |
| 32 | 64 | (16, 8, 1) | 2 | 0 |
| 32 | 96 | (16, 0, 2) | 3 | 8 |
| 32 | 96 | (24, 8, 2) | 3 | 0 |
| 32 | 128 | (24, 0, 3) | 4 | 0 |
| 32 | 6168 | (1152, 0, 144) | 1 | 18 |
| 32 | 6168 | (1152, 0, 144) | 768 | 18 |
| 32 | 6168 | (1160, 8, 144) | 1 | 0 |
| 32 | 6168 | (1160, 8, 144) | 768 | 0 |
| 32 | 12304 | (2304, 0, 288) | 1 | 12 |
| 32 | 12304 | (2304, 0, 288) | 768 | 12 |
| 32 | 12304 | (2312, 8, 288) | 1 | 0 |
| 32 | 12304 | (2312, 8, 288) | 768 | 0 |
| 32 | 18440 | (3456, 0, 432) | 1 | 6 |
| 32 | 18440 | (3456, 0, 432) | 768 | 6 |
| 32 | 18440 | (3464, 8, 432) | 1 | 0 |
| 32 | 18440 | (3464, 8, 432) | 768 | 0 |
| 32 | 24576 | (4608, 0, 576) | 768 | 0 |

Thus each method produces 33 rows; `cuda_event` / `moe.csv` and `record_function` / `moe_kernel_only.csv` together produce **66 rows**.

### Acceptance 5 — existing regressions: PASS with stated skips

The required existing files and every discovered unit-test importer/reference were exercised as described above. Installed-vLLM checks remain skipped; native-source AST execution ran. No unresolved implementation failure remains.

### Acceptance 6 — new tests: PASS

**97 new parametrized cases**: 25 planner/builder, 46 producer/rank/dispatch, and 26 native placement cases. Collection of the two new files plus balanced routing reported **199 tests**, comprising those 97 cases and 102 existing balanced-routing cases. The count check used `pytest --collect-only -q -p no:cacheprovider` on those three files under the CPU torch/native-source environment, with a 45-second timeout.

### Acceptance 7 — zero class and gate parity findings

Native references in this subsection are under `/data/ycfeng/tmp/opencode/vllm-v0.28.0` at the pinned revision.

**Zero class — source inspection supports measurement.** The producer retains positive global M even when a rank owns no routed assignments. `frontier/profiling/moe/moe_vllm_kernel.py:952` obtains the checkpoint Marlin adapter and its `_step` calls `apply` unconditionally at line 981. `mxfp4_marlin_experts.py:100` forwards to native `MarlinExperts.apply`.

Native `experts/marlin_moe.py:759` invokes `fused_marlin_moe`; line 340 performs alignment with `ignore_invalid_experts=True`; lines 135 and 202 call the two Marlin GEMMs; lines 391 and 911 select/call reduction. There is no host early return conditioned on zero rank-local counts. Native launch sites are `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:695` and `:716`, `marlin_moe_wna16/ops.cu:533`, and the expert-map reduction launches around `moe_align_sum_kernels.cu:799` and `:849`. Device threads may have no valid local work, but the host path still launches the operations. No source contradiction requiring this part to stop was found. Actual zero-class CUDA traces remain unmeasured.

**Gate rows — source agrees with L.** Native V4 reduces/scatters attention output before its FFN (`nvidia/model.py:968`, FFN call at `:990`). The native MoE runner applies its gate at `runner/moe_runner.py:863`, before dispatch at `:869`. The selected native configurations enable this SP path through `nvidia/model.py:804` and `:709`.

**Gate implementation — parity difference found and left unchanged.** Native V4 constructs `GateLinear(out_dtype=torch.float32)` at `nvidia/model.py:571–577`, passes it to the MoE factory at `:693`, and requests float32 router logits at `:706`. Producer `moe_impl.py:162–173` uses `ReplicatedLinear(disable_tp=True)` for balanced routing, or a torch fallback. Its `GateLinear` branch at `:153` belongs to grouped top-k routing. Wrapper `moe_wrapper.py:154`, `:217` and `:269–275` use the model dtype for gate weights/inputs, which is BF16 for V4, without requesting FP32 output. Native `router/gate_linear.py:183–241` also contains specialized small-row dispatch and BF16-to-FP32 GEMM/fallback conversion absent from the producer path.

Producer balanced gate routing calls the shared helper with default offset 0 (`moe_impl.py:213`). Native registered routing computes a prefix offset from SP-local metadata (`balanced_routing.py:56–76`), invoked through `RoutingSimulatorRouter` (`routing_simulator_router.py:333`). Concatenated native chunks equal the producer's gathered offset-0 sample, as tested. A rank's isolated gate IDs may differ because its native offset is nonzero; native gate/kernel parity is not claimed.

### Acceptance 8 — GPU time estimate

The accepted H800 job measured 8 rows in 92 seconds, including startup. Scaling by the planned 66 rows gives `92 / 8 × 66 = 759 seconds`, or **0.210833 GPU-hours**, an 8.25× job-time multiplier. This is an estimate, not an H200 measurement; H200 performance and rank-specific routed-weight rebuilds remain unknown. The plan retains C1's **2 GPU-hour G3h allowance**, within the combined **7 GPU-hour G1h–G4h cap**.

### Acceptance 9 — hygiene

Executed `verify_cpu_final.py` under a 180-second outer timeout. It runs both baseline comparisons, the builder check, independent cold imports, syntax compilation without bytecode, the specified unused-import analyzer, and `git diff --check`. Its subprocesses have explicit timeouts. Results and exact commands are retained in `cpu_checks_final.json` and `cpu_check_commands_final.txt`.

- All 14 changed/new Python files compile and have **no unused imports** according to `/data/ycfeng/tmp/unused_imports.py`.
- `git diff --check`: PASS.
- Independent cold imports of `frontier.profiling.moe.main`, `frontier.moe_ep_workload` and `frontier.profiling.moe.moe_rank_plan`: exit 0, with no loaded torch, vLLM or CUDA modules in the minimal environment. Main's pre-existing optional torch import at line 43 still imports torch when installed; that behavior was not changed.
- Frozen balanced-strategy SHA-256 is `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`, equal to the imported attention constant.

| Python file | Final lines |
| --- | --- |
| `moe_ep_workload.py` | 945 |
| MoE `main.py` | 1230 |
| `moe_wrapper.py` | 955 |
| `moe_impl.py` | 613 |
| `moe_rank_plan.py` | 110 |
| `moe_profile_identity.py` | 14 |
| G3h `prepare_package.py` | 214 |
| `test_balanced_routing.py` | 380 |
| `test_dsv4_moe_profiling.py` | 224 |
| `test_measurement_family_selector.py` | 1006 |
| `test_moe_routing_input_contract.py` | 132 |
| `test_moe_rank_plan.py` | 128 |
| `test_moe_rank_routing.py` | 186 |
| `test_profiling_governance_minimal_red.py` | 2071 |

All changed production and new modules are below 2,000 lines. The existing governance test is the retained size exception: **2,071 before and after**, with only fixture field assignments changed. An unrelated split of that existing test was deferred.

## 4. Package and remaining decisions for Claude

The selected launcher base is the same base as P1h bridge:

`task_memory/task_2026-10-23_v4_remaining_measurements/balanced_decode/h800_decode_balanced_02`.

Its directory name contains H800, but its reusable command runner supports a plain producer command and its submission path takes GPU type and charged group from job configuration. Worker `_run_step` at `launch/glm_producer_worker.py:78` executes arbitrary argv using the source snapshot as cwd; lines 134–135 set source PYTHONPATH. Launcher `glm_producer_jobs.py:258` and `:267` use configured GPU type/charged group. All **530 base manifest inputs** verified. Image, CUDA compatibility path and cloud mount match P1h's recorded values. No unavailable CPU prerequisite was found; actual H200 image/runtime qualification remains with Claude.

The plan contains the assigned settings: destination `package_01`, name `G3H_MOE`, tag `v4-h200-g3h-moe-rank-classes-20261028-01`, device H200 / one GPU / `step_main`, and output `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_moe/g3h_01`.

Remaining lead actions:

1. Review and commit the B12 code, tests and package inputs. G3h is ignored by the repository's current ignore rules, so explicitly include its three new files; they do not appear as ordinary untracked files in `git status`.
2. Replace `package_plan.json.source_revision`, currently the pre-B12 HEAD `5083d0369a5f9b86c333b30fce78154a3364225f`, with the committed B12 source revision. The builder intentionally refuses source bytes differing from the pin.
3. Decide the follow-up for native gate parity before admitting these gate measurements as native V4 serving costs. B12 retains the existing gate and labels the package as isolated eager measurements. A separate authorized change could use native `GateLinear` and validate its dispatch/dtypes; otherwise retain the gate rows as explicitly qualified candidates.
4. Build, qualify and submit the package through the lead's approved workflow. Verify zero-class kernel launches and measured runtime on H200. None of these actions was executed here.
5. Implement the later entry-6 consumer and its class admission independently; consumer, loader and trainer work remain outside B12.

There is no remaining CPU implementation blocker. CUDA execution, installed-vLLM tests, graph-replay equivalence and native gate parity remain limitations rather than fabricated successful checks.

## 5. Review handoff

Inspect the shared algebra additions, lowest-rank/permutation planning, rank propagation in all execution paths, gate L versus routed M plumbing, four-target row preservation and pinned balanced identity. Review the package's separate EP output directories and committed-source checks, and the qualification explaining the gate difference.

Rerun the final required-suite command and `prepare_package.py --check-plan` after committing/pin replacement. The broader 55-module command and its exact argv are available in scratch if a larger regression rerun is needed. Check the generated package against the 33-task list and expected 66 output rows during the lead's later build and GPU validation.

Scratch contains the regression logs/file lists, baseline selector experiment, pytest temporary directories, `verify_cpu_final.py`, `cpu_checks_final.json` and exact command records. `g3h/` currently contains only the plan, expected tasks and builder; `package_01` is absent.
