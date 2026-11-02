B12b is implemented and the assigned CPU checks are complete. Final review and acceptance remain with Claude.

The objective was to make G3h measure the native V4 gate and routed-ID kernel specializations, while applying the B12 review fixes. Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; verified HEAD: `eaee355046ea809b716f2e9abf8b00d18fa72cea`. The writable scope was `frontier/profiling/moe/` excluding frozen `balanced_routing.py`, B12 additions in `frontier/moe_ep_workload.py`, B12/new unit tests excluding B11b files, and `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/`.

Applicable AGENTS.md instructions and the named requirements, design, coverage, and B12 assignment/report artifacts were inspected. No applicable CLAUDE.md was found. B12b supersedes B12's restriction against changing the balanced gate and its unconditional row-identity additions. The rank planner, per-rank grouping, class algebra, gate on SP-local L rows, routing offset 0, and accepted measurement-selector fixture repairs were retained.

Pre-existing B12 modifications included the workload algebra, main/wrapper changes, rank planner and identity modules, and B12 tests. Concurrent attention changes and task-record changes were also present. B12b preserved those outside its assigned edits, including `.cortexkit/`, `.pi/`, and `remaining_measurements/`. No Git state changes, dependency installation, network access, GPU operations, package build, deletion, delegation, or task-record update was performed.

The following 12 paths changed incrementally from the B12 starting state. “New” identifies files introduced by B12b; some other files were already untracked B12 additions.

| Changed path | Change and reason | Final lines |
| --- | --- | ---: |
| [moe_impl.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_impl.py:156) | Shares the existing grouped-sigmoid GateLinear construction with balanced_topk; records `vllm_gate_linear`. Restores the exact HEAD `uniform_topk` body. Adds the single routed-ID dtype policy at lines 315–324 and applies it to shuffling at line 392. | 648 |
| [load_distribution.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/load_distribution.py:86) | The untimed balanced sample calls frozen `balanced_topk(token_offset=0)` directly, preserving IDs without changing the measured uniform_topk implementation. | 275 |
| [moe_wrapper.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_wrapper.py:344) | Prepared samples use the shared ID dtype and carry the selected routing path into grouped execution; the shuffler receives that path. `_build_profile_result` adds rank/class/L identity only for balanced distributions, and `profile()` supplies the distribution. Removes redundant bare hash checks while preserving constructor validation and row metadata. | 965 |
| [moe_vllm_kernel.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_vllm_kernel.py:786) | Adds a routing-path argument with the original standard default; the functional cast at lines 949–952 uses the shared dtype policy. Removes unused `triton` and `fused_moe_kernel` imports found by the required audit. | 1279 |
| [main.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/main.py:385) | Retains lane validation in `parse_args`; removes repeats in `_lane_rank_tasks` and `_dry_run`. Removes the redundant bare dry-run hash call while retaining the checked hash in lane-plan JSON at line 1038. | 1226 |
| [moe_profile_identity.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/moe/moe_profile_identity.py:6) | Moves the attention hash constant import to module scope. Cold imports confirm this introduces neither an import cycle nor torch/vLLM/CUDA loading. | 14 |
| [test_balanced_routing.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_balanced_routing.py:198) | Updates the B12 routing-timer fixture to mock native GateLinear, replacing the withdrawn balanced torch fallback. | 391 |
| [test_moe_rank_routing.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_moe_rank_routing.py:25) | Models constructor-populated routing metadata, expects balanced int64 IDs, and checks constructor rejection against the module-bound strategy pin. | 191 |
| [test_moe_native_v4_gate_and_ids.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_moe_native_v4_gate_and_ids.py) — new | Adds 34 CPU regression cases: 5 gate, 8 functional dtype, 4 shuffling dtype, 4 prepared-sample dtype, 9 non-balanced HEAD-row comparisons, and 4 policy-forwarding cases. | 263 |
| [package_plan.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/package_plan.json) | Records native GateLinear, model-dtype inputs/parameters, FP32 logits, and int64 balanced IDs. Replaces the old gate mismatch qualification with the remaining source/runtime qualifications and adds native gate/model/cuteDSL references. Preserves the B12 source revision. | — |
| [prepare_package.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/prepare_package.py) | Updates builder notes and includes the native-kernel notes in a future package. No package was built. | 217 |
| [native_kernel_notes.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/native_kernel_notes.md) — new | Documents gate dispatch, every measured routed-ID consumer, native census comparisons, remaining differences, and builder handoff. Final source inspection corrected several imprecise line references; test-covered code was unchanged afterward. | — |

`frontier/moe_ep_workload.py` and `moe_rank_plan.py` received no incremental B12b edits. Their final line counts are 945 and 110. The accepted measurement-selector fixture repairs were left intact.

Two implementation choices preserve existing behavior. GateLinear's native default `bias=False` is used rather than adding an argument to the shared constructor, so the existing grouped-sigmoid constructor arguments remain exactly HEAD's arguments. Generated routing samples carry an optional internal routing-path field; externally supplied older samples without that field retain the original standard/int32 functional default. The additional fixed-int32 cast in shuffling was corrected because it is another measured consumer of the routed IDs. No CSV dtype column was added.

The native comparison uses read-only `/data/ycfeng/tmp/opencode/vllm-v0.28.0`, verified revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`. The gate, Marlin, alignment/reduction, and activation source files compared against the configured native-test root have identical hashes. Native file references below are relative to that source root; census references are relative to `h3o_operator_costs/`.

Native V4 constructs GateLinear with hidden size, routed-expert count, `bias=False`, and `out_dtype=torch.float32` at `vllm/models/deepseek_v4/nvidia/model.py:571–577`; router logits are explicitly FP32 at line 706. Native parameters default to the current model dtype through `layers/linear.py:251–253`, under the loader's model-dtype context at `model_loader/base_loader.py:53`. The producer now uses that same GateLinear implementation and supplies `router_config.dtype` before dispatch eligibility is computed (`moe_impl.py:156–180`). GateLinear's default bias is false (`router/gate_linear.py:49–58`).

The wrapper retains model-dtype parameters (`moe_wrapper.py:155`), moves the gate to that dtype/device at line 218, and allocates gate input on L rows in the model dtype at lines 271–278. Its FP32 output-dtype attribute survives the module move. V4-Flash therefore has BF16 input and weight with FP32 logits. Dummy initialization remains at lines 245–250. The producer calls the native gate directly at `moe_impl.py:219–223`, matching `runner/moe_runner.py:863`.

The native dispatch comparison is:

| GateLinear branch | Native source | Producer comparison |
| --- | --- | --- |
| Small-row BF16 cuteDSL GEMM | `router/gate_linear.py:183–189` | Same native call and eligibility; used for at most 16 BF16 rows when available. |
| DSV3 specialized GEMM | `:192–198` | Same implementation; supported hidden sizes exclude V4-Flash's 4096. |
| FP32-weight specialized dispatch | `:203–210` | Same implementation; V4-Flash has BF16 parameters and does not meet the supported shape/weight conditions. |
| Blackwell BF16x3 branch | `:213–219` | Same implementation; its FP32-weight/Blackwell conditions do not apply to the planned H200 BF16 gate. |
| BF16-to-FP32 cuBLAS `torch.mm` | `:222–224` | Same implementation and FP32 output epilogue. |
| Replicated-linear fallback/output conversion | `:227–232` | Same native fallback. |

For H200, hidden size 4096, E256, BF16, source conditions imply that L=1,2,3,4 can select cuteDSL when available and otherwise cuBLAS; L=768 selects cuBLAS. This is an inference from source, not a GPU observation. Eligibility evidence is `gate_linear.py:39–46,83–113,123–145`.

Remaining gate differences are synthetic weights/inputs and an isolated eager timing envelope rather than native integrated execution or graph replay. Native supplies a checkpoint prefix at `nvidia/model.py:576`; the profiler uses dummy parameters. Native balanced routing applies an EP-rank prefix offset (`balanced_routing.py:51–70`); the isolated gate uses offset 0 (`moe_impl.py:232–235`), as accepted by Claude. Only the ID values differ there; the gathered routed sample matches the native offset-0 global IDs. Worker platform, installed vLLM/API compatibility, and cuteDSL availability remain unqualified. The builder records native sources as references rather than a worker overlay (`prepare_package.py:197–199`). No gate implementation or dtype mismatch remains in the source comparison.

Native naive DP/EP prepare returns no indices override (`prepare_finalize/naive_dp_ep.py:103–104`), and simulated routing passes it through (`router/routing_simulator_router.py:323–340`). Frozen `balanced_routing.py:26–28` returns `torch.long` when that override is None. The producer's single policy at `moe_impl.py:315–324` therefore assigns int64 to balanced_topk and preserves int32 for grouped_topk_sigmoid, standard_fused_topk, and uniform_topk. Prepared samples, shuffling, and functional execution all apply this policy.

Every measured routed-ID consumer was traced:

| Consumer | ID-dependent kernel/argument and evidence | Comparison with `evidence/runtime_census/p2_32.txt` |
| --- | --- | --- |
| Shuffling and grouped alignment | `vllm::moe::moe_align_block_size_kernel<long>` and `count_and_sort_expert_tokens_kernel<long>` read int64 IDs. Native integral dispatch: `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:655–656`; launches: 695 and 716. Call sites: `moe_impl.py:398–404`, native `experts/marlin_moe.py:340–346`. | Alignment `<long>` is observed at line 228. Sorting `<long>` follows from the source launch; it is absent from the truncated top-kernel census list. |
| Small alignment alternative | `moe_align_block_size_small_batch_expert_kernel<long,256>` at CUDA lines 658–686. | Requires global E≤64, so it is unreachable for G3h global E256. |
| Workspace sizing | Uses ID shape to determine M/top-k, not ID dtype: producer `mxfp4_marlin_experts.py:53–79`, native `experts/marlin_moe.py:670–691`. | No additional routed-ID specialization. |
| Both Marlin GEMMs | Consume int32 sorted-token IDs, expert IDs, and padded counts, plus FP32 top-k weights; they do not consume the original routed IDs. Native `experts/marlin_moe.py:135–160,202–225`; alignment outputs `moe_align_block_size.py:81–88`; Marlin CUDA `marlin_moe_wna16/ops.cu:398–402,533–538`. | Same Marlin family is observed at line 223. Numeric template literals ending in `l` do not encode routed-ID dtype. |
| SwiGLU clamp/invalid-slot masking | `_swiglu_limit_pad_aware_kernel` receives an int64 ID pointer and indexes an int32 expert map. Native `utils.py:484–513,553–564`; forwarding at `experts/marlin_moe.py:169–185`. | Kernel name is observed at line 226; the displayed Triton name has no `<long>` suffix. |
| EP-aware reduction | `vllm::moe::moe_sum_vec_kernel<c10::BFloat16,long,6,true>` receives int64 IDs and an int32 expert map. Native CUDA lines 799–802 and 819–841; native forwarding at `experts/marlin_moe.py:911–921`; producer adapter `mxfp4_marlin_experts.py:97–112`. | Exact BF16/long/top-k-6/pad-aware specialization is observed at line 227. |
| Reduction alternatives | Dynamic-vector and scalar kernels also specialize their ID type as long: CUDA lines 849–863. | The fixed vector top-k-6 branch is selected by source conditions for aligned contiguous G3h BF16 rows; actual launches remain for GPU verification. |
| Post-alignment gather | Uses int32 sorted-token IDs at `moe_impl.py:406–408`. | No further routed-ID dtype dependency. |

Standalone shuffling retains its existing block-size selection and default `ignore_invalid_experts=False`; native grouped Marlin uses its own rule and `ignore_invalid_experts=True` (`moe_impl.py:390–404` versus native `experts/marlin_moe.py:326–346`). This auxiliary shuffling row should not be charged as an additional native cost outside the grouped envelope. Native int32 expert maps/alignment outputs are intentional and do not represent a remaining routed-ID mismatch. EP dispatch/combine collectives, graph equivalence, and actual zero-local-class launches remain outside these CPU-qualified local scopes.

Verification used `/data/ycfeng/envs/frontier-py310/bin/python`, verified Python 3.10.6. CPU torch was 2.1.0+cu121, loaded through the system package paths; CUDA operations were mocked in the new boundary tests. All executed commands used `ulimit -v 31457280` and timeouts. Scratch, bytecode prevention, and pytest output settings kept generated files in `/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b`.

CPU-test environment:

```sh
PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages
PYTHONDONTWRITEBYTECODE=1
WANDB_DISABLED=true
VIDUR_DISABLE_WANDB=1
TMPDIR=/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b
VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69
```

The executed suite/check entry commands were:

```sh
timeout 100s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/run_required_tests.py
timeout 270s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/run_importing_tests.py
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/verify_cpu_final.py
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g3h/prepare_package.py --check-plan
```

The required runner executes B12's ten required files plus the new native-gate/ID test file, using `pytest -q -ra -p no:cacheprovider`, a scratch basetemp, and a 75-second subprocess timeout. The importing runner executes the 59 discovered/required modules with those pytest options and a 240-second timeout. Exact pytest argv and module lists are retained in [required_command_final.txt](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/required_command_final.txt), [importing_command_final.txt](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/importing_command_final.txt), and [importing_tests_final.json](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/importing_tests_final.json).

The CPU verifier forces minimal `PYTHONPATH=$PWD`, executes both H800 baseline CLI commands, checks G3h plans and the base manifest, runs four cold imports, compiles 17 Python files, runs `/data/ycfeng/tmp/unused_imports.py` on those files, compares HEAD uniform_topk source and gate-constructor ASTs, verifies the frozen hash/source pin/package absence, and executes `git diff --check`. Exact component commands are retained in [cpu_check_commands_final.txt](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/cpu_check_commands_final.txt).

The H800 CLI arguments were unchanged for both `cuda_event` and `record_function`:

```sh
/data/ycfeng/envs/frontier-py310/bin/python -m frontier.profiling.moe.main --models deepseek-ai/DeepSeek-V4-Flash --device h800 --num_gpus 1 --disable_ray --num_tensor_parallel_workers 1 --expert_parallel_sizes 16 --num_tokens_list 12288 6152 32 16 --max_tokens 16384 --load_distributions balanced --num_samples_per_distribution 1 --routing_runtime_path balanced_topk --yes --profile_method cuda_event --dry_run
/data/ycfeng/envs/frontier-py310/bin/python -m frontier.profiling.moe.main --models deepseek-ai/DeepSeek-V4-Flash --device h800 --num_gpus 1 --disable_ray --num_tensor_parallel_workers 1 --expert_parallel_sizes 16 --num_tokens_list 12288 6152 32 16 --max_tokens 16384 --load_distributions balanced --num_samples_per_distribution 1 --routing_runtime_path balanced_topk --yes --profile_method record_function --dry_run
```

Acceptance results:

| Acceptance | Expected | Observed | Status |
| --- | --- | --- | --- |
| B12b 1 / B12 1: H800 grid | Exact JSON equality with both `moe/h800_moe_local_01/plans` baselines | 4 steps per method; no changed plan fields | PASS |
| B12b 1 / B12 2: class/native algebra | All G3h observed/margin classes and lane permutations match native per-rank counts | Covered by required planner/balanced-routing tests; suite passed | PASS |
| B12b 1 / B12 3: producer routing | Correct representative-rank counts and gathered IDs; gate uses L while routed sample uses M | Required CPU tensor/rank-routing tests passed | PASS |
| B12b 1 / B12 4: G3h plan | Ordered tasks equal `expected_tasks.json`; 13 EP16 and 20 EP32 tasks per method | 4 plans; 33 tasks per method, 66 across both; package not built; zero GPU/platform operations | PASS |
| B12b 2: unchanged paths | Exact HEAD uniform_topk; unchanged other gate constructor calls; non-balanced HEAD rows; int32 for other paths | Source/AST equality passed; all 9 frozen-row comparisons and dtype boundary cases passed | PASS |
| B12b 3: native gate | Model parameter dtype, BF16 input, FP32 output, no bias, native backend | All 5 mocked GateLinear boundary cases passed for L=1,2,3,4,768 | PASS |
| B12b 4 / B12 5: required suite | No failures | 325 passed, 2 skipped in 27.23s; exit 0 | PASS |
| B12b 4 / B12 5: importing suite | No failures in tests importing affected modules | 59 modules; 1,702 passed, 24 skipped in 39.40s; exit 0 | PASS |
| B12b 4: new test count | Report new coverage | 34 cases collected and included in both passing suites | PASS |
| B12b 5 / B12 9: hygiene | Line counts, compile/import audit, cold imports, frozen hash, clean diff whitespace | All 17 files compiled and had no unused imports; four cold imports loaded no torch/vLLM/CUDA modules; `git diff --check` passed | PASS |

Required-suite skips are the two installed-vLLM checks in `test_dsv4_moe_profiling.py`. Importing-suite skips comprise five installed-vLLM checks and 19 retired PD-AF contract tests. Native source-extraction tests ran against the configured source root. These skips do not establish installed-worker or GPU parity.

All B12b-modified production Python modules are below 2,000 lines; the largest is `moe_vllm_kernel.py` at 1,279. The audit also included the pre-existing, B12b-untouched `test_profiling_governance_minimal_red.py` at 2,071 lines; it was not refactored. Complete counts and check receipts are in [cpu_checks_final.json](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/cpu_checks_final.json).

Frozen `balanced_routing.py` remains SHA-256 `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`. The G3h `source_revision` remains `5083d0369a5f9b86c333b30fce78154a3364225f`, as required. The launcher base manifest validated all 530 inputs. The existing 66-row extrapolation remains 759 seconds, approximately 0.211 GPU-hours, against the 2 GPU-hour G3h planning cap; this remains an estimate based on H800 data.

Development failures were diagnosed and corrected before the final passes: the new test initially hit an existing model/operator import-order cycle; its fixture import order was corrected. The old B12 timer fixture selected the withdrawn torch fallback and was updated. Constructor-bypassing legacy fixtures lacked routing-path state; forwarding the prepared sample's optional policy preserved their supported standard/int32 behavior. The new forwarding fixture lacked existing quantization state and was repaired. One concurrent required-suite run exceeded its 75-second timeout; final suites ran sequentially and passed. Earlier failure logs remain under `initial_` and `second_` prefixes. Final logs are [required_tests_final.log](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/required_tests_final.log), [importing_tests_final.log](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/importing_tests_final.log), and [g3h_final_plan_check.log](/data/ycfeng/tmp/codex-implement/15de688d-ce25-43c9-8ec1-fe214a5ce2e1/b12b/g3h_final_plan_check.log).

No CPU acceptance blocker or native-source contradiction remains. GPU execution, zero-class kernel launches, actual worker/image/cuteDSL dispatch, and graph-replay equivalence are NOT_RUN. They are limitations requiring later qualification, rather than successful checks or observed parity.

Claude should inspect the shared GateLinear branch and dtype policy, both measured ID conversion boundaries, prepared-sample forwarding, balanced-only row identity, and the independent frozen HEAD-row oracle. Review G3h's source references and remaining qualifications alongside `native_kernel_notes.md`. G3h files are ignored by Git and need explicit inclusion in the eventual commit.

After review, Claude owns committing B12+B12b together, replacing the preserved source revision with the resulting commit, rerunning the required/importing suites and `prepare_package.py --check-plan`, building the package, qualifying the worker image, and checking actual GPU kernel names against the native census. No package build or final acceptance is claimed.
