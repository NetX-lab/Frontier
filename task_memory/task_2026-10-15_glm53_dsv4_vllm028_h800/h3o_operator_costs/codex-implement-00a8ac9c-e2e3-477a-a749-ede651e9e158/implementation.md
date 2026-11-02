**1. Assignment and outcome**

Implemented B13s within the authorized CPU scope. Final acceptance remains with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Authorized writes were limited to `frontier/profiling/attention/`, the assigned unit tests under `tests/unit/`, and the four named G2h files under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/`. Only seven existing files changed; `expected_rows.json` remained unchanged. No new repository files were created.

The work followed the applicable AGENTS.md, the B13r assignment, and the G2h artifacts. No CLAUDE.md was present at the inspected repository locations. No package was built or submitted, no GPU/network/platform action or dependency installation occurred, and no state-changing Git command was used. The accepted bootstrap, base package, package_01, direct block-operator implementation, and concurrent B15 work were preserved.

HEAD advanced during execution from `cc80896069d8b1b7a5df1966b1568b053698512d` to the B15 commit `5896339ebc2daeb5f7ba8614b90cc1d7cb9a8d8d`. The plan's temporary source pin was refreshed to the latter, and the dry-run and full acceptance suite passed afterward. Claude must repin it to the commit containing B13s before building. Both runtime snapshots used for the negative regression were verified byte-identical to those modules at the final HEAD.

One premise needed correction from pinned-source evidence: dummy steps do not sample. Their existing native closure was retained; no synthetic sampling call was added. There is no remaining CPU implementation blocker.

**2. Changed files and mechanism**

All seven assigned files were initially clean. Concurrent B15 changes and the existing untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/` were left intact.

| Changed file | Behavior and reason |
| --- | --- |
| [dsa_vllm_runner.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsa_vllm_runner.py:312) | `close_step_range` now finishes the active step through the existing `end_step`, then clears it. `_wrap_execute_model` at line 413 retains active state while the existing V4 replay range is open. GLM DSA and eager V4 retain their execute-return lifetime. Execute failures close without saving a step. |
| [dsv4_graph_replay_runner.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/frontier/profiling/attention/dsv4_graph_replay_runner.py:313) | `_wrap_sample_tokens` closes the range in `finally`. Sampling failures clear active state before closure, preventing failed steps from being recorded. Existing scope hooks and nested logits ownership remain unchanged. |
| [test_dsv4_runner_operators.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_runner_operators.py:348) | Added eight cases using the real recorder and timer: five real-step kinds, one dummy case, and two failure cases. Strengthened the package-check assertions at line 219 for job names, tags, separate outputs/configs, and one bootstrap each. |
| [test_dsv4_graph_replay_profiling.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/unit/test_dsv4_graph_replay_profiling.py:448) | Updated assertions/stubs that assumed finalization at execute return. Removed a pre-existing unused `GraphNode` import for the required hygiene check. |
| [prepare_package.py](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/prepare_package.py:27) | Added shared `job_definitions` for dry-run validation and generation. Each job has one phase and one bootstrap. Direct planned rows are compared completely against expected rows, rather than comparing only a coordinate subset. The unchanged P1h builder supplies the accepted snapshot/bootstrap construction. |
| [package_plan.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/package_plan.json:1097) | Selects `package_02`, the two required jobs/tags/output directories, the current temporary revision pin, and the revised GPU-time estimate. Immutable replay, native, image, mount, CUDA compatibility and grid inputs were preserved. |
| [README.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/README.md:24) | Documents separate jobs and launcher commands, lifecycle diagnosis, expected sampling/logits rows, ownership review with pinned call sites, lead acceptance checks, and GPU admission estimates. Pinned references were checked and corrected. |

The chosen mechanism uses the existing open step range as the lifetime boundary. `begin_step` still starts the recorder lifecycle; `end_step` still completes it. V4 graph replay opens that range in `_wrap_step`, so execute return defers completion until `_wrap_sample_tokens` closes it. The native dummy path closes it through the existing `run_dummy_step`. No flag, second lifecycle, or copied runner method was added.

**Diagnosis and step-kind evidence — Acceptance 1 and 3: PASS**

Defect 1 is confirmed. The baseline execute wrapper cleared `state.active` when `execute_model` returned. The real `_WorkerState.timed_call` at current lines 288–304 bypasses annotation when active state is absent. Native sampling therefore ran within the open step range but without sampling/logits owner ranges.

Pinned source root: `/data/ycfeng/tmp/opencode/vllm-v0.28.0/vllm/`. In the following table, `M` means `v1/worker/gpu/model_runner.py`; all native references are relative to that root.

| Recorded kind | Sampling after execute? | Pinned evidence | Range completion |
| --- | --- | --- | --- |
| FULL: 336 steps | Yes | FULL forward at M:1635; execute state at M:1689; return None at M:1703; engine `v1/engine/core.py:602–603` calls sampling | One open at replay runner:289; one close at replay runner:326 through state closure |
| PIECEWISE: 336 | Yes | PIECEWISE forward at M:1661, then the same M:1689/1703 and engine:602–603 path | Same single closure |
| First tail: 296 | Yes | Uses the common completed-forward state/return path at M:1689–1703; engine:602–603 samples after None | Same single closure |
| Multi-tail: 7 | Yes | Same common M:1689–1703 and engine:602–603 path; no tail-specific sampling bypass | Same single closure |
| Prefill: 1 | Yes | Eager forward branch at M:1666, then M:1689/1703 and engine:602–603 | Same single closure |
| Dummy: 13 | No | `v1/worker/gpu_worker.py:1210–1212` calls `_dummy_run`; M:695 executes, M:709–715 consumes state, M:767 gathers hidden rows, M:768 returns without sampling | Existing `dsa_vllm_runner.run_dummy_step:689–692` closes once after the native call |

For real steps, `sample_tokens` calls `self.sample` at M:1749. `sample` gathers hidden rows at M:1339 and calls `model.compute_logits` at M:1340. `postprocess_sampled` follows at M:1809. The unchanged hooks map these calls to sampling, nested logits, and sampling respectively (`dsv4_graph_replay_runner.py:334–344`).

CUDA_EVENT replay cases cannot occur: `dsa_vllm_runner.py:677` explicitly selects `(KERNEL_ONLY,)` for graph replay. No CUDA_EVENT-specific sampling handling was added.

**3. Verification — Acceptance 2, 4, 5, 7, 8 and 9**

Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10). Commands used the required 30 GiB virtual-memory limit, timeouts, disabled bytecode/W&B, and scratch temporary/output directories.

The commands below use these path abbreviations:

```bash
P=/data/ycfeng/envs/frontier-py310/bin/python
S=/data/ycfeng/tmp/codex-implement/00a8ac9c-e2e3-477a-a749-ede651e9e158/b13s
G=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner
ulimit -v 31457280
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export TMPDIR="$S/tmp"
```

Native-order negative regression:

```bash
timeout 60s "$P" "$S/head_pytest.py" tests/unit/test_dsv4_runner_operators.py -k native_order_records_sampling_after_execute_returns -q -p no:cacheprovider --basetemp="$S/handoff_head_pytest" > "$S/handoff_head_native.log"
```

Expected and observed: exit 1, **5 failed, 50 deselected**, 0.96 s. Every real-kind case failed at `test_dsv4_runner_operators.py:444`, `assert events == expected`. HEAD closed `frontier_dsa_step|7` where the expected sampling/logits enter/exit events should have appeared. The scratch import hook loads unchanged HEAD runtime snapshots without modifying the worktree. The new tests pass with the implementation, including exact step-index ownership, nested order, one completion, dummy behavior, failure cleanup, and no scopes outside the recorded window.

Final required suite:

```bash
timeout 600s "$P" -m pytest tests/unit/test_dsv4_*.py tests/unit/test_*graph_replay*.py tests/unit/test_*bridge*.py tests/unit/test_profiling_runtime_boundary.py tests/unit/test_*kernel_gap*.py tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsa_query_one_profiling.py tests/unit/test_dsa_extend_profiling.py tests/unit/test_dsa_full_dummy_profiling.py --deselect=tests/unit/test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default -q -rs -p no:cacheprovider --basetemp="$S/handoff_pytest" > "$S/handoff_tests.log"
```

Observed: exit 0, **1,414 passed, 28 skipped, 1 deselected**, 44.96 s. No B15-area failure occurred. Earlier focused B13r verification passed **153 tests**; the final focused B13r plus four GLM DSA suites passed **316 tests, 1 skipped**. Logs are `fix2_tests.log` and `final_tests.log` in scratch.

Skipped tests were:

- One in `test_dsv4_linear_op_profiling.py:84`: Torch unavailable.
- Seven in `test_dsv4_moe_profiling.py:29,51,67,88,100,144,206`: Torch unavailable.
- Nineteen in `test_sglang_graph_replay_orchestration.py` (18 at line 34, one at line 217): Torch unavailable.
- One in `test_dsa_full_dummy_profiling.py:166`: optional native Git-object validation requires `FRONTIER_VLLM_BS_ROOT`, which was unset. The named pinned source was separately read for this assignment.

The known collective-launch pricing failure was deliberately deselected, as required.

Saved-record verification used `check_saved_head.py` and `check_saved.py` to run the replay rows builder, `dsv4_replay_gaps` reader/reducer, runner-row builder, and ownership builder against both saved runs. `compare_saved.py` compares the resulting serialized bytes:

```bash
timeout 30s "$P" "$S/compare_saved.py"
```

Observed: **eight byte comparisons PASS**—gap diagnostics, replay attention rows, runner rows, and ownership summaries for each reference run. G1h-02 produced 4,945 attention rows and zero runner rows; G2h-01 produced 4,945 attention rows and 2,464 runner rows. Saved files were not rewritten.

| Gap class | G1h-02 µs | G2h-01 expected and observed µs | Samples in each run |
| --- | ---: | ---: | ---: |
| cuda_graph | 0.49773414267747085 | 0.4801221185543367 | 169,232 |
| cuda_graph_piecewise | 0.6249537992538311 | 0.6106290602187813 | 289,730 |
| eager | 0.5446049595929223 | 0.5065229453923222 | 30,243 |
| eager_preparation | 1.2208671118771937 | 1.2117857340253784 | 91,170 |

The raw roots checked were `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw/` and `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_01/download/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/`. Comparison artifacts are under scratch `before/` and `after/`; evidence is in `saved_comparison.log`.

GLM preservation: the named DSA suites above passed. The additional `check_glm.py` comparison verified byte-identical HEAD/current CUDA_EVENT records, annotations and actual rows, plus byte-identical KERNEL_ONLY records and annotations (`glm_check.log`).

Split-job dry-run:

```bash
timeout 120s "$P" "$G/prepare_package.py" --dry_run > "$S/handoff_dry_run.log"
```

Observed: exit 0, exact case IDs matching expected rows, engine counts **686 / 296 / 7**, direct coordinates **33 zero + 164 write per method**, **394 total rows**, import probe PASS, two distinct jobs, one bootstrap each, `package_built: false`, and zero GPU/platform operations. `--check-plan` also passed earlier; it is the same parser action.

Hygiene:

```bash
timeout 30s "$P" /data/ycfeng/tmp/unused_imports.py frontier/profiling/attention/dsa_vllm_runner.py frontier/profiling/attention/dsv4_graph_replay_runner.py tests/unit/test_dsv4_runner_operators.py tests/unit/test_dsv4_graph_replay_profiling.py "$G/prepare_package.py"
timeout 30s git diff --check
timeout 30s git status --short -- "$G/package_01"
```

Observed: all five Python files reported `[]` unused imports; diff check was clean; package_01 status was empty. Separate cold child imports used a MetaPathFinder that rejects `torch`, `vllm`, and `cuda`, followed by a `sys.modules` assertion. Both changed runner entry modules and the builder passed with exit 0. The dry-run additionally cold-imported the existing worker entry paths with heavy dependencies blocked.

| File | Final lines |
| --- | ---: |
| dsa_vllm_runner.py | 957 |
| dsv4_graph_replay_runner.py | 628 |
| test_dsv4_runner_operators.py | 493 |
| test_dsv4_graph_replay_profiling.py | 991 |
| prepare_package.py | 188 |
| README.md | 138 |
| package_plan.json | 1,158 |
| unchanged expected_rows.json | 3,304 |

Every changed Python module is below 2,000 lines. Immutable plan inputs, `expected_rows.json`, and `block_operator_grid.json` were compared against the original HEAD and remained equal. package_02 is absent.

**4. Unowned review — Acceptance 6: PASS, CPU evidence only**

Read all three G2h-01 ownership summaries and only the two permitted unique traces: `g1-e0-0000-full.json` and `g1-e0-1040-prefill.json`. Counts include kernels, memcpy and memset; hold calibration is excluded.

| Case kind | (a) predicted sampling/logits ownership | (b) outside hooks | (c) intentional output copies | Total |
| --- | ---: | ---: | ---: | ---: |
| FULL | 3,624 | 0 | 672 | 4,296 |
| PIECEWISE | 3,624 | 0 | 672 | 4,296 |
| first_tail | 3,184 | 0 | 592 | 3,776 |
| multi_tail | 77 | 0 | 14 | 91 |
| dummy | 0 | 273 | 0 | 273 |
| prefill | 11 | 6 | 2 | 19 |
| Total | 10,520 | 279 | 1,952 | 12,751 |

Group (a) is the predicted ownership after the fix, derived from the pinned call order and selected CPU launch correlations; GPU confirmation remains pending. Names below are shortened to template bases. FULL and PIECEWISE have identical counts shown in their shared column. Dummy has none of these group (a) activities.

| Group (a) name | Grids | FULL / PIECEWISE, each | First tail | Multi-tail | Prefill |
| --- | --- | ---: | ---: | ---: | ---: |
| _gumbel_sample_kernel | [b,127,1] | 336 | 296 | 7 | 1 |
| _get_num_sampled_and_rejected_kernel | [b,1,1] | 336 | 296 | 7 | 1 |
| _post_update_kernel | [b,1,1] | 336 | 296 | 7 | 1 |
| _scatter_gather_elementwise_kernel | [1,1,1] | 336 | 296 | 7 | 1 |
| index_elementwise_kernel | [1,1,1], singleton hidden gather [8,1,1] | 702 | 597 | 14 | 3 |
| reduce_kernel (argmax) | [1,1,1], [2,1,1] | 336 | 296 | 7 | 1 |
| vectorized_elementwise_kernel, FillFunctor<int> | [1,1,1] | 336 | 296 | 7 | 1 |
| vectorized_gather_kernel | [b,2,1] | 306 | 291 | 7 | 0 |
| nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT | [2,66,1] | 192 | 155 | 7 | 1 |
| nvjet_sm90_tst_256x16_64x6_2x1_v_bz_TNT | [2,66,1] | 72 | 69 | 0 | 0 |
| nvjet_sm90_tst_512x24_64x3_2x1_v_bz_TNT | [2,66,1] | 72 | 72 | 0 | 0 |
| Memset (Device), logits GEMM preparation | No grid | 264 | 224 | 7 | 1 |

Here b spans 1–24 for FULL/PIECEWISE and first tail; multi-tail uses 2,3,4,5,7,8; prefill uses 1. Vectorized hidden gathers use b >= 2. The selected traces place the memset and LM-head GEMM under `aten::linear -> aten::matmul -> aten::mm`, supporting logits ownership rather than output-copy classification.

Group (b), left unwrapped for Claude's scope decision:

| Kind / work | Count and grids | Pinned call site |
| --- | --- | --- |
| Prefill request-state _apply_write_kernel | 3, [1,1,1] | M:1422 calls add_requests; M:1008 calls request-state writes; gpu/states.py:122–124 writes total_len, all_token_ids and num_computed_tokens; gpu/buffer_utils.py:189 launches each kernel |
| Prefill request-state contents HtoD | 3, no grid | Same calls; gpu/buffer_utils.py:184–186 copies staged contents |
| Dummy arange, elementwise_kernel_with_index | 26, [1,1,1] | gpu/input_batch.py:128 and 169; make_dummy is called outside preparation hooks at M:1503–1508 |
| Dummy scalar subtraction, CUDAFunctorOnSelf_add<int> | 13, [1,1,1] | gpu/input_batch.py:168, logits_indices |
| Dummy CUB DeviceScanInitKernel / DeviceScanKernel | 13 each, [1,1,1] | torch.cumsum at gpu/input_batch.py:155–157 |
| Dummy direct_copy conversion kernel | 13, [1,1,1] | Same cumsum/output conversion at gpu/input_batch.py:155–157 |
| Dummy FillFunctor kernels | 182 total: 169 vectorized, 13 unrolled | gpu/input_batch.py:130,145–146,148,154,159,162–163,165; dummy block-table zero at gpu/block_table.py:175, called at M:1327; slot-buffer fill at gpu/block_table.py:213 via M:1328–1330 and gpu/pcp_manager.py:637 |
| Dummy hidden-row index / vectorized gather | 8 index kernels [8,1,1]; 5 gathers, one each at [2,2,1], [4,2,1], [8,2,1], [16,2,1], [24,2,1] | M:767 in _dummy_run, after execute and outside sample/sample_tokens |

The 182 dummy fills comprise 143 int, 26 long and 13 bool fills. Int grids are [x,1,1], x = 1,2,3,4,7,14,21,28,42; long uses [1,1,1] and [40,1,1]; bool uses [1,1,1]. Identical Fill signatures aggregate across source sites, so individual per-site counts cannot be established without a dummy trace. No additional trace was read to overcome the two-trace limit.

Group (c) is `Memcpy DtoH (Device -> Pinned)`, with no grid: two per real step and none per dummy, totaling 1,952. These are AsyncOutput copies at `gpu/async_utils.py:140,149`, implemented by `async_copy_to_np` at lines 253–254. They intentionally remain unowned.

Detailed evidence is retained in scratch `unowned_compact.json` and `selected_trace_evidence.json`.

**5. Split jobs and GPU estimate**

The builder plans these separate containers under the required output root:

| Job / phase | Tag | Output directory |
| --- | --- | --- |
| G2H_V4_RUNNER_REPLAY / G2H_RUNNER_REPLAY | v4-h200-g2h-runner-20261029-02-replay | /mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_02/replay |
| G2H_V4_BLOCK_OPERATORS / G2H_BLOCK_OPERATORS | v4-h200-g2h-runner-20261029-02-block | /mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_02/block |

Replay bootstraps once using `bridge_config.json`, then runs the bridge report. Block operators bootstrap once using `block_config.json`. This removes the second overlay bootstrap in the same container that caused G2h-01 path B's installed-base mismatch. The accepted bootstrap itself is unchanged.

After committing and repinning, Claude's commands from the G2h directory are:

```bash
timeout 600s /data/ycfeng/envs/frontier-py310/bin/python prepare_package.py
timeout 7200s bash package_02/launch/launch_glm_producers.sh package_02/jobs/G2H_V4_RUNNER_REPLAY.json
timeout 7200s bash package_02/launch/launch_glm_producers.sh package_02/jobs/G2H_V4_BLOCK_OPERATORS.json
```

These commands were inspected/documented, not executed.

Observed G2h-01 replay time was **3,089 seconds, 0.858 GPU-hours (about 0.86)**. With a conservative 120–360 second annotation allowance, replay is estimated at **0.891–0.958 GPU-hours**. The separate block job includes its own 180–480 second startup, a conservative planned-hold allowance of `394 * (3+20) * 0.020 = 181.24 seconds` (warm-up itself does not hold), and 60–180 seconds of other work/tracing: **0.117–0.234 GPU-hours**.

Total additional estimate: **1.008–1.192 GPU-hours**. Against the 7 GPU-hour G1h–G4h cap and about 1.89 already used, projected usage is **2.898–3.082**, leaving **3.918–4.102 GPU-hours**. Path B startup and GPU behavior remain unmeasured.

**6. Remaining work and review handoff**

Claude should inspect the lifetime changes at `close_step_range`, `_wrap_execute_model`, and `_wrap_sample_tokens`; confirm the native-order fixture's wrapper installation order and real timer use; and review the shared job generator, immutable inputs, two configs/tags/output destinations, and README call-site list.

The CPU acceptance conditions are complete. Package generation, GPU execution, final acceptance, commits, and task-record updates remain with Claude. Repin the source revision to the commit containing the fix before building package_02; the temporary current-HEAD pin does not include uncommitted B13s changes.

For GPU acceptance, rerun the dry-run and the listed regression command from the committed tree, then submit the two jobs separately. Check the bridge verdict, four gap values/counts/hold diagnostics against both references, and the 4,945 attention rows' identity fields and kernel counts. Inspect every engine's ownership summary and runner CSV. Expected new runner totals are preparation 989, write 976, zero 499, sampling 976 and logits 976: **4,416 rows**. Sampling and logits must share each real step's index, with logits nested under sampling. Dummy steps must have no sampling rows. The sampling kernels and LM-head GEMM should leave the unowned list; group (b) and intentional output copies should remain.

Check path B independently for native layout, all planned coordinates, and all 394 rows. Its GPU behavior was deliberately left as B13r implemented it. Any decision to wrap group (b) calls is outside this assignment.

Scratch root: `/data/ycfeng/tmp/codex-implement/00a8ac9c-e2e3-477a-a749-ede651e9e158/b13s/`. Principal handoff evidence: `handoff_head_native.log`, `handoff_tests.log`, `handoff_dry_run.log`, `saved_comparison.log`, `glm_check.log`, `cold_imports.log`, the `before/` and `after/` outputs, and the ownership/selected-trace evidence JSON files.
