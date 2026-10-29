**1. Assignment and outcome**

Implemented A, B, and C in `/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail`. The caller identified the starting revision as clean `d47f2ea5` on `feat/step5-dp-wave-tail`; no pre-existing source edits were encountered. No Git operations were performed.

The allowed write directories were `frontier/config/`, `frontier/events/`, `frontier/execution_time_predictor/`, `frontier/scheduler/replica_scheduler/`, `tests/unit/`, and `tests/integration/`. Changes stayed within the assignment’s named production files, two new production modules, two new test files, and the permitted existing tests and workload runner. Scratch files, logs, caches, and smoke outputs used private `/tmp`.

The implementation follows design section 7 in `task_memory/task_2026-10-15_step5_preview_h800/design_dp_wave_tail.md` from the read-only `step5-preview` worktree. Root AGENTS.md, the F001–F004 review, task requirements/plan/progress, relevant tickets, and the supplied native vLLM queue, busy-loop, and DP dispatch sources were inspected. No closer applicable AGENTS.md or CLAUDE.md was found.

Settled behavior is preserved: explicit PREFILL capacity requires exact VLLM V1, FCFS, PP1, speculation/GDN/prefix caching off, and chunked prefill disabled; grouped KV allocation is allowed. Cadence pacing applies only to profiled PREFILL DP dummy passes. Mixed cohorts retain device-end blocking. PREFILL handoff remains at output pop. The new dispatch barrier uses original forward ordinals rather than stage-rebound provisional identities.

Outcome is **implemented**: assigned edits and checks are finished. This does not claim that the full unit suite is green or that Claude accepted the changes.

**2. Changes and mechanism locations**

Thirteen existing files changed and four files are new:

| File | Change and reason |
| --- | --- |
| [cluster_config.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/config/cluster_config.py:512) | Added the optional PREFILL capacity override with PD mode metadata and omission when unset. |
| [cluster_role_config.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/config/cluster_role_config.py:170) | Propagates that override into the PREFILL role configuration. |
| [execution_time_predictor_config.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/config/execution_time_predictor_config.py:47) | Added `dp_dummy_step_host_profile`, including the default device-blocking behavior in its help text. |
| [dp_dummy_step_host_time.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/execution_time_predictor/dp_dummy_step_host_time.py:28) — **new** | Implements schema, provenance, admission, run-scope, PP1/DP>1, exact PREFILL role, and finite positive cadence validation. DECODE returns `None`; CPU overhead modeling is rejected. |
| [base_execution_time_predictor.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/execution_time_predictor/base_execution_time_predictor.py:112) | Loads the profile under the measured Step forward-agreement requirement. The getter at line 156 returns milliseconds or `None`, including for custom predictors that omit Base initialization. |
| [shared_prediction_model_manager.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/execution_time_predictor/shared_prediction_model_manager.py:147) | Binds the profile’s absolute path and dataset fingerprint into predictor identity. |
| [vllm_v1_dp_dispatch.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/scheduler/replica_scheduler/vllm_v1_dp_dispatch.py:8) — **new** | Owns dispatch records, paced device-pass identities, host deadlines/ticks, unfinished snapshots, and invalid completion checks. |
| [vllm_v1_engine_replica_scheduler.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:76) | Integrates the mixin, PREFILL admission/in-flight skip, real/dummy records, host guard, snapshot consumption, and idle condition. The mixin follows the existing mixins, preserving their tested order. |
| [replica_schedule_event.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/events/replica_schedule_event.py:103) | Emits one schedule event per consumed host tick. |
| [batch_stage_end_event.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/frontier/events/batch_stage_end_event.py:169) | Paced dummy device completion drains its ordinal without ending or resuming a host iteration. Existing stage follow-up events are retained. |
| [frontier_workload.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/tests/integration/step5_pd/frontier_workload.py:97) | Selects capacity 2 for both PREFILL and DECODE and updates the explanatory comment. |
| [test_dp_dummy_step_host_time.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/tests/unit/test_dp_dummy_step_host_time.py:1) — **new** | Covers loader acceptance/rejection, predictor loading/getter behavior, and file fingerprint binding. |
| [test_step5_dp_dummy_pacing.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/tests/unit/test_step5_dp_dummy_pacing.py:1) — **new** | Uses the existing Step PD event harness to cover CLI propagation, queue admission, dispatch ordering, queued drain, snapshots, ticks, mixed cohorts, two waves, handoff/completion, and exact default-path timing. |
| [test_vllm_v1_max_model_len.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/tests/unit/test_vllm_v1_max_model_len.py:111) | Removes PREFILL from the old disaggregated rejection parametrization and tests chunked rejection and whole-prompt capacity 2 acceptance. |
| [test_decode_output_placeholders.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/tests/unit/test_decode_output_placeholders.py:197) | Updates the existing chunked PREFILL rejection assertion to the assigned error prefix. |
| [test_nixl_read_completion.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/tests/unit/test_nixl_read_completion.py:220) | Repairs the confirmed pre-existing fixture defect by setting `_forward_start_host_delay = 0.0`. |
| [test_step5_mrv2_forward.py](/data/ycfeng/Frontier/.worktrees/step5-dp-wave-tail/tests/unit/test_step5_mrv2_forward.py:401) | Supplies the new completion hook on its SimpleNamespace fixture; existing lifecycle and request-credit assertions remain intact. |

Mechanism mapping:

| Item | Implementation |
| --- | --- |
| A1 | Capacity field: `cluster_config.py:512`; role propagation: `cluster_role_config.py:170`. The existing `_OVERRIDE_FIELDS` needed no change. |
| A2 | PREFILL capacity guard: main scheduler lines 415–430. Grouped allocation is accepted; `_uses_output_placeholders` remains false. |
| A3 | PREFILL in-flight skip: main scheduler line 1167. MONOLITHIC-only reservation stays restricted to MONOLITHIC. Grouped admission’s existing capacity × token-budget calculation and output-pop handoff were retained. |
| A4 | Capacity tests: `test_vllm_v1_max_model_len.py:111`; actual CLI reconstruction and grouped whole-prompt admission: new pacing tests lines 114 and 140. |
| A5 | Workload flags/comment: `frontier_workload.py:97–103`. |
| B | Loader: new profile module line 28; option: predictor config line 47; load/getter: Base predictor lines 112/156; identity binding: manager line 147. |
| C1 | Always-initialized opt-in state: mixin line 11, called from main scheduler line 166. Cadence converts from milliseconds to seconds once. |
| C2 | Real records use `batch._forward_cohort_id` at main line 656; dummy records use `forward_index` at line 768. No provisional ordinal is used. |
| C3 | Per-ordinal agreement and latest-issue-plus-cadence deadline: mixin line 27. Only all-dummy cohorts become paced; records are removed after agreement. |
| C4 | Host guard: mixin line 50 and main line 580. Early events do nothing; a ready event consumes the deadline before ending the open iteration. |
| C5 | Tick consumption: mixin line 66; tick events: `replica_schedule_event.py:103`. |
| C6 | Paced device drain/validation: mixin line 72; event interception: `batch_stage_end_event.py:169`; blocking-end validation: main line 772. |
| C7 | Host deadline added to `engine_loop_idle`: main line 746. Outstanding paced device work does not itself make the host busy. |
| C8 | Snapshot capture/consumption: mixin lines 45/60; synchronization uses it at main line 661. Mixed dummy passes also use their issue snapshot. |
| C9 | Existing busy-loop/synchronization logic remains at main line 602. The event test at pacing-test line 264 verifies immediate formation during drain, existing forward-start wakeup, and completion across two waves. No additional wakeup path was added. |
| C10 | `next_pass_start` at main line 781 is source-identical to the saved original method. NIXL implementation, forward-start agreement, and stage timing files were not edited. |

Ordinary implementation refinements were needed for compatibility: appending the new mixin preserves the existing MRO prefix, and the optional getter tolerates custom predictor fixtures without Base initialization. These remain within the settled design and write scope. No material design decision or scope expansion remains unresolved.

**3. Verification**

All pytest commands ran from the working root with `PYTHONPATH=$PWD:$PWD/tests/unit` and `PYTHONDONTWRITEBYTECODE=1`, using `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python` (**Python 3.10.6**). Long runs used `ulimit -v 31457280` and `timeout 1800`. Every pytest invocation used `-q`, its stated private `/tmp/item4_*` basetemp, `-p no:cacheprovider`, and `-p no:logging`; no `-n` was used.

Evidence is available to the host under:

`/data/ycfeng/tmp/codex-implement/b36e6b25-ccb8-46c8-bbc1-271350c4acd3/tmp/item4_evidence/`

This is the host-visible backing directory for private `/tmp/item4_evidence` in the executed commands.

The focused script executed the following exact Python/test arguments:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit
ulimit -v 31457280
timeout 1800 /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -m pytest tests/unit/test_step5_dp_dummy_pacing.py tests/unit/test_dp_dummy_step_host_time.py tests/unit/test_vllm_v1_max_model_len.py tests/unit/test_kv_block_zeroing_host_time.py tests/unit/test_step5_pd_placement.py tests/unit/test_vllm_v1_engine_idle_time.py tests/unit/test_decode_output_placeholders.py tests/unit/test_nixl_read_cost.py tests/unit/test_nixl_read_completion.py tests/unit/test_step5_mrv2_forward.py tests/unit/test_step5_prefill_target.py tests/unit/test_step5_finish_sync.py tests/unit/test_step5_nixl_delegates.py tests/integration/test_vllm_dp_placement_runtime.py tests/integration/test_vllm_dp_dummy_pass_runtime.py tests/integration/test_stage_admission_pipeline_lanes.py tests/integration/test_online_pdd_forward_groups.py -q --basetemp /tmp/item4_codex -p no:cacheprovider -p no:logging
```

`bash /tmp/item4_evidence/run_focused.sh` was executed three times, redirecting to the logs below:

| Log / code state | Actual result |
| --- | --- |
| `focused01.log`, initial fixtures | **FAIL:** 7 failed, 337 passed, 1 skipped in 556.73 s. Six failures were corrected fixture/assertion issues; one was the known integration failure. |
| `focused_final.log`, corrected fixtures | **FAIL only for known baseline:** 1 failed, 347 passed, 1 skipped in 564.97 s. |
| `focused_postcompat.log`, final production code | **FAIL only for known baseline:** 1 failed, 347 passed, 1 skipped in 560.69 s. |

The remaining failure is unchanged:

`tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`

At line 481, lane 2’s first forward is real, `{2: (0, False)}`, while the test expects dummy, `{2: (0, True)}`. This matches Claude’s supplied baseline. Its code and expectations were not edited.

Other executed pytest commands are listed by their exact arguments following the same Python executable and limit/timeout wrapper:

| Python arguments | Basetemp | Result / log |
| --- | --- | --- |
| `-m pytest tests/unit/test_nixl_read_completion.py::test_decode_step_start_posts_and_checks_the_lane_reads` | `/tmp/item4_nixl_baseline` | **Expected FAIL before repair:** 1 failed; missing `_forward_start_host_delay`. `nixl_baseline.log`. |
| `-m pytest tests/unit/test_dp_dummy_step_host_time.py tests/unit/test_step5_dp_dummy_pacing.py tests/unit/test_vllm_v1_max_model_len.py` | `/tmp/item4_new` | **FAIL during development:** 4 failed, 74 passed in 43.05 s. Test import, batching, and event-trace fixture issues were corrected. `new_tests01.log`. |
| Same three files | `/tmp/item4_new02` | **FAIL during development:** 1 failed, 77 passed in 112.89 s. The transfer-start assertion ran before the queued event; it was moved after final drain. `new_tests02.log`. |
| `-m pytest tests/unit/test_step5_dp_dummy_pacing.py` | `/tmp/item4_pacing_final` | **PASS:** 12 passed in 148.01 s; this preceded the four final added cases. `pacing03.log`. |
| `-m pytest tests/unit/test_step5_mrv2_forward.py tests/unit/test_nixl_read_completion.py` | `/tmp/item4_fixture` | **PASS:** 94 passed in 17.26 s. `fixture_final.log`. |
| `-m pytest tests/unit/test_step5_dp_dummy_pacing.py::test_prefill_cli_override_reaches_role_scheduler_config tests/unit/test_step5_dp_dummy_pacing.py::test_duplicate_tick_during_running_wave_counts_only_one_iteration tests/unit/test_step5_dp_dummy_pacing.py::test_mixed_dummy_pass_also_uses_its_issue_snapshot` | `/tmp/item4_pacing_small` | **PASS:** 3 passed in 9.33 s. `pacing_small.log`. |
| `-m pytest tests/unit/test_module_split_boundaries.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py` | `/tmp/item4_compat` | **PASS on final code:** 62 passed in 46.08 s. `compatibility.log`. |

Each table command included the common `-q --basetemp <stated path> -p no:cacheprovider -p no:logging` arguments.

The required unit command was executed with `FRONTIER_TMP_ROOT=/tmp/item4_scratch`:

```bash
PYTHONDONTWRITEBYTECODE=1 FRONTIER_TMP_ROOT=/tmp/item4_scratch PYTHONPATH=$PWD:$PWD/tests/unit bash -c 'ulimit -v 31457280; timeout 1800 /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -m pytest tests/unit -q -x --basetemp /tmp/item4_codex_unit_final -p no:cacheprovider -p no:logging --deselect tests/unit/test_measurement_family_selector.py'
```

**FAIL during collection:** 0 passed, 1 skipped, 1 error in 6.46 s, missing `torch` in `test_mla_native_profiling_wrapper.py`. `unit_final_x.log`. The earlier identical check used `/tmp/item4_codex_unit` and returned 1 skipped, 1 error in 6.61 s (`unit01.log`).

Removing `-x` and changing basetemp to `/tmp/item4_codex_unit_final_all` produced **0 passed, 2 skipped, 38 deselected, 15 collection errors in 17.83 s** (`unit_final_all.log`). The earlier run used `/tmp/item4_codex_unit_all` and produced the same counts in 18.08 s (`unit02.log`).

The collection errors are missing `torch` in these 13 modules: `test_mla_native_profiling_wrapper`, `test_moe_fused_event_contract`, `test_moe_gating_constructor_boundary`, `test_moe_load_distribution_contract`, `test_moe_native_admission`, `test_moe_routing_input_contract`, `test_native_dummy_initialization`, `test_native_full_structure`, `test_native_profiling_model_type_policy`, `test_profiling_timing_stats_contract`, `test_step5_linear_op_activation`, `test_step5_native_position_cyclic`, and `test_vllm_rocm_attention_wrapper_increment9`; missing `matplotlib` in `test_sim_walltime_scaling_plot`; and missing `aiohttp` in `test_step5_canonical_native`.

To inspect runnable tests despite collection errors, the unit command was also executed without `-x`, with `--continue-on-collection-errors` and basetemp `/tmp/item4_codex_unit_continue`. **Raw diagnostic result before final compatibility fixes:** 354 failed, 8,145 passed, 283 skipped, 38 deselected, 41 errors in 1,345.60 s (`unit_continue.log`).

A saved-source importer ran the same diagnostic with basetemp `/tmp/item4_baseline_unit_continue`, ignoring the two new test files: **350 failed, 8,099 passed, 283 skipped, 38 deselected, 41 errors in 1,188.80 s** (`unit_baseline_continue.log`). The comparison contains **389 common failing/error IDs**, six candidate-only failures, and two baseline-only failures. All six candidate-only failures are corrected and covered by final focused/compatibility checks: the MRO-order test, the DECODE fixture’s old PREFILL message assertion, and four MRv2 fixture cases. The GDN test shared an ID but gained a missing-profile-state error; the getter repair addresses it and its final isolated check passes.

Baseline methodology limitation: pytest initially rewrote current test files while the importer substituted original production modules. The two baseline-only failures were the new PREFILL capacity tests exercising old production code, not pre-existing failures. The repaired NIXL fixture was also used in that diagnostic. The importer was corrected to precede pytest’s assertion rewriter. A subsequent original-source fixture run over `test_vllm_v1_max_model_len.py`, `test_nixl_read_completion.py`, `test_decode_output_placeholders.py`, and `test_step5_mrv2_forward.py` returned **1 failed, 131 passed in 18.52 s**, reproducing only the original missing-delay defect (`baseline_fixtures_corrected.log`, basetemp `/tmp/item4_baseline_fixtures_corrected`). The preceding imperfect fixture probe returned 3 failed, 130 passed (`baseline_fixtures.log`, basetemp `/tmp/item4_baseline_fixtures`).

The corrected original-source collection command used `/tmp/item4_evidence/baseline_runner.py tests/unit -q`, basetemp `/tmp/item4_baseline_collection_final`, the same deselection, and `--ignore` for both new files. It reproduced **2 skipped, 38 deselected, 15 collection errors in 18.24 s** (`baseline_collection_final.log`). The first baseline collection probe had 16 errors because its relocated `__file__` broke a `priority_cases.json` lookup; that importer artifact was corrected and is not a repository failure (`unit_baseline.log`, basetemp `/tmp/item4_baseline_unit`).

The complete unrelated unit-failure list, including parameterized node IDs and baseline messages, is in [common_unit_failures.md](/data/ycfeng/tmp/codex-implement/b36e6b25-ccb8-46c8-bbc1-271350c4acd3/tmp/item4_evidence/common_unit_failures.md); paired candidate/baseline messages are in [unit_comparison.json](/data/ycfeng/tmp/codex-implement/b36e6b25-ccb8-46c8-bbc1-271350c4acd3/tmp/item4_evidence/unit_comparison.json). These cover 71 unit modules. Observed causes include missing optional dependencies, an unavailable collective-sim checkout, unavailable pinned reference paths, missing legacy debug scripts, stale operator/metadata contracts, and read-only default metrics paths. Examples of concrete prerequisites are the collective-sim subtree and `outputs/metrics/meta_llama_llama_2_7b_hf`; child-process tests also reported missing `plotly`. The full diagnostic was not rerun after final compatibility repairs, so its raw counts must not be presented as final-code unit acceptance.

The smoke command was executed twice, using `/tmp/item4_smoke` and then `/tmp/item4_smoke_postcompat`:

```bash
bash -c 'ulimit -v 31457280; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 METRICS_OUTPUT_DIR=/tmp/item4_smoke_postcompat PATH=/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin:$PATH; timeout 1800 bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir /tmp/item4_smoke_postcompat/cache'
```

**PASS:** all 12 cases in both runs (`smoke.log`, `smoke_postcompat.log`). The cache override is the only smoke-command adaptation; it keeps generated caches within approved scratch storage.

Additional executed checks:

- The saved original-source trace, run with the assigned Python through `/tmp/item4_evidence/no_profile_baseline.py`, completed successfully. Its exact values are frozen in the final no-profile test: real formation at 0.0 s, three dummy formations at 0.001 s, PREFILL output at 0.36100000000000027 s, DECODE outputs at 0.7222827635200005 and 1.082282763519996 s, and request completion at 1.082282763519996 s. Final focused verification passed exact equality. This is a controlled CPU fixture, not native timing qualification.
- `wc -l frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` returned **1,949**: **PASS**, below 2,000.
- Python 3.10.6 AST parsing of all 17 changed/new files, introduced-whitespace checks, changed production-module size checks, and original/final `next_pass_start` source comparison all **PASS**.
- Final source diff and hashes were written to `source.diff` and `final_source_hashes.json` in the evidence directory. No Git diff command was used.
- Worktree artifact inspection found only an empty `outputs/metrics` directory, left intact. No `cache/` or `.pytest_cache/` directory and no bytecode for either new production module were present. Smoke outputs and caches stayed under private `/tmp`.

The queued-pass tests observed no negative counts or stranded lanes. The real event harness drains forwards in order across two waves, forms the next real request at arrival during drain, wakes peers through the existing agreement path, and completes PREFILL output/handoff and final request execution. Duplicate paced completions with no blocking dummy pass raise before decrementing the counter.

**4. Remaining work and decisions for Claude**

No implementation design blocker remains. Full unit acceptance is blocked by the documented environment/resource and baseline test problems. Providing dependencies, reference resources, or adjusting affected tests’ output configuration belongs to Claude; no dependency installation or extra writable-directory workaround was performed.

The 77-case default fidelity matrix, native DP4 backlog diagnostic, G/E reruns, and measured profile/native qualification were **NOT_RUN** in this assignment. The exact small default-path test and smoke matrix do not substitute for those gates. No native profile asset or estimator was added beyond the assigned loader/consumer work.

Claude retains review, acceptance, commits, and task-record updates. The known late-join integration failure remains for Claude’s separate investigation.

**5. Review handoff**

Review the PREFILL capacity guard and in-flight skip, dispatch ordinal ownership, host deadline/tick handling, snapshot consumption, and paced completion interception. Confirm the existing mixin order and output-pop handoff remain appropriate. The review diff is [source.diff](/data/ycfeng/tmp/codex-implement/b36e6b25-ccb8-46c8-bbc1-271350c4acd3/tmp/item4_evidence/source.diff).

Rerun the focused command, smoke matrix, and 77-case fidelity matrix for acceptance. Rerun the full unit command in an environment that can collect the missing-dependency modules, then assess its baseline failures separately. Final focused evidence is [focused_postcompat.log](/data/ycfeng/tmp/codex-implement/b36e6b25-ccb8-46c8-bbc1-271350c4acd3/tmp/item4_evidence/focused_postcompat.log). The implementation remains scoped to profiled PREFILL PP1 DP>1 pacing and the explicit whole-prompt queue contract; native timing accuracy is not established by these CPU checks.
