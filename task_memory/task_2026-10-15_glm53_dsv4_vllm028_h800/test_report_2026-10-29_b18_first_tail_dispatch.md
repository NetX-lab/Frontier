# Test report: B18, first-tail graph dispatch and the A10/A9 tools

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Added check 11, the 77-case fidelity matrix at `b6c28858`. |
| 2026-10-29 | Created: B18 (Codex `d06b0795`, stopped by the lead before Acceptance 4-5) review, one lead correction and lead checks. |

## Scope

Assignment: [h3o_operator_costs/codex_assignment_B18_first_tail_dispatch_and_a10_driver.md](h3o_operator_costs/codex_assignment_B18_first_tail_dispatch_and_a10_driver.md). Codex handoff: `/data/ycfeng/tmp/codex-implement/d06b0795-8e70-483d-a5ff-b9e04a9d1352/b18/report_draft.md` (scratch; its logs are in the same directory).

The lead stopped the Codex run during Acceptance 4 (the full P2-32 replay). The lead's profile showed that the operator path reaches 0.44 s of simulated time per 300 s of wall time, so the 146 s replay needs about a day. Waiting would have blocked the serial Codex queue. Acceptance 4 and 5 move to the lead after B20 (the speed assignment, [h3o_operator_costs/codex_assignment_B20_v4_operator_path_speed.md](h3o_operator_costs/codex_assignment_B20_v4_operator_path_speed.md)).

## Delivered

| Area | Files | Behavior |
| --- | --- | --- |
| Classifier | `frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py` (714 lines) | Under `full_and_piecewise`, `_build_decode_cuda_graph_metadata` takes the per-row decode flags from `dsv4_request_is_decoding(batch)`. That function applies the V2 runner rule: a row is prefilling while its computed tokens are below the prompt length (`gpu/model_runner.py:1077-1099`), and a forward with any prefilling row has no uniform decode count (`worker/utils.py:601-607`). A NIXL first tail on DECODE is therefore a mixed row and dispatches PIECEWISE. `full_decode_only` and `piecewise` keep the request-level flags |
| Test lanes | `tests/unit/test_dsv4_dp_sync.py` (515 lines) | `decode_lane` builds a first tail as the runtime does: prefill complete, all request flags true, a DECODE role and real `KVBlockTableWrites.new_rows`. The 12 `FORWARDS` expectations and the native tuple matches are unchanged |
| Dispatch tests | `tests/unit/test_dsv4_graph_dispatch.py` (new, 109 lines) | 18 cases: the runtime first tail with decodes (fails on `e52ecec4`), the first tail clearing the uniform count with and without a dummy peer, the initial prefill chunk per role, steady decodes in the three modes, and the unchanged other modes |
| A10/A9 tools | `h3o_operator_costs/a10/a10_common.py` (45), `a10_driver.py` (174), `a10_gate.py` (69), `compare_cache_runs.py` (78) | The driver reuses the diagnostic `h4a4` cases and config builder, drops the step-cost override and sets the operator-path fields of decision 3. Its receipt records the resolved tables with hashes and row counts, the A1 facts, the KV admission, the wall times, the failing batch on error and the cache inventory. The gate reuses the existing `run_gate.py normalize` and `gate` commands; its matrix differs from the diagnostic one only in the label and the routing admission. The cache comparison reuses the fidelity comparator and cache classifier |
| Tool tests | `tests/unit/test_dsv4_a10_tools.py` (new, 218 lines after the lead correction) | 10 tests: the operator config and resolved inputs for both cases, the matrix difference, the gate command reuse, the A9 equal and difference controls, hash and lock handling, copied-root rebasing, and the receipt phases |

## Lead review

- The classifier change is the single selection of decision 1. Codex's audit of the other request-flag users (attention features, runner scopes, KV and request accounting, the backend query-width split, the EP wave inputs, the dummy forward) keeps them; I checked the audit list against the code paths named in the assignment and agree.
- **Lead correction.** `test_dsv4_a10_tools.py` builds the P2 configs, and `SimulationConfig` sets the process-wide `sys_arch` and graph mode in `global_vars`. Run before `test_dp_dummy_forward_pricing.py` and `test_dsv4_structural_execution.py`, it made three of their tests fail (a PD predictor needed a cluster config, and the DP dummy synchronization took the PD scheduler path). Codex's regression order ran the tool test after those files, so it did not show. I added the reset fixture that `test_config_owned_contracts.py` already uses (reset before and after each test). With it, the three tests pass in both orders (check 2).
- The A10 driver fields match decision 3 of the assignment. No better existing owner exists for them in the recorded native case configs.

## Checks

Environment: Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6 (no torch, no vLLM); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`; `TMPDIR=FRONTIER_TMP_ROOT=/data/ycfeng/tmp/lead_b18/tmp`; `ulimit -v 31457280` (smoke 62914560). Lead scripts and logs: `/data/ycfeng/tmp/lead_b18/` (`env.sh`, `run_tests.sh`, `run_structural.sh`, `order_check.sh`, `run_smoke.sh`, `logs/`).

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | RED on `e52ecec4` (Codex, `red_e52.log`: `git archive e52ecec4 frontier` into scratch, then the new runtime-state test) | Fails: FULL where PIECEWISE is required | `AssertionError: assert 'FULL' == 'PIECEWISE'`, 1 failed | PASS |
| 2 | Test order (`order_check.sh`): the two affected files alone; after `test_dsv4_dp_sync.py` and `test_dsv4_graph_dispatch.py`; after `test_dsv4_a10_tools.py` | All pass in every order | Before the correction: 49, 94 passed, then 3 failed with 56 passed (`logs/structural_before_fixture.log` holds the same 3 failures in the full group). After: 49, 94, 59 passed | PASS after the correction |
| 3 | Structural group (11 files: DP sync, graph dispatch, A10 tools, DP dummy pricing, V4 structural PD and execution, source execution, attention runtime and workload, features, graph measured; `run_structural.sh`) | All pass | 286 passed, rc 0 | PASS |
| 4 | Consumer group (`test_dsv4_operator_consumer.py`, `test_dsv4_runner_consumer.py`, `test_dsv4_runner_ledger.py`, `test_dsv4_kv_block_table_writes.py`) on the committed tables | All pass | 52 passed in 538 s, 7 pandas mixed-type warnings from the attention reader | PASS |
| 5 | Codex regression of 49 files (`run_regression.py`, list in its handoff: the V4 suites, the scheduler and graph-mode suites that reach `_build_decode_cuda_graph_metadata`, and 10 integration files) | Only the known failure | 1,172 passed, 19 skipped, 1 failed: `test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` (lane 2 `(0, False)` against `(0, True)`), the failure known since B17 | PASS (1 failure outside B18) |
| 6 | Codex additional suites (6 scheduler constructor files; `test_online_pdd_forward_groups.py`) | All pass | 105 and 2 passed | PASS |
| 7 | Dummy smoke matrix (`run_smoke.sh`, `PYTHON_BIN` set, output `/data/ycfeng/tmp/lead_b18/smoke`) | 12 of 12 | 12 of 12, rc 0 | PASS |
| 8 | A9 tool controls (Codex, `a9_equal.log`, `a9_difference.log`) | Equal pair reports EQUAL; injected TTFT change and extra cache file are named | EQUAL with exit 0; exit 1 naming `metrics/request_metrics.csv` and `extra_model_123456abcdef.pkl` | PASS |
| 9 | `git diff --check` and module sizes | Clean; touched modules at or below 2,000 lines | Clean; 714, 515, 109, 218 and the four tools below 200 | PASS |
| 10 | A10 replay start (Codex, `timeout 14400s a10_driver.py p2_32 <scratch>/p2_32`) | Not an acceptance item here; facts recorded | Status INITIALIZED, initialization 497.4 s. A1: both roles non-dummy, no step-cost file beside the inputs, no step-cost table bound. Seven tables, each used by PREFILL and DECODE: attention replay 4,945 rows, kernel gap 4, linear 41, runner block 197, runner replay 2,941, MoE 33, H200 collective summary 384. KV admission per DECODE lane 109,934 blocks / 2,940 = 37.39 against native 37.34 (2,944 blocks): DIFFERS, unchanged from the diagnostic run. Stopped by the lead at about 3 s of simulated time | Recorded |
| 11 | 77-case fidelity matrix (A12) on the committed `b6c28858`: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_fidelity.sh`, which runs `tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-candidate-b6c28858 --label candidate_b6c28858 --jobs 8 --clean-cache --continue-on-failure`, then `compare` against `/data/ycfeng/tmp/glm53_dsv4/h3o_b/entry2/fidelity/candidate_7003cf1d` | 77 of 77 identical, no failure | `run.log`: source and harness `b6c288588ae2`, `dirty=False`, 77 of 77 cases rc 0 with artifacts. `compare.log`: 77 compared, 77 identical, 0 mismatched, 0 failures on either side, 0 missing, 0 differing definitions, 0 predictor cache name differences (`comparison.json` in `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/fidelity/`) | PASS |

## Limits

- Acceptance 4 (P2-32 to FINISHED) and 5 (the gate reading) are not done in B18. They run after B20, with the merged B18 and B20 code.
- The KV admission difference (2,940 against 2,944 blocks per lane) is unchanged and outside this assignment.
- The A9 tool is checked on small runs only; the real cold and warm P2 pair runs after B20.
