# Test report: B20, caches of the repeated pure computations on the V4 operator path

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: B20 (Codex `4bcc70f2`, outcome partial) review, the merge with B19 and the lead checks on main. |

## Scope

Assignment: [h3o_operator_costs/codex_assignment_B20_v4_operator_path_speed.md](h3o_operator_costs/codex_assignment_B20_v4_operator_path_speed.md). Codex record: [h3o_operator_costs/b20/report.md](h3o_operator_costs/b20/report.md). Working tree `.worktrees/b20-v4-speed`, branch `b20-v4-speed`, commit `ebdf3b28` (base: the B18 commit `b6c28858`), merged into `feat/glm53-dsv4-vllm028` as `dd7b8932` on top of B19.

Codex returned "partial" only because the frozen package 03 plan file was absent from its worktree (`test_package03_plan_json_frozen`). That file is committed on main in `c4df84e3`; the test passes on main (check 1).

## Delivered

| Area | File (lines on main) | Behavior |
| --- | --- | --- |
| Balanced EP materializers | `frontier/moe_ep_workload.py` (948) | `lru_cache(maxsize=None, typed=True)` on `materialize_balanced_ep_workload` and `materialize_balanced_rank_classes`. Both are pure functions of four integers with immutable results. Typed keys keep `bool` and NumPy integers off the cached path, so they still reach validation |
| V4 layer facts | `frontier/execution_time_predictor/deepseek_v4.py` (706) | `DeepSeekV4LayerFacts` holds the operator names and the shared-expert contract per layer and the TP size per operator for one V4 serving predictor, attached on first use (`_layer_facts`). Layer-id validation runs on every call, cached or not |
| Attention prediction memo | `frontier/execution_time_predictor/dsv4_predictor.py` (364) | Each `DeepSeekV4AttentionPredictor` stores the fitted model's value per model name and feature tuple. The memo covers only the interpolation branch. The row-group admission check (B19), exact lookup and the finite and sign checks still run on every call |
| Tests | `tests/unit/test_v4_prediction_memo.py` (175) | 350 cases: cached equals uncached materializers, invalid argument types rejected after hits, counted fitted-model calls, memo isolation per instance, model and features, coverage before the memo, layer facts for every layer and two replica configs |
| Records | `h3o_operator_costs/b20/` | Run, check and comparison scripts, `comparison.json`, `final-dp-dispatch.json`, `profile.json`, and `main_merge_comparison.json` (lead, check 4) |

## Lead review

- **Diff.** The production diff matches the assignment. The cached TP size equals `replica.attn_tensor_parallel_size`, which the uncached code read. The memo is keyed by the model name and the full feature tuple and covers only the `model.predict` call. No scheduler, entity, table or configuration field changed.
- **Reach.** The only simulator caller of the two cached materializers is `dsv4_operator_predictor.py`; the other callers are the profiling modules `moe_wrapper.py` and `moe_rank_plan.py`, which no simulation path imports. `deepseek_v4.py` and `dsv4_predictor.py` are V4-only. Non-V4 runs therefore do not reach the changed code, so the A12 77-case matrix (no V4 case) was not rerun.
- **Merge.** Git merged `dsv4_predictor.py` without a conflict: B19's row-group check is followed by B20's memo branch (lines 334-359). The memo test fixture still set `_frontier_dsv4_bounds`, which B19 removed, so one test failed after the merge with `AttributeError: ... no attribute '_frontier_dsv4_workload_group_ranges'`. Lead fix in the merge commit: the fixture declares row groups 1, 2 and 3 with `max_context_len` in [1, 100]. The coverage test now fails on `padded_rows=9` through B19's "outside measured workload groups" message, which matches the test's `outside` pattern.

## Checks on main (`dd7b8932`)

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6, GCC 11.3.0); `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, W&B disabled, BLAS/OMP/MKL threads 1; `TMPDIR` and `FRONTIER_TMP_ROOT` under `/data/ycfeng/tmp/lead_b20/`. Scripts and logs: `/data/ycfeng/tmp/lead_b20/` (`main_checks.sh`, `logs/`).

| # | Check | Command | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Required suite: memo, predictor, A6 tool, coverage (with the frozen plan), attention runtime, operator consumer, structural PD and execution, source execution | `bash /data/ycfeng/tmp/lead_b20/main_checks.sh` (part 1, `ulimit -v 31457280`, timeout 1800 s) | all pass | 673 passed in 288 s | PASS |
| 2 | Order check: A6 tool and predictor tests before the A10 tools, DP pricing and sync, graph dispatch and features | same script, part 2 | all pass | 146 passed in 13 s | PASS |
| 3 | B20 regression selection (19 files) on main | `B20_REPO=$W B20_SCRATCH=/data/ycfeng/tmp/lead_b20/run_checks bash h3o_operator_costs/b20/run_checks.sh` | all pass; the 2 known skips (no pinned native balanced-routing registry, no retained P1h GPU evidence) | 862 passed, 2 skipped in 585 s | PASS |
| 4 | Cold identity of main (B18+B19+B20) against B20's `base-cold` (`b6c28858`), 8-request P2-32 trace | `B20_SCRATCH=/data/ycfeng/tmp/lead_b20 bash h3o_operator_costs/b20/run.sh $W /data/ycfeng/tmp/lead_b20/main-cold`, then `compare_runs.py <B20 scratch>/base-cold /data/ycfeng/tmp/lead_b20/main-cold` | every parsed artifact equal, no cache name difference, same trace | status PASS: 6 artifacts equal (`config.json`, `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `op_precision_metadata.csv`, `request_metrics.csv`, `system_metrics.json`), 1,758 cache entries each with 0 differences, dispatch observer records equal, trace SHA-256 `cec2c473…` on both sides | PASS |
| 5 | Scenario coverage of check 4 (local dispatch observer) | from `main_merge_comparison.json` | both PREFILL lanes, PIECEWISE first tails, multi-request FULL forwards, DP dummies | lanes 0 and 1; 8 first tails, all PIECEWISE; 115 multi-request FULL forwards; 138 dummy forwards (equal on both sides) | PASS |
| 6 | Dummy smoke matrix | `PYTHON_BIN=$PY bash examples/architecture/run_dummy_smoke_matrix.sh` | 12 of 12 | "Completed 12 profiling-independent dummy smoke case(s)" | PASS |
| 7 | Whitespace and sizes | `git diff --check` on the B20 commit; `wc -l` | clean; touched modules at most 2,000 lines | clean; 948, 706, 364, 175 | PASS |

Timing of check 4 (single measurements on a shared 64-CPU host; checks 3 and 6 ran at the same time as the main run):

| Run | Initialization s | `Simulator.run` s | Simulated s | Peak RSS KiB |
| --- | ---: | ---: | ---: | ---: |
| B20 `base-cold` (`b6c28858`) | 510.1 | 1,139.3 | 2.7025 | 997,996 |
| B20 `candidate-final-cold` (Codex) | 520.1 | 170.8 | 2.7025 | 998,380 |
| main `main-cold` (lead) | 588.2 | 215.8 | 2.7025 | 998,240 |

Observed speedup of `Simulator.run`: 6.67 times (Codex, quieter host) and 5.28 times (main, loaded host).

## Codex results kept as reported

- 300 s warm window of P2-32: 2.005 s simulated, so 0.00668 simulated seconds per wall second. Estimated full replays without initialization: P2-32 about 6.1 h, P2-64 about 12.7 h. The P2-64 figure assumes twice the cost per simulated second because the EP rank count doubles; it was not measured.
- Remaining largest costs in the window: the EP wave scheduling path (`prepare_moe_wave_from_inputs` 247.7 s cumulative), `ExecutionTime.__init__` 48.5 s, EP lane batch building 40.6 s, `materialize_layer_ep_workload` 32.7 s, `build_payload_bytes` 30.3 s.
- A separate warm run recorded the final descriptors after `_synchronize_dp_descriptors`; its artifacts equal the cold candidate's (`final-dp-dispatch.json`). Those final snapshots give 114 multi-request FULL forwards. The local dispatch observer of check 5 records before synchronization and gives 115.

## Limits

- The three README assertions in `tests/unit/test_pdd_public_surface_docs.py` fail at the base too (Codex `base-audit.log`); outside B20 and not in the required selection.
- The known DP late-join failure (`test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`) is unchanged and not in the selection.
- The caches are unbounded. The memory growth is shown for this scenario and the 300 s window (about 200 MiB over the cold base), not for the full replays; the A10 runs will record their peak RSS.
- The replay estimates come from one profiled window at the start of P2-32; later phases of the trace can run at a different rate.
