# Task: dispatch the D first tail like the V2 runner, and the A10 / A9 run tools (stage 4 assignment B18, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created after the lead's first operator-path probe of P2-32 at `e52ecec4`. Covers the graph-mode classifier fix for the D first tail, the corrected A13 (c) test lanes, the A10 operator-path driver and client-gate reading, and the A9 cold/warm comparison. |

## Goal

Stage 3 is complete (B16-B17b, A12 PASS at `ffcd85e7`). Stage 4 starts with the A10 end-to-end replays of P2-32 and P2-64 on the operator path. The lead's first probe stopped on a Frontier dispatch error, not on a missing measurement:

| Item | Evidence |
| --- | --- |
| Probe | `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/probe/p2_32_a.log` and `p2_32_a/receipt.json`: P2-32 input, the operator-path fields below, HEAD `e52ecec4`, cold cache. Initialization 604 s, then `EXIT 1` on `DeepSeekV4CoverageError` for a `first_tail` row at graph mode `FULL` |
| Table | `attention_graph_replay_kernel_only.csv` has `first_tail` rows only at `PIECEWISE` (1,515 rows), which is the native behavior |
| Native rule | The V2 runner marks a row as prefilling when `num_computed_prefill_tokens < prefill_len` and sets `has_prefill` when any row is (`vllm/v1/worker/gpu/model_runner.py:1077-1099`). `get_uniform_decode_token_count` returns `None` when `has_prefill` (`vllm/v1/worker/utils.py:601-607`), so a forward with a first tail never selects a FULL descriptor (`cudagraph_utils.py`, `_init_candidates`: FULL descriptors carry `uniform_token_count = decode_query_len`). A D request that received its prompt KV through NIXL is added as a new runner row and replays the prompt's last token, so it is prefilling in that forward |
| Frontier rule | `_build_decode_cuda_graph_metadata` (`frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py:70-74`) counts a row as decoding from the request-level `request.is_decoding`. On the DECODE role the NIXL first tail is already prefill-complete in Frontier, so the row counts as a one-token decode, the batch is uniform and dispatches FULL. The V4 workload already uses the native row rule: `dsv4_request_is_decoding(batch)` (`frontier/attention/dsv4_features.py:132-148`) clears the flag for a new row in `batch.kv_block_table_writes` with query 1 and context `num_prefill_tokens - 1`, so the attention phase becomes `first_tail` while the graph mode stays `FULL` |
| Why A13 (c) passed | `decode_lane(..., first_tails=n)` in `tests/unit/test_dsv4_dp_sync.py:165-180` builds a first tail with `request_is_decoding=False`, a state the DECODE role never produces. At runtime the request is prefill-complete, `request_is_decoding` is `True`, and the tail is recognized only through `kv_block_table_writes` |
| Lead check | A scratch worktree (`.worktrees/probe-first-tail-graph`, detached at `e52ecec4`, lead diagnostic only) replaces the classifier's decode list under `full_and_piecewise` with `dsv4_request_is_decoding(batch)`. Probe b on P2-32 passes the point where probe a stopped and was still running when this assignment was submitted (log `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/probe/p2_32_b.log`, receipt `probe/p2_32_b/receipt.json`, read-only). If it has ended when you reach Acceptance 4, read its last lines and its receipt. That edit is not a deliverable; the fix below is |

`full_and_piecewise` is admitted for the `deepseek_v4` profile only (`frontier/config/config.py:634-646`), so the fix changes V4 runs only.

B18 delivers:

1. The classifier fix and its tests.
2. The A10 driver and the client-gate reading of one A10 run.
3. The A9 comparison of a cold-cache and a warm-cache run.

## Approach and decisions

### Read first

- [design.md](design.md) §1 (A1, A9, A10, A13), §3.7 (the DP synchronization and the first-tail mode).
- `frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py:44-172`, `vllm_v1_engine_replica_scheduler.py:440-475` (block-table writes are recorded before the classifier runs) and `:720-735` (dummy forwards), `frontier/scheduler/utils/dp_dummy_forward.py:159-215` (`_synchronize_dp_descriptors`).
- `frontier/attention/dsv4_features.py` (`dsv4_request_is_decoding`, `build_dsv4_features` phase rule), `frontier/execution_time_predictor/deepseek_v4.py:90-110`, `dsv4_runner.py:45-60`.
- `tests/unit/test_dsv4_dp_sync.py` (`decode_lane`, `FORWARDS`, `synchronized_forward`, `native_tuples`) and its fixture `tests/fixtures/dsv4_native_dp_sync_rows.json`.
- The diagnostic tools to reuse: `task_memory/.../h4a_p0_diagnostic/h4a4_driver.py` (`CASES`, `BINDING`, `build_config`, `kv_admission`, `outputs`) and `h4a4_gate.py` (`CASES`, `native_layout`, `matrix`, `main`); `tests/comparison/e2e_gate/run_gate.py`; `tests/e2e/refactor_fidelity/compare.py` (`compare_artifact_directories`, `classify_cache_differences`, `substitutions_for`).
- The lead's scratch probe driver, as an example only: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/probe_driver.py`.
- The pinned vLLM source lines cited above.

### Decisions

1. **The classifier.**
   - Under `full_and_piecewise`, `_build_decode_cuda_graph_metadata` takes the per-row decoding flags from `dsv4_request_is_decoding(batch)`, the one implementation of the V2 row rule. Do not write a second implementation. The other graph modes keep the request-level flag; they model other runners and their results must not change (A12).
   - A forward with a D first tail then has `is_mixed_batch = True`, dispatches PIECEWISE with tokens padded and requests not padded, and makes the DP-synchronized uniform count `None` for every lane of the forward (`_synchronize_dp_descriptors` already reads `is_mixed_batch`).
   - Audit every other V4-path use of the request-level flag that feeds the dispatch, the DP descriptor, `original_decode_batch_size`, `padded_decode_batch_size`, the maximum query length or the routed rows. List each use with its verdict; change only those that disagree with the V2 row rule, and give the native line for each change. The semantic uses that drive KV accounting and request metrics stay request-level (`frontier/attention/deepseek_v4.py:318` records `semantic_is_decoding` for that reason).
   - Dummy forwards keep `_dummy_run(1, uniform_decode=True)` as today.
2. **Tests.**
   - Correct `decode_lane` so a first tail has the DECODE-role runtime state: a prefill-complete request whose prompt KV arrived, `request_is_decoding=True`, query 1, context `num_prefill_tokens - 1`, a new row in `batch.kv_block_table_writes`, and the batch's `_cluster_type` set to DECODE. Build the writes through the existing recorder or the existing data type, not a test-only stand-in with a different shape. The `FORWARDS` expectations (counter row 428 and the others) must hold unchanged against the native tuples.
   - Add classifier tests: the D NIXL first tail alone, with steady decodes, and beside a dummy lane gives PIECEWISE with uniform count `None`; steady decodes stay FULL; a MONOLITHIC first prefill chunk and a PREFILL forward keep their current results; `full_decode_only` and `piecewise` results do not change for the same batches.
   - A test that fails on HEAD `e52ecec4` and passes after the fix: a DECODE batch built the runtime way (as above) with one first tail and steady decodes, through the classifier, the synchronization and `build_dsv4_features`, gives phase `first_tail` at mode PIECEWISE.
3. **The A10 driver** (new, `h3o_operator_costs/a10/a10_driver.py`).
   - Usage `a10_driver.py <case> <output root>` with `p2_32` or `p2_64`. It imports `h4a4_driver` by path and reuses `CASES` (input, digest, `num_blocks`, `max_concurrency`), `build_config`, `kv_admission` and `outputs`; do not copy them.
   - It drops the diagnostic table override (`case["overrides"]`, the step-cost attention input) and the `table` entry, and sets the operator-path fields: `decode_cuda_graph_mode = full_and_piecewise`, `replica_config_network_device = h200_dgx`, `analytical_kv_cache_transfer_config_nixl_read_completion = engine_step`, `cluster_config_decode_replica_scheduler_config_max_concurrent_batches = 2`, and `cudagraph_capture_sizes` read from `tests/fixtures/dsv4_native_dp_sync_rows.json` `capture_sizes`. If one of these values has a better existing owner (a recorded native config of the case), read it from there and say so. The run id names the operator path, not `step_cost`.
   - The receipt (`receipt.json`, written at start, after initialization and at the end) records: label `A10`; case; the source commit and the dirty paths under `frontier/` and `data/`; Python version; input path and digest; the `added_fields` and `override_diff` of `build_config`; the path, SHA-256 and row count of every V4 profiling table the predictors read (attention replay, linear-op, MoE, runner block and replay, kernel gap, network tables), taken from the resolved config and not from a hand list; the A1 facts (`enable_dummy_mode` false, no `dsv4_step_cost.json` beside any resolved input, no step-cost table bound by the predictor); the KV admission of `kv_admission`; initialization and run wall times; the metrics directory; and the cache directory's file names with SHA-256 at the end of the run (lock files listed by name only).
4. **The A10 reading** (new, `h3o_operator_costs/a10/a10_gate.py`).
   - Usage `a10_gate.py <case> <frontier run dir> <output dir>`. It imports `h4a4_gate` by path and reuses `CASES`, `native_layout` and the `run_gate.py` normalize and gate calls.
   - The matrix cell differs from `h4a4_gate.matrix` in two places only: the label states the A10 operator-path reading (not DIAGNOSTIC), and routing is `MATCH` with a reason that cites R0 and A13 (b) (design §3.7: native `balanced_topk` and `determine_expert_map` give the same per-expert and per-EP-rank counts as Frontier at DP2 / EP16 and DP4 / EP32). Reuse `matrix` by parameter if a small change makes that possible; otherwise write the cell once in `a10_gate.py`.
   - It prints the four metrics against native with their absolute and relative errors from the gate output, without changing `run_gate.py`.
5. **The A9 comparison** (new, `h3o_operator_costs/a10/compare_cache_runs.py`).
   - Usage `compare_cache_runs.py <cold run root> <warm run root>`. It compares the two metrics directories with `compare_artifact_directories` (path substitutions from `substitutions_for` or the same mechanism) and the two cache directories' file names with `classify_cache_differences`, and checks that the warm run trained no model (its cache file set equals the cold run's at the end). It exits non-zero on any difference and prints each one.

## Out of scope

- Any change to tables, predictors, the gap path or `run_gate.py`.
- The A10 runs themselves beyond the checks below, the 77-case fidelity rerun, A6, A7, A8, A11 (the lead runs or assigns them).
- P0 and H800.
- Any GPU, platform or network action.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `e52ecec4` or later). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md).
- **Writable paths:**
  - `frontier/scheduler/replica_scheduler/vllm_v1_iteration_policy.py`;
  - other `frontier/` modules only where the audit of decision 1 finds a disagreement (list them); `frontier/execution_time_predictor/dsv4_predictor.py` is not among them (B19 owns it in parallel);
  - `tests/unit/test_dsv4_dp_sync.py`, a new classifier test file (for example `tests/unit/test_dsv4_graph_dispatch.py`), a new test file for the `a10/` tools, and the existing test files whose expectations the fix changes (list each with the reason). B19 owns `tests/unit/test_dsv4_predictor.py`, `tests/unit/test_v4_attention_coverage.py` and `tests/unit/test_v4_attention_heldout.py`; do not edit them;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/` (new).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b18/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No moves. Delete only code that this change supersedes, and list each deletion.
  - Python 3.10. Keep every touched module at or below 2,000 lines and report the line counts of touched modules above 1,000 lines.
  - No magic literals in code paths; no guards or defaults for unreachable states; no fallback that hides a missing row.
  - Memory: `ulimit -v 31457280` for tests, `ulimit -v 62914560` for a simulator run, and a `timeout` on every command. A P2 run initializes for about 10 minutes on a cold cache and about 1 minute on a warm one.
  - Do not search under `/data/ycfeng/tmp/glm53_dsv4/` recursively; list the directories named here.
  - Search with `/usr/bin/grep` and quote `--include='*.py'`.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. Run pytest from the repository root.
  - The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Acceptance

1. **Fix.** The new runtime-state test fails on `e52ecec4` (show the failure) and passes after the fix. The classifier tests of decision 2 pass. The audit list of decision 1 with verdicts.
2. **A13 (c).** `test_dsv4_dp_sync.py` passes with the corrected `decode_lane`; every `FORWARDS` expectation and native-tuple match is unchanged.
3. **Regression.** The suites `test_dsv4_dp_sync.py`, `test_dp_dummy_forward_pricing.py`, `test_dsv4_structural_pd.py`, `test_dsv4_structural_execution.py`, `test_dsv4_source_execution.py`, `test_dsv4_attention_runtime.py`, `test_dsv4_attention_workload.py`, `test_dsv4_features.py`, `test_dsv4_operator_consumer.py`, `test_dsv4_runner_consumer.py`, `test_dsv4_runner_ledger.py`, `test_dsv4_kv_block_table_writes.py`, `test_dsv4_graph_measured.py`, and the other scheduler and graph-mode suites that reach `_build_decode_cuda_graph_metadata` (find them, including `tests/integration/`). The dummy smoke matrix (`PYTHON_BIN=/data/ycfeng/envs/frontier-py310/bin/python bash examples/architecture/run_dummy_smoke_matrix.sh`, 12/12). The known HEAD failure `test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` may stay; report any other failure.
4. **Driver.** `a10_driver.py p2_32 <scratch>` with the fix runs to completion (status `FINISHED`) and writes the receipt of decision 3; report its A1 facts, table list, KV admission and wall times. If another typed coverage error or runtime error stops the run, stop this part: report the operator, the coordinate, the batch and the native evidence for it, and do not change tables or predictors.
5. **Reading.** `a10_gate.py p2_32 <that run> <scratch>` gives the four metrics with errors; report them as observed values, not as a pass or fail.
6. **A9 tool.** `compare_cache_runs.py` on two small runs (or two copies of one run, with one deliberate difference injected into a copy in scratch) reports the equal case as equal and the injected difference by name. The lead runs the real cold and warm P2 runs.
7. **Records.** Line counts; the deleted code, if any.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary; the audit list.
- The results of Acceptance 1-7, with the commands, the expected and actual values, and the counts.
- The A10 receipt summary and the four metrics of the P2-32 run.
- Scratch paths, deviations and their reasons, and limits.
- If the code, the native source or a run contradicts a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Continue with the parts that do not depend on it.
