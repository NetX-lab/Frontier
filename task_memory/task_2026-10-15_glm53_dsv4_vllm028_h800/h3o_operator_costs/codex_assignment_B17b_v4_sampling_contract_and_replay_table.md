# Task: key V4 sampling by the native logits-processing branch and commit the replay table (stage 3 assignment B17b, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created after B17 (`c8c5e593`) and the G2h-04 acceptance. Covers the sampling coordinate (`needs_logits_processing`, b), the two-source replay promotion, the committed `model_runner_replay_kernel_only.csv`, the PREFILL sampling admission and the removal of the test-only table overrides. |

## Goal

B17 prices the four V4 runner operators. Two parts stayed open:

1. `dsv4_sampling` is keyed by b alone, and PREFILL sampling raises `DeepSeekV4CoverageError` (`frontier/execution_time_predictor/dsv4_runtime.py:103-108`).
2. No replay table (preparation and sampling) is committed. The tests read a scratch G2h-02 table through `FRONTIER_DSV4_REPLAY_TABLE` and a cached model root through `FRONTIER_DSV4_TEST_MODEL_ROOT` (`tests/unit/test_dsv4_operator_consumer.py:58-82`).

Native vLLM 0.28 runs one of two sampling branches per batch, and the roles use different ones:

| Item | Evidence |
| --- | --- |
| The branch | `vllm/v1/worker/gpu/sample/sampler.py:74-87` sets `needs_logits_processing` per request (logit bias, which `min_tokens > 0` with stop tokens turns on in `logit_bias.py:89-107`; penalties; bad words; temperature outside {0, 1}; min_p, top_k, top_p). `apply_sampling_params` (`sampler.py:192-199`) skips the FP32 logits copy and every processing kernel when no request of the batch needs it |
| MONOLITHIC and unified DECODE: on | The native harness sends `min_tokens == max_tokens` with `ignore_eos` and temperature 0 (`tests/comparison/dp_placement_pp/vllm_replay.py:507-520`). The PD proxy forwards the original body to D (`task_memory/.../h200_mainline/p2_32_reference/package_01/tests/comparison/calibration/pd_proxy.py:141`). The native decode traces have `_bias_kernel` (`evidence/v4_operators/p2_32_decode_dp0.txt`) |
| PREFILL: off | On the vLLM 0.28.0 branch the proxy sets the P request `max_tokens = 1` and removes `min_tokens` and `min_completion_tokens` (`pd_proxy.py:103-108`). The native P trace has no `_bias_kernel`; its sampling kernels after the logits all-gather are two index kernels, `_gumbel_sample_kernel`, the argmax reduce, the scatter-gather, the int fill, `_get_num_sampled_and_rejected_kernel` and `_post_update_kernel`, plus the row gather before the LM head (`evidence/v4_operators/p2_32_prefill_dp0.txt`, real NONE forward, post section) |
| G2h-04 measures "on" | Accepted ([../test_report_2026-10-29_g2h_runner_04.md](../test_report_2026-10-29_g2h_runner_04.md)): `_bias_kernel` and the FP32 copy in every real step's sampling scope, 11 kernels per scope, 976 rows |
| G2h-02 measures "off" | Accepted ([../test_report_2026-10-29_g2h_runner_02.md](../test_report_2026-10-29_g2h_runner_02.md)); its requests carried no `min_tokens`. The lead's check (`/data/ycfeng/tmp/g2h_accept/sampling_02.log`): no `_bias_kernel` and no FP32 copy on any of the 976 real steps, 9 kernels per scope, the same kernels as the native P forward above (the row gather appears as one of two gather kernels, `vectorized_gather_kernel` in 910 steps and `index_elementwise_kernel` in 66) |

Per-case sampling cost (kernel median + memcpy median) is 1.63 times higher with the branch on (median 0.0506 against 0.0302 ms); preparation differs by 1.7 %, within run-to-run spread.

B17b delivers:

1. The sampling coordinate (`needs_logits_processing`, b) in the registry, the reducer, the loader, the trainer and the runtime.
2. A promotion script that takes the preparation and sampling rows from G2h-04 and the sampling rows from G2h-02, verifies each run's declared branch against its traces, and writes the committed replay table and the manifest.
3. The PREFILL sampling admission (the typed error removed), the `runtime_limitations` text, and tests that use the committed tables without environment overrides.

## Approach and decisions

### Read first

- The B17 assignment, report and test report: [codex_assignment_B17_v4_runner_operators_and_pd.md](codex_assignment_B17_v4_runner_operators_and_pd.md), [codex-implement-9dc571f6-ff2f-414e-9558-5d1cd02f2be2/implementation.md](codex-implement-9dc571f6-ff2f-414e-9558-5d1cd02f2be2/implementation.md), [../test_report_2026-10-29_b17_v4_runner_operators.md](../test_report_2026-10-29_b17_v4_runner_operators.md) (the finding section).
- The two run reports above.
- The B17 code: `frontier/execution_time_predictor/dsv4_runner.py`, `frontier/profiling/model_runner/dsv4_runner_tables.py` (`reduce_replay_rows`, `_replay_records`), `h3o_operator_costs/h200_tables/promote_runner_tables.py`, `dsv4_operator_predictor.py` (`load_dsv4_runner_dataframe`, `train_dsv4_runner_models`, `DeepSeekV4OperatorPredictor.runner`), `dsv4_runtime.py` (`add_runner_scopes`), `frontier/model_architectures.py:704-709`, and the tests `test_dsv4_runner_tables.py`, `test_dsv4_runner_consumer.py`, `test_dsv4_operator_consumer.py`, `test_dsv4_runner_ledger.py`.
- The pinned sampler: `vllm/v1/worker/gpu/sample/sampler.py`, `logit_bias.py`, `vllm/sampling_params.py:656-685`, and `vllm/v1/worker/gpu/model_runner.py:1122-1130, 1339-1353` (sampling b is the number of real requests).

### Inputs (read-only)

- **Replay, branch on (G2h-04):** `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_04/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/`, run tag `v4-h200-g2h-runner-20261029-04-replay`.
- **Replay, branch off (G2h-02):** `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_02/download/replay/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/`, run tag `v4-h200-g2h-runner-20261029-02-replay`.
- Each directory holds `dsv4_raw_eN.jsonl`, `dsv4_raw_eN.runner.csv` and `dsv4_raw_eN.ownership.json` for N = 0, 1, 2 (the e0 JSONL is several hundred MB; stream it). List these directories; do not search under `/data/ycfeng/tmp/glm53_dsv4/` recursively.
- **Block rows (G2h-03):** as in B17; the committed block table must come out with the same SHA-256 (`e0a6dadf...c3139cd`).
- **Descriptor join:** the committed `attention_graph_replay_kernel_only.csv`, unchanged.

### Decisions

1. **The coordinate.**
   - `dsv4_sampling` is keyed by (`needs_logits_processing`, b): the native batch-level branch as 0 or 1, and the number of real requests. Use the native name `needs_logits_processing` for the column.
   - The registry `dsv4_runner.py` holds the role mapping, with a comment citing the evidence above: MONOLITHIC 1, unified DECODE 1, PREFILL 0. The value is a property of the request contract that Frontier simulates (fixed output lengths, enforced natively with `min_tokens == max_tokens`) and of the vLLM 0.28 PD proxy. It is not a model, hardware or tuning value. Roles outside the admitted three never reach sampling (B17 rejects them earlier), so the mapping needs no other entries and no default.
   - The kernel that identifies the "on" branch in the traces (`_bias_kernel`) is a registry constant beside the coordinate, read by the reducer.
2. **Promotion from two runs.**
   - `promote_runner_tables.py` takes the replay run (preparation and sampling rows) and a second run whose sampling rows only are added; each run declares its branch value. Choose plain argument names (for example `--replay-logits-processing` and `--sampling-raw-dir`, `--sampling-run-tag`, `--sampling-logits-processing`).
   - For each declared run, the reducer checks every real step's sampling scope: the identifying kernel is present in every step if the run declares 1, and absent from every step if it declares 0. A mismatch raises and names the run, the case and the step. The two runs must declare different values; preparation comes from the replay run only.
   - Each sampling row carries its branch value, its run tag and its source case. The manifest lists both replay runs with SHA-256 and row counts, and its source run tags name all three sources (block, replay, sampling).
   - The activity count stays constant per coordinate, now (`needs_logits_processing`, b); the two branches have different counts (9 and 11 kernels in the lead's check), which is expected.
   - Remove the G2h-02 "structure fixture" guard and update the docstring: G2h-02 is now the accepted source of the "off" branch.
3. **Commit-ready table.** Run the promotion into `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/` (block from G2h-03, replay from G2h-04 with 1, sampling from G2h-02 with 0) and write `h200_tables/runner_manifest.json`. A second run gives the same SHA-256 for both tables. Report the row counts, the preparation keys, the sampling keys per branch and the per-key cost ranges.
4. **Runtime.** `add_runner_scopes` prices sampling at (`needs_logits_processing` of the batch's role, `len(batch.requests)`). Remove the PREFILL typed error. An unmeasured (branch, b) still raises `DeepSeekV4CoverageError` naming the coordinate.
5. **Loader and trainer.** `load_dsv4_runner_dataframe` and `train_dsv4_runner_models` use the two-column sampling coordinate; exact lookup and duplicate means as in B17. The cache identity covers the committed replay table through the existing A9 path.
6. **Tests without overrides.** Remove `FRONTIER_DSV4_REPLAY_TABLE`, `FRONTIER_DSV4_TEST_MODEL_ROOT` and the skip when the table is absent; the tests use the committed tables. Keep the trained fixture module-scoped so training runs once per module.
7. **`runtime_limitations`.** Replace "sampling is measured under the native min_tokens=max_tokens contract with generation-config EOS stop-token bias only; PREFILL sampling without that contract is unmeasured" with the two measured contracts: MONOLITHIC and unified DECODE requests with `min_tokens` equal to `max_tokens` (logits processing on) and PREFILL requests from the vLLM 0.28 PD proxy without `min_tokens` (off); other sampling parameters are unmeasured. Update the tests that pin the text.

## Out of scope

- Any change to the preparation coordinate, the block table, the attention table, `kernel_gap.csv` or the gap path.
- Per-request sampling parameters in Frontier's request model.
- The `add_requests` staged copies (a known unpriced limit).
- E2E replays, the 77-case fidelity matrix and the A11 cross-check (the lead runs them).
- Any GPU, platform or network action.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `c8c5e593` or later). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md).
- **Writable paths:**
  - `frontier/execution_time_predictor/dsv4_runner.py`, `dsv4_operator_predictor.py`, `dsv4_runtime.py`, `dsv4_training.py`;
  - `frontier/profiling/model_runner/dsv4_runner_tables.py`;
  - `frontier/model_architectures.py` (the V4 `runtime_limitations` text only);
  - `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/model_runner_replay_kernel_only.csv` (new) and `model_runner_block_kernel_only.csv` (rewritten by the same script; it must stay byte-identical);
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_runner_tables.py` and `runner_manifest.json`;
  - `tests/unit/`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b17b/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No moves. Delete only code that this change supersedes, and list each deletion.
  - Python 3.10. Keep every touched module at or below 2,000 lines and report the line counts of touched modules above 1,000 lines.
  - No magic literals in code paths: the branch values, the identifying kernel and the coordinate names come from the registry. No guards or defaults for unreachable states, and no fallback that hides a missing row.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. Stream the raw JSONL.
  - Search with `/usr/bin/grep` and quote `--include='*.py'`.
- **Environment:**
  - Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Run pytest from the repository root.
  - The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Acceptance

1. **Tables.** The promotion writes the block table with the B17 SHA-256 and the replay table: preparation rows from G2h-04 (989 rows, 138 keys), sampling rows from G2h-04 with 1 and from G2h-02 with 0 (976 each), with constant activity counts per key. A second run reproduces both SHA-256 values. A run declared with the wrong branch (G2h-02 as 1, or G2h-04 as 0) raises and names a case and step; this can be a unit test on a small fixture plus one real-input check.
2. **Consumer.** One worked example per branch at b = 8 (coordinate, row medians, cost, activity count, gap), matching the table by hand. `DeepSeekV4CoverageError` for an unmeasured (branch, b). Both training paths give the same predictions, and changing the replay table changes the cache key.
3. **Roles.** Non-dummy PREFILL, unified DECODE and MONOLITHIC predictors (TP8/DP2/EP16, `skip_cpu_overhead_modeling=True`) on the committed tables alone price a prefill forward (PREFILL, branch 0), a first-tail decode forward and a FULL decode forward (DECODE and MONOLITHIC, branch 1), and a dummy forward with no sampling. `DECODE_ATTN` and `DECODE_FFN` still raise.
4. **A1.** On the committed tables alone, MONOLITHIC and PDD non-dummy forwards read no `_dummy_execution_time`, including PREFILL sampling.
5. **A3.** The ledger sentinel's expected table changes only where PREFILL sampling becomes 1 on a real prefill forward of the PREFILL role.
6. **Regression.** The B17 suites (runner tables, runner consumer, operator consumer, runner ledger, block-table writes, structural PD and execution, GLM structural), the dummy smoke matrix (`bash examples/architecture/run_dummy_smoke_matrix.sh`, 12/12), and evidence that non-V4 predictions and metrics are unchanged.
7. **Records.** The `runtime_limitations` text and its tests; the line counts.

## Expected return

- Report language: English.
- Changed, new and deleted paths, each with a summary.
- The results of Acceptance 1-7, with the commands, the expected and actual values, and the counts.
- The SHA-256 and row counts of both tables and the manifest summary.
- The expected effect on H200 steps from the tables: sampling time and gap for a b = 1 PREFILL forward (branch 0) and a b = 8 DECODE FULL forward (branch 1), and the difference against B17's b-only keys.
- Scratch paths, deviations and their reasons, and limits.
- If the code, the native source or the measured traces contradict a decision above, stop that part. Report the evidence and the options rather than choosing a different design. Continue with the parts that do not depend on it.
