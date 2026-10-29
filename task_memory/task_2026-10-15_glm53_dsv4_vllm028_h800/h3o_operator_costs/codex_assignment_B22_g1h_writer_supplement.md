# Task: plan the G1h supplement for the writer-count extremes, and the table promotion that merges it (stage 4 assignment B22, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created after lead probe c P2-32 failed on a writer count outside the measured G1h groups (requirements item 37, D-G1H-WRITER-EXTREMES). |

## Goal

The A10 replays of P2-32 and P2-64 stop on the first V4 attention request whose features fall outside its measured workload group (B19 joint admission, `frontier/execution_time_predictor/dsv4_predictor.py:319-361`). Lead probe c P2-32 stopped after 3,178 s:

```text
DeepSeekV4CoverageError: V4 coverage missing for operator dsv4_compressor_fused_compress_store,
key ('dsv4_compressor_fused_compress_store', 'csa', 'PIECEWISE', 'first_tail'),
group (..., 16): num_compressed_writes=8 outside [1.0, 7.0]
```

The batch (DECODE, id 1544413) has 11 rows of query 1: one first tail at 6,143 cached tokens and ten decodes at 6,243-7,151 cached tokens, spaced about 100 apart; 8 of the 11 rows complete a ratio-4 group in that step. The G1h grid (C1 §4.1, `frontier/profiling/attention/v4_attention_windows.py`) gives each class only the native observed writer maxima of C1 §3.4 (7 at own 11).

B22 amends the grid by decision D-G1H-WRITER-EXTREMES and prepares, on CPU, everything the lead needs to measure and merge the difference: the supplement package plan, the coverage checks that prove the union of G1h-02 and the supplement covers the amended grid and the replay requests, and the promotion that writes the merged table. No GPU work, no package build.

## Decisions (requirements item 37, D-G1H-WRITER-EXTREMES)

1. **Writer counts.** For each serving class (W, own) of the FULL, PIECEWISE and first-tail categories and each context profile, the compression mixes are (0, 0), (k4, 0) and (k128, k128) as today, plus (own, 0) and (own, own). Every mix is raised to (max(n4, t), max(n128, t)) for t first tails, and duplicate windows are removed in grid order. Apply it in `required_windows` so the case builder and the checker keep sharing one grid definition. The multi-tail windows, dummies and prefill do not change.
   - Lead count with the current module: the amended grid adds 432 windows to the 1,041 (144 FULL, 144 PIECEWISE, 144 first tail; W8 72, W16 216, W24 144). Classes with own ≤ 4 gain none because their observed maxima already equal own.
   - Reason: with one query token per row, any subset of rows can complete a group in one step, so t..own writers are reachable at each ratio; the C1 maxima describe one native trajectory.
2. **Supplement package.** A new planning directory `h3o_operator_costs/g1h_supplement/` measures only the amended-grid windows that G1h-02 did not measure, plus a fixed set of repeat windows:
   - Repeat windows: for each of the three serving categories and each W in C1 `widths`, the window with the largest own of that W, the high profile and the (k4, 0) mix after the tail floor. They are G1h-02 windows measured again; they report the difference between the two packages and are left out of the promoted table.
   - The G1h-02 record stays as it ran: do not regenerate or edit `g1h/pilot_cases.json`, `g1h/coverage_grid.json`, `g1h/gpu_time_estimate.json`, `g1h/package_plan.json` or `g1h/package_02/`. Add one sentence to `g1h/README.md` saying that D-G1H-WRITER-EXTREMES later amended the grid and that `g1h_supplement/` measures the difference.
   - Reuse the G1h machinery (the `g1h/build_cases.py` limits and timing model, the P1h package builder through `g1h/prepare_package.py`'s loading pattern). Do not copy it.
   - Package plan fields follow `g1h/package_plan.json`, with destination `package_01`, name `G1H_SUPPLEMENT`, job tag `v4-h200-g1h-supplement-20261029-01`, output `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_s01`, and `source_revision` left as the current main HEAD. The lead replaces it after the commit.
3. **Merged table.** `h200_tables/promote_tables.py` writes the attention table from the G1h-02 rows and the supplement rows without the repeat windows. The V4 kernel gaps (G4h) stay those of G1h-02 only. Every other promoted table is unchanged.
4. **Admission is unchanged.** Do not change B19's joint admission or any runtime predictor behaviour.

## Work

1. **Grid.** Amend `required_windows` (decision 1). Update `tests/unit/test_v4_attention_coverage.py` and the other tests that count or list the grid.
2. **Supplement cases.** In `g1h_supplement/`, generate `pilot_cases.json`, `coverage_grid.json`, `gpu_time_estimate.json` and `package_plan.json`. The supplement case list is: the amended-grid windows not covered by the G1h-02 raw records (use the existing coverage check on `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw`), plus the repeat windows. Every supplement case must reach its window under the existing FCFS case prediction, as in B11b. If a new window cannot be placed, correct the construction, not the window; if the native scheduler cannot produce it within the native context range, stop that part and report the evidence.
3. **Union coverage check.** Show that the G1h-02 raw records plus the supplement predicted steps cover every window of the amended grid, with zero missing windows and zero prediction mismatches. Extend the existing `tests/analysis/v4_attention_coverage.py` for several plans and raw inputs if it needs it; do not write a second checker.
4. **Replay closure check.** The diagnostic census of the lead (`h3o_operator_costs/a10/coverage_census.py`) records each distinct V4 attention request of a replay (`attention_requests.jsonl.gz`) with the measured group ranges of the current models (`measured_group_ranges.json`), and `check` writes `coverage_misses.json`. Add a check that recomputes the group ranges from the G1h-02 table rows plus the supplement predicted steps, exactly as `train_dsv4_attention_models` computes them, and lists every census request outside them.
   - Compute the ranges with the trainer's own code: move the group-range computation of `train_dsv4_attention_models` into a function of `dsv4_predictor.py` that both use. The trained models must be unchanged (check: the `_frontier_dsv4_workload_group_ranges` of every model trained from the current table is equal before and after).
   - Census roots (still running; read them as they are when you check, twice: at the start and before you finish): `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/census/p2_32` and `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/census/p2_64`. Do not write into them. Copy what you read into your scratch directory and report the record counts.
   - A request that stays outside the union for a reason other than decision 1 (another column, an unmeasured padded_rows, a missing model key): report it with its features and the native rule that makes it reachable or not. Do not extend the grid beyond decision 1; the lead decides.
5. **Promotion.** Extend `promote_tables.py` for decision 3, with the supplement raw directory as an input. Without the supplement input it must reproduce the current tables byte for byte. Test the merge on a small synthetic supplement raw file built from existing G1h-02 records (for example, a few records with new case ids), and check that repeat windows are left out and the gaps are unchanged.
6. **Package difference report.** A script that reads the supplement and G1h-02 rows of the repeat windows and reports, per scope and target (time and gap), the relative difference of the medians. Test it on synthetic rows. The lead runs it after the GPU job.
7. **GPU time.** Estimate the supplement with the existing timing model, per engine and in total, and give the envelope. Compare with the remaining G1h-G4h allowance: the cap is 7 GPU-hours and about 3.93 are used. If the estimate is above 1.5 GPU-hours, propose a reduction with what it gives up, and keep the full list in the files.

## Acceptance

1. **Unchanged plans.** The package 03 plan rebuilt from `p1h_bridge/pilot_cases.json` equals `p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json`; the 64 GLM DSA plans that B11 compared stay identical (B11 Acceptance 1-2).
2. **Amended grid.** Window counts by category before and after, and the number of duplicates removed. The lead's count above (1,041 + 432) holds or the difference is explained.
3. **Supplement plan.** Case and measured-step counts by category and engine; the repeat-window list; every supplement case's predicted measured step is assigned to its window (each request's cached and query tokens and writers at each ratio for one example per category and W).
4. **Union coverage.** Zero missing windows, zero predicted steps outside the grid, zero mismatches, and every observed native decode serving class of the three C1 cases inside (`--c1-json` with `evidence/coverage/{p2_32,p2_64,p0}.json`).
5. **Replay closure.** On the census copies: requests and calls checked per case; misses under the current ranges (expected: those of decision 1, including the probe c batch: csa PIECEWISE first tail, padded rows 16, 8 writers); misses left under the union ranges (expected: none from decision 1). Trained models' group ranges equal before and after the refactor.
6. **Promotion.** Byte-identical reproduction without the supplement; the synthetic merge test passes; the kernel-gap table is unchanged in both.
7. **Tests.** The B11 Acceptance 2 files, `tests/unit/test_v4_attention_coverage.py`, the B19 predictor and coverage tests (`tests/unit/test_dsv4_predictor*.py`, `tests/unit/test_v4_prediction_memo.py`) and the new tests pass; report the counts.
8. **Hygiene.** Line counts of the changed and new modules (each at or below 2,000); no unused imports (`/data/ycfeng/tmp/unused_imports.py`); a cold import of the changed `frontier/profiling` modules loads no `torch`, `vllm` or `cuda` module.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b22-g1h-writers` (Git worktree, branch `b22-g1h-writers` from main). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [../requirements.md](../requirements.md) (items 35 and 37), [c1_coverage.md](c1_coverage.md) §3.4 and §4.1, [g1h/README.md](g1h/README.md), [codex_assignment_B11b_g1h_grid_amendment.md](codex_assignment_B11b_g1h_grid_amendment.md), [codex_assignment_B19_v4_joint_coverage_and_heldout_accuracy.md](codex_assignment_B19_v4_joint_coverage_and_heldout_accuracy.md), [h200_tables/manifest.json](h200_tables/manifest.json).
- **Writable paths:**
  - `frontier/profiling/attention/v4_attention_windows.py`, `frontier/profiling/attention/v4_attention_cases.py`;
  - `frontier/execution_time_predictor/dsv4_predictor.py` (the group-range function only);
  - `tests/analysis/v4_attention_coverage.py`, new analysis scripts under `tests/analysis/`;
  - `tests/unit/test_v4_attention_coverage.py`, the B11 Acceptance 2 test files if they count the grid, and new unit tests;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h_supplement/` (new);
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/h200_tables/promote_tables.py`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h/README.md` (one sentence).
- **Read-only inputs:** the G1h-02 raw records and package under `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/`, the census roots, `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b22/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
  - No GPU, network, job submission, package build or package installs. No deletions or moves of files. Do not write under `data/profiling/`; promotion writes to scratch in tests.
  - Do not stop, signal or read with heavy tools the running simulations under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/`; no recursive `grep` or `find` over `/data/ycfeng/tmp/glm53_dsv4/`.
  - Python 3.10. No magic literals in code paths: grid values come from the C1 table, engine values from the plan arguments. No guards for states the code cannot reach. No second way to do what the code already does.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
- **Environment:** `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary of the change.
- The results of Acceptance 1-8 with the commands, the expected and actual values, and the counts.
- The GPU estimate and, if needed, the reduction proposal.
- Any window that could not be placed and any replay request left outside the union, with the evidence.
- Scratch paths, deviations and their reasons, and limits.
