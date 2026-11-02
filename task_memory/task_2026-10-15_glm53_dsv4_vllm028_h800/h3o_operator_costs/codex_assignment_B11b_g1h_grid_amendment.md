# Task: complete the G1h attention grid with the lead's amendment, and fix the B11 review findings (stage 2 assignment B11b, file entry 14, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: follow-up of B11 (`9abfeda9`) after the lead's review and decision D-G1H-GRID-AMENDMENT. |

## Goal

B11 (request `9abfeda9-9f73-4dec-9cb7-ddb151fbbdac`, report `codex-implement-9abfeda9-9f73-4dec-9cb7-ddb151fbbdac/implementation.md`) built the G1h case builder, the case prediction, the coverage check and the G1h planning files. Its full-grid check fails for two reasons it reported correctly. The lead has decided both (below). B11b applies the decisions, makes every window of the amended grid executable, fixes the review findings, and regenerates the G1h files.

The B11 changes are still uncommitted in the working tree. B11b edits them in place. The lead commits B11 and B11b together after review.

## Decisions (requirements D-G1H-GRID-AMENDMENT; [c1_coverage.md](c1_coverage.md) §4.1 amendment)

1. **First-tail mixes.** A first tail has 6,143 cached tokens and query 1, so it writes at ratio 4 and at ratio 128 in its step. With t first tails, each mix target (n4, n128) becomes (max(n4, t), max(n128, t)). Remove duplicate windows after this change, keeping the first one in grid order.
   - For one tail: (0, 0) → (1, 1); (k4, 0) → (k4, 1); (k128, k128) is unchanged.
   - The multi-tail windows already use (tails, tails) and do not change.
   - Apply the rule where the windows are defined (`required_windows` in `v4_attention_windows.py`), so the builder and the checker share it. The check in `_contexts` that rejects a mix below the tail count then guards a state the grid cannot produce. Remove it.
2. **"Near" in the context profiles.** A continuing request is near a profile target when its cached tokens are within one ratio-128 group of the target, that is, within `max(ratios) − 1` = 127 tokens: low 6,144-6,271, high 7,039-7,166. Continuing requests stay inside `cached_low`..`cached_high` (6,144-7,166). Cached 6,143 belongs only to first tails.
   - The allowance is part of the grid definition. Define it once in the C1 table of `v4_attention_windows.py`, derived from the ratios, with a one-line comment that names the decision.
   - Remove the other ways of setting it: the `context_tolerance` parameter of `build_attention_cases`, `check_steps` and `assign_windows`; the `--context-tolerance` options of `build_cases.py` and `tests/analysis/v4_attention_coverage.py`; and `coverage_context_tolerance` in `g1h/package_plan.json`. Tests that need a different allowance monkeypatch the table.
3. **Measured steps.** Keep 0 warm-up steps and 1 measured step per case (B11's choice, accepted).

## Residual placements

The lead applied decisions 1 and 2 in a scratch probe ([/data/ycfeng/tmp/b11_review/probe_amended.py](/data/ycfeng/tmp/b11_review/probe_amended.py), run from the working root with the B11 environment and argument `127`). Of 1,041 windows, 1,036 get a case. Five first-tail windows fail in `_place`:

| W | own | profile | mix | B11 error |
| --- | --- | --- | --- | --- |
| 8 | 6 | low | (5, 1) | `FCFS placement did not converge after 32 predictions` |
| 16 | 6 | low | (5, 1) | same |
| 24 | 18 | high | (1, 1) | `infeasible pilot: recorded 0 of 1 predicted steps` |
| 24 | 18 | high | (7, 1) | same |
| 24 | 18 | high | (4, 4) | same |

The lead has not diagnosed them. Possible causes, to be checked, not assumed:

- W24 own 18 high: the first guess `shift = ceil(sum / chunk_budget) + 2` may not count the two tail steps or the 6,143-token chunk cap of engine 1. Early requests could then reach `max_model_len − 1` and stop before the measured step, and `predict_case_requests` raises before the iteration can correct the lengths.
- W8 and W16 own 6 low: the fixed-point update may oscillate, for example against the last-step remainder that `_align_context_sum` handles for decode cases only.

Make the placement reach these windows by correcting the construction (the first guess, the update, or the alignment), not by changing the windows, the allowance or the engine limits. Keep the prediction the single source of truth: a case is accepted only when its predicted measured step is assigned to its window. If a window cannot occur under the native scheduler within the native context range, stop that part and report the evidence (the predicted request states and the native source rule) and the options. Do not choose a different grid.

## Review findings to fix

- **R1. Function-local imports of CPU modules.**
  - `dsa_vllm_runner.py` (`_WorkerState` step matching) imports `requests_match_case` inside the method.
  - `v4_attention_cases.py` imports `build_pilot_cases` and `predict_case` inside the window loop.
  - `dsa_case_prediction.py` imports only `dsa_plan`, so no import cycle requires this. Move both to the module tops. The runner's function-local imports of `vllm` stay; they keep GPU packages out of the CPU import.
- **R2. String literals for existing constants.** `v4_attention_cases.py` writes `"PIECEWISE"`, `"FULL"`, `"NONE"`, `"dummy"`, `"prefill"`, `"decode"` and `"first_tail"`, and `v4_attention_windows.py` compares `step["phase"]` with `"dummy"` and `"prefill"`. Use the `DP_*` and `PHASE_*` constants of `dsa_plan.py`. Category labels such as `"FULL"` used as grid category names may stay if they are defined once.
- **R3. Hard-coded message values.** The `_contexts` placement error states "ratio-128 cached boundaries in the native range are 6271..7039". Derive any values in a message from the table, or drop them.
- **R4. Stale blocker text.** `build_cases.py` writes fixed `execution_blockers` strings, and the grid manifest writes `context_tolerance_status: provisional`. Remove both. The manifest records the allowance from the table and names decision D-G1H-GRID-AMENDMENT.
- **R5. README.** Rewrite `g1h/README.md` for the amended, complete grid: counts by category, the two decisions with their reasons, the placement corrections, the GPU estimate and the reproduction commands. Keep the multi-tail construction and its native-source evidence. Keep it in plain English.

## Acceptance

1. **Package 03 unchanged.** As B11 Acceptance 1: the plan rebuilt from the package 03 arguments and `p1h_bridge/pilot_cases.json` equals `p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json`.
2. **GLM plans unchanged.** The 64 GLM DSA plans that B11 compared stay identical.
3. **Prediction on package 03.** All 175 measured steps of the 25 package 03 cases match, as in B11 Acceptance 3.
4. **Complete grid.**
   - `build_cases.py` regenerates `pilot_cases.json`, `coverage_grid.json`, `gpu_time_estimate.json` and `package_plan.json` with `complete: true`.
   - `v4_attention_coverage.py cases` then `check --predicted --require-complete-grid` exits 0, with zero missing windows, zero predicted steps outside the grid and zero mismatches.
   - Every observed native decode serving class of the three C1 cases (`--c1-json` with `evidence/coverage/{p2_32,p2_64,p0}.json`) lies inside the grid.
   - Report the counts of requested windows, cases, shared cases and measured steps by category (FULL, PIECEWISE, first tail, multi-tail, dummies, prefill), and the number of duplicate windows removed by decision 1.
5. **Coverage check on package 03 raw records** runs as in B11 Acceptance 5. Report its covered windows and the steps outside the grid.
6. **Tests.**
   - The B11 Acceptance 2 test files and `tests/unit/test_v4_attention_coverage.py` pass. Update the tests for the amended grid and the fixed allowance.
   - Add tests for: the mix amendment and its duplicate removal; the allowance boundaries (a continuing context at target ± 127 is near, at ± 128 is not; a continuing context at 6,143 is outside every profile); and each of the five residual windows reaching its window.
   - Report the test count.
7. **GPU time.** Recompute the estimate per engine and in total from the P1h timing inputs. Compare it with the C1 G1h share (1.5 GPU-hours) and the G1h-G4h cap (7). If it is above 3 GPU-hours, propose a reduction with what it gives up, and keep the full grid in the files.
8. **Hygiene.** Line counts of the changed and new modules (each at or below 2,000); no unused imports (`/data/ycfeng/tmp/unused_imports.py`); a cold import of the changed and new modules exits 0 and loads no `torch`, `vllm` or `cuda` module.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree, HEAD `eaee3550`). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Concurrent uncommitted work:** B12 (`68caa2f4`) changed `frontier/moe_ep_workload.py`, `frontier/profiling/moe/`, several `tests/unit/test_*moe*` and routing tests, and `h3o_operator_costs/g3h/`. Do not edit those files.
- **Task records:** [../requirements.md](../requirements.md) (item 35, D-G1H-GRID-AMENDMENT), [c1_coverage.md](c1_coverage.md) §4.1, [codex_assignment_B11_g1h_attention_windows.md](codex_assignment_B11_g1h_attention_windows.md) (the B11 grid, format and acceptance, still valid except where this file changes them).
- **Writable paths:**
  - `frontier/profiling/attention/` (`dsa_plan.py`, `dsa_vllm_runner.py`, `dsa_case_prediction.py`, `v4_attention_windows.py`, `v4_attention_cases.py`);
  - `tests/analysis/v4_attention_coverage.py`;
  - `tests/unit/test_v4_attention_coverage.py` and the B11 Acceptance 2 test files if they need updates;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h/`.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b11b/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - Leave `source_revision` in `g1h/package_plan.json` at its B11 value; the lead replaces it after the commit. Do not build the package.
  - No GPU, network, job submission or package installs. No deletions or moves of files.
  - Python 3.10. No magic literals in code paths: grid values come from the C1 table, engine values from the plan arguments. No guards for states the code cannot reach. No second way to do what the code already does.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
- **Environment:** `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Expected return

- Report language: English.
- Changed paths, each with a summary of the change.
- For each residual window: the cause found, the correction, and the predicted measured step (each request's cached and query tokens, and its writers at each ratio).
- The results of Acceptance 1-8, with the commands, the expected and actual values, and the counts.
- Scratch paths, deviations and their reasons, and limits.
