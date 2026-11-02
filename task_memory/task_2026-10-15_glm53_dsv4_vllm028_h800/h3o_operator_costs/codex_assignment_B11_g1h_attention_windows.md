# Task: the G1h attention window cases, their package plan and the coverage check (stage 2 assignment B11, file entry 14, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B11 of stage 2 (G1h, [c1_coverage.md](c1_coverage.md) §4.1; [design.md](design.md) §6 entry 14). |

## Goal

G1h measures the V4 attention scopes on 1 H200 through the graph-replay route that P1h qualified (package 03). G1h must measure every attention window class of [c1_coverage.md](c1_coverage.md) §4.1, so that the trained V4 attention predictor (design §6 entry 1, B7) covers the forwards of the H200 P2-32 and P2-64 replays.

B11 delivers, all on CPU:

1. **Case list.** A builder that generates the G1h pilot cases from the C1 §4.1 grid, and the extensions of the pilot case format and the case runner that these cases need.
2. **Prediction per measured step.** For each case, the predicted request state of each measured step: the dispatch mode, the width W, the requests R, the own requests, the first tails, each request's cached tokens, and the requests that write a compressed entry at each ratio.
3. **Coverage check.** A CPU tool that reads the raw replay records of a run, assigns each recorded step to its C1 class and window, compares it with the prediction, and lists the classes and windows that are covered or missing.
4. **Package plan.** A G1h `package_plan.json` and `pilot_cases.json`, built the same way as P1h package 03, with the estimated GPU time.

The lead commits the code, builds the package from the commit and submits G1h through /ds4-exec.

## Approach and decisions

### Existing mechanism (read first)

- **Pilot cases:** `build_pilot_cases` and `_pilot_peer` in `frontier/profiling/attention/dsa_plan.py`. `build_dsv4_profile_plan` in `dsv4_attention_plan.py` loads them. The case fields are `case_id`, `engine_index`, `phase` (`prefill`, `decode`, `dummy`, `first_tail`), `batch_size`, `prompt_len` and `dp_peer` (`num_tokens`, `cg_mode`, `uniform_token_count`, `max_query_len`). The engine fields are `engine_index` and `prefill_query_len`.
- **Case runner:** `run_case` and `_WorkerState.begin_step` in `dsa_vllm_runner.py`.
  - A decode case submits `batch_size` prompts of `prompt_len` tokens and records `measured` steps after `warmup` steps with no prefill rows.
  - A `first_tail` case runs on an engine with `prefill_query_len` = 6143. It first brings `batch_size − 1` requests to decode. Each repeat then adds one request of `prompt_len` tokens, whose last prompt token runs as a one-token first tail.
  - A `dp_peer` sets the synchronized descriptor that the other lane would force (see `_wrap_dispatch` in `dsv4_graph_replay_runner.py`).
- **P1h package 03:**
  - case list: `p1h_bridge/pilot_cases.json` (25 cases);
  - package plan: `p1h_bridge/package_plan.json`;
  - builder: `p1h_bridge/prepare_package.py`;
  - built package: `p1h_bridge/package_03/` (`plans/FULL_AND_PIECEWISE.json` is its plan);
  - driver: `tests/comparison/calibration/dsv4_graph_bridge_check.py`.

  The limits of package 03 are 8 measured and 2 warm-up decode steps, 3 measured and 1 warm-up prefill repeats, a step hold of 200 ms, and a chunk budget of 8,192.
- **P1h package 03 results (H200):**
  - download: `/data/ycfeng/tmp/glm53_dsv4/h200/p1h_bridge_03/download/P1H_BRIDGE/`;
  - raw records: `bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e{0,1}.jsonl`;
  - run logs: `bridge/FULL_AND_PIECEWISE/run_e{0,1}.log` and `bridge/bridge_run.json`.

  Engine 0 took 576 s, of which about 432 s were engine start-up (compile and autotuning), and its 21 cases took about 90 s. Engine 1 took 90 s in all.

  The recorded steps hold each request's `num_computed_tokens`, `num_scheduled_tokens` and `prompt_len`, and the step's `cudagraph_dispatch`. They are the test data for the prediction and for the coverage check.
- **C1 evidence:** `evidence/coverage/{p2_32,p2_64,p0}.json` and `.txt` (the observed decode classes of the three cases, sections A and B), together with `c1_workload.py`, `summarize.py` and `union.py`.

### The grid (C1 §4.1; decisions)

The lane windows below are those of one TP1 reduced engine. W is the dispatched width (`cudagraph_dispatch.num_tokens`), R the requests the attention sees, "own" the lane's real requests and "tails" its first tails.

1. **FULL serving.**
   - Classes: (W, own) for W ∈ {1, 2, 4, 8, 16} with own = 1..W, and W24 with own = 17..24. That is 39 classes, with R = W and padding requests of sequence length 0.
   - When W is above the capture size that `own` dispatches to by itself, a FULL `dp_peer` of `num_tokens` W sets the width (as `e0-decode-b8-p6144-full16`).
2. **PIECEWISE serving.**
   - The same 39 (W, own) classes, once with no tail (a PIECEWISE peer, as `e0-decode-b2-p6144-piecewise4`) and once with one first tail among the own requests (R = own).
   - Seven multi-tail windows at the observed (own, tails): (2, 2), (3, 2), (3, 3), (4, 4), (5, 2), (7, 2) and (8, 3), each at the W it dispatches to.
3. **Context profiles.** Cached tokens of the requests at the measured steps:
   - **low:** all near 6,144;
   - **high:** all near the top of the native range, 7,166, within the engine's `max_context_len` of 7,168;
   - **spread:** evenly over 6,144-7,166.

   Every FULL and PIECEWISE serving class is measured at all three. A first tail has 6,143 cached tokens by definition, so the profile applies to the other requests.
4. **Compression mixes.**
   - Request i writes a ratio-r entry in a step when floor((c_i + q_i) / r) > floor(c_i / r), for r ∈ {4, 128}, with c_i its cached tokens and q_i its query tokens. A ratio-128 write is also a ratio-4 write.
   - For each class and profile, the measured steps must include three mixes, given as (n4, n128), the numbers of requests writing at each ratio:
     - (0, 0);
     - (k4(b), 0);
     - (k128(b), k128(b)).
   - Here b = own. k4(b) and k128(b) are the per-b maxima over the three cases of C1 §3.4, for example b = 4 gives (4, 4) and b = 8 gives (7, 4). For a b that C1 did not observe (above 13), use the largest observed values capped at b: k4 = min(b, 7) and k128 = min(b, 4).
   - The 8 measured steps of a decode case advance every context by one token per step, so a case can cover several mixes when its requests' cached tokens are placed accordingly. Choose the placement, the number of measured steps and the number of cases per class. Fewer cases is better, provided every (class, profile, mix) window is predicted to occur.
5. **Dummies** (as package 03's `dummy` cases):
   - decode FULL dummies with R = W, and PIECEWISE dummies with R = 1 and query W, for W ∈ {1, 2, 4, 8, 16, 24};
   - the prefill FULL W1 dummy, and the NONE W1 dummy on the short-context branch, which is a dummy beside a NONE peer of 6,144 tokens (`e0-dummy-none6144`).

   When two dummies have the same coordinate on the reduced engine, measure it once and say so.
6. **Prefill serving.** NONE W6144: one request of 6,144 query tokens with no cached tokens (`e0-prefill-p6144`).

Every case is measured on every layer of the reduced engine, so each window covers all three layer variants and both planner roles. The 1,024-row multi-stream threshold is crossed only by the prefill case. That is expected; do not add cases for it.

### Format and runner extensions

- **Per-request context.** The case format needs a per-request prompt length (or an equivalent that sets each request's context) to realize the profiles and mixes. A first-tail case needs a tail count and an optional `dp_peer`, because PIECEWISE windows with a tail at a W above the natural width need a peer. `build_pilot_cases` currently rejects a `dp_peer` on a `first_tail` case.
  - Extend `DsaCase`, the pilot keys and their validations.
  - Keep every existing field's meaning. The package 03 `pilot_cases.json` must give the same plan, byte for byte (Acceptance 1).
  - Validate every new field with the existing strict-key and range checks: KV capacity, `max_context_len`, the chunk budget and `max_num_seqs`.
- **Runner.**
  - `run_case` submits the per-request prompts.
  - `begin_step` matches a step to its case using the case's own definition: decode rows only; or the first tails plus continuing decodes, with the given tail count.
  - The step record already carries the request list, so the coverage check needs no new record field. Add a field only if the check cannot be done without it.
- **Multi-tail construction.** Several first tails in one step need their 6,143-token chunks to finish in the same earlier step. With the 8,192-token budget, one chunk of 6,143 fits per step. Find a construction that the native scheduler produces. One option is a separate engine with a larger budget, used for the multi-tail cases only. Show from the native source that the change does not alter the measured step's kernels, or name the kernels it could alter.
- **Prediction.** Predict each measured step's request state from the native scheduling rules: FCFS chunked prefill under the budget, decodes scheduled before new chunks, one token per decode step, `prefill_query_len` capping a chunk on the first-tail engine, and the warm-up and skip counts of `_case_window`. Validate the prediction on the package 03 records: for each of the 25 cases, the predicted and recorded per-request `num_computed_tokens` of every measured step, and the dispatch mode and width. Report any mismatch with the case and step. If the prediction cannot match the records, report the evidence and options. Do not widen the grid to compensate.

### Coverage check

- Add a CLI module under `tests/analysis/`, for example `v4_attention_coverage.py`, with two subcommands:
  - `cases`: writes the predicted windows of a pilot case list;
  - `check`: reads the raw records of a run (`dsv4_raw_e*.jsonl`), assigns every recorded step to (class, profile, mix), compares the steps with the prediction, and lists each window as `covered` or `missing`, plus any recorded step outside the grid.
- The class definitions and the k4 / k128 table live in one place, and the case builder and the checker both import them.
- Read the raw files one record at a time; the engine 0 file is 80 MB.
- Also run `check` over the observed native classes: every decode serving class in the C1 JSON files of the three cases must fall inside the grid. Report any that do not.

### Package plan

- Add a new directory `h3o_operator_costs/g1h/` with:
  - `pilot_cases.json`, generated by the builder;
  - `package_plan.json`, following `p1h_bridge/package_plan.json`, with destination `package_01`, name `G1H_ATTENTION` and output directory `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_01`;
  - a builder that reuses the P1h package mechanism.
- Do not change `p1h_bridge/` or its packages. The G1h builder can import or copy `p1h_bridge/prepare_package.py`.
- The builder requires a committed `source_revision`. Set it to the current HEAD and state in the report that the lead replaces it after committing B11. Do not build the package; the snapshot check refuses uncommitted sources, which is correct.
- **GPU time.** Estimate the job's GPU time from the package 03 timings: engine start-up per engine, and seconds per step including the 200 ms hold. Give the per-engine and total figures against the G1h share of the G1h-G4h cap: C1 estimates 1.5 GPU-hours, and the cap for G1h-G4h is 7. If the estimate is above 3 GPU-hours, propose a reduction (fewer measured steps, or merged cases) with what it gives up, and keep the full grid in the files.

## Out of scope

- Any GPU, platform or network action. Building or submitting the package.
- The V4 attention predictor, the loader and the trainer (B7 and B8 are committed), and the other producers (entries 15-17).
- H800 G1. The same builder must work for H800 later, but no H800 files are written now.
- Changes to the census, the B10 files or the collective benchmark.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Concurrent work:** assignment B10 (`tests/analysis/v4_collective_kernels.py`, `tests/analysis/v4_runtime_census.py`, `tests/unit/test_v4_collective_kernels.py`, `h3o_operator_costs/evidence/collective_kernels/`) may still be uncommitted. Do not edit those files.
- **Task records:** [../requirements.md](../requirements.md), [../plan.md](../plan.md), [design.md](design.md), [c1_coverage.md](c1_coverage.md), [stage1a_replay_route.md](stage1a_replay_route.md), [s1_stream_graph.md](s1_stream_graph.md).
- **Writable paths:**
  - `frontier/profiling/attention/` (the pilot format, the plan, the case runner);
  - `tests/analysis/` (the coverage check; new files only, apart from the B10 files above, which stay untouched);
  - `tests/unit/`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h/` (new).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b11/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No deletions or moves.
  - Python 3.10. Keep modules at or below 2,000 lines (`dsa_plan.py` has 1,058 and `dsa_vllm_runner.py` 930).
  - No magic literals in code paths. The grid values come from C1 and live in one table; the engine values come from the plan arguments. No guards for states the code cannot reach. No second way to do what the pilot format already does.
  - The GLM DSA plans that share `dsa_plan.py` must not change (Acceptance 2).
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
- **Environment:** `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. The pinned vLLM source is `/data/ycfeng/tmp/opencode/vllm-v0.28.0` (revision `2cf0a6915ce544dc493a0990f2ea38d81601128a`), read-only.

## Acceptance

1. **Package 03 unchanged.** Rebuilding the plan from the package 03 plan arguments and `p1h_bridge/pilot_cases.json` (see `prepare_package.py`) gives JSON equal to `p1h_bridge/package_03/plans/FULL_AND_PIECEWISE.json`.
2. **Existing tests.**
   - `tests/unit/test_dsv4_pilot_cases.py`, `test_dsv4_graph_replay_profiling.py`, `test_dsv4_attention_profiling.py`, `test_dsa_attention_profiling.py`, `test_dsa_extend_profiling.py`, `test_dsa_query_one_profiling.py` and `test_dsv4_features.py` pass.
   - Any other test that imports the changed modules passes too; find them with grep.
3. **Prediction on package 03.** For every recorded step of the 25 package 03 cases, the predicted per-request `num_computed_tokens`, the dispatch mode and the width equal the recorded ones. Report the step count and any mismatch.
4. **G1h grid.**
   - `check` on the predicted G1h windows lists every (class, profile, mix) window of the grid as predicted, and no predicted step outside the grid.
   - Every observed native decode serving class of the three C1 cases lies inside the grid.
   - Report the counts of cases, windows and measured steps by category (FULL, PIECEWISE, multi-tail, dummies, prefill).
5. **Coverage check.** `check` on the package 03 raw records runs, and lists the windows it covers and the steps outside the G1h grid (for example `e0-decode-b4-p7150`).
6. **New unit tests** for the format extensions, the multi-tail matching, the prediction (including the rejection of an infeasible case) and the coverage check. Report the count.
7. **GPU time** estimated per engine and in total, as above.
8. **Hygiene:** line counts; no unused imports (`/data/ycfeng/tmp/unused_imports.py`); a cold import of the changed and new modules exits 0 and loads no `torch`, `vllm` or `cuda` module.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- The multi-tail construction and its native-source evidence.
- The results of Acceptance 1-8, with the commands, the expected and actual values, and the counts.
- Scratch paths, deviations and their reasons, and limits.
- If the native source or the package 03 records contradict a decision above (for example, a class that the reduced engine cannot dispatch, or a mix that no placement reaches within `max_context_len`), stop that part. Report the evidence and the options rather than choosing a different grid.
