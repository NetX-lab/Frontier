# Task: record the V4 sampling scopes after `execute_model` returns, and run the two G2h paths as separate jobs (stage 2 assignment B13s, a fix of B13r found by the G2h-01 run, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created after the G2h-01 run (`exp-1029-113750-923871`): path A ran, but recorded no sampling scope; path B failed in its bootstrap. |

## Goal

The G2h-01 job ran B13r's package `g2h_runner/package_01` on one H200. Local copy: `/data/ycfeng/tmp/glm53_dsv4/h200/g2h_runner_01/download/`.

| Path | Observed | Status |
| --- | --- | --- |
| A, `G2H_RUNNER_REPLAY` | Bridge verdict PASS. The four kernel gaps match G1h-02: same sample counts, values 0.7-7.0 % lower. All 4,945 attention rows have the same identity fields and kernel counts as G1h-02; per-operator median ratios are 0.976-1.009. | accepted |
| A, step scopes | `dsv4_raw_e{0,1,2}.runner.csv` has only `dsv4_input_preparation` (989 rows), `dsv4_block_table_write` (976) and `kv_cache_block_zero` (499). It has **no `dsv4_sampling` and no `dsv4_compute_logits` row**. The traces have no `…|-1|dsv4_sampling` or `…|-1|dsv4_compute_logits` annotation, for example `traces/g1-e0-0000-full.json` and `traces/g1-e0-1040-prefill.json`. The sampler kernels (`_gumbel_sample_kernel`, the argmax reduce, `_post_update_kernel`, `_get_num_sampled_and_rejected_kernel`) and the LM-head GEMM (`nvjet_sm90_tst_*` grid [2, 66, 1]) are unowned in `dsv4_raw_e0.ownership.json`. | defect 1 |
| B, `G2H_BLOCK_OPERATORS` | Failed after 1.6 s: `ValueError: Installed base mismatch: models/deepseek_v4/nvidia/model.py`, raised at `native_dummy_bootstrap.py:54` (`G2H_BLOCK_OPERATORS/logs/block_operators.log`). | defect 2 |

**Defect 1, sampling scopes (lead diagnosis; confirm or correct it).**

- `dsa_vllm_runner._wrap_execute_model` (`frontier/profiling/attention/dsa_vllm_runner.py:406-429`) sets `state.active` for the call to `execute_model` only. It resets the value to `None` in `finally`.
- The pinned V2 runner returns `None` from `execute_model` and samples later. The engine then calls `sample_tokens` (`vllm/v1/worker/gpu/model_runner.py:1707`), which calls `self.sample(...)` (1749; `model.compute_logits` at 1340 inside `sample`) and `self.postprocess_sampled(...)` (1809).
- `_WorkerState.timed_call` (`dsa_vllm_runner.py:272-304`) runs the function without an annotation when `active is None`. So the step scopes wrapped in `_RUNNER_STEP_CALLS` (`dsv4_graph_replay_runner.py:325-339`) record nothing inside `sample_tokens`.
- The step range itself is right: `_wrap_step` opens it, and `_wrap_sample_tokens` (313-322) closes it after `sample_tokens`.
- The unit tests did not see this. `tests/unit/test_dsv4_runner_operators.py:40-80` replaces `state.timed_call` with a stand-in that always annotates, and calls `runner.sample()` directly. No test runs `execute_model` and then `sample_tokens` through the real `_WorkerState`.

**Defect 2, bootstrap run twice in one container.**

- `native_dummy_bootstrap.py` is the accepted bootstrap, copied unchanged from the base package by `p1h_bridge/prepare_package.py:27-29`. It requires the installed `model.py` to have `base_sha256`, then writes the overlay.
- G2h-01 ran this bootstrap twice in one container, once per phase. The first phase had already written the overlay, so the second check failed.
- G2h is the first job that runs the bootstrap twice; every earlier job ran it once.

## Decisions

1. **Keep the step active through `sample_tokens` (V4 graph replay).**
   - Every step scope wrapped by `_wrap_runner_step_scopes` that runs inside `sample_tokens` is recorded with the step's index:
     - `sample` → `dsv4_sampling`;
     - `model.compute_logits` → `dsv4_compute_logits`, nested as B13r decided;
     - `postprocess_sampled` → `dsv4_sampling`.
     The annotations lie inside the open step range, as the input-preparation scopes do.
   - Choose the mechanism yourself and state it. Constraints:
     - the GLM DSA runner path keeps its behaviour, and its records and rows stay byte-identical;
     - no new flag or option;
     - no second step lifecycle next to `begin_step` and `end_step`.
   - **CUDA_EVENT handling.** `end_step` (`dsa_vllm_runner.py:244-270`) collects the CUDA-event calls when `execute_model` returns, before sampling runs.
     - If a graph-replay plan can contain CUDA_EVENT cases, the sampling events must reach the same step's calls.
     - If such cases cannot exist, show the line that rules them out. Then add no handling for them.
   - Check the pinned source for every recorded step kind: FULL, PIECEWISE, first tail, multi-tail, dummy and prefill.
     - Does `sample_tokens` follow `execute_model`?
     - Does a dummy step sample?
     Report file:line for each answer. The step range must still close exactly once per recorded step.
2. **Test the native call order.** Add a unit test with stand-in runner objects that runs the real wrappers in the native order:
   - the case recorder's `execute_model` wrapper outside `_wrap_step`;
   - `execute_model` returns `None` and sets `execute_model_state`;
   - then `sample_tokens` calls `self.sample` → `self.model.compute_logits` → `self.postprocess_sampled`.
   Use the real `_WorkerState.timed_call` with a stand-in `torch.profiler.record_function` that records the names entered and exited.

   Assert:
   - the order is: step range open, hold, input-preparation scopes, `dsv4_sampling`, `dsv4_compute_logits` inside it, `dsv4_sampling` again for `postprocess_sampled`, step range close;
   - every scope name carries the recorded step index.

   Show that this test fails on the current HEAD and passes after the change.
3. **Unowned review on G2h-01 (CPU, evidence only).**
   - Read `download/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e{0,1,2}.ownership.json`. These files are small.
   - Read at most the two traces named above.
   - For each case kind, split the unowned names and grids into three groups:
     - (a) kernels that the fixed sampling scopes will own;
     - (b) kernels of the four B13r operators that run outside every wrapped call, for example the three `_apply_write_kernel` [1, 1, 1] calls in the prefill step, and the `arange`/`Fill` kernels of the dummy steps;
     - (c) work that stays unowned on purpose, for example the `AsyncOutput` output copy and the `Memcpy DtoH`.
   - For each line in (b), give the pinned call site. Do not wrap any group (b) call in this assignment; the lead decides from the list.
4. **Run path A and path B as two jobs.** Change the G2h builder (`g2h_runner/prepare_package.py`) and its plan (`g2h_runner/package_plan.json`):
   - Build a new destination, `package_02`. Do not change or delete `package_01`.
   - Write two job files, each running the accepted bootstrap once in its own container:
     - `G2H_V4_RUNNER_REPLAY`, phase `G2H_RUNNER_REPLAY`;
     - `G2H_V4_BLOCK_OPERATORS`, phase `G2H_BLOCK_OPERATORS`.
   - Job tags `v4-h200-g2h-runner-20261029-02-replay` and `v4-h200-g2h-runner-20261029-02-block`.
   - One output directory per job, under `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_02/`.
   - The bootstrap, the base package, the image, the mount, `cuda_compat_path`, the 989 case IDs and the block grid stay unchanged.
   - The dry-run check covers both jobs: the planned rows and case IDs equal `expected_rows.json`, and the import probe passes.
   - Update `g2h_runner/README.md`:
     - the two jobs and their commands;
     - the lead checks (four gaps, attention rows, the unowned summary and the new sampling rows);
     - the GPU time.

   For the GPU time, use the observed G2h-01 path A: job 3,089 s, 0.86 GPU-hours. Path B is a separate start-up plus the planned calls. The G1h-G4h cap is 7 GPU-hours, of which about 1.89 are used.

## Out of scope

- Any GPU, platform or network action. Building or submitting the package; the lead builds `package_02` from the commit.
- `native_dummy_bootstrap.py`, the base package and `package_01`.
- Wrapping the group (b) calls of decision 3.
- The path B module (`frontier/profiling/model_runner/dsv4_block_operators.py`), except where the split jobs need its arguments. The module has not run on a GPU yet; leave its behaviour as B13r made it.
- **Concurrent work.** B15 runs in parallel in the same worktree. Do not edit or rely on its files:
  - `frontier/cc_backend/`, `frontier/operators/`, `frontier/execution_time_predictor/`;
  - `frontier/scheduler/utils/`, `frontier/profiling/collectives/`, `frontier/metrics/constants.py`, `frontier/model_architectures.py`;
  - `tests/integration/run_glm53_comm_replay.py`, `tests/integration/run_dsv4_structural.py`;
  - the tests that `git status` shows modified or new outside your own files.

  Run only the tests named in Acceptance. If one of them is a B15-modified file and fails in a B15 area, the failure is not yours; report it.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records:** [codex_assignment_B13r_v4_runner_operators.md](codex_assignment_B13r_v4_runner_operators.md) and [g2h_runner/README.md](g2h_runner/README.md). Do not search `task_memory/` beyond these files and the `g2h_runner/` directory.
- **Writable paths:**
  - `frontier/profiling/attention/` (the graph-replay runner, and the case recorder only if your mechanism needs it);
  - `tests/unit/`, for the new test and the B13r runner tests;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/`: `prepare_package.py`, `package_plan.json`, `README.md` and `expected_rows.json`, without building a package.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b13s/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
  - No GPU, network, job submission or package installs. No deletions or moves.
  - Python 3.10. No magic literals. No guards for states the code cannot reach. No copy of a runner method.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. No recursive grep or find over the vLLM tree, the download tree or the trace directories; name the files.
  - Keep every module at or below 2,000 lines.
- **Environment:** the same as B13r:
  - Python `/data/ycfeng/envs/frontier-py310/bin/python`;
  - `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`;
  - `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`;
  - pinned vLLM source `/data/ycfeng/tmp/opencode/vllm-v0.28.0`, read-only.

## Acceptance

1. **Diagnosis.** Confirm or correct defect 1 with file:line for each step, in the pinned source and in the runner.
2. **Native-order test** (decision 2): it fails on HEAD and passes after the change. Report the failing assertion on HEAD.
3. **Step lifecycle:**
   - the step range opens and closes once per recorded step, for every step kind;
   - the CUDA_EVENT finding of decision 1 is backed by file:line;
   - the B13r tests still pass: `test_dsv4_runner_operators.py` and `test_dsv4_graph_replay_profiling.py`, updated only where they stubbed the old lifetime.
4. **Saved records unchanged.**
   - The rows builder, `dsv4_replay_gaps.py` and the runner-row builder give the same output before and after the change, on both:
     - the G1h-02 raw records (`/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw/`);
     - the G2h-01 raw records.
   - Expected gap values on G2h-01: `cuda_graph` 0.4801221185543367, `cuda_graph_piecewise` 0.6106290602187813, `eager` 0.5065229453923222, `eager_preparation` 1.2117857340253784 µs.
5. **GLM unchanged.** The GLM DSA runner tests pass. Name them.
6. **Unowned review** (decision 3): the three groups per case kind, with counts and call sites.
7. **Split jobs:**
   - the builder's dry-run check passes on the current HEAD and lists the two jobs, their tags, their output directories and one bootstrap call each;
   - `package_01` is untouched (`git status` on it is clean).
8. **Existing tests pass:**
   - `test_dsv4_*`, `test_*graph_replay*`, `test_*bridge*`, `test_profiling_runtime_boundary.py` and the kernel-gap tests;
   - the known failure `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default` is deselected;
   - report any skipped tests.
9. **Hygiene:**
   - line counts;
   - no unused imports (`/data/ycfeng/tmp/unused_imports.py`);
   - a cold import of each changed entry module loads no `torch`, `vllm` or `cuda` module.

## Expected return

- Report language: English.
- Changed paths, each with a summary.
- The results of Acceptance 1-9, with the commands, the expected and actual values, and the counts.
- The mechanism chosen and why.
- The step-kind table (does it sample, does it close its range).
- The unowned review.
- The GPU-time estimate.
- Scratch paths, deviations and limits.
- If the pinned source contradicts a decision above, stop that part. Report the evidence and the options. One example: `sample_tokens` does not follow every recorded `execute_model`.
