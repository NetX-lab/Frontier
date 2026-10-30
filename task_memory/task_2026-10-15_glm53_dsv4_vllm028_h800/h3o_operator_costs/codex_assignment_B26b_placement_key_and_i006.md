# Task: DP load-report key under host run-ahead, and the failing late-join placement test (stage 4 assignment B26b, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created from the B26 review (open item of `h3o_operator_costs/b26/report.md`) and issue I-006. |

## Goal

Two separate items in the DP placement area. Report them separately.

**Part A: the report key of `vllm_load_balancing` under `max_forwards_in_flight`.** B26 (merged into main as `50ed2061`) lets a DP host form forwards ahead of the GPU, up to `max_forwards_in_flight = K` unfinished forwards. `VllmLoadBalancingClusterScheduler._next_report_key` (`frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py:167-185`) still keys a load report by the stage-0 forward group that the lane's next batch joins on the device (`stage_execution_context.joinable_forward_group_id`). With K unset that group equals the host's formation index. With K set it does not: in the B26 probe (`/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/placement-key-probe.json`), with K = 19, lane 1 forms forward 23 at 0.124 s and reports under key 4.

vLLM keys the report by the host's busy-loop step counter and wave:

- `DPEngineCoreProc._maybe_publish_request_counts` stamps `SchedulerStats` with `step_counter` and `current_wave` (`vllm/v1/engine/core.py:2120-2135`);
- `run_busy_loop` publishes before and after each engine step (`core.py:2147-2212`); `_has_global_unfinished_reqs` increments `step_counter` once per busy-loop iteration, and a wave end resets it;
- the coordinator keeps a report only when its `(wave, step)` is newer than the last one (`vllm/v1/engine/coordinator.py:380-400`).

The step counter is a host count. It does not wait for the GPU, so under run-ahead it runs ahead of the device by up to K forwards.

**Part B: issue I-006.** `tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` fails on main (lead run on `88fbef52`: 18 passed, 1 failed; log `/data/ycfeng/tmp/main_placement.log`). The assertion at line 481 expects lane 2 to start forward group 0 with a dummy forward, `{2: (0, True)}`, and gets a real batch, `{2: (0, False)}`. The test file last changed in `5f907f66`; 143 commits follow it on main. Read `issues.md` I-006.

## Work

### Part A

1. Key each load report of a run-ahead lane by the host formation index of the forward it belongs to, following the rules of the class docstrings (an admission that fills the pipeline holds its key until the lane's next completion; a completion without a held key uses the key of the lane's next forward). Use the identity that B26 already keeps (`_batch_creation_counter`, the `DeviceForward` FIFO of `frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py`); do not add a second counter of the same thing.
2. With K unset, every key, every report time and every placement must stay the same as on main.
3. Check whether the "pipeline still has room" test at admission (`vllm_load_balancing_cluster_scheduler.py:128-132`, `lane.num_running_batches < self._num_pipeline_stages`) still names the native publish point under run-ahead, from the native busy loop above. Change it only if it does not, and give the native lines that decide it.
4. Tests: add a K regression to `tests/integration/test_vllm_dp_run_ahead_runtime.py` (or the placement runtime test, with its fixtures) that covers successive admissions and completions on two or more lanes with forwards queued ahead, a held key released by a completion, and a stale report that the balancer must drop. Each report key must equal the host formation index of its forward.

### Part B

1. Find the first commit on main after `5f907f66` at which the test fails. Export trees with `git archive <commit> | tar -x -C <scratch dir>` (no checkout, no `git bisect`) and run only this test in each tree. Record each commit tried and its result.
2. Find the cause in that commit's change.
3. Decide, from the native source, which side is wrong:
   - if the test still states the native behavior (an idle lane joins the peer's first MoE room with a dummy forward; a request that arrives during that dummy forward starts the lane's next forward, because `run_busy_loop` schedules it only after `execute_dummy_batch` returns), fix the simulator;
   - if the commit changed the native behavior that the test should follow, or the test's setup no longer reaches its shape, fix the test, and keep every assertion that still holds.
4. Before changing simulator code, state which runs the fix can change: the DP placement cases of the fidelity matrix, the A10 P2-32 and P2-64 replays (prefill requests 1 s apart, four PREFILL lanes for P2-64 and two for P2-32, `round_robin` placement), or other cases. If the fix changes a path that A10 uses, implement it, and report the effect on the 32-request short check of B26 (below).

## Acceptance

1. Part A: in the probe of B26 rerun with your change, every report key under K = 19 equals the lane's formation index; with K unset, the probe output equals the B26 probe output for K unset.
2. Part A: the new regression passes; the tests below pass, with no failure other than the I-006 test before Part B is fixed.
3. Part B: the test passes; the report names the first failing commit, the cause, the side fixed and the native lines that decide it.
4. Fidelity:
   - With K unset and the B26 32-request short check (`a10_driver.py p2_64 <root> --trace <file> --without-run-ahead`, trace and cache as in `h3o_operator_costs/b26/report.md`), `request_metrics.csv` and `system_metrics.json` are byte-identical to B26's K-unset outputs, unless Part B changes the simulator on that path; then give the differences and their cause.
   - The same for K = 19 against B26's K = 19 outputs.
5. Run these and report the counts:
   - `tests/integration/test_vllm_dp_placement_runtime.py`, `tests/integration/test_vllm_dp_run_ahead_runtime.py`, `tests/integration/test_vllm_dp_dummy_pass_runtime.py`, `tests/unit/test_vllm_dp_load_balancer.py`, `tests/unit/test_cluster_scheduler_dp_lanes.py`, `tests/unit/test_dp_dummy_forward_pricing.py`, `tests/unit/test_vllm_v1_engine_idle_time.py`, `tests/unit/test_dsv4_dp_sync.py`, `tests/unit/test_forward_sync_state.py`, `tests/unit/test_nixl_read_completion.py`, `tests/unit/test_monolithic_mixed_forward_sync.py`, `tests/unit/test_dsv4_a10_tools.py`;
   - `tests/integration/run_dsv4_structural.py --dp 4 --case offline` and `--case online`;
   - `bash examples/architecture/run_dummy_smoke_matrix.sh`.
6. Hygiene: line counts of the changed files (each at or below 2,000); no unused imports (`/data/ycfeng/tmp/unused_imports.py`); `git diff --check` clean.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b26b-placement-key`. It is a Git worktree on branch `b26b-placement-key`, created from the main commit that adds this file. Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records (read-only), in the main tree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, task directory `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`:** `issues.md` (I-006), `h3o_operator_costs/b26/report.md`, `h3o_operator_costs/codex_assignment_B26_dp_dummy_run_ahead.md`, `requirements.md` (item 43). The B26 scratch `/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/` is read-only; it holds the probe, the short-check trace and outputs.
- **Native source (read-only):** `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` (`vllm/v1/engine/core.py`, `vllm/v1/engine/coordinator.py`).
- **Writable paths, in the working root:**
  - `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py`;
  - `frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py`;
  - for Part B only, if the cause is there: `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`, `frontier/events/batch_stage_end_event.py`, `frontier/events/replica_schedule_event.py`, `frontier/scheduler/utils/dp_dummy_forward.py`;
  - `tests/integration/test_vllm_dp_placement_runtime.py`, `tests/integration/test_vllm_dp_run_ahead_runtime.py`, `tests/unit/test_vllm_dp_load_balancer.py`;
  - your report under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b26b/`.
  - If you need another path, ask in the report; do not change it yourself.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b26b/`. Keep it below 10 GB; exported trees can go there and need not be removed.
- **Restrictions:**
  - The lead commits. No Git commands that change state; `git diff`, `git status`, `git show`, `git log`, `git rev-parse` and `git archive` to a pipe are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
  - No GPU, network, job submission or package installs. No deletions or moves outside your scratch. Do not write under `data/`.
  - Four A10 runs and a fidelity matrix are running under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/`. Do not stop or signal them, and do not write into their directories. No recursive `grep` or `find` over `/data/ycfeng/tmp/glm53_dsv4/`.
  - Python 3.10.
  - No magic literals in code paths, no guards for states the code cannot reach, no second way to do what the code already does, no temporary patches, scaling factors or fallbacks that hide errors.
  - Memory: `ulimit -v 62914560` for simulations, `ulimit -v 31457280` for tests and analysis, and a `timeout` on every command.
- **Environment:** `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, BLAS/OMP/MKL threads 1. For the A10 driver, follow `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10.sh` (its `PYTHONPATH`, `TMPDIR` and `FRONTIER_TMP_ROOT` settings), with your own scratch paths and a byte copy (`cp -a`) of the cache.

## Expected return

- Report language: English. Write it to `h3o_operator_costs/b26b/report.md` and return its content.
- Part A and Part B separately: changed paths, each with a summary of the change, and the native lines that decide each rule.
- Part B: the commits tried with their results, the first failing commit and the cause.
- Acceptance 1-6 with the commands, the expected and actual values, and the counts.
- Scratch paths, deviations and their reasons, and limits.
