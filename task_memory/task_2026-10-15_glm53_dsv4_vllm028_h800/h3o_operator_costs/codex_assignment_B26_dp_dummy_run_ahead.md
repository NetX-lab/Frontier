# Task: model the native launch rule of DP dummy forwards: the host runs ahead of the GPU by up to K forwards, in lockstep across the DP group (stage 4 assignment B26, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created after the native P2-64 wave run (`exp-1030-144848-164721`) located the P2-64 TTFT gap (requirements item 43, D-DP-DUMMY-RUN-AHEAD). |

## Goal

On the A10 replay of P2-64, Frontier's TTFT is 417.1 ms against native 741.2 ms (-43.7%, FAIL); TPOT, E2E and throughput pass. P2-32 passes. Section 3.5 of `test_report_2026-10-30_a10_operator_path_replay.md` (main tree, task directory) gives the measured cause. Read it first. In short:

1. Every forward, real or dummy, runs a CPU all-reduce over the DP group (`dispatch_cg_and_sync_dp`, native `vllm/v1/worker/gpu/model_runner.py:1408`), so the lanes' hosts advance in lockstep, one forward at a time.
2. The native dummy path (`_dummy_run`, `model_runner.py:633`) issues one FULL CUDA graph replay and does not wait for it. The host runs about 4.68 ms per dummy against 23.81 ms of GPU time, so it runs ahead of the GPU until 19 forwards are unfinished; then the launch blocks and the host is paced by the GPU.
3. The 32-step DP sync counts host iterations. A wave therefore ends with about 18 dummies (about 440 ms of GPU work) still queued, and the next prefill on that lane waits behind them on the GPU. This produces the TTFT sawtooth of P2-64 (about 290 to 745 ms). P2-32 is flat because its requests arrive after the queue has drained.

Frontier's host instead waits for every dummy forward to end: `VLLMv1EngineReplicaScheduler.on_schedule` returns no batch while `_dummy_forwards_in_flight` is non-zero (`frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`). A wave never ends with queued dummies, so a prefill starts at once.

A FIFO model of the prefill side (`h3o_operator_costs/a10/wave/fifo.py`) with Frontier's prices predicts the mean prefill-side time to rise from 300.0 ms (no run-ahead) to 514.9 ms (K = 19). The model with native prices gives 544.9 ms against the measured 536.7 ms.

Fidelity gain of this change to the core scheduler: it adds the measured host run-ahead of DP dummy forwards, which no existing mechanism represents. With the field unset, every existing result stays identical.

## Decision (requirements item 43, D-DP-DUMMY-RUN-AHEAD)

New field `VllmV1SchedulerConfig.max_forwards_in_flight: Optional[int] = None` (K), with `omit_if_none` like `max_concurrent_batches`, and a help text in the same style.

- **`None`:** the current path, unchanged. The host waits for each dummy forward to end.
- **Validation:** an exact `int` >= 1 (reject `bool` and everything else) in `__post_init__`, in the style of the `max_concurrent_batches` check.
- **Where K applies:** only on lanes with `_runs_dp_dummy_passes`. A replica scheduler built with K set on any other lane fails in its constructor with a clear message naming the field and the reason. So does one with `num_pipeline_stages > 1`: the run-ahead depth is measured only for one stage (section 3.5).

With K set, a lane follows these rules:

- **(a) No wait for a dummy forward.** Forming a dummy pass does not hold the host until the dummy ends.
- **(b) The K bound.** After forming its forward n (a batch or a dummy pass), the lane starts its next iteration only once its forward n−K has ended on the device. When the host leaves an iteration, at most K of its forwards are unfinished; this matches the native launch of forward n, which waits for the device end of forward n−K. Count every forward, batches and dummy passes alike. The device end of a forward is the end of its last stage on the device, not the later output pop (`hold_batch_output`).
- **(c) Per-forward lockstep.** After forming its forward n, the lane starts its next iteration only once every lane of its DP group (`_dp_group()`) has formed its forward n. `_batch_creation_counter` already counts a lane's forwards, and the forward-cohort identity of the pre-MoE rooms relies on equal counts across lanes. So the condition is that no lane of the group has a smaller counter than this lane.
  - The lane whose formation completes forward n wakes the lanes that wait for it at that time. Reuse the release path of the DP sync: `_dp_sync_released_lane_ids`, consumed by `ReplicaScheduleEvent` through `consume_dp_sync_release`. Do not add a second wake path.
  - A request that reaches a lane while it waits under (b) or (c) goes into a later forward, not the one already formed. Native schedules a forward before its all-reduce and launch.
- **(c2) Wave start.** The lane that starts a DP wave (`_run_dp_iterations`, "An engine that receives a request starts a wave on every engine") also wakes the other lanes at that time; native broadcasts `START_DP_WAVE` to every engine. With K unset, the existing wake through the forward's first room stays.
- **(d) Batches** keep the existing batch-queue rules (`_blocking_batch_id`, `max_concurrent_batches`, `_unpopped_outputs`, the pop that issues a dummy pass when the iteration formed nothing). Rules (b) and (c) add to them.
- **(e) The wave counter** (`_end_iteration`, `VLLM_DP_SYNC_INTERVAL`) still counts host iterations. With K set, a dummy iteration ends when the host leaves it: the later of the (b) and (c) releases. The DP sync, the held outputs and the wave end keep their current logic. A wave may now end while dummies are still unfinished on the device.
- **(f) Device order.** Each lane's forwards run on its stage in formation order, behind the forwards already queued there; the stage queue's lane-scoped `global_id` order already gives this. Verify that, with several forwards queued per lane, the shared stage execution context (`stage_execution_context.py`) and the pre-MoE rendezvous rooms (`frontier/scheduler/utils/dp_dummy_forward.py`) still pair the k-th forward of every lane. Change them only if a test shows they do not.
- **(g) Zero-cost spin.** Keep the meaning of `next_pass_start`. A group whose forwards cost zero, and whose lanes only wait for posted NIXL READs, jumps to the earliest READ wire end instead of spinning at one time. Apply it to the iteration end of rule (e).
- **(h) Unfinished-forward state.** `_dummy_forwards_in_flight` is set to `_num_stages` on every pass. With K set, several passes are unfinished at once, so the state must track every unfinished forward of the lane in formation order. Keep a single source of truth: do not track the same forwards in two parallel structures.
- **(i) Iteration-end bookkeeping.**
  - `_last_iteration_end` (read for `Batch.engine_idle_time`, consumed at `sklearn_execution_time_predictor.py:7065` when a CPU-overhead table has engine-idle buckets) is the iteration end of rule (e).
  - NIXL READs that a dummy pass found are reported when the iteration that ran the pass ends. With K unset this is the dummy end, as now.
  - NIXL READs that a pass posts keep using the time the pass is formed.
  - `engine_loop_idle` is true when the host waits for work. Dummy forwards still queued on the device do not make it busy; a host held by (b), (c), a blocking batch or the DP sync does.
- **(j) Placement-policy hooks.** If any consumer reads dummy-forward ends or `_dummy_forwards_in_flight`, keep its meaning under K and say how. One example is the load reports of `VllmLoadBalancingClusterScheduler`. Check every reader with `grep`.

## Work

1. The config field and its validation.
2. The scheduler and event changes for rules (a)-(j), with the smallest change that keeps one code path. With K unset the code must take today's path: the existing tests and the lead's 77-case fidelity matrix must give identical results.
3. Line count. `vllm_v1_engine_replica_scheduler.py` has 1,918 lines, and every changed file must stay at or below 2,000 (AGENTS.md "Python Module Size and Naming").
   - First remove dead, redundant or overly defensive code you find in the module.
   - If it is still over, move the DP busy-loop state and methods into a child module `frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py`, named after vLLM's `run_busy_loop`, with identical behaviour.
   - Report which you did and the line counts.
4. Tests. Extend `tests/integration/test_vllm_dp_dummy_pass_runtime.py`, or add `tests/integration/test_vllm_dp_run_ahead_runtime.py` with the same fixtures. Cover:
   - config validation (None, 1, 19, 0, -1, `True`, 2.0);
   - rejection on a lane without DP dummy passes and on PP 2;
   - K unset: forward start and end times equal to the current code on an existing DP dummy scenario;
   - an idle DP group runs ahead to exactly K unfinished forwards per lane, then one new forward per device end;
   - lockstep with a lane held by a batch: the idle lanes stop one forward ahead and resume when the busy lane forms its next forward;
   - a request that arrives while its lane waits under (b) or (c) is in the forward after the waited one;
   - a wave that ends at the DP sync with queued dummies: the next prefill on that lane starts on the device when the last queued forward ends;
   - the k-th forwards of all lanes pair in one room (equal cohort ids) with several queued per lane;
   - the zero-cost spin of rule (g);
   - the NIXL report time of rule (i) on a DECODE lane with `nixl_read_completion='engine_step'`.
5. The A10 driver (`h3o_operator_costs/a10/a10_driver.py`): add `vllm_v1_scheduler_config_max_forwards_in_flight` with the value 19. The value is a measured input: name it as a constant whose comment cites report section 3.5, observation 4, and the run id. It must reach both PDD roles through the base scheduler config; check this on the built P2-64 and P2-32 configs. Adjust `tests/unit/test_dsv4_a10_tools.py` if its expectations list the fields. If the observer `a10_observer.py` assumes one unfinished dummy per lane, fix it so its rows still equal the stage ends.
6. Docs: one row for the new field beside the `max_concurrent_batches` row of `docs/general/glm53-dsv4-vllm028.md` (line 156), in the same style. Do not edit `README.md`.
7. A short check on the operator path (non-dummy V4 predictors, the A10 driver), two runs from one copied warm cache:
   - Trace: a 32-request prefix of `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/trace.csv` with `num_decode_tokens` set to 2. Write it to scratch and record its SHA-256. Pass it with `a10_driver.py p2_64 <root> --trace <file>`.
   - Cache: a byte copy (`cp -a`, not hard links) of `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_g1hs_r2/cache` (312 MB) into each run root before the run.
   - Runs: one with K unset and one with K = 19. Pass the unset run with the driver's `fields` override, or a flag you add for it, and record which.
   - Outputs: per request, arrival, prefill stage start and end on its PREFILL lane, and TTFT. From the observer rows, also the number of forwards queued ahead of the prefill on its lane when it was formed, and the gap between the end of the previous forward on that lane and the prefill's stage start. Report the mean and range of the prefill-side time (arrival to prefill end) and of TTFT.

## Acceptance

1. With K unset, the tests named below pass unchanged. On the scenario of item 4, K unset gives forward times equal to the base commit.
2. The new tests pass. They show the K bound, the lockstep, the wave-end queue, the pairing, the zero-cost spin and the NIXL report time as the rules state.
3. Both PDD roles of the built P2-64 and P2-32 configs carry K = 19, and the receipt records it.
4. Short check:
   - K unset: `request_metrics.csv` and `system_metrics.json` are byte-identical to a run of the base checkout `/data/ycfeng/Frontier/.worktrees/b26-base` (detached at the branch point, created by the lead; read-only) on the same trace and a separate copy of the same cache, and no cache file is added.
   - K = 19: each prefill starts on its lane's device at the end of the previous forward there, with a gap of at most 1 µs, whenever forwards were queued ahead of it.
   - K = 19: the prefill-side times rise in a sawtooth between waves; report the series. The lead compares the mean with the FIFO model, which predicts about 509-515 ms for the 128-request trace.
5. Run these and report the counts:
   - `tests/integration/test_vllm_dp_dummy_pass_runtime.py`, `tests/integration/test_vllm_dp_placement_runtime.py` (one known pre-existing failure; name it if it appears), `tests/unit/test_dp_dummy_forward_pricing.py`, `tests/unit/test_vllm_v1_engine_idle_time.py`, `tests/unit/test_dsv4_dp_sync.py`, `tests/unit/test_forward_sync_state.py`, `tests/unit/test_nixl_read_completion.py`, `tests/unit/test_dsv4_structural_execution.py`, `tests/unit/test_cluster_scheduler_dp_lanes.py`, `tests/unit/test_vllm_dp_load_balancer.py`, `tests/unit/test_monolithic_mixed_forward_sync.py` and `tests/unit/test_dsv4_a10_tools.py`;
   - the new or extended test file;
   - `tests/integration/run_dsv4_structural.py --dp 4 --case offline` and `--case online`;
   - `bash examples/architecture/run_dummy_smoke_matrix.sh`;
   - the release starting checks in AGENTS.md "Tests".
6. Hygiene: line counts of the changed files (each at or below 2,000); no unused imports (`/data/ycfeng/tmp/unused_imports.py`); `git diff --check` clean.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b26-dp-dummy-run-ahead` (Git worktree, branch `b26-dp-dummy-run-ahead` from the main commit that adds this file; `git merge-base HEAD feat/glm53-dsv4-vllm028` gives it). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records (read in the main tree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, task directory `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`):**
  - `requirements.md` (items 42, 43);
  - `test_report_2026-10-30_a10_operator_path_replay.md` (sections 3.1-3.5);
  - `h3o_operator_costs/a10/wave/` (scripts and outputs of section 3.5);
  - `issues.md` (I-005).
- **Native source (read-only):** `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`: `vllm/v1/engine/core.py` (`DPEngineCoreProc.run_busy_loop`, `step_with_batch_queue`), `vllm/v1/worker/gpu/model_runner.py`.
- **Writable paths (in the working root):**
  - `frontier/config/replica_scheduler_config.py`;
  - `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`, and the child module `frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py` only if Work item 3 needs it;
  - `frontier/events/batch_stage_end_event.py`, `frontier/events/replica_schedule_event.py`, `frontier/events/cluster_batch_end_event.py`;
  - `frontier/scheduler/utils/dp_dummy_forward.py`, `frontier/scheduler/replica_stage_scheduler/stage_execution_context.py` and `frontier/scheduler/cluster_scheduler/base_cluster_scheduler.py`, only if rule (f) or (j) needs it;
  - the test files named above and the new test file;
  - `docs/general/glm53-dsv4-vllm028.md`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py` and `a10_observer.py`;
  - your report under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b26/`.
  - Ask in the report if another path is needed. Do not change it yourself.
- **Read-only inputs:** the warm cache named in Work item 7 (copy it; do not write into it), `data/`, the native wave artifacts under `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-wave-20261030-01/` if you need them.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b26/`. `/data/ycfeng` has about 220 GB free; keep the scratch below 10 GB.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
  - No GPU, network, job submission or package installs. No deletions or moves outside your scratch. Do not write under `data/`.
  - Four A10 runs are simulating under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/`. Do not stop or signal them, and do not write into their roots. No recursive `grep` or `find` over `/data/ycfeng/tmp/glm53_dsv4/`.
  - Python 3.10.
  - No magic literals in code paths: K comes from the config.
  - No guards for states the code cannot reach.
  - No second way to do what the code already does: reuse the DP-sync release path, `_dp_group()`, `_batch_creation_counter` and `next_pass_start`.
  - No temporary patches, scaling factors or fallbacks that hide errors.
  - Memory: `ulimit -v 62914560` for simulations, `ulimit -v 31457280` for tests and analysis, and a `timeout` on every command.
- **Environment:** `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, BLAS/OMP/MKL threads 1. For the A10 driver, follow `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10.sh` (its `PYTHONPATH`, `TMPDIR` and `FRONTIER_TMP_ROOT` settings), with your own scratch paths.

## Expected return

- Report language: English. Write it to `h3o_operator_costs/b26/report.md` and return its content.
- Changed paths, each with a summary of the change. Give the diff of the scheduler's DP iteration and dummy-pass methods.
- How each rule (a)-(j) is implemented and tested, including the consumers found for (j).
- Acceptance 1-6 with the commands, the expected and actual values, and the counts.
- The short-check series of Work item 7: a table per run plus the means and ranges.
- Scratch paths, deviations and their reasons, and limits.
