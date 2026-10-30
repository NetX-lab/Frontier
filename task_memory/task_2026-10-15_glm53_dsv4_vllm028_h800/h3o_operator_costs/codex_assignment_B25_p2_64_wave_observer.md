# Task: record the native DP wave restart on every P2-64 engine and build the diagnostic package (stage 4 assignment B25, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created after the P2-64 TTFT diagnosis (A10 report section 3) and YC's approval of one instrumented native run (requirements item 42, D-P2-64-WAVE-DIAGNOSTIC). |

## Goal

Frontier's P2-64 TTFT is 43.7% below native vLLM. The native scheduler lifecycle records place almost all of the gap inside the single 6,144-token prefill step on the P side. A prefill that arrives while the previous DP wave still runs takes about 300 ms. A prefill that arrives after the wave ended finishes only about 450-540 ms after that wave end, which adds a mean wait of 264 ms. Evidence and the source reading: `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/test_report_2026-10-30_a10_operator_path_replay.md`, section 3.

The source reading of vLLM-BS `2cf0a691` finds no delay on the wake-up path:

1. The front end sends `FIRST_REQ` when it sees no running wave.
2. The coordinator broadcasts `START_DP_WAVE`.
3. `DPEngineCoreProc.add_request` sends `start_wave` for a request of a finished wave.
4. An idle engine runs dummy steps once `engines_running` is set.

Each forward of the V2 model runner starts with a CPU all-reduce over the DP group (`vllm/v1/worker/gpu/dp_utils.py:16-39`, `sync_cudagraph_and_dp_padding`, gloo `cpu_group`). The real prefill on one lane therefore cannot start before every other lane of the role joins that all-reduce.

One native P2-64 run with timing records must show where the delay sits. The candidates are:

- the start-wave message: late from the front end or coordinator, or late out of the engine's input queue;
- the idle engine: late to start its first dummy step;
- the worker: late to join the DP all-reduce, or the all-reduce itself slow;
- none of these, in which case the delay is device work after the all-reduce.

## Work

1. **Engine-core records.** Add a native observer for every engine core of both roles. Start from `tests/comparison/calibration/native_scheduler_lifecycle.py` (the `LifecycleObservedScheduler` selected through `scheduler_cls`) and keep its records and contract.
   - **Busy loop.** For every iteration of `DPEngineCoreProc.run_busy_loop`, record:
     - begin and end of `_process_input_queue`, and whether it blocked;
     - begin and end of `_process_engine_step`, and whether a model step was executed;
     - begin and end of `execute_dummy_batch`;
     - begin and end of `_has_global_unfinished_reqs`, with `step_counter`, whether the all-reduce ran, and its result;
     - `current_wave` and `engines_running` after the iteration;
     - the `wave_complete` emission.
   - **START_DP_WAVE.** Record when each message enters the engine's input queue (from the input thread) and when `_handle_client_request` handles it. Record its wave, the excluded engine index, and `engines_running` before and after.
   - **Adds.** Record the enqueue time of each add request. Record the wave handling in `DPEngineCoreProc.add_request`: `request_wave`, `current_wave` before and after, `engines_running` before and after, and whether `start_wave` was sent.
2. **Worker records.** Record on TP rank 0 of every engine. Add other TP ranks only if the overhead check shows they are cheap.
   - Begin and end of each `sync_cudagraph_and_dp_padding` call, with the exchanged values (this rank's tokens and the reduced result).
   - Entry and return of the worker's `execute_model`, `execute_dummy_batch` and `sample_tokens`.
   - Install the worker hooks through the existing worker-extension path (`worker_extension_cls`, now `native_full_structure.FullModelInventory`) or the existing observer bootstrap, whichever already runs in the worker processes. Keep the inventory behaviour unchanged.
3. **Coordinator records (optional).** Record the times at which the coordinator receives `FIRST_REQ` / `start_wave` and broadcasts `START_DP_WAVE`, only if an existing hook point runs in the coordinator process without patching vLLM sources. Otherwise report the gap in coverage; items 1 and 2 still separate the candidates above.
4. **Contract (same as the lifecycle observer).**
   - Every wrapped call returns the native result unchanged, and native exceptions propagate unchanged.
   - Observation failures are recorded (`error` records plus the `NATIVE_SCHEDULER_LIFECYCLE_UNVERIFIED` style marker on stderr) and never reach native code.
   - Use CPU timestamps only: no device synchronization, no added collective, no new INFO-level logging.
   - Records stay buffered and are flushed as the lifecycle log is flushed, including once the scheduler drains.
   - Without `FRONTIER_NATIVE_SERVING_EVIDENCE_DIR` nothing is installed.
   - Prefer wrapping instance or class attributes from the harness modules. Do not edit the NATIVE checkout, the runtime source overlay or the installed vLLM.
5. **CPU check.** Extend `tests/comparison/calibration/check_native_scheduler_lifecycle.py`, or add a sibling check, and run it in the vLLM 0.28 environment (`/data/ycfeng/envs/oneshot-opd-py312/bin/python`; its wheel matches NATIVE `2cf0a69` for the scheduler and engine-core files, per the H4b design). It must show:
   - native results unchanged with and without the observer;
   - every record kind written with the expected fields on a driven `DPEngineCoreProc` iteration sequence: idle, start wave, dummy steps, the 32-step sync and wave end, add of a stale-wave request;
   - injected observer failures isolated;
   - the per-iteration observation overhead, measured. Target: at most 50 µs per busy-loop iteration. A dummy step takes about 5 ms, so this stays under 1%.
   - Report the record volume expected for one P2-64 repetition (4 P and 4 D engines, about 150 s), using the native iteration rates from the lifecycle records.
6. **Timeline analysis.** Add `tests/analysis/native_dp_wave_timeline.py` with unit tests on synthetic records. It reads a run copy (`nodes/<host>/...`) and writes, for every formal P-side prefill:
   - arrival, enqueue, add and schedule on its lane;
   - the previous `wave_complete`;
   - on each other P lane, the `START_DP_WAVE` enqueue and handling, the first dummy step after it, and that lane's `sync_cudagraph_and_dp_padding` begin and end for the matching forward;
   - the first-token update.
   - Reuse the classification of `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/ttft/p_blackout.py`: served at once, or held after the wave end.
   - It must place the delay of each held prefill in one of the candidate intervals of the goal.
7. **Package builder.** Add the case `p2_64_wave` to `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/build_package.py`, with the smallest change that keeps its checks: file comparison, sealing, preflight run manifest, R3 Frontier reconstruction, offline launch path and `launch.py --check-only`. The package must equal `h200_mainline/p2_64_lifecycle/package_01` except:
   - case identity, run tag `dsv4-h200-p2-64-wave-20261030-01`, output path and purpose (DIAGNOSTIC; no E2E reference admission; never qualifies H800 cells);
   - the new observer module(s) and their hashes, the changed `scheduler_cls` / worker extension in both `engine.json` files, and the harness commit;
   - one repetition instead of three: 16 warm-up and 128 formal requests (requirements item 42). Use the existing field that sets the repetitions (the trace manifest's `repetitions.repeat_indices` is read by `pd_replay.py:415`). Report every file and check this touches. If one repetition needs changes beyond the workload manifest, its hashes and the checks that read them, stop and report instead of changing the replay driver.

   Do not run the build: it needs a clean harness worktree at a commit, and the lead commits. Show that the builder change works with a dry check that needs no commit, if the builder allows one; otherwise give the command for the lead to run.
8. **Design record.** Add section 9 to `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/design.md`. Cover the question, the records, the analysis, the package changes, the cost and the acceptance, in the style of section 8. Add a Modification History row.
   - Cost: 64 H200, `step_main`, non-preemptible, one repetition. Estimate the wall time from the receipts of `p2_64_lifecycle/package_01` (container start, repetitions, drain); the expected value is about 20-21 GPU-hours.
   - Acceptance:
     - observer files for all 8 engine cores and the 8 TP-rank-0 workers, with no `error` record;
     - full timelines for at least 95% of formal P prefills;
     - the client metrics reported, not gated.

## Acceptance

1. The CPU check passes in the vLLM 0.28 environment with the five points of Work 5, and reports the measured overhead and the expected record volume.
2. `tests/analysis/native_dp_wave_timeline.py` unit tests pass. Run on the existing lifecycle copy `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/diagnostic/inputs/p2_64l`, it reproduces the section 3 classification: 43 served at once, 36 held, 302 in the 260-760 ms group. It marks the new intervals as unavailable there.
3. Existing tests still pass: `tests/unit/test_native_observer_bootstrap.py`, `tests/unit/test_native_glm_observer.py`, and the existing lifecycle CPU check. Report the counts.
4. The builder change, with the dry check or the exact command, and the list of files the package will differ in from `p2_64_lifecycle/package_01`.
5. Hygiene:
   - line counts of the changed files, each at or below 2,000;
   - no unused imports (`/data/ycfeng/tmp/unused_imports.py`);
   - `git diff --check` clean.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/p2-64-wave-harness`, a Git worktree on branch `p2-64-wave-harness` from main `34cc34be`. Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records** (read in the main tree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`): `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`
  - `requirements.md`, item 42;
  - `test_report_2026-10-30_a10_operator_path_replay.md`, section 3;
  - `h4b_lifecycle_measurement/design.md`, sections 2-8: the observer contract, CPU check and package;
  - `h200_mainline/design.md`, section 8;
  - `test_report_2026-10-26_h200_p2_64_lifecycle.md`.
- **NATIVE source (read only):** `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.
  - `vllm/v1/engine/core.py`: `DPEngineCoreProc` from line 1963; `_process_input_queue` at 1401;
  - `vllm/v1/engine/coordinator.py`;
  - `vllm/v1/engine/core_client.py`, lines 1300-1430;
  - `vllm/v1/worker/gpu/dp_utils.py`;
  - `vllm/v1/worker/gpu_worker.py`, lines 1046-1215.
- **Writable paths:**
  - `tests/comparison/calibration/native_scheduler_lifecycle.py`, or new sibling modules in `tests/comparison/calibration/`;
  - `tests/comparison/calibration/check_native_scheduler_lifecycle.py`, or a sibling check;
  - `tests/comparison/calibration/native_full_structure.py`, only for the worker hook installation if that is the existing path;
  - `tests/analysis/native_dp_wave_timeline.py`, and a new unit test file under `tests/unit/`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/build_package.py`;
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/design.md`.
- **Read-only inputs:**
  - `h200_mainline/p2_64_lifecycle/package_01`;
  - `h200_mainline/diagnostic/inputs/p2_64l` and `p2_32l`;
  - the H4b package and records.
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b25/`. `/data/ycfeng` has about 60 GB free, and four long simulations are writing there. Keep scratch output small.
- **Restrictions:**
  - **Git:** no Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
  - **Actions:** no GPU, network, job submission, platform command or package install. No deletions or moves. Do not write under `data/profiling/` or into existing package directories.
  - **Running simulations:** do not stop, signal or read with heavy tools the running simulations under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/`. No recursive `grep` or `find` over `/data/ycfeng/tmp/glm53_dsv4/`.
  - **Code rules:**
    - no magic literals in code paths;
    - no guards for states the code cannot reach;
    - no second way to do what the code already does: reuse `LifecycleLog`, `compact`, the builder helpers and the existing checks;
    - plain names after what the thing is (no `helper`, `manager`, `enhanced` and the like).
  - **Memory:** `ulimit -v 31457280` and a `timeout` on every command.
- **Environments:**
  - Frontier code and the analysis: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM), with `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, and BLAS/OMP/MKL threads 1.
  - Native observer check: `/data/ycfeng/envs/oneshot-opd-py312/bin/python` (vLLM 0.28.0).

## Expected return

- Report language: English.
- Changed and new paths, each with a summary of the change.
- The record schema: each event kind and its fields.
- Where each hook is installed, and why that point runs in the target process.
- Acceptance 1-5 with commands, expected and actual values, and counts.
- The overhead measurement, the record volume estimate and the GPU-hour estimate with its inputs.
- Deviations and their reasons, coverage gaps (for example the coordinator), and limits.
