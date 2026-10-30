# Task: predict each distinct V4 EP lane once per wave, with identical results (stage 4 assignment B24, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created from the lead's cProfile window of the official A10 P2-64 configuration (`probe/p2_64_mid2`). |

## Goal

The A10 replays run on the V4 operator path. The P2-64 replay takes about 5.5 hours of wall time, and every fidelity fix that follows (B26 and later) needs new P2-32 and P2-64 replays. This assignment removes repeated work from the EP wave prediction without changing any result.

Measured hot spots. Input: the lead's 1,200 s cProfile window of the A10 P2-64 configuration, started at simulated 60.0 s with the warm cache (`/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/probe/p2_64_mid2/profile.pstats`, driver `stage4/probe_profile3.py`, launcher `stage4/run_profile3.sh`; read-only for you). The window reached 4.85 simulated seconds.

| Function | Calls | Cumulative | Note |
| --- | --- | --- | --- |
| `predict_ep_wave_phase_times` (`frontier/scheduler/utils/expert_parallel.py:91`) | 12,214 waves | 838.9 s of 1,180.1 s | One loop pass per EP participant: 32 per wave on P2-64 |
| inside it: `_predict_layer_with_kernel_gap` → `predict_structural_execution` | 390,848 | 465.1 s | One full V4 FFN prediction per participant |
| inside it: `phase_getter` (`get_ep_phase_times_ms`) | 390,848 | 151.6 s | Reads and checks the five phases of each prediction |
| inside it: `lane_builder` (`build_ep_lane_batch`) | 390,848 | 109.9 s | Builds a lane batch with synthetic requests |
| inside it: `_assemble_stage` | 390,848 | 56.6 s | One-layer stage per participant |
| inside it: `workload_logger` | 390,848 | 29.2 s | Per-lane workload trace |

Most participants of one wave repeat the same prediction. On the V4 path the prediction for participant `ep_id` depends on:

- the source batch of its DP lane, `source_batches[ep_id // attn_tp]`;
- its destination lane workload, through `grouped_gemm(lane_workload)` and `lane_workload.routed_token_count`;
- values that are the same for the whole wave: the layer, the SP token widths, the cluster type and the serving context of the source.

`grouped_gemm` reads `ep_id` only through the balanced rank class `materialize_balanced_rank_classes(...)[ep_id]`. That class is (the lane's routed rows, its experts with an extra row, the base rows): a function of the lane's `local_token_counts` and the wave's `global_token_counts`. The lead checked sample populations of the kind A10 waves produce (6,168, 12,304, 32, 88, 624 and 2,048 routing rows, top-k 6, 256 experts, EP 32): in each, the 32 lanes have one or two distinct `local_token_counts` tuples. So a P2-64 wave needs at most 4 × 2 = 8 distinct predictions instead of 32, and a P2-32 wave (attention TP 8 × DP 2, EP 16) at most 4 instead of 16.

Only the source TP rank (`ep_id % attn_tp`) and the lane identity differ inside such a group. On this path they reach only metadata: `valid_local_rows`, `request_valid_local_rows` and `source_tp_rank` in `_dsv4_source_rows`, which `frontier/metrics/ep_wave_metrics.py` (`record_ep_wave`) and `metrics_store.py` read from lane records. The phase times do not depend on them. This reading comes from the lead's code inspection; Work 1 must confirm it.

## Decisions

1. **Where.** Only the V4 branch of `predict_ep_wave_phase_times` (the branch taken when `source_batches` is not None), and only when lane records are not captured (`capture_lane_timings` is False). With lane records, or on the non-V4 branch, every participant keeps its own prediction exactly as today.
   - Reason: a lane record carries that lane's own source rows, so it needs its own prediction object. The A10 configuration captures no lane records (`write_metrics` True; `enable_op_level_tracing`, `store_operation_metrics`, `store_frontier_stage_batch_ledger` and `store_frontier_stage_batch_ledger_summary` all False).
   - The step-cost branch of `prepare_moe_wave_from_inputs` (`frontier/scheduler/utils/ep_wave.py:135`) already uses the same condition.
2. **Sharing rule.** Within one call, two participants share one prediction when both hold:
   - the same DP source index `ep_id // attn_tp`;
   - lane workloads equal in every field except `ep_id` and `owned_expert_ids`.
   Build the key from those named fields, so a field added later to `EPLaneWorkload` cannot be silently ignored: compare against the dataclass field list and fail with a clear error if the key does not cover every field except those two. Never key by `id()`. The memo is a local dictionary of the call and is dropped when the call returns.
3. **What a group shares.** The first participant of a group, in participant order, makes the prediction, assembles the stage and calls `phase_getter` once. The others reuse the five phase values. On a cache miss, the predictor therefore trains in the same order as today.
4. **What stays per participant.**
   - The `lane_builder` call and the SP-width assignment on the lane batch.
   - The `workload_logger` call, with that participant's own `lane_workload` and the group's phase values.
   - The six output tuples in participant order, with one entry per participant.
5. **No other change.** `ExecutionTime`, `Request`, `build_ep_lane_batch`, the communication payload validation, `deepseek_v4.py` and the scheduler files are shared or are being edited by B26 (`base_cluster_scheduler.py`). Do not edit them. If Work 1 finds a rank- or lane-dependent input to a phase time other than the rank class, stop and report it; do not widen the scope.

## Work

1. **Audit.** Follow the V4 participant prediction from `predict_ep_wave_phase_times` through `_predict_layer_with_kernel_gap`, `_predict_dsv4_structural_execution`, `predict_structural_execution`, `predict_source_execution`, `add_runner_scopes`, `add_dsv4_operator_gaps`, `DeepSeekV4OperatorPredictor.grouped_gemm` and `gate`, the communication predictions and `_assemble_stage`.
   - List every read of `tp_rank`, `ep_id`, the lane workload and the lane batch, and every write to predictor state (kernel-count scopes, caches, counters).
   - For each read, state whether it feeds a phase time or only metadata.
   - Record the table in the report.
2. **Implementation** as in Decisions 1-4, in `frontier/scheduler/utils/expert_parallel.py`.
3. **Unit tests**, in the new file `tests/unit/test_v4_ep_lane_reuse.py`:
   - On a trained V4 predictor (reuse the `trained` fixture pattern of `tests/unit/test_dsv4_operator_consumer.py`), for an attention TP 8 × DP 2, EP 16 setup and an attention TP 8 × DP 4, EP 32 setup, with different source batches per DP lane (one 6,144-token prefill lane with dummy lanes; decode lanes with different request counts):
     - the six output tuples with `capture_lane_timings=False` equal those with `capture_lane_timings=True` exactly (`==`);
     - the count of predictions with lane records off equals the number of distinct keys, and with lane records on equals the number of participants;
     - each `workload_logger` call receives its own participant's lane workload (`ep_id` matches).
   - The key check of Decision 2 fails when the workload dataclass gains an uncovered field.
4. **Existing test.** `test_ep_wave_uses_the_segment_gap_wrapper_for_every_lane` in `tests/unit/test_dsv4_operator_consumer.py` asserts one prediction per rank. Change it to assert that every prediction goes through the kernel-gap wrapper, and that each rank's routed time equals the prediction of its group. Change nothing else in that file.
5. **Identity runs** against the base (Acceptance 2).
6. **Speed check** (Acceptance 3).

## Acceptance

1. **Unit tests.** The new tests and the changed test pass.
2. **Identity against the base.**
   - Base: the detached worktree `/data/ycfeng/Frontier/.worktrees/b26-base` at `627253d5`. Its `frontier/` and `data/` are identical to your starting commit `a420db84` (only task records differ). B26 also reads it. Run the base from it and do not edit it.
   - Scenario S1: P2-32 of the A10 driver with B20's eight-request trace (arrivals 0.00, 0.05, ..., 0.35 s, 6,144 prompt and 48 output tokens; reuse `h3o_operator_costs/b20/generate_trace.py`, `run_replay.py` and `compare_runs.py`, copied into `b24/` and adjusted). Cold on both sides.
   - Scenario S2: P2-64 with the same trace, warm. Each side gets its own byte copy of `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_g1hs_r2/cache` (312 MB; read-only source, written by the code of `9a4f3471`, whose `frontier/` and `data/` equal the base). Record the SHA-256 inventory of the copies before the runs.
   - Scenario S3: S1 with `metrics_config_store_operation_metrics=True`, so the lane-record path runs. Warm, from a copy of an S1 cache on each side.
   - Pass for each scenario: `compare_artifact_directories` with `substitutions_for` reports no difference, and `classify_cache_differences` reports no predictor cache name difference.
   - Show from the S2 candidate run that groups were shared: the number of participant predictions against the number of participants (for example from a counting wrapper in the run script, kept out of the base run).
3. **Speed.**
   - `Simulator.run` wall time and peak RSS of the S1 and S2 runs, base against candidate.
   - A window on the full A10 P2-64 trace: run base and candidate for 900 s of `Simulator.run` wall time each, without a profiler, from simulated time 0, each with its own warm cache copy. Report the simulated time each reaches. Run the two windows at the same time, so they share the host load.
   - Expected: the candidate reaches at least 1.5 times the base's simulated time. Below that, report a 300 s cProfile window of the candidate and do not widen the scope.
4. **Remaining cost.** The top 15 functions by cumulative time in a 300 s cProfile window of the candidate on the P2-64 trace, each with a one-line cause. Estimate the full P2-32 and P2-64 wall times from the measured rates.
5. **Regression.**
   - `tests/unit/test_dsv4_operator_consumer.py`, `test_stage_reporting_contract.py`, `test_v4_prediction_memo.py`, `test_dsv4_structural_execution.py`, `test_dsv4_structural_pd.py`, `test_dsv4_source_execution.py`, `test_dsv4_dp_sync.py`, `test_moe_rank_plan.py`, `test_moe_rank_routing.py`, `test_dsv4_a10_tools.py`, and every unit or integration test whose name contains `ep_wave`, `expert_parallel` or `moe_ep`.
   - The dummy smoke matrix (`bash examples/architecture/run_dummy_smoke_matrix.sh`, `PYTHON_BIN` set) 12 of 12.
   - Known failures that predate this work and may remain: `test_balanced_routing.py` (2 failures, I-002) and `test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`. Show any other failure on the base too, or fix it if your change caused it.
6. **Records.** `h3o_operator_costs/b24/` holds the scripts, the comparison JSON, the audit table and a `report.md` with the exact commands, Python version, environment, expected and actual values and PASS/FAIL per item. `expert_parallel.py` stays at or below 2,000 lines.

## Writable paths

Your working tree is `/data/ycfeng/Frontier/.worktrees/b24-v4-ep-lane-reuse` (branch `b24-v4-ep-lane-reuse`, started at main `a420db84`). B26 runs at the same time in `.worktrees/b26-dp-dummy-run-ahead`; its changes are not part of your base. All paths below are relative to your working tree.

- `frontier/scheduler/utils/expert_parallel.py`.
- `tests/unit/test_v4_ep_lane_reuse.py` (new) and the one test of Work 4 in `tests/unit/test_dsv4_operator_consumer.py`. The write permission covers `tests/unit/`; edit no other file there.
- `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b24/` (created empty by the lead).

Do not edit the base worktree, the profiling tables, the A10 run roots or the lead's scratch directories. Temporary outputs, cache copies and run roots go under your private scratch directory. Simulations run with `ulimit -v 62914560` and a timeout; analysis runs with `ulimit -v 31457280`. Use `/data/ycfeng/envs/frontier-py310/bin/python` with `PYTHONDONTWRITEBYTECODE=1`, W&B disabled and BLAS/OMP/MKL threads 1. Make no Git commit; the lead reviews and commits.

## Expected return

- The outcome (done, partial or blocked) and the audit table.
- The changed and new files with the reason for each change.
- Per acceptance item: the command, the expected and the actual values, and PASS or FAIL.
- The identity comparison JSON paths, the window results and the profile table.
- Anything left open, with evidence.
