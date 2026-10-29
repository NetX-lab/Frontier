# Task: the A11 forward census and step-class time comparison for the A10 replays (stage 4 assignment B21, CPU)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Added decision 1b, the record of the measured collective requests of a run (exact point or interpolation, and its bracketing points), after the B19 held-out result for the uniform collective curves. |
| 2026-10-29 | Created. Covers the Frontier per-forward record attached by the A10 driver, the comparison with the native serving counters and the native runtime census, and the identity check of the record. |

## Goal

Gate A11 of the H3o design ([design.md](design.md) §1) is a diagnostic cross-check, reported and never a gate:

- the census of graph mode, synchronized widths and dummy counts per step class of each A10 replay (P2-32, P2-64) against the native counters;
- composed against native time per step class (decode-only by batch size, first tail alone, first tail with decodes, prefill) and per phase.

Entry 18 of the design's file table plans this as the A11 part of `tests/analysis/v4_runtime_census.py`. The A10 runs take hours on the operator path, so the Frontier record must exist before they start. This assignment delivers the record, the comparison and their checks. The lead runs the full A10 replays afterwards.

## Approach and decisions

### Read first

- [design.md](design.md) §1 (A11, A13), §2 (the native step classes and modes), §3.7 (DP synchronization), and entry 18 in the file table.
- `tests/analysis/v4_runtime_census.py` (the R1 reducer: per-forward records from the native torch traces, step classes from the `execute_context_<n>(<tokens>)_generation_<n>(<tokens>)` range, segments `prepare`, `dispatch`, `post`).
- In `h3o_operator_costs/evidence/`: `routing/mode_census.py`, `routing/review_counter_checks.py`, `routing/mode_census.txt`, and `runtime_census/p2_32.txt`, `p2_64.txt` (with their `.log` files).
- The native profile runs of the two cases, read-only:
  - `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/`
  - `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/`
  - Each has `nodes/<host>/serving/rank0.json` with aggregated `counters.rows`: `kind` (`serving`, `dummy`, `capture`, `profile`), `phase`, `scheduled_num_requests`, `scheduled_num_tokens`, `successful_forwards`, and `dispatch_evidence` with the classifier inputs, `dp_post_all_reduce` and the final descriptor.
  - `torch_traces/` holds the TP-rank-0 traces that R1 reads.
- The A10 tools: `h3o_operator_costs/a10/a10_driver.py`, `a10_common.py`, `compare_cache_runs.py`, and `tests/unit/test_dsv4_a10_tools.py`.
- Where Frontier finishes a real or a dummy forward on PREFILL and DECODE: `frontier/events/batch_stage_end_event.py`, `frontier/events/replica_stage_schedule_event.py`, `frontier/scheduler/utils/dp_dummy_forward.py` (`_synchronize_dp_descriptors`), `frontier/entities/batch.py` (`DummyForwardBatch`, `DecodeCudaGraphMetadata`) and `frontier/attention/dsv4_features.py` (`dsv4_request_is_decoding`).

### Decisions

1. **The Frontier record is a task-local observer, not simulator code.**
   - A new module in `h3o_operator_costs/a10/` records one row per lane forward, real and dummy, on PREFILL and DECODE: role, replica, DP lane, start and end simulated time, real or dummy, final runtime mode, local tokens and requests, the synchronized width, the classifier inputs (has prefill, max query length, uniform count), the step class and the composed time.
   - The step class uses the native rules: a first tail is a DECODE row that the V2 runner rule counts as prefilling (`dsv4_request_is_decoding`); `first_tail_alone` and `first_tail_with_decodes` are split as R1 splits them; decode-only forwards carry their batch size; PREFILL forwards with a prompt chunk are `prefill`.
   - The composed time is the forward's predicted stage time. Split it into the native segments `prepare`, `dispatch`, `post` when the V4 execution time exposes its scope components; write the scope-to-segment mapping in the report. If a scope has no native segment, list it as its own column; do not drop it.
   - `a10_driver.py` attaches the observer in its own process and writes `forwards.jsonl.gz` in the run root. The receipt records its path, row count and SHA-256.
   - Do not edit files under `frontier/`. If no observation point outside `frontier/` sees every real and dummy forward with its final descriptor, wrap the narrowest existing method in the driver process only, and name it in the report.
   - The observer reads state and never changes it. B20 runs at the same time in another worktree and edits `frontier/moe_ep_workload.py`, `dsv4_operator_predictor.py`, `deepseek_v4.py`, `dsv4_predictor.py` and `dsv4_training.py`. Do not hook inside those modules, so the two merge without conflict.
1b. **Collective requests.** The B19 held-out result ([a6/](a6/), the leave-one-out errors of `_Curve.describe`) shows that most uniform measured collective curves have few payload points. Whether that matters depends on which payloads a run actually requests. The observer also records each call of `MeasuredCCBackend.predict_operator_collective` on both roles, wrapped in the driver process only:
   - Aggregate by (role, curve key, requested bytes, unequal rows or none): the call count, whether the bytes are a measured point or interpolated, the two bracketing measured points and their log-distance ratio `log(bytes / lower) / log(upper / lower)`, and the predicted time.
   - Write it as `collectives.json` in the run root and record its path and SHA-256 in the receipt.
   - The comparison reports, per curve key, the share of calls and of predicted collective time that is exact, and the share that is interpolated.
   - A failing call still raises as today; the wrapper records nothing for it and re-raises the same exception.
2. **The comparison** is a new `tests/analysis/v4_replay_census.py`.
   - It reuses `v4_runtime_census.py` for the native per-forward records (import it; do not copy its functions). It reads the counters of every node of each role, not only the first node.
   - It counts native forwards from `serving` and `dummy` rows of the serving phase. Warmup, capture and profile rows are reported separately and excluded from the comparison; state how the phase field separates them.
   - Census, per role and per lane: forwards by (real or dummy, step class, final mode); synchronized widths by step class; dummy forwards by the mode of their peers. Frontier against native, with counts and differences.
   - Time, per role and step class (decode-only by batch size): the median and p90 of the Frontier composed time and of the native R1 device span, and per segment; absolute and relative errors. The relative error is undefined when the native value is zero. R1 covers only the profiler window; state its forward count beside every native value.
   - Output: one JSON and one text summary per case.
3. **Nothing else changes.** No predictor, table, scheduler or configuration field. A difference the comparison finds is reported, not fixed.

## Acceptance

1. **Unit tests** (new `tests/unit/test_v4_replay_census.py`):
   - Step classes: Frontier batches built as in `test_dsv4_graph_dispatch.py` (first tail alone, first tail with decodes, steady decodes, a dummy lane, a PREFILL chunk) get the class R1 gives the same native forward shape.
   - Counter reading on a small fixture written in the test: phase separation, `successful_forwards` weighting, multi-node aggregation.
   - The comparison arithmetic, including the undefined relative error at a native zero.
2. **Identity of the observer.**
   - Scenario: the P2-32 config of `a10_driver.py` with a trace of 8 requests arriving at 0.00, 0.05, ..., 0.35 s, each with 6,144 prompt tokens and 48 output tokens. Write the trace into the run root and record its SHA-256.
   - Run it with and without the observer, each with a cold cache.
   - Pass: every artifact identical after parsing (`compare_artifact_directories` with `substitutions_for`), and no predictor cache name difference (`classify_cache_differences`). The wall time of `Simulator.run` with the observer is within 2% of the run without it, or report both values.
3. **Record content** of the observed small run:
   - Rows for both PREFILL lanes and both DECODE lanes.
   - At least one PIECEWISE first tail, one FULL decode forward with more than one request, one DP dummy forward, and one NONE prefill.
   - Every forward of the run appears once; show the count against the number of stage-end events of the run.
   - `collectives.json` holds both roles; its call count per role equals the number of `predict_operator_collective` calls counted independently in the same run.
4. **Comparison run** on the small run against the P2-32 native inputs: the JSON and text summaries exist and every native count equals the sum of `successful_forwards` of its counter rows. The values are observations of a small scenario, not a reading of A11.
5. **Regression.** `tests/unit/test_dsv4_a10_tools.py`, `test_dsv4_graph_dispatch.py`, `test_dsv4_dp_sync.py`, and the new test. The dummy smoke matrix (`PYTHON_BIN` set) 12 of 12.
6. **Records.** `h3o_operator_costs/b21/` holds the trace generator, the run and comparison scripts, the identity JSON and a `report.md` with the exact commands, Python version, environment, expected and actual values and PASS/FAIL per item. Touched modules stay at or below 2,000 lines.

## Writable paths

Your working tree is `/data/ycfeng/Frontier/.worktrees/b21-v4-census` (branch `b21-v4-census`). All paths below are relative to it. B19 and B20 run in `.worktrees/b19-v4-heldout` and `.worktrees/b20-v4-speed`; their changes are not part of your base.

- `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/` (the driver edit and the new observer module).
- `tests/analysis/v4_replay_census.py` (new).
- `tests/unit/test_v4_replay_census.py` (new), and `tests/unit/test_dsv4_a10_tools.py` for the receipt change.
- `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b21/` (new).

Do not edit `frontier/`, the profiling tables, the native inputs or the lead's scratch directories. Temporary outputs, caches and run roots go under your private scratch directory. Simulations run with `ulimit -v 62914560` and a timeout.

## Expected return

- The results of Acceptance 1-6, with the commands, the expected and actual values and the counts.
- The observation point chosen, and any method wrapped in the driver process.
- The scope-to-segment mapping of the composed time.
- For the small run: per curve key, the exact and interpolated call shares.
