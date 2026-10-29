# Test report: B21, the A11 forward and collective record of the A10 replays

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: B21 (Codex `fcf9eca4`, outcome complete) review, the lead cleanup, the merge and the start of the official A10 runs. |

## Scope

Assignment: [h3o_operator_costs/codex_assignment_B21_v4_replay_census.md](h3o_operator_costs/codex_assignment_B21_v4_replay_census.md). Codex records: [h3o_operator_costs/b21/report.md](h3o_operator_costs/b21/report.md) and the handoff [codex-implement-fcf9eca4-…/implementation.md](h3o_operator_costs/codex-implement-fcf9eca4-0142-4d72-a3ea-443401fae6fe/implementation.md). Working tree `.worktrees/b21-v4-census`, branch `b21-v4-census`, commit `14fa251d` (base `a27d3ade`, which includes B19), merged into `feat/glm53-dsv4-vllm028` as `d53bab0b`. No file under `frontier/` changed.

## Delivered

| Area | File (lines) | Behavior |
| --- | --- | --- |
| Observer | `h3o_operator_costs/a10/a10_observer.py` (287) | Read-only. In the A10 process only, it wraps `BatchStageEndEvent.__init__` and `handle_event`, `dp_dummy_forward._synchronize_dp_descriptors`, `SklearnMoEExecutionTimePredictor.predict_stage_execution_time` (single-layer source attention, `include_ffn=False`), `BaseClusterScheduler._prepare_moe_ep_wave_plan` and `_get_shared_ep_phase_times_ms`, and `MeasuredCCBackend.predict_operator_collective`. It streams one row per lane forward to `forwards.jsonl.gz` and writes the collective requests to `collectives.json`. The patches are removed when the run ends |
| Driver | `h3o_operator_costs/a10/a10_driver.py` | Attaches the observer by default (`--without-observer` turns it off); optional `--trace`; the receipt records the trace hash, `observer_enabled` and the `a11` record (paths, row and call counts, SHA-256, stage-end counts, scope-to-segment mapping) |
| Comparison | `tests/analysis/v4_replay_census.py` (302) | Reads the counters of every node, uses `successful_forwards` weights and separates phases; imports R1 (`v4_runtime_census.py`) for the native per-forward device spans; census per role, lane, step class and mode; time per step class and segment; collective exact and interpolated shares |
| Tests | `tests/unit/test_v4_replay_census.py` (248), `tests/unit/test_dsv4_a10_tools.py` | Step classes against R1, counter weighting and phases, comparison arithmetic (relative error undefined at a native zero), observer restoration and failure, receipts with the observer on and off |
| Records | `h3o_operator_costs/b21/` | Trace generator, small-run, identity and comparison scripts, `identity.json` (17 of 17 PASS), `comparison_validation.json` (6 of 6 PASS), `report.md` |

Scope-to-segment mapping of the composed time: `prepare` is `dsv4_input_preparation`; `post` is `dsv4_lm_head`, `dsv4_logits_gather` and `dsv4_sampling`; `dispatch` is the V4 attention, mHC, MoE, shared-expert and V4 communication scopes, with attention counted by its critical path; `kv_cache_block_zero` and `dsv4_block_table_write` are their own columns; `coordination_and_overlap` is the composed time minus all of these (signed). EP scopes are those of TP rank 0 of each DP source lane, an attribution, not the EP barrier.

## Lead review

- **Read-only.** The wrappers call the original first and record afterwards. The collective wrapper records nothing when the original raises. The observer requests no extra prediction: the independent collective counts of the plain run equal the observed run's (267,170 calls).
- **Collective classification.** The observer locates the requested bytes in the measured curve to label it exact or interpolated. A request outside the measured range never reaches it, because `predict_operator_collective` raises `MeasuredCCOutOfRangeError` first (`measured_cc_backend.py:260-262`). Counts are prediction calls, so a payload priced twice for one forward counts twice.
- **Lead cleanup** (in `14fa251d`). The driver had two branches written only for the receipt test's fake configuration: `build_config(name, root, fields) if fields else build_config(name, root)`, and a `hasattr(config, "__flat_config__")` guard. `h4a4.build_config` always sets `__flat_config__`, so the driver now always builds with the given fields and always records the trace; the test's fake `build` takes `fields` and its fake configuration carries a trace file, and the test checks the recorded trace hash.
- **Native limits that Codex reported** (observations of the native inputs, kept as unknowns):
  - every native dummy counter row has `phase: null`, so formal dummy counts cannot be separated from warmup ones;
  - multi-request first-tail counter rows lack the context and generation split (`first_tail_unresolved`);
  - R1's profiler window has no warmup or formal tag, so time distributions are labelled `serving` (Frontier) and `profiler_window` (native), not matched by phase.
  - P2-32: 4 nodes, 67,600 weighted observations, 36,108 formal real, 20,750 dummies without a phase, 10,742 excluded (capture and profile), 328 unresolved formal tails. P2-64: 8 nodes, 111,904, 52,494, 41,449, 17,961 and 267.

## Checks

Codex ran with `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python` (Python 3.10.6, same package versions as `frontier-py310`); the lead checks use `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, BLAS/OMP/MKL threads 1, `ulimit -v 31457280`.

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Codex: unit tests (A10 tools, graph dispatch, DP sync, census) | all pass | 74 passed | PASS |
| 2 | Codex: identity of the 8-request P2-32 scenario (arrivals 0.00-0.35 s, 6,144 prompt and 48 output tokens), with and without the observer, each cold | parsed artifacts and cache names identical; run time within 2% or both reported | 0 artifact and 0 cache name differences, 1,758 cache files each; `Simulator.run` 1,140.3 s plain and 1,137.1 s observed (-0.29%, the runs overlapped other work) | PASS |
| 3 | Codex: record content | both lanes of both roles; PIECEWISE first tail, FULL multi-request decode, DP dummy, NONE prefill; one row per stage end; collective counts equal an independent count | 320 rows = 320 stage ends (PREFILL 64, DECODE 256), 0 stale; 138 dummies, 8 PIECEWISE first tails (2 alone, 6 with decodes), 114 FULL multi-request decodes, 8 NONE prefills; collective calls PREFILL 50,192 and DECODE 216,978, equal to the independent counts | PASS |
| 4 | Codex: comparison of the small run with the P2-32 native inputs | JSON and text exist; native counts equal the sums of their counter rows | 6 of 6 checks PASS; 408 R1 profiler-window forwards | PASS |
| 5 | Codex: dummy smoke matrix | 12 of 12 | 12 of 12 | PASS |
| 6 | Lead: the four-file suite in the B21 tree after the cleanup | all pass | 74 passed in 5.7 s | PASS |
| 7 | Lead: on main after the merge, the four files plus the memo, A6 tool and predictor tests | all pass | 478 passed in 12.9 s | PASS |

Collective shares of the small run: every curve key (`all_gatherv/ep/16`, `reduce_scatterv/ep/16`, `all_reduce/tp/8`, `logits_all_gather/tp/8`, `sp_all_gather/tp/8`, `sp_reduce_scatter/tp/8`) is 100% exact in calls and in predicted time. This does not show which payloads the full replays request; the A10 records will.

Step-class times of the small run against the native P2-32 profiler window are in Codex's report (for example DECODE batch 1: Frontier median 19.43 ms against native 18.33 ms; PREFILL real: 245.2 ms against 307.4 ms). They compare different request mixes and are not an A11 reading.

## Official A10 runs started

On main `d53bab0b` at 15:41, with `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_a10.sh <case> <name>` (`a10_driver.py <case> <root>` with the observer, `ulimit -v 62914560`, timeout 86,400 s): run roots `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_32_cold` and `p2_64_cold`. The warm A9 partners start from a copy of each cold cache once the cold run reports INITIALIZED.
