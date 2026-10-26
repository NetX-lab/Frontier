# Test Report: Step-5 allgather_reducescatter collective pricing (calibration item 2a)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Unit, dummy smoke and fidelity checks of `0349c1c6`; short-trace diagnostic with the H800 curves. |
| 2026-10-26 | EP-lane prediction cache checks (`eddb9bf5`, 1.7× faster measured without cProfile) and the merge into `feat/step5-preview`. |

## Scope

Calibration item 2a, steps 2–5 (`plan.md`). Branch `feat/step5-d31-measured-comm`
(worktree `/data/ycfeng/Frontier/.worktrees/step5-d31-measured-comm`):

- `b9cbda0d`: the measured collective backend (`--cc_backend_config_type measured`) and the H800
  TP8/DP2 and TP8/DP4 curves.
- `0349c1c6`: the opt-in `--replica_config_moe_all2all_backend allgather_reducescatter` path.
  Dispatch is priced as a DP-group AllGatherV and combine as a ReduceScatterV, each from every
  rank's own row count. The routed TP all-reduce is added, and a separate shared-expert TP
  all-reduce is added when DP > 1.

Environment: local host, CPU only. Python 3.10.6
(`/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python`).

## 1. Unit tests

Command (from the measured-comm worktree):

```bash
ulimit -v 62914560 && PYTHONPATH=$PWD timeout 1800 /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -m pytest \
  tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_ep_wave_trace_context.py \
  tests/unit/test_expert_parallel.py tests/unit/test_measured_cc_backend.py \
  tests/unit/test_moe_ep_aggregate_admission.py tests/unit/test_moe_ep_sampling_domain.py \
  tests/unit/test_moe_ep_workload_materializer.py tests/unit/test_moe_position_routing.py \
  tests/unit/test_moe_routing_conservation.py tests/unit/test_prefill_ep_wave_materialization.py \
  tests/unit/test_shared_expert_parallel_semantics.py tests/unit/test_stage_reporting_contract.py \
  tests/unit/test_step5_mrv2_forward.py tests/unit/test_vllm_forward_collectives.py \
  tests/unit/test_moe_all2all_backend.py tests/unit/test_step5_moe_pricing.py \
  tests/unit/test_step5_layer_metadata.py \
  -q -p no:cacheprovider --continue-on-collection-errors --basetemp /data/ycfeng/tmp/d31_pytest/agrs3
```

- Expected: every test passes. The new `test_moe_all2all_backend.py` (9 tests) covers config
  validation, the DP forward shape, the per-lane token count, the dispatch payload per layer
  precision, and the two added TP all-reduces.
- Actual: `539 passed, 2 skipped in 27.45s`. **PASS**.

## 2. Dummy smoke matrix

Command: `PATH=<runtime>/bin:$PATH PYTHONPATH=$PWD bash examples/architecture/run_dummy_smoke_matrix.sh`.
Log: `/data/ycfeng/tmp/d31_agrs_fidelity/dummy_smoke.log`.

- Expected: all 12 cases complete.
- Actual: `Completed 12 profiling-independent dummy smoke case(s).` **PASS**.

## 3. Fidelity matrix (default path unchanged)

Both sides ran from new detached worktrees with empty caches:

```bash
run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-baseline-7dce66c  --label baseline  --output-root /data/ycfeng/tmp/d31_agrs_fidelity --python-bin <runtime python> --jobs 8 --clean-cache
run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-candidate-0349c1c --label candidate --output-root /data/ycfeng/tmp/d31_agrs_fidelity --python-bin <runtime python> --jobs 8 --clean-cache
run_matrix.py compare --output-root /data/ycfeng/tmp/d31_agrs_fidelity
```

- Expected: all 77 cases identical, because the new path is opt-in.
- Actual: 77/77 identical; 0 mismatched; 0 failures on either side; 0 differing predictor
  cache file names. Report: `/data/ycfeng/tmp/d31_agrs_fidelity/comparison.json`. **PASS**.

## 4. Short-trace diagnostic (H800 curves used in place of H200)

Purpose: confirm that the new path runs end to end on Step-5-Preview with the `h200_32`
topology (PDD, TP8/DP2/EP16 per role, merged03 H200 operator profiles). It also gives the size of
the added communication and the simulator's cost. The H200 curves were still being measured, so
this run used the H800 curves. Its numbers do not measure accuracy.

- Trace `/data/ycfeng/tmp/d31_agrs_diag/trace_tiny.csv`: 2 requests, 6,144 prompt tokens and
  32 output tokens each, arriving 1 s apart.
- Commands: `/data/ycfeng/tmp/d31_agrs_diag/tiny_{base,agrs_h800}.command.json`, run by
  `/data/ycfeng/tmp/d31_agrs_diag/run_one.py <label> trace_tiny.csv profile` under cProfile.
- `tiny_base`: `--cc_backend_config_type analytical` (the current default path).
- `tiny_agrs_h800`: `--cc_backend_config_type measured`,
  `--measured_cc_backend_config_summary_csv_path .../h800_dgx/step5_tp8_dp2/summary.csv`,
  `--replica_config_moe_all2all_backend allgather_reducescatter`. The measured backend also
  prices every TP all-reduce from the measured curves.

| Run | Return code | Wall time (s) | TTFT mean (ms) | TPOT mean (ms) | E2E mean (ms) |
| --- | --- | --- | --- | --- | --- |
| `tiny_base` | 0 | 577.5 | 580.3 | 20.07 | 1202.6 |
| `tiny_agrs_h800` | 0 | 594.1 | 874.0 | 32.24 | 1873.6 |
| Difference | | +16.6 (+2.9%) | +293.7 | +12.17 | +671.0 |

Observations:

- No `MeasuredCCOutOfRangeError`: every collective call was inside the measured byte range.
- The H800 curves produce benign "Non-monotone measured collective" warnings. These are
  neighbouring medians that differ by about 1–2 µs.
- TPOT rises by about 12 ms per decode step. Each step now prices 88 dispatches, 88 combines
  and two TP all-reduces per MoE layer, all from measured curves. At about 30–40 µs per small
  call, 88 layers add roughly 10–14 ms, so the size is consistent.

Inference, not yet verified: whether the larger TPOT is closer to native can only be judged on
the full workload (step E rerun with the H200 curves). The native `h200_32` means are TTFT
1083.7 ms and TPOT 61.7 ms (`evidence/d31_h200_32_native_metrics.json`).

### Simulator cost

The cProfile output (`tiny_agrs_h800.prof`) shows the following:

- 538 s of 593 s are in the event loop, and 437 s of those are in EP-wave preparation:
  - 5,808 MoE waves;
  - 111,746 single-layer lane predictions, about 3.3 ms each (`_get_layer_execution_time`).
- The rest is spread across Python work: operator lookups (1.4 M calls, 120 s), `Request`
  construction in lane batches (2.0 M calls, 34 s), and repeated Step-5 config validation
  (26,271 calls, 54 s).
- The new path itself costs about 3% more wall time.

The full `h200_32` workload (step E) had run for more than 5 h 47 min at 17:07 UTC without
finishing. Simulator speed therefore limits how fast calibration can iterate.

## 5. EP-lane prediction cache (`eddb9bf5`)

Change: `predict_ep_wave_phase_times` predicts each distinct `(local expert loads, pre-routing
token count)` pair of a wave once. Lanes with a native routing source are still predicted one by
one; the DSV4 per-lane source path is unchanged.

Exactness check: `/data/ycfeng/tmp/d31_lane_memo/check_memo.py` wraps the wave function, predicts
every lane again without the cache and compares the six phase times exactly. It ran on the two
short traces of section 4 (commands `/data/ycfeng/tmp/d31_lane_memo/tiny_*_check.command.json`).

| Trace | Waves | Lanes | Distinct predictions | Mismatches |
| --- | --- | --- | --- | --- |
| `tiny_base` | 5,808 | 92,928 | 11,968 | 0 |
| `tiny_agrs_h800` | 5,808 | 92,928 | 12,144 | 0 |

Result equality (`/data/ycfeng/tmp/d31_lane_memo/compare_exact.py`): the cached runs'
`request_metrics.csv` and every section of `system_metrics.json` equal the section 4 results of
both traces.

Wall time. The section 4 runs ran under cProfile, so they are not a fair baseline. Both versions
were therefore run side by side without a profiler (`/data/ycfeng/tmp/d31_lane_memo/launch_timing.py`;
uncached `1c125189` in `.worktrees/run-agrs-1c125189`, cached `eddb9bf5`). The cached and uncached
results are equal again.

| Trace | Uncached (s) | Cached (s) | Speedup |
| --- | --- | --- | --- |
| `tiny_base` | 329.6 | 195.7 | 1.68× |
| `tiny_agrs_h800` | 342.6 | 196.7 | 1.74× |

The commit message of `eddb9bf5` compares 577.5/594.1 s (under cProfile) with 203.1/209.4 s
(without). That 2.84× overstates the gain; the table above is the correct measurement.

Unit tests: the section 1 command on `eddb9bf5`: `540 passed, 2 skipped`. The new
`test_ep_wave_predicts_lanes_with_equal_inputs_once` covers equal inputs, a different load, a
different pre-routing count and a native routing source. **PASS**.

Fidelity: `run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-candidate-eddb9bf
--label candidate_memo --output-root /data/ycfeng/tmp/d31_agrs_fidelity --jobs 8 --clean-cache`,
then `compare --baseline-label baseline --candidate-label candidate_memo` against the section 3
baseline (`7dce66c0`). Result: 77/77 identical, 0 mismatched, 0 failures, 0 differing predictor
cache file names. Report `/data/ycfeng/tmp/d31_agrs_fidelity/comparison_memo.json`. **PASS**.

## 6. Merge into `feat/step5-preview`

`2acbf30a` merges `feat/step5-d31-workflow` (calibration item 1) and `b3fa995a` merges
`feat/step5-d31-measured-comm` (item 2a with the H200 curves, the runner option and the cache).
Both merges had no conflicts. Unit tests on `b3fa995a`, the section 1 files plus the workflow
files (`test_nixl_read_completion.py`, `test_decode_output_placeholders.py`,
`test_vllm_v1_max_model_len.py`, `test_grouped_kv_cache_scheduler.py`,
`test_pdaf_decode_attn_online_cohort.py`, `test_module_split_boundaries.py`):
`680 passed, 2 skipped`. **PASS**.

## Verification limits

- The H200 curves were not yet installed. The short-trace numbers use H800 curves and two
  requests; they show only that the new path runs and how large its additions are.
- No native measurement exists for the short trace, so no accuracy is claimed.
