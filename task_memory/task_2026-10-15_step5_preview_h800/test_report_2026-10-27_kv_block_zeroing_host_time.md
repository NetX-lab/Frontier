# Test report: calibration item 3, Frontier side (KV block zeroing host time)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | First version: unit, structural, dummy smoke and 77-case fidelity checks of the item 3 Frontier implementation; regenerated NIXL READ profiles. |

## Scope

Item 3 step 3 of `plan.md` (design `design.md` sections 3 and 8). The change is off by default:
without `kv_block_zeroing_host_profile` every host delay is 0. The measured profile does not
exist yet; it comes from the `host_probe` native run (item 3 step 2).

Source: `feat/step5-preview` at `308495a9` plus the uncommitted item 3 diff (committed after
these checks).

Environment:

- Unit tests: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python` (Python 3.10.6).
- Fidelity matrix and structural runner: `/data/ycfeng/envs/frontier-py310/bin/python`
  (Python 3.10.6, the runtime the structural runner requires).

## Results

| Check | Expectation | Actual | Result |
| --- | --- | --- | --- |
| Focused unit tests (9 files) | all pass | 304 passed in 286 s | PASS |
| 77-case fidelity against `e275c83` | numeric artifacts identical; only config fields differ | 77/77 cases produce artifacts; every difference is in `config.json`: the new field `kv_block_zeroing_host_profile` and the known field `nixl_read_completion` (from item 1), both "missing in baseline" | PASS |
| Dummy smoke matrix | all cases complete | 12 cases, rc 0 | PASS |
| Step-5 structural runner, cache lifetime | modified check passes | passes on all 3 cases with the current weight constants (see below) | PASS |
| Step-5 structural runner, as checked in | — | all 3 cases stop at "source storage oracle differs", before the cache check | pre-existing, not caused by this change |

### Commands

Unit tests:

```bash
PYTHONPATH=$PWD:$PWD/tests/unit /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -m pytest \
  tests/unit/test_kv_block_zeroing_host_time.py tests/unit/test_step5_mrv2_forward.py \
  tests/unit/test_monolithic_mixed_forward_sync.py tests/unit/test_step5_cache_scheduler.py \
  tests/unit/test_step5_measured_pd.py tests/unit/test_grouped_kv_cache_manager.py \
  tests/unit/test_grouped_kv_cache_scheduler.py tests/unit/test_dsv4_cache_zero_driver.py \
  tests/unit/test_dsv4_cache_zero_execution.py \
  -q --basetemp /data/ycfeng/tmp/d31_pytest/kvz_final -p no:cacheprovider -p no:logging
```

Log: `/data/ycfeng/tmp/d31_kvz_unit_final.log`.

Fidelity matrix (script `/data/ycfeng/tmp/d31_fidelity_kvz_run.sh`, 8 jobs):

```bash
PYTHONPATH=$PWD OMP_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python \
  tests/e2e/refactor_fidelity/run_matrix.py run --repo-root $PWD --label candidate \
  --output-root /data/ycfeng/tmp/d31_fidelity_kvz --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 8
```

Comparison against the `e275c83` baseline outputs in `/data/ycfeng/tmp/d31_fidelity_dfi` with
`/data/ycfeng/tmp/d31_fidelity_kvz_recompare.py` (a copy of the earlier recompare script with
the candidate root changed). Output `/data/ycfeng/tmp/d31_fidelity_kvz_recompare.log`:
77 cases, 0 identical, 77 mismatched; the only difference kinds are the two `config.json` fields
above, 77 times each, and no other differences.

Dummy smoke: `bash examples/architecture/run_dummy_smoke_matrix.sh` with `PYTHONPATH=$PWD`,
W&B disabled; log `/data/ycfeng/tmp/d31_kvz_smoke.log` ("Completed 12 profiling-independent
dummy smoke case(s).").

Structural runner:

```bash
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python tests/integration/run_step5_structural.py \
  --output-root /data/ycfeng/tmp/d31_kvz_structural01
```

All three cases stop at the weight-memory check: the runner expects 37,750,873,536 parameters
and `other_memory_bytes` 135,020,544; the run reports 38,079,046,000 and 1,444,147,040 (total
weight bytes 41,909,084,704 match). This change touches no parameter accounting, so the stale
constants predate it (inference from the diff, not a rerun on HEAD). The diagnostic
`/data/ycfeng/tmp/d31_kvz_structural_validate.py` reruns `validate` on the same outputs with
the reported values: all three cases pass every remaining check, including the modified
cache-lifetime check (35 cache-state events, 4,113 KV input tokens, 28,898 pool blocks).

## What the new tests check

- `tests/unit/test_kv_block_zeroing_host_time.py` (27 tests): scalar and table lookup with
  linear interpolation; a request count outside the table raises; 19 malformed or mismatched
  profiles (schema, provenance, model profile, device, graph mode, TP, DP, EP, PP, missing role,
  both or neither value form, short or unordered tables, negative or boolean times, a malformed
  row, missing topology); a DP2 profile on a DP4 run; MONOLITHIC, DECODE_ATTN and missing
  cluster types; CPU-overhead modeling on; only batches with `cache_block_zeroing` pay; a
  dummy-mode predictor rejects the profile.
- `tests/unit/test_step5_mrv2_forward.py`, two new tests:
  - lane arrivals (1, 1) s with a 9 ms reset on lane 0 give an agreed start of 1.009 s; arrivals
    (1, 1.004) give 1.009 s; arrivals (1, 1.02) give 1.02 s (the late lane already covers the
    reset). First-attention sync times are start + 1 ms and start + 5 ms on both lanes; the
    launch start, host delay, stage start (unchanged) and DECODE engine-step time are checked;
  - a dummy peer is delayed by the same 9 ms (stage start stays 1.0 s, launch 1.009 s), and no
    delay without a resetting batch; for graph modes `none` and `full_and_piecewise`.
- Trigger: Step-5 capacity sets `zero_new_blocks`; a scheduled prefill records its new block IDs;
  a NIXL read reservation records none (`test_step5_measured_pd.py`: the D-side prompt tail
  resets only its 2 new pages per group, 8 IDs, not the 1,020 READ-filled pages).

## Behavior changes on existing paths

- Step-5 batches now carry `cache_block_zeroing`. No Step-5 timing consumer exists without the
  profile, so timing is unchanged (fidelity above).
- A NIXL read reservation no longer records its blocks for zeroing, for every model with
  `zero_new_blocks`, which matches native (`scheduler.py:1297-1300`). DeepSeek-V4 prices its
  zeroing as a device operator, so a DeepSeek-V4 PD run with NIXL READs now zeroes only the
  locally allocated blocks. The fidelity matrix has no DeepSeek-V4 PD case; the DeepSeek-V4
  zeroing unit tests pass.
- The NIXL READ profile records the full cache capacity, including `zero_new_blocks`. The two
  H200 profiles were regenerated from the same native logs with the new capacity flag:
  `evidence/d31_h200_32_nixl_read_cost_20261027.json` (51.374646939367565 ms, 94 intervals,
  3,447 READs) and `evidence/d31_h200_64_nixl_read_cost_20261027.json` (56.54669704861111 ms,
  182 intervals, 3,456 READs). The READ values equal the old files; only
  `cache_capacity.zero_new_blocks` changed. The old files stay for the earlier runs. Runs from
  this commit on must use the new files.

## Known failures not caused by this change

7 tests in `tests/unit/test_measurement_family_selector.py` fail (missing
`_uses_output_placeholders`, `int(None)`). They also fail on an archive of `308495a9`
(`/data/ycfeng/tmp/d31_head_src`), so they predate this change. They belong to the deferred
stale-test list.

## Limits

- No measured profile exists yet, so no run with a nonzero host delay beyond the unit tests.
  The E and G reruns with the measured profiles verify the effect on TPOT.
- The DeepSeek-V4 read-reservation change is covered by unit tests only.
