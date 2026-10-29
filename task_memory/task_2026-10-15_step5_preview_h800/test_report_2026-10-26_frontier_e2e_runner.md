# Test report: Frontier E2E workload runner (D31 step B) — 2026-10-26

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | First version: runner smoke, short timing runs, full dummy workload and diagnostics-off equality. |

## Scope

D31 step B asks for a Frontier E2E run of the frozen workload that finishes in practical time, with
the diagnostic outputs off and request results unchanged by turning them off.

Runner: `tests/integration/step5_pd/frontier_workload.py` (`run` and `compare`).

Environment:

- Simulator: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python` (Python 3.10.6),
  local host `kun-workspace-vgen2`, `OMP_NUM_THREADS=1`.
- Source: commit `e275c83a` plus the uncommitted runner (committed with this report).
- Trace: `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/trace.csv`
  (128 requests, 6144 input and 1024 output tokens, QPS 1).
- Case `h200_32`: one P and one D replica, each TP8/DP2/EP16, `vllm_load_balancing` v0.28.0,
  `balanced` routing (D31.9), analytical CC, NIXL KV transfer, FULL_AND_PIECEWISE graphs with
  capture sizes up to 256, dummy predictor.

## Commands

```bash
cd /data/ycfeng/Frontier/.worktrees/step5-preview
PY=/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python
$PY tests/integration/step5_pd/frontier_workload.py run --case h200_32 \
  --trace /data/ycfeng/tmp/d31_frontier_b/requests_2x4.csv \
  --output-dir /data/ycfeng/tmp/d31_frontier_b/runner_smoke1
$PY tests/integration/step5_pd/frontier_workload.py run --case h200_32 \
  --output-dir /data/ycfeng/tmp/d31_frontier_b/full_dummy_h200_32
```

The short timing runs `prof1`, `prof2`, `w1x64` and `w32x16` used equivalent hand-written
commands (`cmd.sh` in each directory) before the runner existed; `prof1` keeps the stage-batch
ledger on, the others turn it off.

## Results

| Check | Expectation | Actual | Result |
| --- | --- | --- | --- |
| Runner smoke (2 requests × 4 output tokens) | Return code 0, 2 formal requests, summary written | rc 0, 2 requests, wall 72.8 s | PASS |
| Full workload, dummy mode | Return code 0, 128 of 128 requests | rc 0, 128 requests, wall 2933 s | PASS |
| Ledger on vs off, 1 request × 64 output tokens | Identical `request_metrics.csv` | SHA-256 prefix `3a714ead6aba3aa7` for `prof1` (on), `prof2` and `w1x64` (off) | PASS |
| Single-request wall time | Measured | 147 s (`w1x64`, 1 × 64) | measured |
| Short multi-request wall time | Measured | 121 s (`w32x16`, 32 × 16) | measured |
| cProfile total, 1 × 64 | Diagnostics off is faster | 366 s on, 248 s off | measured |

Output directories are under `/data/ycfeng/tmp/d31_frontier_b/`.

## Limits

- Dummy mode gives placeholder latencies (for example TPOT 1622 ms), so these runs check
  execution and run time only, not accuracy.
- The 2933 s full-workload time is for dummy mode. Measured-profile runs add predictor training
  and lookups; their time is measured in step E.
- `compare` was checked against a synthetic native file built from the full dummy run's own
  metrics, with total throughput raised by 20%
  (`/data/ycfeng/tmp/d31_frontier_b/compare_check/`): the 13 equal metrics reported +0.0% PASS,
  total throughput reported −16.7% FAIL, and the exit code was 1. The field names match
  `workload_metrics.py` (`repetitions`, `mean_total_thpt_tokens_s`, `mean_output_thpt_tokens_s`).
  Real native metrics for `h200_32` are not yet available (step C running).
