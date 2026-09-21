## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-13 | Recorded the third complete H200 PPLX clean boundary replay (run04). |

# H200 PPLX clean first-formal boundary — run04

## Execution

- RJob: `yc26-h200-pplx-clean-20260913-04`; terminal result `Succeeded` /
  `RJobSucceeded`.
- Worker: `tests/e2e/issue26_h200_pplx_clean_boundary_worker.sh`.
- Output root: `analysis/pplx-clean-boundary-20260913-run04/`.
- Analyzer output: `analysis/pplx-clean-boundary-20260913-run04/runtime/boundary_analysis.json`.
- The worker's analyzer invocation was:

  ```bash
  python tests/e2e/issue26_pplx_boundary_analysis.py \
    --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-boundary-20260913-run04/runtime \
    --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-boundary-20260913-run04/runtime/boundary_analysis.json \
    --warmups 10
  ```

- Configuration is the H200 `step_main + h200 + num_gpu_blocks_override=310809`
  row: eight H200 GPUs, Qwen3-30B-A3B dummy, TP4/DP2/EP8/PP1, BF16, eager,
  FLASHINFER, uniform routing, prefix/chunked caching OFF, and
  4096-prefill/1024-output.
- vLLM source commit: `448f2b65e7679ae7490114ad382b6ba79becb3c3`; PPLX backend
  with `VLLM_MOE_DP_CHUNK_SIZE=4096`. All Frontier and per-op diagnostic
  instrumentation was disabled.

## Gate and measurements

The persisted analyzer reports `status=PASS`: ten drained warmup replays of
100 requests, 100 unique formal requests, exactly 1100 client rows, formal
identity `pf4096_dc1024:0` / `cmpl-pf4096_dc1024:0-0`, one selected DP0 lane,
and TP0--TP3 rows satisfying batch size 1, 4096 prefill tokens, and zero decode
tokens.

| TP rank | CUDA event elapsed (ms) |
| ------: | ----------------------: |
| TP0     | 183.881500244 |
| TP1     | 183.869415283 |
| TP2     | 183.598724365 |
| TP3     | 183.461471558 |

| Metric | Value (ms) |
| ------ | ---------: |
| Median | 183.734069824 |
| P90 (inclusive) | 183.877874756 |
| Rank maximum | 183.881500244 |
| Rank spread | 0.420028687 |

Across the three complete PPLX replays (run03, run05, run04), medians are
`182.860076904`, `183.436271667`, and `183.734069824 ms` (mean
`183.343472799 ms`, range `0.873992920 ms`, relative range `0.48%`). This is a
repeatability check for the backend setting, not a cross-backend aggregate.

## Interpretation and limits

The PPLX implementation is capability-valid and stable near 183 ms for this
workload after setting the 4096-token MoE chunk size. It remains
`104.427--105.615 ms` above the accepted native/naive clean references
`79.307357788` and `78.118782043 ms` (approximately 131.67--135.20% slower).
The event pair measures a per-rank model-forward envelope and may include
queued device work; it excludes host wall-clock gaps and is not a pure kernel
sum. Because the worker disables operator/collective instrumentation, this
run cannot identify the internal PPLX overhead or support a Frontier predictor,
communication, CPU, or reconciliation correction. Historical PPLX runs with
the default 256-token chunk are a different protocol and remain separate.
