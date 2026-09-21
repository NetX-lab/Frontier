## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-13 | Recorded the second complete H200 PPLX clean boundary replay (run03). |

# H200 PPLX clean first-formal boundary — run03

## Execution

- RJob: `yc26-h200-pplx-clean-20260913-03` (`Succeeded` / `RJobSucceeded`).
- Worker: `tests/e2e/issue26_h200_pplx_clean_boundary_worker.sh`.
- Output root: `analysis/pplx-clean-20260913-run03/`.
- The worker invoked:

  ```bash
  python tests/e2e/issue26_pplx_boundary_analysis.py \
    --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-20260913-run03/runtime \
    --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-20260913-run03/runtime/boundary_analysis.json \
    --warmups 10
  ```

- Hardware/allocation: H200 `step_main`, selector `h200`, eight GPUs,
  `num_gpu_blocks_override=310809`; preflight records eight H200 devices and
  NV18 peer links.
- Workload: Qwen3-30B-A3B dummy, TP4/DP2/EP8/PP1, BF16, eager,
  FLASHINFER, uniform routing, prefix caching and chunked prefill OFF,
  4096-prefill/1024-output.
- vLLM source commit: `448f2b65e7679ae7490114ad382b6ba79becb3c3` with an empty
  preflight patch. Backend is `VLLM_ALL2ALL_BACKEND=pplx` and
  `VLLM_MOE_DP_CHUNK_SIZE=4096`.
- Frontier per-op, batch, scheduler, routing, and full diagnostic logging are
  disabled. The boundary probe creates one CUDA event pair around the first
  formal model forward and performs one post-forward synchronization solely to
  read elapsed time.

## Gate and result

`analysis/pplx-clean-20260913-run03/runtime/boundary_analysis.json` reports
`status=PASS`:

- ten contiguous, drained warmup records, each with 100 completed requests;
- 100 unique formal requests and 1100 total client rows;
- first formal identity `pf4096_dc1024:0` / `cmpl-pf4096_dc1024:0-0`;
- one selected DP0 lane with TP0--TP3 rows satisfying batch size 1, 4096
  prefill tokens, and zero decode tokens.

| TP rank | CUDA event elapsed (ms) |
| ------: | ----------------------: |
| TP0     | 182.934814453 |
| TP1     | 182.368316650 |
| TP2     | 183.000671387 |
| TP3     | 182.785339355 |

| Metric | Value (ms) |
| ------ | ---------: |
| Median | 182.860076904 |
| P90 (inclusive) | 182.980914307 |
| Rank maximum | 183.000671387 |
| Rank spread | 0.632354736 |

The run05 replay, which uses the same source and settings, measured median
`183.436271667 ms`; the two medians differ by `0.576194763 ms` (0.31%).
This repeatability supports the conclusion that the PPLX path is consistently
near 183 ms under this configuration.

## Interpretation and limits

Relative to the accepted native/naive clean references
`78.118782043--79.307357788 ms`, this PPLX median is `103.553--104.741 ms`
higher (130.57--133.99% slower). The result is an independent PPLX backend
measurement, not a Frontier correction or a claim about pure collective
duration. The event envelope may include queued device work on the forward
stream and excludes host wall-clock gaps; rank-local envelopes must not be
summed. Operator/collective instrumentation was intentionally disabled, so
this run cannot identify the internal source of the PPLX overhead. The prior
default-chunk PPLX result near 542 ms is a separate protocol setting.
