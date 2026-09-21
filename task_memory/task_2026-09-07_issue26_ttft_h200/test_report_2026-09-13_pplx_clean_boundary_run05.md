## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-13 | Recorded the first complete H200 PPLX clean boundary replay (run05). |

# H200 PPLX clean first-formal boundary — run05

## Execution

- RJob: `yc26-h200-pplx-clean-20260913-05`
- Terminal platform result: `Succeeded` (`RJobSucceeded`)
- Worker: `tests/e2e/issue26_h200_pplx_clean_boundary_worker.sh`
- Output root: `analysis/pplx-clean-boundary-20260913-run05/`
- Analyzer: `tests/e2e/issue26_pplx_boundary_analysis.py`
- Analyzer command (executed by the worker):

  ```bash
  python tests/e2e/issue26_pplx_boundary_analysis.py \
    --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-boundary-20260913-run05/runtime \
    --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-boundary-20260913-run05/runtime/boundary_analysis.json \
    --warmups 10
  ```

- Hardware and allocation: H200, `step_main`, selector `h200`, eight GPUs,
  `num_gpu_blocks_override=310809`; the preflight records eight H200 devices
  with 143771 MiB each and NV18 peer links.
- Frozen workload: Qwen3-30B-A3B dummy model, TP4/DP2/EP8/PP1, BF16, eager,
  FLASHINFER, uniform routing, prefix caching OFF, chunked prefill OFF,
  4096-prefill/1024-output.
- vLLM source: `/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt`,
  commit `448f2b65e7679ae7490114ad382b6ba79becb3c3`; the preflight patch is
  empty.
- Communication mode: `VLLM_ALL2ALL_BACKEND=pplx` with
  `VLLM_MOE_DP_CHUNK_SIZE=4096`. This is an experimental PPLX runtime
  setting, not a Frontier production configuration.
- Warmup/formal contract: ten complete drained warmup replays of 100 requests
  each, followed by 100 formal requests; Frontier instrumentation, per-op,
  scheduler, routing, and full diagnostic loggers were disabled. The worker
  records only one independent CUDA event pair for the selected first formal
  model forward and performs one synchronization to read the elapsed time.

## Criteria and evidence

The persisted analyzer output is
`analysis/pplx-clean-boundary-20260913-run05/runtime/boundary_analysis.json`.
It reports `status=PASS` and confirms:

- ten warmup phase records, each with `completed_requests=100` and contiguous
  phase boundaries (full drain before the next phase);
- 100 unique formal requests and exactly 1100 client rows;
- first formal client identity `pf4096_dc1024:0`, server identity
  `cmpl-pf4096_dc1024:0-0`;
- a single selected DP lane (`DP0`) with all TP0--TP3 boundary rows;
- batch size 1, 4096 total/prefill tokens, and zero decode tokens;
- source commit and PPLX backend provenance in every boundary row.

The four independent model-forward CUDA-event envelopes are:

| TP rank | CUDA event elapsed (ms) |
| ------: | ----------------------: |
| TP0     | 183.506790161 |
| TP1     | 183.467132568 |
| TP2     | 183.122406006 |
| TP3     | 183.405410767 |

The derived first-formal statistics are:

| Metric | Value (ms) |
| ------ | ---------: |
| Median | 183.436271667 |
| P90 (inclusive) | 183.494892883 |
| Rank maximum | 183.506790161 |
| Rank spread | 0.384384155 |

For scale context only, the accepted native/naive clean references are
78.118782043 and 79.307357788 ms. The PPLX median is therefore 104.129--105.317
ms higher (131.30--134.82% slower). This comparison is reported as an
independent backend result; values are not combined across implementations.

## Interpretation and limits

The PPLX path is now capability-valid and has a complete clean first-formal
boundary artifact. The 183.436 ms envelope shows that PPLX remains materially
slower than the native/naive reference even with the 4096-token MoE DP chunk
size. It does not establish which PPLX sub-operation causes the overhead: the
boundary-only worker intentionally disables operator, collective, scheduler,
and routing instrumentation. A prior PPLX run using the default 256-token chunk
size measured approximately 542 ms; that historical result and this run are
different PPLX protocol settings and must not be used as a fitted correction.

The event spans are per-rank model-forward envelopes. They may include queued
device work on the forward stream and exclude host wall-clock gaps; they are
not pure kernel sums and must not be added across ranks. This PPLX semantic
variant cannot replace the accepted native clean reference, close the Frontier
CUDA/operator gate, or justify a Frontier predictor/communication/CPU
correction. No production code or clean/diagnostic reconciliation was changed.
