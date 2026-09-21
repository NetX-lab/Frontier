## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-09 | Recorded a fresh H200 batch-only reproduction after the diagnostic warmup review. |

# H200 Batch-Only Reproduction

## 1. Test Script Information

- Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`
- H200 worker script: `tests/e2e/issue26_h200_diagnostics_worker.sh`
- Identity validator: `tests/e2e/issue26_diagnostic_identity_analysis.py`
- Launch receipt: `task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-batch-repro-20260909-01/launch.sh`
- RJob: `yc26-h200-batch-repro-20260909-01` (`step_main`, `--positive-tags=h200`, 8 GPUs); terminal phase `Succeeded`
- vLLM checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, commit `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`, clean tree
- Image: `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`
- GPU runtime: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0, CUDA toolkit 12.8.93

The exact worker command was:

```text
bash tests/e2e/issue26_h200_diagnostics_worker.sh \
  task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-batch-repro-20260909-01/runtime \
  batch
```

The worker used the historical H200 batch-only settings: Qwen3-30B-A3B-Instruct-2507 dummy weights, BF16, TP4/DP2/EP8/PP1, `--max-model-len 16384`, `--max-num-batched-tokens 16384`, `--max-num-seqs 1024`, `--num-gpu-blocks-override 310809`, block size 16, eager execution, FLASHINFER attention, naive all-to-all, uniform MoE routing, chunked prefill OFF, and prefix caching OFF. The client ran 3 complete warmup rounds and 100 formal requests at QPS 2 with 4096 prefill and 1024 decode tokens (`seed=20260908`).

The local verification command was:

```text
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_diagnostic_identity_analysis.py \
  --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-batch-repro-20260909-01/runtime/batch \
  --mode batch \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-batch-repro-20260909-01-validation.json
```

The validator used the CPU environment `dev-vidur-v03-hopper-e2e`, Python 3.13.13.

## 2. Validation Criteria

The run is valid only when all of the following hold:

- The client contains exactly 400 rows: 300 warmup rows and 100 formal rows with unique formal IDs.
- Every request has 4096 prompt tokens and 1024 observed completion tokens.
- All eight `(DP, TP, PP)` workers have monotonically increasing, unique batch IDs and valid request/token vectors.
- A first formal prefill row is selected by request identity, with a request token count of 4096, `batch_num_prefill_tokens=4096`, `batch_num_decode_tokens=0`, and no decode token in that row.
- TP0--TP3 in the same DP lane observe the same first formal request and batch ID.
- The reproduced outer batch span returns to the documented H200 scale of roughly 70--110 ms. This metric is the CUDA-event span emitted by the batch logger and remains distinct from clean request TTFT.

## 3. Test Results and Evidence

The identity validator returned `PASS` with 400/400 client rows, 100/100 formal IDs, and all 8 worker files. Three warmup phases completed before the formal phase; the client log records four completed 100-request replays.

The first formal request in DP0 is `cmpl-pf4096_dc1024:0-0`. TP0--TP3 all selected `batch_id=4405` and the exact vector `[4096]`:

| DP | TP | Batch ID | Prefill tokens | Decode tokens | `batch_execution_time_ms` |
| -- | -- | -------: | -------------: | ------------: | -------------------------: |
| 0 | 0 | 4405 | 4096 | 0 | 67.7816619873 |
| 0 | 1 | 4405 | 4096 | 0 | 67.8107833862 |
| 0 | 2 | 4405 | 4096 | 0 | 67.7643508911 |
| 0 | 3 | 4405 | 4096 | 0 | 67.7260131836 |

For this aligned first batch:

- Median: **67.7730064392 ms**
- P90: **67.8020469666 ms**
- Rank max (outer span): **67.8107833862 ms**
- Rank spread: **0.0847702026 ms**

As a stability check over all 48 formal 4096-prefill rows in DP0, the per-rank medians were 71.3563, 71.3968, 71.3625, and 71.4104 ms; the aligned rank-spread median/P90/max were 0.1027/0.2264/0.2929 ms. The corresponding DP0 outer-span median/P90/min/max were 71.4304/80.0553/67.8108/122.7342 ms. The occasional 122 ms row is a later mixed batch containing multiple prefill requests; it is not the first single-request boundary.

The first formal request's client record reports `client_ttft_ms=101.445412` ms. That value includes dispatch and post-forward processing and is intentionally not used as the batch CUDA span.

## Verdict and Limits

**PASS for the H200 batch-only scale and identity semantics.** The fresh first formal boundary is 67.73--67.81 ms across TP0--TP3, and the formal prefill distribution returns to the documented approximately 70--110 ms range. The earlier 17--20 second rows came from diagnostic warmup/initialization or heavily instrumented AR runs, not this batch-only boundary.

This report does not claim operator parity, CUDA attribution closure, clean TTFT parity, or an E2E gate. The worker still enables scheduler/batch diagnostics and the emitted CUDA-event span can include host-induced device gaps. Subsequent normal/skip or clean/diagnostic reconciliation must use a separate run and preserve this baseline.
