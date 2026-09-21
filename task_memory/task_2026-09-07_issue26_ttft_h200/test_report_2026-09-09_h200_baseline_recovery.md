## Modification History

| Date       | Summary of Changes                     |
| ---------- | -------------------------------------- |
| 2026-09-09 | Recovered the pre-H800 H200 batch-only recipe and revalidated its first formal span. |

# H200 Baseline Recovery Before H800

## 1. Test Script Information

The historical 70--110 ms result comes from the batch-only diagnostic lane, not the
full operator or record-function lanes. The executable chain is:

```text
tests/e2e/issue26_h200_replay_worker.sh
  -> tests/e2e/issue26_h200_uniform_groundtruth_worker.sh
  -> tests/e2e/issue26_h200_diagnostics_worker.sh ... batch
  -> tests/e2e/issue26_token_id_client.py
  -> tests/e2e/issue26_diagnostic_identity_analysis.py --mode batch
```

The authoritative completed run is:

```text
task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-historical-replay-02/
```

Its H200 launcher uses `step_main`, `--positive-tags=h200`, eight GPUs, 64 CPUs,
400 GiB memory, and image digest
`sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.
The vLLM source is pinned to
`361d941c97fcec52e544f74b7ab91c54192de9c9`, with a clean tree. The GPU runtime is
Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0, and CUDA 12.8.93.

The frozen model/runtime settings are Qwen3-30B-A3B-Instruct-2507 dummy weights,
BF16, TP4/DP2/PP1/EP8, eager execution, FLASHINFER attention, naive all-to-all,
chunked prefill OFF, prefix caching OFF, `max_model_len=16384`,
`max_num_batched_tokens=16384`, `max_num_seqs=1024`, block size 16,
`num_gpu_blocks_override=310809`, seed 0, and
`VLLM_MOE_UNIFORM_ROUTING=1`. The client sends 100 formal requests after three
fully drained 100-request warmup replays, at QPS 2, with 4096 prefill and 1024
decode tokens (`seed=20260908`).

The identity and statistic check was rerun with:

```text
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
  tests/e2e/issue26_diagnostic_identity_analysis.py \
  --run task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-historical-replay-02/batch/runtime/batch \
  --mode batch \
  --output /data/ycfeng/tmp/issue26-historical-replay-02-validation-rerun.json
```

## 2. Validation Criteria

The first formal boundary is selected by request identity, not by the smallest
`batch_id`. It must satisfy all of these predicates:

```text
request_id = cmpl-pf4096_dc1024:0-0
batch_size = 1
batch_num_prefill_tokens = 4096
batch_num_decode_tokens = 0
request_num_tokens = [4096]
```

Only DP0 TP0--TP3 are used for the reported same-lane outer span. Warmup rows,
mixed prefill/decode rows, client TTFT, operator scopes, and record-function
spans are separate metrics and do not enter this baseline.

## 3. Test Results and Evidence

The rerun validator returned `PASS`, with 400/400 client rows, 300 warmup rows,
100 formal IDs, and all eight worker files. The first formal request is in
`batch_id=4769` on DP0:

| TP rank | `batch_execution_time_ms` |
| -------: | -------------------------: |
| TP0 | 77.1250228882 ms |
| TP1 | 77.1454391479 ms |
| TP2 | 76.9347839355 ms |
| TP3 | 77.1880340576 ms |

Derived same-boundary statistics are:

| Metric | Value |
| --- | ---: |
| Median | 77.1352310181 ms |
| P90 (linear percentile over four ranks) | 77.1752555850 ms |
| Rank max | 77.1880340576 ms |
| Rank spread | 0.2532501221 ms |

The client log records four completed 100-request replays, so the first formal
row is after three drained warmups. The result is therefore a direct reproduction
of the documented H200 batch-only scale.

The previously observed 10,000+ ms values came from selecting warmup/startup
rows (for example `batch_id=0`, whose request identity is
`warmup:pf4096_dc1024:r0:0-0`) or from high-intrusion operator instrumentation.
Those rows include initialization/JIT/queue effects and are not the first formal
forward boundary.

The following values remain intentionally separate:

```text
batch-only formal boundary:       76.9348--77.1880 ms (this report)
reduced communication diagnostic: approximately 86.41--86.44 ms
full CUDA-event/operator probe:    approximately 116.21 ms
record-function diagnostic:        approximately 150--153 ms
```

The latter three are instrumented or differently scoped diagnostics and cannot
replace the batch-only baseline.

## 4. Exact-Pin Fresh Run Status

To independently replay the same source commit, an isolated clone at
`/data/ycfeng/tmp/issue26-vllm-diagnostics-historical-361-20260909` was checked
clean at `361d941c97fcec52e544f74b7ab91c54192de9c9`. RJob
`yc26-h200-batch-repro-historical-361-20260909-02` uses the same H200 allocation,
image, and case. At report time its platform phase is `Starting` with the worker
pending because no matching H200 node has capacity; no timing artifact has been
produced by this fresh job. The task-local worker now explicitly exports
`VLLM_MOE_UNIFORM_ROUTING=1` before the mode manifest and server are created.

This pending platform job does not invalidate the completed authoritative run.
It only means that an additional fresh source-pin repetition remains pending.

## 5. Scope Boundary

This report establishes the H200 batch-only reference and its warmup/identity
semantics. It does not claim operator parity, CUDA attribution closure, clean
official TTFT parity, or an E2E gate. H800, `normal/scalar_sync/skip`, post-MoE
RCA, and clean/diagnostic span reconciliation remain paused until YC accepts this
baseline.
