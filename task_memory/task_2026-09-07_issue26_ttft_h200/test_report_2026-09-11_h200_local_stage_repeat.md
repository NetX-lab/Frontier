## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-11 | Recorded the second H200 local-MoE completion capture and rank-stability RCA. |

# H200 Local-MoE Completion Capture Repeat

## 1. Test Script Information

- RJob: `yc26-h200-completion-local-stage-20260911-03`
- Cluster: H200 `step_main`, selector `h200`, 8 GPUs, 64 CPU, 400 GiB.
- KV-cache setting: `--num-gpu-blocks-override 310809`.
- Diagnostic source: `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab`, commit `eb4c9a1394ef136f134b5a36538847ec01679c68`.
- Worker entrypoint: `tests/e2e/issue26_h200_completion_capture_worker.sh`.
- Effective chain: `issue26_h200_diagnostics_worker.sh batch` -> `issue26_token_id_client.py` -> `issue26_diagnostic_identity_analysis.py --mode batch`.
- Workload: Qwen3-30B-A3B, 4096 prefill / 1024 decode, TP4/DP2/EP8/PP1, BF16, eager, FlashInfer, uniform routing, prefix caching OFF, chunked prefill OFF.
- Runtime: Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0, NVIDIA H200 driver 570.124.06.
- Reproducible identity command:

```bash
python tests/e2e/issue26_diagnostic_identity_analysis.py \
  --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-completion-local-stage-03/batch/runtime/batch \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-completion-local-stage-03/identity_validation.json \
  --mode batch
```

- Event extraction and summary: `/data/ycfeng/tmp/build_local03_summary.py`; its output is persisted in `analysis/h200-completion-local-stage-03/completion_local_stage_first_formal_summary.json`.

## 2. Validation Criteria

The run is admissible only after three drained 100-request warmup replays, 100 unique formal requests, exactly 400 client rows, identity validator PASS, and the request-specific first formal batch predicates. The RCA compares the full `local_moe_apply` CUDA-event envelope with the following `combine` and `tp_ar` rows for DP0 TP0-TP3, using the persisted `batch_id`.

The formal target is request `cmpl-pf4096_dc1024:0-0`, `batch_id=3882`, batch size 1, 4096 scheduled/prefill tokens, zero decode tokens, `request_num_tokens=[4096]`, and `batch_dp_token_counts=[4096,1]`.

## 3. Test Results and Evidence

### Execution gates

| Gate | Observed result |
| --- | --- |
| RJob | `Succeeded`, worker succeeded=1 |
| Warmup replay 0/1/2 | 100 completed each; all phase barriers drained |
| Formal phase | 100 requests, 100 unique IDs |
| Client rows | 400 |
| Batch boundary logs | 8 (DP0/DP1 × TP0-TP3) |
| Completion logs | 8 (global ranks 0-7) |
| Identity validator | `PASS` |

### DP0 first formal outer spans

| TP rank | Outer batch span (ms) |
| ---: | ---: |
| TP0 | 91.194595337 |
| TP1 | 91.198524475 |
| TP2 | 91.387741089 |
| TP3 | 91.428733826 |
| Rank max | 91.428733826 |
| Rank spread | 0.234138489 |

These values remain in the established H200 70-110 ms diagnostic scale. DP1 first formal batch is a different request (`cmpl-pf4096_dc1024:2-0`, `batch_id=3899`) and is excluded from the DP0 comparison.

### Completion and local-stage evidence

| TP | local CUDA sum (ms) | local median / P90 / max (ms) | combine CUDA sum (ms) | TP AR CUDA sum (ms) | local→combine CUDA median (ms) | local-end→combine-start host median (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP0 | 35.018848 | 0.564512 / 0.998237 / 1.043456 | 3.937664 | 21.707904 | 0.003008 | 0.031404 |
| TP1 | 32.306528 | 0.526208 / 0.988675 / 1.011872 | 4.073664 | 25.163552 | 0.002976 | 0.031819 |
| TP2 | 47.142784 | 0.991264 / 1.007197 / 1.132320 | 3.941408 | 6.539872 | 0.003008 | 0.031907 |
| TP3 | 25.494016 | 0.524272 / 0.528346 / 0.747936 | 4.396160 | 32.188768 | 0.003104 | 0.030914 |

For all four ranks, the target contains 48 `local_moe_apply`, 48 `combine`, and 48 `tp_ar` rows across 48 layers. The local input shape is `[4097, 2048]`, BF16, `numel=8,390,656` for the target local stage.

Layer alignment gives the same critical-path ordering for local-stage end and combine start: TP2 is latest on 37/48 layers, TP0 on 10/48, and TP1 on 1/48. The rank with the latest local stage has the shortest inclusive TP AR sum (TP2: 6.539872 ms), while earlier ranks carry more AR wait (TP3: 32.188768 ms). The local-to-combine intervals stay around 0.003 ms CUDA and 0.031 ms host on every rank.

### RCA and limits

**Evidence:** In this repeat, TP2's local `quant_method.apply()` envelope is 47.142784 ms, 14.835520 ms above TP0 and 21.648768 ms above TP3. TP2 is latest at the local-stage end and combine start on 37/48 layers. Its following TP AR scope is only 6.539872 ms, while earlier ranks record 21.707904-32.188768 ms. The gap immediately after local completion is sub-0.004 ms CUDA and approximately 0.031 ms host on all ranks.

**Inference:** The late participant in this run is explained by rank-local local-MoE device execution or queued device work before combine. The post-MoE TP AR scope is primarily an inclusive rendezvous/wait measurement; summing it across ranks would double count the same collective wait. The late rank changed from TP1 in `-02` to TP2 in `-03`, so the evidence supports per-run local execution variation rather than a fixed TP-rank hardware cause.

**Unknown:** `local_moe_apply` is one envelope around `quant_method.apply()`. This capture does not distinguish grouped-GEMM kernel time, expert token population, stream idle time, host enqueue delay, or NCCL internal work. The diagnostic path performs batch-level synchronization and JSONL writes, so these measurements do not close the clean CUDA gate and do not authorize clean/diagnostic span reconciliation or Frontier production correction.

The first 144 completion rows without `batch_id` are initialization/warmup records and were excluded. Target rows were selected only where the persisted completion metadata contained `batch_id=3882`; no minimum-batch or mixed-batch selection was used.
