## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-11 | Recorded the completed H200 local MoE stage diagnostic capture and its first-formal RCA evidence. |

# Test Script Information

- Worker: `tests/e2e/issue26_h200_completion_capture_worker.sh`
- Standard chain: `issue26_h200_replay_worker.sh -> issue26_h200_uniform_groundtruth_worker.sh -> issue26_h200_diagnostics_worker.sh batch -> issue26_token_id_client.py`; identity validation was run afterward with `tests/e2e/issue26_diagnostic_identity_analysis.py --mode batch`.
- RJob: `yc26-h200-completion-local-stage-20260911-02`
- Launch artifact: `analysis/h200-completion-local-stage-02/launch.sh`
- H200 allocation: `step_main`, `h200`, 8 GPUs, 64 CPUs, 400 GiB, `num_gpu_blocks_override=310809`.
- Diagnostic vLLM checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab`, commit `eb4c9a1394ef136f134b5a36538847ec01679c68`.
- GPU environment: Python 3.10.16, Torch 2.8.0+cu128, CUDA 12.8.93, FlashInfer 0.3.0.

# Validation Criteria

The run was admissible only after three fully drained 100-request warmup replays and a 100-request formal replay completed. The client artifact had to contain 400 rows and the identity validator had to report `PASS`. The selected boundary had to be request `cmpl-pf4096_dc1024:0-0`, one request, 4096 prefill tokens, zero decode tokens, DP0 TP0–TP3, and `batch_dp_token_counts=[4096,1]`.

The diagnostic source added asynchronous CUDA events around the non-chunked `quant_method.apply()` path (`phase=local_moe_apply`) and preserved the existing `combine` and `tp_ar` completion rows. It added no per-collective synchronization and did not modify Frontier production code. The result remained diagnostic-only because each forward flush synchronizes once and writes completion JSONL.

# Test Results and Evidence

Status: `PASS_DIAGNOSTIC_CAPTURE_LOCAL_STAGE`.

- RJob phase: `Succeeded`; worker completed normally.
- Warmups: replay 0, 1, and 2 each completed 100 requests and drained before the next replay.
- Formal phase: 100 unique requests completed; `client.jsonl` contains 400 rows.
- Identity validator: `PASS`; DP0 TP0–TP3 had complete formal batch logs.
- First formal batch: `batch_id=3880`, request `cmpl-pf4096_dc1024:0-0`, `batch_size=1`, `batch_num_tokens=4096`, `batch_num_prefill_tokens=4096`, `batch_num_decode_tokens=0`, `request_num_tokens=[4096]`, `batch_dp_token_counts=[4096,1]`.
- Clean comparison boundary is unavailable in this run because completion instrumentation was enabled. The recorded DP0 TP0 outer span was `90.690338135 ms`; this value is diagnostic and is not substituted for a clean span.

The selected local stage contained 48 rows per rank, all with shape `[4097,2048]`, `torch.bfloat16`, and 8,390,656 elements:

| DP0 TP rank | local CUDA sum (ms) | local median (ms/layer) | local P90 (ms) | local max (ms) | local→combine CUDA gap median (ms) | local→combine host gap median (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 27.315488 | 0.525776 | 0.759235 | 1.001888 | 0.002912 | 0.030232 |
| 1 | 47.901280 | 0.993168 | 1.016387 | 1.114208 | 0.003008 | 0.032006 |
| 2 | 33.951136 | 0.664320 | 0.980876 | 1.018016 | 0.002944 | 0.030880 |
| 3 | 29.063264 | 0.526864 | 0.903171 | 1.002176 | 0.002944 | 0.031338 |

TP1 was the latest local-stage end on 47/48 layers and the latest combine start on 47/48 layers. Its local stage was 14.0–20.6 ms longer in aggregate than the other ranks, while the subsequent local-to-combine gap remained approximately 0.003 ms on the CUDA stream and 0.03 ms on the host. This is direct evidence that the late arrival in this run originates inside the local MoE `quant_method.apply()` scope; it is not explained by post-MoE AR duration or by a host delay between local completion and combine submission.

The paired completion values also show the expected compensation: TP1's post-MoE TP AR sum was only `4.770560 ms`, while TP0/TP2/TP3 were `30.354400/23.484864/28.319840 ms`, because earlier ranks waited inside the inclusive AR event scope. Rank-local AR sums therefore remain non-additive.

# Limits

- The local scope does not decompose the kernels inside `quant_method.apply()` and cannot distinguish an expert-shard kernel-duration difference from internal queued work.
- Uniform routing and identical `[4097,2048]` boundary shapes establish equal outer input shape, but this run did not persist per-expert token counts. A remaining expert-shard or GPU execution explanation requires a routing-count or kernel-level follow-up.
- CUDA event intervals and host timestamps do not prove a specific NCCL or CPU mechanism. The run does not close the Frontier-vLLM operator gate, clean/diagnostic reconciliation, or production correction.
