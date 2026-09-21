## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recomputed the native Nsight run-03 category unions after removing the overbroad `collective` attention-name heuristic and added control-window idle accounting. |

# Native/naive Nsight category reclassification addendum

The earlier run-03 capture used `--trace=cuda,nvtx,nccl --nccl-trace=api-coll,gpu`, with all Frontier instrumentation disabled. Its artifact gate passed (ten drained warmups, 100 formal requests, 1100 client rows, complete first-formal capture), but the RJob had a post-client shell failure. The accepted clean native batch-only references remain `78.118782043--79.307357788 ms`.

The original parser classified any operation containing the word `collective` as communication. That incorrectly placed FlashInfer's `CollectiveEpilogue` attention kernel in the communication bucket. The parser was corrected in commit `108c4c48`; only explicit NCCL/all-reduce/broadcast/all-gather/all-to-all/cross-device/P2P names remain communication candidates. The corrected artifact is:

```text
analysis/nsys-profiler-20260913-run03-postprocess/nsys_breakdown_api_window_reclassified.json
```

The API capture window is `100.826970 ms`. Values below are per-device interval unions clipped to that window. `control-window idle` is `window - all_activity_union`, which includes leading/trailing periods with no observed GPU activity; `internal idle` is the prior envelope-only gap. Category unions overlap when kernels from different categories run concurrently, so category values must not be added as a serial total.

| Device | Compute pure-kernel union (ms) | Communication pure-kernel union (ms) | Memory-kernel union (ms) | All-activity union (ms) | Control-window idle (ms) | Internal idle (ms) | Category overlap (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H200 (0) | 26.967 | 80.553 | 2.027 | 98.693 | 2.134 | 0.821 | 7.452 |
| H200 (1) | 26.840 | 74.954 | 2.118 | 93.041 | 7.786 | 0.798 | 7.454 |
| H200 (2) | 26.406 | 74.805 | 2.165 | 97.394 | 3.433 | 2.110 | 4.286 |
| H200 (3) | 40.857 | 50.312 | 2.606 | 67.962 | 32.865 | 3.255 | 22.491 |
| **Median** | **26.903** | **74.880** | **2.142** | **95.217** | **5.610** | **1.465** | **7.453** |

The corrected classification moves about 3.4 ms of FlashInfer attention activity per H200 device from communication to compute. Communication remains dominated by NCCL and vLLM cross-device kernels: `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`, `ncclDevKernel_Broadcast_RING_LL`, and `vllm::cross_device_reduce_1stage`. The largest individual all-reduce/cross-device kernels remain approximately 28 ms, so the communication path is still a material residual candidate.

This is a profiler diagnostic decomposition. The Nsight API window is roughly 21--29% above the clean 78--79 ms reference, and the per-device values do not carry a persisted TP/DP identity. Nsight capture and CUDA profiler stop can change launch placement and drain queued work. The corrected table therefore supports category and queue hypotheses, but does not establish a clean 79 ms pure-kernel budget, authorize a Frontier correction, or justify clean/diagnostic span reconciliation.
