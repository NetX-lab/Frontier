## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the completed H200 naive Kineto run-03 audit, decomposition, and post-client harness failure. |

# H200 Native/Naive Kineto Diagnostic — Run 03

## Outcome

The H200 native/naive profiling run produced a complete formal artifact set after ten drained warmup replays. The worker process was marked failed only after the client and traces had been written: the worker had an invalid Bash arithmetic expression in its final row-count check. That expression is fixed and committed in `f1f5ba74`; the run-03 result remains an artifact-complete diagnostic, while the platform job status remains `Failed`.

The selected formal Kineto marker window is substantially slower than the accepted clean native reference. It must therefore be used for kernel/category attribution only:

- accepted clean native batch-only reference: `78.118782043--79.307357788 ms`;
- run-03 Kineto formal marker span: `110.268503906--114.339026367 ms` across DP0 TP0--TP3;
- DP0 median: `112.731364746 ms`;
- DP0 P90: `114.025485645 ms`;
- rank maximum: `114.339026367 ms`;
- rank spread: `4.070522461 ms`.

This is profiler-perturbed diagnostic evidence. It does not replace the clean reference and does not authorize a Frontier timing correction or clean/diagnostic reconciliation.

## Execution

- RJob: `yc26-h200-naive-kineto-20260913-03`
- RJob terminal phase: `Failed`
- Failure stage: worker teardown/validation after the client completed
- Failure evidence: `tests/e2e/issue26_h200_naive_profiler_worker.sh:109` reported a Bash arithmetic syntax error in the post-client row-count expression.
- Exact worker: `tests/e2e/issue26_h200_naive_profiler_worker.sh`
- Exact client: `tests/e2e/issue26_naive_profiler_client.py`
- vLLM source: `/data/ycfeng/tmp/vLLM-BS`
- vLLM source commit: `46f7b179fd3bf42b9616dc4670cba419afdb2085`
- Fixed worker commit: `f1f5ba74`
- Persistent output: `analysis/naive-profiler-20260913/run-03/`
- Independent audit: `analysis/naive-profiler-20260913/run-03/run-03_gate_audit.json`
- Parser output: `analysis/naive-profiler-20260913/run-03_kineto_breakdown_formal.json`

The worker command used the standard H200 recipe:

```text
H200 / step_main / h200 / 8 GPUs / num_gpu_blocks_override=310809
Qwen3-30B-A3B dummy weights
TP4 / DP2 / PP1 / EP8
BF16 / FLASHINFER / eager
uniform routing
prefix caching OFF
chunked prefill OFF
4096-prefill / 1024-output
VLLM_ALL2ALL_BACKEND=naive
VLLM_FRONTIER_INSTRUMENTATION=0
```

The client executed ten complete drained warmup replays of 100 requests each, followed by 100 formal requests. The exact client command embedded in the worker was:

```bash
"$PY" "$WORKERS/issue26_naive_profiler_client.py" \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --row pf4096_dc1024 --prefill-tokens 4096 --decode-tokens 1024 \
  --requests 100 --warmups "${ISSUE26_WARMUPS:-10}" --qps 2 --seed 20260908 \
  --output "$RUN_ROOT/runtime/client.jsonl"
```

No Frontier batch/operator/scheduler/routing/CUDA-event logger was enabled. The only profiler was vLLM's built-in `torch.profiler` / Kineto, started after warmup:9 drained and stopped from the first formal request's first-token callback.

## Gate evidence

The independent audit reports:

| Criterion | Observed | Result |
| --- | ---: | --- |
| Warmup replays | 10 | PASS |
| Warmup completions | 100 each | PASS |
| Formal requests | 100 | PASS |
| Client rows | 1100 | PASS |
| Unique request identities | 1100 | PASS |
| Target formal request | `pf4096_dc1024:0` | PASS |
| First formal first-token event | present | PASS |
| GPU traces | 8 | PASS |
| Additional async CPU trace | 1 | diagnostic |
| DP0 TP0--TP3 traces | complete | PASS |
| First formal marker group | 48 attention + 48 MoE markers per selected trace | PASS |
| RJob worker exit | non-zero | FAIL (post-client shell check only) |

The client record for `pf4096_dc1024:0` reports prompt 4096, observed completion 1024, response ID `cmpl-pf4096_dc1024:0`, and client TTFT `142.335657 ms`. The client TTFT is also profiler-perturbed and is not used as the clean batch metric.

## Formal-window decomposition

The parser selected the complete 48-layer formal marker envelope for DP0 TP0--TP3. Values are per-rank interval unions inside that marker window; durations are in milliseconds.

| Rank | Marker span | Compute pure union | Communication pure union | Memory pure union | All-kernel union | Idle/non-kernel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP0 TP0 | 113.293891 | 28.769890 | 61.339904 | 3.792161 | 93.901955 | 19.391936 |
| DP0 TP1 | 110.268504 | 27.821732 | 71.663575 | 3.674188 | 103.159496 | 7.109008 |
| DP0 TP2 | 114.339026 | 28.975674 | 22.931490 | 3.908914 | 55.816078 | 58.522948 |
| DP0 TP3 | 112.168839 | 28.391640 | 72.972239 | 3.720315 | 105.084194 | 7.084645 |

For the four ranks, the median category unions are:

- compute: `28.580764648 ms`;
- communication: `66.501739746 ms`;
- memory: `3.756238281 ms`;
- all-kernel union: `98.530725586 ms`;
- idle/non-kernel: `13.250471680 ms`.

In this selected window, the parser found zero cross-category interval overlap. The displayed category unions therefore add to the all-kernel union for each rank, and all-kernel union plus idle/non-kernel equals that rank's marker span. This is a per-rank interval accounting identity; it is not permission to sum rank-local communication values across TP ranks.

## Direct trace audit

The eight GPU traces contain H200 device metadata with compute capability 9.0, CUDA 12.8 runtime/driver metadata, and CUPTI 26. Every GPU trace has `distributedInfo.backend=nccl`, `world_size=8`, and NCCL `2.27.3`. The selected DP0 traces expose ranks 0--3 and process labels:

```text
VLLM::Worker_DP0_TP0_EP0
VLLM::Worker_DP0_TP1_EP1
VLLM::Worker_DP0_TP2_EP2
VLLM::Worker_DP0_TP3_EP3
```

The selected trace activity includes:

- `vllm::unified_attention_with_output`: 144 occurrences (48 layers × 3 marker events);
- `vllm::moe_forward`: 144 occurrences;
- `nccl:broadcast`: 1152 records;
- `ncclDevKernel_Broadcast_RING_LL`: 576 records;
- `ncclDevKernel_AllReduce_Sum_bf16_RING_LL`: 241 records;
- `_C_custom_ar::all_reduce`: 194 records;
- `void vllm::cross_device_reduce_1stage<...>`: 194 records;
- `gpu_memcpy` activity: 653 records;
- `gpu_memset` activity: 2 records.

The communication classification includes NCCL broadcasts, all-reduces, cross-device reductions and related collective kernels. The memory classification includes memcpy/memset and copy/cast/transpose/index activity. Compute is the remaining GPU-kernel class. CPU launch records and asynchronous stream work are not treated as additional serial latency.

## Interpretation and limits

**Observed:** run-03 is approximately 33.4--36.2 ms slower than the accepted clean reference at the DP0 median/max level. All four ranks execute complete marker groups and expose the same distributed world size; the run does not show a missing 20 ms operation or a rank-specific reduction in operation count.

**Observed:** rank-local communication and idle values are complementary. TP2 has the smallest communication union (`22.931490 ms`) and the largest idle/non-kernel interval (`58.522948 ms`), while TP1/TP3 have communication unions near 72 ms and idle near 7 ms. Run-02 showed the same type of rank-dependent redistribution on a different TP rank.

**Inference:** the category spread is primarily a marker-window/stream-queue placement effect under Kineto capture. A collective's participant wait and queued device work can be represented partly in a collective kernel interval and partly in the non-kernel gap, depending on the rank's local event alignment. The evidence does not support adding the four communication unions as serial work or claiming that TP2 omitted operations.

**Limit:** Kineto records and the marker callbacks perturb launch scheduling and extend the observed window. The selected span includes queued device work and event-boundary effects. It does not provide a clean, no-profiler split of host launch delay versus device idle. A low-perturbation CUPTI/Nsight measurement would be required for that separation.

## Relation to the PPLX lane

PPLX remains a separate fused-protocol experiment. Its repaired first-formal boundary artifact is `542.338012695 ms` median on the observed DP1 lane, with the canonical DP0 lane absent. That result is not mixed into this native/naive decomposition and does not justify a Frontier correction.

