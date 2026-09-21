## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated fresh eight-rank primitive measurements; retained the ideal EP abstraction under YC's latest decision; prepared independent multi-size physical calibration with heldout validation. |

# D019 communication measurements

**Measurement validation PASS: 8 ranks, 64 operation rows, 448 positive finite samples. Numerical communication calibration remains open.** All recorded numerical dispatch/combine checks pass, exact groups/message sizes match, and each stored median/minimum/maximum was recomputed from raw samples. The source frozen in the GPU run matched the benchmark at commit `08f16e58`; that verified baseline was committed before the subsequent sweep extension.

YC's current decision supersedes the optional protocol implementation proposal in `d019-communication.md`: **retain Frontier's current ideal EP abstraction**. Naive DP/TP decomposition is diagnostic evidence and optional future documentation. No protocol selector, broadcast implementation, physical-DP shared schema or extra forward phase will be implemented in this step.

## Runtime and execution

- Source: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-profiles-02/runtime/communication/rank_0.json` through `rank_7.json`.
- Root-managed job: `yc26-h200-d019-profiles-20260908-02`, one dedicated eight-H200 `step_main` allocation; communication ran serially after the other profile mode.
- GPU Python: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Python 3.10.16, Torch 2.8.0+cu128, CUDA 12.8, `NVIDIA H200` on all ranks.
- vLLM commit: `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`. The reviewed CUDA communicator, custom-AR, PyNCCL and naive manager files have no diff from the earlier source audit at `8453dd342c6aa2721aaf4b410998aab2f38bc2ec`.
- Exact launch and environment recipes: `analysis/h200-d019-profiles-02/launch.sh` and frozen `issue26_h200_d019_profiles_worker.sh`. The launch uses the approved image digest `sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.

The executed command inside that worker was:

```bash
/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  tests/performance/issue26_h200_collective_microbenchmark.py \
  --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-profiles-02/runtime/communication
```

`PYTHONPATH` selected the diagnostic vLLM checkout; `VLLM_ALL2ALL_BACKEND=naive`; instrumentation was off. Existing platform NCCL settings were preserved, including `NCCL_P2P_DISABLE=0`, `NCCL_NVLS_ENABLE=0` and `NCCL_PXN_DISABLE=1`. No `NCCL_ALGO` or `NCCL_PROTO` override was added. The rank artifacts retain the full NCCL environment.

## Identity and implementation selection

TP groups are `[0,1,2,3]` and `[4,5,6,7]`; DP groups are `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`; EP is all eight ranks. BF16 local hidden bytes are `[16,777,216,4,096]`, corresponding to real4096 and dummy1 rows of hidden2048. Global hidden bytes are16,781,312. Local router bytes are1,048,576 and256; global router bytes1,048,832.

Ranks0–3 recorded `tp_custom_eligible=false` and `tp_runtime_selected=pynccl_all_reduce`; ranks4–7 recorded `true` and `custom_all_reduce`. Every rank recorded a live custom communicator with the8-MiB maximum, enabled PyNCCL and no symmetric-memory communicator. This directly verifies the source-based selection correction; custom-enabled configuration is not a sufficient algorithm identity.

## All measured operation medians

Values below are **microseconds per call**, with each rank's median of seven48-call CUDA-event blocks after20 warmups. Columns are parallel physical ranks and must not be summed.

| Operation | rank0 | rank1 | rank2 | rank3 | rank4 | rank5 | rank6 | rank7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tp_runtime_local | 99.258 | 98.871 | 98.907 | 98.581 | 16.783 | 16.787 | 16.467 | 16.569 |
| tp_pynccl_local | 99.160 | 98.825 | 98.414 | 98.519 | 15.277 | 14.890 | 14.937 | 14.823 |
| dp_runtime_global | 78.391 | 78.613 | 78.857 | 77.999 | 78.316 | 78.709 | 79.166 | 78.020 |
| naive_hidden_multicast | 95.339 | 94.831 | 95.368 | 95.729 | 95.146 | 94.827 | 95.563 | 95.837 |
| naive_router_multicast | 67.429 | 66.445 | 65.130 | 65.692 | 67.677 | 66.808 | 65.641 | 65.980 |
| naive_dispatch | 136.349 | 138.557 | 134.429 | 135.378 | 136.102 | 138.046 | 134.444 | 135.352 |
| naive_combine | 78.617 | 79.202 | 80.304 | 78.591 | 78.637 | 79.426 | 81.009 | 79.007 |
| naive_combine_post_tp | 200.156 | 200.481 | 199.407 | 199.872 | 198.055 | 198.550 | 197.739 | 197.966 |

`naive_dispatch` includes the hidden and router multicasts; `naive_combine_post_tp` includes both combine and the following TP reduction. These composition rows overlap their primitive rows. Their event durations include allocation, submission and synchronization effects; adding the medians of separately run components does not reconstruct a measured compound interval. All naive rows are diagnostic only under the retained ideal-model decision.

## Matched attention TP4 comparison

Only ranks0–3's16-MiB TP4 PyNCCL operation matches the first real attention AR. The small custom dummy group is a separate implementation and shape. Pooling the28 real-group samples gives runtime median98.904997us, range98.444005–99.347999us; direct PyNCCL median98.673667us, range98.363996–99.412670us. The difference between these medians is0.231331us. This is a controlled API-path comparison, not an attribution of the entire runtime wrapper's CPU overhead.

| Matched scope | Frontier predicted ms | Actual ms | Absolute error ms | Signed relative error |
| --- | ---: | ---: | ---: | ---: |
| One16-MiB TP4 runtime AR | 0.372905067 | 0.098904997 | 0.274000070 | +277.033595% |
| 48 equivalent AR calls | 17.899443200 | 4.747439861 | 13.152003339 | +277.033595% |
| Zero-launch counterfactual,48 calls | 3.499443200 | 4.747439861 | 1.247996661 | -26.287782% |

The first matched row is a primitive benchmark. Its48-call equivalent is not a new first-forward result. The existing lower-density in-context48-layer attention AR measured5.537248–5.708768ms; its higher span than the4.747440-ms primitive equivalent reflects a context difference that this benchmark does not uniquely explain. Keep it as an independent validation check.

## What the evidence proves about50us per step

The current model gives6ring steps ×50us =300us per call before adding any bytes or link latency. **This term alone is more than three times the entire observed approximately99-us primitive event span.** The50-us per-step setting is incompatible with this measured H200 TP4 case. Source and existing16-MiB kernel evidence show one PyNCCL API operation and a RING_LL device collective; ring rounds do not establish six50-us host submissions.

Simply dropping the term gives72.905067us, which underpredicts the primitive by26.287782%. The single-size result cannot separate effective protocol bandwidth, a fixed collective floor, link latency, launch cost and arrival effects. Therefore no replacement numeric value, bandwidth factor, or per-op scale is fitted from this result. No production parameter has changed.

## Follow-up sweep — now executed; physical calibration rule

The follow-up sweep is now complete; see `d019-communication-sweep.md` and its JSON for actual values, kernel joins, the passing heldout estimate and unresolved in-context gap. The retained measurement design follows. Root requested and approved a bounded primitive sweep at8,12,16,24,32MiB on the same TP4 real group, retaining the dummy4-KiB group as a separate family. The16-MiB row is held out from parameter fitting. The benchmark extension keeps20 warmups,48 calls and7 blocks; each size is measured through runtime and direct PyNCCL. A single profiler context **after all timing** captures one runtime operation per size with explicit range labels to establish kernel family. Capture times are never fitting inputs.

All sweep sizes are at least8MiB and therefore fail custom's strict-less-than8-MiB threshold. Source selection establishes PyNCCL for them, but it does not establish that NCCL chooses the same device algorithm at every size. The existing K trace establishes RING_LL at16MiB only. Fit no common curve until the fresh signature trace verifies a common family or identifies where it changes. Do not force `NCCL_ALGO`/`NCCL_PROTO` just to make the samples fit.

For a verified constant ring family, fit `T(B)=a+b*(1.5*B)` on8/12/24/32MiB primitive samples, with a positive physical slope and a nonnegative inferred fixed floor after the existing link-latency term. The existing case-specific fields can represent measured effective bandwidth and an amortized per-step floor for the fixed TP4 domain. The amortized floor is not evidence of actual per-step CPU launches and must be documented accordingly. Do not modify shared defaults or fit across the dummy custom family.

Acceptance before applying a parameter correction: fitting stability across repeats/ranks; no family mismatch; heldout16-MiB prediction within10% and consistent with its repeat spread; independent lower-density in-context comparison reported; no total-error cancellation hiding known op defects. If the intercept is negative, the model is visibly nonlinear, or heldout fails, the affine parameterization is rejected rather than clipped. A one-size event-derived bandwidth is not an independently identified link constant.

Pending: independent review of the completed fit and candidate case-parameter scope, application if accepted, and fresh full-forward validation. Sweep collection, kernel identity joins and primitive heldout validation are complete in the follow-up report. No implementation of the deferred naive protocol is pending in the active scope.
