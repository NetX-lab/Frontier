## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated the five-size H200 sweep, all kernel signatures and heldout prediction; derived one existing case-parameter candidate with explicit scope and remaining in-context gap. |

# D019 ideal communication parameter calibration

**PASS:8 ranks,80 operation rows,560 non-profiled event samples,40 uniquely joined kernel launches.** The fixed-bandwidth single-parameter model passes the independent16-MiB holdout with−0.644124% error. Recommended candidate: case-local `nvlink_allreduce_launch_overhead_us=4.384788772964477`, replacing50.0 while retaining bandwidth450GB/s, efficiency0.8 and link latency0.5us. This is a proposed calibrated parameter, not an applied production change or a completed CUDA-forward correction.

The job's later single-GPU timing-context phase failed independently. That failure does not invalidate the completed, correctly serialized communication phase. Root owns the overall execution manifest. This report claims only communication acceptance. D020 remains in effect: keep the current ideal EP abstraction; no naive protocol, selector, broadcast, shared DP schema or new forward phase.

## Actual execution and identity

Root-managed job `yc26-h200-d019-profiles-20260908-03`, separate8H200 `step_main` allocation. Source: `analysis/h200-d019-profiles-03/runtime/communication/rank_0.json` through `rank_7.json`, plus corresponding `rank_N_kernel_signature.json`. Frozen scripts, launch and execution receipt are in that run directory. The GPU image is `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.

GPU conda `vllm-bs-0.10.2`, Python3.10.16, Torch2.8.0+cu128, CUDA12.8, vLLM `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`. The original platform NCCL settings, eager/inference mode and custom-allreduce configuration were preserved. The executed command was:

```bash
/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=8 \
  tests/performance/issue26_h200_collective_microbenchmark.py \
  --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-profiles-03/runtime/communication \
  --tp-message-bytes 8388608 12582912 16777216 25165824 33554432 \
  --capture-tp-kernels
```

These are collective message-size controls, not additional E2E prefill cases. For every requested DP0 size, ranks0–3 hold that BF16 message in theirTP4 group; ranks4–7 retain the original4-KiB dummy input in a separateTP4 group. Each operation has20warmups and seven48-call eager event blocks. Runtime and direct PyNCCL are measured separately. All560 durations are positive and finite, per-block/per-call conversion and stored median/min/max match recomputation, and all numerical sum assertions pass. The source bytes run on GPU match the benchmark extension committed as `882ac05b`; only that owned benchmark increment was committed.

## Kernel identity and contamination control

After all latency measurements, one profiler context captured one runtime collective per requested size, per rank. Kernel→CUDA launch joins use exact correlation IDs; launch→named message-size range joins require matching CPU pid/tid and timestamp containment. All40kernels have exactly one matching launch and exactly one message-size range; no missing launch/device correlation exists. This is stronger than inferring the algorithm from a Python API name.

All20real-lane captured kernels are `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)`. All20dummy-lane kernels are `vllm::cross_device_reduce_1stage<__nv_bfloat16,4>`, separately recorded and excluded from fitting. Real8-MiB captures use24blocks of640threads, whereas12/16/24/32MiB use24blocks of544threads. The common RING_LL family is established; launch geometry is not falsely claimed identical.

Profiler startup and rank skew severely inflate early captured kernel durations (for example the first rank0 captured8-MiB kernel lasts8.436ms). **No profiler duration enters any fit or validation latency.** Those records supply identity only. The event measurements were completed before capture began.

## Primitive timings and heldout result

Units are microseconds per collective. A size's central value is the median of28samples across the four parallel real ranks; it is a descriptive summary, not28statistically independent observations. Ranks are never summed. The fit gives each training size equal weight and excludes16MiB entirely.

| Size MiB | Runtime median us | Direct PyNCCL median us | Direct sample range us | Fixed-bandwidth predicted us | Signed error versus direct |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 61.479665 | 61.279001 | 60.851336–61.978002 | 64.261266 | +4.866699% |
| 12 | 81.681666 | 81.173666 | 80.838665–81.739331 | 81.737533 | +0.694643% |
| 16 (held out) | 99.754001 | 99.857002 | 99.545995–100.508004 | 99.213799 | -0.644124% |
| 24 | 133.540332 | 134.268334 | 133.747329–137.616664 | 134.166333 | -0.075968% |
| 32 | 170.996000 | 172.562997 | 171.904663–174.136003 | 169.118866 | -1.995869% |

The training maximum absolute error is4.866699%; leave-one-training-size-out maximum is6.488932%. The16-MiB heldout prediction99.213799us versus99.857002us differs by0.643203us/−0.644124%, satisfying the10%criterion. It sits slightly below the observed repeat range99.546–100.508us; this small but resolved model bias is disclosed rather than called measurement noise. Independently measured runtime-path16-MiB median99.754001us also agrees to approximately0.54%.

Per-rank fitted aggregate floors are29.300400,29.432733,29.336736 and29.195899us. Seven round-index fits using each round's cross-rank median range29.144065–29.995316us. These checks establish stability of the scoped estimate; they are not formal independent-sample confidence intervals.

## Identifiability and mapping to existing parameters

For the current fully intra-serverTP4 ring estimator, with per-rank logical input `B` bytes:

```text
T_us(B) = (1.5 * B) / (450e9 * efficiency) * 1e6
          + 6 * nvlink_latency_us
          + 6 * nvlink_allreduce_launch_overhead_us
```

The multiplier1.5 is the existing ring traffic contract `2*(n−1)/n` atn=4, and6is the existing `2*(n−1)`step count. With450GB/s, efficiency0.8 and latency0.5us held fixed, only one free coefficient remains. In MiB units the fixed slope is4.369066666667us/MiB. Equal-weight least squares for that fixed slope gives:

```text
training sizes = [8, 12, 24, 32] MiB
aggregate floor a = mean(T_measured(m) - 4.369066666667*m)
                  = 29.30873263778686 us
launch-field candidate = (a - 6*0.5) / 6
                       = 4.384788772964477 us/step
```

This identifies an **aggregate eager collective floor in a fixedTP4/RING_LL/8–32MiB domain**. It does not separately measure host CPU launch, NCCL fixed device work, link latency, allocation or arrival skew. Dividing by6maps that aggregate into the existing estimator's parameterization; it does not prove there are six CPU launches or that overhead scales linearly across other group sizes. The candidate is therefore case-local and must not replace a global hardware default or become an independently measured CPU-overhead row.

A two-parameter affine fit was evaluated as a diagnostic: direct PyNCCL intercept25.026650us, effective-efficiency0.760757 and heldout−1.321211%. It is unnecessary because the fixed-bandwidth one-parameter model already passes. More importantly, `nvlink_efficiency` affects every collective, including the retained ideal EP8 all-to-all. A TP4 RING_LL fit would not qualify the EP8 abstraction's effective bandwidth. Keeping efficiency unchanged avoids introducing that unsupported cross-family change.

The physically checked H200450GB/s direction and topology remain fixed. Efficiency0.8 remains the existing modeling assumption, not a newly measured raw NVLink wire property. The five-size experiment validates the resulting scoped predictor empirically without claiming unique microscopic parameters.

## First-forward effect and remaining limits

The existing field is passed by `CollectiveSimCCBackend._build_scenario` into `IntraServerConfig`. `estimate_intra_server_ms` multiplies it by the existing step count **only for allreduce**, so changing this field leaves ideal all-to-all dispatch/combine untouched. First-forward attentionTP4 AR uses the16-MiB payload and six steps. The current eager CUDA-event case does not trigger `_should_strip_collective_sim_allreduce_launch_overhead`, which requires a kernel-only captured decode path. Candidate first-forward48-layer attention AR is4.762262367ms, versus the current17.8994432ms, a13.137180833-ms model correction.

The candidate remains a backend AR knob, not a message-size-conditional predictor. Later small custom-AR decode batches and different group sizes are outside this fit's qualified domain. They must not be marked calibrated by this first-forward result. No general custom-versus-NCCL runtime model is introduced here.

The independent older low-density in-context attention totals are5.537248/5.649952/5.708768/5.644864ms. The candidate4.762262ms is−13.995863%/−15.711454%/−16.579858%/−15.635481% versus those intervals. Therefore the isolated primitive holdout passes, while an **in-context communication-only10%gate does not pass**. That context gap is not used to increase the fitted floor. Existing context measurements include arrival/submission behavior and should be reassessed after the other first-forward corrections; this report does not identify their entire residual as CPU time or pure wait.

Pending: independent review of this candidate and its field scope, root application to the existing case configuration if accepted, and fresh first-forward validation. No production config or backend source was changed by lane C. The deferred naive modeling is not an active blocker.

## Reproducible fit command

Run from the active Frontier worktree with conda `dev-vidur-v03-hopper-e2e`, Python3.13.13:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PYCODE'
import json, statistics
from pathlib import Path
p=Path('task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-profiles-03/runtime/communication')
ranks=[json.loads((p/f'rank_{i}.json').read_text()) for i in range(4)]
train=[8,12,24,32]
values={m:statistics.median(s['per_call_ms']*1000 for r in ranks for row in r['measurements'] if row['operation']==f'tp_pynccl_{m*2**20}_bytes' for s in row['samples']) for m in [8,12,16,24,32]}
slope=1.5*2**20/(450000*0.8)
floor=statistics.mean(values[m]-slope*m for m in train)
launch=(floor-6*0.5)/6
prediction=floor+slope*16
print(json.dumps({'training_sizes_MiB':train,'slope_us_per_MiB':slope,'floor_us':floor,'launch_candidate_us_per_step':launch,'heldout_prediction_us':prediction,'heldout_actual_us':values[16],'heldout_error_percent':100*(prediction-values[16])/values[16]},indent=2))
PYCODE
```

Observed outputs: floor29.30873263778686us; candidate4.384788772964477us/step; heldout prediction99.21379930445353us; actual99.85700249671936us; error−0.6441242738955248%. Full identity joins, individual rank medians, repeat stability, leave-one-size-out checks and alternate fits are preserved in `analysis/d019-communication-sweep.json`.
