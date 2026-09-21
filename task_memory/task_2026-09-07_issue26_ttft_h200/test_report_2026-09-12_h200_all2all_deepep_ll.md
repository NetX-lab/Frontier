## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-12 | Recorded the completed H200 ten-warmup DeepEP low-latency run and source-grounded RCA. |

# H200 alternative all2all backend test report

## Result

The formal DeepEP low-latency run completed all execution and identity gates,
but it is not a valid normal prefill calibration reference. The first formal
DP0 batch-only CUDA event span is **719.293090820 ms** (median across TP0--TP3),
which is about **8.61x** the warmup-normalized native/naive reference from the
same backend experiment (`83.551776886 ms`). The source and vLLM deployment
documentation explain this result: DeepEP low-latency is intended for
decode-dominated workloads, while this case is a 4096-token prefill. The LL
path uses a 256-token-per-DP-rank chunk size and executes the low-latency
dispatch/combine protocol repeatedly for every MoE layer.

This result is evidence about backend/workload semantics. It does not authorize
a Frontier communication correction, a fitted residual, or clean/diagnostic
span reconciliation.

## Fixed execution

- Cluster: H200, `step_main`, `h200`, one 8-GPU node, BF16, eager execution.
- Workload: Qwen3-30B-A3B-Instruct-2507 dummy weights; 4096 prefill / 1024
  output; TP4/DP2/PP1/EP8; uniform routing; prefix caching OFF; chunked prefill
  OFF; FlashInfer attention.
- Capacity: `--num-gpu-blocks-override 310809`, `max_model_len=16384`,
  `max_num_batched_tokens=16384`, `max_num_seqs=1024`.
- Runtime: Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0, CUDA 12.8.93,
  pinned image
  `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.
- All three timing arms used ten complete drained 100-request warmup replays,
  followed by 100 formal requests. The batch manifests record
  `warmup_rounds=10`, `formal_requests=100`, and `op_profile_selected=false`.
- The standard chain was used:
  `tests/e2e/issue26_h200_replay_worker.sh` ->
  `tests/e2e/issue26_h200_uniform_groundtruth_worker.sh` ->
  `tests/e2e/issue26_h200_diagnostics_worker.sh batch` ->
  `tests/e2e/issue26_token_id_client.py` ->
  `tests/e2e/issue26_diagnostic_identity_analysis.py --mode batch`.

The backend-specific RJobs and launch wrappers were:

| Backend | RJob | Launch wrapper | vLLM source commit |
| --- | --- | --- | --- |
| native/naive | `yc26-h200-all2all-naive-20260912-01` | `/data/ycfeng/tmp/issue26-h200-all2all-naive-20260912-launch.sh` | `eb4c9a1394ef136f134b5a36538847ec01679c68` |
| DeepEP high-throughput | `yc26-h200-all2all-deepep-ht-20260912-02` | `/data/ycfeng/tmp/issue26-h200-all2all-deepep-ht-20260912-02-launch.sh` | `150fa4a1cf46500c22d5fa585ebc9801a43e5c2d` |
| DeepEP low-latency | `yc26-h200-all2all-deepep-ll-20260912-05` | `/data/ycfeng/tmp/issue26-h200-all2all-deepep-ll-20260912-05-launch.sh` | `150fa4a1cf46500c22d5fa585ebc9801a43e5c2d` |

DeepEP LL required the rebuilt wheel
`/data/ycfeng/tmp/issue26-deepep-rebuild-extracted-20260912-02/` after the
capability probe detected a host/device NVSHMEM mismatch in the original wheel.
The corrected capability RJob `yc26-h200-deepep-ll-capability-20260912-04`
passed on all eight ranks (`hint=545260672`, `max_tokens_per_dp_rank=256`,
buffer PASS, barrier PASS, `torchrun.exit_code=0`).

## Formal boundary and measurements

The first formal row was selected by request identity, not by minimum batch ID:

```text
request_id              = cmpl-pf4096_dc1024:0-0
batch_size              = 1
request_num_tokens      = [4096]
batch_num_prefill_tokens= 4096
batch_num_decode_tokens = 0
DP lane                 = DP0
TP ranks                = TP0--TP3
batch_dp_token_counts   = [4096, 1]
```

The validator command for the LL batch artifact was:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python \
tests/e2e/issue26_diagnostic_identity_analysis.py \
  --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/\
h200-all2all-deepep-ll-20260912-05/batch/runtime/batch \
  --mode batch --warmups 10 \
  --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/\
h200-all2all-deepep-ll-20260912-05-identity_validation.json
```

The validator returned `status=PASS` for all eight workers. Each backend has
1100 client rows (1000 warmup rows plus 100 formal rows), and each batch arm
contains the eight rank logs and complete formal drain.

| Backend | TP0 | TP1 | TP2 | TP3 | Median | P90 | Rank max | Rank spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native/naive | 83.554787 | 83.607101 | 83.546211 | 83.548767 | 83.551777 | 83.591407 | 83.607101 | 0.060890 |
| DeepEP high-throughput | 86.499840 | 86.509506 | 86.207138 | 86.473053 | 86.486446 | 86.506606 | 86.509506 | 0.302368 |
| DeepEP low-latency | 719.277649 | 719.308533 | 719.308899 | 719.197388 | 719.293091 | 719.308789 | 719.308899 | 0.111511 |

All values are milliseconds and are the reduced batch-only CUDA event rows. The
batch logger is diagnostic instrumentation, but these rows exclude the old
full diagnostic outer span, operator profiling rows, and full record-function
traces. The manifests explicitly retain `op_profile_selected=false`.

The LL raw DP0/TP0 sequence around the first formal request is also material:

| Batch ID | Prefill | Decode | DP token counts | Span (ms) |
| ---: | ---: | ---: | --- | ---: |
| 10490--10494 | 0 | 14, 11, 8, 5, 2 | `[n,n]` | 66.62--67.51 |
| 10495 (formal) | 4096 | 0 | `[4096,1]` | 719.278 |
| 10496 | 8192 | 1 | `[8193,4096]` | 1870.592 |
| 10497 | 4096 | 3 | `[4099,12289]` | 2583.210 |
| 10498 | 0 | 4 | `[4,4100]` | 744.316 |

The following mixed-batch values are evidence of backlog after the expensive
prefill. They are not added to the first-formal span and are not used as a
Frontier correction.

## Source-level RCA

### 1. Backend selection and semantic intent

`vllm/distributed/device_communicators/cuda_communicator.py:84-103` selects
`DeepEPLLAll2AllManager` when `VLLM_ALL2ALL_BACKEND=deepep_low_latency` and
selects `DeepEPHTAll2AllManager` for the high-throughput arm. This is a real
protocol change, not a label-only change.

The vLLM deployment guide at
`docs/serving/expert_parallel_deployment.md:17-23,182-190` describes
`deepep_high_throughput` as the prefill-oriented backend and
`deepep_low_latency` as the decode-oriented backend. The current experiment is
the opposite of the LL target: a single 4096-token prefill first-forward.

### 2. The LL path forces a 256-token chunk size

`vllm/envs.py:130` defines `VLLM_MOE_DP_CHUNK_SIZE=256`, and
`vllm/model_executor/layers/fused_moe/config.py:320` uses it as
`FusedMoEConfig.max_num_tokens`. The LL constructor receives that value at
`vllm/model_executor/layers/fused_moe/layer.py:173-195` as
`max_tokens_per_rank`; it also materializes an 8-rank, 128-global-expert,
16-local-expert DeepEP buffer in
`vllm/distributed/device_communicators/all2all.py:202-242` with
`low_latency_mode=True`.

For the formal `[4096,1]` DP input, the 4096-token dispatcher extent requires
`ceil(4096/256)=16` chunk iterations. The model has 48 MoE layers, so the same
chunked path is entered repeatedly across the layer stack. This is a control
flow consequence of the source and is distinct from a single one-shot
4096-token all2all.

### 3. Each chunk invokes full LL dispatch and combine

`vllm/model_executor/layers/fused_moe/layer.py:1696-1806` copies each chunk to
staged hidden/router buffers, calls `quant_method.apply`, and loops over the
dispatcher extent in `max_num_tokens` increments. The LL path is selected by
`layer.py:1820-1828`.

Inside `deepep_ll_prepare_finalize.py:151-165`, LL dispatch uses
`low_latency_dispatch(..., max_tokens_per_rank=256, async_finish=False,
return_recv_hook=True)`. The corresponding combine at lines `205-235` uses
`low_latency_combine(..., async_finish=False, zero_copy=False,
return_recv_hook=False)`. Thus each chunk waits for the synchronous LL
protocol choices and returns through a separate combine call; there is no
source evidence that the 16 prefill chunks are fused into one prefill launch.

The measured 719 ms is therefore best explained as repeated low-latency
protocol and chunk scheduling overhead on a prefill workload. The adjacent
66--68 ms decode-only rows and the 1.87--2.58 s mixed rows corroborate that the
formal prefill leaves queued work behind it. The raw rows do not isolate a
single kernel or assign a pure communication duration, so the per-chunk cost is
an inference from control flow, not a separately measured kernel number.

### 4. Reduction semantics differ from native/naive

`vllm/model_executor/layers/fused_moe/layer.py:1588-1611` states that PPLX,
DeepEP HT, and DeepEP LL combine paths already reduce shared expert outputs and
skip the later tensor-model-parallel all-reduce. Native/naive reaches the
separate post-MoE TP reduction path. This makes DeepEP timing a fused protocol
variant; it cannot be decomposed into Frontier's independent dispatch,
combine, and post-MoE TP-AR predictors.

## PPLX status

The PPLX capability artifact
`analysis/h200-pplx-capability-20260912-03/capability.json` is `status=FAIL`
with no imported `pplx_spec` and no distributed handle. Earlier attempts failed
while linking NVSHMEM device symbols. PPLX therefore has no formal timing value;
this is an unresolved capability/build result, not a claim that PPLX is
semantically unsupported.

## Interpretation and limits

DeepEP HT is the only tested alternative whose documented target matches this
prefill case. Its fused EP participant graph is structurally closer to
Frontier's ideal EP phase than native/naive, but its 86.486446 ms median is
3.51% slower than native/naive in the same reduced batch-only experiment.
DeepEP LL is capability-valid but semantically mismatched to this prefill and
is 8.61x slower. Neither result is a production replacement for Frontier's
ideal EP number.

The prior clean integrated comparison remains separate:

```text
Frontier integrated = 59.190354384 ms
vLLM clean reference = 78.118782043--79.307357788 ms
```

The vLLM excess is 18.928427659--20.117003404 ms. Because Frontier's compute
prediction is itself about 9.740 ms higher than observed vLLM compute, the
non-compute remainder implied by that comparison is approximately
28.668427659--29.857003404 ms after adding the compute overestimate. This is a
sign correction to the earlier “only 9--10 ms remains” interpretation; it does
not identify all of that remainder as communication.

No CUDA gate or operator gate is closed. No Frontier source, predictor, CPU
add-on, fitted residual, or clean/diagnostic reconciliation was changed from
these measurements.

