## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-09 | Recorded synchronization-boundary source checks, existing trace evidence, and diagnostic A/B execution status. |

# Synchronization-boundary post-MoE AR RCA

## 1. Test Script Information

- Source checks: `vllm/model_executor/layers/fused_moe/layer.py`, `vllm/distributed/device_communicators/all2all.py`, `vllm/distributed/parallel_state.py`, `vllm/distributed/device_communicators/cuda_communicator.py`, `vllm/distributed/device_communicators/pynccl.py` in `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`.
- Existing trace evidence: `analysis/h200-rca-01/runtime/kernels/frontier_profiler_traces/frontier_batch_4734_*.json` and corresponding 4735/4736 files.
- Diagnostic source check: `python -m py_compile vllm/model_executor/layers/fused_moe/layer.py` in `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab`.
- Baseline command: RJob `yc26-h200-rank-stability-20260909-01`, frozen diagnostic commit `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`.
- Scalar A/B command: RJob `yc26-h200-ar-scalar-20260909-01`, isolated diagnostic commit `e60f4dfd5e8f362a4f0625dd4de3fcfee1516efd`, mode `VLLM_FRONTIER_DIAG_MOE_AR_MODE=scalar_sync`.
- Environment: H200 `step_main`, image `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`, Python `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`.

## 2. Validation Criteria

The measurement must begin at the actual DP-pair combine boundary and distinguish:

1. DP-pair completion/arrival skew.
2. Host submission delay after combine.
3. Queued CUDA work before the TP AR call.
4. TP AR payload/backend cost.

For each matched batch/layer/rank, the target values are combine start/end, TP AR start/end, `ar_start - combine_end`, and cross-TP arrival/scope spread. The scalar A/B keeps the TP collective participant rendezvous but replaces the normal hidden-state payload with a one-element BF16 all-reduce. The skip mode is timing-only and semantically invalid.

Acceptance for this RCA is causal classification with direct evidence. It does not require a Frontier calibration gate and cannot use rank-local scope sums as an additive latency term.

## 3. Test Results and Evidence

### Source and trace evidence — PASS

- The vLLM layout is `ExternalDP × DP × PP × TP`: TP groups are `[0,1,2,3]` and `[4,5,6,7]`; DP groups are `[0,4]`, `[1,5]`, `[2,6]`, `[3,7]`.
- `NaiveAll2AllManager.combine()` calls `dp_group.all_reduce()` and slices the local DP interval. It synchronizes only each DP pair.
- `reduce_output()` calls the DP combine and then immediately calls `tensor_model_parallel_all_reduce()` on the TP group. There is no TP-wide barrier between them.
- The collective path enqueues NCCL on the current CUDA stream without a host or device-wide synchronization.
- Existing GPU trace evidence shows the latest combine arrival is the shortest post-MoE AR scope in 48/48 layers for batch 4734, 45/48 for 4735 and 30/48 for 4736. For batch 4734 first layer, combine-start spread is about 1.31 ms median and 1.665 ms maximum while combine duration is about 0.08 ms. This establishes that pairwise combine does not close the TP arrival skew.
- Routing evidence shows local routed-token counts `4176/4083/4296/4081` across TP0–TP3 for one DP0 batch, a 5.2% spread. Existing rows do not record post-combine `states.shape`, so they do not yet prove a payload-size difference across TP ranks.

### Diagnostic source A/B — PASS_PRECHECK

- `normal`, `scalar_sync`, and `skip` mode parsing is explicit and fail-fast for unknown values.
- `scalar_sync` calls the same TP collective with a one-element tensor and returns a cloned partial result for timing only.
- `skip` returns a clone without the TP collective and is explicitly invalid for inference correctness.
- The patched source compiles successfully and is committed only in the isolated diagnostic copy. The frozen diagnostic checkout was restored to its original clean commit before the baseline job was reused.

### GPU execution — H200_SKIP_INVALID; H800_BOUNDARY_RERUN_PENDING

- Baseline RJob `yc26-h200-rank-stability-20260909-01` succeeded after its 100-request client phase; its persistent operator rows contain the four rank totals reported below.
- Scalar RJob `yc26-h200-ar-scalar-20260909-04` succeeded after its 100-request client phase; its source diff and output manifest are persistent under `analysis/h200-ar-scalar-04/`.
- Bounded skip RJob `yc26-h200-ar-skip-20260909-01` is `Failed`: the client rejected `warmups=0` before sending requests (`issue26_token_id_client.py:120` requires at least three warmups). All operator and batch files are zero bytes, so it provides no skip timing evidence.
- Because the H200 skip run failed, the same bounded diagnostic is queued on H800 with corrected `warmups=3`: `yc26-h800-ar-skip-20260909-01`. Matching H800 normal and scalar jobs are also queued so all three modes can emit the new boundary JSONL from diagnostic commit `0d633a946`.
- Current H800 platform state: `yc26-h800-ar-normal-20260909-01`, `yc26-h800-ar-scalar-20260909-01`, and `yc26-h800-ar-skip-20260909-01` are all `Starting`/waiting for worker allocation. No H800 boundary rows have been interpreted yet; the jobs are preserved for FIFO scheduling.

### Scalar payload-isolation result — PASS_DIAGNOSTIC

For the first formal request (`cmpl-pf4096_dc1024:0-0`), each TP rank recorded 48 post-MoE collectives. The normal and scalar runs are separate fresh batches (`4193` and `4179`) with the same frozen case and are compared only as diagnostic distributions:

| Mode | TP0 AR sum (ms) | TP1 | TP2 | TP3 | Outer span TP0–TP3 (ms) |
| --- | ---: | ---: | ---: | ---: | --- |
| normal | 30.775808 | 19.624416 | 6.887840 | 26.885280 | 90.073540 / 90.056641 / 90.090881 / 90.070847 |
| scalar_sync | 26.297856 | 0.884320 | 25.805088 | 18.144064 | 89.217377 / 89.209632 / 89.204926 / 89.239075 |

The one-element TP collective retains participant synchronization but removes the normal hidden-state payload. The rank-local spread remains large (`23.887968 ms` normal; `25.413536 ms` scalar), while outer span remains within `0.034149 ms` across scalar ranks and is close to the normal diagnostic span. This rejects payload size as the primary explanation for the rank-local gap in this diagnostic comparison. It supports a late-participant/queue attribution; it does not yet distinguish local routed work from host scheduling or device queue delay.

## Current RCA conclusion

The proven root cause of the apparent rank-local post-MoE AR gap is a missing TP-wide alignment after a DP-pair-only combine. A late DP pair reaches the first TP-wide AR later; earlier TP ranks then record more inclusive wait, while the late participant records a shorter AR scope. Existing traces plus scalar payload isolation reject normal payload size as the primary cause. The remaining split between local routed workload, host enqueue delay and queued device work requires boundary timestamps and a bounded skip run before deciding whether clean/diagnostic span reconciliation is warranted.

### H800 cluster separation and rerun checkpoint (2026-09-09)

- H200 jobs use `--charged-group=step_main --positive-tags=h200` and the H200 probe, while H800 jobs use `--charged-group=codesign --positive-tags=h800` and the dedicated H800 probe. The two clusters are recorded and scheduled independently.
- `yc26-h800-ar-normal-20260909-01` failed before worker startup with platform RDMA admission error (`no healthy devices present ... mellanox.com/mlnx_rdma`); it has no timing artifacts.
- `yc26-h800-ar-scalar-20260909-01` and `yc26-h800-ar-skip-20260909-01` produced only environment logs because the H200 probe was incompatible with H800 (`NVMLError_NotSupported`, then `CUDA_HOME: unbound variable`). They are invalid timing runs.
- Corrected probe: `/data/ycfeng/tmp/issue26_h800_environment_probe.sh` (H800 GPU assertion, NVML capability logging, sequential `CUDA_HOME`/`PATH` exports). Reruns submitted with diagnostic source `0d633a946`, `warmups=3`, and `requests=1`: `yc26-h800-ar-normal-20260909-02`, `yc26-h800-ar-skip-20260909-02`, `yc26-h800-ar-scalar-20260909-03`. At checkpoint, all three remain `Starting`/waiting for H800 allocation; no boundary rows exist yet.

### Cluster-separated memory-safe rerun checkpoint (2026-09-09)

- H800 `-02/-03` runs reached the worker probe (`H800_RUNTIME_PROBE_PASS`) but failed during EngineCore KV-cache initialization. H800 exposes ~79.19 GiB and had 477 MiB free when vLLM attempted another 2.37 GiB allocation with the inherited H200 `--num-gpu-blocks-override 310809`. Each rank emitted 144 boundary rows from initialization profiling; all `server.batch.*` and `server.ops.*` files remained zero and no client request was sent. These rows are invalid for first-forward reconciliation.
- H800 workers now use `--num-gpu-blocks-override 4096`; reruns `yc26-h800-ar-{normal,scalar,skip}-20260909-04` are submitted with `codesign + h800`.
- Independent H200 reruns `yc26-h200-ar-{normal,scalar,skip}-20260909-05` use `step_main + h200`, the H200 probe, and the original `310809` block override. Their outputs are separate from H800 outputs and will only be reconciled within the same cluster.
