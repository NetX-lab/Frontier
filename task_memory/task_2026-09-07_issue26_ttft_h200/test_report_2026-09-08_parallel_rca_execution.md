## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded fresh H200 RCA execution, independent client/batch checks, partial profiler failure and communication supplement. |

# D018 parallel execution verification

## Execution

Workspace: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
CPU conda: `dev-vidur-v03-hopper-e2e`; Python3.13.13, `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`.
GPU conda: `vllm-bs-0.10.2`; Python3.10.16, `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`; Torch2.8.0+cu128, CUDA12.8.93, FlashInfer0.3.0.
Pinned image: `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.

The full platform allocation, company-proxy setup and worker command are preserved in the following executable launch records. They reproduce the same H200 step_main scenario; an actual rerun must use a new output generation rather than append to completed artifacts.

```bash
bash /data/ycfeng/tmp/issue26-h200-network/launch-rca-01.sh
bash /data/ycfeng/tmp/issue26-h200-network/launch-rca-comm-01.sh
```

Persistent copies: `analysis/h200-rca-01/launch.sh`, `analysis/h200-rca-comm-01/launch.sh`, each directory's `worker_snapshot.sh`, `run_manifest.json`, and runtime `mode_manifest.json`/`server_command.json`. Both use `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908` commit8453dd342c6aa2721aaf4b410998aab2f38bc2ec. No external checkout mutation occurs during the running supplement.

Independent full client and worker-local batch checks:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/runtime/batch --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/batch_validation.json --mode batch > /data/ycfeng/tmp/issue26-h200-network/rca-batch-validation-01.log 2>&1
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/runtime/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/operator_batch_validation.json --mode batch > /data/ycfeng/tmp/issue26-h200-network/rca-operator-batch-validation-01.log 2>&1
```

## Criteria and observed evidence

- H200 environment/topology must match the approved case. PASS: both0844and0761report eight NVIDIA H200 compute checks, per-GPU NVLink topology and H200_RUNTIME_PROBE_PASS. Effective communication mode uses exactly the three requested TP AR scopes, CUDAevent/default/per_scope, runtime_meta0 and uniform routing. Prefix caching, eager mode, token/KV limits and all other server settings are retained in the exact command records.
- Each completed serving mode must contain400unique completed4096/1024requests, including exactly300warmups and100formal requests; each formal prefill must appear once perTP on exactly oneDP. PASS separately for batch/route and all-scope CUDA-event modes; both independent analyzers exit0. Detailed op selection/coverage is verified by the operator-specific analyzer, not by the batch-only check.
- Record-function mode must retain complete activity coverage before kernel values are accepted. FAIL for the full mode: launcher exits1 and client request60 receives no streamed token. Server traceback reaches `gpu_model_runner.py:2455 -> v1/utils.py:449 -> v1/utils.py:208`, raising `RuntimeError: Non-positive CUDA time for op tensor_parallel_allreduce: 0.0us`. Raw third DP0 profile has5missing device correlations on everyTP; second profile has2missing onTP1/TP2. First profile has3077launches and0missing correlations perDP0TP rank and is retained as bounded diagnostic evidence. No zero replacement or error suppression is applied.
- The first numerical batch comparison is Frontier65.790076904ms versus batch-only vLLM80.335617065ms: absolute error14.545540162ms, relative error18.106% (actual denominator). This is a comparison of predicted forward and measured CUDA-event span, not official TTFT acceptance.
- The separate full-event firstbatch116.210144ms and kernel-profile153.403ms demonstrate substantial instrumentation/run sensitivity relative to batch-only80.335617ms. These differences must not size a production repair. Kernel tracing identifies host submission gaps and collective waiting as concrete perturbation mechanisms.

Communication-only completion and independent identity verification PASS:400unique completed clients,100formal prefill identities across8workers;24selected rank-batches and2328scope rows pass the dedicated analyzer. Worker marker DIAGNOSTIC_EXECUTION_COMPLETE selection=communication is present; systemd launcher ExecMainStatus=0 and inactive. The RCA01 launcher remains failed with exit1 from the documented kernel collector failure; no active GPU launcher remains. The full three-aligned-global-batch operator gate is not claimed. Clean E2E remains the independently completed replay02 reference; D018 introduces no new calibrated production result.

Communication completion check, exit0:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_diagnostic_identity_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-comm-01/runtime/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-comm-01/batch_validation.json --mode batch > /data/ycfeng/tmp/issue26-h200-network/rca-comm-batch-validation-01.log 2>&1
```

Firstbatch communication scope totals and their timing limits are preserved in `analysis/first-batch-op-rca/communication/summary.json`: attentionTP5.537248–5.708768ms versusFrontier17.899440ms; extraMoETP4.792544–28.413280ms with rank-dependent waits; wholeforward86.412033–86.443520ms. These are diagnostic values, not a fitted clean correction.

## Reports and practical limits

`analysis/parallel_rca_summary.md` integrates the findings. `test_report_2026-09-08_dp_workflow_controls.md` preserves source, same-run joins and paired DES commands. `test_report_2026-09-08_first_batch_op_rca.md` preserves the fresh predictor command, per-op gap table, kernel analysis and the interrupted sequential time-limit probe. No historical-version numerical evidence is accepted, no production CPU constant is added, and deferred gated-SiLU/routing-import work remains deferred.
