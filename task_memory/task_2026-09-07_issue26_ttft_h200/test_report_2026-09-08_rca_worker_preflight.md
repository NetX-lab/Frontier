## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified RCA diagnostic orchestration and recorded H200 execution preflight. |

# RCA worker preflight

Status: PASS for source/preflight checks; H200 diagnostic results pending.

## Execution

Working directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907

```bash
bash -n tests/e2e/issue26_h200_diagnostics_worker.sh
git diff --check
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_selection.py --checkout /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908 --output /data/ycfeng/tmp/issue26-op-selection-root-check-01.json
```

CPU Conda dev-vidur-v03-hopper-e2e, Python3.13.13. Full launch command: analysis/h200-rca-01/launch.sh; source snapshot: analysis/h200-rca-01/worker_snapshot.sh; exact image/source/configuration: analysis/h200-rca-01/run_manifest.json. Company proxy lowercase/uppercase synchronized.

## Criteria and evidence

- PASS: shell parses; whitespace checks pass; source selection helper checks7cases, mask[false,true,true,true,false], unchanged model/collective call sites. Detects accidentally profiling warmup/all batches or changing execution call paths.
- PASS: exact diagnostic branch/commit, clean tree, remote identity and observed tip recorded; output absent before launch.
- PASS: actual H200 environment probe reports8NVIDIA H200 devices, each compute test32.0, activeNVLinks and all peer NVLink status0.
- Source issue recovered: runtime metadata outside an active scope is unsupported; removed enablement, committed26a57780. Actual mode manifests must confirm disabled runtime metadata before accepting samples. No metadata-dependent shape equality is claimed.
- Initial root selection command used a nonexistent filename and exited2; corrected exact command above passes.

These checks establish diagnostic execution readiness and limited selection correctness, not numerical parity. The modes remain separate: route/batch (operators off), CUDA-event scope spans, and record-function kernel traces. Existing clean replay02 is the E2E reference; instrumented runtimes never substitute for it.

## Communication scope supplement

Observed full-scope diagnostic inflation116.210144vs80.335617ms motivated a lower-density control. Existing `VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES` selects `expert_parallel_allreduce,attn_post_proj_tp_allreduce,tensor_parallel_allreduce`. No logger source change. Workercommit e108696a passes `bash -n tests/e2e/issue26_h200_diagnostics_worker.sh` and `git diff --check`. Exact second launch and manifest: analysis/h200-rca-comm-01/launch.sh and run_manifest.json. H200step_main predict-only passed with3eligible nodes; actual worker scheduledgpu-h200-0761 after a briefqueue. Runtime scope counts, rank dispersion and whole-batch inflation are the acceptance evidence; source/launch PASS does not establish numerical fidelity.
