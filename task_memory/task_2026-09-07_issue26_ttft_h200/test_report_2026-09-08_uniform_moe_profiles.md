## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Completed774 fresh uniform_topk rows; documented and repaired merge output namespace/permissions. |

# Fresh uniform-routing MoE profiling

Status: completed; fresh data and corrected merge validated PASS. This collection supports only the current 4096/1024 H200 calibration case.

## Execution

Worker: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_h200_uniform_moe_profiles_worker.sh`.

The exact rlaunch command and immutable initial worker snapshot are recorded in `runs/h200-uniform-moe-profiles-01/run_manifest.json` and `worker.sh`. The worker records the complete profiler CLI via shell tracing in `runtime/environment.log` and each context has a separate profiler log.

Pinned image: `hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`.
Python: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, conda environment `vllm-bs-0.10.2`, Python 3.10.16. vLLM source commit `46f7b179fd3bf42b9616dc4670cba419afdb2085`.
H200 allocation: `step_main`, eight GPUs, 64 CPU cores, 409600 MiB. Company HTTP proxy is configured in the launcher and worker.

## Criteria

Each of prefill_hot and standalone_legacy must produce 387 fresh rows: 43 token points, three load distributions, and three seeds. Every row must use uniform_topk routing, round_robin_uniform assignment, uniform weights, BF16, CUDA_EVENT, TP1, and EP8. All four timing medians must be positive and finite. The complete merged dataset must retain all 774 rows, without dropped rows or standard_fused_topk rows. Selected context filters must each return exactly387 rows.

## Evidence

- Shell syntax and embedded Python compilation: PASS. An initial extra closing parenthesis was caught and corrected before launch.
- predict-only: PASS, three available eight-GPU nodes. See `analysis/uniform-moe-profiles-01/predict-only.log`.
- RJob `yc26-h200-uniform-moe-profiles-20260908-01` created at 2026-09-08 00:44:23 UTC; scheduled on `gpu-h200-0043.lgcm.sh.istep.fun`.
- Runtime probe: PASS, eight NVIDIA H200 GPUs with active NVLink and successful matrix multiplication.
- Preflight follow-up identified that the existing merge tool requires outputs beneath the task supplements directory. The initially prepared runtime/merged output must be corrected before publication; raw profile collection is unaffected.

## Limits

This is new measured operator data, not clean request-level E2E evidence. Existing grouped GEMM profiling omits the gated activation; this collection does not repair that independent known issue. Uniform routing implementation metadata does not establish identical batch composition between Frontier and vLLM.

## Completion and recovery

Both GPU collections completed on the first RJob:387 prefill_hot rows in38 seconds and387 standalone_legacy rows in29 seconds (profiling-loop durations). The worker subsequently exited1 because the merge output was outside the tool's required supplements namespace. The original launch snapshot and failure stack trace are preserved under `runs/h200-uniform-moe-profiles-01/worker.sh` and `runtime/environment.log`.

CPU recovery used the exact command vector in `analysis/uniform-moe-profiles-01/cpu-merge-command.json` with `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, conda `dev-vidur-v03-hopper-e2e`, Python3.13.13. The first CPU attempt failed with PermissionError because the supplements parent is root-owned. Created only the new approved `supplements/moe-uniform-01` directory with owner/group10250 through `sudo -n install -d -o10250 -g10250`; second attempt returned0. Both failure and successful logs are preserved as `cpu-merge.log` and `cpu-merge-retry.log`.

Reproducible merge invocation (requires the declared output paths to be fresh; for another iteration select a new supplements directory):

```python
import json
import subprocess
from pathlib import Path
case = Path("task_memory/task_2026-09-07_issue26_ttft_h200")
command = json.loads((case / "analysis/uniform-moe-profiles-01/cpu-merge-command.json").read_text())
subprocess.run(command, check=True)
```

Selected final evidence:

- `supplements/moe-uniform-01/moe.csv`:774 rows,1843915 bytes.
- `supplements/moe-uniform-01/moe.summary.json`: all387 base rows retained,387 added, dropped_rows empty.
- `supplements/moe-uniform-01/moe.complete.json`: COMMITTED; both artifact hashes and sizes validated.
- `analysis/uniform-moe-profiles-01/validation.json`: PASS; exact128 experts/topk8/hidden2048/intermediate768, zero duplicate identities, all four timing medians finite and positive, both context filters387 rows. Each context includes43 token points ×3 distributions ×3 seeds (0,1,2). All source columns and row values remain unchanged after merge (floating-point comparison tolerance1e-12).

The durable worker was corrected only after the launched process had exited: its future merge output uses `supplements/<run-generation>/moe.csv`, and verifies both the existing tool byte count and hash. The launched worker snapshot remains unchanged. Shell and embedded Python syntax passed after correction. The corrected merge path was validated through the CPU recovery; a second GPU collection was unnecessary because raw measurement generation succeeded unchanged.

No pending work remains in this profiling subtask. The parent calibration task owns CPU Frontier execution and the separate clean vLLM rerun.
