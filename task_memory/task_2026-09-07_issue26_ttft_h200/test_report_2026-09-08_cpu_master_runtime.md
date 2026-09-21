## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified CPU master simulator runtime, real collective backend, and isolated preparation worker. |

# CPU master Frontier preparation

## Execution

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
Python: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, conda environment `dev-vidur-v03-hopper-e2e`, Python 3.13.13.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_simulator_runtime_check.py task_memory/task_2026-09-07_issue26_ttft_h200/analysis/cpu-master-runtime-01/runtime.json
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_collective_sim_zero_payload.py -k test_real_predictor_retains_intra_server_latency -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/issue26-cpu-master-htsim-01
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_cpu_frontier_worker.py --config /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/config/frontier_run_01.json --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/cpu-master-runtime-01/prepare --prepare-only
```

The last command was executed again to verify rejection of an existing output directory. Reproduce the successful preparation with a new output path; the recorded directory must remain unchanged.

## Criteria and evidence

- PASS: nine dependency imports, full `frontier.simulator.Simulator` import, and native RandomForest fit/predict. Exact package versions and paths: `analysis/cpu-master-runtime-01/runtime.json`.
- PASS: the real Scenario-to-predictor-to-htsim runner boundary for zero and 32768-byte single-server EP8 payloads: `2 passed, 3 deselected in 0.64s`. The expected latency is `(7 * 0.5 + (7 / 8 * payload) / (450e9 * 0.8) * 1e6) / 1000` ms, including 0.0035 ms at zero payload; network contribution is zero. This is backend/runtime validation, not a vLLM timing measurement.
- PASS: preparation executes the runtime audit, captures the full command and settings, allocates new caches below `/data/ycfeng/tmp`, and creates no simulator metrics or `frontier.log`. Existing output is rejected with `FileExistsError`, exit 1. Exact receipt and negative traceback: `analysis/cpu-master-runtime-01/worker-validation.json` and `existing-output-rejection.log`.
- The initial attempt to read a worktree-local `task_memory/env_handbook.md` reported that the file was absent. The authoritative original-repository handbook was then read; no environment failure or installation occurred.

## Scope and limitations

The worker accepts a frozen JSON config and an exclusive output directory; it preserves scenario settings while replacing metric/output/cache paths. It checks required input files and the selected collective_sim/nvlink_analytic backend. The preparation test used the existing baseline config only to validate command serialization and isolation; its profile/routing semantics are not certified for the new uniform run.

No GPU, Docker, network, dependency installation, model-cache reuse, or numerical simulation was performed. The CPU environment has sklearn 1.9.0, numpy 2.5.0, pandas 2.3.3, scipy 1.18.0; these differ from the H200 image, so fresh predictor training is mandatory. A complete uniform profile configuration and the chosen fresh arrival trace are pending from the main task. Launch after these inputs are frozen; remove `--prepare-only` and supply a new output directory.
