## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified source-equivalent count snapshots and real DES load-feedback callbacks. |

# D016 snapshot-policy validation

Authorization: YC explicitly approved RR correction plus the separate vLLM DP policy with load snapshots. Receipt: repairs/d016_human_review.json. Change marker: D016_VLLM_DP_SNAPSHOT.

## Environment and execution

Working directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907.
CPU conda environment: dev-vidur-v03-hopper-e2e; Python3.13.13.
Pinned reference source: /data/ycfeng/tmp/vLLM-BS/vllm/v1/engine/coordinator.py at46f7b179fd3bf42b9616dc4670cba419afdb2085, verified against upstream v0.10.2.

Focused checks:

```bash
export TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_vllm_dp_load_balancer.py tests/unit/test_cluster_scheduler_dp_lanes.py tests/unit/test_prefix_cache_scheduler_frontier.py tests/unit/test_pdaf_config_contract.py tests/unit/test_request_generator_decode_bound_count.py -q -p no:cacheprovider --tb=short
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/integration/issue26_dp_coordinator_reference.py --source /data/ycfeng/tmp/vLLM-BS/vllm/v1/engine/coordinator.py --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/dp_coordinator_reference_check.json
git diff --check
```

Real DES verification preserves the current Qwen TP4/DP2/PP1/EP8 and4096/1024 shape, with three requests from the beginning of the already validated current-task trace. It explicitly uses dummy operator timing for control-flow verification. It is not a new numerical ground-truth pair.

Prepared command: /data/ycfeng/tmp/issue26-dp-snapshot-validation-9jxdy2bq/runtime/command.json. This contains the complete expanded frontier.main invocation and fresh cache/output paths.
Input config: config/frontier_dp_snapshot_diagnostic.json.
Preparation and execution:

```bash
export TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_cpu_frontier_worker.py --config task_memory/task_2026-09-07_issue26_ttft_h200/config/frontier_dp_snapshot_diagnostic.json --output /data/ycfeng/tmp/issue26-dp-snapshot-validation-9jxdy2bq/runtime --prepare-only
systemd-run --user --unit=yc26-cpu-dp-snapshot-validation-20260908-01 --property=MemoryMax=8G --property=WorkingDirectory="$PWD" --property=StandardOutput=file:/data/ycfeng/tmp/issue26-dp-snapshot-validation-9jxdy2bq/runtime/frontier.log --property=StandardError=inherit --setenv=FRONTIER_LOG_LEVEL=ERROR --setenv=PYTHONPATH="$PWD" --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=TMPDIR=/data/ycfeng/tmp --setenv=WANDB_DISABLED=true --setenv=VIDUR_DISABLE_WANDB=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/integration/issue26_dp_snapshot_diagnostic.py --command /data/ycfeng/tmp/issue26-dp-snapshot-validation-9jxdy2bq/runtime/command.json --output "$PWD/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/dp_snapshot_des_feedback.json"
```

The commands above are the recorded execution. A repetition must choose a fresh output directory/unit and prepare a new command.json; the worker intentionally rejects an existing output directory.

## Criteria and evidence

- PASS:73 focused checks in5.69s. Weighted waiting/running scores, fixed tie, local increments, snapshot replacement, same-step gathering, prior-step counts, heartbeat, unchanged-report suppression, timeout ties, config discovery and existing regression coverage pass.
- PASS: Actual upstream coordinator loop and Frontier model agree at6001millisecond observations over a scripted6-second count sequence. Observed publications:70,170,270,370,470,5470ms. Artifact: analysis/dp_coordinator_reference_check.json.
- PASS: Final DES service exited0;159907events processed;3/3unique requests completed4096prefill+1024decode tokens. Actual owners: request0->DP0, request1->DP0, request2->DP1.
- PASS:2049real post-step callbacks observed: DP0=1025, DP1=1024. Every report equals the corresponding real V1 scheduler load. Observed load transitions include DP0(1waiting,1running)->(0,2)->(0,1)->(0,0), DP1(0,1)->(0,0). This verifies waiting/admitted/completed populations through actual simulation callbacks, beyond supplied-state unit checks.
- PASS: Both lanes end with waiting0/running0. Current case config selects vllm_load_balancing, collective_sim and nvlink_analytic, prefix caching OFF.
- PASS: git diff --check.

Selected durable evidence:

- analysis/dp_snapshot_des_feedback.json
- analysis/dp_snapshot_final_des_validation.json
- analysis/dp_snapshot_final_des_evidence/config.json
- analysis/dp_snapshot_final_des_evidence/cluster.json
- analysis/dp_snapshot_final_des_evidence/request_metrics.csv
- analysis/dp_snapshot_final_des_evidence/metrics_ground_truth.jsonl

Raw event trace, stage ledger and logs remain under /data/ycfeng/tmp/issue26-dp-snapshot-validation-9jxdy2bq/runtime.

## Failures and corrections

- An initial manual registry probe passed a string to get_class, which requires ClusterSchedulerType; it raised ValueError. Using the existing enum API passed all six registrations. No registry behavior was changed to accommodate the probe.
- Initial direct reference-script invocation lacked PYTHONPATH and raised ModuleNotFoundError: No module named 'frontier'. Setting PYTHONPATH to the active worktree corrected the environment.
- Review found that unchanged counts must not be treated as a coordinator input at a publication deadline. Removed that premature timer advancement and added the same-time peer-report regression. The final source-reference and DES checks ran after this correction.
- The earlier DES pass was preliminary; final evidence above comes from the later callback-observed run after the boundary correction.

## Numerical limits and pending E2E

Dummy operator timings do not establish latency accuracy. Predicted TTFT, actual TTFT, absolute error and relative error for a fresh repaired pair: NOT AVAILABLE. The previous18.739% gap is not a result of this repair.

The coordinator check models deterministic zero-latency transport and empty bootstrap state. Real IPC latency and warmup-ending snapshot phase are not measured or fitted. Matching the algorithm does not guarantee identical placement across different execution timelines.

The requested full clean replay remains at232completed warmup records, last output03:31UTC, without100formal rows. Bounded platform queries return EOF; an explicit company-proxy connection returns502. Full numerical config config/frontier_dp_snapshot_candidate.json is prepared and points to the new clean arrival trace, which does not yet exist. No old arrival trace is substituted for this formal run. Restore platform visibility, obtain valid fresh clean/batch outputs, then execute that config and compare batches and official TTFT.
