## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-11 | Recorded the completed H200 same-node normal/skip ABBA replay and first-formal CUDA span validation. |

# Test Script Information

- Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`
- GPU worker script: `tests/e2e/issue26_h200_paired_replay_worker.sh`
- Launch wrapper: `/data/ycfeng/tmp/issue26-h200-paired-20260911-02/launch.sh`
- Reproducible worker command:

  ```bash
  bash tests/e2e/issue26_h200_paired_replay_worker.sh \
    task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-paired-normal-skip-02
  ```

- RJob: `yc26-h200-paired-normal-skip-20260911-02`
- Cluster configuration: `charged-group=step_main`, `positive-tags=h200`, `gpu=8`, `cpu=64`, `memory=409600`, `num-gpu-blocks-override=310809`.
- Workload: Qwen3-30B-A3B, TP4/DP2/EP8/PP1, BF16, eager, FLASHINFER, 4096-prefill/1024-output, uniform routing, prefix caching OFF, chunked prefill OFF.
- Python environment: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0.
- Arm order: `normal_1 -> skip_1 -> skip_2 -> normal_2`, all on node `rjob-288c5724ad84ba30-52045d57e7718f09-0`.

# Validation Criteria

- Each arm completes 3 drained 100-request warmup replays and 100 formal requests.
- Each arm records 400 clean rows and 400 batch rows, with full drain and identity validator PASS.
- First formal batch satisfies request `cmpl-pf4096_dc1024:0-0`, batch size 1, 4096 prefill tokens, 0 decode tokens, DP0 TP0–TP3, and `batch_dp_token_counts=[4096,1]`.
- Report median, P90, rank max, rank spread over DP0 TP0–TP3.
- Compare `skip_1 - normal_1` and `skip_2 - normal_2`; use the result only to test whether minimal post-MoE AR bypass inflates the outer batch span.

# Test Results and Evidence

All four arms returned `PASS_STANDARD_IDENTITY_AND_DRAIN` and worker exit code 0. The first-formal DP0 TP0–TP3 statistics are:

| Arm | Median (ms) | P90 (ms) | Rank max (ms) | Rank spread (ms) |
| --- | ---: | ---: | ---: | ---: |
| normal_1 | 81.084751129 | 81.096882629 | 81.099166870 | 0.101020813 |
| skip_1 | 80.058879852 | 80.318404388 | 80.426269531 | 0.510169983 |
| skip_2 | 80.260639191 | 80.642729950 | 80.786048889 | 0.630340576 |
| normal_2 | 84.223361969 | 84.401792145 | 84.403327942 | 0.473953247 |

Pair deltas:

- `skip_1 - normal_1`: `-1.025871277 ms` (`-1.2652%` median), P90 `-0.778478241 ms`, rank max `-0.672897339 ms`.
- `skip_2 - normal_2`: `-3.962722778 ms` (`-4.7050%` median), P90 `-3.759062195 ms`, rank max `-3.617279053 ms`.

The two normal arms differ by `+3.138610840 ms` (normal_2 slower), so the bypass delta magnitude is not stable even though both deltas are negative. Every median remains in the established H200 normal 70–110 ms scale. The result rejects the earlier 5× slowdown as a consequence of commenting only the post-MoE TP AR call under the standard warmed chain. It does not identify the remaining completion/queue residual and does not authorize clean/diagnostic span reconciliation or a production profiling correction.

Primary artifacts:

- `analysis/h200-paired-normal-skip-02/{normal_1,skip_1,skip_2,normal_2}/identity_validation.json`
- `analysis/h200-paired-normal-skip-02/{normal_1,skip_1,skip_2,normal_2}/standard_result.json`
- `analysis/h200-paired-normal-skip-02/worker_exit.txt` (`worker_exit_code=0`)
- `analysis/h200-paired-normal-skip-02/worker.log` (all four `ARM_COMPLETE` markers)
