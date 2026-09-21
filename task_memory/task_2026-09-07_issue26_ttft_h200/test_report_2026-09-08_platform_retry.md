## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded recovered platform queries and fresh H200 replay submission. |

# Platform retry and new replay

## Execution

CPU master, Bash, brainctl through MemoryMax=2G scopes. No Python environment is required for these platform checks. Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.

```bash
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy"
export NO_PROXY="$no_proxy,127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
timeout 30s systemd-run --user --scope -p MemoryMax=2G brainctl --request-timeout=15s get rjob yc26-h200-historical-replay-20260908-01 -n shai-core -o jsonpath='{.status.phase}'
timeout 30s systemd-run --user --scope -p MemoryMax=2G brainctl --request-timeout=15s get replica yc26-h200-historical-replay-20260908-01-cfcd2084 -n shai-core -o jsonpath='{.status}'
timeout 30s systemd-run --user --scope -p MemoryMax=2G bash /data/ycfeng/tmp/issue26-h200-network/predict-historical-replay-02.sh
```

Exact new workload command: `runs/h200-historical-replay-02/launch.sh`. Persistent launcher: user service `yc26-h200-historical-replay-20260908-02.service`, running `/bin/bash /data/ycfeng/tmp/issue26-h200-network/launch-historical-replay-02.sh`. Launcher output: `/data/ycfeng/tmp/issue26-h200-network/launcher-historical-replay-02.log`.

## Criteria and evidence

- Platform visibility PASS after synchronizing proxy variable case. Before synchronization, inherited uppercase HTTPS_PROXY used a localhost proxy and uppercase NO_PROXY did not bypass the platform domain; queries repeatedly returned EOF. Lowercase company variables alone were insufficient. Consecutive RJob, replica, and log queries then exited0.
- Old run completeness FAIL: replica exitCode0, reasonCompleted, finishedAt2026-09-08T03:31:16Z; client has232warmup rows and0formal rows, only replay0/replay1 completion records, and no full replay completion marker. Platform Succeeded is not numerical acceptance. Early worker termination cause remains unresolved.
- Fresh run preflight PASS: clean and diagnostic vLLM checkout branches, commits, clean trees, origin URLs, and current remote tip verified. Both mode manifests retain the approved TP4/DP2/EP8, BF16, uniform router, prefix-off/chunking-off/eager, three drained warmups, and100formal4096/1024 requests.
- H200 prediction PASS: eligible8GPU nodes1091,0819,0276. New job acknowledged and scheduled on gpu-h200-1091. Image digest unchanged; no H800 allocation.
- New runtime and numerical verification remain pending at this report's creation. No previous numerical data enters the new generation. No latency values are inferred from process exit codes.
