## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded eight-H200 clock and endpoint-overhead checks. |

# Minimal Prefill Endpoint Recorder: GPU Clock Check

Execution: tests/e2e/issue26_h200_endpoint_worker.sh sourced the exact image environment probe and invoked /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python tests/performance/issue26_prefill_endpoint_probe.py <fresh-output>/clock-check. Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0. Worker yc26-h200-endpoint-20260908-0018 used step_main, eight H200 GPUs, and the same pinned image. The vLLM measurement extension was uncommitted experimental code; this is preflight validation, not clean formal parity evidence.

Exact launcher receipt:

```bash
#!/usr/bin/env bash
set -euo pipefail
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy" ALL_PROXY="$all_proxy" NO_PROXY="$no_proxy,127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"
exec /kubebrain/rlaunch --detach --name yc26-h200-endpoint-20260908-0018 --charged-group=step_main --private-machine=group --positive-tags=h200 --gpu=8 --cpu=64 --memory=409600 --backoff-limit=1 --max-wait-duration=2h --enable-sshd=false --image hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc --volume /data:/data --workdir /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907 -- bash tests/e2e/issue26_h200_endpoint_worker.sh /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-endpoint-runtime-01
```

Criteria: for each GPU, 64 unique endpoint records; existing natural output synchronization completes the recorded CUDA event; host anchor bracket <=100 us; independent current GPU/host clock brackets separated by <=100 us. The 100 us limit is a provisional instrumentation precision criterion, not a TTFT error tolerance. Formal case admission must additionally compare uncertainty and overhead with its newly measured TTFT.

Procedure: BF16 (4096,2048) x (2048,2048) matmul, three warmups, 64 interleaved baseline/recorded pairs per GPU. Recorder initialization synchronizes only before measured work. Each measured completion uses the existing output .item() synchronization analogue; no added batch synchronize.

PASS: 512/512 unique endpoint records across eight H200 GPUs; anchor interval widths 14.8546–19.8241 us; independent interval separation 0 us on all GPUs; observed mean recorder overhead 17.9810–22.0366 us. PREFILL_ENDPOINT_CLOCK_PASS observed. This short-duration check does not establish long-run clock drift, full-vLLM overhead, scheduler request-ID joins, or canonical TTFT parity.

Detailed numeric rows are in runs/h200-endpoint-runtime-01/clock-check/clock_overhead.json, raw endpoint records beside it, and the full worker log in runs/h200-endpoint-runtime-01/environment.log. All are new measurements from this task and are infrastructure checks, not operator predictor training rows.
