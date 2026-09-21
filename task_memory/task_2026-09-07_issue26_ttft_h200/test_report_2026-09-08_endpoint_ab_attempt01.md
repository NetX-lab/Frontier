## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Preserved the failed first A/B attempt and scoped client correction. |

# Endpoint A/B attempt 01

## Execution

Worker: `tests/e2e/issue26_h200_endpoint_ab_worker.sh`, invoked with the absolute task run path `runs/h200-endpoint-ab-01` through `/data/ycfeng/tmp/issue26-h200-network/launch-endpoint-ab-01.sh`. That script retains the exact image digest, H200 step_main allocation, proxy configuration, and full command.
Python: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Python 3.10.16; Torch 2.8.0+cu128; FlashInfer 0.3.0. vLLM head db3fde591 plus the retained integration diff in the run root.

## Criteria

Each mode must complete three full 100-request warmup replays followed by 100 unique formal requests at 4096 input / 1024 output tokens. All replay barriers must drain, and the endpoint mode must join queue arrivals with TP-rank completion records. No incomplete attempt is admitted as clean parity evidence.

## Evidence

**FAIL.** All 100 replay-0 requests completed with 4096 input and 1024 output tokens. Before replay 1 could proceed, aiohttp failed writing a POST with `ClientConnectionResetError: Cannot write to closing transport`, followed by `ClientOSError: Can not write request body`. Client failure triggered worker EXIT cleanup. The server's first shutdown message followed the client failure; no preceding engine/CUDA error was observed.

The long-lived client session reused a pool across streamed replays. vLLM's HTTP keepalive timeout is 5 seconds (`vllm/envs.py`). The scoped correction makes the connection lifetime request-local with `TCPConnector(force_close=True, limit=concurrency)`, retaining the same arrivals and no automatic retry. Attempt 02 is the actual workload verification.

Additional local mock check could not run: the host default Python and the CPU simulator conda environment do not have aiohttp. No dependency was installed for this optional check. The image already has aiohttp and is running the corrected real workload. Syntax checks passed; that alone does not validate transport behavior.

Canonical simulation predicted/actual/absolute/relative error: unavailable; this integration attempt produced no formal comparison.

## Corrected client: actual workload PASS

Attempt 02 baseline completed 400/400 unique requests, all with 4096 input tokens and 1024 observed output tokens. `client.log` records four non-overlapping drained replay intervals and 100/100 unique formal IDs. The same worker command uses fresh output `runs/h200-endpoint-ab-02`, launched by `/data/ycfeng/tmp/issue26-h200-network/launch-endpoint-ab-02.sh`. No connection reset recurred across any boundary. This validates the scoped client repair; endpoint-on A/B validation is still running.
