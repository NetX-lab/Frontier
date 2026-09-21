## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Invalidated the initial H200 normal/skip outer-span pair after warmup and DP-lane audit. |

# H200 diagnostic warmup and lane audit

The previously reported `18087.072266 ms` normal and `19656.929688 ms` skip values are not valid warmed first-forward batch spans for the requested DP0 comparison.

## Evidence

The diagnostic client is configured with `--warmups 3 --requests 100`. The batch logger starts its own `batch_id` sequence when Frontier instrumentation becomes active, but the recorded `request_ids` show that `batch_id=0` is still a client warmup request:

```text
normal DP0 batch_id=0: cmpl-warmup:pf4096_dc1024:r0:0-0, prefill=4096, 17960.621094 ms
skip   DP0 batch_id=0: cmpl-warmup:pf4096_dc1024:r0:0-0, prefill=4096, 19641.011719 ms
```

`VLLM_FRONTIER_TRACE_SKIP_WARMUP=1` suppresses Frontier tracing during vLLM engine/internal warmup. It does not suppress the client warmup rounds, which are deliberately logged after the server becomes ready.

The first formal request is assigned to DP1 in both runs, rather than DP0:

```text
normal DP1 batch_id=1024: cmpl-pf4096_dc1024:0-0, prefill=4096, 1194.216797–1194.692993 ms across TP0–TP3
skip   DP1 batch_id=0:    cmpl-pf4096_dc1024:0-0, prefill=4096, 1070.157959–1220.434448 ms across TP0–TP3
```

The normal formal request follows a DP1 warmup (`cmpl-warmup:...:r1:0-0`) and 1024 decode batches. The skip formal request has no preceding DP1 client warmup in its batch file. Consequently, the two formal rows also do not have equivalent per-lane warmup state.

## Consequence

The prior H200 normal/skip pair must be excluded from causal span comparison. Its 1.57 s skip-minus-normal difference combines a cold client-warmup row, different DP-lane warmup state, independent processes, and heavy diagnostic instrumentation. It cannot be used as post-MoE AR cost or as evidence that skip increases the batch span.

## Required rerun contract

The next H200 run must select a formal `cmpl-pf4096_dc1024:0-0` row after three completed client warmup rounds **on the same DP lane**, then restrict analysis to that lane's TP0–TP3 rows. The run must record an explicit request-to-batch identity and distinguish engine warmup, client warmup, formal prefill, and decode batches. The 70–110 ms historical reference can only be compared after confirming that the measurement uses the same batch-only scope and a low-overhead logger configuration; the current boundary logger and per-layer host file writes are known to introduce host-visible gaps into CUDA event spans.
