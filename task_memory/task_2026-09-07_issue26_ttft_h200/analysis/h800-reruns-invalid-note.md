## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-09-09 | Confirmed all three H800 diagnostic reruns initialize then fail CUDA OOM; boundary rows are invalid pre-request artifacts. |

H800 job names:

- `yc26-h800-ar-normal-20260909-02`
- `yc26-h800-ar-skip-20260909-02`
- `yc26-h800-ar-scalar-20260909-03`

All platform phases report `Succeeded`, but this only means the worker process exited. Every run has zero-byte `server.batch.*.jsonl` and `server.ops.*.jsonl`, no client success artifact, and 144 boundary rows/rank generated before API readiness. Every server log reports:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.37 GiB. GPU total 79.19 GiB; 477.06 MiB free; 78.71 GiB in use.
RuntimeError: Engine core initialization failed
```

The H800 probe confirms physical `NVIDIA H800`, 81559 MiB. This is a memory-capacity/configuration failure, not a post-MoE timing result. Do not reconcile these rows. A valid H800 run requires a memory-feasible model/runtime configuration (for example, lower KV-cache allocation if model weights fit); changing this would alter the frozen case and needs parent-level scope decision.
