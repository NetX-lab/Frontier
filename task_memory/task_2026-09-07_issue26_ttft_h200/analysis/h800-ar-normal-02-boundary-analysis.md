## Modification History

| Date | Summary of Changes |
|---|---|
| 2026-09-09 | Audited H800 normal-02 boundary artifacts; run invalid due CUDA OOM during engine initialization. |

# H800 normal-02 boundary audit

## Verdict

This run cannot provide valid first-forward timing evidence. Although each rank has 144 boundary rows (48 `combine_return`, 48 `tp_ar_call`, 48 `tp_ar_return`) with shape `[16384, 2048]`, `server.log` records CUDA out-of-memory during engine/KV-cache initialization before the API server became usable. The boundary rows were emitted during model initialization/profiling, not a served first-forward request.

## Evidence

- `operators/mode_manifest.json` declares `warmup_rounds=3`, `formal_requests=100`, and diagnostic mode `normal`.
- Each `moe_boundary.rank{0..7}.jsonl` contains exactly 144 rows: 48 rows for each phase and 48 distinct layers. This confirms logger execution but does not establish request execution because request/batch IDs are absent.
- Every inspected boundary row uses `shape=[16384,2048]`, `numel=33554432`, `dtype=torch.bfloat16`.
- `operators/server.log` reports at 08:35:44 `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.37 GiB; GPU capacity 79.19 GiB, only 477.06 MiB free`, followed by `RuntimeError: Engine core initialization failed`.
- All `server.batch.*.jsonl` and `server.ops.*.jsonl` files are zero bytes; no client log or formal request artifact is present.

## Diagnostic calculations (non-evidentiary)

Host timestamps show large offsets between ranks even for layer 0 (combine-return relative offsets approximately rank0 3.186 ms, rank1 44.527 ms, rank2 234.068 ms, rank3 543.864 ms, rank4 0 ms, rank5 62.858 ms, rank6 236.981 ms, rank7 553.860 ms). Per-rank `tp_ar_call - combine_return` gaps are approximately 0.7–13.1 ms on layer 0, and scope durations range from 0.875 to 43.85 ms. These values are initialization/profiling artifacts under a failed run and must not be interpreted as TP participant arrival or AR wait behavior.

## Consequence

Do not align this run with H200 normal/scalar/skip or use it for RCA. A valid H800 rerun must complete engine initialization, produce a client success response, and emit non-empty request/batch/operator artifacts before boundary reconciliation. The OOM likely requires reducing H800 KV-cache allocation (`num_gpu_blocks_override`) or using the validated H800 memory setting in the worker recipe; this is an execution configuration issue, not evidence about post-MoE AR causality.

## Cross-mode H800 status check

The corresponding H800 `skip-02` and `scalar-03` artifacts exhibit the same pattern: 144 boundary rows per rank but zero-byte batch/operator streams, and their `server.log` files contain the same 2.37 GiB CUDA OOM during initialization. Their platform phase is `Succeeded` because the worker exited cleanly after the diagnostic command; this does not mean the vLLM case succeeded. All three H800 modes therefore require a memory-corrected rerun before reconciliation.
