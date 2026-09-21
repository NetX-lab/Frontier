## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-11 | Recorded the completed H200 CUDA completion capture for the first formal 4096-prefill batch. |

# H200 CUDA Completion Capture

## Test Script Information

- Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`
- Worker script: `tests/e2e/issue26_h200_completion_capture_worker.sh`
- Standard diagnostic chain invoked by the worker:
  `issue26_h200_diagnostics_worker.sh batch` -> `issue26_token_id_client.py`.
  The diagnostic worker uses the existing environment probe and the approved batch replay path.
- Persistent output:
  `task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-completion-normal-03/`
- RJob: `yc26-h200-completion-normal-20260911-03`
- Diagnostic vLLM source: `/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab`
- Diagnostic vLLM commit: `63ef916e7c5817980f05a9f520fe5b4368393101`
- H200 allocation: `charged-group=step_main`, `positive-tags=h200`, `gpu=8`, `cpu=64`, `memory=409600`, one node.
- Workload: Qwen3-30B-A3B, TP4/DP2/EP8/PP1, BF16, eager execution, `FLASHINFER`, uniform routing, 4096 prefill tokens and 1024 output tokens, prefix caching OFF, chunked prefill OFF, dummy model loading.
- vLLM runtime settings: `max_model_len=16384`, `max_num_batched_tokens=16384`, `max_num_seqs=1024`, block size 16, `num-gpu-blocks-override=310809`, seed 0, naive all-to-all backend.
- GPU environment: `/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python`, Python 3.10.16, Torch 2.8.0+cu128, FlashInfer 0.3.0, CUDA 12.8.93.
- Exact launch command recorded in `analysis/h200-completion-normal-03/launch.sh`:

  ```bash
  /kubebrain/rlaunch --detach \
    --name yc26-h200-completion-normal-20260911-03 \
    --charged-group=step_main --private-machine=group --positive-tags=h200 \
    --set-env=ISSUE26_DIAGNOSTIC_VLLM_SOURCE=/data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab \
    --set-env=ISSUE26_DIAGNOSTIC_VLLM_COMMIT=63ef916e7c5817980f05a9f520fe5b4368393101 \
    --gpu=8 --cpu=64 --memory=409600 --backoff-limit=1 --max-wait-duration=2h \
    --enable-sshd=false \
    --image hub.i.basemind.com/vllm-0.10.2/frontier-env@sha256:b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc \
    --volume /data:/data \
    --workdir /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907 \
    -- bash tests/e2e/issue26_h200_completion_capture_worker.sh \
      task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-completion-normal-03
  ```

- Reproducible in-worker command (requires a fresh output directory and the same diagnostic source):

  ```bash
  bash tests/e2e/issue26_h200_completion_capture_worker.sh \
    task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-completion-normal-03
  ```

  The recorded output directory already exists; the worker intentionally rejects reuse of an existing probe/output directory. The command above documents the worker invocation and is not a request to rerun this completed experiment.

## Validation Criteria

- Complete three drained 100-request warmup replays followed by 100 formal requests.
- Produce 400 client rows and complete drain/shutdown with worker exit code 0.
- Pass the eight-worker identity validator and retain the formal request identity.
- Select the first formal DP0 batch with request `cmpl-pf4096_dc1024:0-0`, batch size 1, 4096 prefill tokens, 0 decode tokens, and `batch_dp_token_counts=[4096,1]`.
- Retain eight rank-separated completion files (`server.moe_completion.rank0.jsonl` through `rank7.jsonl`).
- For DP0 TP0-TP3, record 48 `combine` rows and 48 post-MoE `tp_ar` rows per rank, covering all 48 model layers.
- Compare rank-local CUDA completion timings, rank start skew, and the outer batch boundary span. Treat the capture as diagnostic evidence, because it adds a batch-level synchronization and per-forward file I/O.

## Test Results and Evidence

Status: `PASS_DIAGNOSTIC_CAPTURE`.

The RJob completed with platform status `Succeeded` and worker exit code 0. The identity artifact reports eight workers and `status=PASS`. Each worker completed the full replay; DP0 TP0-TP3 reported 5362 total batches, 1442 formal batches, and first formal `batch_id=3920`. DP1 workers independently reported first formal `batch_id=3934`; only DP0 is used for the requested first-batch analysis.

The selected first formal batch is:

| Field | Observed value |
| --- | --- |
| Request | `cmpl-pf4096_dc1024:0-0` |
| `batch_id` | 3920 |
| `batch_size` | 1 |
| `batch_num_tokens` | 4096 |
| `batch_num_prefill_tokens` | 4096 |
| `batch_num_decode_tokens` | 0 |
| `batch_dp_token_counts` | `[4096, 1]` |

The outer batch boundary spans from the eight batch boundary logs for DP0 TP0-TP3 were:

| TP rank | Outer span (ms) |
| ---: | ---: |
| TP0 | 88.206016541 |
| TP1 | 88.172286987 |
| TP2 | 88.200737000 |
| TP3 | 88.078498840 |

Across these four ranks, the rank max is `88.206016541 ms`, the minimum is `88.078498840 ms`, and the rank spread is `0.127517700 ms`.

Completion timing for the same batch is:

| TP rank | Combine sum (ms) | Combine median (ms) | Post-MoE TP AR sum (ms) | TP AR median (ms) | TP AR max (ms) | Previous TP AR -> next combine event gap median (ms) | Host event span (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP0 | 3.950784 | 0.081936 | 27.861600 | 0.619088 | 1.199680 | 1.075168 | 84.138217 |
| TP1 | 3.929312 | 0.081776 | 29.591680 | 0.623024 | 1.459936 | 1.075648 | 83.750530 |
| TP2 | 3.936416 | 0.082160 | 28.826112 | 0.618544 | 1.379776 | 1.082400 | 83.380903 |
| TP3 | 3.938368 | 0.081824 | 4.802144 | 0.100000 | 0.102112 | 1.597984 | 85.374932 |

All four DP0 ranks have exactly 96 completion rows: 48 `combine` and 48 `tp_ar`. The rank start spread is `1.909675 ms` median and `2.222244 ms` maximum for `combine`; it is `1.920461 ms` median and `2.222540 ms` maximum for `tp_ar`. TP3 is the latest `combine` and `tp_ar` starter on all 48 layers, and its previous-TP-AR-to-next-combine gap is larger than TP0-TP2 on all 48 layers.

These observations support a rank-level collective-arrival interpretation: TP0-TP2 enter the post-MoE TP AR earlier and their event durations include waiting for the late rank, while TP3 enters later and therefore records a shorter rank-local event duration. The inclusive rank-local TP AR sums cannot be added across ranks to form a physical collective duration. The near-equal outer spans are consistent with the late rank's larger pre-combine gap compensating for its shorter recorded TP AR duration.

Primary evidence files:

- `analysis/h200-completion-normal-03/completion_first_formal_summary.json`
- `analysis/h200-completion-normal-03/completion_summary.log`
- `analysis/h200-completion-normal-03/identity_validation.json`
- `analysis/h200-completion-normal-03/identity_validation.log`
- `analysis/h200-completion-normal-03/batch/runtime/batch/server.batch.dp0.tp{0,1,2,3}.pp0.jsonl`
- `analysis/h200-completion-normal-03/batch/runtime/batch/server.moe_completion.rank{0,1,2,3}.jsonl`
- `analysis/h200-completion-normal-03/launch.sh`
- `analysis/h200-completion-normal-03/submit.log`

## Limitations and RCA Boundary

- This is a diagnostic capture, not a clean CUDA batch-span measurement. The completion source performs one batch-level `torch.cuda.synchronize()` before flushing and writes per-forward completion records. These actions can perturb later scheduling.
- CUDA event-to-event intervals do not independently separate host enqueue delay, stream idle time, already queued work, or NCCL internal completion. The selected rows all report CUDA stream 0, but the capture records only the start-event stream metadata.
- The result closes the observed rank-level arrival skew for this selected batch: TP3 is consistently late across all 48 layers and its short post-MoE TP AR duration is compatible with late collective arrival. It does not identify which preceding local kernel, host scheduling segment, or queue event causes TP3 to start late.
- The result does not establish a clean-vs-diagnostic reconciliation, does not close the Frontier-vLLM operator gap gate, and does not authorize a Frontier production profiling/predictor/communication correction or a CPU add-on.
- The observed outer span is in the established H200 diagnostic scale of roughly 70-110 ms, but it must not be substituted for the clean normal/skip ABBA spans.

