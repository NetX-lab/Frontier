## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Recorded the completed H200 PPLX clean boundary replays for runs 03 and 05, their gates, and measurement limits. |

# H200 PPLX clean boundary replays — runs 03 and 05

## Execution

Both runs used the frozen H200 calibration case and the independent clean-boundary
recorder. The worker source is
`tests/e2e/issue26_h200_pplx_clean_boundary_worker.sh`; it runs from the Frontier
worktree mounted at `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.
The vLLM compatibility source was pinned to
`/data/ycfeng/tmp/issue26-vllm-pplx-boundary-20260913-wt` at commit
`448f2b65e7679ae7490114ad382b6ba79becb3c3`.

The two RJobs were:

- `yc26-h200-pplx-clean-20260913-03`, output
  `analysis/pplx-clean-20260913-run03/`, phase `Succeeded`.
- `yc26-h200-pplx-clean-20260913-05`, output
  `analysis/pplx-clean-boundary-20260913-run05/`, phase `Succeeded`.

Both used H200 `step_main`, selector `h200`, one 8-GPU worker,
`num_gpu_blocks_override=310809`, TP4/DP2/EP8/PP1, BF16, eager execution,
FlashInfer, uniform routing, prefix caching OFF, chunked prefill OFF,
`VLLM_ALL2ALL_BACKEND=pplx`, and `ISSUE26_MOE_DP_CHUNK_SIZE=4096`.
Each used ten complete drained 100-request warmup replays followed by 100
formal requests at QPS 2, for 1100 client rows.

The worker fix committed as `a26bf034` keeps the launch root separate from probe
output: probe artifacts are under `<run>/preflight`, and replay artifacts are
under `<run>/runtime`. This avoids the environment probe's intentional
already-existing-root rejection when `launch.sh` is present.

## Criteria and evidence

| Criterion | Run 03 | Run 05 |
| --- | --- | --- |
| RJob terminal phase | `Succeeded` | `Succeeded` |
| Warmup replay lines | 10 (`replay` 0–9) | 10 (`replay` 0–9) |
| Client rows | 1100 | 1100 |
| Formal rows / unique IDs | 100 / 100 | 100 / 100 |
| First formal client ID | `pf4096_dc1024:0` | `pf4096_dc1024:0` |
| First formal server ID | `cmpl-pf4096_dc1024:0-0` | `cmpl-pf4096_dc1024:0-0` |
| First formal predicates | batch size 1, 4096 prefill, 0 decode | batch size 1, 4096 prefill, 0 decode |
| Selected DP lane | DP0 | DP0 |
| TP boundary rows | TP0–TP3, one each | TP0–TP3, one each |
| Analyzer | `boundary_analysis.json`: `status=PASS` | `boundary_analysis.json`: `status=PASS` |

Primary analyzer artifacts:

- [run 03 boundary_analysis.json](/data/ycfeng/stepfun-performance-optimization/Frontier/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-20260913-run03/runtime/boundary_analysis.json)
- [run 05 boundary_analysis.json](/data/ycfeng/stepfun-performance-optimization/Frontier/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/pplx-clean-boundary-20260913-run05/runtime/boundary_analysis.json)

Independent recomputation from the four TP JSONL rows gives:

| Run | TP0 (ms) | TP1 (ms) | TP2 (ms) | TP3 (ms) | Median (ms) | P90 (ms) | Rank max (ms) | Rank spread (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 03 | 182.934814 | 182.368317 | 183.000671 | 182.785339 | 182.860077 | 182.980914 | 183.000671 | 0.632355 |
| 05 | 183.506790 | 183.467133 | 183.122406 | 183.405411 | 183.436272 | 183.494893 | 183.506790 | 0.384384 |

The two run medians differ by `0.576195 ms` (`0.315%` relative to run 03).
The observed rank spread is sub-millisecond in both runs.

## Interpretation and limits

These are valid PPLX clean-boundary artifacts for the selected first formal
batch. The boundary is an independent CUDA-event envelope from model-forward
entry to return on each rank, with one post-end `torch.cuda.synchronize()` only
to read the elapsed event. Frontier per-op, batch, scheduler, routing,
completion, and full diagnostic loggers were disabled.

The envelope can include queued device work already ordered on the model
forward stream and excludes host wall-clock gaps. It is therefore not a
pure-kernel sum, not a rank-local communication sum, and not the old full
instrumented outer span. These runs do not provide per-operator attribution,
communication-kernel breakdown, queue-wait attribution, or a clean/diagnostic
reconciliation input. The `~183 ms` PPLX boundary should be compared only as a
PPLX protocol variant against the accepted native/naive clean reference; it
must not be used to modify Frontier predictors, communication models, CPU
accounting, or production settings.
