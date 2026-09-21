## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-10 | Recorded actual artifact validation, failed capture attempts and queued paired job. |

## Test Script Information

Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`.

Analyzer: `tests/e2e/issue26_standard_bypass_result.py`. Commands run from repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/e2e/issue26_standard_bypass_result.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-post-moe-bypass-03 --no-historical-reference --output /data/ycfeng/tmp/issue26-h800-paired-20260910/verified-bypass.json
PYTHONDONTWRITEBYTECODE=1 python3 tests/e2e/issue26_standard_bypass_result.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h800-standard-replay-176000-repro-01 --source /data/ycfeng/tmp/issue26-vllm-diagnostics-ar-ab --no-historical-reference --output /data/ycfeng/tmp/issue26-h800-paired-20260910/verified-normal.json
bash -n tests/e2e/issue26_h800_paired_replay_worker.sh
```

The output uses exclusive creation; select fresh output filenames when repeating. Local system Python3.12.3, no conda activation. GPU execution uses pinned image and vllm-bs-0.10.2 Python3.10.16.

## Validation Criteria

Each standard arm: clean and batch modes each400complete rows,3drained100-request warmups,100unique formal requests,4096prompt/1024output. Exact first request cmpl-pf4096_dc1024:0-0 with own4096scheduled tokens, DP0 TP0-3, single request,4096prefill/0decode. Uniform routing1, operator/boundary/profiler capture OFF. New pairs use explicit current-arm results without a historical numerical reference.

## Test Results and Evidence

Direct existing-artifact analysis PASS; this is analyzer validation, not fresh GPU measurement.

| Existing arm | Median ms | Four-rank P90 ms | Rank max ms | Spread ms | Clean/batch rows | Drain |
| --- | --- | --- | --- | --- | --- | --- |
| Normal | 77.483665466 | 77.486233521 | 77.486846924 | 0.026206970 | 400/400 | PASS |
| Minimal bypass | 77.935920715 | 78.081732941 | 78.142173767 | 0.222427368 | 400/400 | PASS |

Analyzer outputs persist in the exact paths above. Shell syntax and git diff whitespace checks PASS. Analyzer commit a27b80f5. Same-node paired worker remains uncommitted pending actual run.

Three prior capture jobs generated zero formal timing and zero trace:

- yc26-h800-nsys-first-formal-20260910-01: preflight only, fixed nsys executable check failed; no vLLM artifacts.
- yc26-h800-nsys-first-formal-20260910-02: environment.log line181: `Nsight Systems executable is unavailable in the worker image.`
- yc26-h800-torch-profiler-first-formal-20260910-01: environment.log line179: `fatal: detected dubious ownership in repository at /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`.

Platform Succeeded does not qualify these as test successes. Source f827a4ecc has crossed CUDA/Torch counters and a prefix environment-name mismatch; capture functional validation FAIL/not executed. Canonical normal and skip sources exclude these edits.

New job yc26-h800-paired-normal-skip-20260910-01: submitted15:24:33UTC, Pending15:28UTC, codesign-default position4/H800quota0. Same allocation normal_1->skip_1->skip_2->normal_2. No new latency or causal conclusion yet. Allocation submission missed predict-only; restore handbook preflight next allocation. Monitoring system scope failed interactive authentication; user scope with MemoryMax2G passed.
