## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Derived and validated the uniform clean01 engine interval for all 100 formal requests. |

# Uniform clean engine interval

PASS_ALGEBRAIC_DECOMPOSITION. Only `runs/h200-uniform-clean-01/runtime/clean/server.request_metrics.jsonl` and `analysis/uniform-clean-01/request_id_map.csv` contribute numerical evidence. The previous decomposition supplied the method, not data. All 100 per-request rows, raw duration operands, IDs, counts, and checks are retained in `uniform_clean_engine_prefill_decomposition.json`.

| Duration (ms) | Mean | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Official server TTFT | 127.380511760712 | 86.163759231567 | 246.113300323486 |
| First SCHEDULED to first EngineCoreOutput | 86.785534396767 | 80.065043643117 | 130.710240453482 |
| Remaining official interval | 40.594977363944 | 1.831116154790 | 117.320906370893 |

Source verification: run commit `46f7b179fd3bf42b9616dc4670cba419afdb2085`, empty run uncommitted patch, matching local HEAD and unchanged relevant source files. `vllm/v1/engine/output_processor.py:273-318` records model execution as `(last_token_ts - scheduled_ts)*1000`, and TPOT as `(last_token_ts - first_token_ts)/(decode_tokens-1)*1000`. `vllm/v1/metrics/stats.py:140-195` retains the first SCHEDULED timestamp and sets first/last token timestamps from EngineCoreOutput's monotonic timestamp. Therefore:

```text
scheduled_to_first_engine_output_ms = request_model_execution_time - tpot * (decode_tokens - 1)
remaining_official_interval_ms = official_server_ttft_ms - scheduled_to_first_engine_output_ms
```

Validation:400 unique server rows; exactly100 unique formal client/server/Frontier ID joins covering IDs0–99;300 warmup rows excluded; every formal row has4096prefill/1024decode tokens; all inputs and derived durations finite and nonnegative; official TTFT equals its mapping value exactly. Maximum per-row absolute recomposition error is0ms. These are algebraic and identity checks, not measured simulation error.

The strongest limit is that **40.595ms is not a CPU-only measurement or a Frontier correction**. It includes queue waiting and work outside the engine interval. The86.786ms engine interval includes host/device work and any intervening preemption, and ends at first EngineCoreOutput, not a separately timed GPU prefill endpoint. Only duration intervals are combined; subtraction of frontend wall-duration and engine monotonic-duration does not establish a single-clock causal partition or microsecond endpoint accuracy.

## Reproducible computation

Execution used `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, conda environment `dev-vidur-v03-hopper-e2e`, Python3.13.13. Run from `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`:

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'CHECK'
import csv, json, math, statistics
from pathlib import Path
task = Path('task_memory/task_2026-09-07_issue26_ttft_h200')
mapping = list(csv.DictReader((task / 'analysis/uniform-clean-01/request_id_map.csv').open()))
records = [json.loads(line) for line in (task / 'runs/h200-uniform-clean-01/runtime/clean/server.request_metrics.jsonl').open()]
by_id = {row['request_id']: row for row in records}
assert len(records) == len(by_id) == 400 and len(mapping) == 100
assert {int(row['frontier_request_id']) for row in mapping} == set(range(100))
values = []
for entry in mapping:
    row = by_id[entry['server_request_id']]
    assert row['request_num_prefill_tokens'] == 4096 and row['request_num_decode_tokens'] == 1024
    engine = row['request_model_execution_time'] - row['tpot'] * 1023
    remaining = row['ttft'] - engine
    assert row['ttft'] == float(entry['server_ttft_ms'])
    assert all(math.isfinite(x) and x >= 0 for x in (row['ttft'], engine, remaining))
    values.append((row['ttft'], engine, remaining))
for name, column in zip(('official', 'engine', 'remaining'), zip(*values)):
    print(name, 'mean', statistics.mean(column), 'min', min(column), 'max', max(column))
CHECK
```

No production/test files, config, historical numeric evidence, GPU resources, or network were modified or used. An initial guessed `official-uniform-clean-01` analysis path was absent; file inventory located the correct `uniform-clean-01` path before computation.
