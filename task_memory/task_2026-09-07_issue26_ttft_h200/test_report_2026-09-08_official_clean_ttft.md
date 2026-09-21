## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded the fresh official server TTFT baseline and exact request joins. |

# Official clean TTFT generation01

Execution: full worker `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_h200_official_ttft_worker.sh`, command `bash tests/e2e/issue26_h200_official_ttft_worker.sh /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-official-clean-01/runtime`. Working directory `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`. RJob yc26-h200-official-clean-20260908-01; H200 step_main,8GPUs, gpu-h200-0043; pinned image b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc; vLLM commit46f7b179f clean. Runtime Python /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python version3.10.16; Torch2.8.0+cu128; FlashInfer0.3.0. Full launch receipt/settings are in runs/h200-official-clean-01/run_manifest.json.

Validation command: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_official_ttft_analysis.py --run task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-official-clean-01/runtime/clean --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/official-clean-01`, Python3.13.13. Full validator path `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_official_ttft_analysis.py`.

Criteria: 400 unique client and server completions; all requests have4096 input/1024 observed output tokens; three fully drained warmup replays;100 formal requests joined once each; finite positive official TTFT and queue timestamps; explicit prefix caching false, eager, no chunking, TP4/DP2/EP8 and KV blocks310809 observed in startup.

Evidence: PASS. 100 formal requests;300 warmup requests excluded. Official server mean TTFT=115.982880592ms. Client mean TTFT=120.504552970ms (secondary). Observed formal queue-arrival span=47.098747927s. Worker completion marker OFFICIAL_TTFT_EXECUTION_COMPLETE. All four client phase barriers passed.

Outputs: analysis/official-clean-01/validation.json, request_id_map.csv, frontier_queue_arrivals.csv. Raw client/server JSONL and startup logs remain in runs/h200-official-clean-01/runtime/clean.

Limits: Frontier predicted value, absolute error and relative error are pending new profiling/simulation. The115.983ms server metric has not been relabeled queue-to-prefill or GPU-only latency. The4.522ms mean client/server difference is not attributed to CPU; transport and different boundaries also contribute. This PASS validates the new ground-truth evidence, not calibration acceptance.

## Additional source-grounded engine interval decomposition

Execution environment: `/usr/bin/python`, Python3.12.3, no active conda. Source inspection of vLLM46f7b179f `output_processor.py:273-318` and `metrics/stats.py:140-195` proves:

- `request_model_execution_time = (last_engine_output - first_SCHEDULED) *1000` despite its misleading pure-forward comment.
- `tpot * (decode_tokens-1) = (last_engine_output - first_engine_output) *1000`.
- Subtraction therefore recovers `first_SCHEDULED -> first_engine_output` as an interval in milliseconds.

Reproducible calculation from the active worktree:

```bash
/usr/bin/python - <<'PYCODE'
import csv,json,statistics
from pathlib import Path
p=Path('task_memory/task_2026-09-07_issue26_ttft_h200')
ids={r['server_request_id'] for r in csv.DictReader((p/'analysis/official-clean-01/request_id_map.csv').open())}
rows=[json.loads(line) for line in (p/'runs/h200-official-clean-01/runtime/clean/server.request_metrics.jsonl').open()]
rows=[r for r in rows if r['request_id'] in ids]
assert len(rows)==100
engine=[r['request_model_execution_time']-r['tpot']*(r['request_num_decode_tokens']-1) for r in rows]
remaining=[r['ttft']-v for r,v in zip(rows,engine)]
assert all(v>0 for v in engine) and all(v>=0 for v in remaining)
print(statistics.mean(engine),statistics.mean(remaining))
PYCODE
```

PASS:100 formal joins; engine interval mean80.554247736ms (range71.709448937–128.379150294ms), remaining official interval mean35.428632856ms (range2.415037248–73.327394202ms). The means sum to official server115.982880592ms. Per-request values and source provenance: `analysis/clean_engine_prefill_decomposition.json`.

Limits: the remaining interval combines queue waiting and work before/after the engine interval. It is neither a CPU-only measurement nor an approved additive correction. The engine interval includes host/device execution and ends at output publication, not an independently clock-mapped GPU forward endpoint. This calculation subtracts durations, not wall-clock and monotonic timestamps from each other. No TPOT acceptance or repair scope is added.

Independent review `/root/moe_profile_contract_review`: ACCEPT as request-level diagnostic. Raw100-row recalculation and per-row algebraPASS; durations originate in different clock domains, so the remainder is not a single-clock causal partition or microsecond-accuracy proof.
