## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded completed fresh CPU uniform replay and request-level comparison. |

# Fresh uniform Frontier verification

## Execution

Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`. Branch: `task/issue26-ttft-h200-20260907`; commit `9e3d1874fca51491d7f14b4788394811d3e48ffa`. Collective-sim revision e564935d3874d8c71b52a554ab7c9a72e5e19f68. CPU master conda environment dev-vidur-v03-hopper-e2e, Python3.13.13, sklearn1.9.0. Full runtime audit: `runs/cpu-uniform-frontier-01/runtime/runtime.json` (PASS, no import errors). H200 hardware parameters and this task's fresh H200 GPU measurements are inputs; simulation itself uses CPU only.

Launch UTC: 2026-09-08T01:21:57.648626+00:00. Exit receipt UTC: 2026-09-08T02:02:25.622414+00:00. Wall elapsed: 2427.974s. Simulation end:104.95909645254737s. Exit0 and FRESH_FRONTIER_EXECUTION_COMPLETE observed. Source code remained unchanged during the numerical run.

Exact worker command:
```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_cpu_frontier_worker.py --config /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/config/frontier_uniform_run_01.json --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/cpu-uniform-frontier-01/runtime
```
Exact persistent launch is the service_command array in `runs/cpu-uniform-frontier-01/run_manifest.json`; it sets WorkingDirectory, TMPDIR, PYTHONDONTWRITEBYTECODE and MemoryMax18G. Exact expanded simulator command and effective paths are in runtime/command.json and runtime/settings.json. Fresh scratch: /data/ycfeng/tmp/issue26-cpu-frontier-e9hg9fiy; predictor, collective and htsim caches are isolated there. Output guards require a new directory for any reproduction; the recorded run is preserved.

Exact completed result check:
```bash
TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/e2e/issue26_frontier_baseline_analysis.py --metrics /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/runs/cpu-uniform-frontier-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime --mapping /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/uniform-clean-01/request_id_map.csv --output /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/task_memory/task_2026-09-07_issue26_ttft_h200/analysis/uniform-frontier-01
```

The validator was queued behind successful creation of runtime/exit_code.txt=0, then executed once. Existing output paths must not be reused by the validator.

## Criteria and observed evidence

PASS:100unique arrivals +100unique completion events;100unique CSV rows;100current formal vLLM ID joins;4096prefill/1024decode tokens on every request;300vLLM warmups excluded. Queue-arrival offsets match at1e-9s; seconds/ms conversions, endpoint order, per-request TTFT/E2E formulas and system/CSV/JSON mean consistency passed. All expected runtime files and six metrics artifacts exist, are nonempty and were created after launch. Detailed file checks are in the final run manifest.

PASS:27effective config predicates covering ATTN TP4/DP2, MoE TP1/EP8, PP1,128experts/topk8, H200, balanced/uniform_topk, prefixOFF/chunkOFF, KV310809blocks×16 and collective_sim/nvlink_analytic. Evidence: analysis/uniform_frontier_effective_config.json. The original declared collective layout is materialized locally later; serialized null runtime fields do not describe its effective per-call group.

Numerical input provenance: current-task82linear rows and2026attention rows, plus774newuniformMoE rows (387per gating context). All predictors were trained from scratch. Current clean vLLM run is H200 step_main uniform-clean01, commit46f7b179f, pinned image recorded in its manifest. Same-population actual router/materializer checks passed24/24 across8H200 GPUs; actual Frontier first prefill records128experts×256assignments per layer. Neither check claims identical distributed scheduler batches across implementations.

## Primary mean comparison

| Metric | Value |
| --- | ---: |
| Frontier queue-visible arrival to prefill completion | 103.510401224623ms |
| Official vLLM server request TTFT | 127.380511760712ms |
| Signed difference (Frontier minus vLLM) | -23.870110536088ms |
| Absolute error of means | 23.870110536088ms |
| Relative error of means | 18.739217016908% |

Formula: abs(mean_frontier-mean_vllm)/abs(mean_vllm). Raw numerical comparison is outside10%. D006 formal gate remains NOT_EVALUATED_UNADJUSTED_BASELINE because missing official-server critical-path work is not independently accounted and workflow/operator parity is incomplete.23.870110536ms is not a CPU estimate. Request rows: analysis/uniform-frontier-01/request_comparison.csv; machine summary: analysis/uniform-frontier-01/summary.json.

## Additional same-case diagnostics

These do not introduce new acceptance targets. Their count/window operands and boundary limits are retained in analysis/uniform-frontier-01/secondary_metrics.json. All100requests are eligible for TPOT; denominator1023 excludes the first generated token. Frontier first_decode_token_completed_at equals prefill_completed_at for every request; tpot_ms recomposes exactly from request endpoints. E2E/throughput compare the respective queue-visible versus official frontend boundaries and remain unadjusted.

| Metric | Frontier | vLLM | Absolute error | Relative error |
| --- | ---: | ---: | ---: | ---: |
| tpot (ms) | 57.583941406 | 78.336091547 | 20.752150141 | 26.491174% |
| request_e2e (ms) | 59011.882459432 | 80264.999897480 | 21253.117438048 | 26.478686% |
| request_throughput (requests/s) | 0.952752104 | 0.787335508 | 0.165416596 | 21.009671% |
| token_throughput (tokens/s) | 4878.090773499 | 4031.157800173 | 846.932973327 | 21.009671% |

Formal throughput windows are104.95909645254737s and127.01065683364868s; numerators100requests and512000total input+output tokens. The secondary calculation reads only this run's200request events and the400current uniform vLLM records, selects the same100mapped IDs, averages tpot_ms/tpot and request_e2e_time_ms/request_e2e_time, and divides the stated numerators by maxcompletion-minarrival. Exact source paths and all numerator/denominator values are in secondary_metrics.json.

## Internal accounting and limits

First-request TTFT65.79007690374297ms matches the bounded first-prefill EPwave endpoint within2.5704e-10ms.48-layer component sum65.790074082ms closes within2.822e-6ms of the request endpoint (log rounding). Duplicate attention parent records were counted once; EP phases use a maximum, not an8-lane sum. Evidence: analysis/uniform-frontier-first-prefill.json/.md and analysis/uniform_frontier_first_prefill_endpoint.json.

The current uniform clean engine interval decomposition passed100formal joins: official127.380511761ms = firstSCHEDULED-to-firstEngineCoreOutput86.785534397ms +remaining40.594977364ms. The remaining duration includes queue/outside-engine work and is not directly additive to Frontier. Evidence: analysis/uniform_clean_engine_prefill_decomposition.json/.md.

No simulation execution failures occurred in this generation. D012 focused code checks and their recovered test-fixture failures are recorded separately in test_report_2026-09-08_d012_routing_runtime.md. Current source-confirmed profiling activation omission and separate moe_sum omission remain uncorrected; timing impact is unmeasured. D013 scoped activation proposal awaits YC, and no corresponding source change or GPU rerun has begun.

## Reproduce secondary diagnostics

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'CHECK'
import csv, json, math, statistics
from pathlib import Path
p=Path('task_memory/task_2026-09-07_issue26_ttft_h200')
m=list(csv.DictReader((p/'analysis/uniform-clean-01/request_id_map.csv').open()))
fr={r['request_id']:r for r in map(json.loads,(p/'runs/cpu-uniform-frontier-01/runtime/metrics/qwen3_a3b_30b_moe/online_serving/runtime/metrics_ground_truth.jsonl').open()) if r['event_type']=='request_completion'}
gt={r['request_id']:r for r in map(json.loads,(p/'runs/h200-uniform-clean-01/runtime/clean/server.request_metrics.jsonl').open())}
pairs=[(fr[int(r['frontier_request_id'])],gt[r['server_request_id']]) for r in m]
assert len(pairs)==len(fr)==100
for f,g in pairs:
    assert f['num_decode_tokens']==g['request_num_decode_tokens']==1024
    assert f['first_decode_token_completed_at']==f['prefill_completed_at']
    assert math.isclose(f['tpot_ms'],1000*(f['completed_at']-f['first_decode_token_completed_at'])/1023,abs_tol=1e-8)
for a,b in [('tpot_ms','tpot'),('request_e2e_time_ms','request_e2e_time')]:
    x=statistics.mean(f[a] for f,g in pairs)
    y=statistics.mean(g[b] for f,g in pairs)
    print(a,x,y,abs(x-y),100*abs(x-y)/y)
fw=max(f['completed_at'] for f,g in pairs)-min(f['arrived_at'] for f,g in pairs)
gw=max(g['completion_time'] for f,g in pairs)-min(g['arrival_time'] for f,g in pairs)
assert fw>0 and gw>0
for numerator in [100,512000]:
    x,y=numerator/fw,numerator/gw
    print('throughput',numerator,x,y,abs(x-y),100*abs(x-y)/y)
CHECK
```
