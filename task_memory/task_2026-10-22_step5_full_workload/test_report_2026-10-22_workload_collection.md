# D25 native workload collection: verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded121 CPU passes and historical TTFT endpoint/counter analysis before native staging. |

## CPU execution

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-preview`. Python `/data/ycfeng/envs/asyncopd-py312/bin/python`3.12.13 (Clang22.1.3), pytest9.0.2. Existing simulator interpreter `/data/ycfeng/envs/frontier-py310/bin/python`; no environment installation.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_step5_priority_runs.py tests/unit/test_step5_canonical_native.py tests/unit/test_step5_client_metrics.py tests/unit/test_canonical_client_metrics.py -q --tb=short -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d25_workload_parent01 --junitxml=/data/ycfeng/tmp/d25_workload_parent01.xml > /data/ycfeng/tmp/d25_workload_parent01.log 2>&1
git diff --check
```

Observed: **121 passed in104.09s**, zero failures/errors; diff check PASS.

- Real local HTTP servers and the unchanged token-ID replay client execute all432 requests, three16+128 phases, exact arrivals/template IDs and six strict post-phase boundaries for the new `workload` purpose at DP2/DP4.
- Injected HTTP failure remains visible while subsequent healthy repetitions finish. Reference/runtime qualification stays UNVERIFIED.
- Standalone sealed-stage dry-runs accept H80032/64 workload collection without fabricated readiness. Changing the purpose to canonical still rejects missing readiness; existing per-repeat30% tests and final10% gate tests pass.
- Native latency aggregation emits nearest-rank p90. Independent128-value expectation: mean64.5, p90116 and p95122.
- Frozen inputs, token formulas and shared client/normalizer checks pass. These CPU checks are not native completion evidence.

## Historical TTFT above5s

Artifact: `historical_ttft.json`, including exact source paths/SHA256, all16 non-warmup request joins, endpoint components, means and native P-counter deltas. Sources are the existing H80032 qualification03 and H80064 qualification02 packages; each has8 concurrent samples.

| Corresponding p90 request, seconds | H80032 | H80064 |
| --- | ---: | ---: |
| Client TTFT |4.791508447|9.093688630|
| Dispatch to proxy arrival |0.008332813|0.005286112|
| Proxy arrival through P response |4.217981735|8.357245632|
| P response through first client token |0.565193899|0.731156886|

These are components of each actual p90 request, not sums of separate percentiles. Ordered same-host monotonic endpoints close exactly to TTFT. The H80064 p90 exceeds5s by4.093689s;8.357246s (about92%) precedes the P response. It is meaningfully high relative to both the requested threshold and the available32-GPU run.

Population means: P-response interval2.663758s versus5.567062s, client TTFT3.241113s versus6.418956s. The native P counters between verified after-warmup/after-concurrent boundaries each count exactly8 requests: mean `request_queue_time_seconds`0.794069s versus0.156125s, mean `request_prefill_time_seconds`1.832679s versus3.846671s. Thus these observations locate the dominant delay on the P side; they do not support attributing the64-GPU regression solely to its reported queue metric. Counter names are retained literally; unclosed server-side intervals are not assigned to an unmeasured operator.

The32/64 historical API frontend counts differ (1 versus4), as do DP/EP width and physical nodes. These data do not determine how much delay each difference caused. The new full runs pin one frontend per role, preserve native topology/configuration, and supply three complete windows for a better-controlled comparison. Source/API/communication attribution remains pending their evidence.

## Native acceptance

Pending fresh package sealing, submissions, complete workload evidence, terminal inspection and updated p90/throughput analysis.
