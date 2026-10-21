# D25 native workload collection: verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Verified both full workloads, prompt regeneration,64 independent audit and complete p90/throughput analysis. |
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

## Native execution and identity

Both jobs reached **Succeeded**,32 GPUs at2026-10-21T19:10:05Z and64 GPUs at19:11:19Z. All12 containers exited0 with zero restarts. Primary read actual replica/container APIs and checked each image digest, role/lane inventory, eight distinct H800 UUIDs per node, all96 UUIDs unique across the two jobs, empty worker observation errors, `workload_returncode=0`, and staged tool hashes. Both workers verify50 locked native source hashes before rendezvous (`native_pd_worker.py:135–138`). Actual versions: vLLM `0.1.dev22912+g8224bb82f`, torch `2.13.0+cu130`, aiohttp `3.14.5`. Evidence/hashes: `native_receipt.json`, with platform originals at each package's `parent_terminal01.json`.

Observed container GPU time: **12.966667 GPU-hours (32)** and **26.564444 GPU-hours (64)**. These are summed container runtime ×8, not queue time or billing estimates. Both are below their48/96GPU-hour caps.

## H80032 full workload: PASS

Primary command:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B tests/integration/step5_pd/workload_metrics.py --stage /data/ycfeng/tmp/step5_d25_workload_20261022/h800_32_workload01/stage --output task_memory/task_2026-10-22_step5_full_workload/h80032_metrics.json > /data/ycfeng/tmp/d25_h80032_metrics.log 2>&1
```

This invokes the existing strict `native_client_rows` reader, checks every token-delta endpoint/count, joins every proxy record one-to-one, and recomputes all stored repetition metrics exactly. **432/432 requests complete,48 warmups,384 measured, six verified post-phase boundaries.** No measured TTFT exceeds5s; maximum4.906835s. All measured phases have exact P local128×6128, D external128×6128 plus128×16 local tokens,1024 rank READs totaling143,244,394,496bytes, and zero native NIXL failure counters. `h80032_phase_checks.json` records these checks.

| Repeat | p90 TTFT(s) | p90 TPOT(ms/token) | p90 E2E(s) | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
|1|1.916213|91.911522|95.665251|5041.079914|720.154273|
|2|1.686839|69.468726|72.556134|5140.158997|734.308428|
|3|1.642572|68.620203|71.837178|5123.030353|731.861479|

Pooled384-request nearest-rank p90: **TTFT1.760660s, TPOT87.792540ms/token, E2E91.639790s**. Three-window arithmetic mean total throughput5101.423088tok/s; output-only728.774727tok/s. Warmups and inter-window pauses are excluded from every throughput denominator.

TTFT is below the user's concern threshold. Decode timing is less stable: first-round p90 TPOT91.912ms versus69.469/68.620ms, with peak client in-flight97 versus74/73. Runtime logs contain an inference-time `_direct_scores` JIT warning at18:59:22 during the first measurement; this supports incomplete shape warmup but does not quantify the entire slowdown. Token-delta gaps (`h80032_token_gaps.json`) max1.042s/0.619s/3.673s across the three rounds, so a single long compilation stall is not established as the explanation. These observations should not be described as fully stable steady-state performance.

Both roles log actual runtime graph rows. Shutdown-period errors/warnings are retained in `h80032_log_events.json`: P AsyncLLM output-handler errors start19:09:49, D errors19:09:56, with TCPStore/NCCL peer-close messages. These follow the final measured output at19:09:31.564Z; worker stop receipts record SIGINT and successful workload return. This is not an error-free-server claim, and native qualification remains a separate unverified gate.

## H80064 full workload: PASS

Independent audit: [`h80064_audit/test_report_2026-10-22_h80064_native_metrics.md`](h80064_audit/test_report_2026-10-22_h80064_native_metrics.md), with immutable author receipt. Primary reran the same `workload_metrics.py` command above with `h800_64_workload01` and `h80064_metrics.json`, then repeated both sizes after adding prompt regeneration. Final logs: `/data/ycfeng/tmp/d25_h800{32,64}_metrics02.log`. The initial regeneration attempt raised `KeyError: vocab_size`; using the existing Step adapter's `local_model_config(...,version='step5')` resolved its nested model metadata. Both final commands exit0. This post-run analysis correction changes no native record.

**432/432 complete,48 warmups,384 measured, six post-phase idle boundaries.** Primary reconstruction verifies144 frozen prompt templates per scale against sent and accepted hashes, pinned model metadata and prompt/output vocabulary bounds. This closes the independent audit's prompt-regeneration/vocabulary limitation, while raw accepted arrays remain unretained. All pooled means/p90/min/max and average throughput crosscheck the independent64 audit within1e-9.

| Repeat | p90 TTFT(s) | p90 TPOT(ms/token) | p90 E2E(s) | Total tok/s | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
|1|1.952135|99.983773|103.978045|4947.598965|706.799852|
|2|1.903968|89.337139|93.053702|5028.266863|718.323838|
|3|1.885374|78.902957|82.290655|5021.293927|717.327704|

Pooled384-request nearest-rank p90: **TTFT1.911773s, TPOT93.557086ms/token, E2E97.515587s**. Three-window arithmetic mean total throughput4999.053252tok/s; output-only714.150465tok/s. Zero measured TTFT>5s; maximum2.473084s. Native per-phase source/read/failure counters pass the same exact workload checks as32.

### TTFT and normality conclusion

- Both measured populations are below the user's5s concern threshold, including every individual measured request.
-64 first-round warmup has12 TTFT values above5s; maximum11.506277s. The maximum decomposes into7.580721s through P response and3.904503s from D headers to first token, with about20.999ms elsewhere. Both roles log first-inference Triton JIT and D pending KV work; TCP setup is milliseconds. The dominant delay is native startup/warmup, but per-request device/JIT attribution is not established. These warmups are excluded from the reported measurement population, not discarded from evidence.
- Both jobs still log `_direct_scores` JIT inside the first measurement.64 native mean ITL falls82.390→73.930→67.572ms, agreeing with client TPOT drift; this is not just a client-clock artifact. Correlated token gaps do not establish a complete causal account of the slowdown.
-64 mean total throughput is about2.01% below32, while pooled p90 TPOT/E2E are about6.57%/6.41% higher. This fixed QPS1 workload shows no benefit from doubling GPUs. Fixed offered load and finite completion windows also prevent interpreting this ratio as maximum scaling efficiency.
- Workload completion and statistics pass. Fully warmed, stable performance and native/calibration qualification remain **UNVERIFIED**. Shutdown-period EngineDeadError/TCPStore logs are retained, with all requests complete and local stop receipts0; no server-error-free claim is made.
