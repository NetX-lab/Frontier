# D21-NATIVE-REPLAY: logger root cause and historical H20064 reuse

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Diagnosed actual multi-API logger suppression, fixed caller and assessed retained64-GPU basic proof. |

## Outcome

**Root cause resolved at source/caller level; actual GPU replay remains UNVERIFIED.** The locked CLI defaults API frontend count to role DP size. Native `StatLoggerManager` deliberately disables textual statistics for multiple API clients, including graph tables, while retaining Prometheus. P0 logs explicitly show that condition; this is not evidence of a broken MRv2 statistics field or failed graph capture.

**Old H20064 basic proof is reusable but PARTIAL, not G3/G4 or formal-ready.** Do not repeat64 GPUs solely for its startup-gauge diagnostic. Actual lane-labelled replay, content/lifetime and formal prerequisites remain missing. Parent owns any further platform decision.

Code commit: `6838bc10135faf5bea1bc48321837caa938f85a1` on `feat/step5-priority-runs`, based on42dda4c. Only native caller/config/staging verification and focused CPU tests changed; no native kernels, simulator, thresholds, request/accounting or auxiliary error semantics changed.125 affected CPU checksPASS. Fresh P0 qualification03 is prepared/verified offline, **not submitted**.

## Decisive evidence and transport chain

Paths: **V** = `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`; **P0** = `/data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01`; **H64** = `/data/ycfeng/tmp/step5_d18_native_20261021/qualification02`. Node paths below are under each `stage/evidence/nodes/`; H800 suffix is `.host.platform.shaipower.com`, H200 suffix is `.lgcm.sh.istep.fun`. Log line references are LF-delimited physical lines.

- P0 P head `gpu-h800-0083/server.log:1,26,375–376` and D head `gpu-h800-0216/server.log:1,28,381–382`: default api_server_count2, two API processes, then `AsyncLLM created with api_server_count more than 1; disabling stats logging to avoid incomplete stats`.
- H64 P head `gpu-h200-0024/server.log:1,26,410–414`; D head `gpu-h200-0878/server.log:1,28,421–424`: same cause, count4. All original argv include `--cudagraph-metrics`, none includes an explicit API count, `--disable-log-stats` or aggregation flag. `server_env_set` contains CUDA/NIXL role settings, no harness logging override; the full inherited environment was not retained. The actual INFO records and native suppression warning settle this cause without guessing inherited settings.
- `V/entrypoints/launchers/cli_args.py:441–448` registers supported integer `--api-server-count` (defaultNone). `V/entrypoints/cli/serve.py:69–77,107–156`: headless count0; internal-DP head defaults to full DP, not local size1. An explicit count1 selects single API process, without changing TP/DP/EP or internal load balancing.
- `V/v1/worker/gpu/model_runner.py:1810–1837,2100,2204`: MRv2 constructs graph stats from the post-DP-sync descriptor for real batches and carries them via ExecuteModelState to ModelRunnerOutput. `V/v1/worker/gpu/cudagraph_utils.py:83–91` records unpadded/padded token counts, padding and runtime mode; fullgraph dispatch calls actual replay, separately exercised by the existing CPU source test.
- `V/v1/core/sched/scheduler.py:2144,2510–2524,3021–3058` carries `model_runner_output.cudagraph_stats` into SchedulerStats and sends each step's stats to only one frontend. With multiple API clients no individual logger has a complete population, which explains the intentional suppression. `V/v1/engine/async_llm.py:194–204,886–896` builds the manager and records the stats with `outputs.engine_index` rather than dropping lane attribution.
- `V/v1/metrics/loggers.py:1328–1363` is decisive: client_count>1 suppresses LoggingStatLogger only; Prometheus is still appended. `:129–134,237–241,329–338` consumes graph stats and logs a preceding EngineNNN throughput line followed by the table. `V/compilation/cuda_graph.py:40–124` groups real observed runtime modes/counts and resets each emitted interval. Thus capture output alone is insufficient, and successful Prometheus collection does not imply a text graph logger existed.
- `V/entrypoints/launchers/api_server/entry.py:89–103` forwards client_count; `V/entrypoints/launchers/utils/server_utils.py:75–87` invokes periodic logging. `V/v1/engine/core_client.py:716–732` manages all local+remote EngineCores in pure internal LB, not just local lane0. One API frontend therefore retains per-engine logging for every request-owner lane; no requirement for32/64 separate graph counters is introduced.
- Complete P0 and H64 logs retain startup, native capture, requests and requested shutdown. There are0 EngineNNN headers and0 FULL runtime rows in both packages, not merely a truncated grep selection. Those historical logs/outcomes remain unchanged. The tested native transport is source-level evidence, not retrospective proof that those runs executed FULL replay.

## Minimal correction and fresh offline package

- `tests/integration/step5_pd/canonical_run_h200.json`: freeze `engine.api_server_count=1`, inherited by all four D21 cases. The existing name is retained; this is the shared definition, not a new H200-only path.
- `tests/comparison/dp_placement_pp/vllm_replay.py`: emit supported `--api-server-count 1` only on API heads. Headless peers must omit a positive count because native rejects that combination. No new logging level, environment toggle, retry, collector, parser or validator is added.
- `tests/integration/step5_pd/canonical_stage.py`: verify the frozen count and bind the native CLI/API-entry/periodic-logger sources. Existing source files and native sampling/graph/NIXL flags otherwise remain unchanged.
- Tests exercise actual builder→node_plan→server_command→locked CLI defaulting→locked logger construction for DP2/DP4. Before correction both cases fail specifically because no text logger is built while Prometheus survives; afterwards every DP owner remains managed, both role heads select one frontend, all headless peers select none. All four hardware/scale preflights also assert the caller flags. CPU tests do not prove GPU replay or finite/content correctness.

Fresh package: `/data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage`. Source6838bc1,37 staged pins,50 native source pins; standalone verify, exact offline submission dry-run and `bash -n`PASS. Local receipts are `fresh_stage_build.json`, `fresh_stage_verify.json`, `fresh_stage_dry_run.json` in this directory. P0 remains4×8 H800/codesign/non-preemptible, TP8/DP2/EP16/PP1,16 warmups→verified idle→8 concurrent,7168/8192/128, no formal reference admission. Output path in the config is prospective only; no cloud read/write occurred here.

**Recommendation, no submit:** use this fresh generation for the next parent-authorized P0 qualification, not sealed01/02. Retain complete process-labelled P/D head logs and each Engine000/001 line preceding graph tables; require healthy generation with observed nonzero FULL decode rows on both D owner lanes, while recording actual P runtime modes rather than demanding invented P FULL rows. Check both required idle boundaries and all request/rank accounting separately. A single API process changes the frontend configuration: do not merge prior multi-API timings with fresh clean-reference timings. Other future case generations must adopt the same corrected shared definition; no other new stage is built here.

## Historical H20064 evidence reuse

Native source commit `2a1070be38ed6fb75dc223c46b0ea407ee908ce6`, immutable image `sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`; launch receipt `H64/launch_receipt_d19.json:3–20` binds creator,8 replicas×8 GPUs, local mount and historical **preemptible=no**. Platform success at08:20:59UTC is recorded by primary spec§0:101; this child did not query the platform. All8 retained worker results finish with workload returncode0, requested SIGINT/server returncode0 and no observation errors; all8 drained/native_stopped proofs exist. Native source47/47 pins match the frozen export, tools27/27 and metadata3/3 match stored pins.

| Role/lane | H200 node | MRv2 line | KV capacity / concurrency at7168 | Capacity log line |
| --- | --- | ---: | --- | ---: |
| P/0 | 0024 |115|1,229,781 /171.57x|214|
| P/1 | 0052 |30|1,230,272 /171.63x|125|
| P/2 | 0057 |30|1,230,272 /171.63x|125|
| P/3 | 0760 |30|1,229,781 /171.57x|125|
| D/0 | 0878 |157|1,229,781 /171.57x|224|
| D/1 | 0938 |64|1,230,272 /171.63x|127|
| D/2 | 1080 |64|1,230,272 /171.63x|127|
| D/3 | 1192 |64|1,229,781 /171.57x|127|

All64 GPU UUIDs are distinct H200s. All32 EP ranks per role have explicit model-load lines (23.14GiB each); observed TP8/DP4/EP32/PP1 is the correct64 topology. Backend initialization reports native FP8, MTP None, context7168, graph configured and chunk/prefix caching off. This is usable startup/rank-load/capacity evidence, not sustained128-request occupancy, full finite-state/exclusion census, useful/dummy-forward accounting or complete NIC/NUMA placement proof.

`H64/stage/evidence/replay/client_requests.jsonl:1–24`:24/24 HTTP200, streamed/completed, no validation errors;16 warmup +8 concurrent; every accepted prompt6144 and final output1024 IDs, usage and prompt hashes agree. `proxy_requests.jsonl:1–24` joins all24 with P/D HTTP200, nixl_pull and remote_num_tokens6128. Concurrent dispatch span1.568ms. Summary `:1970–2048,2050–2128` verifies warmup and final idle across P/D lanes0–3 with running/waiting/KV=0; P completion population after warmup[3,3,5,5], final[5,5,7,7]; D warmup[4,4,4,4], final[6,6,6,6]. Each lane adds2 concurrent completions. Startup's missing gauges at`:570–1968` remain a diagnostic under D21, not grounds to discard these later observed boundaries.

Concurrent8 metrics (`summary:19–560`) show P local49,024 tokens; D external49,024 +local128: **P6128; D6128 external+16 local per request**, not full6144 P compute. D64 rank-reads transfer8,952,774,656 bytes (16 reads/2,238,193,664 bytes per lane); transfer-time sum9.607027s, post-time sum0.606488s. All24 totals are192 reads/26,858,323,968 bytes; P147,072 local tokens, D147,072 external+384 local. Both roles' failure/notification/expiry deltas and local-cache-hit counters remain0. These aggregate rank-read statistics do not establish content or GPU-visible lifetime ordering.

Original `summary:2133–2137` remains runtime/referenceUNVERIFIED, client_workload_verified=false, **REFERENCE_UNSTABLE** due its then-required startup idle check. The child row/idle judgment is not a rewrite of that summary. Head shutdown errors are preserved: P head`:846–900` and D head`:915–952` show EngineDeadError only after requested zero-timeout shutdown/forced API termination, after the final successful metrics read. They are observed shutdown errors, not an inferred mid-workload request failure; the run is nevertheless not promoted to an error-free clean reference.

**Reuse disposition:** retain the hardware/scale-specific startup, measured capacity, full rank-load, fixed-length completion, actual read-count/prompt-split and later idle proof. Actual replay remains missing for the demonstrated same logger cause; no content/lifetime or clean-reference claim is reusable. Historical non-preemptible policy is not retroactively changed, but this package is not a launch recipe under D21. If a new64-H200 observation is eventually necessary, parent uses a **new wholly-preemptible generation**, not historical canonical02; first validate the cheaper P0 caller fix and target the remaining evidence rather than blindly rerunning for initial gauges. H20032's own three-repeat/four-error30% release plus H20064 own readiness still precede any64 formal collection.

## Exact case-aligned G3 witnesses/reference still missing

Authority: ticket08:30–36,61–70, primary spec:252,267–278. Neither package contains these witnesses; enabling text metrics cannot supply them:

1. Canonical6144/1024 at each hardware/scale's actual TP/DP/EP/native identity: complete per-request/rank raw-KV and side-summary contents, registered component/group/dtype, source/target logical→physical page mapping, matched6128 transferred tokens, the observed16-token D tail and D first scheduled tokens. Byte totals and equal random seeds are not content or P/D parameter/buffer-identity proof.
2. Independent functional/known-state reference with verified parameter/buffer/scales identity and its own content/output comparison. Existing paused maxlen2048 R2's `reference_census` is intra-run consistency only, not an independent reference; short255–1537/output17/128/256 capture does not qualify canonical length.
3. Request-local source retention→target reservation→READ→network completion→GPU-visible readiness→D execution→source/target release witnesses, with local-clock ordering and component/alias ownership; final idle alone is not lifetime proof.
4. Actual before/on/after region/page boundaries, partial/open-tail behavior, next decode region crossing, differing local page IDs, SWA transferable ranges/full-history sparse KV, reset/reuse and target-capacity pressure.
5. Concurrent owner-lane handoffs, retained-state reuse races, cancellation/pending-READ abort and conservation for every request/token/state component; native useful and required DP dummy forwards remain separately accounted. No broad G3 implementation or substituted reference is introduced by this subtask.

**Formal-ready NO** for P0 and historical H20064: graph evidence restoration is preparation, not actual replay/G3/G4/profile admission/four-metric accuracy. Parent owns next authorized native generation and independent prerequisite completion.
