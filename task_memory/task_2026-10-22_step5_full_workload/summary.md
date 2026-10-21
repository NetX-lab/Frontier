# D25 H800 full workload completion

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Archived both completed native workloads, verified metrics and stability limits. |

## Outcome

Completed the requested H80032 and H80064 native Step workload collection. Each job executes three repetitions of16 warmups plus128 measured6144/1024 requests atQPS1, using the same locked dummy-weight model/image/settings and one API frontend per role. Both jobs Succeeded; all864 requests complete, all12 containers exit0/restarts0, and all96 H800 UUIDs are distinct.

| H800 GPUs | Job | Pooled p90 TTFT(s) | Pooled p90 TPOT(ms/token) | Pooled p90 E2E(s) | Mean total/output tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
|32|`exp-1022-024130-721461`|1.760660|87.792540|91.639790|5101.423/728.775|
|64|`exp-1022-024130-735731`|1.911773|93.557086|97.515587|4999.053/714.150|

P90 uses nearest rank over384 measured requests per scale. Throughput is the arithmetic mean of three windows, each first measured dispatch through last measured token; warmups and gaps between windows are excluded. Full means and repetition values are in the metrics JSONs.

## Verification and deliverables

- Source `e998210` adds the explicit workload purpose and p90 through the existing stager/replay;121 CPU checks pass, preserving canonical accuracy-release requirements.
- `tests/integration/step5_pd/workload_metrics.py` reuses the strict population reader and reports exactly recomputed metrics. Both final native-artifact executions pass;144 prompts per scale regenerate from frozen templates and pinned vocabulary metadata and match accepted hashes. All output IDs and timestamp/count contracts validate.
- This directory: `test_report_2026-10-22_workload_collection.md`, `native_run.md`, `native_receipt.json`, `h80032_metrics.json`, `h80064_metrics.json`, `historical_ttft.json`, `h80032_phase_checks.json`, `h80032_token_gaps.json`, `h80032_log_events.json`, and independent `h80064_audit/{receipt.json,test_report_2026-10-22_h80064_native_metrics.md}`.
- Local sealed packages/evidence: `/data/ycfeng/tmp/step5_d25_workload_20261022/h800_{32,64}_workload01/stage`; exact cloud locations and launch commands are in `native_run.md`.
- Observed container GPU time12.966667/26.564444GPU-hours, within48/96 ceilings.

## Interpretation and open work

No measured request exceeds5s TTFT. Historical64 burst TTFT and first-round64 warmup outliers are analyzed separately in the reports. In-measurement JIT and repeat-dependent decode latency remain, so these are complete workload measurements rather than a fully warmed/stable reference.64 GPUs do not improve performance at this fixed offered load. Native qualification, G3/reference agreement, admitted profiles and final simulator accuracy remain unverified.

Independent D23 CPU capacity-review/full-CLI work was interrupted twice by provider HTTP502; its saved owner sessions and next actions are preserved in the main task progress. D24 bounded-capture CPU implementation is complete; its new native diagnostic remains pending after the prioritized D25 collection.
