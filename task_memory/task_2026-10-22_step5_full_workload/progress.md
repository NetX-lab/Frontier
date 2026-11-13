# D25 progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Both full workloads completed and primary-verified; metrics, warmup analysis and stability limits archived. |

- completed: initial historical audit confirmed basic-only populations before D25.
- completed: read-only personal quota snapshot `/data/ycfeng/tmp/step5_quota/d25_h800_workload01.json`: codesign quota128 H800; no personal nonterminal jobs; 13 returned nodes had8 usable GPUs. Capacity is a snapshot, not a reservation.
- completed: existing canonical stager/replay supports explicit `workload` purpose and p90.121 checks pass in104.09s, including real HTTP three-repeat execution and strict formal release. Details in `test_report_2026-10-22_workload_collection.md`.
- completed: historical64-GPU p90 TTFT9.093689s decomposes into0.005286s dispatch-to-proxy,8.357246s through P response,0.731157s thereafter. Retained hashes/rows/counter deltas: `historical_ttft.json`.
- completed: sourcee998210 committed; fresh packages from clean checkout each verify37 tool/50 native pins, real StepMind Python dry-run and ResourceConfig construction, isolated ports. Exact records: `native_run.md`.
- completed: H80032 `exp-1022-024130-721461` and H80064 `exp-1022-024130-735731` both Succeeded. All12 containers exit0/restarts0;96 unique H800 UUIDs, complete role/lane/source/image evidence. Container GPU-hours12.966667/26.564444. See `native_receipt.json` and `native_run.md`.
- completed: each scale has432/432 successful requests,48 warmups and384 measured, six post-phase idle boundaries. Primary `workload_metrics.py` reproduces all stored per-repeat values and all144 frozen prompt templates; prompt/output vocabulary bounds pass.64 independent audit and primary pooled metrics agree within1e-9. Final logs `/data/ycfeng/tmp/d25_h800{32,64}_metrics02.log`; an earlier prompt-regeneration checker failed on a top-level `vocab_size` assumption, then used existing `local_model_config(...,version='step5')` to correctly resolve nested metadata. The error affected only post-run analysis; original native records are unchanged.
- completed: pooled p90 TTFT1.760660/1.911773s, TPOT87.792540/93.557086ms, E2E91.639790/97.515587s; mean total throughput5101.423/4999.053tok/s. Zero measured TTFT>5s. First-round64 warmup outliers and historical64 basic TTFT are decomposed in the reports.
- completed with limits: first-round decode slowdown and in-measurement JIT remain observed;64 GPUs show no performance gain under this workload. Fully warmed/steady-state behavior and native/accuracy qualification are not established. D23 capacity/actual-main subagents twice failed with provider HTTP502; these independent CPU tasks remain pending, with owner/artifact pointers in the main task handoff.
