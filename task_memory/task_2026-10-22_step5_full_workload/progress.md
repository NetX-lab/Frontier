# D25 progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Verified121 CPU checks and historical TTFT decomposition; preparing fresh packages. |

- completed: historical audit confirms basic-only populations; no complete canonical workload result.
- completed: read-only personal quota snapshot `/data/ycfeng/tmp/step5_quota/d25_h800_workload01.json`: codesign quota128 H800; no personal nonterminal jobs; 13 returned nodes had8 usable GPUs. Capacity is a snapshot, not a reservation.
- completed: existing canonical stager/replay supports explicit `workload` purpose and p90.121 checks pass in104.09s, including real HTTP three-repeat execution and strict formal release. Details in `test_report_2026-10-22_workload_collection.md`.
- completed: historical64-GPU p90 TTFT9.093689s decomposes into0.005286s dispatch-to-proxy,8.357246s through P response,0.731157s thereafter. Retained hashes/rows/counter deltas: `historical_ttft.json`.
- in-progress: commit and seal fresh packages from clean source, with isolated ports.
- pending: sealing, parent dry-run/identity acceptance, two native submissions, metrics and TTFT analysis.
