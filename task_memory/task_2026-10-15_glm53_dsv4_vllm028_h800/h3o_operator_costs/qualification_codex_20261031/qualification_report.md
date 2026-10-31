# P2 H200 packed KV, memory and admission audit

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Captured the exact retained P2-32/P2-64 sources and classified payload, memory, capacity and admission conditions. |
| 2026-10-31 | Reconciled native startup counters, preserved original authorization and recorded the limits of diagnostic evidence. |

The original P2-32 and P2-64 clean references retain R4 PASS. Their hardware, startup cache geometry and memory scalars are available. Complete qualification remains OPEN: the source-derived packed extent is different from component-byte accounting, the retained inputs do not bind an accepted H200 NIXL fit, and the formal coverage and promotion conditions are not discharged by these observations. Retained B29 P2-32 passes the numerical 10% predicate; P2-64 fails repeat01 TTFT. No ticket is closed by this audit.

This is a local investigation using existing artifacts. It performs no GPU, platform or cloud operation, source repair, replay, new gate evaluation, broad trace census or receipt rewrite. F2, F1a, A6, A10, UUID selection and links/KV02 remain lead-owned. The user's reported run state is routing context only and was not independently inspected.

## Authority and evidence locations

The controlling main worktree is `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, observed at `5d1754d0eb11d6cd1fe7a164f7f3161aa2f43f07`. The report branch is `b30b-ex-situ-link-probes`, based on `dbd3ff6f7e2cfdf30050945039df7b66cced7d14`. Main requirements decision 46 and the current route control: measured CC remains formal, F1-F4 remain diagnostic, and ingress A is approved. A replacement of a formal input requires the separately recorded decision described by that route. (`S:A02`, `S:A03`)

The [source index and selected evidence](evidence/qualification_evidence.json) contains 160 exact local paths, byte lengths and SHA-256 values. The `S:` prefix below names a key under `/sources` in that JSON. Authority/source excerpts retain original line numbers; case JSONs retain the listed pointers. The 96 rank records retain only the memory/cache/configuration fields needed here, and 12 startup logs are inspected for selected rows within their first 1,600 lines. No profiling trace is parsed.

| Source keys | Exact path or selector |
| --- | --- |
| `S:A02`, `S:A03`, `S:A04` | Main `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/{requirements,plan,progress}.md`; decision 46, current route and current handoff. |
| `S:A05`, `S:A06`, `S:A07`, `S:A08`, `S:A09` | The same task's `tickets/{05-dsv4-flash-family,09-predictor-bundle,11-h200-regression-delivery,18-native-ground-truth-h200,19-comparison-and-acceptance}.md`; current checkpoints and acceptance. |
| `S:A10` | The same task's `source_spec_v2/Frontier_GLM53_DeepSeekV4Flash_vLLM028_H80064_Spec_v2.md:393-402,666-678`; sections 5.5 and 10.4. |
| `S:P32_run_check_status.groundtruth_clean`, `S:P64_run_check_status.groundtruth_clean` | Main task `h200_mainline/diagnostic/inputs/p2_{32,64}r/run_check_status.groundtruth_clean.json`; `/runs/<run_tag>/status` and `/checks`. |
| `S:P32_run_manifest`, `S:P64_run_manifest` | Those input directories' `run_manifest.json`; `/authorization`, `/written_before_launch_utc`, `/resources`, `/groundtruth_commit`. Launcher hashes and terminal receipts are separately retained. |
| `S:NIXL` | `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:960-1062,1388-1457,2412-2446`; SHA-256 `4860f51b6ac3301c243794b7626b5536f6b0ec41204e426ee6b61bc0e1d0bdb7`. |
| `S:GROUP_COUNTS`, `S:PAYLOAD_SOURCE`, `S:CAPACITY_SOURCE` | Main `task_memory/task_2026-10-21_v4_grouped_nixl/native_retention.md:45-59`, `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py:19-65,107-136`, `frontier/attention/memory.py:211-224`. |
| `S:NATIVE_MEMORY`, `S:NATIVE_COUNTERS` | Pinned native `vllm/v1/worker/gpu_worker.py:515-565,754-806` and `vllm/utils/mem_utils.py:305-326`. |
| `S:P32_F4`, `S:P64_F4` | `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/f4/p2_{32,64}_init/planner_blocks.json`; `/planner`, `/native_num_blocks`, `/decode_lanes`. |
| `S:P32_F4_B28`, `S:P64_F4_B28` | Main task `h3o_operator_costs/codex_resume_20261031/f4_p2_{32,64}_b28_comparison.json`; `/statuses`, `/metrics`. Existing comparisons are reused. |
| `S:P32_B29`, `S:P64_B29` | Main task `h3o_operator_costs/codex_resume_20261031/p2_{32,64}_warm_k19b29/reading.json`; `/verdict`, `/repetitions`. All 24 existing metric rows are retained. |
| `S:P32_B29_CONFIG`, `S:P64_B29_CONFIG` | `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_{32,64}_warm_k19b29/metrics/deepseek_ai_deepseek_v4_flash/online_serving/a10_p2_{32,64}_operator_path/config.json`; `/kv_cache_transfer_config`, `/cluster_config/replica_scheduler_config`. |

The native downloads are already present locally, as bound by each `download_receipt.json` and `cloud_inventory.json`:

- P2-32: `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-canonical-20261025-01/artifacts`.
- P2-64: `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-canonical-20261025-01/artifacts`.

Every selected `nodes/<host>/result.json`, `server.log` and `serving/rank<n>.json` matches its existing inventory hash and byte length. All exact filenames and selected log line numbers are retained in the source index and `/cases/p2_*/startup_logs`. Rank pointers are `/inventory/memory`, `/inventory/workspace_bytes`, `/inventory/cache/num_blocks`, `/inventory/effective_graph`. (`S:P32_cloud_inventory`, `S:P64_cloud_inventory`)

## Criterion and evidence table

SATISFIED means the stated prerequisite is supported within its named scope. PARTIAL or MISSING identifies a remaining evidence condition. NUMERIC_PASS/FAIL is the existing numerical predicate only. MISMATCH is a diagnostic comparison result; this audit introduces no new capacity gate.

| ID | Criterion | P2-32 | P2-64 | Evidence and scope | Minimal next action for the lead |
| --- | --- | --- | --- | --- | --- |
| Q1 | Actual H200 and native configuration/topology | SATISFIED, original reference | SATISFIED, original reference | 4/8 nodes, 32/64 H200 ranks, P16+D16/P32+D32, TP8, DP2/DP4, EP16/EP32, PP1. Retained vLLM 0.28.0 and torch 2.13.0+cu130; dummy load, FP8 KV, utilization 0.85, effective FULL_AND_PIECEWISE. Original R4 and all node/rank artifacts agree. `S:P32_run_check_status.groundtruth_clean`, `S:P64_run_check_status.groundtruth_clean` | Reuse the original configuration and receipts; bind any successor to its own effective configuration. |
| Q2 | Original request population, repetition, lifecycle and host graph admission | SATISFIED, original R4 | SATISFIED, original R4 | Each has three independently warmed 128-row formal repeats, 7 idle boundaries, every rank and 6 phases/rank, ordered shutdown, Succeeded/exit0/restart0. R4 graph coverage is successful host returns/replay returns; it claims no device completion or timing. `S:P32_run_check_status.groundtruth_clean`, `S:P64_run_check_status.groundtruth_clean` | Reuse all three exact repeats and preserve these coverage limits. |
| Q3 | Packed registration geometry on the clean reference | SATISFIED | SATISFIED | 96 registration rows: one region, 1,002,240 B stride; 109,934/112,914 blocks. P2-32 example `nodes/gpu-h200-0082.lgcm.sh.istep.fun/server.log:209`; P2-64 example `nodes/gpu-h200-0040.lgcm.sh.istep.fun/server.log:239`. `S:NIXL`, `S:P32_nodes/gpu-h200-0082.lgcm.sh.istep.fun/server.log`, `S:P64_nodes/gpu-h200-0040.lgcm.sh.istep.fun/server.log` | Preserve physical stride separately from logical component bytes when binding a payload model. |
| Q4 | Actual request-level selected descriptors and packed payload | PARTIAL | PARTIAL | Existing source/CPU control selects [24,2,2,2,16] live blocks at T6144; 46 packed descriptors imply 46,103,040 B/lane. The simulator reports 39,521,664 component B/lane. Registration is observed; request selection is source-derived and has no independent wire-byte witness here. `S:GROUP_COUNTS`, `S:NIXL`, `S:PAYLOAD_SOURCE`, `S:A10` | Use the lead's approved payload/links evidence to bind selected descriptor lengths and rank receive ranges. Do not relabel component bytes as wire bytes. |
| Q5 | H200 NIXL timing/payload fit admitted for this configuration | PARTIAL; H200 binding missing in inspected inputs | PARTIAL; H200 binding missing in inspected inputs | Both archived configurations use 126.2 Gbit/s and 0.0647 ms. Existing provenance says H800 two-node point-to-point fit. Neither an accepted H200 replacement fit nor its admission is bound by these inputs. Formal measured CC remains permitted by decision 46. `S:P32_B29_CONFIG`, `S:P64_B29_CONFIG`, `S:EX_SITU`, `S:A02` | Finish the already assigned H200 links/KV analysis and fit binding. Report diagnostic results separately; preserve the formal input until the controlling decision changes it. |
| Q6 | Native startup non-KV and cache-budget measurements | SATISFIED, startup scope | SATISFIED, startup scope | All 96 rank records retain exact requested, available-KV, consumed and transient-headroom scalars; model-load deltas and workspace lists are present. Every requested-minus-available identity passes. Actual graph memory is logged at 2.35-2.38 GiB. `S:P32_nodes/gpu-h200-0082.lgcm.sh.istep.fun/serving/rank0.json`, `S:P64_nodes/gpu-h200-0040.lgcm.sh.istep.fun/serving/rank0.json`, `S:NATIVE_MEMORY`, `S:NATIVE_COUNTERS` | Reuse this case-specific ledger and preserve counter overlap and precision limits. |
| Q7 | Independent allocation-category and per-phase peak accounting | PARTIAL | PARTIAL | Inventory explicitly marks independent stage peaks unverified. Allocator high-water includes resident allocations; workspace, free-memory consumption, model-load delta and graph logs cannot be summed as disjoint categories. Spec 10.4 excludes a new exact quantized capacity framework. `S:P32_nodes/gpu-h200-0082.lgcm.sh.istep.fun/serving/rank0.json`, `S:P64_nodes/gpu-h200-0040.lgcm.sh.istep.fun/serving/rank0.json`, `S:NATIVE_COUNTERS`, `S:A10` | Retain the limitation; request further phase/category evidence only if a concrete existing acceptance question needs it. Do not add a new capacity-framework prerequisite. |
| Q8 | F4 autonomous planner agrees with native pool size | MISMATCH: +6,100 blocks, +5.548783816% | MISMATCH: +7,709 blocks, +6.827319907% | Planner 116,034/120,623 versus native 109,934/112,914; requested memory and non-KV terms differ. Existing F4/B28 comparisons have five equal metric artifacts per scale. These low-pressure replays do not establish capacity accuracy. `S:P32_F4`, `S:P64_F4`, `S:P32_F4_B28`, `S:P64_F4_B28`, `S:PLANNER_SOURCE` | Keep F4 diagnostic and record the accounting difference. No source repair or replay is part of this sidecar. |
| Q9 | Measured capacity is representable and bound to this case | SATISFIED, measured-capacity isolation | SATISFIED, measured-capacity isolation | Existing B29 scheduler has explicit 109,934/112,914 blocks. This report records capacity_source=measured for these audited overrides, bound to the native hardware/topology/configuration; it makes no autonomous-capacity claim. Spec 10.4 permits this isolation mode. `S:P32_B29_CONFIG`, `S:P64_B29_CONFIG`, `S:A10` | Preserve the measured source and configuration binding in final bundle records; do not transfer these values across layouts or hardware. |
| Q10 | All four numerical errors <=10% in every retained B29 repeat | NUMERIC_PASS: 12/12; maximum 5.607176024% | NUMERIC_FAIL: repeat01 TTFT 12.254940665% | Reused reading rows; P2-64 native TTFT 779.8887274002482 ms, Frontier 684.313826604259 ms, absolute error 95.57490079598915 ms. No gate is rerun and no pooled verdict replaces a repeat. `S:P32_B29`, `S:P64_B29`, `S:A09` | The lead owns ongoing comparisons and cause diagnosis. Keep all repeats and the unchanged 10% predicate. |
| Q11 | Numerical part of same-model/H200 32-to-64 promotion <=30% | NUMERIC_PASS: all 12 finite errors <=30% | Not a 64-to-successor predicate | P2-32 meets the numerical part only. Qualification, generation/workload binding and the intended successor's basic receipt remain separate. `S:P32_B29`, `S:A09` | After the already assigned qualification work passes, bind the exact admitted 32 case and successor in a future pre-launch receipt. |
| Q12 | Historical P2-64 clean reference/basic-validation evidence | Original R4 preserved | SATISFIED, original authorization/basic scope | P2-64's original manifest explicitly authorizes clean native reference and basic validation; both original jobs were historically non-preemptible and terminal Succeeded. This is evidence for their original purpose and does not produce a new formal promotion. `S:P32_run_manifest`, `S:P64_run_manifest`, `S:P32_platform_terminal_receipt`, `S:P64_platform_terminal_receipt`, `S:A07`, `S:A08` | Bind applicable existing basic evidence to any intended successor; preserve its original authorization and verdict. |
| Q13 | Complete promotion receipt existed before a new formal64 submission | MISSING future receipt/binding | No historical promotion binding; no retroactive claim | Neither original manifest has promotion/admission fields. The limited receipt-filename check found none in the two named retained roots. This does not prove global absence. Current 11/18/19 require qualified reference/coverage, successor/basic binding and a receipt before formal64 launch; current resource class applies to new work. `S:P32_run_manifest`, `S:P64_run_manifest`, `S:A07`, `S:A08`, `S:A09` | The lead records a complete receipt before any eligible new formal64 campaign under standing authority. Never backfill the 2026-10-25 jobs as a pre-launch promotion. |
| Q14 | Complete 05/09/11/19 formal qualification and acceptance | OPEN | OPEN | Hardware/R4, startup measurements and numerical predicates do not discharge fresh A6, qualified coverage, payload/timing binding and required promotion. Current checkpoints retain these open conditions. `S:A03`, `S:A05`, `S:A06`, `S:A07`, `S:A09` | Continue the existing owners' A6, fit and qualification path; retain the current open tickets and diagnostic/formal distinction. |

## Packed payload reconciliation

For T6144 the retained CPU execution of extracted native methods selects the following group suffixes. These are not new GPU observations. (`S:GROUP_COUNTS`)

| Group | Live blocks | Component B/page | Packed descriptor B/page | Component B/lane |
| --- | ---: | ---: | ---: | ---: |
| compressed | 24 | 1,002,240 | 1,002,240 | 24,053,760 |
| swa_0 | 2 | 823,680 | 1,002,240 | 1,647,360 |
| swa_1 | 2 | 786,240 | 1,002,240 | 1,572,480 |
| csa_state_0 | 2 | 870,912 | 1,002,240 | 1,741,824 |
| hca_state_0 | 16 | 656,640 | 1,002,240 | 10,506,240 |

| Total | 46 | Varies | 1,002,240 | 39,521,664 |

Packed descriptor extent is `46 * 1,002,240 = 46,103,040 B/lane`. The component shortfall against that extent is `6,581,376 B/lane`, or `14.27536231884058%` using packed extent as the denominator. The pinned worker creates one descriptor per packed block and local descriptor lengths use block stride; remote addressing also uses registered block lengths. This establishes the source-derived distinction. It does not observe a request's native transport byte count or justify multiplying by all TP ranks: rank receive scope, replication and deduplication still need their own actual descriptor binding. (`S:NIXL`, `S:PAYLOAD_SOURCE`, `S:A10`)

## Native memory and capacity reconciliation

Values below are exact retained counters in bytes per physical rank. Multiple values are observed rank variants; they are not confidence intervals. (`S:P32_nodes/gpu-h200-0082.lgcm.sh.istep.fun/serving/rank0.json`, `S:P64_nodes/gpu-h200-0040.lgcm.sh.istep.fun/serving/rank0.json` and all 96 records in `/cases`)

| Counter | P2-32 B/rank | P2-64 B/rank |
| --- | ---: | ---: |
| requested_memory | 127,603,313,869 | 127,603,313,869 |
| model_load_delta_bytes | 13,605,207,552 | 9,006,153,216 |
| available_kv_cache_memory_bytes | 110,181,056,717 / 110,197,833,933 | 113,167,401,165 / 113,184,178,381 |
| total_consumed | 15,447,621,632 / 15,464,398,848 | 12,461,277,184 / 12,478,054,400 |
| peak_activation_memory | 1,957,858,304 | 1,957,858,304 |
| cudagraph_memory_estimate | 0 | 0 |
| workspace_bytes (sum of each recorded list) | 1,610,612,736 | 3,221,225,472 |
| requested minus available KV (pre-graph non-KV) | 17,405,479,936 / 17,422,257,152 | 14,419,135,488 / 14,435,912,704 |
| pre-graph non-KV minus model-load delta | 3,800,272,384 / 3,817,049,600 | 5,412,982,272 / 5,429,759,488 |

Every rank satisfies `requested_memory - available_KV = total_consumed + peak_activation_memory`. The pinned implementation defines consumed memory as a free-memory delta covering persistent allocations and adds transient headroom. Here `cudagraph_memory_estimate=0`; the later startup logs report actual graph usage of 2.35-2.38 GiB rounded to two decimals. The peak_activation field in this native implementation is transient headroom, not a separately isolated phase peak. (`S:NATIVE_MEMORY`, `S:NATIVE_COUNTERS`)

The pre-graph non-load residual is 3,800,272,384-3,817,049,600 B for P2-32 (about 3.54-3.55 GiB) and 5,412,982,272-5,429,759,488 B for P2-64 (about 5.04-5.06 GiB). These residuals are arithmetic differences. They do not independently identify activation, workspace, communication or graph allocations. The allocator high-water fields include resident KV and later persistent allocations, so adding them to workspace or graph counters would double-count. Independent stage peaks remain explicitly unverified.

| F4 term | P2-32 | P2-64 |
| --- | ---: | ---: |
| requested_memory_bytes | 128,687,957,606 | 128,687,957,606 |
| parameter_memory_per_device_bytes | 12,393,714,140 | 7,794,659,804 |
| non_kv_cache_overhead_bytes | 0 | 0 |
| physical_block_size_bytes | 1,002,240 | 1,002,240 |
| reserved_blocks, included in returned pool | 1 | 1 |
| total_memory_gb input | 141 | 141 |
| Planner pool | 116,034 | 120,623 |
| Native pool | 109,934 | 112,914 |
| Excess blocks | 6,100 | 7,709 |
| Relative excess against native | 5.548783815743992% | 6.827319907186% |

The existing planner formula reproduces both outputs: `floor((requested - parameter - overhead) / packed_stride)`. The native minimum retained available-KV value divided by that stride reproduces its observed pool at each scale. F4 requested budget is 1,084,643,737 B higher than the native requested budget, and its parameter term is 1,211,493,412 B (about 1.13 GiB) below the clean native model-load delta at both scales. The additional native non-load residual also differs by scale. These terms explain why equating analytical parameters alone with native non-KV consumption is unsupported; they are not a proposed calibration constant. (`S:P32_F4`, `S:P64_F4`, `S:PLANNER_SOURCE`, `S:CAPACITY_SOURCE`)

The existing F4/B28 comparison reports FINISHED/FINISHED and five byte-identical metric artifacts at each scale. This audit verifies that recorded comparison JSON and does not replay it or repeat the byte comparison. The low-pressure workload does not discriminate these pool sizes. The original F4 input audit's earlier statement that this was unverified is superseded by that existing comparison for metric equality only; capacity accounting remains unresolved. (`S:P32_F4_B28`, `S:P64_F4_B28`, `S:EX_SITU`)

Spec 10.4 permits native measured capacity to isolate execution/scheduling errors, provided the existing allocator can express it and the record says `capacity_source=measured` with the topology/configuration scope. The audited explicit pools satisfy representability and this report supplies that source classification for these exact cases. This is separate from autonomous capacity accuracy. The spec does not request a new exact quantized weight-capacity framework. (`S:A10`, `S:P32_B29_CONFIG`, `S:P64_B29_CONFIG`)

## Numerical and promotion conditions

The table reuses existing B29 relative errors in percent; no comparison or gate is executed. Every row remains independent. (`S:P32_B29`, `S:P64_B29`)

| Case/repeat | TTFT error % | TPOT error % | Request E2E error % | Total token THPT error % |
| --- | ---: | ---: | ---: | ---: |
| P2-32/repeat01 | 4.722918375726 | 0.100175341144 | 0.015943882935 | 0.134548186228 |
| P2-32/repeat02 | 5.447453481087 | 0.192043544505 | 0.092723369841 | 0.090973777281 |
| P2-32/repeat03 | 5.607176023818 | 0.173713143990 | 0.269220048079 | 0.047913549558 |
| P2-64/repeat01 | 12.254940665008 | 0.529031900371 | 0.870327866558 | 0.457663161175 |
| P2-64/repeat02 | 5.675932671490 | 3.656982823073 | 3.710081961603 | 0.022442923649 |
| P2-64/repeat03 | 4.727235248268 | 2.433412047035 | 2.493894115007 | 0.170758391514 |

P2-32's worst row is repeat03 TTFT: native 365.3656886308454 ms, Frontier 344.8789913386803 ms, absolute error 20.486697292165104 ms, relative error 5.607176023817675%. P2-64's failing row is repeat01 TTFT: native 779.8887274002482 ms, Frontier 684.313826604259 ms, absolute error 95.57490079598915 ms, relative error 12.254940665008354%. Exact native/Frontier values, units, numerators, denominators and source signed differences for all 24 rows are retained in `/cases/p2_*/b29_metrics`.

The P2-32 numerical 30% predicate is also satisfied, but a promotion receipt additionally needs the admitted same-model/H200 comparison, qualified reference and coverage, source/table generations, workload hashes, all three repeats, and the intended P2-64 successor plus its own basic-validation binding. Every value must be finite, and a zero native metric has undefined relative error and cannot qualify. The receipt must exist before a new formal64 submission. This report is not such a receipt. (`S:A09`)

The preserved original manifests were written at 2026-10-25T15:23:22.997270Z (P2-32) and 2026-10-25T15:54:37.229732Z (P2-64). Both bind the temporary original H200 authorization; P2-64 explicitly includes clean native reference and basic validation. Neither manifest has a promotion/admission binding field. Historical terminal receipts are non-preemptible/Succeeded with exit0/restart0. No retrospective promotion or change to their original R4 purpose follows. (`S:P32_run_manifest`, `S:P64_run_manifest`, `S:P32_platform_terminal_receipt`, `S:P64_platform_terminal_receipt`)

The only receipt-filename search is recorded in `/receipt_presence_check`: `rg --files -g '*promotion*' -g '*admission*' -g '!*.py' -g '!*.jsonl'` over main task `h200_mainline/` and `h3o_operator_costs/codex_resume_20261031/`. It returns 1 with no matches. This is limited to those retained roots and filename patterns; it is not a global absence or full content search. A future receipt must be bound regardless of filenames.

Current Tickets11/18 prescribe the current resource class and pre-launch conditions for new formal64 work, while their current checkpoints explicitly preserve historical authorization. The lead remains responsible for any launch-time personal reservation check and resource binding. This audit does not inspect platform reservations, submit a successor or request renewed budget approval. (`S:A07`, `S:A08`)

The minimal remaining route is to finish the already assigned hardware/payload/fit and A6 qualification evidence, keep the existing F comparisons diagnostic, bind any eligible future promotion before submission, and apply all four unchanged 10% predicates to every admitted repeat. Tickets05/09/11/19 remain OPEN. No additional source implementation, GPU probe, replay or broad census is assigned by this report.

Validation commands, actual PASS/FAIL values and scope limits are in the [test report](test_report_2026-10-31_p2_qualification.md); the [verification result](evidence/verification.json) binds the retained source and report checks.
