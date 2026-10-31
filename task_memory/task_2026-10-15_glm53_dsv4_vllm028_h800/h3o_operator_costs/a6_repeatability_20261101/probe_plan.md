# A6 workload repeatability probe

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Froze the hypothesis, selection, analysis, resource budget and CPU checks before submission. |

## Purpose and decision

Tickets 05/09 remain open because four attention scopes fail A6 p90. Exact-feature lookups alone force the failures. The retained independent G2h runs do not reproduce a persistent first-CSA-layer/reuse-layer contrast. Under D-NO-INTERRUPT, measure repeated reconstructions of the same workload geometry before changing measurement inputs or predictor features. This probe is diagnostic; no resulting row replaces a published input and no result closes A6 or an E2E gate.

Hypothesis: the large single-step timing contrasts vary across independently rebuilt cases even when request geometry, native graph dispatch and kernel ownership stay equal. Record every repetition, including extremes. Large within-case variation or unstable signed contrasts supports insufficient precision of individual observations. Stable repeated values would instead justify inspecting what state differs between the earlier run and this probe. Neither outcome alone proves a cache, clock, concurrency, input-content or instrumentation cause.

## Fixed cases and execution

Four failing cases and four adjacent controls are selected before measurement:

| Failing case | Adjacent control | Scope used for selection |
| --- | --- | --- |
| g1-e0-0263-full | g1-e0-0262-full | index_compressor_fused_compress_store |
| g1-e1-0729-first_tail | g1-e1-0730-first_tail | compressor_fused_compress_store |
| g1hs-e0-0101-full | g1hs-e0-0100-full | q_projection |
| g1hs-e0-0254-piecewise | g1hs-e0-0253-piecewise | index_fused_query_prepare |

Each case is independently reconstructed 16 times, then measured for one step: 128 case IDs, 96 on engine 0 and 32 on engine 1, executed sequentially on one GPU. The JSON order is fixed: repeat-major order within each engine, with each failing case followed by its control. `warmup=0, measured=1` preserves decision 35. The existing runner drains or aborts a case before the next case begins. Engines, capture sizes, seed 1234, 200000-us hold, balanced routing, reduced physical TP1/target TP8, model and native source pins are unchanged.

The source is frozen G1h `a689ed05f462eb269b9dffe640da151e6614a5ef` in `/data/ycfeng/Frontier/.worktrees/g1h-supplement-build`. The existing package builder and launcher are reused. No production source edit is needed. Request geometry is fixed; random prompt contents vary under the existing sequential seeded generator, and GPU cache/clock state is not reset. This limits causal interpretation. Frozen G1h sampling differs from the later corrected G2h sampler, so this is an attention repeatability probe, not sampling qualification.

## Analysis and acceptance

1. Require all 128 measured steps, exact `phase`, `requests` and `cudagraph_dispatch` equality to the pre-recorded predictions, native source identity, graph-node ownership and complete worker artifacts. Record any mismatch without dropping the case.
2. Reuse the production replay reducer. Retain per-case, per-scope, per-layer-group timings and kernel/sample counts. For the four failing scopes, report every positive measurement, median, sample standard deviation, coefficient of variation, p10/p90, and absolute/relative difference from the original row. A zero denominator remains undefined.
3. Split each sequence into repetitions 0-7 and 8-15. Report their medians and signed/absolute relative differences. Compare first-layer/reuse-layer contrast signs and magnitudes. Summaries describe measurement stability; they do not introduce a new formal threshold or replace A6's existing median/p90 limits.
4. If raw identity and counts agree but timings vary, evaluate whether a repeat-based measurement generation is justified. If geometry or counts differ, explain that mismatch before interpreting times. Do not tune RF or add planner/layer features based only on this probe. Any new table generation retains the original evidence and the unchanged qualification population.
5. Keep one probe submission. A failure is diagnosed before considering another allocation. No automatic GPU retry.

## Resources and evidence

One non-preemptible H200 in step_main, 16 CPU, 128 GB; image digest `2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`. Local source mount is the frozen build worktree. Cloud output: `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/a6_repeatability_20261101_01`. All retained cloud content stays under the personal directory.

Existing G1h timing model: `2*486.104 + 1568*0.205 + 128*0.2 = 1319.248 s`, or **0.366457778 GPUh**. The payload timeout is 5400 s, **1.5 GPUh** exposure. The prior G1h-G4h ledger is approximately **4.52/7 GPUh**; prior usage plus payload timeout is approximately **6.02/7**, leaving **0.98 GPUh** for platform overhead/reserve. Prior usage is approximate and the timeout begins at the payload, so neither figure is an exact platform bill. Record observed container and platform elapsed time after completion, including startup/failure time.

CPU checks: `verification.json` confirms 546 manifest files, 527 source snapshot files, all eight original definitions, all 128 clone definitions/predicted steps and their equality to the retained native steps. `payload_check.json` captures actual ResourceConfig with no platform submission. `quota.json` at 2026-10-31T17:56:41Z contains the complete personal nonterminal job list: all six existing jobs are preemptible; non-preemptible reservations are zero. The quota-group API returned HTTP403; free capacity is unknown. The new job stays within the personal 32-GPU non-preemptible ceiling and may queue normally.
