# CORE review repair progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Inspect frozen review, overlay and actual published routing delegation; no peer changes. |
| 2026-10-23 | Verify R1 actual inherited/delegated callable negative and record23PASS. |

TaskID:D29-CORE-REVIEW-REPAIR. Baseline d96b3279 confirmed. Only root-owned development submodule3dac checkout differs initially; preserve gitlink unstaged.

- completed: root review and both exact overlays read; six exclusive paths frozen. Existing source math/registry and existing verifier reused. No new API/recursive verifier.
- in-progress R1: actual published chain inspected at `/data/ycfeng/tmp/d29_published_sources01/source/vllm/model_executor/layers/fused_moe/router/{fused_moe_router,base_router,routing_simulator_router}.py`; PRICING C1 independently lists actual `_compute_routing`, simulator and selected strategy callable in verifier. Hypothesis: with class/public/inherited selectors unchanged, changing only delegated callable remains admitted; constructor test must go RED before adding that chain.
- pending R2/R3: actual automatic producer/trainer and full artifact validator seams; user explicitly confirmed them. No full CLI or peer import.
- retained prior terminal: original seam03 notification arrived,exit1,1555.208038s,no timeout,internal2400s. Missing old collective input after metadata initialization fixed2a862c25. Raw evidence `/data/ycfeng/tmp/d29-core-position-main-seam03/{command.json,execution.json,run.log}`. Current branch does not include original-worktree terminal docs5c141339; preserve selected terminal here without writing there.

- completed R1: existing constructor/full-receipt test now loads the actual exported FusedMoERouter/BaseRouter/Simulator/strategy class definitions into sealed CPU fixture modules. No generic recursion, alternative verifier or production strategy implementation added. Foreign implementation executes through real inherited selection first; class, selectors,module file/spec/hash unchanged. RED01 wrong-result dtype failure was fixture-only and not claimed as RED. RED02:_compute_routing/simulate_routing/route_tokens3DID NOT RAISE;prior_select_experts1PASS. GREEN01:23PASS/11deselected9.21s at same constructor/verifier plus source,timer,replay,oracle negatives. Production3added lines;test repair34added/7deleted=41;R1gross44<=230,baseline totalsmoe_impl56/62,test303/350. Published export absence explicitlySKIP,not invented native sources.
