# CORE review repair progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Inspect frozen review, overlay and actual published routing delegation; no peer changes. |
| 2026-10-23 | Verify R1 actual inherited/delegated callable negative and record23PASS. |
| 2026-10-23 | Verify mandatory descriptor through automatic shared/independent training and preflight. |
| 2026-10-23 | Complete R3 current-owner selected proof coverage; record regression limits and182/230 effective budget. |

TaskID:D29-CORE-REVIEW-REPAIR. Baseline d96b3279 confirmed. Only root-owned development submodule3dac checkout differs initially; preserve gitlink unstaged.

- completed: root review and both exact overlays read; six exclusive paths frozen. Existing source math/registry and existing verifier reused. No new API/recursive verifier.
- in-progress R1: actual published chain inspected at `/data/ycfeng/tmp/d29_published_sources01/source/vllm/model_executor/layers/fused_moe/router/{fused_moe_router,base_router,routing_simulator_router}.py`; PRICING C1 independently lists actual `_compute_routing`, simulator and selected strategy callable in verifier. Hypothesis: with class/public/inherited selectors unchanged, changing only delegated callable remains admitted; constructor test must go RED before adding that chain.
- pending R2/R3: actual automatic producer/trainer and full artifact validator seams; user explicitly confirmed them. No full CLI or peer import.
- retained prior terminal: original seam03 notification arrived,exit1,1555.208038s,no timeout,internal2400s. Missing old collective input after metadata initialization fixed2a862c25. Raw evidence `/data/ycfeng/tmp/d29-core-position-main-seam03/{command.json,execution.json,run.log}`. Current branch does not include original-worktree terminal docs5c141339; preserve selected terminal here without writing there.

- completed R1: existing constructor/full-receipt test now loads the actual exported FusedMoERouter/BaseRouter/Simulator/strategy class definitions into sealed CPU fixture modules. No generic recursion, alternative verifier or production strategy implementation added. Foreign implementation executes through real inherited selection first; class, selectors,module file/spec/hash unchanged. RED01 wrong-result dtype failure was fixture-only and not claimed as RED. RED02:_compute_routing/simulate_routing/route_tokens3DID NOT RAISE;prior_select_experts1PASS. GREEN01:23PASS/11deselected9.21s at same constructor/verifier plus source,timer,replay,oracle negatives. Production3added lines;test repair34added/7deleted=41;R1gross44<=230,baseline totalsmoe_impl56/62,test303/350. Published export absence explicitlySKIP,not invented native sources.
- completed R2: valid existing producer rows first pass actual manager preflight. Drop routing_input, then call actual automatic shared `_train_ffn_models_for_cluster` or independent `_train_moe_models`;RED01 both reach forbidden estimator/metadata sentinel. Invalid descriptor already rejected,regex corrected to actualsource_frames. Add required routing_input and unconditionally reuse existing source validator, no trainer/helper edit. GREEN01:36PASS/3deselected68.81s including both automatic negative paths,valid producer/actual cold-warm fit/cache andlegacy defaults. R2delta47;test125/135,runtime103/133;combined repair91/230. Next R3 complete actual-main artifact validation, no CLI.
- completed R3: actual validator with full CPU artifact seam retains6144/1024,P6128+D16/1023 single decode,READpayload/3.5ms/readiness/reservation/release andgraph/topology. Existing count dict slicing first exposedTypeError;not claimedRED. Temporary existing-dict indexing letsRED02 expose16DID NOT RAISE for removed actualdummy-owner gate/router/prepare/allproofs/operator andpeer-onlyliveowner. Final path deletesexpert-emptycensus,usesexisting source_lane_index_for_ep foractualownerlive0 andgetsselectednativecontextfromdispatchplan.buffers.compute_context. Singleexistingselectedregistryplusbind_operator_queryrequiresapplicableops,exactCOMPUTEproofcorrespondence,finitepositiveequalphase durationandintegralpositiveactualcount;BF16/deferrednoinapplicableprepare. GREEN02:36PASS/3SKIP2.49s;union72PASS/3SKIP70.53s;R1final23PASS9.50s. R3delta91;total182/230,effectivetotal3450/4404,allcapsASTwhitespacePASS.
- verification limit: py312 integration collection andactualwrappergatheredASTseam hitexistingfastenersmissing;no depsinstalledorproductionchanged. Temporaryrunnerglobalsfixedthenstoppedatrealsameenvironmentblocker. Expandedoldfixtures73PASS/1SKIP/3FAIL;all3failcorrectmandatoryrouting_inputadmissionbecauseunownedtest_moe_grouped_gemm_padded_features.py:_position_source_rowsomitsdescriptor. Rootmustupdateitsfixture,butnoownershipgrantedsofileuntouched. Freshactualproducernegative/positivefit/cachetestsPASS;thisisnotlegacybalancedregression.

## 1. 规划完成情况

**汇总：** PASS 2｜BLOCKED/FAILED/SKIPPED 0｜TODO 2
| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| R1 bound delegated provenance | PASS |e67a9e65;23PASS |
| R2 supplied descriptor auto admission | PASS |36PASS;real shared/independent seam |
| R3 actual source-owner proof coverage | TODO |existing registry/mapper inspected |
| final regression/accounting/handoff | TODO |91/230current;no peer edits |

## 2. 偏离检测

**结论：**未偏移，正常推进。**检测窗口：**taskstart至R2GREEN｜**当前前沿：**R3。
| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR |actual inheritance fixture verifiesR1thenstops;R2existingproducerreused |
| D2 细节反复重试 | CLEAR |dtype hypothesis narrowedR1fixture;effectiveRED02thenGREEN |
| D3 支线问题修补 | CLEAR |threeuserreviewfindingsonly |
| D4 可并行任务停滞 | CLEAR |singleownerboundedverificationresources;extraagentsprohibited |
| D5 超范围镀金 | CLEAR |sixpaths;91/230;noalternativeverifier |
| D6 分析空转 | CLEAR |R1committed,R2verified |
| D7 路线违背 | CLEAR |newworktree;noWIPimport/fullCLI |
**文档检查：**S1–S6均未发现明显缺陷。**归因：**无。

## 3. 纠正建议

**分支：**A：spec/tickets合理，无需改动。**需停止的偏离行为：**无。
**接下去的路线：**1.R3actualvalidatorRED→GREENandBF16/FP8positive；2.affectedregression/overlayaccounting→committedhandoff。完成标志均为planacceptance，不含nativequalification。

## 4. 最值得关注的 3 点

1. BoundstrategychainactualRED/GREEN,R1existingselectorretained。
2. Automatictrainer/preflightrejectmissingdescriptorbeforeestimator，validfitscovered。
3. R3mustderivecurrentownerandselectedpreparefromexistingevidence;remaining139repairgross。
