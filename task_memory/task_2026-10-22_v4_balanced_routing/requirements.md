# Balanced routing alignment requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded explicit D-V4-BALANCED-RUNTIME-SELECTOR approval for the shared interface exception and14-file/500-line total scope. |
| 2026-10-22 | Recorded YC's routing/order steering and the active V4 runtime binding obligation. |
| 2026-10-22 | W128 recorded root's combined-budget integration finding and required complete binding estimate. |
| 2026-10-22 | Recorded approved23/900 aggregate scope and B6/250 allocation; native-final controls awaiting local reallocation. |

## [Original Request]

> 补充任务执行的2个规范的要求：1.routing暂时统一采用balanced distribution方案对齐（frontier和vllm的routing都采用均衡分布的设定） 2.校准过程中，先进行workflow和gpu侧的对齐（包括op gap计算），最后再进行cpu overhead（gap）的计算测量，不要在当前阶段就引入cpu overhead的大批分析、测量和suits，推迟到矫正最后。

Decision `D-BALANCED-ROUTING-AND-CALIBRATION-ORDER` is recorded in main GLM/V4 requirements. Implement the smallest existing-mechanism binding for balanced token-to-expert distribution, with verified matching scope and integerization on Frontier/native. Do not substitute random-uniform routing or infer balance from weights. Historical measurements retain their original routing identity.

Scope covers the active V4 prefill supplement and native32 successor. Existing source approvals/budgets remain; no extra GPU run or GLM resumption follows from this record. Keep the source2cf/C26 runtime lock; use existing supported registration/configuration rather than an unreviewed native patch. If matching semantics require a new interface or materially different policy, present the exact evidence and decision before expanding.

## Acceptance

## [Original Request] Shared-interface exception

Root asked:

> 是否批准新增每案例的 moe_gating_routing_runtime_path 选择项，并注册 balanced_topk，贯通producer/consumer和运行绑定？建议作为算子身份配置例外，在新路径GPU耗时尚未测得时实施；默认沿用现有行为，缺匹配实测仍typed拒绝。该routing变更含策略、调用链和测试，最多14个维护文件/500行，不增加已批GPU预算。

YC answered:

> 批准接口例外 (Recommended)

Decision `D-V4-BALANCED-RUNTIME-SELECTOR` approves the optional per-case runtime selection, central registry entry, producer/consumer/run binding and necessary tests within14 maintained files/500 changed lines including the existing211-line strategy slice. This explicitly permits the identity field before measured GPU latency is available. Preserve default behavior and typed missing coverage. Root coordinates file ownership and counts the combined diff. GPU budgets and native/C26 source remain as already approved.

## Acceptance criteria

- Identify effective routing entry on the actual H800 model/backend, including SP/TP/DP/EP and padded/valid token semantics.
- Match deterministic per-expert assignment counts and remainder ownership for the real decode/prefill shapes on both sides; verify actual binding rather than labels alone.
- Bind routing source/configuration to runtime identity, sealed files and actual worker initialization, including any graph-capture constraints.
- Complete focused correctness/review and update new packages before submission. Never relabel old unbalanced data as balanced.
- Defer CPU-overhead/gap studies until workflow and GPU/operator-gap alignment are complete.

## [Original Request] W128 combined-budget steering

Root reports shared410f062 at nine files/377 lines including strategy; mandatory cluster_role_config forwarding and possible prediction_model_registry identity correction may raise shared scope to11 files/about417 lines, leaving three files/83 lines under14/500. Do not use code golf, generated-file logic or weakened actual-origin/environment/runtime controls to force that remainder. Return a precise complete native32/prefill dependency and file/line estimate before further combined-scope expansion. Same balanced module and explicit policy must bind both configurations, actual runtimes and source identities; old unbalanced rows must not have an indistinguishable accepted execution identity. Frozen R2 and shadow negatives, native/C26 bytes and all old packages remain unchanged. Root owns review/final sealing.

## [Original Request] W128-V4-BALANCED-NATIVE-FINAL

YC approved D-V4-BALANCED-SCOPE-23-900. Verbatim question: “boss YC，是否批准将 routing 的总实现上限从 14 文件/500 changed lines 修订为 23 文件/900 changed lines？完整接线预计约 854 行，涵盖 typed identity 修复、native/prefill 实际绑定、测量行身份传递及两个新包 builder。目前两条分支已合计 15 文件/522 行，这是我的总量分配失误，后续源码编辑和封包已暂停。GPU 预算仍为已批的 1×H800/900s 与 32×H800/60min。” Verbatim answer: “批准 23 文件/900 行 (Recommended)”. Root records retain authority; the earlier allocation overrun is historical, not erased.

Current allocation: shared A12 files/450 routing lines, native B existing five files plus maintained native builder, six files/250 cumulative routing lines, prefill C five files/200 lines. B alone writes tests/unit/test_native_reader_binding.py, including root-supplied prefill integration negatives. Complete native actual environment/router/factory/simulator/MoERunner origins, real hash-bound Frontier input and builder/source/checker wiring. Root selects absent candidate_h800_p0_06; preserve05 and all old packages. Retain original builder bytes/hash in this worktree and add three dispatch pins separately under existing488-line dispatch budget (must remain<=500), then four routing pins to74. No generated routing code injection, native/C26 edits, GPU/platform/cloud/SSH/Docker, seal/submission or delegation.

Root supplies the actual775-field flat SimulationConfig at root task_memory/task_2026-10-22_v4_balanced_routing/balanced_frontier_input/frontier.json, SHA256747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897 and receipt.json. It reconstructs through existing create_flat_dataclass(SimulationConfig), not a miniature settings object or old calibration runner. Validate source/runtime/workload binding without claiming numerical pricing/graph/memory qualification or introducing H4 fits as an H5 dependency. Root revalidates role materialization on A final source. Observer errors stay soft evidence gates in32GPU workloads. Stop expansion and report exact reallocation before additional local scope.
