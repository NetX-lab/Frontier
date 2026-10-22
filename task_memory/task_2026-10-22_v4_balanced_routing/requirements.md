# Balanced routing alignment requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Preserved primary verbatim approvals and merged the delegated C implementation/R1 contract. |
| 2026-10-22 | YC approved D-V4-BALANCED-SCOPE-23-900; replaced the aggregate bound while retaining the recorded earlier overrun. |
| 2026-10-22 | Recorded the actual aggregate scope overrun and an unapproved complete-scope amendment proposal. |
| 2026-10-22 | Recorded explicit D-V4-BALANCED-RUNTIME-SELECTOR approval for the shared interface exception and14-file/500-line total scope. |
| 2026-10-22 | Recorded YC's routing/order steering and the active V4 runtime binding obligation. |

## [Original Request]

> 补充任务执行的2个规范的要求：1.routing暂时统一采用balanced distribution方案对齐（frontier和vllm的routing都采用均衡分布的设定） 2.校准过程中，先进行workflow和gpu侧的对齐（包括op gap计算），最后再进行cpu overhead（gap）的计算测量，不要在当前阶段就引入cpu overhead的大批分析、测量和suits，推迟到矫正最后。

Decision `D-BALANCED-ROUTING-AND-CALIBRATION-ORDER` is recorded in main GLM/V4 requirements. Implement the smallest existing-mechanism binding for balanced token-to-expert distribution, with verified matching scope and integerization on Frontier/native. Do not substitute random-uniform routing or infer balance from weights. Historical measurements retain their original routing identity.

Scope covers the active V4 prefill supplement and native32 successor. Existing source approvals/budgets remain; no extra GPU run or GLM resumption follows from this record. Keep the source2cf/C26 runtime lock; use existing supported registration/configuration rather than an unreviewed native patch. If matching semantics require a new interface or materially different policy, present the exact evidence and decision before expanding.

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

## Historical scope-amendment proposal

Root independently counted the current shared WIP against9d44675:10 Python files/413 changed lines. Native binding commitca1ca86 adds five distinct Python files/109 changed lines. The aggregate15/522 already exceeds the approved14/500; separate branch delivery does not exempt these changes. Root owns this allocation error. Further source edits, combined acceptance, packaging and launch are paused pending a decision; retain existing WIP and evidence.

The real V4 mixed-routing estimator test fails in both training orders. The existing typed contract and selected serializer omit routing identity, while disk hashes already differ. The proposed fix reuses ResolvedLayerContract and its serializer, adding two maintained files and approximately15 lines rather than another registry key mechanism. Shared completion therefore estimates12 files/428 lines. The complete binding estimate in the native owner's design is11 additional files/426 lines, giving23 files/about854 lines overall.

Root proposes a single aggregate amendment to at most23 maintained source/test/binding files and900 changed lines. It covers typed identity, native actual source/Frontier-input binding, maintained prefill startup adapter and constructed-model evidence, row propagation/consumer rejection, two fresh-generation builders and focused controls. This is an estimate and a request, not approval. Existing GPU budgets, frozen native/C26 source, historical packages and the calibration phase order remain governed by their existing decisions.

## [Original Request] D-V4-BALANCED-SCOPE-23-900

Root asked:

> boss YC，是否批准将 routing 的总实现上限从 14 文件/500 changed lines 修订为 23 文件/900 changed lines？完整接线预计约 854 行，涵盖 typed identity 修复、native/prefill 实际绑定、测量行身份传递及两个新包 builder。目前两条分支已合计 15 文件/522 行，这是我的总量分配失误，后续源码编辑和封包已暂停。GPU 预算仍为已批的 1×H800/900s 与 32×H800/60min。

YC answered:

> 批准 23 文件/900 行 (Recommended)

The current aggregate limit is23 maintained source/test/binding files and900 added-plus-deleted lines. It supersedes the earlier14/500 bound prospectively and authorizes completion of the retained work. Root counts the union across worktrees, including both maintained builders and the successor adapter. This does not erase the earlier allocation error. All behavioral acceptance criteria and GPU budgets remain unchanged.
## Prefill delegated implementation contract

The following is the historical C delegation, not a new user approval. Its original200-line limit is superseded by the current root allocation within the unchanged approved aggregate ceiling.

Implement balanced prefill wiring in five maintained files only: successor adapter `tests/comparison/calibration/profile_dsv4_prefill.py`, `dsv4_attention_runner.py`, `dsv4_attention_rows.py`, `execution_time_predictor/deepseek_v4.py`, and the existing configured-capacity builder. Maximum cumulative added-plus-deleted routing lines: 200. Root approval D-V4-BALANCED-SCOPE-23-900 is authoritative in the root worktree requirements and plan. Ordinary merge of shared `410f062` is permitted. Native owner alone writes integration tests.

Require explicit balanced_topk registration before profiling CLI/model construction, flat strategy origin and hash binding, actual constructed model registry/router/environment/native source evidence, raw-to-CSV propagation, and typed rejection of unbalanced or missing balanced identity by the explicit-balanced isolated consumer. Preserve absent-selector behavior and immutable historical packages. Strategy SHA256 must remain `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`.

Preserve native2cf/C26 bytes and the configured7168/8192/128/effective128, physicalTP1/eager/six-layer, sequential6144/C0 warm2/measured3 geometry for both methods. Amend the existing builder only; root selects the fresh balanced generation, pins final committed source, builds/seals/reviews, and performs GPU operations. No new harness/flag/policy/fingerprint, CPU-overhead work, GPU, Docker, platform, SSH, or cloud operations.

Deliver a commit, exact five-file diff/count, focused existing checks and direct CPU controls, native-owner integration cases, identity schema, builder baseline and outstanding hardware qualification.

## [Original Request] W128-V4-BALANCED-PREFILL-R1

Spec review found P2: `routing_columns` checks runner keys only and accepts non-MoERunner class values under matching names. Replace the key-only comparison with the exact expected runner mapping, preserving other controls. Reproduce RED through actual `routing_columns` and `read_isolated_scope`, then GREEN and legacy positive through the same direct CPU controls. No new large or persistent harness; no native/B/shared file edits. Only the existing C source file needs a change; do not modify builder, build/seal, or perform GPU work.

C had189 routing changed lines. Root reserves12 lines for six final builder generation/source replacements; tentative combined target is<=202 and root owns allocation. Report the exact net cumulative baseline diff and new patch delta before any overrun. Root selects `h800_prefill_balanced`, `PREFILL_WARM_6144_C0_BALANCED`, matching tag/cloud output/task identity and source. Root will reuse the native builder's current-clean-HEAD snapshot mechanism instead of embedding a self-referential HEAD literal. This supersedes prior builder-finalization guidance; builder remains untouched by this R1 task.

Deliver source/report commit, RED/GREEN evidence, exact net counts and line references. Review remains source-only until root joint checks and image/runtime evidence.
