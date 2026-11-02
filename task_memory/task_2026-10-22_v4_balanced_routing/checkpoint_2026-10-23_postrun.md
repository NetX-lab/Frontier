## 1. 规划完成情况

**Modification History**

| Date | Change |
| --- | --- |
| 2026-10-23 | Read-only post-run checkpoint: classified native06 terminal/admission result, independent RCA/client acceptance, D-Q10 isolated prerequisite and the authorized versioned reader correction. |

**汇总：** PASS 15｜BLOCKED/FAILED/SKIPPED 3｜TODO 4

任务ID：`W128-V4-POSTRUN-CHECKPOINT`。主 worktree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`（HEAD b62c155）。下文 R 指 `task_memory/task_2026-10-22_v4_balanced_routing/`，P 指 `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/`。本检查只写本文件，不运行测试/Git/平台操作，不做源码质量结论；PASS 仅表示记录中的子项完成。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| R1：实际 routing 路径、整数分配与接口决定 | PASS | 策略6ca6777及源码比较完成（R/requirements.md:27–48）；实际运行证据由R4-B行承担 |
| R2-A：shared selector、typed identity | PASS | 186dfef、独立复审、strict77，合并292c489 |
| R2-B：native 来源、完整输入及 effective-DP | PASS | c491e6a两P2关闭、ce9ceba merged133PASS |
| R2-C：prefill 注册、inventory、行及 consumer 身份 | PASS | 合并84c806f、最终39b2bf8 |
| R2：总范围、独占分配及 corrective decision | PASS | 当前23/953=A432/B319/C202；dispatch491、32GPUh不变 |
| R3-C：固定源 default fidelity | PASS | 39b2bf8 strict77=77/77、451 artifacts、clean source/harness |
| R3-C：当前源184项收据 | PASS | root184PASS，零失败/错误/跳过（R/test_report_2026-10-23_prefill_acceptance.md:45） |
| R3-B：merged/source review 及最终 fidelity | PASS | merged133PASS；e9e79d4 source-only APPROVE；该 revision strict77=77/77 |
| R4-C：新 prefill package/image | PASS | 529 sealed inputs、60 source/519 tool/20 binary及实际 adapter 检查完成 |
| R4-C：已批1H800补测及 isolated 数据接收 | PASS | exp-1023-023708-916112，500s，两个6-row表，8 cloud hashes；仅 ACCEPTED_ISOLATED_DATA |
| R4-B：最终 native06 package/joint/image/freeze/seal | PASS | R3ca8ddcea/R229523700冻结、43输入seal/manifest8eb3f794、locked image2286e853、双Python check-only INPUT_HASHES_PASS、已提交（R/test_report_2026-10-23_native_ready.md:26） |
| R4-B：已批32H800 native06 实际运行 | PASS | exp-1023-043123-830898 Succeeded4/4、exit0/restarts0、1364/1263/1386/1388s=12.002222GPUh<32GPUh（R/native06_platform_terminal.json:8–10,123） |
| 父18：native reference 准入（frozen R3 全判据） | FAILED | 8判据7PASS，`rank_runtime_graph` FAIL：`Actual balanced routing strategy/source/environment identity mismatch`；reference 仍 UNVERIFIED（R/test_report_2026-10-23_native_ready.md:42） |
| native06 routing identity RCA | PASS | 32/32 rank 复现同一拒绝；FP8-only `moe_backends` 为空，43个MXFP4/MARLIN owner 与 routing maps 独立一致；策略/环境/origin/hash 全部匹配（P/native06_acceptance_20261023/routing_rca.md:13,33–54,80–97） |
| native06 客户业务与三轮稳定性独立验收 | PASS | 432=48warmup+384formal、6144/1024、7边界、10/0类；明确不主张 native admission（P/native06_acceptance_20261023/client_report.md:10,26–34,67,91） |
| 版本化 R4 reader 修正（D-V4-NATIVE06-ROUTER-OWNER-READER） | TODO | 已授权、实现进行中：≤3非core文件/300行、无新GPU、保留R3/R2与原始FAIL；尚无提交（`../v4-native-routing-reader/.../requirements.md:15,21–27`；同目录 progress.md:11–20） |
| D-Q10：独立 isolated prefill 前提（源码+矩阵） | PASS | ACCEPTED_ISOLATED_PREREQUISITE：144unit/94exact/legacy102-72/clean576d278 strict77=77/77、451 hashes、519 cache、serving_admission=false（`../v4-prefill-contract/.../primary_acceptance.json`） |
| D-Q10：primary merge | TODO | 分支 b7a179a 已提交于 `feat/v4-prefill-contract`；主源合并由 root 拥有、无 merge hash |
| 父05/09：CPU measured-consumer 剩余集成单元 | TODO | D-Q10 已覆盖 isolated reader/contract，family→实际 per-scope consumer 调用链仍未完成 |
| 父05/09：qualified balanced decode/graph 及完整 cost coverage | BLOCKED | 缺实测 balanced decode/graph/TP8-PD/first-tail：历史 q1 无 balanced 身份不能补标签（P/tickets/09-predictor-bundle.md:39,47） |
| 父19：完整 non-dummy path／四指标比较 | BLOCKED | 等已准入 reference 与合格完整 coverage（P/tickets/19-comparison-and-acceptance.md:116,130） |
| 14份当前状态同步（另一 owner 进行中） | TODO | 主 worktree 有7份文档未提交修改；本检查按指示不写、不重复 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次 checkpoint `R/checkpoint_2026-10-23_native06.md`（04:13）至本次（~05:25）。**当前前沿：** 版本化 R4 reader（含 root 复审/冻结）‖ D-Q10 primary merge ‖ 父05/09 剩余 CPU 消费单元。

最短主线：R4 reader 修正并复审/冻结 → root 对未变 native06 证据的再评估决定 + 下游 graph/phase/counter 判据 → 18 准入（或明确拒绝）→ 合格 coverage → 19。D-Q10 合并在主线上独立并行，不阻塞 native 侧。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只读诊断止于结论：image 校验复用既有脚本仅改51/517→74/18计数常量；client 复用 frozen R2 helper；RCA 脚本明确标注 diagnostic。没有为未被前沿触达的情形扩展 harness（R/test_report_2026-10-23_native_ready.md:14,19,40；P/.../routing_rca.md:99–143） |
| D2 细节反复重试 | CLEAR | 提交前三次失败各自不同且有事前假设：startup 控制缺 node_plan 属错误输入、代理残留属环境、外层 gate 的 `replay_timeout_s` 位置错误；每次不同原因一次收敛。identity FAIL 未盲目重跑，改为 RCA 定位枚举缺陷（R/test_report_2026-10-23_native_ready.md:18,20,28,30） |
| D3 支线问题修补 | CLEAR | client 侧一条内存断言误抄 native05 的50/50车道数，仅移除该无支持断言后重跑既有 helper，单次收敛且属既有验收范围；RCA 的 nested-class/serialized-identity 排查直接服务于被拒判据（client_report.md:75–81；routing_rca.md:87–97） |
| D4 可并行任务停滞 | CLEAR | 真实并行已发生：CPU D-Q10（另一 worktree）与 native06 运行/RCA/client 同时推进，D-Q10 已验收；余下 05/09 CPU 单元受3 slot 与既定"不写14份文档、不重复其工作"的排他限制，属记录的资源冲突而非无依赖者闲置（R/progress.md:38；本次请求约束） |
| D5 超范围镀金 | CLEAR | reader 明确限定为"只改 owner 枚举、不改验收条件"，3文件/300行上限、零 core；D-Q10 保持 isolated、serving/full bundle 仍 false；无对已 PASS 项的重做（reader requirements.md:21–27；contract primary_acceptance.json:14–15） |
| D6 分析空转 | CLEAR | 两份报告都产出决定而非更多笔记：RCA 给出"保留事实足以做 reader-only 修正、无需新采集"，client 给出业务PASS/准入NOT CLAIMED；随后 root 立即授权实现（routing_rca.md:147–151；client_report.md:34；reader requirements.md:15） |
| D7 路线违背 | CLEAR | 冻结 R3 FAIL 未被绕过或改标：未改 checker/raw/验收规则、未用7PASS推断下游、未追溯准入；reader 为版本化 successor 且保留原判据；未提交新 GPU 或扩大预算（R/test_report_2026-10-23_native_ready.md:42；reader requirements.md:15,22–26） |

**文档检查：**
- S1 CLEAR：18 等真实 runtime/reference 准入、19 等已准入 reference+完整 coverage 均为真边；reader 修正不需 GPU、D-Q10 合并不等 native 冻结，两者无假边。
- S2 CLEAR：reader 的复审/复跑/冻结与仓库 ≥50-scenario gate 由 root 拥有（reader requirements.md:27）；"R4 后下游判据由谁评"仍在同一 root 决定范围内，未见无主阻塞。
- S3 CLEAR：RCA 证明所需 owner/child/backend/source 事实已在32/32 rank 保留，故"准确枚举即可通过"可达；追溯准入由 root 显式决定，属决策而非不可达标准。
- S4 CLEAR：3文件/300行、精确集合一致、拒绝缺失/多余/空覆盖可区分完成；client 无跨轮数值阈值且未新增（client_report.md:67）。
- S5 HIT：`../v4-prefill-contract/task_memory/task_2026-10-23_v4_prefill_contract/checkpoint_2026-10-23.md:51` 仍写"不开展 strict77 支线"，而其 requirements.md:31、summary.md:32 已记录 root 以仓库 gate 取代该决定且 77/77 已通过；R/plan.md:25 的"RCA ‖ client 验证"仍写作当前路线，未含 05:19 之后的 reader 授权（该文件正由14份同步 owner 修改）。
- S6 CLEAR：reader 在独立 worktree、3文件上限内可自验；D-Q10 合并、05/09 消费、14份文档写入分属不同 worktree/owner，未见重叠或必须拆票。

**归因：** 无执行模式 HIT；S5 为状态同步滞后，未见其导致错误串行或错误运行。

## 3. 纠正建议

**分支：** B：存在状态同步缺陷，先补文档；不改变已同意的 scope、验收条件或 GPU 预算。

**文档更新（先执行）：**
1. `R/plan.md:25`（及 `R/progress.md` 当前段）：把"RCA ‖ client 业务验证"标为已完成，改为"R4 版本化 reader 实现中（≤3文件/300行、无新GPU）→ root 复审/复跑/冻结 → 未变证据再评估决定"（可直接执行；由在运行的14份同步 owner 或 root 落地）。
2. `P/tickets/18-native-ground-truth-h200.md:213–235` 与 `P/progress.md`/`P/summary.md` 当前段：记录 native06 运行 Succeeded/12.002222GPUh 但 frozen R3 判据 FAILED、owner 枚举缺陷、R4 授权与"通过枚举不等于准入"；保留 native05 UNVERIFIED（可直接执行）。
3. `../v4-prefill-contract/task_memory/task_2026-10-23_v4_prefill_contract/checkpoint_2026-10-23.md:51`：将"不开展 strict77 支线"标注为已被 root 的仓库 gate 取代（requirements.md:31/summary.md:32 已是当前），不改该文件其余历史正文（可直接执行）。
4. `R/summary.md`/`R/review.md` 当前收尾段：补 D-Q10 ACCEPTED_ISOLATED_PREREQUISITE 与 primary merge 未完成、无 merge hash；同时补 reader 授权（可直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. R4 reader 修正（`/data/ycfeng/Frontier/.worktrees/v4-native-routing-reader`）：完成 hash-bound successor、真实 collector FP8 理解 RED（实际形状 MXFP4/RoutedExperts/MoERunner）、32 rank 精确一致与严格负例 → 完成标志：≤3非core文件/300行、仅替换枚举 seam、其余判据逐项保留、只产出诊断 receipt、root 复审复跑后冻结，且不主张准入。
2. 并行：D-Q10 primary merge（root，`feat/v4-prefill-contract` b7a179a）→ 完成标志：记录合并后源/hash 并在合并修订上复跑受影响检查；不主张 TP8/PD/graph/serving coverage。
3. 并行：14份状态同步收尾（现 owner）→ 完成标志：上述1–4项落地，历史与原始失败原文保留。
4. slot 空出后：父05/09 剩余 CPU measured-consumer 单元（D-Q10 family/scope → 实际 per-scope consumer 调用链）→ 完成标志：合格 scope 可达既有实测 consumer，缺 balanced decode/graph/TP/first-tail 仍 typed 拒绝。
5. R4 冻结后：root 对未变 native06 证据的再评估决定 + 下游 graph/phase/padding/counter 判据实际评估 → 完成标志：决定有记录、下游判据按 R4 实评、18 准入或明确拒绝；无新 GPU 除非另行批准。
6. 之后：18 准入 → 合格 workflow/GPU/operator-gap → CPU-overhead → 19 四指标。

**暂缓事项：** 旧 prefill 首 fault GPU kernel、q1 事件/稳态异常、CUPTI/GLM 扩展 → 保持在 `P/future.md` 原证据/owner 下。

## 4. 最值得关注的 3 点

1. native06 是"运行成功、准入失败"：4/4 Succeeded、exit0/restarts0、12.002222GPUh，但 frozen R3 只差 `rank_runtime_graph` 的 owner 枚举；7PASS 不能读作下游已验证，也未获追溯准入。
2. R4 是本窗口最易产生新偏离的点：3文件/300行与"必须拒绝缺失/多余/空枚举"是硬边；若 R4 暴露新的下游失败必须回 root，不得就地放宽、改原记录或改判据，且 reader 通过不等于 18 准入。
3. D-Q10 只是 isolated 前提：94 scope/77 矩阵/451 hash/519 cache 不覆盖 physical TP8、PD、graph、serving；primary merge 未完成且无 merge hash，CPU-overhead 仍最后；native 成功与 isolated 接受都不构成新 GPU 授权。
