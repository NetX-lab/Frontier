## 1. 规划完成情况

**Modification History**

| Date | Change |
| --- | --- |
| 2026-10-31 | 根据 Codex 接续记录、当前 tickets 和实际 artifacts 检查路线；识别 S5，执行方随后修正文档。 |

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 2｜TODO 17

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 00 baseline audit | PASS | ticket 00 的三个验收项已记录完成。 |
| 01 runtime qualification | TODO | P0/P2 R4 references 已通过；完整 qualification 未关闭。 |
| 02 observer and gate | TODO | canonical client gate 已实现；完整 observer qualification 未关闭。 |
| 03 inventory | TODO | source/cache contracts 已实现；完整 native 验收未关闭。 |
| 04 GLM family | SKIPPED | D-DSV4-ONLY-GLM-PAUSED。 |
| 05 DSV4 family | TODO | B16–B28 已提供 operator path；旧 A6 仍有三个 attention scopes FAIL。 |
| 06 communication | TODO | B29 A8 全 exact；B30b 的 H200 link/NIXL probes 待完成。 |
| 07 NIXL PD | TODO | H4b/B29 已实现；F3 硬件与 packed payload 验证未完成。 |
| 08 workflow | TODO | B24/B26 已实现并验证；完整验收仍未关闭。 |
| 09 predictor bundle | TODO | 非 dummy operator path 已运行；fresh A6 和完整资格仍待核验。 |
| 10 H800 cases | TODO | 保留 P0 R4 和 diagnostic 结果；当前 H200 主线结束后继续。 |
| 11 H200 cases | TODO | B29 P2-32 数值 PASS，P2-64 第一次 TTFT FAIL。 |
| 12 D-PD | TODO | 已有 lifecycle 实现与证据；完整 contention disposition 未关闭。 |
| 13 D-ASYNC | TODO | decision 43/B26 已提供 run-ahead 语义；F2 比较未完成。 |
| 14 D-ROUTING | TODO | 已接受 balanced routing 和 MATCH；旧 ticket 状态不能代表当前完整验收。 |
| 15 D-DP-PD | TODO | 已有 P2 reference/placement；完整 disposition 未关闭。 |
| 16 GLM case | SKIPPED | D-DSV4-ONLY-GLM-PAUSED。 |
| 17 DSV4 case | TODO | P0/P2 scoped bindings 可复用；完整四个 case 的工作未完成。 |
| 18 native references | TODO | p2_32r/p2_64r R4 均 PASS；完整各硬件 qualification 未完成。 |
| 19 comparison | TODO | 五份 canonical readings 已重读；F1–F4 与 final admission 未完成。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进；发现 S5 文档缺陷。
**检测窗口：** B29 验证完成至 Codex 接续｜**当前前沿：** 06/09 的 B30b，11/19 的 F1–F4 比较

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | A8/gate 复核直接决定 11/19 的现状；不扩展 harness。 |
| D2 细节反复重试 | CLEAR | F1a P2-64 旧 attempt 无终态；一次 fresh replay 保留旧证据，未形成重复调参。 |
| D3 支线问题修补 | CLEAR | 本窗口操作均服务 decision 46 和 Codex 迁移。 |
| D4 可并行任务停滞 | CLEAR | F1a CPU replay 与 B30b preparation 同时推进。 |
| D5 超范围镀金 | CLEAR | 无新增正式输入、门限或模型。 |
| D6 分析空转 | CLEAR | 复核得到逐次 FAIL 和 F4/B28 identity；随即更新入口并执行 probes。 |
| D7 路线违背 | CLEAR | 最新 decision 46 允许 measured CC、批准 ingress A；用户明确指定 direct Codex RJob。 |

**文档检查：**
- S1 CLEAR：最新用户决定允许 F comparisons 与 ingress 并行；当前调度未增加虚假依赖。
- S2 CLEAR：A6 归 05/09，F1/F3 归 06/07，F comparisons 归 11/19；无未归属 blocker。
- S3 CLEAR：保持现有逐次门限；不以 pooled 值或历史资源授权补写 promotion。
- S4 CLEAR：每个 probe、comparison 和 admission 均有可核验输出。
- S5 HIT：plan 的 2026-10-25 入口仍限定 12 小时 H200；summary 写 H200 deferred；09/19 写 comparison 未开始；progress 的旧 handoff 写 ingress deferred。与 requirements item 21/46、B29 commit a69a5213 和 readings 冲突。
- S6 CLEAR：沿用已有 tickets，B30b 仅交付 probes/fits；lead 单独负责 GPU 与主文档。

**归因：** 无执行偏离；S5 是执行入口未同步。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. requirements.md：记录 Codex CLI 原文和 direct RJob 执行决定（可直接执行）。
2. plan.md、spec.md、progress.md、summary.md：把 H200 operator path、B29、F1–F4 和当前 32-GPU non-preemptible runtime 规则写入当前入口；旧路线标为历史（可直接执行）。
3. tickets 05–11、17–19：记录已实现与未验收部分；逐次 P2-64 FAIL、A6 和历史 admission 均保留（可直接执行）。

**需停止的偏离行为：无**

**接下去的路线：**
1. 06/09 B30b：核验 dry run、旧 plan preservation 和 cost → 完成标志：两项已授权 H200 probes 有业务输出和终态证据。
2. 并行：11/19 F1a P2-64 replay ‖ B30b 的 two-node/one-GPU jobs；每项有 fresh receipt 和 canonical reading。
3. 11/19 F1b/F2/F3：从测量拟合，运行 diagnostic variants，与固定正式输入比较；所有四项及每次 reference 都报告。
4. 05/09/19：检查最新 A6、hardware/payload/promotion prerequisites 后再讨论 formal acceptance；numeric PASS 不自动完成 ticket。

## 4. 最值得关注的 3 点

1. P2-64 第一次 TTFT 779.888727400 ms 对 684.313826604 ms，误差 12.254940665%，正式 reading 仍 FAIL。
2. F4 与 B28 的五个 metric artifacts 在两种规模上逐字节相同；其与 B29 的差异来自 delay 配置，不能归因于 planner。
3. measured CC 仍是正式输入；diagnostic 替换正式输入和 B30a 默认 ASTRA 行为合并仍需 YC 新决定。
