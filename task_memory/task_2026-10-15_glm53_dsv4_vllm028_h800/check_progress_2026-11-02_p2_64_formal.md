Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | R5 已验收；generation02 抢占与104文件复制已核验；按现有第三次额度恢复，停止引用旧 Running/待审批状态。 |
| 2026-11-02 | RCA 与独立 CPU 复现完成；保留时钟缺口，停止可选控制实验的反复扩展，下一步为具体恢复方案审批。 |
| 2026-11-02 | 以 Failed 终态和 74/74 文件复制校验替换排队状态；RCA、CPU 复现与用量核对继续。 |
| 2026-11-02 | harness 最终分支整合已验证；记录并停止 D1 额外测试，修正 ticket 18 的 S5 旧路线。 |
| 2026-11-02 | 核对 D-P2-64-FORMAL-B 当前阶段；确认 S5 旧状态并修正当前入口，保留历史验收。 |

## 1. 规划完成情况

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 7｜TODO 12

表中是完整 ticket 状态；本阶段已完成 native 启动修复、R5 验收、generation02 终态与复制，并未完成 ticket18/19 的正式 P2-64 验收。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 00 基线审计 | PASS | ticket00 recorded done。 |
| 01 运行环境与 native qualification | TODO | 已有 H200/source 证据，完整跨配置条件未关闭。 |
| 02 observer/reader | TODO | R5 在5bd68e9e验收；旧完整 observer 条件仍未全闭。 |
| 03 模型/cache/routing inventory | TODO | 已有结构与 native 局部证据，ticket未整体关闭。 |
| 04 GLM family | BLOCKED | D-DSV4-ONLY-GLM-PAUSED。 |
| 05 DSV4 family | TODO | H200 operator path 已交付；完整最终范围未关闭。 |
| 06 communication | TODO | P2所需实测输入已合格，完整硬件范围未关闭。 |
| 07 NIXL PD | TODO | 现有线长计价修复已验证，完整ticket未关闭。 |
| 08 workflow | TODO | P2现有路径已验证；保留其余未闭条件。 |
| 09 predictor bundle | TODO | P2所需输入已合格，完整交付未关闭。 |
| 10 H800 P0/P1 | BLOCKED | 当前路线要求独立资源方案；本轮未获新增H800资源。 |
| 11 H200 P2 | TODO | P2-32正式PASS；新P2-64参考不完整。 |
| 12 D-PD | BLOCKED | ticket仍记录未闭native生命周期/争用证据；非本轮前沿。 |
| 13 D-ASYNC | BLOCKED | ticket仍记录native证据依赖；非本轮前沿。 |
| 14 D-ROUTING | BLOCKED | ticket完整决策未关闭；本轮不扩大routing工作。 |
| 15 D-DP-PD | BLOCKED | 完整native决策未关闭；保留其证据边界。 |
| 16 GLM case | BLOCKED | GLM暂停。 |
| 17 DSV4 case | TODO | P2绑定已用；完整硬件范围未关闭。 |
| 18 native reference | TODO | generation02终态/复制通过；0完整repeat，第三代准备中。 |
| 19 comparison/acceptance | TODO | P2-32PASS及晋级receipt已签；P2-64新参考与完整用量未闭。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 前次启动故障收尾checkpoint至本次generation02终态验收｜**当前前沿：** 18第三代准备/运行；19精确容器时钟。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 39f9d20f和5bd68e9e解决实际启动/reader依赖；153+25测试及PASS_EXACT已止步，不重复。 |
| D2 细节反复重试 | CLEAR | 第一代端口机制已修复，第二代两API头healthy后被平台抢占；不同已证实原因，当前仅用剩余第三次额度。 |
| D3 支线问题修补 | CLEAR | GLM/EP-count/collector-hash旧债未重开。 |
| D4 可并行任务停滞 | CLEAR | 第三代package与两exact-job只读时钟/事件查询并行，lead独立验收并维护记录。 |
| D5 超范围镀金 | CLEAR | 第三代仅现有CASES新增身份条目，源/工作负载/资源均不变。 |
| D6 分析空转 | CLEAR | 已落地终态和104文件验收，并启动新generation准备。 |
| D7 路线违背 | CLEAR | requirements:263明确允许抢占后的新generation，累计上限3jobs；平台动作均走/ds4-exec。 |

**文档检查：** S1/S2/S3/S4/S6未发现本阶段新缺陷。S5：plan/progress/summary/tickets18/19当前段仍有Running或等待上一代状态；旧checkpoint还写修复待批准，和D-P2-64-STARTUP-RECOVERY、5bd68e9e及Stopped终态冲突。
**归因：** 无执行偏离；S5是跨session状态滞后，已按实际产物同步。完整ticket的历史局部状态不当作新的前置门槛。

## 3. 纠正建议

**分支：** B：spec/tickets存在缺陷，先更新文档再继续。

**文档更新（先执行）：**
1. plan/progress/summary/tickets18/19当前入口：generation02改为已抢占且复制通过，后续依赖为第三代完整参考（可直接执行，已应用）。
2. requirements续接记录及本checkpoint：引用既有第三次额度和debug批准；第四次需新决定，不把executor单轮禁止重提当作用户未授权（可直接执行，已应用）。
3. recovery test report：记录真实client/proxy/repetition产物，纠正空replay.log推断（可直接执行，已应用）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 18：审查第三代package、独立seal/check-only并提交 → 完成标志：仅身份/路径变化，源和工作负载保持冻结且预检通过。
2. 18：通过/ds4-exec提交第三个64H200全preemptible job并监控 → 完成标志：三repeat完成及terminal/copy证据；不自动创建第四job。
3. 并行：19的缺失时钟一次只读跟进；不足则精确总量继续null，不阻塞已授权运行。
4. 18/19：完整参考的pair覆盖soft gate → R5 → all-request/warm → 每repeat四项≤10%及reference充分性结论。

**暂缓事项：** GLM、H800新资源、EP-count和旧collector-hash测试债继续保留于future.md及既有报告。

## 4. 最值得关注的 3 点

1. 第二代越过启动故障，随后8replicas被抢占；只完成首次warmup的一部分，不能产生新校准PASS。
2. 三个GPUjobs额度已用两个；第三代沿已批准配置恢复，第四次必须另作决定。
3. 104文件已独立核验；两次正式尝试均缺容器结束时钟，已知subtotal不能当作完整GPU-hours。
