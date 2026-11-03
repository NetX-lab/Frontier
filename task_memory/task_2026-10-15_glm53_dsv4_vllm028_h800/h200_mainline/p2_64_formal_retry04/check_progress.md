Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | 第四代提交验收后的自查；确认已应用上轮监控与 launcher 生命周期纠正。 |

## 1. 规划完成情况

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 7｜TODO 12

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 00 基线审计 | PASS | ticket00 记录 done。 |
| 01 运行环境与 native qualification | TODO | 部分 H200 证据完成，完整范围未关闭。 |
| 02 observer/reader | TODO | R5 已验收；完整 observer 条件仍未全闭。 |
| 03 模型/cache/routing inventory | TODO | 源码与局部 native 证据已记录；完整条件未关闭。 |
| 04 GLM family | BLOCKED | D-DSV4-ONLY-GLM-PAUSED。 |
| 05 DSV4 family | TODO | 当前 H200 operator path 已交付，完整范围未关闭。 |
| 06 communication | TODO | P2 输入已合格，完整硬件范围未关闭。 |
| 07 NIXL PD | TODO | P2 所需计价已验证，完整 ticket 未关闭。 |
| 08 workflow | TODO | 已有 P2 路径证据，其余条件保留。 |
| 09 predictor bundle | TODO | 当前 P2 输入合格，完整交付未关闭。 |
| 10 H800 P0/P1 | BLOCKED | 新 H800 资源仍需具体方案与批准。 |
| 11 H200 P2 | TODO | P2-32 正式 PASS；P2-64 新参考待完成。 |
| 12 D-PD | BLOCKED | 完整生命周期/争用证据未关闭。 |
| 13 D-ASYNC | BLOCKED | native 证据依赖仍在。 |
| 14 D-ROUTING | BLOCKED | 完整 native 决策未关闭。 |
| 15 D-DP-PD | BLOCKED | 完整 native 决策未关闭。 |
| 16 GLM case | BLOCKED | GLM 暂停。 |
| 17 DSV4 case | TODO | 当前 P2 绑定已用，完整硬件范围未关闭。 |
| 18 native reference | TODO | 第四提交获批并创建，资源与 launcher 存活验收 PASS；真实运行待完成。 |
| 19 comparison/acceptance | TODO | 完整新参考与每次重复的四指标未取得；旧缺失时钟仍单列。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 3d2964a5 的前次 checkpoint 至本次第四提交验收｜**当前前沿：** 18 已提交 job 的独立监控；19 依赖完整参考，CPU 后处理命令已就绪。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用原包/R5/Frontier replay，只修正包外授权文字和必要的新身份命令，未重跑旧测试。 |
| D2 细节反复重试 | CLEAR | 本窗口恰好一次新提交；监控独立 label，只读精确 job/replica；未重复旧失败 API。 |
| D3 支线问题修补 | CLEAR | 没有开展 GLM、EP-count、H800 或旧时钟深挖。 |
| D4 可并行任务停滞 | CLEAR | 独立监控与 lead 提交验收/后处理准备并行；比较确实依赖 native 输出。 |
| D5 超范围镀金 | CLEAR | 冻结包未变；没有新增功能、模块或配置机制。 |
| D6 分析空转 | CLEAR | 用户决定已落实为真实 job 与可复核提交证据。 |
| D7 路线违背 | CLEAR | D-P2-64-FOURTH 授权先记录后提交；Pi 正常退出后 launcher81539 仍存活；未调用 stop。 |

**文档检查：** S1–S6 均未发现当前范围新缺陷；requirements/current plan/progress/summary/tickets18/19 已同步最新批准和实际提交，历史候选报告保留原时点限制。
**归因：** 无。本轮严格执行上次 D2/D7 的纠正，不改写先前事故。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 18：保持 launcher81539，独立 Pi p64-formal4-monitor-1103f 读取 exp-1103-131952-259161 → 完成标志：真实8节点/64GPU/source/API、三轮正式完成及终态证据；未调度则保留队列和实际原因。
2. 18→19：终态立即保存容器时钟，按需要执行一次0GPU复制并独立核验 → 完成标志：原始产物与本地哈希一致，缺失项明确记录。
3. 19：coverage→R5→每次重复的 all-request/warm 四指标，复用已完成 Frontier replay → 完成标志：全部重复的结果、误差、限制作出明确结论。

**暂缓事项：** GLM、EP-count、新 H800 资源、旧缺时钟深挖和已有 collector-hash 测试债继续留在原 reports/future.md；不扩大本轮。

## 4. 最值得关注的 3 点

1. 第四个 job 已创建，64H200 全部可抢占；提交前仅47可用卡的快照说明可能等待容量，不能据此重复提交。
2. 提交 Pi 正常退出后 launcher81539 仍存活，独立监控已接手；不能再用提交 label 的 stop 修复监控。
3. 新 native 参考尚未完成，P2-64 正式指标和精确用量仍待终态；第五次提交没有授权。
