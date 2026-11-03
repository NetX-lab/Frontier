Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | 清理完成后的主线恢复自查；沿用第四作业和已验收后处理，配额阻塞尚未解除。 |
| 2026-11-03 | 第四代提交验收后的自查；确认已应用上轮监控与 launcher 生命周期纠正。 |

## 1. 规划完成情况

**汇总：** PASS 1｜BLOCKED/FAILED/SKIPPED 9｜TODO 10

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 00 基线审计 | PASS | ticket00 记录 done。 |
| 01 运行环境与 native qualification | TODO | 部分 H200 证据完成，完整范围未关闭。 |
| 02 observer/reader | TODO | R5 已验收；完整 observer 条件未全闭。 |
| 03 模型/cache/routing inventory | TODO | 源码与局部 native 证据已记录，完整条件未关闭。 |
| 04 GLM family | BLOCKED | D-DSV4-ONLY-GLM-PAUSED。 |
| 05 DSV4 family | TODO | H200 operator path 已交付，完整范围未关闭。 |
| 06 communication | TODO | P2 输入已合格，完整硬件范围未关闭。 |
| 07 NIXL PD | TODO | P2 所需计价已验证，完整 ticket 未关闭。 |
| 08 workflow | TODO | 已有 P2 路径证据，其余条件保留。 |
| 09 predictor bundle | TODO | 当前 P2 输入合格，完整交付未关闭。 |
| 10 H800 P0/P1 | BLOCKED | 新 H800 资源仍需具体方案与批准。 |
| 11 H200 P2 | TODO | P2-32 正式 PASS；P2-64 新参考未完成。 |
| 12 D-PD | BLOCKED | 完整生命周期/争用证据未关闭。 |
| 13 D-ASYNC | BLOCKED | native 证据依赖仍在。 |
| 14 D-ROUTING | BLOCKED | 完整 native 决策未关闭。 |
| 15 D-DP-PD | BLOCKED | 完整 native 决策未关闭。 |
| 16 GLM case | BLOCKED | GLM 暂停。 |
| 17 DSV4 case | TODO | 当前 P2 绑定已用，完整硬件范围未关闭。 |
| 18 native reference | BLOCKED | 第四作业 Pending、0 replicas；平台报告 Insufficient GPU quota。 |
| 19 comparison/acceptance | BLOCKED | 缺完整第四代参考；R5 和 Frontier replay 已完成，可在产物到齐后复用。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 清理完成的 checkpoint/提交 `0eca0e38` 至本次主线恢复｜**当前前沿：** ticket18 的已提交第四作业；ticket19 后处理有真实产物依赖。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 复用 poll_once.py、冻结包、R5 和 Frontier replay；未重跑已验收测试。 |
| D2 细节反复重试 | CLEAR | 有限只读复查，无新提交、参数变更或失败 API 重试。配额等待不是修复尝试。 |
| D3 支线问题修补 | CLEAR | 用户要求的清理已结束；当前执行只服务 ticket18/19。 |
| D4 可并行任务停滞 | CLEAR | Pi 状态查询与 lead 记录/验收并行；四指标依赖尚未产生的 native 输出。 |
| D5 超范围镀金 | CLEAR | 没有实现、模型、数据格式或范围扩展。 |
| D6 分析空转 | CLEAR | 已落实上轮停止清理调查的纠正；本轮只取得新状态并保留后处理入口。 |
| D7 路线违背 | CLEAR | 同一已授权 job/UID，独立 monitor label，无 stop、信号或第五次提交。 |

**文档检查：** S1–S6 均未发现当前路线的新缺陷。最新明确决定与实际作业一致；旧报告保留历史时点。状态时间戳随本轮证据更新，不改变 scope 或验收条件。
**归因：** 无。清理阶段的 D6 和第三作业的 D7 均保留历史记录，本轮没有继续这些行为。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 18：保持 launcher81539，按需通过独立 Pi 读取 exp-1103-131952-259161；Pending 时有限复查后保留排队，避免反复查询代替产物交付。完成标志：8 节点、64 H200、source/API、三轮正式请求及终态证据。
2. 18→19：Running 时及时保存 start clocks；终态首先保存 end clocks/terminal receipt，再按需复制并独立核验。完成标志：原始产物与本地哈希一致，缺失项和精确用量限制明确。
3. 19：pair coverage soft gate → R5 → 每次重复的 all-request/warm 四指标，复用已完成 Frontier replay。完成标志：三轮四指标、误差、判定及限制完整记录。

**暂缓事项：** GLM、EP-count、新 H800 资源、旧缺时钟深挖及 collector-hash 测试债，继续保留原 reports/future.md；清理已完成。

## 4. 最值得关注的 3 点

1. 当前阻塞是平台 GPU 配额；第四作业仍排队，暂无 worker 业务结果。
2. launcher81539 保持存活；监控在独立 label 中有限执行，不能对提交 label 调用 ds4 --stop。
3. 后处理已具备脚本与 Frontier 结果，等待完整 native 参考；P2-64 和整体 calibration 未完成，没有第五次提交授权。
