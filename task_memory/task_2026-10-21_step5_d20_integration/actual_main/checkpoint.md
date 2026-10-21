## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Self-check after actual-main reproduction. |

## 1. 规划完成情况

**汇总：** PASS2｜BLOCKED/FAILED2｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 当前输入和CLI检查 | PASS | 正确P16/D16与v0.28.0输入已生成 |
| 无修改真实入口runner | PASS | 子进程实际执行frontier.main |
| offline/online执行 | FAILED | 两者相同production初始化异常 |
| 完整请求/transfer/metrics验收 | BLOCKED | 依赖parent graph-policy修复 |
| 成功路径assertions与最终提交 | TODO | 修复后重新核实实际artifact schema |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本子任务开始至首次真实入口失败｜**当前前沿：** parent source修复及随后rerun

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只新增满足实际入口验证的最小runner，失败后停止扩展 |
| D2 细节反复重试 | CLEAR | 两种模式各一次，未参数绕过或重复修复 |
| D3 支线问题修补 | CLEAR | 未修改production或其他组件 |
| D4 可并行任务停滞 | CLEAR | parent独立负责fidelity/native；此任务禁止新agent |
| D5 超范围镀金 | CLEAR | 未扩展正式工作负载或native admission |
| D6 分析空转 | CLEAR | 最初文件定位偏多，但随后形成2秒真实失败证据并停止调查 |
| D7 路线违背 | CLEAR | 保持full_decode_only并交回parent source问题 |

**文档检查：** S1–S6均未发现明显缺陷；parent已明确拥有production blocker。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. parent修复policy-to-measurement边界 → 完成标志：真实CLI越过初始化。
2. 本任务rerun两模式 → 完成标志：6144/1024、tail16/cached6128、完整transfer与metrics。
3. 并行：parent原fidelity/native任务保持原owner，不重复运行。

## 4. 最值得关注的3点

1. 当前是确定性production blocker，不能将fixture预训练成功当完整main成功。
2. 不得把CLI改成full来隐藏真实配置错配。
3.8192容量与synthetic3.5ms只用于CPU contract，正式native/accuracy仍未验证。
