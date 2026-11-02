## 1. 规划完成情况

Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | S5 验收完成后检查路线；P0 timeout 保留，转向已分配的单次尾部计时。 |

**汇总：** PASS 8｜BLOCKED/FAILED/SKIPPED 3｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A–H/G2/I1–I4 与已验收的独立输入 | PASS | 原验收和固定 source/input 继续保留，无重做 |
| R3/R4 retained-run 与 metadata-only | PASS | run_metadata_acceptance.json；未扩展为全 trace 或 whole-DP |
| S1/S2 scope CPU/source | PASS | 84a66948，50/50 两次独立检查与冻结 source |
| S3 mirrored package/image/SDK | PASS | 08b38905，原 package 和 18 inputs 保留 |
| Fresh prelaunch gates 与 seal | PASS | 0f2fb6cd；fresh receipts 先于唯一提交 |
| S4 单次 scope GPU 执行 | PASS | exp-1025-021228-212042，Succeeded/exit0/restart0，额度 1/1 用满 |
| S5 受限 runtime acceptance | PASS | results/primary_acceptance.json；R2 findings closed，79 新绑定检查 |
| Metrics source 与两个 exact controls | PASS | 0d0fe648，所有 non-clock 字段和解压 index exact |
| P0 pilot/full-output correction | FAILED | child124/no finalJSON；原 affected 额度使用 1/8，七个未执行 |
| 新 metrics-tail diagnosis | TODO | requirements 的单独分配；timer owner 与 primary 写入范围分开 |
| R6 完整 trace review/synthesis | BLOCKED | R5 仍 24/32，八个 P0 完整输出缺失 |
| R7 完整受限 evidence archive | BLOCKED | 完整 R5/R6 标准未满足 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 0d0fe648 的 source/control checkpoint 至本次 S5 验收｜**当前前沿：** metrics-tail timer 的 source/AST 验证与单次分段诊断

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 79 retained bindings 直接完成 S5 验收；timer 只测当前缺失的 batch 分段 |
| D2 细节反复重试 | CLEAR | 已失败的单次 pilot 没有重跑；GC/metrics 的历史尝试均有事前假设，下一步先测未观测成本 |
| D3 支线问题修补 | CLEAR | 只完成 S5 和记录 P0 失败，无 unrelated 修改 |
| D4 可并行任务停滞 | CLEAR | primary 收口 S5 时，Hooke 同时准备 P0 timer，写入范围不重叠 |
| D5 超范围镀金 | CLEAR | 原 package/source/control 不重做；无 native integration 或新 admission |
| D6 分析空转 | CLEAR | 源码核对定位到 batched_metrics；结果落实为有一次额度的计时工具 |
| D7 路线违背 | CLEAR | GPU 额度保持用满 1/1；60s/30GiB/max2 与所有冻结归因规则保持 |

**文档检查：** S1–S6 当前均无未解决缺陷。原 current 段的 R2/pilot pending 与实际结果冲突已在主 plan/progress/summary 原位同步；原 handoff 已更新，旧 snapshot 标为历史。新 metrics-tail blocker 已分配给独立步骤，未改变验收标准。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 完成 timer 的 source/AST/tiny-fixture evidence 检查；完成标志：去掉 telemetry 后仍为原函数，tiny outputs exact，冻结输入不变。
2. Primary 执行新分配的唯一 P0 timing diagnostic；完成标志：group construction、global sort、union recurrence 的完成/截断时间及 GC overlap 都明确记录，原失败不覆盖。
3. 根据实测原因记录新的 bounded correction、exact-control 和 fresh affected-rank 额度；完成标志：允许范围和预期可核对，执行前写入，最终全部 32 ranks 有完整合格输出后才推进 R5/R6/R7。

## 4. 最值得关注的 3 点

1. S5 仅接受锁定 image/H800/1024-element 普通 pointwise probe 的 P1–P6；native serving ownership、timing、dependency、cost/reference 仍未验收。
2. P0 的完整 index 不代替 final JSON，也不能继承旧 pilot 的 useful-step 数；R5 保持 24/32。
3. 旧 GPU/pilot/control 额度不复用；新诊断只有一次，且不自动授权 correction、full controls 或 fresh affected-rank 执行。
