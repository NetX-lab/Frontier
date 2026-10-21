# Scope-transition self-check

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Check the measured vertical slice and correct serial zero-reservation guard discovery. |

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 1｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| R2 当前 P1 regression/disposition | PASS | cb09a27、636 tests、独立复审通过，分支保留于 b6ba7e5 |
| D20 A 接口与生命周期定位 | PASS | 实际源码和共享6cc09dc已读；两个共享接口扩展已获批准 |
| computed-context 切片 | PASS | f3781ef，实际 offline/online 生命周期，70 tests |
| D20 B,C measured PD 实现 | TODO | 已划定独占代码与明确拒绝边界 |
| D20 D integration tests | TODO | 使用用户指定的真实 event/scheduler seam |
| D20 E 实际 measured main | BLOCKED | 计算 owner 的 provider/bundle 集成尚待提供 |

## 2. 偏离检测

**结论：** 已偏离：命中 D2。
**检测窗口：** 前次 scope checkpoint / f3781ef 至今｜**当前前沿：** D20 B/C measured 实际生命周期

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | R2 当前回归闭环后停止；不扩展 staging |
| D2 细节反复重试 | HIT | zero-query READ 在多层 positive-only guard 上连续阻塞；没有预先查全 allocation 调用链 |
| D3 支线问题修补 | CLEAR | 当前切换由用户明确要求 |
| D4 可并行任务停滞 | CLEAR | Request slice 并行完成84 tests；主线实现 event/predictor/allocator 接线 |
| D5 超范围镀金 | CLEAR | 只做现有 PD owners，排除第二引擎和 observer |
| D6 分析空转 | CLEAR | 当前定位直接服务 measured-mode guard 和 first-D shape 实现 |
| D7 路线违背 | CLEAR | D20 已明确解除 CPU 实现等待，native 验收仍保留 |

**文档检查：** S5：requirements 的独占范围尚未记入随后批准的 transfer predictor/info 和 grouped allocator 最小接口扩展。S1–S4、S6 无明显缺陷。
**归因：** D2 执行偏离；S5 是随后批准事项的记录滞后，不是重复试错的原因。

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. requirements 的 explicit decisions：记入两项已批准的最小共享接口扩展（可直接执行；已确认，无新范围申请）。

**需停止的偏离行为：**
- D2：停止每撞到一层 guard 就重跑；先核对 reserve→can_allocate→allocation plan→allocate 的完整调用链，再验证同一真实场景。

**接下去的路线：**
1. B/C：一次完成已查全的 zero-query reservation 合同 → 完成标志：原 measured 正例穿过 READ/ready/first-D 并释放全部 ownership。
2. 并行：PD CPU fixture 集成 ‖ computation owner 提供共享 profile/provider 接口。
3. D/E：提交 affected regression；provider 就绪后执行小型实际 main → 完成标志：命令、输出、token/ownership 平衡可核验。

## 4. 最值得关注的 3 点

1. zero-query READ 必须保留真实 prefix frontier；不能通过虚构1-token计算绕过 allocator guard。
2. Request 已复用现有 recompute cursor；25-token tail 只产生1个输出，独立84 tests通过。
3. measured provider/bundle 和 T2/native 输入仍待 computation owner/primary；CPU fixture 不授予 native qualification。
