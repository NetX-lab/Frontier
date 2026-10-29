**Modification History**

| Date | Change |
| --- | --- |
| 2026-10-24 UTC / 2026-10-25 HKT | 第三次合并预览问题后停止预览排查，回到既有回归与证据归档前沿。 |

## 1. 规划完成情况

**汇总：** PASS 10｜BLOCKED/FAILED/SKIPPED 3｜TODO 5

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 原批准、owner与scope | PASS | 116路径、18 overlays与原预算不变。 |
| Native19与T2 engine limits源码/CPU | PASS | 既有7168/8192/128绑定未变。 |
| Derived image/source records | PASS | 既有实际导出和发布绑定保留。 |
| CORE、registry、PRICING已验收源码 | PASS | 当前生产source9566/global9580。 |
| Ledger完整输出与caller controls | PASS | count0/3、66/6与30原始证据保留。 |
| 最终146候选audit | PASS | 126 bindings，source9682/global9696。 |
| 最终Standards/Spec | PASS | 两轴No findings。 |
| Required P6128/D16/BF16 shapes | PASS | 992 events/80 lanes；6/6耗尽。 |
| Fresh strict77与原auditor | PASS | 77/77；每侧451 artifacts/519 cache names。 |
| 101/197/53/52 supplements | PASS | 独立真实IDs/status与环境一致。 |
| Final freeze与生产整合 | TODO | 冻结完成；尚待canonical unit，未合入。 |
| Canonical full-unit IDs/status | TODO | 原supervisor73779运行，最后观察约92%。 |
| Snapshot-copy有限诊断 | TODO | 原owner，最多一次control，最终consumed count待回执。 |
| Initialization有限诊断 | TODO | 原owner只读source与小旧回执。 |
| 当前pricing+emission+flush | TODO | 恢复真实生产写入路径，尚未测量。 |
| 完整6144/1024 CLI | FAILED | 最近main08两模式2400s timeout。 |
| Worker/cloud条件 | BLOCKED | quota确认；环境、吞吐、全workload内存待核验。 |
| 最终T2/package/native/G3/E2E | BLOCKED | 等待完整workflow与封包。 |

## 2. 偏离检测

**结论：** 已偏离：命中D2。
**检测窗口：** 回归checkpoint后至合并预览解码错误｜**当前前沿：** canonical unit ‖ copy/init；mandatory证据归档及当前计时准备。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 预览重复问题归入D2；原comparators与auditor未扩展。 |
| D2 细节反复重试 | HIT | Git2.34.1无merge-tree --write-tree；旧三参数输出过大，再读取因binary patch UTF8解码失败；准入尚未需要这个预览。 |
| D3 支线问题修补 | CLEAR | 无其他代码或环境修复。 |
| D4 可并行任务停滞 | CLEAR | unit与copy/init继续；当前计时和worker recipe读取并行。 |
| D5 超范围镀金 | CLEAR | 无新增source、flags、schema或预算。 |
| D6 分析空转 | CLEAR | 仅恢复原生产调用路径；预览细节归入D2。 |
| D7 路线违背 | CLEAR | 无提前合并、新shape、pricing或worker操作。 |

**文档检查：** S1–S6未发现新的明显缺陷；既有S5标题已澄清。
**归因：** D2执行偏离。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。

**需停止的偏离行为：**
- D2：停止merge-tree预览及其二进制解析；mandatory gates后直接执行授权合并，按真实冲突逐hunk保留双方意图。

**接下去的路线：**
1. 原canonical unit完成后执行已准备的原判据wrapper → 完成标志：全部8620 IDs/status及FAIL/ERROR/SKIP集合一致，raw exit1保留。
2. 归档strict/unit与四个补充套件原始证据并提交记录 → 完成标志：source/version/hash bindings齐备。
3. 并行copy/init → 完成标志：原额度内ALLOC/NO_ALLOC/精确BLOCKER。
4. mandatory gates后生产整合及父端绑定 → 完成标志：四文件与146一致，126 bindings及预算通过。

## 4. 最值得关注的3点

1. 预览不是当前准入前置条件；已停止无收益的工具细节排查。
2. 原unit仍运行，候选仍UNADMITTED；strict和补充套件不重跑。
3. 当前计时、worker/cloud与完整CLI仍是后续验收，原运行上限不增。
