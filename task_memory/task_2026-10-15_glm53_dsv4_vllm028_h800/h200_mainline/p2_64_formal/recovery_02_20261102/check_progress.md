## 1. 规划完成情况

| Date | Change |
| --- | --- |
| 2026-11-03 | Modification History: R5 验收里程碑自查；同步 Running 放置证据，后续聚焦新参考和计量。 |

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 1｜TODO 3

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| P2-32 与晋级 receipt | PASS | 1a7950ed；formal PASS 保留 |
| 第一代 P2-64 | FAILED | b0ceff53；启动失败，无请求结果 |
| 第一代 RCA 与复制 | PASS | 74/74 文件；CPU 端口分叉复现 |
| D-P2-64-STARTUP-RECOVERY | PASS | 用户批准方案 B 及相关 debug |
| ticket18 native/harness/package | PASS | a06a72a4/f33306d4/39f9d20f；CPU、image、seal/check-only 通过 |
| ticket19 R5 reader code/control | PASS | r5_lead_acceptance.json；153+25 tests、实际 graph-entry 和旧参考 PASS_EXACT |
| ticket18 第二代执行 | TODO | exp-1102-233941-576872；八个独立节点 Running，尚无完整参考 |
| ticket19 新参考比较 | TODO | 等待 terminal/copy 后 coverage、R5、逐次及 pooled readings |
| ticket19 GPU 计量 | TODO | 第一代仍缺7个结束时间；第二代尚未终态 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 第二代提交自查（39f9d20f 后）至 R5 验收｜**当前前沿：** ticket18 现有运行，ticket19 新参考处理与计量

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | R5 是新 source admission 的必要修复；实际入口和旧参考验证已完成，停止扩展与重复测试 |
| D2 细节反复重试 | CLEAR | 先复现 exec namespace 失配，再修正真实调用链；r5_initial_call_chain_failure.json 与最终验收记录闭环 |
| D3 支线问题修补 | CLEAR | 14 个 unchanged collector-hash tests 单独记入报告，不展开修复 |
| D4 可并行任务停滞 | CLEAR | R5 验收与现有 GPU job 放置/运行并行；比较确需 native 输出 |
| D5 超范围镀金 | CLEAR | 仅接受指定 startup source；没有新模型、kernel、routing 或指标规则 |
| D6 分析空转 | CLEAR | R5 从失败转为已验收；三文件 hashes 与验收绑定一致 |
| D7 路线违背 | CLEAR | 现有第二代只提交一次，旧 generation 与已完成 CPU replay 保留 |

**文档检查：** S5：plan/progress/tickets18/19/summary 的当前状态仍称 R5 in-progress，与最终 PASS 证据矛盾；本轮已同步。S1/S2/S3/S4/S6 未发现新缺陷。
**归因：** 无

## 3. 纠正建议

**分支：** B：spec/tickets 存在缺陷，先更新文档再继续

**文档更新（先执行）：**
1. plan/progress/tickets18/19/summary/recovery report：记录 R5 code/control PASS、八节点 Running 和仍待核验的 native 结果（可直接执行，已完成）。

**需停止的偏离行为：** 无

**接下去的路线：**
1. 提交已验收 R5/tests/compare 与必要记录；Pi 继续现有 job，核验硬件/source/native 启动。
2. terminal receipt → hash-identical copy → pair coverage soft gate → R5 → unchanged all-request gate/warm reading；完成标志为三次重复均有明确 validity/readiness 和四指标结果。
3. 更新第二代 actual GPU-hours 与 ticket19；第一代缺时钟继续明确 exact null，不重复已无新增证据的旧时钟查询。

**暂缓事项：** 14 个既有 collector-hash tests → test_report_2026-11-02_p2_64_recovery.md 已记明，后续独立处理。

## 4. 最值得关注的 3 点

1. 第二代已在八个独立节点 Running；需要 actual native 输出才能证明修复成功。
2. R5 code/control 已验收，应转到新参考处理；旧 control 的 PASS 不能代替新运行结论。
3. 第一代 exact GPU-hours 仍缺7个 finish clocks；不能将估算并入已完成精确账本。
