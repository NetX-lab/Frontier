## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Checked the failed prefill control and corrected the route to one separately allocated native-stop observation. |

## 1. 规划完成情况

**汇总：** PASS 6｜BLOCKED/FAILED/SKIPPED 5｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Scope限定验收 | PASS | 原GPU1/1已验收，不重跑。 |
| Interval owner/source单位交付 | PASS | owner_evidence_gate：17files/90checks。 |
| Primary独立unit | PASS | 四fixtures/36checks。 |
| Synthetic pair | PASS | exact240groups/24checks，本次10.34%耗时下降。 |
| 新decode complete control | PASS | 37checks/142567rows/cleanEOF。 |
| 两项source review | PASS | source_reviews_binding：两No findings，仅source范围。 |
| 新prefill complete control | FAILED | SIGSEGV/-11，无finalJSON；controls2/2spent。 |
| 原native/segment诊断及P0pilots | FAILED | 已用额度，原CENSORED/ABORTED/timeout保留。 |
| Interval source/control最终验收 | BLOCKED | 缺完整prefill control，source reviews不能代替。 |
| 新affected ranks | BLOCKED | allowance0，需source/control最终验收。 |
| R6/R7完整验收 | BLOCKED | R5仍24/32，retained24准备已完成。 |
| 新DP1 SIGSEGV native-stop观测 | TODO | prefill_fault04已先分配；source/static PASS，review/唯一执行待完成。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** unit/scale checkpoint至prefill失败归档｜**当前前沿：** DP1 pre-metrics native fault的唯一观测。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 新工具仅为当前SIGSEGV取fault PC/native stack；未扩展metrics工具。 |
| D2 细节反复重试 | CLEAR | 两controls各一次；新allocation有不同DP1/SIGSEGV问题，旧P0GDB不重跑。 |
| D3 支线问题修补 | CLEAR | native失败直接阻塞complete-control admission。 |
| D4 可并行任务停滞 | CLEAR | source双审与controls并行完成；affected/fullR6有真实缺输出依赖。 |
| D5 超范围镀金 | CLEAR | frozen runtime/input与原来源未编辑，新signal查询在明确额度内。 |
| D6 分析空转 | CLEAR | 失败boundary/archive已确定，采取一个具体native-stop观测。 |
| D7 路线违背 | CLEAR | 60s/30GiB/max2与0newcontrol/affected/GPU保持；旧实验spent。 |

**文档检查：** S5当前handoff旧unit/scale/control pending说法需要本次结果更新；其余S1–S4/S6未见明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** B：先同步当前失败/新native观测状态，再继续。

**文档更新（先执行）：**
1. progress当前handoff/summary/plan：source review通过与control最终验收BLOCKED分别记录，新native allocation与已用额度清楚绑定（可直接执行）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 新native source只读review → 完成标志：78/90GDB、70/100controller，无实际blocking finding，command/pins冻结。
2. 唯一DP1native观测 → 完成标志：原runtime/cgroup/terminal/log与native marker或CENSORED事实归档，不声称control/curePASS。
3. 按native观测取得的证据选择后续修复；完整control gap关闭之前，不执行affected或R6/R7。

## 4. 最值得关注的 3 点

1. Prefill fault在metrics之前；不能据此推断新的union代码出错或修复完成。
2. Source双审No findings与fullcontrol失败同时成立；R5仍24/32。
3. 所有旧controls/diagnostics均spent，新增只限一个明确DP1native观测。
