## 1. 规划完成情况

### Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Diagnose and correct the non-baseline binary projection before canonical unit rerun. |

**汇总：** PASS 6｜BLOCKED/FAILED/SKIPPED 1｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Source/input preflight | PASS | frozen46fe/treebbb/3dac保持clean |
| supplement101 | PASS | 101PASS，same79IDs |
| metadata | PASS | 197PASS，planned1ID替换单列 |
| catalogue | PASS | 53PASS |
| strict77/audit | PASS | 77/77，完整hash/cache证据 |
| First canonical unit admission | FAILED | raw8036/279/32/267，但额外binary使14旧SKIP的执行环境改变 |
| Corrected canonical rerun | TODO | 原namespace已恢复，同1800s suite重新运行 |
| Existing governance52 | PASS | fresh52PASS，same52IDs，无binary投影 |
| Finalguards/archive/commit | TODO | 等待canonical结果 |

## 2. 偏离检测

**结论：** 已偏离：命中D1；已纠正，不重复独立有效结果。
**检测窗口：** strict checkpoint至今｜**当前前沿：** canonical unit比较及最终guards

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | HIT | 不必要的binary投影扩展optional执行环境，推迟canonical unit；baseline XML本可确认缺binary |
| D2 细节反复重试 | CLEAR | 首次env差异后检查旧skip原因，单一原因已证实；仅一次针对性环境纠正 |
| D3 支线问题修补 | CLEAR | 不修复历史279/32或生产代码；纠正只服务canonical门禁 |
| D4 可并行任务停滞 | CLEAR | canonical unit与gov52并行，无agents/poll |
| D5 超范围镀金 | CLEAR | binary可用性实际结果保留但不qualify，不新增suite |
| D6 分析空转 | CLEAR | exactIDs和old skipmessages确定原因，立即恢复原recipe |
| D7 路线违背 | CLEAR | 同一binary投影计入D1不重复计数；不改source/test/skip/gates |

**文档检查：** S5：初始progress/short把“目前存在的binary”误当作“79时可用”，已更正。S3：planned parser-test replacement与绝对旧ID-preservation冲突仍单列；S1/S2/S4/S6无明显缺陷。
**归因：** D1执行偏离；user要求复用既有namespace，不应追加binary可用性。

## 3. 纠正建议

**分支：** B：先纠正自身环境记录，再完成原门禁；不改已批准source/suite。

**文档更新（先执行）：**
1. progress/short/fullunit：更正历史假设，保留实际SKIP-to-PASS及所有first-run证据（可直接执行，已完成）。
2. baseline_environment/reconciliation_preflight.json：绑定原namespace、无binary、新输出名及不变gate（可直接执行，已完成）。
3. planned parser-ID replacement：rawpredicate继续标记loss，admission由primary处理（必要时需用户确认，无test alias授权）。

**需停止的偏离行为：**
- D1：停止为optional native tests增加基线没有的可用性；恢复原namespace，不改任何test skip或gate。

**接下去的路线：**
1. canonical unit → 完成标志：原namespace、1800s、same frozen source的真实XML及全部exactIDs。
2. final比较 → 完成标志：source additions/planned replacement/env observations分别具名；gov5失误由fresh52证据核对。
3. finalguards/archive/scopedcommit → 完成标志：真实状态及限制完整，oldraw不删除。

## 4. 最值得关注的 3 点

1. 8036/279/32/267是实际first-run结果，但不是历史环境等价的canonical gate。
2. 无binary原namespace已恢复，无exclusion/skip/生产改动；strict/short不重复。
3. oldSKIPs不转换为baselinePASS，optional结果不用作native/G3/T2 qualification。
