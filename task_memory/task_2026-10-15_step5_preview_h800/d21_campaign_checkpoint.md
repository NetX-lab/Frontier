# D21 campaign checkpoint — 2026-10-21

**Modification History**

| Date | Change |
| --- | --- |
| 2026-10-21 | Check D21 documentation delivery against final requirements and correct obsolete campaign edges without expanding runtime work. |
| 2026-10-21 | Parent checks all30 tickets after measured-code integration and observed P0 runtime replay. |

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 18｜TODO 8

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 01 基线复用 | PASS | plan§4记为DONE |
| 02 原生版本锁定 | PASS | 原生image/source已冻结 |
| 03 模型清单 | PASS | plan§4记为DONE |
| 04 G1 kernels/state | PASS | 全部10项已记录完成 |
| 05 早期物理测量 | TODO | 局部数据不能替代完整准入 |
| 06 单角色原生资格 | TODO | simulation_scope完成；完整历史census保留部分状态 |
| 07 observer | BLOCKED | 需要消费方对应的原生验证；准备已完成 |
| 08 G3内容/生命周期 | TODO | case-aligned实现已有owner |
| 09 分track原生资格 | TODO | 四条basic均PARTIAL；P0双lane FULL已观察 |
| 10 compute profiles | TODO | D22小样本严格通过；P0生产者实现中 |
| 11 routing统计 | BLOCKED | 所需原生/observer证据未齐 |
| 12 通信测量 | TODO | 已有诊断数据；正式准入未完成 |
| 13 NIXL争用 | BLOCKED | 08/12未完成 |
| 14 metrics normalization | BLOCKED | CPU实现完成；原生准入未齐 |
| 15 混合模型完整路径 | TODO | representation/structural完成；整票未验收 |
| 16 measured compute | BLOCKED | CPU代码已集成；真实profile/native验收未齐 |
| 17 measured PD | BLOCKED | CPU路径已集成；08/15整体验收未齐 |
| 18 容量/取消 | BLOCKED | 正式17未完成 |
| 19 graph/host | BLOCKED | 07/17未完成 |
| 20 placement | BLOCKED | structural完成；正式08/17未齐 |
| 21 routing/通信预测 | BLOCKED | 11/12/13/17未齐 |
| 22 bundle/cost replay | BLOCKED | CPU整链执行中；正式profiles及依赖未齐 |
| 23 workload/runtime冻结 | TODO | 统一workload完成；各track runtime尚未全冻结 |
| 24 单独H800诊断票 | SKIPPED | D12并入28 |
| 25 单独fit-freeze票 | SKIPPED | D12并入28 |
| 26 matrix/holdouts | SKIPPED | D12取消 |
| 27 单独H200矩阵冻结 | SKIPPED | D12由23替代 |
| 28 四指标校准 | BLOCKED | CPU comparator完成；真实clean输入未齐 |
| 29 同HEAD回归 | BLOCKED | 最终28未完成；已有阶段回归 |
| 30 支持验收 | BLOCKED | 05/29及全部正式证据未齐 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 上次D21-DOCS checkpoint至bc8ae27及P0 generation03判定；实际CLI问题另追溯至首次运行｜**当前前沿：** D20实际请求完成、08 case-aligned G3、10 P0 profiles；09 P0逐lane replay观测本轮完成

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | idle-DP是完整CLI实际缺口；logger是09实际缺口，已交付并停止重复basic验证 |
| D2 细节反复重试 | CLEAR | actual_main/progress在各次运行前记录缺失identity、6145物理token及均匀expert周期假设；结果逐步收窄到不同coverage缺口 |
| D3 支线问题修补 | CLEAR | 新修正均对应16/22或09的已执行失败 |
| D4 可并行任务停滞 | CLEAR | 3个实现槽位分别用于实际CLI、case-aligned G3和P0 producer；独立工作未等待正式30%结果 |
| D5 超范围镀金 | CLEAR | 复用原predictor/allocator/comparator；未恢复retired矩阵 |
| D6 分析空转 | CLEAR | bc8ae27完成代码集成；generation03新增实际FULL证据；后续报告只记录本轮结果 |
| D7 路线违背 | CLEAR | D20允许CPU实现，D21允许独立basic；未将其当作正式10%或64解锁证据 |

**文档检查：** S1–S6均未发现新的明显缺陷。plan§4–6区分CPU里程碑和整票验收，08/10/22已有明确owner；旧硬件规则与retired票已标为历史。运行状态同步不改变spec或acceptance。
**归因：** 无。

## 3. 纠正建议

**分支：** A：当前spec/tickets合理，无需改变范围或验收。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 16/22实际CLI：offline/online完整6144/1024请求、D tail、READ/release和最终metrics通过；合成fixture仍不提供accuracy证据。
2. 并行08与10：完成case-aligned独立reference/content/lifetime证据，以及硬件匹配的P0 profiles。P0 replay已观察，后续09只补尚缺的G4 accounting。
3. prerequisites齐备后采集32卡三轮clean全128请求，运行28原四指标10% gate；对应32四项每轮<=30%才释放64正式校准。最终29在完成的干净checkout验证，不为清除文档dirty标记反复运行当前矩阵。

## 4. 最值得关注的 3 点

1. P0双lane FULL replay观测已完成；G3和完整G4仍未完成，避免为已解决的logger问题再次basic重跑。
2. 323项合并检查及297个不涉及schema的artifact对一致，尚未证明完整实际CLI和正式精度；当前最近的软件里程碑仍是完整请求执行。
3. P0 producer与case-aligned G3并行推进；10%最终门槛、同硬件30%释放条件和全session H200不可抢占32卡规则保持不变。
