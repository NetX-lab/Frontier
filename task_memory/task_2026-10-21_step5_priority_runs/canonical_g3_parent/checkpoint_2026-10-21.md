# Canonical G3 partial-delivery checkpoint

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Check the five delegated implementation steps and resume the unfinished native-reference slice. |

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

范围：`step5-canonical-g3` 的 `canonical_g3/plan.md` 五项步骤；PASS 仅表示该步骤的记录已完成。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 恢复纠正后的 R2 与 P1 检查 | PASS |7d46abe；父端复验包括旧 source/trace 负控制 |
| Canonical capture、caller 与状态控制 | TODO |32请求与 abort 已实现；reset/reuse/pressure 闭环未完成 |
| 核对独立 reference 可行性 | PASS |锁定源码证明无 transfer metadata 可完整本地执行；父端已选择此复用方式 |
| 受影响 CPU 验证 | PASS |父端658项通过；不代表 native G3 |
| 完整实现与可执行封包交接 | TODO |已有部分提交；reference 和状态控制完成后才可封包 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** D21-CANONICAL-G3 分配至父端部分交付复验｜**当前前沿：** reference/状态控制实现、完整 CLI 执行和 P0 profiles。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR |本轮 capture/abort 能力直接服务 ticket08 缺失验收项；父端一次受影响复验后停止扩大测试 |
| D2 细节反复重试 | CLEAR |复用已纠正 R2；新增失败记录对应明确的测试时钟/断连观测问题，未调整 native 行为 |
| D3 支线问题修补 | CLEAR |改动归属 ticket08 与其共享请求通路 |
| D4 可并行任务停滞 | CLEAR |capture 与 reference 源码核对均推进；完整 CLI 和 P0 profiles 另有 owner |
| D5 超范围镀金 | CLEAR |仅实现既有内容、生命周期、abort 与 reference 要求 |
| D6 分析空转 | CLEAR |已交付7d46abe可执行 capture/abort；父端依据源码完成 reference 选择并恢复实现 |
| D7 路线违背 | CLEAR |保持 P0、锁定 native 路径、原阈值；未把部分 CPU 通过当作 G3 |

**文档检查：** 在本五步计划及 ticket08 相关要求内，S1–S6 均未发现明显缺陷。完整主任务的其他票不在本次检查范围。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动验收要求。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. ticket08：完成现有 D 引擎的独立本地执行、完整 resident identity 与状态比较，并实现 reset/reuse/显式压力控制 → 完成标志：真实调用与判定代码齐备、受影响验证通过，能够生成新的完整 stage。
2. 并行：完整 measured CLI 的请求/READ/释放/final metrics 验收 ‖ P0 native profiles 的缺失 coverage 生产。
3. 父端核验完整 stage 和资源后执行 native G3；实际内容、独立 reference 与生命周期证据满足原标准后才能解除正式校准前置。

## 4. 最值得关注的 3 点

1. 658项 CPU 通过只关闭部分代码验证；G3 内容与完整生命周期证据仍未完成。
2. 复用 D 引擎无需新增 GPU 或替换 reference；完整参数/buffer/scales 同一性及状态比较仍是硬要求。
3. 下一交付应是完整调用/判定与可执行 stage；完整 CLI 和 P0 profiles 同时推进，避免再次停在缺口报告。
