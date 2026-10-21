# Mainline checkpoint: D20 implementation correction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Self-check after D21 source integration/P0 native completion and D22 delivery: correct stale implementation/run status, prioritize real-main mapping and case-aligned native evidence. |
| 2026-10-21 | Key-deliverable self-check afterb68ce4d: bounded communication startup fix is complete; unchanged native case requeued, compute/PD implementation remains the mainline. |
| 2026-10-21 | Key-deliverable self-check after e2be122: client CLI implementation completed; next route is compute/PD integration, not further client tooling. |
| 2026-10-21 | D20 self-check: map all30 tickets, check seven drift patterns and six document defects, apply the approved implementation-edge correction and current-state updates. |
| 2026-10-21 | Record the requested analyzer checkpoint, parent verification of tickets22/28, and the bounded execution adjustment. |

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 12｜TODO 13

PASS仅表示票据已记录完成；06按已批准的simulation scope统计。

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
|01 基线复用|PASS|票据done|
|02 native reference|PASS|锁定记录完成|
|03 模型清单|PASS|票据done|
|04 kernel/state|PASS|G1范围完成|
|05 物理测量|TODO|覆盖处置完成，数据准入未完|
|06 single-role|PASS|simulation scope关闭|
|07 observer|BLOCKED|native验证前置未齐|
|08 R2|TODO|P1源码修复完成；旧包不覆盖6K/1K，case-aligned G3未完成|
|09 四条native任务|TODO|P0请求和idle通过，实际replay/G3仍缺；H20032与H80064基础验证推进|
|10 compute profiles|TODO|D22小探针18/18严格候选通过；完整coverage和linear质量未准入，历史FAIL保留|
|11 routing|BLOCKED|票据前置未齐|
|12 communication|TODO|H20002已Succeeded，完整shape与质量准入未完成|
|13 NIXL contention|BLOCKED|依赖08/12|
|14 metric normalization|TODO|D20客户端CLI已合并并执行；native验收未完|
|15 mixed model path|TODO|结构已完成，whole-ticket未关闭|
|16 measured compute|TODO|完整compute/FFN代码已交付；真实CLI映射缺口正在修复|
|17 measured PD|TODO|stable slice201项通过，整链仍待真实main完成|
|18 capacity/cancel|BLOCKED|完整生命周期前置未齐|
|19 graph/host|BLOCKED|正式前置未齐|
|20 placement|TODO|结构完成，native与预算未完|
|21 routing/communication|BLOCKED|native与预测前置未齐|
|22 predictor bundle|TODO|真实manager生成4个bundle，每个34artifacts；初始化修复后到达idle-DP coverage缺口|
|23 canonical freeze|TODO|workload完成，runtime/metric准入未完|
|24 H800 representative|SKIPPED|D12退役|
|25 old revalidation|SKIPPED|D12退役|
|26 H800 matrix|SKIPPED|D12退役|
|27 H200 matrix|SKIPPED|D12退役|
|28 final comparison|TODO|比较代码已执行；measured Frontier和真实GT未齐|
|29 three-model regression|BLOCKED|依赖完整比较与共同源码|
|30 acceptance|BLOCKED|mandatory tickets未完成|

## 2. 偏离检测

**结论：** 未偏移，正常推进；当前文档存在S5状态滞后，需要同步。
**检测窗口：** 启动修复b68ce4d检查点至本次D21/D22集成｜**当前前沿：**16/22真实main、09各track基础证据、10 D22实测、08 case-aligned G3。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
|D1 工具打磨|CLEAR|封装tests同名包遮蔽直接阻塞09；真实子进程复现后两行修复，38项回归与实际StepMind Python即收尾|
|D2 细节反复重试|CLEAR|D22基于已证实event开销并经YC批准；导入缺陷只改一次；两次审查限流后停止重试|
|D3 支线问题修补|CLEAR|CLI graph映射、原生封装和D22均对应16/22、09、10的实际阻塞|
|D4 可并行任务停滞|CLEAR|compute/PD交付、D21文档与运行、D22均有交付；当前main修复与原生判定继续并行|
|D5 超范围镀金|CLEAR|保留既有比较器和publisher；新增package标识只用于已到达的提交路径|
|D6 分析空转|CLEAR|4个bundle实际生成，274/204项父端检查通过，P0已完成平台运行|
|D7 路线违背|CLEAR|H200非抢占32，D22另1卡抢占；64卡仅基础验证，30%解锁与最终10%未变|

**文档检查：** S5：plan§6仍称完整FFN/delegates未交付、baseline运行中，09/10状态也落后本次实物。S1/S2/S3/S4/S6未发现明确矛盾；既有08拥有G3阻塞，64 basic/formal边界准确。
**归因：** 无执行偏离HIT；S5需要按实际回执更新。

## 3. 纠正建议

**分支：** B：修正状态记录，不改变范围或验收条件。

**文档更新（先执行）：**
1. plan§6、progress与tickets09/10/16/22：同步代码交付、已完成/已提交运行及真实CLI初始化阻塞（可直接执行）。

**需停止的偏离行为：** 无。限流审查不再重试；保留未完成标记并由parent完成关键检查。

**接下去的路线：**
1. 16/22：global graph映射已修复，补齐真实request-free idle-DP coverage后再跑offline/online → 6144/1024请求、tail16、READ与metrics全部完成。
2. 并行09：判定P0实际输出并推进H80064/H20032基础验证；H20064优先复核旧基础证据，避免重复占卡。
3. 并行10：D22原18+4完整probe已通过；后续采集required per-track coverage，历史FAIL保留。
4. 08：依据P0日志明确case-aligned内容/生命周期/reference缺口后推进；22/28正式准入仍需G3及完整profiles。

## 4. 最值得关注的 3 点

1. 完整bundle已产生；真实CLI当前关键路径是idle-DP measured coverage，初始化已修复。
2. P0平台Succeeded，H20032已运行；64卡基础验证与30%正式解锁必须分别记录。
3. 当前没有任何正式四指标误差结果；G3、画像准入及完整native证据仍是验收依赖。
