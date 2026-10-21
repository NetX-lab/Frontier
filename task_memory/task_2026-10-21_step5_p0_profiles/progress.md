# D21-P0-PROFILES progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Inspect source/native evidence and start executable no-metadata capability. |

- completed: switched session to clean bc8ae27;read AGENTS,D20–D22,ticket10,measured handoff,actual-main records and source scopes.
- observed: consumer requires explicit one-slot eager no-metadata identity,zero block-table/score extent and distinct fusion; producer currently always builds metadata and emits1. Ordinary serving consumer uses full448 block-table columns; producer defaults to live width. Eager aggregate trace counts are not yet exported as per-replay root counts required by the consumer.
- completed A/B: existing input/metadata/forward supports one-slot eager request-free; explicit boolean identity and zero table extent serialize through unchanged consumer. Native full-table allocation and serving128seq/8192token capacity are explicit. Real timing collectors/scoped protocol unchanged. Actual trace association exports per-replay root counts,not aggregate/sample counts. Cache mutation fault is rejected before timing. Variable positive contexts preserve pages; history preparation and full references split by request only when the8192 budget requires it.
- completed C source: configured P0 native35 capture buckets from retained H800 init log,7168 context,128seq,8192budget,448columns. Independent P/D command packages include attention,linear and EP16 MoE;all10 actual CPU CLI dry-runs returned0. Prepared P4event+4GPU-active;D20event+4GPU-active. These are representative layers0/3 and widths1/2,not full admission or an observed-width census.
- completed verification checkpoint: initial meaningful native forward RED absent `has_attention_metadata` constructor; initial GREEN143PASS. Expanded producer/consumer/native helpers281PASS;latest focused seams70PASS including explicit cache-mutation and budget/page-offset fault checks. Counts overlap,not additive. First combined broad run hit120s shell timeout;rerun uses unlimited background timeout,not a quality retry.
- completed D CPU:448 final focused PASS;latest-result union998PASS/4 confirmed inherited failures/1002IDs. All980 starting bc8ae27 IDs retained,22 added. Actual starting-source comparison reproduces4 execution-disabled assertion failures;one archive-no-git environment failure is reported separately. A new dataclass-default mismatch in the old boundary CPU matcher/expected description was corrected narrowly;no native boundary experiment was run. Original D18 one renamed runtime-disabled test already predates this task;no new test removal.
- in-progress C/D: source commit,sealed new role stages,full hash/CLI preflight and final parent-only receipt. Exact commands/counts/limits recorded in test report and evidence/.
- limits: native H200 D22 PASS reused as protocol evidence only;not H800 time. Actual variable widths/arrays remain unobserved;DSA partial-summary contexts remain unsupported by repeated timing. No further causal experiment is needed.

## 1. 规划完成情况

**汇总：** PASS2｜BLOCKED/FAILED/SKIPPED0｜TODO2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A native idle-DP | PASS | 真实 metadata/forward seam RED→GREEN；无 metadata/slot/history，cache mutation 被拒绝。 |
| B consumer-shaped rows | PASS | 未改 consumer 的 loader 接受新 producer rows；物理448列、预算及 per-replay root count 有定向检查。 |
| C sealed P0 package | TODO | 两个独立 role 的10条实际 CLI dry-run 均通过；等待 source commit/seal。 |
| D regression/handoff | TODO | 聚焦验证通过；正在分类旧 baseline 与当前 bc8ae27 的继承冲突。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** bc8ae27 assignment 至首次 producer 里程碑｜**当前前沿：** C/D

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只扩展现有 stage/preflight/judge 以运行新的已准备 workload。 |
| D2 细节反复重试 | CLEAR | RED 是缺失接口；剩余失败已定位为旧 execution-disabled 断言，未改参数重试 native quality。 |
| D3 支线问题修补 | CLEAR | 页号、预算和 count 是现有 consumer 的直接阻断项；不修无关 simulator。 |
| D4 可并行任务停滞 | CLEAR | 明确禁止 delegation；CLI dry-runs 与测试并行，代码共享文件串行修改。 |
| D5 超范围镀金 | CLEAR | 无新 predictor、admission、canonical/G3 或 GPU action。 |
| D6 分析空转 | CLEAR | 已实现 callable 和实际可执行的10条 producer command。 |
| D7 路线违背 | CLEAR | H800/codesign/nonpreemptible；旧 stages/FAIL 不变，沿用 D22 独立 cohorts。 |

**文档检查：** S1–S6 均未发现明显缺陷；本次边界明确允许 prepared coverage 不等于 full admission。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. D：完成受影响测试和 bc8ae27 继承错误分类 → 不引入新增失败、不隐藏旧失败。
2. C/D：提交 source、封存两个新 role stages、执行哈希/CLI preflight → 交给 parent 唯一提交 GPU 的命令及预算。

## 4. 最值得关注的 3 点

1. 新 idle-DP 是真实无 metadata 调用，不是普通 q1 替代。
2. 两个 role 的所有 producer CLI 已可执行；native H800 时间仍需 parent 实测。
3. 其他90个 global layers、variable widths/contexts、DSA partial contexts 和线性质量仍是明确准入缺口。
