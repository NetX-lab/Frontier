# Forward Agreement Source Review

**Modification History**

| Date | Change |
| --- | --- |
| 2026-10-22 | Record self-check after489 affected PASS; not a correctness re-review. |
| 2026-10-22 | Advance local closure to PASS after verified correction commit. |
| 2026-10-22 | Resume interrupted fixed-capacity follow-up; check route after source change. |
| 2026-10-22 | Final independent source review No findings; primary integration32fb934 preserves all16 reviewed source files. |

## Current independent source review

- Reviewer: native reviewer session `ses_ea077118cffeNBkYQnihHe1SZ7`.
- Target: `ef476fe`; complete diff `5c2319cd..ef476fe`, retained at
  `/data/ycfeng/tmp/d23_forward_final_ef476fe.diff`. Inspected task records,
  D23 design, native candidate/DP semantics and first-event tests.
- **Standards: No findings. Spec: No findings.** Original cap3 P2 is resolved:
  FULL request buckets respect capacity; wider PIECEWISE remains valid; both
  local selection and final DP dispatch use the same resolver.
- The reviewer withdraws the hidden-phase example as a current Step bug:
  Step thinking is rejected, ordinary requests have depth1, and no phase cap
  override is reachable. The final fixed `_config.batch_size_cap` binding
  states the native graph-creation invariant; it is not a reproduced bug fix.
- Reviewer confirmed the primary242-PASS/33.92s XML and earlier221+268 records;
  the reviewer did not execute tests. Existing hold/resume, promotion/reset,
  NONE widths, ownership/null fields and strict coverage conclusions stand.
- Primary merged at `32fb934`; all16 changed core/test files match reviewed
  `ef476fe` bytes exactly. Clean strict77 and fresh full CLI remain pending;
  source review does not establish native/profile/accuracy qualification.

The following checkpoint is retained as the historical interrupted-owner state.

## 1. Historical planning checkpoint

**汇总：** PASS 2｜BLOCKED/FAILED/SKIPPED 0｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| F1 native/config 映射与 override 可达性 | PASS | `progress.md` 记录固定 max_num_seqs 映射和2个 Step thinking 拒绝 PASS；未证明当前 workload bug。 |
| F2 固定 config capacity 与支持状态测试 | PASS | line109 单行替换；真实 V1 config harness 和普通 profile 断言，修改前46个支持状态用例 PASS。 |
| F3 affected 验证、记录、commit、handoff | TODO | source 替换已完成；需跑 stage/engine 和直接 public predictor 受影响测试、提交。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** fixed-capacity follow-up 开始至中断恢复后的 source 替换｜**当前前沿：** F3

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | harness 仅绑定真实 V1 config 和支持状态 profile；没有扩展矩阵或 producer。 |
| D2 细节反复重试 | CLEAR | 无失败修正循环；中断恢复后先读 actual diff/progress，保留已有工作。 |
| D3 支线问题修补 | CLEAR | diff 仅 agreement、focused forward tests 和本任务记录。 |
| D4 可并行任务停滞 | CLEAR | F1 -> F2 -> F3 是真实依赖；接下来独立 CPU suites 可并行，用户禁止 delegation。 |
| D5 超范围镀金 | CLEAR | 单行来源替换，无新增 branch、flag、cost、registry 或 unsupported state。 |
| D6 分析空转 | CLEAR | source/reachability 已形成明确 invariant disposition，F1/F2 完成。 |
| D7 路线违背 | CLEAR | 不声称虚假 RED，支持状态 baseline 已留证；无 main/D25/platform access。 |

**文档检查：** S1–S6 均未发现明显缺陷；`plan.md` F1 -> F2 -> F3 依赖真实、切片有界，未要求不存在的可达反例。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. F3：并行跑 affected stage/engine 与直接 public predictor tests → 完成标志：PASS 且检查 log/XML，记录精确命令与限制。
2. F3：更新 disposition/summary，读 requirements 并提交 scoped diff → 完成标志：精确 commit receipt 和 clean branch。

## 4. 最值得关注的 3 点

1. 固定 `_config.batch_size_cap` 对应 native graph creation；mutable admission cap 不代表 graph capture capacity。
2. Step thinking 被配置拒绝；此 follow-up 是 invariant 改善，不是已复现当前 workload bug。
3. cap3/PIECEWISE 和双到达顺序需要保留；parent 的 re-review/merge/CLI/strict77 与 D25 jobs 不在本切片。
