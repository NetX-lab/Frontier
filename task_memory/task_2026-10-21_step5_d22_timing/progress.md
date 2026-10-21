# D22-TIMING progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record clean baseline, approved scope and implementation route. |

- completed: inspected clean `79c8b45`, applicable AGENTS, parent D21/D22 requirements, real timing seam, wrapper, judge, D18 stage builder and 851-test command.
- completed A: real-seam RED receipts `/data/ycfeng/tmp/d22_timing_red01.xml` (absent reference) and `d22_timing_red02.xml` (only five windows, expected independent cohort). First GREEN run passes the injected event-overhead case; seven old exact-count expectations identify required extra reference work and were updated without removing test IDs.
- completed B: independent event layer reference is separate from per-iteration operator samples; raw boundary samples/statistics are explicitly diagnostic. Strict judge verifies reference normalization/count/minimum/labels and still applies 5% mean-versus-reference and candidate CV. Existing eager GPU-active and scoped rules remain unchanged. Existing CLI now measures each cell exactly once; CPU failure injection proves no quality retry. Existing stage/launcher can materialize preemptible H200 resources with unchanged defaults.
- completed C: final affected873 PASS/14 existing warnings in51.20s, all851 baseline IDs/verdicts preserved and22 added;0 missing/changed. Exact logs/XML: `/data/ycfeng/tmp/d22_timing_20261021/affected02.{log,xml}`, `baseline_comparison02.json`. Compact final log/comparison and original RED XMLs retained under `evidence/`. Prior862 and focused253 counts overlap, not additive. `git diff --check` PASS. Final changes since first affected pass add explicit short-reference/invalid-reference cases and remove dead retry metadata branches.
- pending D: source commit, fresh preemptible sealed stage, source/package preflight and parent handoff.
- scope clarification: existing CLI quality retry conflicts with D22's explicit no-quality-retry contract; remove it in existing CLI, retaining single-attempt metadata. Resource wiring extends existing stage/launcher only to select `preemptible=True`; no platform imports/calls outside CPU doubles.

## 1. 规划完成情况

**汇总：** PASS 2｜BLOCKED/FAILED/SKIPPED 0｜TODO 2

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| A timing seam | PASS | 真实 seam RED->GREEN；独立执行、独立 primed events、内部事件开销只在诊断保留。 |
| B producer/judge/policy | PASS | focused253 PASS；质量失败不重试；资源 CPU double 证明 preemptible。 |
| C final regression | TODO | 首轮862 PASS保留851 IDs，最后小改后待重跑。 |
| D immutable stage/handoff | TODO | 依赖最终回归和 source commit。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本次任务开始至当前 CPU checkpoint｜**当前前沿：** C，然后 D

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 测试和 judge 仅覆盖 D22 必需的新参照协议。 |
| D2 细节反复重试 | CLEAR | 首轮七个失败均是额外参照工作的旧计数预期，无产品修复循环。 |
| D3 支线问题修补 | CLEAR | 无独立故障排查或新诊断实验。 |
| D4 可并行任务停滞 | CLEAR | 父任务禁止 nested delegation；封存依赖已验证 source。 |
| D5 超范围镀金 | CLEAR | preemptible 只扩展已有资源选择，满足 D21。 |
| D6 分析空转 | CLEAR | 已交付协议实现和862 CPU passes。 |
| D7 路线违背 | CLEAR | 无平台调用、阈值变化、开销扣除或历史 verdict 改写。 |

**文档检查：** S1–S6 均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. C：最终 affected regression + baseline ID comparison → 全部原851 IDs PASS。
2. D：commit verified source、封存原始完整 probe、离线 preflight → handoff 后停止；GPU 由 parent 执行。

## 4. 最值得关注的 3 点

1. 新协议 CPU 已通过，不代表 native timing 通过。
2. 新 stage 必须 preemptible；parent 仍负责真实 quota/priority 检查。
3. 原 D18/H800 FAIL 和 stage02 永远保留，不重新计算为 PASS。
