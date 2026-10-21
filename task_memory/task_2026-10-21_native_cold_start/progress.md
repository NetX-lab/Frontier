# Native cold-start progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Started diagnosis and scoped implementation. |

- completed: clean worktree HEAD `e55c83c263f86f48972f137858723a07c6f5e691`; read parent diagnosis, captured response, harness and frozen checker. Both P/D captures have all-lane zero native counters but no running/waiting/KV sample lines.
- completed: root checker SHA256 confirmed `9116ab7c83da2b4e7cafb871c272af56c8322954eccd56abf2eeac1b55906b55`. Its first-boundary checker currently requires three zero gauges, so a task-local successor is necessary.
- completed: captured-response RED (`observe_idle` returned False with both missing-gauge errors); first-boundary GREEN at actual replay/SSE seam completes 432 canonical requests with explicit zero counters, empty gauge maps, manifest/PID/health-bound readiness. Post-boundary targets are 16/144/160/288/304/432.
- completed: successor copied from exact frozen root bytes (SHA256 equal before edits); only `native_idle_evidence` and native replay proof binding changed. Eight replay scenarios and seven readiness mutations pass. Root checker remains unchanged.
- completed: focused regressions. Iteration01: 36 PASS / 9 FAIL (new optional keyword in existing test stub, fractional fixture counter, worker fixture absent provenance); cause-specific corrections keep keyword absent when proof is unavailable, leave later counter number semantics unchanged, and add explicit fixture process receipts. Iteration02: 59 PASS / 1 FAIL (fault-injection fixture accidentally changed its original missing-file error to missing-key); corrected fixture setup only. No native or transport change. Broader oneshot-py312 attempt: 12 FAIL / 14 ERROR due to unrelated installed `tests` shadowing fresh subprocesses; existing asyncopd-py312 + py310 simulator avoids that environment issue. Final 241 PASS / 21 SKIP / 0 ERROR,113.62s. Evidence retained in this directory.
- completed: report, narrow successor diff, canonical/checker/collector byte-identity evidence. Five production/test files and453 changed lines. Read back requirements before completion.
- completed: code/test/artifact commit `1a2d6e9` (`fix(calibration): prove dedicated native cold start before warmup`); indexed diff check passes. No unrelated files staged.
- pending outside this slice: parent review/freeze/repoint and future authorized native run.

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 实际捕获响应 RED | PASS | 两个 missing-gauge 错误可重复。 |
| 首次证明及 readiness | PASS | 432 请求 replay 与 readiness 负例通过。 |
| 后续严格 freshness | PASS | 失效初始 baseline 不再放行后续边界。 |
| successor 与回归 | PASS | 完整候选 checker 路径及最终241测试通过。 |
| 报告、归档及 commit | PASS | 报告/identity已归档，本地commit1a2d6e9；parent冻结另行处理。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 最终回归至commit1a2d6e9｜**当前前沿：** parent交付

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 测试仅覆盖批准边界及负例。 |
| D2 细节反复重试 | CLEAR | 解释器/namespace 问题已定位；各回归失败对应独立原因，无参数反复调整。 |
| D3 支线问题修补 | CLEAR | HTTP502 未修复。 |
| D4 可并行任务停滞 | CLEAR | 用户禁止 delegation；共享文件串行修改。 |
| D5 超范围镀金 | CLEAR | 无 CLI flag、模型/Frontier core 修改。 |
| D6 分析空转 | CLEAR | 已形成可执行证明及 replay 结果。 |
| D7 路线违背 | CLEAR | 全局 checker/native collector 只读。 |

**文档检查：** S1–S6 均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets 合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 交付commit、报告、successor及narrow diff → 完成标志：parent获得明确冻结/repoint责任。

## 4. 最值得关注的 3 点

1. 冷启动是专用证明，未生成任何缺失 gauge 的零值。
2. 原 V4 仍是 431/432 且首次边界缺少证明，不追认。
3. parent 仍需 review/freeze/repoint successor 和新的 run generation。

Ranked source-backed hypotheses: (1) first-step-only native gauge publication causes the initial failure (confirmed by parent source chain); (2) missing initial counter baseline disables later freshness (confirmed by inspected replay control flow); (3) stale or unrelated readiness could falsely imply a dedicated service (existing readiness contains only role/url, requiring a bound Popen/manifest seam). No frontend count, transport or simulator change is planned.
