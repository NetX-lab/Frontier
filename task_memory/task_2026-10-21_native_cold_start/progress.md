# Native cold-start progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Started diagnosis and scoped implementation. |
| 2026-10-22 | Reopened full-checker admission on primary P1; captured RED, corrected validation and completed full focused rerun. |

- completed: clean worktree HEAD `e55c83c263f86f48972f137858723a07c6f5e691`; read parent diagnosis, captured response, harness and frozen checker. Both P/D captures have all-lane zero native counters but no running/waiting/KV sample lines.
- completed: root checker SHA256 confirmed `9116ab7c83da2b4e7cafb871c272af56c8322954eccd56abf2eeac1b55906b55`. Its first-boundary checker currently requires three zero gauges, so a task-local successor is necessary.
- completed: captured-response RED (`observe_idle` returned False with both missing-gauge errors); first-boundary GREEN at actual replay/SSE seam completes 432 canonical requests with explicit zero counters, empty gauge maps, manifest/PID/health-bound readiness. Post-boundary targets are 16/144/160/288/304/432.
- completed: successor copied from exact frozen root bytes (SHA256 equal before edits); only `native_idle_evidence` and native replay proof binding changed. Eight replay scenarios and seven readiness mutations pass. Root checker remains unchanged.
- completed: focused regressions. Iteration01: 36 PASS / 9 FAIL (new optional keyword in existing test stub, fractional fixture counter, worker fixture absent provenance); cause-specific corrections keep keyword absent when proof is unavailable, leave later counter number semantics unchanged, and add explicit fixture process receipts. Iteration02: 59 PASS / 1 FAIL (fault-injection fixture accidentally changed its original missing-file error to missing-key); corrected fixture setup only. No native or transport change. Broader oneshot-py312 attempt: 12 FAIL / 14 ERROR due to unrelated installed `tests` shadowing fresh subprocesses; existing asyncopd-py312 + py310 simulator avoids that environment issue. Final 241 PASS / 21 SKIP / 0 ERROR,113.62s. Evidence retained in this directory.
- completed: report, narrow successor diff, canonical/checker/collector byte-identity evidence. Five production/test files and453 changed lines. Read back requirements before completion.
- completed: code/test/artifact commit `1a2d6e9` (`fix(calibration): prove dedicated native cold start before warmup`); indexed diff check passes. No unrelated files staged.
- pending outside this slice: parent review/freeze/repoint and future authorized native run.

## P1 review correction

- completed: primary independently reproduced the earlier241 PASS/21 SKIP, but that test omitted worker/lifecycle/terminal predicate assertions. Added actual `native_workers` PASS; `/data/ycfeng/tmp/native_cold_start_p1_red.xml`:1 FAIL in2.58s. Worker says `Head readiness endpoint mismatch: host0`; incomplete nodes also fail lifecycle/terminal. This supersedes the earlier completion claim for full checker admission.
- completed: ranked predictions before fix: (1) old whole-object equality rejects metadata despite successful pinned binding (confirmed); (2) identity disagreement would also reject requests/idle (ruled out in RED); (3) lifecycle fixture itself incomplete would still fail after worker correction (ruled out in GREEN).
- completed: shared checked `dedicated_services` receipt from replay with `native_workers`; new-proof worker compares the complete fresh readiness object with the source-pinned receipt. Validation of all fields remains in the existing pinned validator. If receipt rebinding fails, new-proof worker fails. Legacy whole-object `{role,url}` comparison remains unchanged. No production harness/native changes for P1.
- completed: targeted GREEN1 PASS in18.88s; seven expected predicates PASS, rank-runtime-graph FAIL as expected. All16 missing/wrong readiness-field cases rejected by both request/idle and worker predicates; legacy exact and extra-field verdicts identical to frozen checker. Final code/test count495 after formatting.
- completed: affected+existing seams final241 PASS/21 SKIP,132.05s. Retained `p1_predicates_{red,green}.json`,16 new-proof/2 legacy cases in `p1_readiness_negatives.json`,JUnit,`p1_review_receipt.json`. Canonical/root checker/collector hashes unchanged; candidate5efd00ea6544d64440c535d06ad415b6a4d1a25b96ce8863cf4944568f37ec36. Exact code cumulative diff includes actual successor; P1 fix diff is against746b744. Report/test report/summary now withdraw earlier incomplete predicate proof.
- completed: P1 code/test/evidence commit `35fe8a08488d269fb9aa7325021e82cccb53f57d`. Source/document whitespace check passes; exact raw diffs/JUnit preserve captured whitespace. Sole writer has stopped successor edits; only this completion receipt update follows. Await primary acceptance, no freeze/package.

## 1. 规划完成情况

**汇总：** PASS 6｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 实际捕获响应 RED | PASS | 两个 missing-gauge 错误可重复。 |
| 首次证明及 readiness | PASS | 432 请求 replay 与 readiness 负例通过。 |
| 后续严格 freshness | PASS | 失效初始 baseline 不再放行后续边界。 |
| successor 与原回归 | PASS | 原241测试通过，但不作为完整predicate验收。 |
| P1全checker校验 | PASS | 完整focused241 PASS;七预期predicate PASS，一预期native qualification FAIL。 |
| 报告、归档及 commit | PASS | P1证据/diff已归档，commit35fe8a0；parent接受前禁止冻结。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 完整回归至commit35fe8a0｜**当前前沿：** primary acceptance交付

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 测试仅覆盖批准边界及负例。 |
| D2 细节反复重试 | CLEAR | 一次明确字段比较假设即可消除RED。 |
| D3 支线问题修补 | CLEAR | HTTP502 未修复。 |
| D4 可并行任务停滞 | CLEAR | primary明确指定sole writer，无delegation。 |
| D5 超范围镀金 | CLEAR | 无 CLI flag、模型/Frontier core 修改。 |
| D6 分析空转 | CLEAR | 已形成可执行证明及 replay 结果。 |
| D7 路线违背 | CLEAR | 全局 checker/native collector 只读。 |

**文档检查：** 旧S5完成声明已用P1实测纠正；requirements/plan/report/progress一致。S1–S6无未解决缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：最新spec/plan合理，完成声明已纠正。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. 交付commit35fe8a0、cumulative/narrow diff和receipt → 完成标志：parent可独立复核，sole writer停止，不冻结。

## 4. 最值得关注的 3 点

1. P1已用完整predicate断言关闭；七PASS不意味着native qualification。
2. 原 V4 仍是431/432且首次边界缺少证明，不追认；全局checker和collector未动。
3. parent接受前不冻结/打包；commit后sole writer停止。

Ranked source-backed hypotheses: (1) first-step-only native gauge publication causes the initial failure (confirmed by parent source chain); (2) missing initial counter baseline disables later freshness (confirmed by inspected replay control flow); (3) stale or unrelated readiness could falsely imply a dedicated service (existing readiness contains only role/url, requiring a bound Popen/manifest seam). No frontend count, transport or simulator change is planned.
