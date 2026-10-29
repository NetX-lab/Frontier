# D29 plan-key reuse progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Start the allocated key-only public-interface RED/GREEN slice. |
| 2026-10-24 | Complete focused tests, production replay, exact-byte verification and two-path accounting. |

Task: `D29-PLAN-KEY-REUSE`; session `ses_e963098ecffebqCOtdysKGY6Xe`.
Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-plan-key`, branch `fix/d29-plan-key`, base `087db1bfcb3dbc2377d4032f650b29c48fb104fd`.

## Scope and state

- completed: read parent/local requirements and plans, amendment, glossary, repository instructions and parent domain receipt.
- completed: verify target did not exist; create isolated worktree without modifying root `.rebuttal/` or protected snapshot baseline.
- completed: extend existing identity regression by11 lines. Public seam: exact canonical key, retained successful string identity, detached dictionaries and repeated accepted bytes/bytearray encoding failures. Existing caller-sequence/replacement coverage is retained.
- completed: meaningful RED: exact canonical equality passed, then repeated public key object identity failed;1FAIL in1.57s, exit1. Raw `/data/ycfeng/tmp/d29_plan_key_reuse/01/{red.log,red.xml}`.
- completed: implement successful canonical string retention in existing `cache_key` only. Encoding completes before publication, so exceptions leave no cached value.
- completed: focused GREEN1PASS; candidate affected tests124PASS/1SKIP in13.11s; protected baseline124PASS/1SKIP in13.09s. The unchanged torch-dependent test is skipped because torch is absent; no failure or native qualification is inferred from the skip.
- completed: protected cold reporting plan and body replay on production snapshot+key. Full208 event/cost/proof/key/order, whole JSONL and ledger bytes,64 wrong-key negatives,3902-byte error recovery and all existing snapshot checks PASS. Main binding+waves read the collective file once; cold/warm reads0.
- completed:192 live cache-key calls/wave remain, but warm real encodings192->0. Numeric288, fresh FFN proofs144, collective64 and actual point16 remain. No diagnostic key/byte adapter or numerical/proof cache.
- completed: warm pricing median0.161006936->0.119742442s; separately measured emission median0.051601212->0.038155627s. These are bounded CPU-fixture measurements, not full Simulator/CLI/native qualification.
- completed: Git step accounting source5gross/+3, tests11gross/+11; total16gross/+14. Cumulative D29 path counts139/140 and311/311; donors93/14/236 unchanged. Source/input hashes and submodule3dac53c unchanged during probes.
- completed: full diff and changed-file list archived in `evidence/implementation.diff` and `evidence/changed_files.txt`; exact commands/environment/results/hashes in `evidence/evidence.json`.
- completed: verified source/test commit `c4e06a61`, containing only the two allocated paths after all required bounded checks.
- completed: records archive and primary handoff prepared; the separate records commit contains this final state, manifest and full review diff.
- pending (primary-owned): Standards/Spec dual review, parent verification, integration/final114-path accounting and broader frozen/full-workflow gates.

Expected implementation: the already-qualified5-gross `cache_key` hunk; no `to_dict`, import/helper/API or encoder/protocol change. At most16 combined gross/14 retained if the11-line existing-test extension suffices. Actual Git counts decide acceptance.

The accepted output body remains SHA256 `1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`; warm timing samples must not replace the original cold reporting plan. No diagnostic byte/key adapter, numerical/proof cache, native cache-key change, full fixture scan, full CLI/GPU/network or further delegation.

## 1. 规划完成情况

**汇总：** PASS6｜BLOCKED/FAILED/SKIPPED0｜TODO1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Public-interface RED | PASS | red.xml：canonical通过，重复identity失败。 |
| 两路径实现/GREEN | PASS | green.xml；成功编码后才发布字符串。 |
| Focused tests | PASS | 两侧124PASS/1SKIP；同一torch依赖缺失。 |
| Actual-wave parity/timing | PASS | verification.json：完整字节与工作量一致。 |
| Accounting/source commit | PASS | 16gross/+14；c4e06a61仅两路径。 |
| Records handoff | PASS | evidence、报告、manifest与full diff随本记录单独提交。 |
| Primary downstream gates | TODO | 双轴review、parent verification、integration/union/full workflow归primary。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** public RED至今｜**当前前沿：** primary review交付

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | script只完成要求的计数/字节/归档；验证后停止扩展。 |
| D2 细节反复重试 | CLEAR | RED针对重复编码；GREEN和完整wave已完成。 |
| D3 支线问题修补 | CLEAR | 未触碰H3、donors、encoding、baseline或peer。 |
| D4 可并行任务停滞 | CLEAR | 按无进一步delegation约束完成两个验证项。 |
| D5 超范围镀金 | CLEAR | 仅成功key retention和11行现有test扩展。 |
| D6 分析空转 | CLEAR | mainline已有验证、accounting和source commit产物。 |
| D7 路线违背 | CLEAR | 保留cold reporting plan、真实snapshot owner与完整live调用。 |

**文档检查：** S1-S6均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/tickets合理，无需改动。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. Records提交：核对manifest/source hashes，单独提交records → 完成标志：不包含source变更。
2. Primary交付：提交commits、changed-file list和full diff → 完成标志：可启动两轴review。

## 4. 最值得关注的 3 点

1.192 live key calls不减少；实际warm编码192->0，numeric/proof/point工作不变。
2. torch-dependent SKIP不是native qualification；full CLI与完整workflow未执行。
3. H3继续UNQUALIFIED/unallocated；下一步只有records提交和primary review。
