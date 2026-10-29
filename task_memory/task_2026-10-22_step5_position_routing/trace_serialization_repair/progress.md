# D29 trace serialization repair progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Initialize the isolated repair and evidence plan. |
| 2026-10-23 | Close focused source/byte/proof verification and code commit; record the final route checkpoint. |

- completed: Read root requirements/plan/AGENTS, allocation, seam evidence, and
  captured-wave scripts. Create the approved worktree/branch from full parent
  revision `767da5437609301a6183dbfdcfd87b2402a32007`; no parent/peer edits.
- completed: Unchanged-source focused baseline: 98 passed / 1 known mapper
  fixture failure in 9.19s; raw log and JUnit XML retained in scratch. Source
  and six selected test files match frozen79 exactly (`git diff` empty).
- completed: Remove the sole `TraceEvent.to_dict` method, replace `asdict` with
  `fields`, and serialize at the existing flush write site only.
- completed: Current-source focused suite matches baseline: 98 passed / 1 known
  failure, 99 identical statuses. Four real-disk files contain eight events with
  identical event bytes. Commands/logs/XML are recorded in the test report and
  selected `evidence/` artifacts.
- completed: Captured598-event replay, real seven-event producer, declared-field
  contracts, injected exception/buffer order checks, exact AST scope and whitespace.
  Replay8,353,308 bytes and required SHA256 match; seven alternating measurements
  give0.373688s baseline versus0.066070s candidate median.
- completed: Explicitly source-bound complete actual layer60 wave:208/208 proof
  and duration matches, zero mismatches; five iterations produce identical
  14,464,252-byte trace bodies. Warm flush median0.131004s ->0.023990s. Pricing
  variation is not attributed to the repair. Full traces remain in scratch.
- completed: Code commit `390bb5a2`, sole path `frontier/metrics/trace_store.py`,
  exact+7/-7=14gross. Donor remains+40/-1=41gross below45 and untouched. Owner
  and global ceilings remain unchanged; final-union audit is parent-owned.
- completed: Repair records, final-source receipt, and selected evidence hashes
  verified; committed390bb5a2 blob equals the measured candidate hash. Records
  commit is the docs-only child of390bb5a2 on the delivered branch.
- parent-owned/pending: Independent review; final frozen-union strict77/full-unit,
  CLI, workflow admission, and native/GPU qualification.

Reason for the source change: `asdict` recursively copies already-normalized
JSON-compatible metadata at every flush. The sole consumer immediately dumps
the result; removing its undocumented method permits direct declared-field
serialization without an aliased-return interface or a second serializer.
Expected result: identical supported JSONL bytes and lower serialization time;
pricing, operator durations, and all proofs remain unchanged.

Environment setup: unprivileged system and user systemd scopes failed before
tests (interactive authentication required / user job failed). The existing
noninteractive sudo permission works; subsequent runs use
`timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env ...`.
Child Python remains brainpp, CPython3.10.6 at
`/data/ycfeng/envs/frontier-py310/bin/python`. No system configuration was changed.

Initial checkpoint found D2 environment retry drift (two scope variants without
a recorded hypothesis). Correction applied: all subsequent memory-heavy checks
used the confirmed sudo scope. No permission/configuration work continued.
Auxiliary verification errors were diagnosed and fixed only in new scratch:
AST-list dumping and duplicate pytest-current symlink counting. The complete-wave
prerequisite failure was missing the pinned submodule; a local shared clone at
gitlink3dac53c populated only this worktree, no network or peer edits.
See the test report for original failure evidence and exact commands.

Record staging: copied raw pytest logs/XML and the exact Git diff retain their
original whitespace. A whole-record `git diff --cached --check` reports trailing
spaces in those verbatim artifacts (including blank diff context lines).
Source whitespace and maintained Markdown/JSON are checked separately; do not
trim raw evidence or change its recorded hashes to satisfy a prose-style check.

## 1. 规划完成情况

**汇总：** PASS 5｜BLOCKED/FAILED/SKIPPED 0｜TODO 0

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 独立源码身份/范围 | PASS | 767da543 独立分支，相关源与79一致 |
| baseline与单文件实现 | PASS | 98/1旧失败；已落地唯一序列化路径 |
| pytest/字节/contract/AST | PASS | 99状态不变、598event与真实producer逐字节一致、AST14gross |
| 完整208 wave | PASS | 208/208完整proof无差异；flush0.1310 ->0.0240s |
| 记录和提交 | PASS | 代码390bb5a2与final receipt一致；records归档已完成 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 初次checkpoint纠正后至本checkpoint｜**当前前沿：** Parent review/freeze/gates

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | scratch仅修复验收必需问题；检查已结束不继续扩展 |
| D2 细节反复重试 | CLEAR | 已确认scope固定；各新scratch错误定位后一次修正通过 |
| D3 支线问题修补 | CLEAR | mapper旧失败只记录，不修复 |
| D4 可并行任务停滞 | CLEAR | pytest和replay并行；计时wave串行避免竞争 |
| D5 超范围镀金 | CLEAR | 仅分配源文件，无新增产品机制 |
| D6 分析空转 | CLEAR | 已落实源修改 |
| D7 路线违背 | CLEAR | 独立分支，无主分支或peer修改 |

**文档检查：** S1–S6 均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/plan合理，无需改动。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. Parent独立review代码390bb5a2与本目录diff/budget/evidence；完成标志为parent单独的review结论。
2. Parent最终union freeze/accounting -> strict77/full-unit/full CLI；本agent不代跑这些未分配gate。
3. 本agent交付code与docs-only子commit，不merge/cherry-pick primary或peer；full workflow仍未完成。

## 4. 最值得关注的 3 点

1. 98/1旧mapper失败不变，原始suite仍FAIL；comparison PASS不隐藏它。
2. 598字节、208完整proof和当前source path/hash均已落实；flush改善不等于完整CLI提速。
3. 14gross低于20，donor41低于45；parent-owned final gates仍未完成。
