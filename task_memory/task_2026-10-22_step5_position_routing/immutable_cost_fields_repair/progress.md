# D29 immutable cost fields progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Parent recovers owner records after quota error, verifies committed source and closes independent component reviews. |
| 2026-10-23 | Record isolated source/authority setup and baseline check. |
| 2026-10-23 | Close actual-source differential, known failure comparison,8gross/0retained accounting and code commit; archive review-ready records. |

- completed: Worktree/branch feat/step5-d29-cost-fields from48d34106. Root authority,
  requirements, plan and relevant AGENTS read. Only this worktree's submodule is
  initialized from a local shared clone, detached at3dac53c; no network or peer
  modification. Gitlink unchanged.
- completed: Baseline and candidate six-file pytest both98PASS/1FAIL, exact99ID/
  status sets and failure message identical. Knownfailure is
  test_mla_core_native_op_tracing.py::test_frontier_stage_batch_component_ledger_dense_attention_uses_shared_mapper,
  unexpected allow_structural in the old fixture. No attempted repair.
- completed: Rebound retained setup to48d34106 current trace+prepared source;
  original/derived SHA and only3path/HEAD replacements recorded. Candidate wrapper
  changes only output label/location and two actual source hash expectations.
- completed: Only two declared-field substitutions and their imports. Git+2/-2
  per source,8stepgross, retained1aee0af45/22 unchanged, delta0. Exact textual/AST
  expected change and whitespace checks pass; gitlink remains3dac53c.
- completed:208full events/proofs/order,2,889,655eventbytes and full2,309,545-byte
  ledger exactly equal; wholeJSONL fixedheader also equal.64cost objects are
  scalar/string/numeric tuples. Snapshot/mapping mutation checks pass;
 64wrong-plan-key errors and3902-byte lateJSONerror prefix/restoration unchanged.
- completed: Emission-only median58.067739->51.181030ms (-11.859785%), asdict
 336->208,deepcopy26,336->21,472,plan.to_dict192/key64unchanged. No pricing or
  fullCLI timing attribution.
- completed: Source commit7540414341d5b81a86b18b9562d1506c0ba74285 after all focused
  checks. Tested source hashes equal committed blobs. New records/diff/receipt are
  delivered by the separate records commit carrying this archive.
- completed: Independent Standards/Spec reviews bothNo findings; parent actual
  208event/fullledger/64negative/snapshot/error checks pass on75404143.
- completed: Initial131PASS/14SKIP traced to13 missingHTSIM cases plus existing
  Torch absence. Local3dac build and affected-file rerun54PASS close the13
  environment skips.145 distinct latest IDs resolve to144PASS/1TorchSKIP;
  overlapping suites are not summed. Parent receipt and exact commands retained.
- completed: Code merged to feat/step5-preview. Owner API quota failure happened
  after source commit; parent recovered the uncommitted records and archived
  the selected raw evidence. This supersedes the owner handoff checkpoint below.
- parent-owned: final union/freeze/regressions and fullCLI1024/2400s acceptance.
  No whole-workflow PASS asserted.

## 1. 规划完成情况

**汇总：** PASS 4｜BLOCKED/FAILED/SKIPPED 0｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| Authority/source baseline | PASS |48d34106真实trace+prepared绑定、旧probe保留 |
| 两处字段替换 | PASS |75404143仅两source路径，+4/-4 |
| 完整差分/negatives | PASS |208完整byte/ledger一致，64错key和snapshot检查通过 |
| pytest/AST/Git accounting | PASS |98/1同状态，8gross/retained0，45/22不变 |
| 记录交付/parent验收 | TODO |新records就绪；parent独立review/freeze/fullworkflow未执行 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 新allocation开始至此checkpoint｜**当前前沿：** records交付/parent review

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR |仅复用验收必要scratch，验证已完成不扩展 |
| D2 细节反复重试 | CLEAR |baseline/candidate/final checks无新失败修复循环 |
| D3 支线问题修补 | CLEAR |旧mapper失败原样保留，不修无关问题 |
| D4 可并行任务停滞 | CLEAR |pytest期间准备binding；计时避开测试资源竞争 |
| D5 超范围镀金 | CLEAR |仅8gross，两既有路径无新API/test/cache |
| D6 分析空转 | CLEAR |已实施、验证并提交source75404143 |
| D7 路线违背 | CLEAR |保留snapshot/key/flush，无unallocated controls |

**文档检查：** S1–S6 均未发现明显缺陷。
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/plan合理，无需改动。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. 交付独立records commit与精确receipt/diff；完成标志是仅新目录records、源码blob与实测一致。
2. Parent独立Standards/Spec review -> union freeze/accounting -> requiredregressions。
3. Parent结合其余残留成本再评估wholeCLI1024/2400s；局部11.86%不是workflow admission。

## 4. 最值得关注的 3 点

1.8stepgross替换既有addedlines，retained增长0，不能重复计owner消费。
2.98/1原始suite仍FAIL；99状态/消息不变比较PASS，旧failure未隐藏。
3.208完整proof与所有snapshot/key/error检查相同；fullworkflow仍parent-owned。
