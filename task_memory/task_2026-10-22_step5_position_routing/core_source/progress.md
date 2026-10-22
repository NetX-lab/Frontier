# CORE/SOURCE progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Read approved designs, sole-writer matrix and baseline source. |
| 2026-10-23 | Record verified runtime/source consumers, timer replay and concrete peer boundary. |

- completed: baseline clean, branch feat/step5-position-core at 1aee0af; scope.json grouped by exact path yields 44 paths/4404 gross lines.
- completed: first count/source/tensor/provenance interfaces, 53 PASS/1 SKIP in frontier-py310; actual source tensors/counts in asyncopd-py312 4 PASS/1 deselected. AST and whitespace PASS. Current 7 code/test paths, 713 gross/4404.
- completed: first interface commit `c995fdc`, 713 gross code/test lines, receipt delivered to root.
- completed: runtime/source threading: constructors resolve registry before parent initialization, shared scheduler passes actual bound DP sources, immutable full-source cache separates lane reorder/padding/position/context, ratio tables are not synthesized for the new policy. Source/load/conservation consumer tests pass.
- completed: independent descriptor/launcher/provenance interfaces: explicit image reaches existing ResourceConfig launcher; generated artifacts use the existing copier/seal; actual GateLinear/router/norm callables use the full receipt; legacy native_facts remains absent; producer-specific imports remain distinct under shared identity.
- completed: singleton timer `_profile_cell_callable` switches to RECORD_FUNCTION through tracer parser, restores/clears on call/parser failure, retains original event durations and carries paired kernel_count. Real CPU DeviceTimer/record_function annotations and parser executed; transport alone is synthetic. Mask sentinel is excluded before map/bincount.
- observed: selected consumer check 215 PASS/19 SKIP/4 deselected; Py3.12 timer/router/norm/oracle check 18 PASS; actual linear attachment/legacy source test 34 PASS. Actual old fixed-probe stage/verifier PASS, 540 sealed files/18 attention rows/16 linear rows. Exact commands in dated reports.
- completed: runtime commit `ac48363`,11 code/test paths,+176/-54=230 gross beyond first interface commit.
- in-progress: commit source/timer step and archive partial handoff. Current29/44 changed paths,+1470/-95=1565/4404; no file allowance exceeded. Canonical input parser/executor serialization test added to the existing output-metadata file and verified; latest source identity suite35 PASS.
- blocked: selected trainers/preflight/schema/physical price consumers and new fixture/main. PRICING `7b86614` provides the selected registry context contract, but import/merge is root-only and not authorized. Its uncommitted `NativeRoutedExperts.prepare_source/prepared_context/apply_prepared` does not yet provide the complete schema/range/numerical plan fixture receipt. No replacement registry/schema/plan is created.
- pending root-owned: import/coordinate peers, selected integration qualification, full-unit/strict77/review, actual image/source receipt and runtime/E2E runs. Native actual setter/factory candidate is readable; published derived image/full receipt remain unavailable.

## 1. 规划完成情况

**汇总：** PASS 2｜BLOCKED/FAILED/SKIPPED 2｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| count/source/tensor/verifier interfaces | PASS | test_report_2026-10-23_interfaces.md; interface_receipt.md |
| scheduler/runtime/cache | PASS | actual bound DP sources and constructor/cache consumer tests |
| selected trainer/profiling consumers | BLOCKED | selected registry import authorization and completed native/schema facts required |
| source package/producers | TODO | independent launcher/gate/norm/timer tested; position MoE main and full new package remain peer-dependent |
| actual-main fixture/runner | BLOCKED | completed selected native/schema/finite-plan consumers required |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** c995fdc 至今｜**当前前沿：** scoped commits/peer integration coordination

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | actual scheduler/source/timer consumers settle required scoped interfaces |
| D2 细节反复重试 | CLEAR | fixture mixed-root and JSON canonicalization failures each corrected once; final tests pass |
| D3 支线问题修补 | CLEAR | no unrelated fixes |
| D4 可并行任务停滞 | CLEAR | runtime and independent source work both advanced; user prohibits extra agents |
| D5 超范围镀金 | CLEAR | 29 owned paths,1565/4404 gross; no new unallocated module |
| D6 分析空转 | CLEAR | real source/timer/launcher changes and tests delivered; remaining dependency named precisely |
| D7 路线违背 | CLEAR | no peer import/platform/source fallback |

**文档检查：** S1–S6 均未发现明显缺陷；implementation progress is updated, not treated as a scope change.
**归因：** 无。

## 3. 纠正建议

**分支：** A：spec/plan合理，无需改动。
**需停止的偏离行为：** 无。
**接下去的路线：**
1. Commit verified runtime/source slices → 完成标志：scoped clean commits with dated tests/receipt/quota.
2. Root supplies registry import authorization and completed PRICING numerical/schema/finite-plan payload → 完成标志：one actual selected contract available, without alternate registry/fallback.
3. Resume selected trainers/preflight/wrapper/main, then new fixture/main → 完成标志：all selected source/gathered/empty physical exact pairs and preserved workflow actual-main checks pass.

## 4. 最值得关注的 3 点

1. First count/source/tensor/verifier interfaces are ready; runtime/source consumers preserve the same owners.
2. Shared trainer/schema/fixture work is blocked by real peer API/import prerequisites, not quota exhaustion;2839 gross lines remain.
3. CPU plumbing is verified; published-image/full receipt, native/GPU/runtime/E2E qualification remain UNVERIFIED and root-owned.
