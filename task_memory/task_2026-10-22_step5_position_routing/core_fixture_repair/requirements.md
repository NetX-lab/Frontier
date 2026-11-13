# Affected CORE fixture requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record the root's test-only allocation, accepted seams, immutable registry integration and verification boundaries. |

TaskID: **D29-CORE-AFFECTED-FIXTURES**. Worktree `/data/ycfeng/Frontier/.worktrees/step5-core-review-fixes`, branch `feat/step5-core-review-fixes`, base `e49b26f849ed5edc87b4c443ce753db64d8e5a07`.

[Original Request] Continue in the same worktree. Own only `tests/unit/test_moe_routing_runtime.py` and `tests/unit/test_moe_grouped_gemm_padded_features.py` for this step; separate <=80 gross relative to e49b26f8. Effective caps110(runtime) and210(padded), original review228/230 unchanged, CORE4404/global10000/109paths unchanged. Read the absolute root `repairs/affected_fixture_decision.md` and updated `core_review_scope_amendment.json`, applying them after `registry_scope_amendment.json`; do not overwrite the root overlay with an older copy. Development merge of immutable registry a5189e199208022a341b377b0fb8c542030907f3 is authorized; no peer edits, independent registry review remains separate.

[Original Request] `_position_source_rows` must provide a canonical descriptor and matching census through existing producer/materializer, preserving actual shared/independent trainer, preflight, fit/cache assertions. Runtime fixture must use formal Step model/group/selected layer and real existing registry marker/view/physical rows, preserving positive exact plan context and foreign-context negative. No production fallback or minimal-state bypass. Reproduce the five current failures before changing tests. Run both complete affected files and coherent producer/registry/trainer/consumer selection; retain every new failure, not exclusions. No other production/test/registry edits, dependencies, new CLI/GPU runs or independent public mode. Commit the scoped tests, preserve root-owned gitlink, and archive English report/diff/budget/commands locally under core_fixture_repair.

[Explicit Decision] Root acceptance criteria and zero-sum allocation are in `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/repairs/affected_fixture_decision.md` and `core_review_scope_amendment.json`. The original delivery/review artifacts remain unchanged. This task qualifies fixture correctness and CPU consumer plumbing, not source/image/native/GPU/E2E qualification.

[Explicit Decision] Seams are the five named failing tests, both full affected test files, and the coherent full seven-file union. Reuse existing `bound` fixture, production source descriptors/forward agreement/materializer, shared predictor layer binding, and actual registry/trainer APIs. No new registry or independent public predictor behavior.
