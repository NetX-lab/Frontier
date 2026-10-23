# Affected CORE fixture repair

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Allocate two existing test paths after inspecting five integration failures. |

Decision: D29-CORE-AFFECTED-FIXTURES. Root uses existing in-scope correctness
authority. Base e49b26f849ed5edc87b4c443ce753db64d8e5a07. CORE remains4404,
effective union109 paths, total10000. No production change is authorized by this
test-only allocation. Original seven-path repair remains228/230.

## Evidence and scope

- `/data/ycfeng/tmp/d29-core-review-legacy-union01.log`: three tests in
  `tests/unit/test_moe_grouped_gemm_padded_features.py` fail with missing
  `routing_input`. `_position_source_rows` supplies physical/live/masked counts
  but no actual source descriptor. Supply the canonical descriptor and matching
  source census through existing mechanisms; retain both trainer/cache checks.
- `/data/ycfeng/tmp/d29-core-review-registry-joint02.log`: two cases in
  `tests/unit/test_moe_routing_runtime.py` fail at actual routed-group resolution
  because the old model stub has no model_type. Use actual Step model/config,
  selected layer/group and existing registry bindings/physical rows. Retain
  exact context/plan recovery and foreign-context rejection.

The two paths receive a combined80-gross repair cap relative to the base. Effective
path caps are110 for runtime tests and210 for padded-feature tests. Transfer45
unused runner,15 wrapper and20 fixture-producer allowance; apply the maintained
core_review_scope_amendment.json after registry_scope_amendment.json. No new path,
typed identity, cache hash, independent public mode or qualification relaxation.

## Acceptance

Reproduce the five failures, repair their fixtures, run both complete affected
test files and the coherent combined producer/registry/trainer/consumer suite.
Record exact commands, failure IDs, results and final scope accounting. Do not
delete assertions, skip the failures, patch production admission, or edit the
registry owner's files. Commit the test step and return its immutable receipt;
parent independently verifies and integrates it after review.
