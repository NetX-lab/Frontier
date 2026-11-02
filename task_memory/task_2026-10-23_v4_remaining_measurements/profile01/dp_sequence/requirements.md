# Diagnostic DP sequence recording

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded W128-DP-SEQUENCE-IMPLEMENTATION owner allocation and explicit decisions. |

## Requests and decisions

[Original Request] Task ID W128-DP-SEQUENCE-IMPLEMENTATION: implement/verify one bounded non-core change, commit and stop. Dedicated worktree `/data/ycfeng/Frontier/.worktrees/v4-dp-sequence-record`, branch `feat/v4-dp-sequence-record`, base `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`. Parent requirements lines94–102 are authoritative.

[Original Request] Own only `tests/comparison/calibration/{native_full_structure.py,native_serve_profile.py}` within40 gross changed lines combined and `tests/unit/{test_native_v2_serving_inventory.py,test_native_serve_profile.py}` within140 combined; total180. Own records only in this directory. No native/C26/frontier/frozen checker, parent document, old artifact/package or unrelated edits; no GPU/platform/network/package/run or further delegation. CPU-only tests may read the pinned native checkout.

[Original Request] Retain optional `steps[].dp_post_all_reduce_sequence` from the actual explicit DP group raw getter via the existing post-all-reduce hook and inventory thread-local side state. Never add it to `dispatch_evidence`, base inventory rows/dedup keys or hooks. No callback/interface/config/schema version. Reset every execution, including paths without current dispatch and native failures. Enrich before final detached-row/capacity admission.

[Original Request] Read `kwargs['group']` strictly; missing/throwing getters produce existing bounded soft observation errors, leaving the counter absent while preserving the independent CPU vector and native return/exception identity. Update only local realistic-group fixtures, including the local frozen-oracle namespace. No WORLD/default inference, offset subtraction, native-step-index claim or duplicate-collective branch beyond existing ambiguity rejection.

[Original Request] Verify nonzero startup, repeated/zero vectors, successive executions, successful-reduction ordering, inactive dispatch, missing/raising getter isolation, native return/exception identity, resets, detached rows/byte capacity, unchanged frozen acceptance/rejections. Establish baseline first. Focus suites: native_v2_serving_inventory/native_serve_profile/native_serving_inventory/native_groundtruth_check; adjacent suites only where imports justify them. Bind existing77-case matrix using retained root `stream_probe/matrix_binding.json` and hashes, not a new capture.

[Decision] Pairing semantics remain within `(run generation, role, ordered DP group ranks)`, with lane/rank as member identity, not part of the same cross-member join key. Counter is optional diagnostic evidence, never a calibration predicate or native-reference admission. Exact-image WS2 counter proof is inherited evidence, not native worker pairing.

[Decision] Root owns independent review, final integration and new native collection. Deliver committed source/records, hashes, per-file +/- budgets, source.diff, commands/results, matrix binding, frozen-byte checks and unresolved limits, then stop.

[Original Request — bounded envelope correction] Root found pinned `gpu_worker.py:1210–1212` directly calls runner `_dummy_run`, bypassing Worker.execute_model/profile.steps. Carry the same optional raw counter in existing annotation/context records while native dispatch is active, via `context()` without hooks or fabricated steps. Post-execution sample_tokens/last_step context must use the detached step's counter, not a later dummy execution's thread-local counter. Preserve absence without active dispatch or a recorded step counter. Verify actual source-backed ordinary/direct-dummy paths, stale protection and retained bytes, within the unchanged four-file/180-line allocation; no additional calibration predicate.
