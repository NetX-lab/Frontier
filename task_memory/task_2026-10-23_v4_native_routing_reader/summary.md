# Native06 routing reader correction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived verified bounded correction and root-owned acceptance limits. |

Task ID: W128-V4-NATIVE06-ROUTING-READER. Decision: D-V4-NATIVE06-ROUTER-OWNER-READER. Baseline b62c155; branch fix/v4-native-routing-reader.

## Outcome

Added a 52-line hash-bound R4 successor that replaces only frozen R3's incomplete FP8-only owner enumeration. Complete nonempty owner/child/backend agreement comes from independent initialization metadata; fully qualified module identities and supported recorded consumer tuples are checked before unchanged strict routing/source/strategy and downstream predicates. Frozen R3/R2 are untouched.

## Deliverables

- `task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py`.
- `tests/unit/test_native_reader_binding.py`: actual unchanged MXFP4 collector RED/R4 GREEN, FP8 legacy path, all 32 actual-rank positives, strict identity negatives and dependency tamper controls.
- `task_memory/task_2026-10-23_v4_native_routing_reader/{requirements.md,plan.md,progress.md,checkpoint.md,test_report_2026-10-23_routing_reader.md,summary.md}`.
- `task_memory/task_2026-10-23_v4_native_routing_reader/native06_r4_diagnostic.json`: new diagnostic receipt, never original analysis.
- `task_memory/task_2026-10-23_v4_native_routing_reader/integrity_and_matrix_scope.json`: immutable evidence and production/matrix unchanged-byte reachability evidence.
- `task_memory/task_2026-10-23_v4_native_routing_reader/correction.diff`: source/test diff artifact generated from the verified staged correction.

Exact maintained delta: two files, 216 additions/seven deletions, 223 changed lines; zero core edits. R4 SHA256 `67f3192302b6d12c0b8296581ee8b2788a772bce80b158ad45bc349eacb32a7b`.

## Observed verification

Focused regressions: 75 PASS. Final existing native reader/dispatch suite: 208 PASS. Actual identity census: 32/32 ranks pass R4 with 43 independent owners each and reject under frozen R3. Full read-only R4 `check()` diagnostic returns eight predicates PASS, including 32 physical ranks and six phases/rank; no downstream failure emerged. Every other predicate and frozen replacement path is unchanged.

199 original-file hashes unchanged, including original R3 FAIL analysis; 43 package inputs and frozen R3 `ca8ddcea83c852c27295c8a73453746d755c51055e703bd94df48e8f1af7d917` / R2 `29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a` match. 744 regular production/matrix-input files are byte-identical to b62c155, and the collective-sim gitlink is unchanged/uninitialized. No GPU, platform, SSH, cloud, Docker, raw-schema, collector, native, policy or metric change.

## Open work and acceptance boundary

Implementation and verification scope is complete. Root owns independent review/rerun, successor freeze, comprehensive-matrix capture accounting and any native/reference re-admission. No new admission is granted by this delivery. Root may reuse the prior 77/77 capture only after mapping the supplied unchanged-byte and reachability evidence to it; otherwise the affected default matrix is still required.
