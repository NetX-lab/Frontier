# DP sequence implementation completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Archived W128-DP-SEQUENCE-IMPLEMENTATION source/CPU delivery and explicit limits. |

## Outcome

Completed the bounded diagnostic-envelope implementation on `feat/v4-dp-sequence-record`, dedicated worktree `/data/ycfeng/Frontier/.worktrees/v4-dp-sequence-record`. Source commit: `bb50186e417f99080d71948b38f6006e954a8c41`; base: `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`.

The existing post-all-reduce observer strictly reads the actual `kwargs['group']` getter after successful native reduction, independently of CPU-vector capture. Its raw counter is stored only in inventory thread-local side state. Both existing Worker and V2 execution entries reset that state. Optional `dp_post_all_reduce_sequence` is emitted before final detached step admission and through the existing context/annotation envelope while native dispatch is active. Direct dummy annotations retain `step=None`; post-execution sample_tokens context reads the saved detached step's counter, not a later dummy execution's counter.

Frozen base dispatch keys, row/dedup behavior, hook statements, schemas and native return/exception identity are unchanged. Getter failures are existing bounded soft observation errors and leave the field absent. No counter offset, default group, new callback/config/interface, hook, collective, calibration predicate or native-reference admission was added.

## Deliverables and line budget

| Source | Added | Deleted | Gross |
| --- | ---: | ---: | ---: |
| `tests/comparison/calibration/native_full_structure.py` | 8 | 0 | 8 |
| `tests/comparison/calibration/native_serve_profile.py` | 9 | 1 | 10 |
| `tests/unit/test_native_v2_serving_inventory.py` | 59 | 6 | 65 |
| `tests/unit/test_native_serve_profile.py` | 71 | 2 | 73 |
| Total | 147 | 9 | 156 |

Production18/40 and tests138/140, total156/180; no expanded allocation. All records are confined to `task_memory/task_2026-10-23_v4_remaining_measurements/profile01/dp_sequence/`:

- `requirements.md`, `plan.md`, `progress.md`, `summary.md` and `test_report_2026-10-24_dp_sequence.md`.
- `source.diff` and `source_binding.json`: exact source/base file hashes and +/- budget.
- `baseline.json`, `red.json`, `green.json`, `green2.json`, `final.json`, `final2.json`, `context_red.json`, `final_context.json`: exact CPU commands, original errors, results and log/XML hashes.
- `frozen_native_binding.json`, `matrix_binding.json`: frozen/native bytes and explicit inherited77-case evidence.
- `records_manifest.json`: per-record SHA256 and exact source/base revision, excluding its own bytes to avoid self-reference.

Source set SHA256: `52c2596a8705b2002ad3a2de45360de45592be5fe806fde992f5d4b735ae838a`. Source.diff SHA256: `71ca353c76af4f5b392291f7226328c0d6547a7aefca7eff5e3a970656f04a78`. Final records commit and manifest SHA256 are returned in the delivery message; there is no self-referential commit hash in the record bundle.

## Observed validation

Final separate CPU suites: V2 **46PASS**, profile **36PASS**, inventory **28PASS**, groundtruth **78PASS/21SKIP**, directly composed native_full_structure **6PASS**. Total **194PASS/21SKIP**, zero failures/errors. The unchanged skips are19 unavailable approved C26 source controls and2 missing external legacy snapshot controls; these are not passed checks.

Source-backed native dispatch and Worker controls cover startup counter37, post-reduction38, repeated/zero matrices39/40/41, inactive dispatch, missing group/getter and throwing getter, native marker/exception identity, pending-send failure before runner entry, zero/no-forward reset and final-row detachment/bytes. Ten existing FULL/PIECEWISE cases execute the unedited native Worker.execute_dummy_batch body, prove direct dummy annotation39 with no Worker step, saved-step sample_tokens context38, absence without current dispatch/step and exact retained byte accounting.

Frozen R2 matches its existing pin `29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a`; tracked R3 equals base bytes, SHA256 `ca8ddcea83c852c27295c8a73453746d755c51055e703bd94df48e8f1af7d917`. Existing R3 acceptance/negative controls pass, including rejection of a counter added inside dispatch_evidence. Native Worker/V2/DP files match pinned2cf0a691 Git objects; DP source matches retained exact-image SHA256 `3c882f85109ba47e473953351d166c4377ceb995205e91cbf128ca5075775a5d`.

The existing77-case comparison is explicitly bound, not rerun or newly captured: inherited source5f7d0950, original comparison4a1b53e6, comparison SHA256 `7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6`. Five relevant domain Git trees are identical between original/inherited/base and remain unchanged in the candidate.

## Remaining work and limits

No owner implementation issue remains. Root still owns independent review, integration, fresh native collection and actual cross-DP pairing. This delivery establishes source/CPU behavior only; it does not establish native startup offsets, full native Worker symmetry, GPU timing, serving-cost calibration or historical profile02 recovery. Pair only within run generation/role/ordered DP group ranks, with lane/rank as member identity; never use a local Worker step or subtract an offset to infer alignment. No GPU/platform/network/package operation or further delegation was performed.
