# D29 native collective cache-key completion

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive the accepted three-gross-line source, frozen verification, independent reviews/audit and primary integration. |

## Overview

The native collective prediction key now reuses the existing validated run-fixed
bytes rather than encoding the entire private artifact on every call. The full
canonical plan key, instance-local LRU, numerical behavior and all pre-lookup
admission/count/evidence checks remain. No new API, flag, numerical/proof cache,
source path or donor edit was added.

Source376983d2 is integrated on feat/step5-preview as c668a938. Scope records
1dd96179/65a2766d are integrated as04d4319a/9584941c; frozen records are eda4413b.

## Deliverables

- `frontier/cc_backend/backends/collective_sim_cc_backend.py`: one key-expression replacement.
- `tests/unit/test_collective_sim_cc_backend.py`: one assertion in the existing identity test.
- `task_memory/task_2026-10-22_step5_position_routing/repairs/native_collective_key_scope_amendment.json`: unchanged two-path allocation and one-line zero-sum transfer.
- This component's `review.md`, `test_report_2026-10-24_native_key.md`, `evidence/frozen_regressions/`, `evidence/frozen_scope/`, `verify_integration.py` and `evidence/primary_integration.json`: review, execution, accounting and integration evidence.

## Observed verification

- RED1FAIL to GREEN1PASS; focused baseline/candidate each124PASS/1existingSKIP.
- Canonical CPU208-event wave:19 file comparisons, complete costs/proofs/keys/
  order, snapshot isolation,64 wrong-key cases and3902-byte failure body equal.
  Live288 numerical/144 fresh proof/64 collective/16 point calls stay unchanged.
- Frozen strict77:77/77 identical;451 artifact hashes and519 cache names per side.
- Frozen unit:8028PASS/279FAIL/32ERROR/281SKIP, rawexit1; all8620 IDs/states equal
  the accepted baseline. Existing failures and the old raw presenceFAIL remain.
- Independent Standards and Spec reviews: No findings on each axis.
- Independent scope audit: zero violations,114 allocated/113 changed,
  CORE4261/4404, PRICING3955/3956, native1350/1582, source9566/9942,
  global9580/10000. Step3gross/net1 increases cumulative budget by1, not3.
  Backend93/93, test237/237, donor139/139;17 zero-sum overlays and124 bindings.
- Primary integration:9 scope artifacts,24 frozen artifacts/scripts and124
  bindings rechecked;122 non-key digest unchanged; all non-task source equals
  clean376983d2. No suite/probe rerun. Unrelated untracked files are untouched.

The isolated key/LRU experiment used8,033,853 unchanged input bytes,8 retained
plan keys and64 warm lookups per sample. Median2.806010747s becomes94.784us,
with64,271,208B of extra retained artifact strings eliminated. This is only an
expression/LRU measurement with placeholder values, not native pricing/CLI timing.

## Remaining limits

The CPU wave does not enter the changed native branch. Synthetic native-schema
tests do not qualify native hardware. The full6144/1024 offline/online CLI within
2400s, native/GPU, G3 and E2E gates remain open in the parent task. Snapshot-copy,
initialization-local and required-shape investigations keep separate ownership.
