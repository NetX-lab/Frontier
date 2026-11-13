# D21-P0-PROFILES plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Define producer-first implementation,existing consumer compatibility and bounded P0 handoff. |
| 2026-10-21 | Parent completes MRv2 source/CPU correction and immutable-stage audit. |

Dependency:A -> B -> C -> D. No delegation/platform operations.

## Current MRv2 correction follow-up

Initial stage01 serving binding is superseded;both packages are NOT FOR SUBMISSION. Historical CPU/seal results below are not proof of the corrected serving path.

Dependency:E -> F -> G -> H. No GPU or delegation. Stop generic test expansion after affected checks settle this fix.

| Step | Status | Deliverable | Acceptance |
| --- | --- | --- | --- |
| E | completed | Actual MRv2 RED/source binding | Locked worker→_dummy_run→execute_model CPU path reaches metadata after synchronized sizing;initial P0 grid fails the serving metadata assertion. |
| F | completed | Native helper/cost identity correction | Explicit descriptor input;MRv2 preparation,Step state/builder,FULL restage,negative slots,padding and live-vs-scratch semantics;no no-metadata fallback or core edits. Initial7 source checks PASS. |
| G | completed | Affected CPU regression | Parent465PASS;ordinary FULL/PIECEWISE and D22 gates unchanged. |
| H | completed | Corrected records/immutable-stage handoff | Old-stage3415/3415 hashes per role and17 native pins match;report/handoff identify the remaining descriptor and consumer gaps. No invented fresh stage. |

## Historical initial phase (serving-idle conclusion superseded)

| Step | Status | Deliverable | Acceptance evidence |
| --- | --- | --- | --- |
| A | completed | Native no-metadata workload/callable | Actual seam RED->GREEN;448 focused PASS include no builder/metadata/slot/history,positive scopes and cache-mutation fault. |
| B | completed | Consumer-shaped observations | Real448-column allocation,128/8192 capacity,per-replay root count,independent roles and unchanged consumer loader checks PASS. |
| C | completed | Bounded P0 package | Two fresh sealed role stages;each1H800/codesign/non-preemptible,3415payload hashes and533 committed source payload matches. Both actual sealed preflights PASS. |
| D | completed | Regression/seal/handoff | Latest998PASS/4 inherited FAIL,all980 starting IDs retained;source a7551a3;exact commands,budget,manifests and explicit full-admission gaps archived. Parent alone launches GPU. |

Use existing Python3.12 profiling environment/offline dependency site and Python3.10 consumer environment where needed. Raw logs/cache under `/data/ycfeng/tmp/step5_p0_profiles_20261021`; compact reports/evidence here. Full native admission remains parent-owned/pending.
