# Routing reader correction progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Started bounded reader correction from clean b62c155 worktree. |
| 2026-10-23 | Verified focused MXFP4 RED/R4 GREEN and strict actual-rank identity controls. |
| 2026-10-23 | Completed 208-test final regression, diagnostic 8/8 PASS, immutable evidence and matrix-scope inspection. |

Task ID: W128-V4-NATIVE06-ROUTING-READER.

## Status

- completed: Requirements, primary RCA/JSON and FINAL_FREEZE inspected. Local R3 matches `ca8ddcea83c852c27295c8a73453746d755c51055e703bd94df48e8f1af7d917`. The unchanged comprehension excludes exact `Mxfp4MoEMethod`; separate initialization records preserve MXFP4 MARLIN consumers and fully qualified runner/child identities.
- completed: R4 replaces exactly the single R3 owner enumeration line. Independent runner/child/backend sets must agree and be nonempty; supported recorded consumer tuples are centralized. Frozen R2 loading remains hash-bound by unchanged R3.
- completed: Focused persistent regressions: 75 PASS, 53 deselected in 5.64s. The real unchanged collector comprehension excludes an actual-shaped MXFP4 child, frozen R3 rejects it, and R4 accepts independently collected initialization/routing records. All 32 actual ranks pass R4 routing identity and reject under frozen R3. Missing/extra/wrong independent facts, consumers and source/strategy identities reject; legacy absence and dependency tamper controls pass.
- completed: Final native reader/dispatch regression: 208 PASS in 51.23s. Added FP8/TRITON real-collector legacy controls and exact unchanged collector-source hash assertions; R4 source unchanged after the diagnostic.
- completed: New diagnostic receipt `native06_r4_diagnostic.json`: unchanged actual native06 returns eight predicates PASS, including 32 physical ranks and six phases per rank. No downstream failure emerged. Original R3 analysis remains FAIL and no admission is granted.
- completed: Before/after 199 original-file hashes unchanged; all 43 package freeze pins and frozen R3/R2 unchanged. 744 regular production/matrix-input files byte-equal to baseline b62c155; collective-sim gitlink unchanged and uninitialized. Reachability evidence is in `integrity_and_matrix_scope.json`.
- completed: Requirements reread after root's matrix-accounting update, final records and exact source/test allocation verified. Source and records are ready for the scoped commit and handback; no more implementation work remains.
- pending (root-owned): Independent review, rerun, R4 freeze, comprehensive-matrix capture reuse decision and native/reference re-admission.

## Evidence and decisions

Read-only initial inspection used `/usr/bin/python3 -B` (Python 3.10.6, no conda activation). First actual rank contains 1,224 initialization modules and 365 backend entries: 43 MXFP4 MARLIN, 43 attention and 279 FP8 linear entries. Every recorded MXFP4 row is `{quant_method: Mxfp4MoEMethod, backend: MARLIN, experts: MarlinExperts, prepare_finalize: MoEPrepareAndFinalizeNaiveDPEPModular, input_dtype: "None"}`. Original analysis remains seven PASS and routing identity FAIL.

The repair gain is complete independent owner enumeration from already retained actual metadata. Existing `moe_backends` cannot provide that completeness because its collector is FP8-only. No collector or raw-record change is needed or authorized.

Focused command (working directory is this worktree):

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_reader_binding.py -k 'balanced_strategy_actual_import_and_missing_identity or test_r4' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-focused --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-focused.xml
```

Read-only original-byte snapshot: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-original-hashes.json`. It monitors 199 unique original case/package/native/source files and validates 43/43 package freeze pins and both frozen source hashes. No original analysis write or GPU/external action occurred. Final maintained source/test allocation: two files, 216 additions and seven deletions, 223 changed lines; zero core edits. R4 is 52 lines; the one existing test file has 164 additions/seven deletions.

Final regression command and exact diagnostic body are retained in `test_report_2026-10-23_routing_reader.md`. Initial merged regression passed 201 tests in 50.62s; final FP8 legacy/source-hash controls increased it to 208 PASS. Diagnostic check ran once through a non-main import and `check()` only; UTC 2026-10-22T21:25:47.385681+00:00 to 21:25:52.431852+00:00 (task naming follows the requested local date 2026-10-23).

One auxiliary inspection failed because it treated the uninitialized collective-sim gitlink as a missing regular file. The diagnosis was confirmed by git tree mode 160000 and unchanged gitlink d28fe917370c2d55230c3ea226e5660785b44aca. Correcting only the inspection yielded zero changed regular files and unchanged gitlink; no repository/source behavior was altered. This is not a downstream checker failure.

The latest root requirement tightens comprehensive-matrix accounting. No local capture is reused or gate waived here. The task supplies exact unchanged-byte and reachability evidence; root maps it to the prior 77/77 capture or runs the affected default matrix.
