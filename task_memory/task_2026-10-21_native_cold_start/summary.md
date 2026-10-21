# Native cold-start completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archived W128-NATIVE-COLD-START implementation and verification. |
| 2026-10-22 | Closed primary P1 readiness rejection; withdrew incomplete full-checker assertion claim. |

Implemented the approved first-boundary-only proof in `tests/comparison/calibration/pd_replay.py`, with manifest-bound actual Popen/health readiness receipts in `tests/comparison/calibration/native_pd_worker.py`. Missing metrics are never synthesized. Later six boundaries retain all-lane zero gauges, KV-release and cumulative completion freshness; an unverified initial baseline cannot disable freshness. Client failures and native exceptions retain original behavior.

Regression files: `tests/unit/test_native_cold_start.py` and `tests/unit/test_native_pd_worker.py`. Candidate checker: `task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py`, copied from root SHA256 `9116ab7c83da2b4e7cafb871c272af56c8322954eccd56abf2eeac1b55906b55`; latest candidate SHA256 `5efd00ea6544d64440c535d06ad415b6a4d1a25b96ce8863cf4944568f37ec36`. Root checker is untouched. Frozen-to-current delta `checker_successor.diff` changes only native first-boundary proof and receipt binding, including complete validated worker readiness.

Primary P1 identified that the previous241 passed/21 skipped result omitted worker/lifecycle/terminal assertions and could not prove full-checker admission. Exact RED: worker rejected new metadata as `Head readiness endpoint mismatch: host0`. The correction reuses the pinned full receipt for new-proof worker validation; legacy exact `{role,url}` checks remain unchanged.

Latest observed verification:241 passed,21 skipped,132.05s; all seven expected full-checker predicates PASS, rank-runtime-graph and overall status intentionally FAIL for missing native source/rank/graph qualification. All16 missing/wrong readiness-field cases rejected; two legacy controls retain frozen-checker verdicts. Exact commands/results in `test_report_2026-10-21_native_cold_start.md`; receipt `p1_review_receipt.json`, exact cumulative five-file code diff `cumulative_code_vs_parent.diff` (includes actual successor), P1-only `p1_fix.diff`. Five production/test files and495 cumulative changed lines excluding the unchanged frozen checker copy. Canonical workload byte-identical to68f891b; collector byte-identical toe55c83c.

Code/test/evidence commit: `1a2d6e9` (`fix(calibration): prove dedicated native cold start before warmup`). No merge or push.

P1 correction code/test/evidence commit: `35fe8a08488d269fb9aa7325021e82cccb53f57d` (`fix(calibration): admit bound cold-start worker readiness`). Successor edits stop here pending primary acceptance.

Open/deferred: primary acceptance is required before freeze/package/repoint. This agent stops successor edits on handoff; future native launch verification is not authorized in this slice. Old V4 remains unverified (missing initial proof,431/432 completions,repeat02 warmupHTTP502). No transport/native/core/metrics change or GPU/network/platform operation was performed.
