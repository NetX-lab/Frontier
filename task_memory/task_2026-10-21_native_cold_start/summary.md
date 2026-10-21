# Native cold-start completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archived W128-NATIVE-COLD-START implementation and verification. |

Implemented the approved first-boundary-only proof in `tests/comparison/calibration/pd_replay.py`, with manifest-bound actual Popen/health readiness receipts in `tests/comparison/calibration/native_pd_worker.py`. Missing metrics are never synthesized. Later six boundaries retain all-lane zero gauges, KV-release and cumulative completion freshness; an unverified initial baseline cannot disable freshness. Client failures and native exceptions retain original behavior.

Regression files: `tests/unit/test_native_cold_start.py` and `tests/unit/test_native_pd_worker.py`. Candidate checker: `task_memory/task_2026-10-21_native_cold_start/check_groundtruth_run_successor.py`, copied from root SHA256 `9116ab7c83da2b4e7cafb871c272af56c8322954eccd56abf2eeac1b55906b55`; candidate SHA256 `48e311358aa23ae56e6f89b0b8686dced5d6ff5d7b3bcf10bb638dab9d091771`. Root checker is untouched. Narrow delta `checker_successor.diff` changes only native first-boundary proof and receipt binding.

Observed verification: 241 passed,21 skipped,113.62s; nine captured-response replay scenarios, seven readiness negatives, snapshot negatives and full successor checker branch. Exact commands/results in `test_report_2026-10-21_native_cold_start.md`, retained JUnit and JSON evidence alongside this file. Five production/test files and453 changed lines excluding the unchanged frozen checker copy. Canonical workload byte-identical to68f891b; collector byte-identical toe55c83c.

Code/test/evidence commit: `1a2d6e9` (`fix(calibration): prove dedicated native cold start before warmup`). No merge or push.

Open/deferred: parent reviews/freezes/repoints the candidate for new generations; future native launch verification is not authorized in this slice. Old V4 remains unverified (missing initial proof,431/432 completions,repeat02 warmupHTTP502). No transport/native/core fix or GPU/network/platform operation was performed.
