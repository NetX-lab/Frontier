# Cold-start fixture dependency correction

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded the baseline-reproduced staged-module mismatch and a bounded test-only scope amendment. |

The expanded required regression found `test_successor_full_checker_rebinds_pinned_launch_receipts` failing because it stages the current `pd_replay.py` beside an older `vllm_replay.py`. The latter lacks `NATIVE_WORKER_EXTENSIONS`, already imported by the accepted H version. This is not a G2 replay regression or a missing Python package.

Evidence: `/data/ycfeng/tmp/glm53_dsv4/g2_cold_start_child_stderr.log` retains the isolated child ImportError. `/data/ycfeng/tmp/glm53_dsv4/g2_cold_start_import_control/receipt.json` shows baseline313c8167 with the old staged peer fails identically, while replacing only that peer with this repository's module passes. Original fixture files remain hash-identical.

Root authorizes one additional existing test path, `tests/unit/test_native_cold_start.py`, at most25 changed lines within the unchanged750-line G2 total. The fixture must stage and hash-bind both current modules together and retain the existing launch/admission assertions. No frozen checker, package, historical run or production code changes. This extends the source allocation to seven existing non-core files; root remains the sole implementer.

The initial two reviews cover the original six-file artifact. This additional test correction must be included in the final review follow-up and final source manifest.
