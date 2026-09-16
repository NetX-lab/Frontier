## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded fresh W00 unprofiled paired baseline before production changes. |

# Performance RCA

W00: candidate `7b59d1fd`, pinned main `0515589a`; Python 3.12.3, one subprocess at a time, 3 repetitions, alternating pair order by repetition, thread limits 1, same existing run_case.py and disabled reporting. 18/18 success. Focused tests completed before measurement. Source files unchanged during campaign; task/test preparation is recorded in Git dirty provenance.

| Case | Base run s | Candidate run s | Ratio | Events / requests |
| --- | --- | --- | --- | --- |
| small_dense | 0.014485311 | 0.021181218 | 1.462255 | 176 / 2 |
| longer_dense | 0.028880736 | 0.041498117 | 1.436879 | 352 / 4 |
| representative_moe | 9.981733260 | 10.056389081 | 1.007479 | 3560 / 2 |

Command: `/usr/bin/python tests/performance/measure_pr33_paired.py --baseline ../pr33-r12-baseline-20260915 --candidate . --output /data/ycfeng/tmp/pr33-w00-20260916/paired --repetitions 3`. Full samples including init/total/RSS: `w00_paired_samples.json`. Final-source paired verification and W03/W04 ablations remain pending; no D02 waiver.
