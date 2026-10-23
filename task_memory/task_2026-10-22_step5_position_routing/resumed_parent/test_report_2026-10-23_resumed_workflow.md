# Resumed D29 workflow validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify archived three-suite results without rerunning tests;retain pending long-run verdicts. |
| 2026-10-23 | Freeze e41abc07 and launch fullCLI;independent regressions/T2 binding run in parallel. |

Status:IN_PROGRESS. No new run verdict is claimed.

Source:e41abc073500d61f224eda4fca1123e74e40544d,clean
/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05;
submodule3dac53c544e546574cb6e8078cd14053d0916113. Source-freeze receipt is adjacent.
Environment:kun-workspace-vgen2,frontier-py310 Python3.10.6,existing dependencies.

Expected:both actual modes complete6144/1024 workloads within2400s each and their
business metrics/trace validators pass. Constructor-only checks do not settle this.
Fresh source-gated regressions and T2 rebinding are independent prerequisites and
will retain their own receipts;results are not summed.

Exact root command,from the frozen worktree:
```bash
timeout 5100s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d29-parent-position-fixture03/fixture_manifest.json --output-root /data/ycfeng/tmp/d29-parent-position-main06 --mode all > /data/ycfeng/tmp/d29-parent-position-main06.log 2>&1
```

Output paths were absent before launch. The source and submodule stay frozen;
root status records are written on feat/step5-preview instead. Do not modify
retained inputs,old runs,workload bounds or the backend placement gate. This
CPU/synthetic-cost run supplies no native/GPU numerical qualification.

## Completed independent short suites

Parent reparsed the three archived JUnit XMLs under resumed_regressions/evidence,
asserted exact counts and zero failure/error/skip,and hashed each XML in
short_suites_receipt.json. Supplement101:101PASS/8.598s;metadata196:196PASS/8.208s;
catalogue53:53PASS/2.581s. This is verification of delivered results,not another
test execution. The owner's reports retain exact commands and environments.
The suites are separate and their counts are not summed. strict77,full-unit and
fullCLI verdicts remain pending.
