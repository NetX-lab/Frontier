# Resumed D29 workflow validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Accept raw-XML regression reconciliation and T2 static binding;preserve main06 interruption and record main07 command. |
| 2026-10-23 | Verify archived three-suite results without rerunning tests;retain pending long-run verdicts. |
| 2026-10-23 | Freeze e41abc07 and launch fullCLI;independent regressions/T2 binding run in parallel. |

Status:IN_PROGRESS for full CLI;fresh e41 regression and static T2 checks PASS.

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
The suites are separate and their counts are not summed.

## Final e41 regression acceptance

Owner commands, environments and raw receipts are retained in
../resumed_regressions/test_report_2026-10-23_unit_parent04.md,
../resumed_regressions/test_report_2026-10-23_strict77_final04.md and their evidence/
directory (commits b5a74416/5ceb68c9). Parent reparsed original candidate,baseline
and supplement XMLs and verified each evidence SHA256 without rerunning suites.
regression_acceptance.json records the result and receipt hashes.

- strict77 final04:77/77 identical,zero comparison differences;451 artifact hashes
  and519 cache names per side. PASS.
- Full unit:8015PASS,279FAIL,32ERROR,281SKIP. Raw exit1 is preserved. The exact
  baseline267FAIL/31ERROR sets are retained;the only additions are12 aiohttp
  dependency failures and1 Torch collection error. All12 same IDs pass in the
  Python3.12 supplement;the affected Torch module has16 passing tests there.
  No baseline case loss or PASS-to-SKIP. PASS for the agreed regression-delta gate.
- The first parent auxiliary checker mistakenly treated numeric receipt counts
  as arrays. It was corrected to derive ID sets directly from raw XML;the corrected
  checker passed. No production or test behavior changed.

## T2 static binding acceptance

Preparation checkout:/data/ycfeng/Frontier/.worktrees/step5-d29-t2-e41,
records e8e91070/d7d6ad8c. Runtime revision remains e41abc07.
Parent rehashed all63 index entries and compared all26 complete bound files with
the frozen Git revision and actual integrated05 disk. Source tree,engine limits
7168/8192/128 and required unset of FRONTIER_STEP5_PROFILE_MANIFEST match.
PASS_STATIC_BINDING_NOT_RUNTIME in t2_e41_binding_receipt.json. The first parent
check counted only20 external-to-packet repository index entries;the documented
26 also include4 native-source files and2 inputs. Checking the exact26 resolves
that checker error. No package defect or source change was found.
Target staging is pending;16H800/two-node/30-minute resources remain unapproved.

## Interrupted main06 and unchanged main07 rerun

main06 has no completion receipt after the server restart and no remaining
matching process;online never started. main06_interruption.json retains observed
facts and hashes. Verdict:INTERRUPTED_UNVERIFIED,not a timeout or normal failure.

The fresh directory was absent before this unchanged command was launched from
the frozen integrated05 worktree:

```bash
timeout 5100s env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/integration/run_step5_measured_main.py --fixture-manifest /data/ycfeng/tmp/d29-parent-position-fixture03/fixture_manifest.json --output-root /data/ycfeng/tmp/d29-parent-position-main07 --mode all > /data/ycfeng/tmp/d29-parent-position-main07.log 2>&1
```

main07 is still running;its final workload/trace validation remains pending.
