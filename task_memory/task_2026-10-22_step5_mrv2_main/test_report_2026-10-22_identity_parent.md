# D23 fixed identity parent verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record parent96 checks, complete iteration and source/ledger audit. |

Worktree `/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf`, source7e98e5c,
record HEAD3ece1ce9ffea09e9f22b4642223186721182161a. Python3.10.6,
`/data/ycfeng/envs/frontier-py310/bin/python`. This is minimal Python execution
enablement; D29 defers native CPU-overhead/gap work until workflow/GPU alignment.

## Exact executed commands

```bash
timeout 60s env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_model_identity_reuse.py tests/unit/test_step5_contract_reuse.py tests/unit/test_step5_model_bundle.py -q --basetemp=/data/ycfeng/tmp/d23_identity_parent_tests01 --junitxml=/data/ycfeng/tmp/d23_identity_parent_tests01.xml
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_parent_identity_verify01 --max-runtime 30 --resume-state /data/ycfeng/tmp/d23_actual06_exact_q1_state/state.pkl --start-emitted 1 --outputs 3
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_actual06_iteration.py --output /data/ycfeng/tmp/d23_parent_identity_audit01 --max-runtime 30 --resume-state /data/ycfeng/tmp/d23_actual06_exact_q1_state/state.pkl --start-emitted 1 --outputs 3 --audit
```

Logs use the same basename as each output/XML under `/data/ycfeng/tmp`.
Unit checks:96 PASS in35.53s. Plain public intervals7.982853/8.115971s, versus
frozen parent9.451313/9.549744s. Still RED against2.012202s optimistic bound;
exit1 is this expected latency verdict. Audit intervals10.924394/10.897711s
include audit overhead and are not performance comparisons.

Parent independently compared event bytes, parsed stage JSON, lane digest,
event/lane counts, collective deltas, and full source_truth.json against the
frozen parent controls. Every comparison PASS;1168 events,5632 lanes and12224
CC hits/zero misses unchanged. Receipt
`/data/ycfeng/tmp/d23_identity_parent_comparison02.json`.

Independent source review session `ses_e9e8b2f76ffeL18tOAXFSAdgoY` reports No
findings. It checked full supplied diff, source/tests and author299-check
evidence without rerunning it. Marker cache is selected-estimator/full-marker
bound; per-call selection, paired identity and dynamic validation/proofs remain.

Remaining: merged legacy77, corrected D26 fixture and complete1024 output/
offline/online CLI. Timeout, native/workflow/GPU and final accuracy remain open.
No larger CPU-overhead campaign was started.
