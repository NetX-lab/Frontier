# Current-source cohesive joint timing

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 UTC / HKT | Collect and archive the one finite two-call execution; no runtime call is repeated. |

The unique execution passes complete CPU-fixture output parity on both cold/warm attempts. This closes cohesive timing only; complete6144/1024CLI runtime/RSS, finalT2/package/native/G3/E2E remain unverified.

## Authority, command and environment

Decision:D29-CURRENT-JOINT-TIMING-TWO-CALLS-EXECUTE-20261025, YC verbatim answer continue. Exactly2/2 ordinary D1/layer60 pricing attempts; warmups0, profiling0, retries0, required-shape calls0. Required-shape remains6/6.

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -B task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/joint_timing.py --bindings task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25/joint_bindings.json --output-root /data/ycfeng/tmp/d29_current_joint_20261025_01
```

Working directory:/data/ycfeng/Frontier/.worktrees/step5-preview. Records HEAD6a15009e1684d12501f99487e301d66cea1d96df before/after; admitted source2ca5e57113b0bc99ab84e926e5bb492b3670072e equals frozen146 outside task_memory. Copied Python3.10.6, -B, single OMP/OpenBLAS/MKL/NumExpr thread. Actual execution2026-10-25T03:57:26.692272Z–03:57:32.852978Z, rawexit0, outerwall6.160573354s. ConfiguredMemoryMax32,212,254,720B and whole-process timeout60s.

## Observed timings

| Component (seconds) | Cold | Warm |
| --- | ---: | ---: |
| Pricing | 0.844966512 | 0.088767464 |
| Plan and timing | 0.000020486 | 0.000709893 |
| MetricsStore emission including ledger flush | 0.060888903 | 0.165885018 |
| Final trace/ledger flush and close | 0.022945238 | 0.024321925 |
| Joint interval | 0.928821139 | 0.279684300 |

MetricsStore construction and trace-header initialization precede each timed interval. Validation follows it. Cached-fixture setup3.806491587s is recorded separately and is not complete-main startup. The process ru_maxrss is1,315,135,488B; this is observed fixture-process memory, not configuredMemoryMax or full-workload RSS. No effective cgroup peak is recorded. Two observations provide no distribution or whole-workflow2400s claim.

## Required full output and actual observations

Both attempts satisfy PASS_COMPLETE_OUTPUT_PARITY:16ordered EP lanes,208operators,144compute proofs. Exact full pricing-reference bytes, sorted ledger JSONL rows, trace-header identity, all proofs/keys/order and captured-source identities pass; actual newly allocated destination request/batch identities are bound on each wave. Both final trace buffers are empty; ledger count16 each.

| Full artifact | Cold bytes | Warm bytes |
| --- | ---: | ---: |
| Pricing payload | 647510 | 647510 |
| EP lane ledger | 2309529 | 2309646 |
| Operator trace | 2889856 | 2891377 |

The unique execution verifies25source bindings/sixinputs/262fixture metadata before/after. The archive independently matches actual output sizes/hashes to the execution receipts, both pricing hashes to the retained reference, and all decompressed bytes to the originals. Current five-worktree ancestry and unchanged2ca/146source comparisons pass; composed audit and mandatory regressions are retained, not rerun.

## Exact retained evidence

resumed_execution/evidence/joint_timing_2026-10-25/ contains business_execution.json, command_execution.json, command.stdout, command.stderr, archive_joint.py, archive.json and six complete lossless gzip output files under outputs/{cold,warm}/. Original output11,695,428B becomes798,661 compressed bytes; no field, row, key, proof or ordering is removed. archive.json records every compressed and original digest and byte count.

Archive command:timeout60s sudo-n systemd-run --scope -p MemoryMax=30G runuser-u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_archive_joint_20261025_01.py. Rawexit0, archivewall1.005787683s, zero new runtime/platform calls. Original scratch roots:/data/ycfeng/tmp/d29_current_joint_20261025_01 and /data/ycfeng/tmp/d29_current_joint_capture_20261025_01.

All original canonical test failures and completed-worker Pod403 remain retained. This report allocates no source change, joint/shape retry or native/GPU run. Next:complete6144/1024offline/onlineCLI, <=2400s per actualmain child, full validation outside that cap, actual workload memory/controller evidence, then freshT2/source/package/native/G3/E2E.
