# CPU runtime diagnosis and correction review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Parent reproduces runtime RED and verifies collective/trace controls; scopes fixed-contract reuse and independent initialization diagnosis. |

## Diagnosis accepted at the measured boundary

Source74980bb adds diagnostic scripts and records, without production changes.
The parent read the actual cache-replay path and reran the measurements. It
retains the actual05 typed artifacts, exact feature keys, durations, counts,
precision and evidence identities. It bypasses initialization/fitting only to
isolate existing production query methods; it is not full stage/CLI acceptance.

Parent commands, cwd=`/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf` for the
first command and parent `step5-preview` after merging74980bb for the controls:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --iterations 2 --no-profile --output /data/ycfeng/tmp/d23_parent_red01 > /data/ycfeng/tmp/d23_parent_red01.log 2>&1
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_collective.py --output /data/ycfeng/tmp/d23_parent_cc01 > /data/ycfeng/tmp/d23_parent_cc01.log 2>&1
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --trace --no-profile --output /data/ycfeng/tmp/d23_parent_trace01 > /data/ycfeng/tmp/d23_parent_trace01.log 2>&1
```

Environment: existing frontier-py310 Python3.10.6, CPU only, unchanged fixture02.

| Parent measurement | Result |
| --- | --- |
| 64 real FFN lane queries | 0.532956s; RED; exact source truth PASS |
| Cache replay loading | 1.468053s |
| ATTN_TP8 allreduce cold / identical hit | 1.884812s / 0.128823ms; same0.122465ms prediction |
| EP16 all-to-all cold / identical hit | 2.490454s / 0.201464ms; same0.0199104ms prediction |
| Enabled trace and JSONL flush | 228 events,172466 bytes,40.468442ms; persistence truth PASS |

The runtime profile in the author's report attributes most measured query work
to repeated fixed metadata, precision and operator-contract construction.
These controls support that diagnosis for this boundary. They do not establish
the distribution of the entire original run's costs or all cold collective misses.
Original initialization remains a separate unprofiled interval.

## Scoped source correction

Implementation owner: `ses_e9f519775ffe3Cv0bhEzJSKNhH`, task
`D23-MRV2-CONTRACT-REUSE`, same isolated performance worktree.

- Reuse fixed physical-layer metadata and expensive precision/contract binding
  within one predictor lifetime. Inspect supported configuration lifetime first.
- Preserve the cheap existing per-query family/model registry selection rather
  than adding a second model cache and invalidation mechanism.
- Continue dynamic feature validation, exact coverage/bounds, paired count and
  source-proof checks. Do not cache feature-dependent outputs or change costs.
- Cover distinct layers, precisions, measurement families, predictor instances
  and nested context restoration with real repeated-query regressions.
- Keep new logic in existing small Step modules and minimize giant-module wiring.
  Source review, public stage checks and full CLI acceptance remain parent-owned.

The diagnostic per-query budget is derived from the original2400s bound and
observed setup, not a native accuracy threshold. Its maximum-query verdict alone
does not prove complete workload runtime; retain cold/steady observations and
source truth instead of changing behavior to satisfy a synthetic timing number.

## Independent initialization diagnosis

Session `ses_e9f3e2ed0ffejPdfKHpqPHhXWY`, task `D23-MRV2-LOAD-DIAG`, owns a short
profile of the original manager/typed input loader in a separate worktree.
This assignment permits diagnosis only, with unchanged source rows and no broad
fit, full retry, cache mutation or production change. Its result will determine
whether an additional initializer correction is justified.

Full offline/online completion, READ/readiness/release, final metrics and native
qualification remain unverified. The full execution bound remains2400s.
