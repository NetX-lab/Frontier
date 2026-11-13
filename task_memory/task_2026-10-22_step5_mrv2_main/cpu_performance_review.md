# CPU runtime diagnosis and correction review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Parent reproduces runtime RED and verifies collective/trace controls; scopes fixed-contract reuse and independent initialization diagnosis. |
| 2026-10-22 | Parent verifies 7.21x query improvement and repeated initializer loads; records source-change invalidation requirements and pending source review. |
| 2026-10-22 | Loader reuse independently accepted with parent206 checks and exact frame/hash proof; merged224 PASS and actual06 launched. |

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
The whole initialization interval has not been profiled. The later prefix and
warm-cache controls below identify repeated input processing within that interval.

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

Session `ses_e9f3e2ed0ffejPdfKHpqPHhXWY` completed `D23-MRV2-LOAD-DIAG` at
`32a1f60`. The parent reproduced its existing command from `step5-mrv2-load`:

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/profile_step5_load.py --config /data/ycfeng/tmp/d23_main_actual05_offline/offline/metrics/stepfun_ai_step_5_preview/offline_batch/step5_measured_cpu_main_offline/config.json --output /data/ycfeng/tmp/d23_load_parent01 --rows 100 --event-cache-hash 23b7026a --assert-no-repeated-loads > /data/ycfeng/tmp/d23_load_parent01.log 2>&1
```

Observed: `RED_REPEATED_LOAD`, exit 1. The already-deduplicated DECODE role still
loads twice, taking 4.655791s, with no fitting or cache read. Warm PREFILL takes
5.101961s: loader time 4.600761s, two cache reads 0.003788s. The original event
cache loads independently in 0.523899s. Source size/mtime and original model
configuration checks pass. These are prefix measurements, not a complete
attribution of the original 998/1004s initialization. The author's extrapolated
747s is not a measured full-run value.

The same owner completed `D23-MRV2-LOAD-REUSE`: share completely validated
inputs and union/hash results within manager initialization and reuse the first
feature result where equivalent. Keep source records for both roles, all row
checks, measurement and topology separation, and existing estimator hashes.
`test_step5_attention_input_union.py:107-148` explicitly changes the second
source and then its shape policy on the same manager. These supported changes
must still invalidate the corresponding identity. Source content fingerprints
remain authoritative; a path-only or mtime-only reuse key is insufficient.
Release retained frames at initialization completion or failure. Runtime query
reuse and loader reuse have separate file ownership.

Parent acceptance:187 input/bundle/manager checks plus19 CPU torch-helper checks
pass. The unchanged probe returns GREEN, DECODE0 loads/0.028414s and warm
PREFILL2.574255s. Exact frame/attrs/JSON and all four archived root/count hashes
match. Independent source review returns No findings. Detailed commands,
reproduction setup corrections and evidence are in
`test_report_2026-10-22_cpu_reuse.md` and `load_parent_receipt.json`.

## Fixed-contract parent verification

Target `ee28bf7`, records `7834dd5`, in `step5-mrv2-perf`. Parent command:

```bash
timeout 60s env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-mrv2-perf PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_cpu.py --mode ffn --layers 3 88 --iterations 2 --no-profile --trace --output /data/ycfeng/tmp/d23_parent_contract_verify01 > /data/ycfeng/tmp/d23_parent_contract_verify01.log 2>&1
```

Observed: 64 lane queries take 0.073913s versus the parent's 0.532956s baseline,
a 7.210579x improvement. Every corresponding cost dictionary and artifact
descriptor is exactly equal. Source truth passes. Enabled trace retains 456
events, 344788 bytes and passing persisted proof in 0.069498s. The unchanged
maximum-query latency gate is still RED; this is not full CLI acceptance.

The parent also reproduced both reported structural failures on the unchanged
parent core before merging this optimization. Both fail at
`replica_stage_schedule_event.py:81` because the old test's `SimpleNamespace`
lacks `take_forward_start_batch`, before any cost lookup. Evidence:
`/data/ycfeng/tmp/d23_contract_baseline_structural01.{log,xml}`. The combined parent
run includes these tests: 255 PASS, 2 FAIL, zero errors/skips in 108.370s. Failure
IDs and messages exactly equal the two baseline failures. The combined run is
not described as all-pass; the optimization introduces no new failure in it.

Independent reviewer `ses_e9f8b6cffffeSZiBfBvZkmXBQa` inspected the complete
`74980bb..ee28bf7` diff and returned **No findings** for the source phase. It
confirmed predictor-instance lifetime, fixed model/role/TP/EP inputs, dynamic
registry/family selection, fresh batch features and nested context restoration.
No estimator or prediction output is cached by this change. Parent cost/artifact
equality and final test counts above were independently executed by the parent;
the reviewer did not rerun them. Short-loop performance and source acceptance
do not establish full CLI completion or native accuracy.

Full offline/online completion, READ/readiness/release, final metrics and native
qualification remain unverified. The full execution bound remains2400s.
