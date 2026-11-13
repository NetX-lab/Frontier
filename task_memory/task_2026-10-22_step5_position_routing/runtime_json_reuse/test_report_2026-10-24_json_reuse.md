# D29 current-query JSON reuse verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete four protected-output arms and24 balanced timing samples; archive the corrected auxiliary-header check. |

## Source and environment

Frozen source: `dbe19f531ff68d9edafc480418e8927c40eb1c60` at
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-08`;23 source bindings
match before/after. Protected body SHA256:
`1cabc04f024e6e1c8ee8d9902e74fee6e5dff4d95faeeaab0f8902e67d6d8555`.
Python3.10.6: `/data/ycfeng/envs/frontier-py310/bin/python`. All math thread
variables are1, Python bytecode writes disabled. Source/fixture caches are not
edited; candidate methods are explicitly installed from two scratch copies.

## Commands

Preparation, from `/data/ycfeng/Frontier/.worktrees/step5-preview`:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=1G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/runtime_json_reuse/prepare.py /data/ycfeng/tmp/d29_runtime_json_reuse01
```

The following exact launcher was used for each row below, with the indicated
script/arguments and log path. The common raw root is
`/data/ycfeng/tmp/d29_runtime_json_reuse01`.

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_runtime_json_reuse01/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_runtime_json_reuse01/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_runtime_json_reuse01/cache/mpl NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_runtime_json_reuse01/cache/numba /data/ycfeng/envs/frontier-py310/bin/python -B SCRIPT ARGUMENTS > LOG 2>&1
```

| Invocation | SCRIPT / ARGUMENTS | LOG under raw root | Expected / actual |
| --- | --- | --- | --- |
| Baseline | `/data/ycfeng/tmp/d29_runtime_json_reuse01/registry/run.py baseline` | `baseline.log` | Protected208-event output / PASS |
| Registry | `/data/ycfeng/tmp/d29_runtime_json_reuse01/registry/run.py candidate` | `registry.log` |160 fewer loads, same output / PASS |
| Point | `/data/ycfeng/tmp/d29_runtime_json_reuse01/point/run.py candidate` | `point.log` |160 fewer dumps, same output / PASS |
| Combined | `/data/ycfeng/tmp/d29_runtime_json_reuse01/combined/run.py candidate` | `combined.log` | Both reductions, same output / PASS |
| Balanced timing | `task_memory/task_2026-10-22_step5_position_routing/runtime_json_reuse/paired_timing.py /data/ycfeng/tmp/d29_runtime_json_reuse01` | `paired.log` |24 full-pricing payload matches, GC enabled / PASS |

Final archive check:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=1G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/runtime_json_reuse/verify.py /data/ycfeng/tmp/d29_runtime_json_reuse01 > /data/ycfeng/tmp/d29_runtime_json_reuse01/verify02.log 2>&1
```

## Observed results

| Arm | loads / dumps | Same-process wall median ms | Wall mean ms | Mean GC ms |
| --- | --- | --- | --- | --- |
| Baseline |1136 /704 |90.982 |101.850 |2.115 |
| Registry |976 /704 |89.274 |91.365 |2.014 |
| Point |1136 /544 |87.092 |104.933 |18.691 |
| Combined |976 /544 |88.348 |124.540 |39.242 |

Six samples per arm,24 total; no exclusions, GC disablement or GC subtraction.
CPU medians90.976/89.269/87.087/88.341ms closely track wall medians. Registry
median saves1.707ms, point3.889ms and combined2.634ms. Independent-process warm
medians207.332/86.672/87.205/161.763ms are inconsistent and are not accepted
speedup numbers. Paired GC callbacks show placement-sensitive pauses, but do not
uniquely explain the earlier independent-arm spread.

Every arm preserves288 numerical lookups,144 fresh proofs,64 collective prices,
16 actual points and192 live key calls. Canonical JSON calls fall624→464 only
for the point/combined arms.19 per-candidate exact comparisons include15 complete
208-event JSONLs, two pinned auxiliary trace headers, ledger and pricing payload.
Pricing SHA256 is `3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`;
whole canonical JSONL `1850a9b6a90cc7bb087b881e72d8cc00f5a669fa0be0f54bb25f3361a38d7e11`;
ledger `006dc7e8f76e2c8d2f59a42d3163aae75d474a4149eb08a3e85d688e948482bf`.
Snapshot/ownership checks and64 wrong-key rejections pass.

### Archive-checker failure and correction

`verify.log` retains `AssertionError: ('registry', 'late-json-failure/op_traces.jsonl')`.
The first checker treated all19 trace files as clock-pinned. The original protected
body pins emission headers at lines27-30, but lines123-150 initialize the profile
and late-error stores with real time. The failure assertion intentionally compares
only the exact3902-byte event prefix; the profile has no flushed event body.

The corrected checker retains each raw header/full-file hash, asserts all other
header fields equal, compares the failure bodies exactly and requires profile
bodies to be empty.3902-byte prefix SHA256 is
`f5e7f5a67660096a71d14abfe690dbc432d09a480c85c9999dea0cd16979f1ec` on all four arms.
All canonical full files remain strict byte comparisons. `verify02.log` passes;
no original artifact, protected body, production source or probe run was changed.

Final record checks: five diagnostic scripts parse; all seven retained artifact
hashes match. Full staged `git diff --cached --check` exits2 only at the raw
`candidate_source.diff:29,47` blank-context markers required by unified-diff
syntax. The artifact bytes/hash are preserved. The remaining staged task changes
pass this command:

```sh
git diff --cached --check -- . ':(exclude)task_memory/task_2026-10-22_step5_position_routing/runtime_json_reuse/evidence/candidate_source.diff'
```

## Disposition and limits

**NO_PRODUCTION_ALLOCATION.** The tested controls remove real repeated codec work
but provide small/noisy marginal savings; the point control also expands an
existing private method signature. They do not address the large full-workflow
gap. No extended current-model replacement/error matrix was run. Only the
canonical FP8 DECODE1/layer60/16EP scenario and protected negatives are verified;
no wholeCLI/native/GPU/G3/E2E acceptance follows. Scope remains9565 source and9579
global; current-model validation, full output and6144/1024/2400s remain required.
