# Required Shape Attempts

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Record YC-approved supplemental3/3PASS,total6/6 used,selected cloud archive and timely local cleanup. |
| 2026-10-24 | Record three failed shape attempts, the checker-only correction, preserved evidence, and exhausted invocation allowance. |

## Result

**Shape qualification remains UNVERIFIED.** All three allowed invocations exited1
after production pricing and before emission. Primary's checker incorrectly
required a positive duration for every compatibility phase slot. The first
unselected slot, `add_attn_residual=0.0`, triggered the assertion in all cases.
This is a checker boundary error, not observed failure of the price predictor.

| Case | Layers priced | Numerical / fresh FFN proof / collective / point calls | Raw result | Complete emission/ledger comparison |
| --- | --- | --- | --- | --- |
| P6128 | 60 | 288 /144 /64 /16 | FAIL,exit1 | UNVERIFIED |
| First DECODE16, remaining prompt replay | 60 | 288 /144 /64 /16 | FAIL,exit1 | UNVERIFIED |
| BF16 | 88 only | 256 /128 /64 /16 | FAIL,exit1 | UNVERIFIED;89/90 not reached |

The call counts are read from completed `layer*.prof` files. They establish
which methods ran, not the correctness of all returned values. No complete
payload or JSONL was produced, so no output parity or final event census is claimed.

## Commands and Environment

Scratch: `/data/ycfeng/tmp/d29_required_wave_shapes_20261024`.
Driver: `workload_shape_validation/run_case.py`; parameters: `cases.json`.
Python: `/data/ycfeng/envs/frontier-py310/bin/python -B`,version3.10.6
(`python -B -VV`:GCC11.3.0). Version inspection is not a shape invocation.
No full CSV/trace scan, model training, full initialization, native hardware,
platform operation or full CLI run was performed.

The following exact command pattern ran once per row of the substitution table,
with the shown shell substitutions expanded before execution:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_required_wave_shapes_20261024/cache_${CASE} XDG_CACHE_HOME=/data/ycfeng/tmp/d29_required_wave_shapes_20261024/cache_${CASE} MPLCONFIGDIR=/data/ycfeng/tmp/d29_required_wave_shapes_20261024/cache_${CASE} NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_required_wave_shapes_20261024/cache_${CASE} timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/workload_shape_validation/run_case.py --parameters /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/workload_shape_validation/cases.json --case ${CASE} --output /data/ycfeng/tmp/d29_required_wave_shapes_20261024/${CASE} > /data/ycfeng/tmp/d29_required_wave_shapes_20261024/${CASE}.log 2>&1
```

| Invocation | CASE | Scope recorded in raw log |
| --- | --- | --- |
| 1 | `prefill6128` | `run-rc137a7024cba48d39be4ef651c951c81.scope` |
| 2 | `first_decode16` | `run-rfbcadae509cd443ba055baa2563246de.scope` |
| 3 | `bf16` | `run-r56cf0627bd9b4ba29f4124ce76e1e743.scope` |

Each invocation has the original60s/30GiB/single-math-thread limit. The complete
three-invocation allowance is consumed. **No fourth invocation is authorized or run.**

## Diagnosis and Checker Correction

Frozen `frontier/entities/execution_time.py:955-969` deliberately projects
compatibility slots, including absent operators as0.0. The production emitter
at `frontier/metrics/ep_wave_metrics.py:91-93` does not emit nonpositive slots.
The actual applicable scopes are separately checked for positive, finite,
paired exact evidence at `frontier/execution_time_predictor/moe_operator_times.py:139-141`.

Primary corrected only the task-local checker: skip exactly-zero slots only
when their names have no compute proof or collective plan. A selected zero,
negative or NaN duration still fails. Production behavior and source allocation
are unchanged. Independent static check D29-SHAPE-DRIVER-PREFLIGHT
(`ses_e9217fb19ffe995qZCog0dwXN5`) identified the same blocker; it delivered after
the three attempts and did not prevent their failure.

The AST-extracted checker control passes5/5 cases: unselected-zero skipping,
positive selected retention, selected-zero compute rejection, selected-zero
collective rejection, and negative/NaN rejection (the last two are separate cases
within the five test rows). This is **checker-only evidence**, not a shape rerun.
The first control's AST selector failed on a nested tuple unpack; filtering the
exact three-name target fixed that selector. The retained failed shape driver is
`evidence/failed_run_case.py`,SHA256`d98924fbb6b64f0fe0f6def14e74a7d47a67d36e00275e61600e0caabda89f75`.
The corrected driver isSHA256`e099270ef4c8bfdf36336d4c4d494d8ddecdafbc57003efabb07762f416a1d9e`.

## Evidence and Limits

- `evidence/raw_probe_failures/`:16 byte-preserved raw logs, stage/failure
  receipts, cProfile files, preflight, checker control and post-failure guard.
  `artifact_sha256.json` binds all16 files. The failed driver is preserved separately.
- Under YC's subsequent completed-file cleanup authorization,the16 exact scratch
  duplicates are deleted after archive SHA256 verification. Canonical raw logs,
  stage/failure receipts and profiles remain byte-preserved in the task archive;
  active fixture and driver inputs are not selected. `storage_readiness/cleanup_receipt.json`
  records the298,219B removal;command paths above describe the original runs.

## Explicitly Approved Supplemental Runs

YC selects **追加3次**,one per category,retaining60s/30GiB and no GPU/fullCLI.
Original rawFAIL results above are not waived. Corrected driverSHA256e099270e…
and the original frozen dbe19f53/cached inputs are unchanged.

Working directory:`/data/ycfeng/Frontier/.worktrees/step5-preview`.
Exact common command,with the case/output values from the following table:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/d29_required_wave_shapes_retry_20261024/cache XDG_CACHE_HOME=/data/ycfeng/tmp/d29_required_wave_shapes_retry_20261024/cache MPLCONFIGDIR=/data/ycfeng/tmp/d29_required_wave_shapes_retry_20261024/cache NUMBA_CACHE_DIR=/data/ycfeng/tmp/d29_required_wave_shapes_retry_20261024/cache timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-22_step5_position_routing/workload_shape_validation/run_case.py --parameters task_memory/task_2026-10-22_step5_position_routing/workload_shape_validation/cases.json --case CASE --output /data/ycfeng/tmp/d29_required_wave_shapes_retry_20261024/CASE > /data/ycfeng/tmp/d29_required_wave_shapes_retry_20261024/CASE.log 2>&1
```

P6128 finishes and is inspected before the independent D16/BF16 commands run
concurrently. Python3.10.6/GCC11.3.0;all observed RLIMIT_AS values are
32212254720B/32212254720B. Profiler times are not a speedup benchmark.

| CASE | Logical/physical rows | Layers | Events | Ledger lanes | Exit/verdict |
| --- | --- | --- | ---: | ---: | --- |
| prefill6128 | 6128/6129 | 60 | 208 | 16 | 0/PASS |
| first_decode16 | 16/32 | 60 | 208 | 16 | 0/PASS |
| bf16 | 1/2 | 88,89,90 | 576(192each) | 48 | 0/PASS |

All992 emitted events match actual lane durations,operator/layer/EP/phase order,
complete compute proofs,full collective plans/costs/keys and80lane-ledger rows.
Each FP8 wave retains288numerical/144fresh-proof/64collective/16point calls.
Each BF16 wave retains256/128/64/16,quant_dtype=None and no input-quantize event.
All23 frozen source hashes and262 fixture metadata bindings pass before/after.
The bound scratch collective JSON is intentionally replaced with`b'{'` by the
retained fixture adapter after successful run-fixed input binding;actual pricing
continues using the validated fixed bytes,not that negative-control file.

Whole JSONL observations include the actual header:

| CASE | Bytes | SHA256 |
| --- | ---: | --- |
| prefill6128 | 2904335 | a00b81e4d47f4d700853fef9c850ffa7e7a2a8ea9c75b2bd670bcddab0d673da |
| first_decode16 | 2897050 | b6669453ff5817d361fd17cb4e90d2ef391801eebb4077de271f2fdc0da8a11e |
| bf16 | 7913396 | 7b0a0c65a434ac5f15ccb0c693252c0a78ec2161d655137ea7ad60a94c186c3c |

Original3FAIL plus supplemental3PASS consume the approved6-call ceiling.
There are0 remaining invocations. Stdlib postrun/archival checks invoke no
production method. Their missing-whitespace SyntaxError and mistaken identical-
fixture assumption both occur outside production,before artifact deletion;the
exact errors and causal fixes are recorded in evidence/supplemental/.

## Canonical Cloud Evidence and Cleanup

The23 selected files retain all complete JSONL,pricing payloads,ledger,result/
stage receipts,profiles and logs. The uncompressed tar is40,581,120B,SHA256
566e9327bbdf71df15bbd11eecaef4ea10e48ea791aa71ab7bc62dafce84ecdd;all member
hashes are verified locally,then the full uploaded tar hash is checked remotely.
Canonical worker path:

`/mnt/codesign-exp/ycfeng/frontier/step5_position_routing/task_2026-10-22_step5_position_routing/required_wave_shapes_20261024/selected_shapes_c9c79259cb524e38957612882efb458d.tar`

On shai-ycfeng this is the documented`/mnt/ws-jfs/ycfeng` alias. Only personal
storage access is used;no RJob/GPU control or CPU-host cloud mount occurs.
The cloud archive is rehashed and a verified`.tar.complete.json` sidecar is
written/read back before acceptance,as required for completed large cloud files.
27 declared completed local duplicates/archive/one-byte scratch negative controls
are removed after verification,81,114,685B total. Canonical cloud evidence and
task-local result/log summaries/manifests remain;active fixture/source/scripts
and other-session work are unchanged.

## Qualification Limits

PASS is the observed cachedCPU-fixture production path:layer60 P6128/D16 and
ordinaryD1 BF16layers88-90. It is not all-layer/all-token-shape,real native
hardware,full6144/1024 CLI timing or G3/E2E qualification. P/BF16 andD16/BF16
combinations are not observed. Existing lane-builder cluster label remains
DECODE as noted above;gathered routing geometry and actual role are inspected.
The expected18,712,400 full-run events remain an expectation,not a full census.
- Exact clean frozen source is`dbe19f53`;23 source hashes and262 fixture size/mtime
  bindings pass both preflight and post-failure guards. No source/fixture mutation
  was observed. Mainline native-key integration remains`c668a938/886608ea` and
  cumulative source9566/global9580; these CPU checks do not re-audit that union.
- No projected CLI speedup is inferred from profiled cold pricing. The per-mode
  2400s full6144/1024 CLI acceptance, native admission/runtime, G3 and E2E remain open.
- A future retry requires an explicit decision about the exhausted invocation
  ceiling. Preparing or passing this checker control does not grant that authority.
