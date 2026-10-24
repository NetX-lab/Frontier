# Required Shape Attempts

## Modification History

| Date | Change |
| --- | --- |
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
- Exact clean frozen source is`dbe19f53`;23 source hashes and262 fixture size/mtime
  bindings pass both preflight and post-failure guards. No source/fixture mutation
  was observed. Mainline native-key integration remains`c668a938/886608ea` and
  cumulative source9566/global9580; these CPU checks do not re-audit that union.
- No projected CLI speedup is inferred from profiled cold pricing. The per-mode
  2400s full6144/1024 CLI acceptance, native admission/runtime, G3 and E2E remain open.
- A future retry requires an explicit decision about the exhausted invocation
  ceiling. Preparing or passing this checker control does not grant that authority.
