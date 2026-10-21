# D21-CANONICAL-G3 native diagnostic implementation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archived runnable capture/abort code and explicit incomplete G3 boundary. |
| 2026-10-21 | Parent completed reference/controls/MRv2 census implementation at1171ab6;685 broad and69 final checks passed. |
| 2026-10-22 | Fresh P0 native capture01 verified and submitted; native judgment pending. |

## Outcome

Case-aligned capture, independent full-local D reference, complete resident/cache comparison, reset/reuse/pressure controls and MRv2 useful/dummy forward accounting are implemented at `1171ab6`. Corrected c0a0e63 R2 retention checks remain. Fresh49-file/50-source-pin capture01 is submitted as `exp-1021-235955-532667`,32×H800/codesign/non-preemptible; initial platform identity/mount/resource checks pass. **Native G3/G4 are not qualified.** The reference uses the existing D engine without transfer metadata; no runtime replacement or numerical waiver applies.

## Delivered code

Code commits: initial partial `7d46abe`; completed diagnostic implementation `1171ab6`, on feat/step5-canonical-g3.

- `tests/integration/step5_pd/r2_canonical.{py,json}`: current P0 definition,163 diagnostic requests including three independent references and128 pressure-attempt requests, original canonical warmup/formal prompt IDs, exact native output-ID caller, immutable7168/8192/128/topology/API settings. `definition()` plus existing `build_stage.build()` is the capture path; unit tests drive it end-to-end.
- `tests/integration/step5_pd/r2_stage.py`, `verify_r2_stage.py`: preserve corrected aliases/schema/source bindings; support canonical capture without short-run1024/512 block overrides; standalone stage verifier never calls this G3 qualified.
- `tests/integration/step5_pd/r2_observation.py`, `r2_verdict.py`: restored strict P1 retention/token logic, native pre-abort status without consuming native iterables, full exact canonical client receipts and a separate native pending-READ abort conservation consumer. Normal completion/content/retention checks are not weakened.
- Necessary opt-in caller seams only: `tests/comparison/calibration/pd_proxy.py`, `tests/comparison/dp_placement_pp/vllm_replay.py`, `tests/integration/step5_pd/build_stage.py`. Proxy-native-D-client deadline is a diagnostic attempt; actual native overlap/cleanup must still be proven. Normal clean/native requests do not carry its header.
- `tests/unit/test_step5_canonical_g3.py` and corrected existing R2 unit tests:53 new focused cases; final combined612 PASS. Retained native source/trace retirement replay tests also executed against current imports, without copying their older canonical scope.
- `tests/integration/step5_pd/r2_reference.py`, `r2_controls.py`, `r2_replay.py`: complete resident view hashing, logical native checkpoints, component/alias/reset/pressure checks and direct-D HTTP references. Failed request validation is retained and subsequent healthy waves continue.
- `tests/integration/step5_pd/r2_runtime.py`: source-pinned MRv2 post-DP descriptors, query/context arrays, attention-metadata/slot/table shapes and paired native host forward records. All32 ranks and useful scheduler/model inputs must reconcile; there are no observer GPU reads/fences in this census.
- `tests/unit/test_step5_native_reference.py`, `test_step5_native_forward_census.py`: real CPU Torch state/alias tests, native host-method execution, real local HTTP routing and fault-injection controls. Primary broad685 PASS and final69 PASS; no native qualification claim.

Decisive caller/source anchors: `r2_canonical.py:20,50,87` (case freeze and actual trace preparation); `pd_proxy.py:152` (opt-in D-client cancellation); `r2_observation.py:972` (native pre-abort snapshot); `r2_verdict.py:909,1133` (pending-READ conservation and exact native client counts). Native/reference feasibility anchors are in `reference_decision.md`.

## Runnable offline preparation (not sealing or submission)

```bash
PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B \
  tests/integration/step5_pd/r2_canonical.py \
  --source-root /data/ycfeng/tmp/step5_profiling_20261020/image_src
```

Historical partial output: `capture_plan.json`, retained unchanged. The current command reports implemented diagnostics and pending committed-stage verification. Resource ceiling remains4x8 H800/codesign, non-preemptible,5400s/48 GPU-hours. A launch command will be recorded only after the new stage is sealed and verified.

## Validation and evidence

Exact command/limitations: `test_report_2026-10-21_canonical_g3.md`; final measured command: `cpu_native_verification.json` (612 PASS,28.345s,max childRSS779000KiB). Optional broader check: `cpu_verification.json` (631 PASS/27 setup errors,missing plotly); not hidden or counted as passing. `receipt.json` binds used local source/test evidence, frozen native source and canonical input hashes.

Current primary evidence: `test_report_2026-10-21_parent_reference.md`, `/data/ycfeng/tmp/d21_g3_complete02.{log,xml}` and `/data/ycfeng/tmp/d21_g3_pins_final01.{log,xml}`. The685-case run uses the existing simulator interpreter for metrics subprocesses and has no skips/errors.

## Remaining work

See `reference_decision.md` for the selected existing-D full-local reference and unchanged numerical limits. Native resident identity, state/output equivalence and lifecycle outcomes still require actual evidence.

Inspect the submitted run and judge its native evidence, as specified in `native_run.md`. Neither the128-request pressure attempt nor the frozen10ms abort deadline establishes success without actual allocator/READ/cleanup witnesses. Old maxlen2048 R2 packages remain unchanged. Formal timing and the four10% gates remain open.
