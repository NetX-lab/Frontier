# D21-CANONICAL-G3 partial implementation handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archived runnable capture/abort code and explicit incomplete G3 boundary. |

## Outcome

Case-aligned capture/caller and strict pending-READ abort workflow implemented; corrected c0a0e63 R2 retained. **Full G3 NOT implemented or qualified. No real-image sealed package or submit command.** The material independent-reference choice is returned before changing reference runtime semantics. Native reset/reuse/pressure component/alias conservation and an explicit pressure trigger remain unfinished, not waived.

## Delivered code

Code commit: `7d46abe` on feat/step5-canonical-g3.

- `tests/integration/step5_pd/r2_canonical.{py,json}`: current P0 definition,32-request diagnostic controls, original canonical warmup/formal prompt IDs, exact native output-ID caller, immutable7168/8192/128/topology/API settings. `definition()` plus existing `build_stage.build()` is the implemented capture path; unit test drives it end-to-end. CLI prints an explicitly unsealed/blocking capture plan, not a launchable package.
- `tests/integration/step5_pd/r2_stage.py`, `verify_r2_stage.py`: preserve corrected aliases/schema/source bindings; support canonical capture without short-run1024/512 block overrides; standalone stage verifier never calls this G3 qualified.
- `tests/integration/step5_pd/r2_observation.py`, `r2_verdict.py`: restored strict P1 retention/token logic, native pre-abort status without consuming native iterables, full exact canonical client receipts and a separate native pending-READ abort conservation consumer. Normal completion/content/retention checks are not weakened.
- Necessary opt-in caller seams only: `tests/comparison/calibration/pd_proxy.py`, `tests/comparison/dp_placement_pp/vllm_replay.py`, `tests/integration/step5_pd/build_stage.py`. Proxy-native-D-client deadline is a diagnostic attempt; actual native overlap/cleanup must still be proven. Normal clean/native requests do not carry its header.
- `tests/unit/test_step5_canonical_g3.py` and corrected existing R2 unit tests:53 new focused cases; final combined612 PASS. Retained native source/trace retirement replay tests also executed against current imports, without copying their older canonical scope.

Decisive caller/source anchors: `r2_canonical.py:20,50,87` (case freeze and actual trace preparation); `pd_proxy.py:152` (opt-in D-client cancellation); `r2_observation.py:972` (native pre-abort snapshot); `r2_verdict.py:909,1133` (pending-READ conservation and exact native client counts). Native/reference feasibility anchors are in `reference_decision.md`.

## Runnable offline preparation (not sealing or submission)

```bash
PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B \
  tests/integration/step5_pd/r2_canonical.py \
  --source-root /data/ycfeng/tmp/step5_profiling_20261020/image_src
```

Saved output: `task_memory/task_2026-10-21_step5_priority_runs/canonical_g3/capture_plan.json`. It reports sealed=false/submitted=false/ready_for_formal=false and unresolved prerequisites. Proposed inherited capture skeleton budget only:4x8 H800/codesign,5400s, at most48 GPU-hours. It is not approval or a valid submit instruction for incomplete G3.

## Validation and evidence

Exact command/limitations: `test_report_2026-10-21_canonical_g3.md`; final measured command: `cpu_native_verification.json` (612 PASS,28.345s,max childRSS779000KiB). Optional broader check: `cpu_verification.json` (631 PASS/27 setup errors,missing plotly); not hidden or counted as passing. `receipt.json` binds used local source/test evidence, frozen native source and canonical input hashes.

## One parent decision and remaining work

See `reference_decision.md`: recommended existing-D native full-local-prefill execution, no transfer metadata, with complete resident parameter/buffer/scales identity; equal seeds/census are insufficient. No extra instance, synthetic initialization, numeric tolerance or alternative model has been selected.

After that explicit choice, implement the independent execution/state/output comparison and reference-state reset/reuse/capacity-pressure component/alias controls plus explicit pressure trigger. Then generate/verify a fresh complete package and provide the parent-only submit command. Do not reuse short maxlen2048 R2 stage02 or call concurrent8 capacity pressure. Actual native pending-READ overlap may remain unverified if the frozen10ms attempt misses it; do not retry/tune silently or call a timer proof. Parent retains platform ownership.
