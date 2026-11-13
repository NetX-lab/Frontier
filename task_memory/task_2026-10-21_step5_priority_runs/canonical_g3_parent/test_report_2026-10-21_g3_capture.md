# Parent verification of canonical G3 capture and abort

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record immutable-snapshot parent verification, reference selection and outstanding native work. |

## Scope and environment

Task: D21-CANONICAL-G3. Code7d46abe, records e14e7df. Parent created detached worktree `/data/ycfeng/Frontier/.worktrees/step5-g3-review-e14e7df` at e14e7df to verify delivered bytes while the owner continued separately. Linux; `/data/ycfeng/envs/asyncopd-py312/bin/python` reports Python3.12.13. Simulator subprocesses use the existing `/data/ycfeng/envs/frontier-py310/bin/python`. No dependency or native-runtime changes, GPU execution, sealing or submission.

## Exact command

Working directory: `/data/ycfeng/Frontier/.worktrees/step5-g3-review-e14e7df`.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp \
  STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm \
  FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python \
  /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
  tests/unit/test_step5_canonical_g3.py \
  tests/unit/test_step5_native_pd_r2_stage.py \
  tests/unit/test_step5_native_pd_r2_observation.py \
  tests/unit/test_step5_native_pd_r2_verdict.py \
  tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py \
  tests/unit/test_calibration_pd_harness.py \
  /data/ycfeng/Frontier/.worktrees/step5-r2-correction/tests/unit/test_step5_native_pd_r2_trace_replay.py \
  tests/unit/test_canonical_client_metrics.py tests/unit/test_step5_client_metrics.py \
  --import-mode=importlib -q --tb=short -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d21_g3_parent01 \
  --junitxml=/data/ycfeng/tmp/d21_g3_parent01.xml \
  > /data/ycfeng/tmp/d21_g3_parent01.log 2>&1
```

## Observed validation

| Check | Passing condition | Observation | Result |
| --- | --- | --- | --- |
| Capture/caller/R2/source replay and shared native/metrics regressions | All selected cases pass |658 passed in42.08s | PASS |
| Earlier optional metrics setup failure | Execute the selected cases using the intended simulator interpreter | Same command includes both metrics modules and has no setup failures | PASS |
| Native G3 and complete reference | Actual independent state/identity/lifetime evidence | Not executed; implementation incomplete | UNVERIFIED |

The previous owner's631-pass/27-setup-error record remains unchanged. `ModuleNotFoundError: No module named 'plotly'` came from simulator subprocesses using the native-client interpreter. Setting the existing interpreter-selection environment variable resolves the configuration mismatch; nothing was installed and no tests were skipped or suppressed. No aggregate memory measurement was taken by the parent.

## Parent source review

Reviewer: primary assistant; no independent-review claim. Inspected the13-file changed list from `git diff b6755f7..e14e7df`, the new canonical definition/caller, shared proxy/stream changes, stage builder/verifier, abort observation/consumer, canonical receipt checks and fault tests. Key paths:

- `tests/integration/step5_pd/r2_canonical.py`: frozen P0 shape/config and original prompt-template preparation; no complete-package claim.
- `tests/comparison/calibration/pd_proxy.py:152–172` and `tests/comparison/dp_placement_pp/vllm_replay.py`: opt-in diagnostic cancellation, preserving ordinary streaming behavior.
- `tests/integration/step5_pd/r2_observation.py:972–1022`: pre-abort status observation without consuming the native request iterable; original callable executes once with auxiliary callbacks isolated.
- `tests/integration/step5_pd/r2_verdict.py:909–955,1133–1165,1282–1424`: actual pending state, same-host READ overlap, all-rank readiness, source notification/free and pool conservation; exact normal-request token accounting and explicit missing-evidence outcomes.
- `tests/unit/test_step5_canonical_g3.py:150–303`: actual localhost HTTP path, deadline-only/late/early-free/missing-rank/pool-leak/fence controls and iterable preservation.

No actionable blocker found in this partial capture/abort scope. This review does not accept the unfinished independent reference, pressure/reset/reuse controls or native observations. The deadline merely attempts overlap and never proves it. The source is not merged at this checkpoint.

## Parent reference disposition

The locked `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:34–106` returns zero external tokens when transfer metadata is absent. Primary spec section6 limits conditionalD-S5-REF to changed image, native functional repair or connector. Selecting an independent request on the existing D engine therefore reuses the locked native path under current ordinary implementation authority; it is not a new user approval or reference substitution.

Required next implementation: full resident parameter/buffer/scales identity through existing chunk primitives; independent6144-local/zero-external/1024-output execution after idle; actual state/output checks; explicit reset/reuse/pressure component and alias conservation. Retain existing numerical comparison contracts and bit-exact wire checks. Missing contracts/evidence remain unverified; no tolerance may be invented. Owner resumed in `ses_ea19abd75ffeb7vtGDx4XMF94y`; primary retains platform ownership.
