# Step measured compute CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record the actual shared fit/cache/bundle/cost minimum and scoped existing regressions. |

## Environment and command

Python3.10.6,pandas2.3.3,sklearn1.7.2. Worktree `/data/ycfeng/Frontier/.worktrees/step5-measured-compute`,base `6db2250211a3cf312d17138abdc0cbdcb92449ce`. No Torch/GPU/platform/auth operations were required. Primary pinned collective-sim Python dependency is read through PYTHONPATH;no other-model dataset was copied.

```bash
env PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-measured-compute:/data/ycfeng/Frontier/.worktrees/step5-measured-compute/tests/unit:/data/ycfeng/Frontier/.worktrees/step5-preview/frontier/cc_backend/backends/collective-sim/python \
 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
 /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
 tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_training.py \
 tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_replay.py \
 tests/unit/test_kernel_gap_pricing.py tests/unit/test_on_demand_prediction_contract.py \
 tests/unit/test_execution_time_op_times.py \
 -q -p no:cacheprovider --tb=short \
 --junitxml=/data/ycfeng/tmp/d20_compute_minimum03.xml
```

## Observed results

| Check | Expectation | Actual | Verdict |
| --- | --- | --- | --- |
| New cost/training/bundle/replay tests | All pass | 6+24+45+3=78PASS | PASS |
| Existing selected regression cases | No new failure | 84PASS,1 conflicting baseline assertion | LIMITED |
| Actual CPU fitting | Real sklearn and existing shared atomic cache | Duration/count fitted and reloaded for SWA+DSA | PASS |
| Root ownership | Root once,not child sum | eager fixture1.016ms+23x0.001ms=1.039ms;FULL fixture1.001ms,no extra gap | PASS |
| Serialized proof | Preserve real modality/count/provenance | ExecutionTime copy retains typed cost and23 launches | PASS |
| Cache | Immutable cold/warm and A->B->A | Existing pickle files and bundle bytes/mtime reused | PASS |
| Missing evidence | Explicit error,not replacement values | Missing counts,shape,physical-width and kernel-gap cases rejected | PASS |
| Source coverage binding | All92 source layers resolvable | 69SWA/23DSA typed CPU fixture rows loaded/fitted | PASS |

Combined command:162PASS,1FAIL in24.01s. The failed `test_every_collective_launches_one_kernel_by_default` asserts an absent `OperatorSpec.kernel_count` attribute. `frontier/operators/spec.py`,`frontier/operators/families.py` and the test file are byte-identical to6db2250 (`git diff --exit-code` PASS). `minimum_verification01.json` records hashes and exact test IDs. This is source-equality evidence of the existing assertion conflict,not a claimed execution of the full baseline suite. No unrelated collective default was added and no failure was silently skipped.

## Feedback loop and limits

The first full replay test was RED because producer fixtures allocated live-length258/257-column tables while runtime asked for configured512 columns;decode score_regions differed accordingly. The source coverage guard correctly rejected those physical shapes. The fix was confined to the positive labelled fixture input:near-limit valid lengths allocate512 columns already,with region-aligned DSA decode history. The4104-context mismatch remains a negative test. Focused replay rerun:3PASS in5.33s,XML `/data/ycfeng/tmp/d20_compute_replay02.xml`;the subsequent combined command includes all3 passing cases.

All values above are labelled CPU fixtures,not admitted profiling. The normal full shared-manager constructor,complete FFN/input-normalization coverage and automatic full predictor bundle integration are not yet verified in this first step. No canonical serving,original native strict-repro recovery or any four10% gate is claimed. Parent owns77-case fidelity/integration;this child did not run a redundant full matrix.
