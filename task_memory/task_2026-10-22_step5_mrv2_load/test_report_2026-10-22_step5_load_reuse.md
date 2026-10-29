# D23-MRV2-LOAD-REUSE Test Report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record RED/GREEN, identity preservation, lifetime checks and fast profile. |

## Result
Source/tests commit: `02f566a`; task evidence is a separate follow-up commit.

PASS for the authorized CPU loader/union implementation and focused checks.
The unchanged diagnostic now returns `GREEN_NO_REPEATED_LOAD`, exit0. No
production full initialization, full CLI, GPU, platform or77-case acceptance was
run; these remain parent-owned. Four native profiling tests are unavailable
because this CPU environment does not contain torch. No dependency was installed.

## Source change and boundary reason
`frontier/execution_time_predictor/step5.py` owns a current validated union per
measurement on the existing manager. Every lookup fingerprints each distinct
resolved source once, even on reuse; keys include the complete public serialized
model config, physical topology, shape policy, measurement and source bytes.
An identical source/context is validated once inside the union, while both role
records remain in `step5_attention_inputs` and measurement identity. Changed
content/context replaces only that measurement's current entry; no history or
persistent cache is added. Invalid changed content still fails before fitting.

The input hash calls the original `_get_model_hash` on a miss and retains that
unchanged result only for the current frame and the existing hash-relevant fit
configuration. The original model hash implementation and estimator cache naming
are untouched. Canonical fit config is selected from the same sorted first role
on every lookup, so mutated/replaced configs are not captured as stale objects.
The first validated row feature dictionary is retained, removing only the
duplicate second derivation call, not its validation or feature values.

`prediction_family_trainers.py` changes only the existing Step root-signature
call. `shared_prediction_model_manager.py` adds `finally` cleanup around existing
train/publication initialization. Three weakref tests cover success, training
exception and publication exception. Partial-manager APIs need no flags or
alternate path; their entries are bounded to current measurements. Step627,
trainer1849 and manager703 lines remain below2,000; no size exception is needed.
No source consumer mutates union/attrs in-place: training/registry read them;
NaN filtering creates a local copy. No change to `step5_cost.py`, sklearn MoE,
quant_signature representation, fixture/cache source or numerical/coverage rules.

## Environment and commands
Working directory `/data/ycfeng/Frontier/.worktrees/step5-mrv2-load`.
Python3.10.6 from `/data/ycfeng/envs/frontier-py310/bin/python`, numpy2.2.6,
pandas2.3.3, sklearn1.7.2. CPU only; OMP/OpenBLAS threads1, no bytecode writes.
Use new scratch/basetemp directories for repeat execution.

RED before production changes:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q tests/unit/test_step5_attention_input_union.py -k same_manager_roles --basetemp=/data/ycfeng/tmp/d23_load_reuse_red01 > /data/ycfeng/tmp/d23_load_reuse_red01.log 2>&1
```
Expected/observed2 failures: distinct second-role lookup still loads2; shared
source first lookup loads2 instead of1. After union correction the same selector
with green01 paths passes2 in2.63s. Feature RED before feature correction:
same pytest command for `test_step5_measured_training.py -k
derives_validated_features_once`, feature_red01 paths; observed `[0,3,0,3]`
instead of `[0,3]`. It passes after retaining the original feature dictionaries.

Final focused checks:
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q tests/unit/test_step5_attention_input_union.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_mrv2_inputs.py --basetemp=/data/ycfeng/tmp/d23_load_reuse_tests03 > /data/ycfeng/tmp/d23_load_reuse_tests03.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q tests/unit/test_step5_request_free_profiling.py -k 'not native_forward and not native_metadata and not variable_decode_prefix and not rejects_any_cache_mutation' --basetemp=/data/ycfeng/tmp/d23_load_reuse_request_free03 > /data/ycfeng/tmp/d23_load_reuse_request_free03.log 2>&1
```
Observed93 PASS in28.65s and15 PASS/4 deselected in2.03s. The earlier four-file
check produced104 PASS and4 failures, all `ModuleNotFoundError: No module named
'torch'` in native profiling tests; log `d23_load_reuse_tests02.log` retained.
No suppression or source changes to those tests.

Unchanged fast profile script and flags (only new output directory):
```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/profile_step5_load.py --config /data/ycfeng/tmp/d23_main_actual05_offline/offline/metrics/stepfun_ai_step_5_preview/offline_batch/step5_measured_cpu_main_offline/config.json --output /data/ycfeng/tmp/d23_mrv2_load_reuse_probe01 --rows 100 --event-cache-hash 23b7026a --assert-no-repeated-loads > /data/ycfeng/tmp/d23_mrv2_load_reuse_probe01.log 2>&1
```
Expected/observedGREEN, exit0, zero estimator fit, original config/source truth
and cache SHA256 preservation PASS. Script SHA256 remains
`9321b35820325953759b6b457ffc15716824f8b1ec098fe1e542f97159754414`.

## Measurements: actual API, not full initialization
Baseline is diagnostic04 before fix; parent independently measured its own RED.
Times below are cProfile wall except the explicitly plain event loader.

| Boundary | Before | After | Observed call change |
| --- | ---: | ---: | --- |
| plain event100 loader |0.255889s|0.232461s|feature200 ->100 |
| event100 role union |1.205208s|0.555073s|loader2 ->1 |
| warm KERNEL_ONLY PREFILL |4.986517s|2.666841s|loader2 ->1, cache hits2, fit0 |
| already-trained KERNEL_ONLY DECODE |4.611782s|0.027970s|loader2 ->0, hash1 ->0, fit0 |
| existing71.15MB event root read |0.529611s|0.530418s|unchanged cache-read seam |

The after DECODE path still fingerprints the distinct source once; no mtime-only
shortcut. Full original event CSV was not loaded. No statement here treats
earlier747s prefix extrapolation as measured wall time or claims full998/1004s
acceptance. Generic loader/bundle/replica initialization and full E2E remain
unmeasured; actual full speedup must be accepted by the parent.

## Exact frame/attrs/hash and proof checks
Before edits, actual100-row event prefix and full184-row kernel union were saved
under `/data/ycfeng/tmp/d23_load_reuse_baseline01/` using the unchanged union API,
original actual config and unchanged source files. After both corrections:
`pd.testing.assert_frame_equal` PASS, complete `attrs` equality PASS; reversed
role lookup returns the same frame and original hash. No source column/value,
coverage, dtype, index, order, nested scope, input proof or hash changed.

| Source | Root hash before/after | Count hash before/after |
| --- | --- | --- |
| event diagnostic100 |9688c187 |008badcf |
| full KERNEL_ONLY184 |b1bedbbb |664eca75 |

Persistent regressions verify shared/distinct source paths, both declared role
records, per-distinct-source fingerprint counts, second-role source content
change/restore on the same manager, zero-count malformed change, shape/fitconfig
change/restore, changed runtime/backend/measurement/physical_scope rejection,
changed TP/max_model_len/norm_dtype/rope context revalidation, role-order
invariance, nested request/MRv2 scope identity, separate measurement coverage
and weakref lifetime release. Original source/shape tests remain unchanged.

## Durable evidence and limits
- `reuse_profile_receipt.json`: after fast profile counts/times/source truth.
- `reuse_frame_truth.json`: pre/post union frame/attrs/hash proofs.
- `reuse_checkpoint.md`: phase self-check; records/summary updated in-place.
- Raw JSON/pstats/subsets at `/data/ycfeng/tmp/d23_mrv2_load_reuse_probe01/`;
  before frames at `/data/ycfeng/tmp/d23_load_reuse_baseline01/`; pytest logs above.
- `git diff --check` PASS. Parent owns independent final review/merge,
  full offline/online CLI and77-case fidelity gate. The4 torch-dependent native
  checks remain unverified on this interpreter, not fixed or waived.
