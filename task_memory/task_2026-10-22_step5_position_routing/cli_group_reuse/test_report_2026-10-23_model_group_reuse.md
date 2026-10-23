# D29 fixed-model group reuse verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual dual-consumer RED/GREEN, committed pricing and exact scope. |

Task ID: **D29-MODEL-GROUP-REUSE**.
Code: **3763e5b87a845eac186cfd481f0b56cffb2dd045**.
Baseline: e41abc073500d61f224eda4fca1123e74e40544d.
Submodule:3dac53c544e546574cb6e8078cd14053d0916113, unchanged and isolated.

## Environment

Workdir `/data/ycfeng/Frontier/.worktrees/step5-d29-group-reuse`.
Python `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6; NumPy2.2.6,
pandas2.3.3, sklearn1.7.2. All actual runs used Python `-B`, single OMP/OpenBLAS
threads, scratch TMPDIR and current worktree PYTHONPATH. Probes keep the inherited
30GiB RLIMIT_AS and outer60s timeout. All actual CLI output/cache fields relocate
before config parse. Retained main07/fixture03 and frozen integrated05 are read
only; no full CLI run or external dependency is used.

## Real regression, RED before production edit

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py::test_fixed_predictor_groups_reused_by_queries_and_binding_selection > /data/ycfeng/tmp/d29_group_reuse_red01.log 2>&1
```

Actual result: **1FAIL**, exit1. The first predictor resolved actual groups24
times across repeated layer3/88 query/model-selection/cost and negative calls;
expected1. The failure occurs after real producer, shared trainer, paired models,
query/plan recovery, selected FP32 gate precision, actual source/family negatives,
positive exact duration/count and snapshot equality. This is not a memoization
utility test. Raw log is retained as red.log and at the scratch path.

After the source fix, same focused command with green01 log: **1PASS14.64s**.
The final regression adds foreign source-point and paired-count fault injection,
retains those existing assertions and runs three independent predictors: two
share the same model object, the third has a distinct deep-copied formal model.
Each must call the actual resolver once and receive its own tuple. No global
cache could satisfy that check. The exact group facts are equal for equal model
configurations, but the returned tuples are not shared.

## Full affected selection

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_prediction_measurement_bindings.py tests/unit/test_step5_measured_ffn_replay.py tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_position_routing.py tests/unit/test_step5_measured_training.py > /data/ycfeng/tmp/d29_group_reuse_affected01.log 2>&1
```

**81PASS,1SKIP,0FAIL**,166.49s. The skip is the existing CPU tensor case because
this environment has no torch. The affected selection includes formal source/
constructor behavior, both-trainer/preflight/cache coverage and real shared
consumer/exact-plan tests. Full-unit/strict77 are not delegated or run.

## Unchanged actual pricing loop on committed candidate

Copied the retained diagnosis script (original SHA256
bba92e780743cea23b5c92b852e56c7cabe9867e8c4016b970fc9f599344852e) to a separate
scratch file. Initial changes only rebind source directory, enforced commit and
receipt source_commit to3763e5b8. See probe_source_pin.diff; no point/model/setup
selection changes, no reuse option passed.

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_group_reuse_runtime_loop.py --output /data/ycfeng/tmp/d29_group_reuse_pricing01 --iterations 2 --execution --no-profile > /data/ycfeng/tmp/d29_group_reuse_pricing01.log 2>&1
```

**GREEN,exit0**,cold0.9450197029s,warm0.3690549191s,88 layer descriptions total.
Original main07 point P6128+idle physical1,layer3/EP0,actual formal parsed config,
real Cluster/backend and verified paired bundle remain unchanged. All MoE and
communication cost/proof/plan-key payloads still hash to
**d2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952**.

Another fresh process pricing02 measured0.9965638448/0.3579018088s,88 total, same
hash. Its environment command accidentally spelled PYTHONDONTWRITEBYTECODE as
PYTHONDONTWRITECODE, but Python `-B` still prohibited bytecode; no behavior or
input changes. The fully explicit command above and final counter run use the
correct variable.

An observation-only per-iteration counter was then added to the scratch script;
probe_complete.diff contains every change. Same command with output/log suffix
pricing_count01: **0.9194535781/0.3690149621s**,description counts **88/0**,same
payload hash. These measured cold/warm counts prove the candidate resolves once
inside its first call rather than relying on the diagnostic external reuse.
The script's nonprofiled group_resolutions0 field is an unavailable pstats count,
not evidence of no cold resolution; explicit description count and profile1 are
the decisive evidence.

The script pins the immutable code commit, not the later records-only HEAD.
Future replay should use a worktree at3763e5b8 or explicitly rebind the source pin
and retain that diff. It deliberately rejects existing output directories; choose
a fresh scratch output, do not overwrite archived results.

## Residual profile

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -u /data/ycfeng/tmp/d29_group_reuse_runtime_loop.py --output /data/ycfeng/tmp/d29_group_reuse_profile01 --iterations 2 --execution > /data/ycfeng/tmp/d29_group_reuse_profile01.log 2>&1
```

**GREEN,exit0**,cold1.6370585952s,warm0.3740707878s with profiler overhead.
One actual resolver call1.210s,88 descriptions;12 consumer-local selections
across two pricing calls. Original diagnosis12 resolver calls/1056 descriptions
are reduced to1/88. No prewarm or one-time validation skip is introduced.

Residuals over both calls: backend predict_collective_plan8calls0.548s cumulative;
JSON loads1654calls0.601s cumulative (decode0.594s); unchanged artifact reads
10calls0.098s. The cold resolver cost is still present. Residual profile is stored
as residual_profile.txt and pricing_profile.json. Backend and initialization
validation are diagnostic only; no second repair is included.

## Scope and checks

| Path | e41-relative + / - / gross | Retained gross / ceiling |
| --- | --- | --- |
| frontier/execution_time_predictor/moe_routing_workload.py | 10/3/13 | 143/175 |
| frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py | 1/3/4 | 111/128 |
| tests/unit/test_prediction_measurement_bindings.py | 54/0/54 | 254/260 |

Source17/45,test54/60,total71/105 PASS. Exactly three code/test paths.
Final CORE retained3932/4404; global9175/10000, computed by updating three actual
retained diffs in the existing final handoff ledger. Retained delta59 differs from
step71 because some removed lines are historical additions. budget.json records
both, SHA256s and all ceilings. AST parsing of all three, whitespace and gitlink
pin checks PASS. No donor/primary/frozen/PRICING file edited.

The final staged check excludes verbatim `.log`/`.diff`/profile `.txt` evidence:
those raw artifacts retain original log trailing whitespace and diff context,
so an unrestricted `git diff --cached --check` reports those evidence lines.
Source and maintained Markdown/JSON whitespace checks pass. Raw evidence is not
rewritten merely to hide that diagnostic output.

## Verification limits

No unresolved in-scope test or source failure. Fixed lifetime is established by
production constructor/simulator references in plan.md, not a mutable-config
fallback. All per-point source/context/family/model/duration/count checks remain.
The actual pricing loop uses an existing test construction seam to avoid complete
Simulator initialization; measured one-lane/layer costs are not full6144/1024
runtime acceptance. Primary must review and freeze before deciding residual
repairs or running original offline/online2400s and native gates.
