# C1/C2 compute review repair verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record exact C1/C2 RED/GREEN, JUnit, numerical seams and quota. |

TaskID D29-PRICING-COMPUTE-REPAIR. Reviewer
ses_e9afdb7c4ffejb3hbkjZCeHFKS, D29-PRICING-COMPUTE-REVIEW. Frozen review8e61c2d
checked against current bfe6f2ae/code506b43ac/sub3dac53c before edits. Root re-review
and CORE/integrated full gates remain external acceptance. No peer/import/new source
path/agent/platform/GPU/qualification-criteria changes; F4 identity/F3/F5/F6 retained.

## Environment

Cwd `/data/ycfeng/Frontier/.worktrees/step5-position-pricing`. Existing conda
frontier-py310 Python3.10.6, no torch. Existing asyncopd-py312 Python3.12.13,
torch2.9.1+cu128, CPU tensors only. Published native source root
`/data/ycfeng/tmp/d29_published_sources01/source/vllm`; actual FusedMoERouter.select,
BaseRouter template, simulator/position strategy/factory and native context setter
executed through the original-source fixture with only external dependencies scoped.
No GPU/calibration or complete installed-runtime claim. All commands cap
OMP/OpenBLAS/MKL threads1, disable bytecode/cacheprovider, keep logs/JUnit in tmp.

## Exact commands and JUnit

RED H04 before production edits, log `/data/ycfeng/tmp/d29_c2_red.log`:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_moe_pricing.py -k h04 -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_c2_red --junitxml=/data/ycfeng/tmp/d29_c2_red.xml
```
JUnit21:18 FAIL/3 PASS/0 ERROR/SKIP,1.933s. All wrong/missing native compute
context and activation/scales/top-k/output layout negatives accepted by old H04.
Positive source widths/live0 cases passed. RED result establishes the actual defect.

RED actual native observer, log `/data/ycfeng/tmp/d29_c1_red.log`:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py -k actual_native_router_observer -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_c1_red --junitxml=/data/ycfeng/tmp/d29_c1_red.xml
```
JUnit16:16 FAIL/0 PASS/ERROR/SKIP,2.407s. Missing DispatchObserver.routing
seam prevents any actual router/config/value observation. Original comm-schema
comparison alone had no routing admission proof.

Final14 affected suites, log `/data/ycfeng/tmp/d29_c_final03.log`:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_moe_pricing.py tests/unit/test_comm_operator_families.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_op_trace_utils.py tests/unit/test_step5_contract_reuse.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_package.py tests/unit/test_step5_comm_transport.py tests/unit/test_step5_dispatch_probe.py tests/unit/test_step5_moe_profiling.py tests/unit/test_step5_fixed_probe.py tests/unit/test_execution_time_op_times.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_c_final03 --junitxml=/data/ycfeng/tmp/d29_c_final03.xml
```
JUnit682:529 PASS/153 SKIP/0 FAIL/ERROR,56.033s: PASS. Increase from prior
463/89 is66 new passing cases and64 torch/source observer cases skipped here;
those64 execute separately in torch environment below. Skips are never numerical
qualification. Prior same final suites620=499 PASS/121 SKIP (52.807s; after moving
all validation ahead of pricing52.408s) superseded after explicit BF16 and pad-only
accepted-path regressions. Tests have no silent pass for inapplicable parameter pairs.

Published native original-method observer final, log
`/data/ycfeng/tmp/d29_c1_final03.log`:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py -k actual_native_router_observer -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_c1_final03 --junitxml=/data/ycfeng/tmp/d29_c1_final03.xml
```
JUnit64:64 PASS/0 FAIL/ERROR/SKIP,2.689s: PASS. Both normalization modes,
mixed live/padding and pad-only physical rows, original function called exactly once
with named kwargs. Original tuple/tensor/logit/hidden identity retained, no input
mutation or collective re-execution. Wrong IDs, weights, padding sentinel/zero,
top-k/global experts/normalization/scale/effective strategy/scoped position fail.
Native exception identity remains exact with simultaneous diagnostic failure;
entry/exit diagnostic failures produce unverified evidence, native success retained.

Cold actual backend importer, log `/data/ycfeng/tmp/d29_c_cold.log`:
```sh
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_c_cold --junitxml=/data/ycfeng/tmp/d29_c_cold.xml
```
JUnit10:10 PASS/0 FAIL/ERROR/SKIP,1.255s: PASS. Real constructor/bootstrap,
complete count/cache identities, frozen policy grouping remain unchanged.

Intermediate expanded H04 command (log `/data/ycfeng/tmp/d29_c2_green03.log`):
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_moe_pricing.py -k h04 -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_c2_green03 --junitxml=/data/ycfeng/tmp/d29_c2_green03.xml
```
JUnit36:36 PASS/0 FAIL/ERROR/SKIP,1.640s: PASS. Final affected run broadens
this to66 BF16/FP8 H04 cases including exact same-context but corrupt local/
gathered stride/scale dtype, corrupt top-k/output receipt. Each negative executes
actual H04 and asserts zero compute-query/backend calls. Source/live0 lanes0/8/15
retain all applicable positive duration+count scopes.

## Numerical and interface expectations

C1 expected IDs=(positions*K+slot)%E, then padding IDs=-1/weights=0. Expected
weights are selected unbiased sigmoid of the same actual logits, normalized only
when configured, multiplied by actual configured routed scale. Native E352/K8,
positions6128/6144 with padding0; no invented logits, counts, scale or tolerance in
production. torch.equal and native source arithmetic order used, no rerouting.
Output observations record actual config/strategy/source/logits schema/positions/
mask/IDs/weights and physical/live/assignment totals. Actual resolved
select/_select/_compute/simulator/route callables verified before interception.
Final generic comparison cannot be match if route comparison fails. Selected
validator missing/wrong routing and inconsistent totals negatives pass even with
otherwise valid collective shapes/counts/policy. Legacy unbound semantics unchanged.

C2 full canonical context equality is between current compute point and
`json.loads(plan.buffer_json).compute_context`; source shared binding and complete
plan tensors/layouts are independently checked. Complete point includes backend,
quant/deferral, source/gathered activation+scales, padding/workspace and native
facts/collective_tensors. It is not compared to unrelated engine/comm envelopes.
Per-cell buffer_json stays in exact count/cache/seal, never frozen fit identity,
so valid size-dependent policies retain prior F4a/F4b grouping. All context/layout
validation executes before compute and communication queries. Native FP8 source
activationH4096+FP32 scales32+FP32 weights8+INT32 IDs8, BF16 no scales, BF16 output
H4096 retained; numbers are fixture inspected config, production reads actual fields.

## Failed iterations retained

- C1 green01: indentation typo caused collection error (0 tests), corrected.
- C1 green02:12 FAIL/4 PASS from external vllm fixture module import touching
  VLLM_LOGGING_PREFIX; fixture binds parent vllm/envs module explicitly. No
  production/environment fallback. green03 and green04 each16 PASS.
- C2 green02:3 FAIL/18 PASS when missing compute_context left empty buffer_json,
  which real CollectivePlan correctly rejects before H04. Fixture keeps existing
  supplied_output receipt; green03 later36 PASS.
- C1 green05:2 FAIL/26 PASS, test wrongly expected actual top-k8 on intentionally
  corrupted top-k7. Observer correctly returned discrepancy/actual7; fix assertion,
  final28 PASS then32 PASS; final64 includes pad-only cases.
- Combined green first:18 FAIL due validator referring to nonexistent
  ModelShape.router_top_k; inspected existing schema and use num_experts_per_tok.
  No new classification. green02 JUnit296=196 PASS/100 SKIP/0 FAIL/ERROR,6.362s.
- Exact budget excludes inherited disjoint peer imports, task records and root
  gitlink; earlier raw diff included them but no peer writes occurred.

## Diff, budget and dependency boundary

Six existing owned source/test paths touched. Current repair versus bfe6f2ae:
frontier/execution_time_predictor/moe_operator_times.py +40/-11;
frontier/profiling/collectives/step5_dispatch_probe.py +81/-3;
tests/integration/step5_comm/validate_result.py +7/-1;
tests/unit/test_step5_comm_transport.py +9/-1;
tests/unit/test_step5_dispatch_probe.py +70/-0;
tests/unit/test_step5_moe_pricing.py +53/-6. Total+260/-22=282 repair gross.
Original-baseline approved budget43 paths,+3210/-95=3305/3956,remaining651
(rise238; previously added lines replaced). Summary lists exact43-path ledger.
AST/JSON/whitespace and retained subpatch byte identity PASS. No new sub changes:
tip3dac53c, clean checkout, baseline-to-tip patch remains sha256
c96cace541b72f7627b5f429d55615a17be4cd6c86d1827ef53969d05d766a1d.
Prior actual sub tests12 PASS unchanged; not rerun for this no-sub repair. Final
affected exporter tests execute real joint HTSIM; completed-flow negatives preserved.

Producer/CORE still lacks observed full compute_context in its exact collective
buffer/count receipts; shared/tensor-only handoff is insufficient. Owner cannot
edit/import the peer. See interface_receipt C1/C2 for precise field and failure
boundary. No copied candidate prediction/native classname is accepted as measured
point proof. Global runtime/quant/backend/GPU/T2/G3/qualification and full CORE
integrated acceptance remain root-owned/unverified. This report establishes owned
validation seams and CPU numerical/method behavior only.
