# Observed compute-context producer verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record owned writer/association RED/GREEN, producer positive and budget. |

TaskID: D29-COMPUTE-CONTEXT-PRODUCER. Start:f41d1ee9/code989a883d/sub3dac53c.
Delivered source/tests/records commit:9dbca5b3. Root re-review remains separate.
Workdir:/data/ycfeng/Frontier/.worktrees/step5-position-pricing. No peer import,
new code path, GPU/platform, extra agent, qualification change or submodule edit.

## Root cause and exact writer boundary

Probe previously retained expert names and collective schemas but not their
original tensor references; no point-context writer existed. Raw collector is a
collective microbenchmark and cannot infer native compute context from comm dtype.
Owned DispatchObserver now keeps original completed dispatch outputs, source
inputs, router tuple and expert apply bound arguments; same-reference checks link
the actual route->dispatch->expert. After native execution it reuses
NativeRoutedExperts.prepared_context (fp8_block_experts.py106-163) for backend,
quant, source/gathered layouts, workspace/padding and actual callable verifier.
Combine output geometry comes from original into-output output tensor.

Native source evidence: modular_kernel.py1324-1339 forwards original named expert
operands; fallback.py134-161 chooses one child then calls its apply; all2all.py128-
161 returns original gathered tensors; naive_dp_ep.py155-208 preserves gathered
activation/top-k/scales; Step5MoE.forward model.py1813-1829 computes shared first,
then routed. Files inspected under /data/ycfeng/tmp/d29_published_sources01/source/vllm.
No schema/classification/second builder duplicated.

Exporter reads verified rank reports plus exact routing inputs. Physical/live
vectors, source frame, rank/layer/input, full policy/group and input geometry must
match. Observed compute_context/native_points attach to exact buffer/count keys,
never frozen fit grouping. No corresponding eager observation means no bound
compute point; calibration-only/captured counts cannot satisfy H04. CORE chooser
in its worktree moe_routing_workload.py165-170 already reads the field and compares
count.buffer_receipt; no additional shared API required. No peer edits/imports.

## Commands and observed results

All commands below ran in the workdir above. Observed versions:Python3.10.6
frontier-py310 and Python3.12.13 asyncopd-py312; thread caps and bytecode/temp settings explicit.

RED before production edits, /data/ycfeng/tmp/d29_point_red.log:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py::test_observed_compute_point_retains_original_operands -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_point_red > /data/ycfeng/tmp/d29_point_red.log 2>&1
```
1 FAIL: AttributeError missing DispatchObserver.compute_context. This tests actual
observer invocation/operand identity, not arbitrary durations. GREEN in final suite.

Final probe, including writer->existing prepared context/verifier->association,
native one-call/error/soft diagnostic and stale tensor negatives:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_point_probe_delivery --junitxml=/data/ycfeng/tmp/d29_point_probe_delivery.xml > /data/ycfeng/tmp/d29_point_probe_delivery.log 2>&1
```
JUnit245=166 PASS/79 SKIP/0 FAIL/ERROR,5.589s. Source tests skipped without explicit
native-source fixture variable; executed separately below. CPU kernels are fixture
dependencies, but actual prepared_context/layout/verifier writer executes.

Final exporter, including actual HTSIM and real selected validator->exporter:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_comm_cost_export.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_point_export_complete --junitxml=/data/ycfeng/tmp/d29_point_export_complete.xml > /data/ycfeng/tmp/d29_point_export_complete.log 2>&1
```
JUnit54=54 PASS/0 FAIL/ERROR/SKIP,42.928s. Expected valid complete subset: actual
validator PASS, three affine anchors plus held-out point, one model/four count
keys, one bound key retaining32 rank/layer associations; candidate_only=True and
qualification.status=UNVERIFIED. Actual values match: PASS. The valid_producer
case does not mock validator. Other point exporter tests isolate association with
a declared validator stub; they are not complete-producer proof. Stale context,
source, branch/policy, stride geometry negatives reject; missing/stale validator
point reports mark UNVERIFIED; fit grouping across legitimate size branches intact.

Affected H04, real constructor/importer and validator:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_comm_transport.py tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_moe_pricing.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_point_affected_final2 --junitxml=/data/ycfeng/tmp/d29_point_affected_final2.xml > /data/ycfeng/tmp/d29_point_affected_final2.log 2>&1
```
JUnit184=183 PASS/1 SKIP/0 FAIL/ERROR,47.708s, before one final valid_producer case.
Latest exporter54 supersedes its53 cases; other three unchanged suites131=130
PASS/1 SKIP. Distinct latest affected union430=350 PASS/80 SKIP/0 FAIL/ERROR.

Actual published native router regression:
```sh
env PYTHONPATH=$PWD STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29_published_sources01/source/vllm PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_dispatch_probe.py -k actual_native_router_observer -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_point_native_final --junitxml=/data/ycfeng/tmp/d29_point_native_final.xml > /data/ycfeng/tmp/d29_point_native_final.log 2>&1
```
JUnit64=64 PASS/0 FAIL/ERROR/SKIP,2.644s. Actual source methods execute on CPU;
not installed native GPU routing/backend/count/calibration/graph qualification.

## Failed iterations and verification limits

- observer01 import directly from frontier.model_architectures triggered its
  cold circular import; use the existing profiling/common model-config entry.
- observer02 bound transport suites in asyncopd env failed missing fasteners;
  rerun those unchanged backend dependencies in existing frontier-py310, no fallback.
- exporter01 qualification alone had per-rank policy; calibration had rank0-only
  fixture identity, correctly split fits. Fixture now mirrors real all-rank collector.
- point chain initially compared tuple source-frame fixture with JSON lists;
  serialize exactly as actual reports. Native operand/value equality unchanged.
- first native regression command used FRONTIER_STEP5_NATIVE_SOURCE_ROOT instead
  of fixture STEP5_NATIVE_SOURCE_ROOT:64 SKIP, superseded by64 PASS above.
- Schema/AST and git diff --check PASS. Native/GPU-derived values remain future
  data; code to emit/validate/thread them is implemented. Root re-review/global
  qualification independent; no full-unit or GPU acceptance claimed.

## Diff and quota

Six existing owned source/test paths sincef41: +308/-9=317 gross patch churn.
Baseline1aee owned total43 paths:+3510/-96=3606/3956; remaining350. Counts exclude
inherited disjoint peer imports/task records/root gitlink; include isolated
submodule333 gross unchanged at3dac53c. The original-to-tip budget growth301 is
not the same as revision-to-revision317 churn; both recorded separately.

Exact changed baseline rows: step5_dispatch_probe.py223/10/233;
export_costs.py297/0/297; validate_result.py88/5/93;
test_step5_comm_cost_export.py271/0/271; test_step5_comm_transport.py200/4/204;
test_step5_dispatch_probe.py271/1/272. Full43-row table in summary.md.
