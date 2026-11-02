## 1. Assignment and outcome

B15 implements the CPU collective consumer for the V4 H200 measurements: implementation-aware communication specs and curves, row classification, one measured pricing hook, grouped EP dispatch, ordered SP widths, endpoint pricing, and communication-payload admission for PDD roles.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Scratch root, abbreviated as `S` below: `/data/ycfeng/tmp/codex-implement/5efbdad3-a727-43de-873b-f6928d5eb4e1/b15`.

Interpreter, abbreviated as `P`: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6. The observed HEAD is `cc80896069d8b1b7a5df1966b1568b053698512d`; `git merge-base --is-ancestor 9dac0271 HEAD` exited 0.

Applicable root instructions and the named requirements, plan, design, G5h design/report, C1 coverage, and issue records were inspected. All edits are within the assigned writable paths. The main predictor change is restricted to the hook import and call. The measured CSVs/provenance, task records, attention admission, serving-context PD gate, kernel-gap values, and composition implementation were not edited. No state-changing Git operations, commits, package installation/upgrades, GPU work, network work, deployment, or delegation were performed.

Initial tracked files were clean. The pre-existing untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/` directories remain intact.

**Outcome: partial.** The independent implementation and checks are finished, but the runtime V4 combine-payload migration is deliberately deferred under the assignment's explicit stop rule. Full acceptance also remains open because broader tests expose unrelated failures and fixture prerequisites. Claude retains review, acceptance, commits, and task-record ownership.

### Material blocker: existing combine fixture

`tests/unit/test_dsv4_structural_execution.py::destination()` constructs a 40-row routing workload for TP8/EP64, meaning eight DP source lanes. Eight positive native lanes require at least 64 gathered SP rows. Eight 17-token source lanes produce widths `(3,3,3,3,3,3,3,3)` and 192 gathered rows.

| Quantity | Existing fixture | Physical eight-lane, 17-token cohort |
| --- | ---: | ---: |
| Gathered rows | 40 | 192 |
| BF16 combine input bytes | 327,680 | 1,572,864 |

Consequently, the required equality with the old lane-workload payload cannot hold on this fixture. Evidence is in `S/combine_fixture_blocker.json` and `S/widths.log`.

The runtime combine therefore still uses `_dsv4_ep_output_bytes` and receives `lane_workload` through the structural collective context. Shared `ep_workload` already supports the V4 combine, and ordered widths reach its measured exact lookup, but switching its runtime payload builder and deleting that retained context plumbing awaits Claude's fixture decision. Physical cohorts do satisfy equality: the `(6144,1)` wave has 6,152 gathered rows and 50,397,184 bytes through both calculations.

## 2. Changes

Paths below are relative to the working root. Line counts are actual before/after counts, rather than the brief's approximate counts.

| Changed file | Lines before → after | Behavior and reason |
| --- | ---: | --- |
| `frontier/cc_backend/backends/measured_cc_backend.py` | 251 → 307 | Adds implementation-aware curve resolution, SP/logits curves, equal-lane fitting, exact unequal-combine retention, holdout diagnostics, and `predict_operator_collective`. Legacy methods use the same lookup table. |
| `frontier/operators/spec.py` | 378 → 393 | Adds `CollectiveImplementation`, `CommOperatorSpec.collective_implementation`, `sp_rows_builder`, and context fields `sp_token_widths`/`selected_logits_rows`. |
| `frontier/operators/sequence_parallel_moe.py` | 152 → 168 | Adds the unified measured hook; extends shared TP/EP geometry to V4 and reads IDs dtype from the architecture profile. |
| `frontier/operators/families.py` | 964 → 984 | Registers V4 implementation choices, replaces three dispatch specs with one grouped spec, uses descriptor rows and unpadded SP reduce-scatter bytes, admits PDD payload roles, and converts endpoints to communication specs. Retains the old combine builder pending the fixture decision. |
| `frontier/model_architectures.py` | 1,331 → 1,342 | Adds architecture-owned gathered IDs dtype: GLM int32 and V4 int64, with a clear missing-dtype error when dispatch is built. |
| `frontier/profiling/collectives/vllm_forward_collectives.py` | 293 → 295 | Reads `TOPK_IDS_DTYPE` from the GLM profile; measurement behavior and plans remain identical. |
| `frontier/profiling/collectives/vllm_dsv4_forward_collectives.py` | 150 → 152 | Reads `DSV4_TOPK_IDS_DTYPE` from the V4 profile; plans remain identical. |
| `frontier/execution_time_predictor/deepseek_v4.py` | 804 → 816 | Updates the dispatch phase name, prices embedding/logits through communication specs, records int64 routing IDs, carries widths into payload contexts, and replaces `source_physical_rows` with `source_sp_token_widths`. |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8,626 → 8,626 | Replaces only the old hook import/call. Alias dispatch and all other predictor behavior remain intact. |
| `frontier/scheduler/utils/ep_wave.py` | 193 → 194 | Computes ordered V4 widths once per wave from `compute_rows`, preserving the routing-row total. |
| `frontier/scheduler/utils/expert_parallel.py` | 831 → 832 | Passes the existing EP lane width carrier into V4 structural execution. |
| `frontier/metrics/constants.py` | 351 → 349 | Replaces the three dispatch metric names with `DSV4_EP_DISPATCH_ALLGATHER`. |
| `tests/integration/run_glm53_comm_replay.py` | 110 → 110 | Uses the unified measured hook; output remains byte-identical to the supplied baseline. |
| `tests/integration/run_dsv4_structural.py` | 483 → 493 | Checks one grouped dispatch and selected-row logits presence; removes the obsolete fixed-duration embedding-reduction expectation. |
| `tests/unit/test_comm_operator_families.py` | 1,100 → 1,099 | Updates expected registry membership for the grouped dispatch and communication endpoints. |
| `tests/unit/test_measured_cc_backend.py` | 451 → 463 | Updates row fixtures/classification and verifies the separate logits curve without changing existing curve predictions. |
| `tests/unit/test_dsv4_source_execution.py` | 333 → 337 | Adds the recording backend's all-reduce entry and updates unpadded SP reduce-scatter expectations. |
| `tests/unit/test_dsv4_structural_execution.py` | 455 → 459 | Supplies ordered widths and updates grouped-dispatch/endpoint expectations while preserving the contradictory 40-row destination workload. |
| `tests/unit/test_dsv4_graph_measured.py` | 215 → 215 | Uses the width-returning source helper and preserves its 40-row routing expectation for the valid TP8/DP2 graph cohort. |
| `tests/unit/test_dsv4_dp_sync.py` | 507 → 507 | Reads the unchanged routing total from `widths.gathered_tokens`. |
| **New:** `tests/unit/test_v4_measured_collective_consumer.py` | 0 → 346 | Adds 118 parametrized cases covering implementations, payload grids, exact lookup without name matching, dtype ownership, roles, carriers, wave propagation, and endpoint ownership. |

`frontier/execution_time_predictor/dsv4_composition.py` required no edit: it contained none of the removed dispatch names.

### Interfaces and carrier

The enum is `CollectiveImplementation`, with members `GROUP_COORDINATOR_DIM0` (default), `VLLM_SEQUENCE_PARALLEL`, and `GROUP_COORDINATOR_LAST_DIM`. All GLM specs retain the default, including `logits_allgather`.

The hook remains in `frontier/operators/sequence_parallel_moe.py`:

`measured_collective_time(operator, ctx, backend, size)`

It returns `None` for other backends. For `MeasuredCCBackend`, every communication spec goes through:

`predict_operator_collective(alias, implementation, data_size_bytes, num_devices, comm_domain, sp_rows=None)`.

Exact lookup depends on the resolved EP `reduce_scatterv` curve and unequal ordered rows supplied by the spec. It compares no operator name. Renamed GLM and V4 combine specs are tested.

The V4 width carrier is:

`lane_batch.moe_sequence_parallel_token_widths` → structural `sp_token_widths` argument → `CommPayloadContext.sp_token_widths` → `ep_widths(ctx)`.

`expert_parallel.py` reads the lane attribute when invoking V4 structural execution. Shared dispatch/combine geometry and the combine row builder read widths through `ep_widths`. GLM continues using its existing lane-batch attribute through that reader's fallback. Source attention batches receive no new attribute. Lane workload remains available for routed compute, workload records, and the deferred combine payload.

The IDs field is `ModelArchitectureProfile.gathered_topk_ids_dtype`, required through `require_gathered_topk_ids_dtype()`. Both producers, grouped consumer geometry, and the V4 gating workload label read this source.

## 3. Verification

### Environment and commands

Commands ran from the repository root with `ulimit -v 31457280` and an outer `timeout` (30 seconds for routine reads/comparisons, 60 for artifact validation, 600 for pytest/hygiene, 900 for structural simulation). Child checks also have timeouts and inherit the memory limit.

The minimal environment used `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, and `VIDUR_DISABLE_WANDB=1`. Pytest used scratch basetemps, no cacheprovider, scratch `TMPDIR`/`FRONTIER_TMP_ROOT`, and scratch matplotlib configuration. The interpreter's bin directory was added to `PATH` for tests that invoke `python`.

CPU torch checks used `/usr/local/lib/python3.10/dist-packages`, `CUDA_VISIBLE_DEVICES=""`, and `VLLM_BS_SOURCE_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. Final profiling retries also added `/usr/lib/python3/dist-packages` to reuse its existing pure-Python `setuptools`; no dependency was installed or changed. `TRITON_CACHE_DIR` pointed into scratch. Native source-object tests also required `FRONTIER_VLLM_BS_ROOT` set to the same source root.

Executed command bodies, with `P` and `S` expanded to the paths above:

```bash
P /data/ycfeng/tmp/b15_baseline/h800_curves.py
P tests/integration/run_glm53_comm_replay.py --output-json S/glm_after.json
P S/plans.py S/plans_after.json
P S/compare_baselines.py
diff -u S/h800_existing_before.normalized.json S/h800_existing_after.normalized.json
cmp /data/ycfeng/tmp/b15_baseline/glm53_comm_replay.json S/glm_after.json
cmp S/plans_before.json S/plans_after.json
P S/verify_tests.py focused_verified
P S/verify_tests.py isolated
P S/verify_tests.py torch_complete
P S/verify_tests.py extra
P S/verify_tests.py extra_torch
P S/verify_tests.py native_env
P S/diagnostics.py
P S/widths.py
P tests/integration/run_dsv4_structural.py --output-root S/structural --dp 2 --case both
P S/revalidate_structural.py
P tests/integration/run_dsv4_structural.py --output-root S/structural_online --dp 2 --case online
P S/hygiene.py
git diff --check
```

`verify_tests.py` contains exact file/node selections and invokes pytest with `-q -p no:cacheprovider --basetemp=S/pytest_<mode> --junitxml=S/pytest_<mode>.xml`. The broad run used `pytest.main` over `S/tests_selected.txt`, excluding the two initially uncollectable torch files and deselecting `test_kernel_gap_pricing.py::test_every_collective_launches_one_kernel_by_default`. Its actual output is `S/pytest_all2.log` and `S/pytest_all2.xml`.

### Acceptance results

| Acceptance | Status | Observed result |
| --- | --- | --- |
| 1. H800 preservation | **PASS** | All five existing curve descriptions match field-by-field; all nine exact-combine entries match. Exactly one curve is added: `('logits_all_gather','tp',8)`. Normalized `diff -u` and baseline assertions exit 0. |
| 2. GLM and producer preservation | **PASS** | Replay `cmp` exits 0; SHA256 is `f2116303e30ca97853157a385248c3a72113a706676296afb16d075f080e6e15`. K5 and G5h plan `cmp` exits 0; combined plan SHA256 is `10c87f8dc1675d274658e86a8e2cdebdb774ba786f7787c5d597cac5e92e38dd`. Plans contain 36 GLM scenarios per DP size and 82/46 V4 planned calls. Required GLM/backend/registry test files pass in the focused run. |
| 3. H200 loading | **PASS** | Eight curves, 16 ordered exact-combine entries, and 48 holdouts. Holdouts are exclusively unequal dispatch rows. The 16 exact entries pool three passes each; retained-combine validation remains unchanged. |
| 4. Pricing contract | **PASS for implemented path** | Tests verify every V4 implementation choice, renamed exact GLM/V4 combines, typed vector/size coverage errors with curve keys, and unchanged alias dispatch for other backends. |
| 5. V4 payloads | **PARTIAL** | All 248 physical G5h-grid comparisons equal expected bytes. Grouped dispatch uses int64 IDs. Physical combine calculations agree, but the existing EP64 structural fixture prevents the requested runtime combine migration. |
| 6. Widths | **PASS** | Existing routing totals remain unchanged. Whole-wave test observes `(768,1)` at all 16 dispatch and 16 combine contexts, and source-batch attributes remain identical before/after. |
| 7. PDD payload roles | **PASS** | PREFILL and DECODE payloads equal MONOLITHIC for the same batch; DECODE_ATTN/DECODE_FFN reject. Runtime PD admission gates remain unchanged. |
| 8. Endpoints | **PASS** | Measured-backend unit cases verify layer-0 embedding on real/dummy forwards and layer-42 logits only with selected rows. Offline and online 43-layer structural artifacts pass under the memory limit. |
| 9. Existing tests | **PARTIAL** | Broad and targeted runs cover the 121 matched pytest files. Latest results across separate runs contain 3,661 passing cases, 17 failures, 61 fixture errors, and 23 skips; the required kernel-launch test is deselected. Remaining failures are detailed below. This is a union of separate runs, not one successful full-suite invocation. |
| 10. New tests | **PASS** | 118 new parametrized cases; all pass. |
| 11. Diagnostics | **PASS** | Holdouts, graph/eager medians, G5h payloads, and C1/G5h vector coverage are written to scratch and inspected. |
| 12. Hygiene | **PASS for new changes; existing exception** | All 21 changed modules cold-import with exit 0 and no torch/vLLM/CUDA modules. No new unused imports. Five pre-existing unused-import findings remain in the predictor because edits there are restricted to the hook. All other edited modules are below 2,000 lines; the predictor has no growth. `git diff --check` exits 0. |

For GLM, bypassing the predictor alias dispatch preserves numbers: its communication methods delegate directly to backend methods and return their result (for example `predict_allreduce_time`, around line 8011). The launch-overhead strip around line 5024 returns unchanged when the backend lacks `estimate_intra_server_allreduce_launch_overhead_ms`; this backend lacks it. Legacy methods resolve the same default implementation table. Replay and curve assertions independently confirm preservation.

### H200 loaded curves

| Key | Sizes in bytes | Calls |
| --- | --- | --- |
| `all_gatherv/ep/16` | 8,264; 16,528; 24,792; 33,056; 6,346,752 | `moe_dispatch_all_gatherv` |
| `all_gatherv/ep/32` | Same five sizes | `moe_dispatch_all_gatherv` |
| `all_reduce/tp/8` | 8,192; 16,384; 32,768; 65,536; 131,072; 196,608; 262,144; 50,331,648 | `embedding_all_reduce` |
| `logits_all_gather/tp/8` | `32,320 × b`, b=1–24 | `logits_all_gather` |
| `reduce_scatterv/ep/16` | 131,072; 262,144; 393,216; 524,288; 100,663,296 | `moe_combine_reduce_scatterv` |
| `reduce_scatterv/ep/32` | 262,144; 524,288; 786,432; 1,048,576; 201,326,592 | `moe_combine_reduce_scatterv` |
| `sp_all_gather/tp/8` | 8,192; 16,384; 24,576; 32,768; 65,536; 98,304; 131,072; 6,291,456; 25,165,824 | Attention and final-mHC SP gathers |
| `sp_reduce_scatter/tp/8` | Same eight sizes as embedding all-reduce | `sp_attention_reduce_scatter` |

The other new implementation-specific keys are therefore `sp_all_gather/tp/8` and `sp_reduce_scatter/tp/8`. Existing H800 keys retain their identities. Complete times, descriptions, and exact entries are in `S/h200_coverage.json` and `S/h800_coverage.json`.

The H200 exact vectors are `(1,768)` and `(768,1)` for EP16, plus all 14 unequal four-lane vectors over `{1,768}` for EP32. Native token vectors over `{1,6144}` map to these SP widths. Ordered directions retain different measured medians.

### Payload and width evidence

For graph T=`1,2,4,8,16,24,32`, local SP rows are `1,1,1,1,2,3,4`.

| Operator | Expected bytes, matched by consumer |
| --- | --- |
| Embedding and SP reduce-scatter | `T × 8192`, unpadded |
| SP attention gather | `ceil(T/8) × 8192` |
| Final mHC gather | `ceil(T/8) × 32768` |
| Grouped dispatch | `max(r_d) × 8264` |
| Physical combine | `8 × sum(r_d) × 8192` |
| Logits gather | `selected_rows × 32320`, tested for 1–24 |

At eager T=6144, embedding/SP reduce-scatter is 50,331,648 bytes, attention gather 6,291,456, final mHC gather 25,165,824, and dispatch 6,346,752. Uniform combine is 100,663,296 bytes for DP2 and 201,326,592 for DP4. Mixed DP2 combine is 50,397,184 in both directions. DP4 unequal combine sizes are 50,528,256, 100,794,368, and 151,060,480. `S/v4_payload_grid.csv` records all 248 comparisons, all equal.

The V4 reader and V1 maximum-padding helper agree on 11 of 14 inspected fixtures. They differ on:

- Graph source widths 32/8: V4 SP `(4,1)`, V1 `(4,4)`; gathered rows 40 versus 64.
- Prefill beside dummy: V4 `(768,1)`, V1 `(768,768)`; gathered rows 6,152 versus 12,288.
- Four prefill lanes, one eager: V4 `(13,4,1,768)`, V1 `(768,768,768,768)`; gathered rows 6,288 versus 24,576.

V4 remains on its descriptor reader. Details are in `S/width_comparison.json`.

### Test execution and unresolved checks

The final focused command passed **484 tests in 9.34 seconds**. CPU profiling with existing system setuptools passed **296 tests in 54.12 seconds**. Additional CPU profiling passed **48 tests**, with three vLLM-dependent skips.

The initial broad run returned 17 failed, 3,003 passed, 21 skipped, and one deselected. Targeted retries resolved the torch-dependent failures and initial collection errors. The GDN simulator test that failed on `routing_details missing target_replica_id 143` passes in isolation; the evidence supports test-order dependence rather than a B15 regression.

Remaining latest failures/errors:

- One scheduler late-join expectation in `test_vllm_dp_placement_runtime.py`. It reproduces with original source snapshots loaded through the scratch baseline importer; both old and new code start lane 2 as real rather than dummy. `S/pytest_baseline_placement_confirmed.log` records the same failure.
- Ten `test_colocation_release_review_contracts.py` failures reference absent release-debug scripts or the absent `frontier.config_optimizer` package. Adding the interpreter to PATH resolved only the initial executable lookup problem.
- Six HTTP-harness cases in `test_dp_placement_harness.py` fail importing `aiohttp` in the minimal environment, before HTTP work starts. They were not rerun with network behavior enabled under B15's network exclusion.
- Forty-two native-groundtruth setup errors arise at `tests/unit/test_native_groundtruth_check.py:73`: Python 3.10.6 rejects `TarFile.extractall(..., filter="data")`.
- Nineteen other native-groundtruth setup errors remain after supplying CPU torch: the fixture replaces `sys.modules['torch']` with a fake module, and native DP synchronization then fails with `AttributeError: module 'torch' has no attribute 'SymInt'`. These occur before the collective consumer is exercised.

Remaining skips are 19 intentionally retired PD-AF contracts, the unavailable optional collective-sim build, and three tests requiring an installed vLLM package. The known kernel-launch test is explicitly deselected. The initial 35 native-source-object skips were resolved by setting `FRONTIER_VLLM_BS_ROOT`; that targeted run passed 36 tests alongside the 19 groundtruth fixture errors.

Per-case latest statuses are in `S/verification_remaining.json`; counts and file attribution are in `S/verification_summary.json`. All executed pytest logs/XMLs remain in scratch.

The text search was broadened after the first run and found 14 additional pytest files; those were executed. `S/test_search_command.txt` preserves the search expression, and `S/tests_search_final.txt` contains the complete matched list. Non-pytest matches include the two assigned integration drivers, analysis helpers, and calibration/E2E/native profiling utilities. The two assigned drivers ran; the other utilities were not invoked as standalone workflows with separate profiling inputs or native execution. Their applicable CPU unit coverage was run.

### Diagnostics

`S/h800_holdouts.csv` and `S/h200_holdouts.csv` contain call, group size, tokens, bytes, measured time, uniform-curve value, and relative error for every holdout. H800 has 282 covered holdouts: median absolute relative error 3.691%, maximum 37.606%. H200 has 48 covered holdouts: median 3.055%, maximum 5.875%. These rows do not fit or select runtime curves.

H200 graph/eager shared-size medians, in microseconds:

| Call / group size / bytes | Graph | Eager | Eager − graph |
| --- | ---: | ---: | ---: |
| Embedding / 8 / 8192 | 8.8544 | 10.5824 | +1.7280 |
| Combine / 16 / 131072 | 61.3448 | 56.1632 | −5.1816 |
| Combine / 32 / 262144 | 116.5568 | 112.6464 | −3.9104 |
| Dispatch / 16 / 8264 | 140.9232 | 133.7632 | −7.1600 |
| Dispatch / 32 / 8264 | 257.1328 | 251.3808 | −5.7520 |
| Attention gather / 8 / 8192 | 7.2680 | 8.1488 | +0.8808 |
| Attention reduce-scatter / 8 / 8192 | 9.2480 | 13.2832 | +4.0352 |
| Final mHC gather / 8 / 32768 | 8.4264 | 8.1856 | −0.2408 |

Logits has no size with both modes in the table. The largest shared-point difference is SP reduce-scatter, with eager 43.63% above graph. The loader still pools the modes as assigned. Full values are in `S/h200_graph_eager_pooling.csv`.

C1 P2-32 has 14 windows and five distinct vectors: `(1,1)`, `(2,2)`, `(768,768)`, `(1,768)`, `(768,1)`. Both unequal windows have exact H200 entries. P2-64 has 17 windows and eight distinct vectors; all five unequal windows are covered: `(1,1,1,768)`, `(1,1,768,1)`, `(1,768,1,1)`, `(1,768,768,768)`, `(768,1,1,1)`. Equal vectors use uniform curves. P0/H800 has 14 windows and five distinct vectors; neither of its two unequal vectors has an exact entry in that table.

`S/c1_combine_vectors.json` records every window and table-presence result. `S/g5h_combine_vectors.json` contains 26 grid vector records. `S/diagnostic_summary.json` summarizes diagnostics.

### Structural integration and hygiene

The first structural command completed the offline simulator, then stopped at the obsolete fixed-duration embedding assertion. After updating the validator, `P S/revalidate_structural.py` validated those existing offline artifacts with exit 0. The online command subsequently completed simulation and validation with exit 0. Each case completed two requests, recorded six real stage rows, covered all 43 attention layers and scheduled boundaries, and conserved its ledgers. Maximum stage discrepancy was approximately `2.66e-15` ms and phase discrepancy `6.94e-18` ms. These remain analytical/dummy CPU structural checks, not native latency or assignment evidence.

Artifacts are in `S/structural/offline`, `S/structural_offline_validation.json`, and `S/structural_online/report.json`.

`S/hygiene.py` ran the supplied unused-import checker and cold-imported every changed module. Its baseline import initially had one fixture-relative-path artifact; preserving the original `__file__` corrected that baseline check, recorded in `S/cold_dp_sync_baseline_corrected.log`. All after-change imports exit 0 and load no optional torch/vLLM/CUDA modules. Five unchanged predictor findings are `BaseCCBackend`, `EPLaneWorkload`, `ExecutionTimePredictionModelManager`, `MEMORY_FAMILY`, and `product`. They were not removed because the assigned predictor scope permits only hook edits. `S/line_counts.json`, `S/cold_imports.json`, and the before/after unused-import reports preserve the results. The actual predictor count is 8,626 in this checkout, although the brief quotes 8,666; it has zero growth.

A search over `frontier/` and `tests/` finds no remaining old dispatch names, `source_physical_rows`, or `measured_unequal_combine_time`. Pinned native `vllm/models/common/ops/sequence_parallel.py:23–39`, V4 model SP sites, and `distributed/device_communicators/all2all.py` confirm custom-first SP calls, unpadded reduce-scatter input, and grouped hidden/weights/IDs dispatch.

## 4. Remaining work and decisions for Claude

First, reconcile or separate the artificial EP64 destination fixture from physical communication geometry. A direct physical fixture would use 192 gathered rows for eight 17-token lanes and 64 for eight one-row dummy lanes. After that decision, switch the V4 combine payload to shared `ep_workload` through `_dsv4_ep_payload`, remove `_dsv4_ep_output_bytes`, and remove the structural collective-context lane-workload special case. Retain lane workload for compute and workload records. No different runtime geometry was chosen silently.

Broader acceptance needs separate handling of the pre-existing scheduler expectation, retired release-script/package tests, HTTP harness environment/scope, native-groundtruth tar extraction and fake-torch fixtures, and the three unavailable vLLM-dependent cases. The five existing predictor unused imports also need an explicit wider cleanup scope if acceptance requires their removal.

Remaining PD rejections found, unchanged:

| Location | Message / effect |
| --- | --- |
| `frontier/attention/deepseek_v4.py:155` | `V4 PD attention is unsupported` |
| `frontier/attention/deepseek_v4.py:218` | `V4 attention component requires co-location` |
| `frontier/execution_time_predictor/deepseek_v4.py:568` | `V4 predictor PD phase/graph equivalence remains unqualified` |
| `frontier/execution_time_predictor/deepseek_v4.py:675` | `V4 isolated measured scopes do not qualify PD batches` |
| `frontier/utils/param_counter.py:607` | `V4 parameter estimates require PP1 MONOLITHIC/PREFILL/DECODE`; permits PDD roles, rejects PD-AF roles and PP>1 |

Payload admission alone therefore does not qualify V4 PDD runtime execution. Coverage remains bounded by the measured table; uncovered implementations, sizes, or unequal vectors raise the typed coverage error.

## 5. Review handoff

Inspect the default implementation on all GLM specs; the backend's row-classification and call-only logits override; preservation of old H800 curve identities; the single grouped dispatch and profile-owned dtype; descriptor-row SP payloads; endpoint layer/presence ownership; and the width carrier's lack of source-batch mutation. The retained combine builder and corresponding test-fixture inconsistency are the principal unfinished code change.

Rerun `compare_baselines.py`, the two `cmp` commands, `verify_tests.py focused_verified`, and the structural driver after completing the combine migration. Recheck the payload grid, source mutation assertion, and original-fixture equality specifically. Then rerun applicable broader checks after their separate prerequisites or fixture fixes are settled.

The implementation is reviewable but has not been accepted or committed. Baseline comparisons, commands, logs, diagnostics, source snapshots, and unresolved-case details remain under the scratch root above.
