# D29 native19 implementation archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Complete approved native19 implementation and CPU verification; retain historical design archive below. |
| 2026-10-22 | Close native design findings; update its 19-file estimate and validation limits. |
| 2026-10-22 | Record approved semantics and native design; retain implementation and live-validation boundaries. |
|2026-10-22|Archived actual native route and reference-invariance policy decision; no implementation or launch.|

## Current implementation outcome

TaskID: **D29-NATIVE-IMPLEMENTATION**. Approved native19 implementation is complete
on feat/step5-position-native. Commits: e48cdc0 (early native interface/source
receipt and CPU tests), d12941c (owned image/client/provenance/T2 integration).
The final documentation commit records this handoff. No peer files, historical
captures, original native source, GPU/platform/auth/cloud/SSH/Docker or publication
operations were modified/executed by this stream.

Actual scope: **19 code/test/Dockerfile paths; 1017 added+deleted lines of1582**,
including **99 gross lines in4 actual native files** against the locked base.
Task records/generated receipts/sparse source snapshots are evidence, not a way
to exclude additional native changes. Committed per-path diff matches scope_receipt.json.

### Deliverables (exact paths)
Local task root:
`/data/ycfeng/Frontier/.worktrees/step5-position-native/task_memory/task_2026-10-22_step5_d29_native_balanced/`.

- Native candidate: `/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm`.
  Actual4 paths: forward_context.py, models/step5/nvidia/model.py,
  model_executor/layers/fused_moe/router/router_factory.py and
  model_executor/layers/fused_moe/router/routing_simulator_router.py.
- Local `interface_receipt.json`, `native_source.patch`, `native_source/` retain actual
  setter/factory signatures, source hashes and reviewed4-file bytes. CORE can
  consume set_routing_positions(actual_device_positions) under existing native
  ForwardContext and factory sigmoid/renormalize/routed_scaling_factor arguments.
- `tests/unit/test_step5_native_position_cyclic.py` executes actual exported
  source bodies on CPU, not mocked routing evidence or CUDA qualification.
- `tests/integration/step5_native_image/{build_image.py,Dockerfile.position_cyclic}`
  produce only the authorized4-file overlay. Final unused context:
  `/data/ycfeng/tmp/d29-native-build-context02`; no image build/push occurred.
- `tests/comparison/dp_placement_pp/vllm_replay.py` and
  `tests/comparison/calibration/pd_proxy.py`: derived registry adapter, validated
  supplied immutable identity, effective selector and HTTP capability propagation.
- `tests/integration/step5_pd/{r2_canonical.py,r2_replay.py,r2_stage.py,r2_runtime.py,verify_r2_stage.py}`:
  registered client version through all R2/direct-D calls; conditional imported
  model/context/factory/router root/spec/hash/class/config/backend census; strict
  staged identity/source checks with diagnostic-only native-call boundaries.
- `tests/integration/step5_environment/test_position_cyclic_graph.py`: parent-worker
  T2 genuine PW eager-break/FULL replay and separately context/mask-bound actual
  FP8/BF16 modular layer script. AST/source checks only; GPU NOT_RUN.
- Existing caller/provenance/reference tests in
  `tests/unit/{test_step5_canonical_g3.py,test_step5_native_pd_r2_observation.py,test_step5_native_reference.py}`,
  plus `tests/unit/test_step5_native_image.py`; no old probe or D30 implementation edits.
- Local `scope_receipt.json`, `build_context_receipt.json`,
  `sealed_preservation_receipt.json`, `t2_static_receipt.json`, `notes.md`,
  `test_report_2026-10-23_native_interface.md` and
  `test_report_2026-10-23_native_integration.md` preserve commands, inputs and limits.

### Observed validation
Native/HTTP/client/canonical/reference/provenance/capacity/source suites:
**418 PASS/39.78s**, asyncopd-py312 Python3.12.13. D30steps1/16 andengine/core
source/counter evidence: **48 PASS/67.40s**, frontier-py310 Python3.10.6.
Actual native interface controls11 PASS; final builder-only test1 PASS/0.15s.
Ownership19/gross1017/AST/whitespace and final actual sparse context PASS.
Capture02all56 andcapture03all57 manifest entries remain unchanged. Capture01
was not present at the inspected historical root; no writes occurred there.
Baseline251 PASS remains documented before implementation.

### Root prerequisites and unresolved live evidence
1. Root builds/publishes and inspects the actual changed linux/amd64 image digest,
   then supplies STEP5_POSITION_CYCLIC_IMAGE=repository@sha256:digest for staging.
   No invented final image identity is frozen. Staged native_reference carries the
   exact image/base_image/routing_strategy and clients compare all identity sources.
2. Export actual installed derived bytes/pins53 plus profiling's full installed
   source receipt. The53-entry local base-build listing is50 sealed base pins plus
   3 locally byte-validated base exports, NOT installed derived evidence.
3. Root supplies approved model metadata, explicit per-DP positions/masks and
   allocation for T2 and full serving/G3. See notes.md for case schema/inputs.

Still **UNVERIFIED**: CUDA selected-score arithmetic, NONE/PW/FULL replay,
actual auto FP8/BF16 dispatch/E8M0/collectives, all-rank imported provenance,
image/dependency/library identity, G3 numerical state/bit-exact transfer/full
serving and new profiling/operator-gap/E2E. Workflow then GPU/op-gap, CPU overhead
last. Old capture03 is not reclassified as balanced or submitted.

## Preserved pre-allocation design archive (historical)

The following archive describes the prior design-only stage, not current status.

TaskID: **D29-NATIVE-BALANCED**. Investigation and native design are complete.
YC approved position-cyclic semantics and the necessary native/Frontier/profile
and internal-image work. Implementation awaits a combined numerical scope
decision; no production/native/image/package change has been made by this task.

## Decisive findings

1. Frozen H800 TP8/DP2/EP16 Step routes are modular for both FP8 (`TritonOrDeepGemmExperts`, Fp8MoEMethod) and BF16 (`TritonExperts`, UnquantizedFusedMoEMethod); both execute router.select_experts and accept precomputed top-k. All16 historical native module witnesses agree. FP8 DeepGEMM condition is E8M0 OR valid shape; smallM alone does not mean Triton.
2. Native has only uniform_random/normal_routing simulation settings, no exact balanced strategy. EPLB/round_robin placement is not balanced token choice. Global DP counts/prefix are available without new collective/readback but routing occurs before DP gather.
3. Current Frontier ascending-Hamilton wave counts conflict with independent G3/chunk/q1 invariance. Reset policy changes all16 tail rows in splitP6128+D16/full6144; absolute positions preserve IDs but cannot guarantee every arbitrary wave balanced.
4. MRv2 already produces exact device positions; CPU mirror is optimistic, router has no position input, FULL graph replay bypasses Python. The approved policy needs graph-safe context binding, native strategy/weight/mask wiring and Frontier per-request interval counts. The native design uses the uncompiled Step outer forward to bind stable device positions during capture; actual GPU replay remains to verify.

## Deliverables

Root: `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_d29_native_balanced/`.

- `proposal.md`: exact files/lines, actual route/evidence tier, aggregate and independent-reference contracts, options/costs/minimum complete extension and ready-to-ask semantic decision.
- `native_design.md` (`c9e9274`): four native source files, two tests and thirteen
  image/package/client files; 1003–1582 changed lines, graph/weight/mask verification,
  derived-image recipe and a minimum 53-pin installed-source binding.
- `review.md`: all four native design findings resolved in independent follow-up;
  parent confirms path uniqueness and budget arithmetic. Runtime remains unverified.
- `cpu_receipt.json`, `semantic_receipt.json`: actual frozen CPU method-seam and hypothetical assignment math evidence; seven count cases, independent/chunk/q1/padding counterexamples,16-rank native witness/source hashes.
- `contract_probe.py`, `semantic_probe.py`, `preservation_check.py`: bounded task-scoped read-only CPU/evidence checks, no installed strategy.
- `preservation_receipt.json`: all57 capture03 manifest entries and50 native pins unchanged,3 probe sources compile, no launch markers.
- `test_report_2026-10-22_native_routing_contract.md`: exact interpreter/commands, expectations/values/PASS/UNVERIFIED and original probe-seam/path errors.
- `check_progress_2026-10-22.md`: historical investigation checkpoint, before policy approval.
- `requirements.md`, `plan.md`, `progress.md`: requests and current completion/limitations.

## Validation and boundaries

CPU interpreter `/data/ycfeng/envs/asyncopd-py312/bin/python`, CPython3.12.13/Torch2.9.1+cu128. Read-only probes PASS; no native engine/backend initialization/GPU tensors/collectives. Native image source and sealed capture03 bytes preserved; manifest SHA256 `cdaad24ecded6da6f2eea90ed4ef4a582432db2fd75c56c29aafb22f2f56a57d`.

No actual new routing implementation or current GPU/operator/graph/G3 validation. Historical module receipts and CPU/source evidence are not formal timing qualification. Capture03 remains **sealed, unsubmitted and not balanced**; no attempt/submitted marker/job exists from this task. No CPU-overhead model, suite or measurements.

## Pending scope decision / next action

Combine the native design review with the Frontier/profile design, deduplicate
shared integration files and confirm the complete file/line scope once with YC.
Then implement and verify the approved policy, publish its derived image, bind
new source pins and freeze a fresh generation. Preserve independent G3 criteria
and historical routing identities. Primary owns GPU/platform operations.
Workflow -> GPU/operator-gap alignment -> CPU overhead remains the order.
