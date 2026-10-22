# D29 native19 implementation archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Parent accepts and integrates native19, publishes the image and supplies verified installed-source artifacts. |
| 2026-10-23 | Prepare controlled T2 metadata/cases and bounded execution/resource proposal; static-only, no code change or launch. |
| 2026-10-23 | Complete approved native19 implementation and CPU verification; retain historical design archive below. |
| 2026-10-23 | Repair four implementation findings,485 CPU PASS and1269/1582; narrow thin graph qualification. |
| 2026-10-22 | Close native design findings; update its 19-file estimate and validation limits. |
| 2026-10-22 | Record approved semantics and native design; retain implementation and live-validation boundaries. |
|2026-10-22|Archived actual native route and reference-invariance policy decision; no implementation or launch.|

## Current implementation outcome

TaskID: **D29-NATIVE-IMPLEMENTATION**. Approved native19 implementation is complete
on feat/step5-position-native. Commits: e48cdc0 (early native interface/source
receipt and CPU tests), d12941c (owned image/client/provenance/T2 integration).
Reviewer repair commit:74d5c48 (7 existing owned code paths plus local evidence).
The final documentation commit records this handoff. No peer files, historical
captures, original native source, GPU/platform/auth/cloud/SSH/Docker or publication
operations were modified/executed by this stream.

Actual scope after reviewer repair: **19 code/test/Dockerfile paths;1269 added+deleted lines of1582**,
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
   produce only the authorized4-file overlay. Context:
   `/data/ycfeng/tmp/d29-native-build-context02`; no image build/push by this stream.
   Root completes publication and actual source export; see the parent handoff below.
- `tests/comparison/dp_placement_pp/vllm_replay.py` and
  `tests/comparison/calibration/pd_proxy.py`: derived registry adapter, validated
  supplied immutable identity, effective selector and HTTP capability propagation.
- `tests/integration/step5_pd/{r2_canonical.py,r2_replay.py,r2_stage.py,r2_runtime.py,verify_r2_stage.py}`:
  registered client version through all R2/direct-D calls; conditional imported
  model/context/factory/router root/spec/hash/class/config/backend census; strict
  staged identity/source checks with diagnostic-only native-call boundaries.
- `tests/integration/step5_environment/test_position_cyclic_graph.py`: thin actual
   outer setter/opaque-MoE/router PW eager-break/FULL script plus separate eager
   context/mask-bound FP8/BF16 modules. No MRv2/attention/connector/backend graph
   qualification; AST/source/CPU sequence checks only, GPU NOT_RUN.
- Existing caller/provenance/reference tests in
  `tests/unit/{test_step5_canonical_g3.py,test_step5_native_pd_r2_observation.py,test_step5_native_reference.py}`,
  plus `tests/unit/test_step5_native_image.py`; no old probe or D30 implementation edits.
- Local `scope_receipt.json`, `build_context_receipt.json`,
  `sealed_preservation_receipt.json`, `t2_static_receipt.json`, `notes.md`,
  `test_report_2026-10-23_native_interface.md` and
  `test_report_2026-10-23_native_integration.md` preserve commands, inputs and limits.

### Observed validation
Native/HTTP/client/canonical/reference/provenance/capacity/source suites:
**485 PASS/36.16s** after reviewer repair, asyncopd-py312 Python3.12.13;
previous418 PASS/39.78s. D30steps1/16 andengine/core
source/counter evidence: **48 PASS/67.40s**, frontier-py310 Python3.10.6.
Actual native interface controls11 PASS; final builder-only test1 PASS/0.15s.
Ownership19/gross1269/AST/whitespace and actual native4/snapshot/patch hashes PASS.
Candidate2907 Python files differ from base only in approved4; all12 existing
backend source inputs are unchanged. Native gross remains99;313 lines remain.
Capture02all56 andcapture03all57 manifest entries remain unchanged. Capture01
was not present at the inspected historical root; no writes occurred there.
Baseline251 PASS remains documented before implementation.

### Parent integration and source handoff
Parent independently reproduces485 PASS/36.09s and verifies19 paths/1269 gross.
The original reviewer closes N1–N4 without additional actionable findings.
Accepted native source is integrated at9495cfb; publication records are at67d5b5e.
The actual published Linux/amd64 image is
`hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee`.

The parent task `../task_2026-10-22_step5_position_routing/image_publication/`
contains the verified remote identity, derivation manifest, all2907 installed
Python hashes and53 actual serving pins. The source export is
`/data/ycfeng/tmp/d29_published_sources01/source`, including all12 backend schema
modules. Every exported file matches the actual published-image receipt.
These artifacts supersede the base-build listing for new staging. D30 canonical
source_hashes remains13 including engine.core. Full commands and evidence limits
are in the parent's `repairs/test_report_2026-10-23_image_publication.md`.

### Unresolved live evidence
Controlled T2 metadata and explicit positions/masks are now prepared in the local
`t2_preparation/` directory. Root independently accepts/seals them and still
supplies actual resource/run approval and worker/rendezvous/output identities.
Exact T2 engine max_model_len7168 is not expressible by existing script/helper;
all input positions are bounded below7168 while original metadata1048576 remains.
Do not claim full-serving7168 engine binding from these inputs.

Still **UNVERIFIED**: CUDA selected-score arithmetic, NONE/PW/FULL replay,
actual auto FP8/BF16 dispatch/E8M0/collectives, all-rank imported provenance,
actual launched GPU image identity, G3 numerical state/bit-exact transfer/full
serving and new profiling/operator-gap/E2E. Workflow then GPU/op-gap, CPU overhead
last. Old capture03 is not reclassified as balanced or submitted.

### Four-finding implementation review disposition
Reviewer ses_e9afdb7c4ffejb3hbkjZCeHFKS found4 defects at2c4cc9e; user authorized
repair within native19. All4 are resolved at CPU/source level:
1. Actual native keyword router invocation reaches explicit capture signature;
   native tuple/exception identity and single-call semantics preserved.
2. PW/FULL dummy replay occurs before buffer assertions/replay baseline.
   CPU executes pinned PW dispatch/replay bodies; FULL runtime remains UNVERIFIED.
3. Actual instance-bound Step forward identity/origin is checked; replacements
   are rejected inside the unchanged soft observer boundary.
4. Actual quant/kernel/modular/fallback source objects are checked using native
   checkpoint exclusions/oracle mappings, not names. Host positive fixtures model
   valid FP8/BF16 relationships; wrong class/backend/kernel/monolithic/apply/source
   negatives pass. These fixtures are not actual imported native qualification.
Repair changes7 existing owned paths. Before-fix14 FAIL/20 PASS, focused307 PASS,
final affected485 PASS. Exact commands, errors and limits are in the existing
test report; source/diff accounting is in scope_receipt.json. No native4 changes,
peer edits/new files/delegation or image/GPU/platform action by this stream.
Repair code diff:267 additions+23 deletions (290); baseline native19 gross
increases252 from1017 to1269 because final baseline diff differs from commit
churn. Code diff command: `git diff 2c4cc9e 74d5c48 -- tests`.

## D29-T2-INPUT-PREPARATION outcome

Authorized clean fast-forward6827507 -> accepted9090407. No source/test edits or
source/CPU acceptance rerun. Native19 remains1269/1582 gross; no native patch change.
Input/artifact commit: **aa872fa**,14 local task-record/input paths only.

Deliverables under the existing task's `t2_preparation/`:
- `model_metadata/config.json`: unchanged complete retained78447-byte metadata,
  SHA256e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63,
  official revision364b3b03fed2477aa0953fe855a2c4c8c48ee3cc. No fake model data,
  truncated exclusions, weights or tokenizer are needed by the existing T2 path.
- `cases.json`: controlled/unobserved graph2×17 rows, live16 then4; three module
  frames source6128+dummy/tail16+dummy/paddedq1+emptyDP. Physical/live DP counts
  are[6129/6128,1/0],[1/0,17/16],[4/1,4/0]. Hash7383a897d21ef2201b7f8bfb40bd855aeb89a5da7fecbedec657df96c8ecaef0.
- `provenance.json`: source/revision/hash/image/script/publication identities,
  controlled construction, input7168 bound and explicit engine7168 limitation.
- `execution_proposal.md`, `resource_proposal.json`: exact existing T2/torchrun
  command, neutral cwd/import/mounts/compatibility recipe, TP8/DP2/EP16/noSP/noMTP/
  auto/skip-padding settings. Minimum2nodes/16H800 is derived from existing
  topology validation. Suggested32CPU/128GiB per node,30min/8GPUh, single attempt
  is a root decision proposal, not an allocation, fit guarantee or launch approval.
- `static_review_receipt.json`, `hashes.sha256`: static scope/conformance limits
  and6 artifact checksums. Hash-index SHA256:
  67f1a13307dc5e19353fea4b0f8a50fddc73f04ecb53a2fa533e5cc78a939a29.

Static schema/types/positive physical rows/position bounds/hash/copy/source
agreement and existing static-only probe_plan/topology PASS. No torch/vllm imports,
CUDA initialization or backend/kernel evidence. Exact static commands/results
are appended to the existing native integration test report. Thin graph plus
eager modules remains distinct from MRv2 serving/attention/connector/full backend
graphs, G3/state/transfer and numerical/op-gap/CPU overhead acceptance.

Remaining: root independent accept/seal/readiness and finite-collection resource/
run authorization, actual paths/rendezvous/output retention, actual engine7168
seam if required, all GPU/import/library/backend/runtime evidence. No numerical
state tolerance was selected, and original G3 criteria are preserved.
The exact unbound-engine seam is the existing EngineArgs call in
`frontier/profiling/collectives/step5_dispatch_probe.py:create_native_config:334–341`
and `tests/integration/step5_environment/test_position_cyclic_graph.py:276–279`;
T2 argparse exposes no max_model_len. No code changes are authorized/performed.

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
