# D29 native position-routing investigation and design archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Close native design findings; update its 19-file estimate and validation limits. |
| 2026-10-22 | Record approved semantics and native design; retain implementation and live-validation boundaries. |
|2026-10-22|Archived actual native route and reference-invariance policy decision; no implementation or launch.|

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
