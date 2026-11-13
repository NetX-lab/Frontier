# 04: G1 rank-local kernel and state qualification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-20 | Added review limits to criteria 4 and 5; acceptance status unchanged. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-17 | Accepted criterion1 after new resource/identity generation verification; all ten G1 criteria complete and05/06 released. |
| 2026-10-17 | Accepted criterion5/10 within their literal scopes; combined-package criterion1 remains open for the missing04E mandatory measurements. |
| 2026-10-17 | Accepted criteria 2, 3, 4, 6, 7, 8 and 9 after evidence reconciliation; remaining package, precision and resource dispositions stay open. |
| 2026-10-17 | Added04F native execution/storage/selector evidence and unresolved SWA replay content; no complete criterion is closed. |
| 2026-10-17 | Froze native FFN checkpoint04E with separate process lifetime and raw tensor retention; complete criteria remain open. |
| 2026-10-17 | Added accepted standalone binding subset and remaining precision/FFN execution dependencies. |
| 2026-10-17 | Recorded late input/path fixes and preserved the accepted native worker generation separately. |
| 2026-10-17 | Accepted the 04D paged sparse-decode subset; runtime cache, precision and prefill obligations remain open. |
| 2026-10-17 | Accepted checkpoint04C72native norm/gate cases and216local graph replays on H800; complete-ticket criteria remain open. |
| 2026-10-16 | Accepted primary04B64-case raw-KV/joint-preparation subset; full criteria remain open. |
| 2026-10-16 | Accepted04A subset:82 native case verdicts PASS after checker exception correction; all ten complete-ticket criteria remain open. |
| 2026-10-16 | Primary corrections authorized and verified; submitted G1 job exp-1016-160330-448213. Eight-GPU diagnostic content passed; all G1 criteria remain open. |
| 2026-10-15 | Approved planning ticket created. |
| 2026-10-16 | Released after ticket03 source/design acceptance; standing resources approved, native implementation checkpoint not started. |
| 2026-10-16 | Recorded actual CUDA initialization blocker and user deferral; independent CPU/source preparation is in progress. |
| 2026-10-16 | Prepared82source/math reference cases with absent native results; local checks passed and phase review is pending. |
| 2026-10-16 | Preparation accepted. CUDA forward-compatibility rerun passed; the CUDA blocker is removed, and native G1 work waits for the implementation checkpoint. |
| 2026-10-16 | Specified native checkpoint 04A; its Grok dispatch waits for the user's `/grok-implement` command. A multi-GPU NCCL/P2P diagnostic was submitted in parallel. |

**What to build:** An S03.G1 qualification package demonstrating the locked native kernels and state transitions on legal rank-local shapes, with content-level evidence that can independently unlock early measurement and full-role qualification.

**Blocked by:** `03` Full text-model inventory and observation design.

**External prerequisites:** G1 is accepted for rank-local diagnostics only. A new native functional substitute or reference replacement is not authorized by G1. Ordinary subsequent evidence work has no GLM barrier. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.

**Status:** done

**Review limits — 2026-10-20:** Criterion 4 was checked without a declared numeric tolerance. Criterion 5 retains 128 zero-scale mismatches with `quantizer_diagnostic_verdict=PARTIAL`. Both are diagnostic limits; they are revisited only if ticket 24 needs them to explain an E2E error.

Ticket03 is done. All ten criteria are accepted by the primary; see [the criterion map](../g1_criteria.md) and `../g1_package_acceptance_01.json`. Criterion1 closed only after the new corrective resource generation `exp-1017-083634-286962` produced 168 verified phase records, 271 parent/process-tree/physical GPU samples and six successful fresh children. All 4,358 new tensor files (41,462,170,614 bytes) were individually rehashed in 35 bounded CPU calls and exactly match original04E content. New quantizer generation02 `exp-1017-083634-393146` records actual loaded stable-libtorch and Python wrapper identities. Independent resource and identity reviews returned No findings within their stated scopes; the primary separately completed raw integrity and combined-package acceptance. Original missing04E resource values and generation01 identities remain missing, not waived or retrospectively supplied. [Corrective report](../test_report_2026-10-17_corrective_resources.md).

Tickets05/06 are now ready. Criterion5 remains limited to actual precision/buffer/scale/exclusion/cast/gate diagnostics: 128 quantizer zero-scale equation differences retain PARTIAL/UNVERIFIED. Criterion10 covers ≥50 exact fidelity scenarios and frozen simulator budget dispositions. G1 does not establish G2 full-role/all-rank capacity, PD, a matching native build, arbitrary-activation numerical bounds or clean profiling. H200 deferral and the GLM production barrier remain unchanged. Earlier checkpoint counts and restrictions below are historical outcomes, not current ticket status.

## Historical native checkpoint outcomes

04D ([scope](../checkpoint_04d.md), [results](../test_report_2026-10-17_g1_sparse_decode.md)) adds 26 native BF16 paged sparse-decode cases, 104 exact frames, 78 isolated graph replays and eight safe host assertions on H800 job `exp-1017-034334-792896`. Independent post-run reviewer checked 638,976 output values, 624 input hashes, identities and terminal state: No findings. The pre-submission reviewer did not return a formal verdict before execution; that limit is preserved. Exact dominant-probe fixtures qualify selection-table consumption, addressing, causal/inactive masks and split/combine, not general numerical accuracy, integrated summary/scoring/selection or runtime allocator defaults. Accept only 04D. All ten complete-ticket criteria remain open. Two late harness input/path findings were fixed using existing validators and accepted after three CPU regressions; no fixture/oracle or accepted native evidence changed. See [follow-up](../test_report_2026-10-17_runtime_sources_input_locks.md). Remaining work is tracked in [g1_runtime_precision.md](../g1_runtime_precision.md); targeted installed-owner collection is evidence preparation, not runtime binding or qualification.

04C ([scope](../checkpoint_04c.md), [results](../test_report_2026-10-17_g1_norm_gate.md)) adds72exact norm/residual/gate cases,216isolated operator graph replays and eighthost rejection controls in successful H800 job `exp-1017-023052-337126`. Full nested text config parsing preserves92backbone/3MTP declarations/352experts; no model is constructed. Independent reviewer verified288frames/384tensor records with Decimal70/BF16 reference and accepted the subset. Model/FFN/exclusion application, runtime cache/page/window binding, integrated sparse/prefill execution and model-runner graph qualification remain open. Subsequent work runs on H800; H200 is temporarily deferred, not waived from final acceptance.

04B ([scope](../checkpoint_04b.md), [results](../test_report_2026-10-16_g1_joint.md)) adds64exact raw-KV/joint-preparation cases and272typed snapshots on successful job `exp-1016-193813-683851`. Initial job failed final summary generation after all cases; the diagnosed Path/string tooling fix and distinct rerun preserved identical raw tensors. This exercises parts of raw-KV/norm/RoPE/state/padding criteria with explicit fixtures. Runtime B/W/cache precision, full-model construction, broader finite-precision/FFN classes and graph qualification remain open. Independent post-run review was unavailable; primary performed acceptance.

Checkpoint04A ([scope](../grok_batch_04_g1_native.md)) completed on original job `exp-1016-160330-448213`:82 cases PASS, four host rejection controls PASS, one no-debug observation. Primary reviewed the corrected missing-tail exception check, preserved initial78PASS/4FAIL verdicts and verified six targeted negative mutations. See [native results](../test_report_2026-10-16_g1_native_results.md). Independent post-run review was unavailable due to model-service budget exhaustion.04A exercises parts of criteria3/4/6/7/8/10 only; model construction, TP8, raw-KV writes, fused norm/gate/FFN and graph replay remain for later checkpoints. No full criterion is accepted.

Primary preparation artifacts: [g1_preparation.md](../g1_preparation.md), [g1_preparation.json](../g1_preparation.json), and [CPU verification report](../test_report_2026-10-16_g1_preparation.md). The82planned cases cover initial summary, reset/reuse/copy, padding, selector and conditional-window references. Original preparation fields remain unexecuted for provenance; actual results are stored separately. Missing-tail device assertions were observed in isolated debug-enabled tests; the no-debug control executed without error. Runtime page/window binding and full native qualification remain open.

**Source workstreams:** S03.
**Shared owners:** C02, C03, C04 for inventory, state, and measurement evidence.

## Standalone runtime binding subset

Native job `exp-1017-042926-324906` passed native EngineArgs/text-view/quantization inventory and two fullTP1 Attention cache-view bindings: complete92backbone/95declarations/352experts, Fp8Config1406exclusions, defaultB16/cacheauto→BF16, SWA[511,0]/DSA[-1,-1], raw and summaryaliases. Independent reviewer No findings and primary raw/terminalaudit PASS. [Report](../test_report_2026-10-17_runtime_binding.md). This does not include complete Step5 model cache hooks, dummy tensors, forward or precision/FFN execution and does not complete a full criterion. First size-field harness failure preserved and corrected with samebudget. Continue [native precision/FFN](../g1_precision_ffn.md) and actual SWA/prefill/integrated DSA execution. No H200 resumption.

## Native precision preparation

[checkpoint04E](../checkpoint_04e.md) freezes sixactualnativeclasses/full352experts/exclusions/realTP1groups/nativeDummyModelLoader+reload+postprocessing andthreefixedinputs. GeneralnumericalguaranteesremainUNVERIFIED; finite/zero/outputretentiondonotreplacecompletecontentqualification. Threeexecutionownerbatchesaudited:6/6,7/8withpreservedmissingpath/failedexit1,2/2actualworkspaceimport. [Report](../test_report_2026-10-17_execution_sources.md). Native job `exp-1017-051804-058598` completed all six modules and eighteen input frames on one H800. Creator, actual `codesign` group, locked image/config, current NFS, CPU8/32000Mi, Succeeded/Completed/exit0/restart0 were verified. All native output finite/zero checks and full tensor retention passed. All 352 experts in each MoE remain present; actual backends were Triton. Cloud file integrity and independent CPU mathematical comparisons are complete, including preserved timeout/stride-checker failures and corrected CPU-only analyses. [Report](../test_report_2026-10-17_precision_ffn.md). General numerical bounds and FP8 activation-scale mode remain UNVERIFIED. Per-child memory peaks were measured, but original per-phase and parent memory evidence is missing. The distinct corrective generation supplies new measured evidence; it does not recover these historical values.

## Stateful attention04F observations

Generation01 failed on the harness DSA planner device scope, while SWA retained seven frames. CPU builder scope correction enabled generation02 DSA-only success with 22 frames and genuine state continuation through4110. The 32 read-only CPU operations verify storage, metadata, all capture occurrences and independent selector checks.

Generation03 `exp-1017-070246-633403` completed affected SWA verification after retaining captured metadata ownership. The three replay maximum absolute errors fell from 0.01096–0.01332 to 3.1792e-5, 2.7795e-5 and 3.3179e-5; eager/capture errors were unchanged. Actual full/AOT/FA3 flags, stable metadata pointers, updated sequence lengths, five lifecycle phases and successful terminal state are retained. This supports the harness lifetime diagnosis, not a native-kernel defect or general numerical guarantee.

Projection→norm/RoPE, indexer, gate and output references completed 29 read-only CPU calls/390 comparisons. They remain mathematical discrepancies without invented tolerances; conditional indexer and historical FP8 modes are not inferred from lower errors. [Report](../test_report_2026-10-17_attention_state.md), [checkpoint](../checkpoint_04f.md). Fresh 77-case fidelity is exactly identical to baseline with dirty-tree attribution and clean-cache/performance limits retained. Criteria2/3/4/6/7/8/9 are accepted within their approved G1 scope; remaining criteria are mapped in [g1_criteria.md](../g1_criteria.md).

## Review disposition — 2026-10-18

DONE as `ACCEPTED_G1_RANK_LOCAL_ONLY`, supported by `../g1_package_acceptance_01.json`. The 128 zero-scale discrepancies remain PARTIAL/UNVERIFIED. Do not reopen G1 to prove a general compiled-kernel theorem unless a specific required path is blocked.

## Acceptance

- [x] Deliver S03.G1 with its nested G0 identity, exact legal shape/configuration inventory, minimally validated observations, expected-versus-actual state checks, per-case verdicts, and unresolved native limitations under §§3, 5–7, and 10.
- [x] Exercise real qualified native kernels with dummy weights, unchanged routing, actual precision/exclusions and backend; record devices, rank geometry, initialization/buffer identities, and finite checks without reducing the target model definition.
- [x] Derive test points from locked region size r, page size B, window W, and selector K: r−1/r/r+1/2r±1, B−1/B/B+1, W−1/W/W+1, and K×r−1/K×r/K×r+1. Never substitute remembered fixture constants.
- [x] Verify raw KV writes, closed-summary content, open-region accumulators/counts/positions, reset/reuse, and the next boundary-crossing update against independently known state with declared precision-appropriate tolerances.
- [x] Check native finite-precision buffers, scales, exclusions, norm/residual casts, and headwise gating across reachable classes; distinguish packed/fused physical execution from separate semantic operations.
- [x] Account for valid and padded rows, physical selector-table extent, KV/provider replication, and scratch; show that padding consumes capacity/work without advancing live tokens, regions, or KV frontiers.
- [x] Cover initial/subsequent prefill, short-selector bypass, sparse/SWA paths, and legal decode shapes; record observed staging and graph eligibility without claiming full-model memory fit or all-rank replay.
- [x] Reject wrong precision/layout identities, region-as-token interpretation, fractional KV heads, missing summary content, and padded-token credit with explicit negative evidence rather than accepting successful exits alone.
- [x] Own the minimum observations needed for G1; publish diagnostic limitations so early scope measurements can start without G2 or the complete observer suite. G1 proves local kernel/state behavior, not full-model E2E or PD support.
- [x] Native defects yield BLOCKED_ON_RUNTIME with a minimal reproducer, not a Frontier workaround; any qualification-tooling code changes satisfy §10's ≥50-scenario fidelity checks and preregistered wall-time/memory budgets with over-budget dispositions.
