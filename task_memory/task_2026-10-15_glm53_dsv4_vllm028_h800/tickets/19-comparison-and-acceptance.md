# 19: Four-metric comparison, calibration, and final acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Q2 complete: nixl6144 measured (0.346 container GPUh) and D-P2-Q2-NIXL-TIMING made the H200 affine NIXL pair (390.39 Gbit/s, 0.0237 ms) the formal P2 timing input at the measured coordinate; P2 outputs unchanged by the event proof; payload route waits for YC; ticket stays OPEN. |
| 2026-11-01 | Q1 input qualification ledger complete (`f48457d2`): 27 inputs, 19 QUALIFIED / 6 OPEN / 2 NEEDS_DECISION; D-P2-Q1-CHOICES adopted the independent-pass attention check, kept the original table and started the 16-H200 preemptible nixl6144 measurement; payload route waits for YC; ticket stays OPEN. |
| 2026-11-01 | D-WARM-TWO-FIX complete: NIXL wire-pricing fix `0fb8955d`, EP dispatch count deferred, two fresh-cache CPU replays bit-identical to B29 with verdicts unchanged; ticket stays OPEN. |
| 2026-11-01 | Integration complete atf31b2b2d/approved README24ddbbb6 with prerequisites/34-file equality/validators PASS; only human fix/replay approval pending, retained-population THPT rule preserved. |
| 2026-11-01 | Final warm review/archive accepted and complete at0cd8d5d3; only primary exact skill/prerequisites import/final status commit remains integration TODO, human two-fix/two-replay approval pending. Retained-population THPT rule unchanged. |
| 2026-11-01 | WARM-DOCS follow-up: skill accepted/root main0cc4118a and accounting7/7; correction decision/final review/integration/archive pending, retained-population throughput rule unchanged. |
| 2026-11-01 | Applied D-WARM-TARGET paired-reading contract and completed warm/residual status; preserved all-request FAILs and exact completion/window evidence, formal qualification and review/integration remain unfinished. |
| 2026-11-01 | Completed F2 P2-64 with repeat1 TTFT FAIL, exact A8 coverage, and the full eight-reading F comparison; formal qualification remains open. |
| 2026-11-01 | F2 P2-32 completed with all-repeat numerical PASS and exact collective coverage; only F2 P2-64 remains active. |
| 2026-11-01 | Completed F1a at both scales with failed latency readings; both F2K18 replays and formal qualification remain open. |
| 2026-11-01 | Recorded both F1b terminal coverage failures; no incomplete-request metric gate or formal input change. |
| 2026-11-01 | Synchronized completed A6/F3 evidence and the active F1/F2 comparisons; acceptance criteria unchanged. |
| 2026-10-31 | Synchronized current Codex/H200 operator evidence and pending work; preserved historical snapshots and qualification requirements. |
| 2026-10-25 | D-DSV4-H800-MAINLINE-CORRECTION: pointed the status to the mainline handoff, added the H4a diagnostic comparison milestone, and deferred the H200 cases. |
| 2026-10-23 | W128-V4-NATIVE06-ADMISSION-RECORDS: admitted18 P0 native06 referenceR4 and integrated D-Q10 remove those pending pointers; full19 costs/prediction/four-metric gates remain unfinished. |
| 2026-10-23 | W128-V4-NATIVE06-RUNNING-RECORDS final delta: terminal Succeeded/frozen identity FAIL leaves18 unadmitted; D-Q10 isolated prerequisite accepted, full costs/four-error comparison still open. |
| 2026-10-23 | W128-V4-NATIVE06-RUNNING-RECORDS: native06 is launched/Running, independent D-Q10 source checks/review complete with matrix running; no admitted18/full costs/four-error result. |
| 2026-10-23 | W128-V4-CHECKPOINT-CORRECTIONS-20261023: removed stale-current unfinished B/C assertions; accepted prefill and source/fidelity953 versus unsealed CPU native06/independent D-Q10, comparison still blocked. |
| 2026-10-23 | W128-V4-BALANCED-STATUS-SYNC: fixed Routing milestone current-progress anchor and synchronized accepted A/C source/CPU/fidelity versus pending B/package/native admission; comparison criteria unchanged. |
| 2026-10-22 | W128-V4-S5-STATUS: historical execution/source snapshots now point to current main progress/05/18; comparison contract unchanged. |
| 2026-10-22 | Final checkpoint separates warm-q1 Running identity/business pending from native04 sealed/unadmitted and unsealed05/versioned full-checker repair;GLM pause confirmed,all comparison gates unchanged. |
| 2026-10-22 | Active final acceptance/comparison is DSV4 only under D-DSV4-ONLY-GLM-PAUSED; all GLM tiers parked with no paired-model gate; mean contract and30%/10% unchanged. |
| 2026-10-21 | Added same-model/hardware32-to64 promotion at30% without changing final10% gates; results are separate for all four priority cases. |
| 2026-10-21 | Accepted the approved client metric/input-binding source prerequisite at9f58dbe; native references and measured four-metric acceptance remain open. |
| 2026-10-21 | Comparison now binds the 128-request fixed 6144/1024-token QPS-1.0 case, full-completion THPT window and exact-length population in all 3 independently warmed native repetitions. Preserved four independent ≤10% gates and open runtime admission. |
| 2026-10-21 | Created from the canonical H200 calibration contract (`calibration.md`, decisions Q1–Q12). Replaces ticket 10's per-cell gate with one four-metric result per model; absorbs the delivery portion of ticket 11. |
| 2026-10-21 | Second round: THPT confirmed as total prompt+output token throughput (output-only and request/s are diagnostics); anti-overfitting repetition aligned with the reference's 3-clean-repetition rule. |

## Current Codex checkpoint — 2026-10-31

Use [current plan](../plan.md#current-execution-route--codex-cli-h200-operator-path-2026-10-31) and [current handoff](../progress.md#current-session-handoff--codex-cli-2026-10-31). D-H200-MAINLINE-CONTINUE, decision46 and D-WARM-TARGET apply; Codex uses personal StepMind Python RJobBackend directly. GLM stays paused. Dated snapshots below are historical where they conflict with this entry.

Current deciding comparison: six B29/pass1/pass2 paired warm readings at P2-32/P2-64, all96 metrics/all repeats PASS. Canonical P2-64 repeat01 openers r006/r017/r027/r035/r036 bind uniquely through all eight TP-worker log cohorts, DP ownership and proxy identity; only those five paired requests leave repeat01 on both sides. P2-32 and repeats02/03 exclude none. Original all-request repeat01 TTFT FAIL12.254940665%/10.169032726%/10.277736953% remains unchanged beside the warm reading. P2-64 pooled TTFT684.545783/701.662204/700.865508 ms versus native720.639680 ms (signed errors -5.008591%/-2.633421%/-2.743975%); repeat01 -4.599811%/-1.970116%/-2.066263%. No latency-based selection or input promotion.

WARM-READING/WARM-RESIDUAL complete: [reading report](../h3o_operator_costs/warm_target_20261101/test_report_2026-11-01_warm_reading.md), [residual report](../h3o_operator_costs/warm_residual_20261101/test_report_2026-11-01_warm_ttft_residual.md). Authoritative outputs `../h3o_operator_costs/warm_target_20261101/readings/`, not first iteration. P2-32 coarse13.4–13.8 ms P-side/5.8–6.9 ms post-P unexplained; P2-64 P-step9.7–10.2 ms/READ8.8–9.9 ms partly probe/inferred; ~31 ms price-neighborhood proxy, not same-price long-term average, no offset. Skill accepted: six findings CLOSED/no actionable findings/four walkthroughs PASS. Integration complete: exact warm skill `f31b2b2d` from root0cc4118a/approved README `24ddbbb6`, prerequisites c3ba1712/7077960f/33ff10fd;34 files byte-identical, structure v2_pressure/quick_validate/diff PASS. Prior explicit README/generalization approval resolves old-README pressure failure, no new permission. WARM-FINAL-REVIEW `ses_e68dddb1effeH8TbRzOxiHxLPx` No findings/independent96 warm PASS maximum5.952669%, event/window/repeat/proxy limits inspected; no separate row/hash verifier rerun. Primary accepted diagnostic/warm scope; archive0cd8d5d3 complete. Final document commit not an open acceptance item. Human two-fix/two-CPU-replay approval was later given as D-WARM-TWO-FIX. Full qualification/ticket remain OPEN.

WARM-ACCOUNTING [report](../h3o_operator_costs/warm_accounting_20261101/test_report_2026-11-01_warm_accounting.md)/[validation](../h3o_operator_costs/warm_accounting_20261101/validation.json) is complete, 7/7 PASS. NIXL component39,521,664 B versus packed46,103,040 B per TP handle supports selected pages × existing physical stride for wire pricing only, preserving ownership/component semantics (+0.417203 ms wire, not TTFT). EP dispatch129/43=3 physical kernels/layer versus COMM registry default1 supports dispatch3/combine1 (+0.046836 ms direct eager P-step, no TTFT promise). Primary accepted both findings, NOT code corrections. Stop residual investigations: READ per-worker timings/host-zero remain insufficient. Next is the concrete two-fix decision and, only if approved, affected CPU regressions; no new GPU/host or READ mechanism. Keep the retained-population THPT rule below; the five current interior exclusions do not change endpoints, but that is not a general fixed-window rule.

D-WARM-TWO-FIX is complete ([two-fix report](../h3o_operator_costs/two_fix_20261101/test_report_2026-11-01_two_fix.md)). The NIXL wire-pricing fix is committed at `0fb8955d` (selected pages × `physical_block_size_bytes`, structural payload kept; seven boundary cases PASS; no new unit failure). The EP dispatch count is deferred under D-WARM-EP-DEFER because held-out native traces show three kernels per layer only on unequal DP lanes (494/494 forwards). Both fresh-cache CPU replays on the unchanged B29 inputs are bit-identical to B29. P2-32 all-request PASS; P2-64 all-request FAIL on the preserved repeat01 TTFT −12.254941%; warm 32/32 PASS, maximum |error| 5.675933%. The added 0.417203 ms per transfer never crosses a decode step start, which are at least 18.72 ms apart inside every transfer window, so no READ completion moved. No further correction, GPU run, offset or promotion is approved.

P2 input qualification (Q1, D-P2-INPUT-QUALIFICATION) is complete as a ledger ([ledger](../h3o_operator_costs/input_qualification_20261101/ledger.md), [decisions](../h3o_operator_costs/input_qualification_20261101/decisions.md), commit `f48457d2`): 27 inputs, 19 QUALIFIED, 6 OPEN, 2 NEEDS_DECISION. Under D-P2-Q1-CHOICES the lead adopted the independent-pass exact-point check for the 48 omitted attention workloads (maximum error 3.663151% over 96 comparisons; NONE/prefill maximum 0.971587%), kept the original attention table, and chose to measure H200 NIXL READ timing at the exact 6144-token payload (16 H200, wholly preemptible, about 2.06 GPUh; RJob `exp-1102-030025-971155`, submitted through `/ds4-exec`). The measurement finished (0.346 container GPUh; [nixl6144 report](../test_report_2026-11-01_nixl6144_h200.md)). Under D-P2-Q2-NIXL-TIMING the formal KV transfer input for P2 is now the H200 affine pair 390.39349928 Gbit/s / 0.0236521843 ms (0.968 ms per 46,103,040-B READ, against 2.987 ms from the H800 pair), qualified only at the measured coordinate (6144 tokens, eight pairs, one request in flight) by a physical-fit and per-pass trial median 5% / p90 10% rule. The P2 outputs equal the two-fix outputs: no READ can cross a decode step start ([Q2 report](../test_report_2026-11-01_q2_h200_nixl_timing.md)). The native payload observation route (Spec 5.5) waits for YC. The ticket stays OPEN.

### Retained F comparison checkpoint — 2026-11-01, all-request reading

The original eight-reading F comparison below preserves its metrics and terminal failures; it is not the latest warm-target verdict.

All eight complete comparison readings now exist: B29 P2-32 PASS; B29 P2-64 FAIL on repeat 1 TTFT; F1a P2-32 FAIL on TPOT/E2E; both F4 readings FAIL TTFT and equal B28 metrics. F1a P2-64 is complete: all-repeat TTFT/TPOT/E2E FAIL (pooled37.41/39.83/39.77%), THPT7.40%PASS; its gate ran once. F2K18 P2-32 is complete with allfour metrics PASS in every repeat (worstTTFT5.56523%); P2-64 is now complete and fails repeat01 TTFT12.792204607% (779.888727400ms native versus680.123765687ms predicted); the other11 per-repeat metrics pass. All comparison launchers/watchers are terminal. The collector verifies8readings/128metrics/3unavailable outcomes, with zero A8 violations and unchanged references. Decision46 diagnostic comparisons are recorded; full ticket acceptance remains open. See [F comparison report](../test_report_2026-10-31_f_variant_comparison.md). Collectives-only03 completed; F1b fitting and G5h validation are recorded, and both F1b CPU replays terminated COVERAGE_MISS. The held-out validation fails, and no complete F1b E2E metrics exist; both limitations are retained. F3 measurements are complete but the affine fit fails; no F3 replay is claimed. Fresh A6 completed with qualification failures. All four metrics and all three repeats are required; pooled PASS never overrides a repeat FAIL. Complete 05/09 qualification and formal measured acceptance remain OPEN.

## Historical execution status

Use the [mainline handoff](../progress.md#current-session-handoff--2026-10-25-mainline-correction)
and the [current route](../plan.md#current-execution-route--dsv4-h800-mainline-2026-10-25)
(D-DSV4-H800-MAINLINE-CORRECTION). The next comparison work is
`h4a_diagnostic_comparison` below. The P2 H200 cases are deferred. The snapshots
below are historical evidence, not current execution status or measured acceptance.

A accepted/merged292c489; all three B/C P2s CLOSED. C merged84c806f/final39b2bf8 has184/184tests,A/C26PASS,strict77/77/451artifacts and sealed package/image checks. Its already-spent approved1H800 supplement exp-1023-023708-916112 Succeeded500s/0.138889GPUh, root ACCEPTED_ISOLATED_DATA (`../../task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json`): both6-row CSVs6144/C0/7168/8192/configured+effective128, physicalTP1/eager/six-layer/TP8-local dimensions,1engine/6steps/282calls, byte-identical Python3.12/pandas2.3.3 rebuilds, all8 cloud/local hashes match including7,702,607B trace. Retain anomalies; no independent expert histogram, steady-state, TP8-PD, graph, serving or E2E admission. Historical unbalanced q1 cannot qualify balanced decode.

Native ce9ceba root133 merged PASS and e9e79d4 independent APPROVE/source-only/final strict77=77/77/451hashes/no source-cache findings complete. b62c155 existing model-name case-input correction is committed (+2/-1), actual/ceiling23/953=A432+B319+C202, dispatch491 separate (`evidence/final_union_b62c155.json`). Identical frontier/tests bytes are not a new capture or independently reviewed correction; `native_final_fidelity.json` remains e9e79d4. Historical950/952/argparse exit2/model_name FAILED partial13-file06 and native01–05 preserved; R3/R2 unchanged, native05 remains historical UNVERIFIED.

P0 native06 reference is now ACCEPTED_NATIVE_REFERENCE_R4, reader merged d278659/frozen67f31923/all8/8PASS on unchanged b62c155 run/12.002222GPUh. Client432/432/three rounds/seven boundaries and32ranks/six phases physical graph P NONE/D PIECEWISE+FULL PASS. Final JSON: [routing current root state](../../task_2026-10-22_v4_balanced_routing/progress.md#current-root-state). Thus18 P0 reference subtask PASS, full18 other hardware/prerequisites unfinished; old R3 FAIL/native05 UNVERIFIED unchanged. No Frontier prediction/four-metric calibration/operator-cost/physicalTP8 profiling/independent histogram qualification or new GPU budget.

D-Q10 INTEGRATED_ISOLATED_PREREQUISITE e89b45d/source2f87828, merged144PASS/94exact/legacy102positive72negative,3/197/core55 (`../../task_2026-10-23_v4_prefill_contract/integration_receipt.json`). R4 matrix accounting reuses actual576d27877/77/451actual hashes/519cache names/777runtime-input bytes/same environment+gitlink/unreached changed unit file, NOT new R4 capture. R4 additional2/223/zeroCore under3/300 separate from historical routing953/dispatch491, no net-union claim. Full05/09 qualified costs, complete non-dummy Frontier prediction and four-error comparison still block19; no full ticket PASS, q1 relabeling or newGPUcollection claimed.

Native06 P0 reference is admitted only by final frozen R4 acceptance, not source checks alone. Complete qualified non-dummy Frontier path/four-error comparison remains unfinished; GLM paused/CPU-overhead-last and official mean/total-THPT/30% promotion/10% final criteria unchanged.

## Historical execution snapshot — superseded (2026-10-22)

GLM06 PAUSED_USER_APPROVAL_REQUIRED handoff complete,no fresh06/tests/build/seal/
children. Corrected V4 warm-q1 exp-1022-003156-993689 submitted/observed Running,
1H800/codesign/non-preemptible,hardware business/rows PENDING. First preflight
BEFORE_SUBMISSION;descriptor ONLY aggregate_H800_ceiling32->128,527 pins unchanged.
Native04 SEALED_NOT_ADMITTED_NOT_SUBMITTED after staged PYTHONPATH/checkout mismatch
and missing67th breakable_cudagraph.py pin. Candidate05 UNSEALED and
VERSIONED_CHECKER_FIX ses_ea0af5ae9ffe20NcnuumIBrR5v pending. Old5efd/9116/candidate04
immutable,no04 READY_TO_LAUNCH. Final NEW freeze/seal only after
ALL_COMPUTABLE_FULL_CHECKER_PREDICATES_PASS and all writers' no-more-writes.
No admitted native reference/accepted hardware measurements or four-error result;
DSV4-only/mean/promotion30%/final10% unchanged,P90 still recommendation only.

## DSV4 comparison contract — current

**What to build:** case-bound native/Frontier comparisons for DSV4 and each
priority task in10/11, separate30% promotion receipts for the32-GPU cases,
and final four-metric10% acceptance. No cross-model, hardware or size averaging.

**Blocked by:**17's DSV4 frozen cases using the accepted shared68f891b artifact,
18's admitted DSV4 reference and a complete DSV4 non-dummy Frontier path with
own qualified coverage. No04/16/GLM full-path,tests,calibration or paired-model
completion dependency. D-DSV4-ONLY-GLM-PAUSED sets all GLM tiers/new packages/
measurement-proposal follow-up PAUSED_PENDING_USER_APPROVAL until a new explicit user approval;retain prior
GLM data/receipts/globalUNVERIFIED,not PASS. The FULL KERNEL_ONLY proposal is
parked,not this task's next approval question.

**Status:** PARTIAL — canonical binding, F comparisons, warm/residual/accounting diagnostics, skill acceptance, warm review/archive and integration complete atf31b2b2d/approved README24ddbbb6. D-WARM-TWO-FIX complete: NIXL fix `0fb8955d`, EP count deferred, both CPU replays bit-identical to B29. Q1 ledger complete (`f48457d2`, 19 QUALIFIED / 6 OPEN / 2 NEEDS_DECISION); nixl6144 H200 measurement complete and its affine pair adopted as the formal P2 NIXL timing input (D-P2-Q2-NIXL-TIMING, outputs unchanged); payload route waits for YC. Full measured qualification/final acceptance remains OPEN.

**Warm target (D-WARM-TARGET):** use the same source-and-run-event rule for every case/repeat. An evidenced one-time compile/capture/handshake carrier leaves the warm metric population on both native and Frontier sides of that repeat; keep every repeat and the unchanged all-request reading beside it. Do not subtract a fitted delay, exclude slow requests, reconstruct counterfactual scheduling, or treat a fixed warmup count as coverage evidence. Future native formal windows require lifecycle/server proof that every needed lazy communication pair is opened. One-time warmup costs are not modeled. All128 exact-length request completions and original window endpoints remain recorded even when the warm reading retains fewer pairs.

### Historical source execution checkpoint — superseded (2026-10-22)

Current source checkpoint: cold-start R1 APPROVE,merge7c1b888,TCP30db587,
combined288PASS25SKIP and an accepted exact checker freeze5efd00ea…f37ec36;
native_admission=PENDING,old9116ab7 immutable,no POST retry. V4 candidate03
384/384 formal/431 of432 all stays UNVERIFIED/REFERENCE_UNSTABLE. V4 warm-q1
attempt exits1,business cause not established;no submitted/Running/success
claim. Source PASS and changed scope do not provide numerical reference admission.
P90 TTFT/TPOT/E2E remains recommendation,not approved implementation;official
mean latency/total THPT definitions and thresholds below stay unchanged.

## Milestones

### canonical_metric_binding
- [x] Reuse simulator ground-truth `first_decode_token_completed_at` and native streamed first/last-token records through the existing normalize/gate CLI with `gate: client`. Preserve public CSV/internal TTFT semantics and all four definitions. Bind the actual native manifest to its pinned receipt's producer origin; verify frozen formal/warmup IDs and scheduled arrivals while preserving actual native dispatch jitter. Implementation1400f12 merged/pushed as9f58dbe after P1/P2 re-review, primary153 merged checks and strict77-case equality. [Verification](../q10_client_metrics_20261021/test_report_2026-10-21_client_metrics.md). This is a prerequisite, not native qualification or an accuracy result.

### h4a_diagnostic_comparison (P0 H800-32, DIAGNOSTIC, completed 2026-10-25)

This milestone does not satisfy `frontier_reference_run` or
`comparison_and_calibration`, and it does not count toward the 30% or 10% gates.
Its inputs may include profile04 step-level costs and other listed substitutions
(D-V4-STEP-COST-FIRST-PASS). D-NO-STEP-COST-SHORTCUT excludes whole-step tables from the formal gates.
- [x] Run the P0 case through the existing normalize/gate CLI with `gate: client`
  and report the four errors against each native06 R4 round and their mean.
- [x] Attribute the TTFT error (P queue wait, prefill compute, KV transfer, D first
  step) and the TPOT error (decode step cost, per-lane batch size over time).
- [x] List every substituted input with its source in the H4a test report.

### frontier_reference_run
- [ ] Run the exact68f891b workload on each eligible priority case with measured
  communication:32 GPUs=P16/D16, TP8/DP2/EP16/PP1 per role;64=P32/D32,
  TP8/DP4/EP32/PP1. Use its own hardware-specific compute/collective/NIXL coverage
  for P6144/C0 and D cached6143..7166/attention6144..7167. Missing coverage remains
  typed; no cross-hardware scaling or cost substitution.
- [ ] Confirm non-dummy timing without omissions/fallbacks, native-default MTP off, prefix/chunking/thinking off, graph and quantization on with effective values matching ticket 18. All 128 requests complete with exact lengths; bind native evidence for `max_model_len >= 7168` and `max_num_batched_tokens >= max_model_len` and the actual memory budget.
- [ ] Record client TTFT, TPOT, THPT and per-request E2E with the same definitions, aggregation and window as the native reference, plus p50/p95 and the non-gating diagnostics.

### comparison_and_calibration
- [ ] Compute `abs(frontier − vllm) / abs(vllm)` for mean client TTFT, mean request TPOT, THPT and mean request E2E, independently gated at0.10 for each warm repeat under D-WARM-TARGET. Keep the unchanged all-request reading beside it: full-repetition THPT is `917504 / (last_formal_last_token_completion − first_formal_submission)`, not tokens divided by127 s. Warm THPT uses retained paired token totals over each side's retained-population window (`last retained completion − first retained arrival`), using the existing harness arithmetic; pooling sums token totals and duration totals on both sides. Record both original and retained endpoints. In the current P2 readings the five excluded requests are interior and the window endpoints remain unchanged; this observation is not a fixed-endpoint rule for other populations. Keep every repeat's exclusion IDs/evidence, counts, endpoints and both readings; warmup/startup/idle outside the formal window stays excluded. Output-only throughput, request/s, ITL, internal TTFT and percentiles do not gate.
- [ ] Where a metric misses, localize the cause before changing anything: scheduler behaviour, attention or FFN cost, MoE routing, CUDA graph coverage, host/device overlap, or P/D communication. Record the evidence that identifies the cause.
- [ ] Fix causes, not symptoms. No arbitrary multipliers, no benchmark-specific numerical patches, no duplicated predictor logic, no widened threshold. A fix that needs a Frontier core refactor goes to YC with alternatives first.
- [ ] Re-run after each fix and keep every attempt in the record with its verdict.

### promotion_32_to_64 (DSV4 and same hardware; GLM paused)
- [ ] Bind the admitted32-GPU comparison's model, hardware, topology, runtime,
  exact workload hashes, source/table generations and all three repetitions.
  Record source metric values/units and all four relative errors, plus the
  intended64-GPU successor and its basic-validation receipt.
- [ ] Set promotion true only when EVERY finite error is<=0.30 in EVERY required
  repetition and all reference/coverage checks pass. Missing, undefined,
  unstable, mock or partial evidence means false/unverified. Do not use an
  average across metrics or repetitions. A native value of zero gives undefined
  relative error and cannot qualify.
- [ ] P0-H800-32 can unlock only the same model's P1-H800-64 formal campaign;
  P2-H200-32 can unlock only its P2-H200-64 formal campaign. The receipt must
  exist before formal64-GPU native submission, not be supplied retrospectively.
- [ ] A32-GPU case between10% and30% remains unfinished but may promote. Basic
 64-GPU validation/preparation has no30% prerequisite. Keep final gate.py
  threshold0.10 unchanged; the new30% predicate is a launch eligibility record,
  not a replacement final PASS.

### final_acceptance (DSV4 priority cases only; GLM paused)
- [ ] Publish a separate result for every eligible case containing native and
  Frontier values, all four relative errors, evidence references and verdict.
  Active scope is DSV4 across the four requested tasks,not an extra workload/
  topology sweep. GLM results remain paused,not falsely complete and not needed
  for DSV4 closure. Never claim completion for missing active DSV4 cases.
- [ ] Guard against accidental overfitting without expanding the matrix: compare all3 admitted repetitions from ticket18, retaining all128 exact-length completions and targeted correctness evidence. Independently warm required kernels/graphs/lazy communication pairs and prove coverage before future formal windows, with both roles idle between repetitions. Apply D-WARM-TARGET's paired event exclusion to retained runs uniformly and retain unchanged all-request readings; do not drop a repeat, select by latency, average away unstable admission or substitute deferred SharedGPT.
- [ ] Report any limitation that qualifies the verdict: an unmeasured shape, an unstable reference, a diagnostic that could not be collected. Labelling rules stay in force: `FAIL`, `INSUFFICIENT_EVIDENCE`, `COVERAGE_MISS`, `REFERENCE_UNSTABLE`, `BLOCKED_ON_DECISION`.
- [ ] Record the GPU-hours actually spent, including initialization, compilation, warm-up and failures.
- [ ] Verify affected shared infrastructure for the active DSV4 change and
  retain existing baseline/shared evidence. No new GLM simulation/tests/calibration
  or paired GLM/V4/Step completion gate. Existing reusable commits/evidence can
  be handed to the separate integration owner without restarting paused work.

## Acceptance

Every active DSV4(priority,hardware,size) result is recorded independently;
final PASS requires qualified inputs and all four warm-reading errors<=10% in every admitted repetition under D-WARM-TARGET, with unchanged all-request readings retained.64-GPU
formal runs additionally have a valid pre-launch30% promotion receipt and correct
resource class. Publish native/Frontier values, errors, sources and limitations;
neither basic validation nor30% promotion counts as final acceptance. Paused GLM
all-tier completion is outside current acceptance pending new explicit approval;
preserve archived verdicts and do not waive any active DSV4 criterion.
