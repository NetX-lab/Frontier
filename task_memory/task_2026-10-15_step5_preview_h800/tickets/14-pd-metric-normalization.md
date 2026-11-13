# 14: PD metric normalization through the existing gate

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-21 | D21: retain the single comparator/four10% gates across four tracks; paired32 release30 is a scheduling prerequisite, not a new final tolerance. |
| 2026-10-21 | D20-CLIENT executable normalizer/gate merged at e2be122 after parent183/148 passing checks and review closure; native/whole-ticket acceptance remains open. |
| 2026-10-21 | D20 explicitly approves shared client-metrics reuse and Step source/runtime adaptation under one owner; native validation remains open. |
| 2026-10-21 | Decision D15: the canonical THPT gate is prompt-plus-output `token_throughput_tps`; output-only throughput stays as a diagnostic mapping. |
| 2026-10-20 | Completed D5 T1 simulator endpoint code/tests in 6db2cd9 with exact legacy fidelity and no new known-failure IDs; native normalization/gate acceptance remains open. |
| 2026-10-20 | Recorded the D5 outcome: T1 is large, so the PD client TTFT must end at the D first token. |
| 2026-10-20 | Recorded decision D5: the client TTFT mapping carries the T1 first-token endpoint term. |
| 2026-10-19 | Completed source/design preparation and scoped parent verification; implementation/native/finalacceptance remains blocked by existing gates. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |

**What to build:** A verified native PD metric mapping through the existing normalizer and single gate, separating client, internal, readiness, decode, and throughput intervals while preserving historical metric meanings and refusing incomplete evidence.

**Blocked by:** `03` for tooling_preparation (satisfied); `07` and `08` for native_validation and whole-ticket completion.

**External prerequisites:** Tooling preparation may use03, the existing normalizer/gate and fixed metric definitions. Whole-ticket validation still requires07/08 native evidence and PD-specific pairing. Native metric tooling has no GLM simulator barrier. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.

**Status:** in-progress (D5 T1 endpoint and D20-CLIENT executable normalizer/gate complete); qualified native normalization and whole-ticket gate validation remain blocked

**D31 status (2026-10-25):** Active within plan steps E and G: the existing normalizer and metric scripts produce the comparison metrics.

**Current D21 metric use:** preserve client TTFT, TPOT, prompt-plus-output THPT and per-request E2E independently for each hardware/scale and each of three clean complete-population repetitions. Final acceptance remains four errors<=10%;32 all-four<=30% evidence releases only the corresponding64 formal work. Equality passes; no averaging across metrics/hardware/repetitions hides an error. Basic64 runtime has no30% prerequisite, and H20032 has no H800 accuracy edge. Reuse the existing normalize/gate path; do not relax its10% final gate or create another comparator. P0 H80032 comes first. `../plan.md` defines per-track readiness, manifests and resource policy; native pairing/admission is not waived. Targeted D5 diagnosis now belongs to28, not retired24.

**D20 implementation:** `step5-client-metrics` owns the existing canonical reader/normalizer/gate and Step binding. Reuse1400f12 and its common-workload dependency. Bind Step's real image/package/file identities and source-schema fields; do not label Step as vLLM0.28.0 or manufacture missing clean-run receipts. Execute the existing CLI with complete fixtures and negative controls while retaining native acceptance as a separate criterion.

**D20 delivered:** e2be122 integrates Step/GLM source-format selection, truthful Step runtime/image/source admission, common workload reading and the existing CLI. Parent183 client/native tests and148 legacy/internal tests pass (overlap); Step/shared CPU arithmetic examples compute all four metrics, and the shipped matrix returnsINSUFFICIENT_EVIDENCE without actual admission. Four criteria apply independently per repetition; pooled readings are diagnostic. One documentationP3 was corrected at15a3958. Evidence: `../test_report_2026-10-21_d20_client_integration.md`. Native G3 pairing, formal evidence and integrated fidelity/performance dispositions remain open; no whole-ticket box is closed by CPU fixtures.

**Decision D5 — 2026-10-20:** R1 established that D, not P, produces the first output token after the NIXL read and a 25-token tail recomputation. The T1 term of D5 (the gap between Frontier's P-completion `ttft` and the D first-step completion that native client TTFT ends at) is measured here when native PD timings are mapped, and the mapping decides whether Frontier's client TTFT uses the existing D first-step timestamp ([requirements.md](../requirements.md) decision D5 and [spec §8](../spec.md)). Historical `ttft_ms` keeps its internal prefill-completion meaning.

The implementation phase is active under `../requirements.md`. Routine in-scope instrumentation, incremental extensions and tests use the standing authority; preparation evidence does not satisfy native acceptance.

**Source workstreams:** S05, S09.
**Shared owners:** C06, with existing D-PD/D-ASYNC boundary evidence where applicable.

## Review disposition — 2026-10-18

`14.tooling_preparation` may start after03 in the existing gate/normalizer, with explicit unknown native boundary fields and independently computed fixtures. `14.native_validation` still requires07/08 and completes the whole ticket. Preserve old internal `ttft_ms` and prompt-plus-output throughput semantics.

## Acceptance

- [ ] Deliver the PD metric mapping, normalized native record set, independently recomputed interval examples, gate verification results, and missing-evidence ledger under §§7 and 8; validate after G3 without waiting for G4. Formal sampling/window/stability freezes remain ticket `23` outputs.
- [ ] Preserve historical `ttft_ms` as internal prefill-completion latency and `token_throughput_tps` as prompt-plus-output throughput; verify legacy values and consumers retain their original meanings while explicit new mappings use the same gate.
- [ ] Add distinct client TTFT, internal P-prefill completion, KV/PD-ready, D-first-step, request E2E/completion, TPOT, ITL, and output-only throughput definitions, with explicit token eligibility, numerator, interval endpoints, units, and clock provenance; do not guess formal measurement windows. The canonical THPT gate is the existing prompt-plus-output `token_throughput_tps` (decision D15); output-only throughput and request throughput are diagnostics on the same window.
- [ ] Compute client TTFT from submission to the first client-visible token on one client clock; establish P first-token emission/forwarding and D tail recomputation from G3 evidence. Internal-only timing cannot stand in for client TTFT or unmeasured proxy/CPU terms.
- [ ] Pair request/step/rank/role records through validated identities and causal relationships; never subtract bare cross-node clocks, inherit the old S33/pd_s33 endpoint offset, or select pairing modes by nth-forward order or DP×PP arithmetic without reference-specific evidence and approval.
- [ ] Validate PD-specific correlation beyond the single-role observer result, separating clean and observed records; valid observer pairs establish usable relationships, not clean performance truth, and trace-conditioned examples are not predictive comparisons.
- [ ] Preserve per-cell actual/predicted/absolute-error reporting and relative error only for nonzero actuals, leaving zero-actual relative error undefined. Represent the approved ≤10% mean/applicable-internal and conditional ≤15% p95 client-TTFT/ITL thresholds without inventing eligibility rules or an epsilon.
- [ ] Missing requests, ranks, pairs, or sufficient samples produce INSUFFICIENT_EVIDENCE; unstable reference repeats produce REFERENCE_UNSTABLE. Neither, nor an ineligible/unverified p95 result, can PASS; no cross-cell average replaces individual verdicts.
- [ ] Verify known small fixtures and deliberate missing/duplicate/mismatched records against independently calculated intervals, then verify qualified native records; fixtures prove tooling only, and native normalization alone does not establish Frontier accuracy.
- [ ] Metric-tooling changes satisfy §10's ≥50-scenario baseline/candidate fidelity comparisons and preregistered wall-time/memory budgets, preserving legacy metrics and key intermediates with measured over-budget dispositions; auxiliary errors at ≥32 GPUs remain non-blocking for healthy workloads while affected evidence stays unverified.

## D5 T1 simulator endpoint — 2026-10-20

Completed and committed as **6db2cd9** on `feat/step5-pd-structural`. PD-only CSV `client_ttft` uses existing `first_decode_token_completed_at` minus arrival, in milliseconds; ground-truth rows add `client_ttft_s`/`client_ttft_ms`. The similarly named `decode_first_token_completed_at` is the second output and is deliberately not used. Public `ttft`, `ttft_statistics` and historical `ttft_ms` remain prefill-completion intervals; the comparison/gate contract is unchanged. Nine new regressions and **764 affected tests** pass (eight Torch skips). Full unit: **5,581 passed / 70 skipped / 103 failed / 29 collection errors**, with all **132** failing IDs exactly matching the known stream-consolidation set. Step 4 cases/26 requests, GLM 2 cases/10 requests and legacy DeepSeek-v3 offline/online 2 requests verify the endpoint and preserved old values. Legacy versus `8585748`: **60 raw-identical + 17 additive-only PD cases; all 77 original outputs identical**, 902 artifacts; 179 PD requests independently reconstruct client TTFT. This completes only the simulator T1 endpoint code/test line and advances, rather than closes, this ticket's aligned-client-metric acceptance. Native pairing, explicit client normalization through the same gate, omitted client/proxy terms and clean-cache/provenance/performance qualification remain open. T2 still needs ticket 10 profiles and ticket 24 cells. Exact commands, schema semantics, fixture correction, evidence and limits: `/data/ycfeng/Frontier/.worktrees/step5-pd-structural/task_memory/task_2026-10-20_pd_client_ttft/test_report_2026-10-20_pd_client_ttft.md`. No GPU operation or publication.

## Completed preparation scope — 2026-10-19

`tooling_preparation` source/design preparation is complete in `../pd_metric_preparation_20261019.md`. Parent `../native_request_audit_20261019_01/preparation_document_checks_02.json` verifies its recorded source/document identities within the exact scope in `../test_report_2026-10-19_native_request.md`; no native or predictive acceptance is inferred. The document retains its original inspected snapshot, while current task status is recorded here. Whole-ticket completion still requires 07/08 qualified native PD endpoints and pairing. No acceptance checkbox is closed.
