# 17: Single-request PD handoff from reservation to release

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record approved D26/D27 source delivery and pending integrated workflow acceptance. |
| 2026-10-21 | D20 authorizes measured-mode PD CPU implementation through existing lifecycle owners; native tail/readiness and whole-ticket acceptance remain open. |
| 2026-10-20 | Completed D5 T1 simulator endpoint code/tests in 6db2cd9, verified Step/GLM/DeepSeek and exact legacy values; T2/native criteria stay open. |
| 2026-10-20 | Recorded the D5 outcome: T1 is large (align the endpoint), and T2's tail rule is 16–31 tokens. |
| 2026-10-20 | Recorded decisions D5 (assess the first-token and tail mismatch before aligning it) and D7 (Thinking Mode stays off). |
| 2026-10-20 | Merged the structural part into `feat/step5-preview` at `4b3f662`; published with PR #50 at `7266981`. |
| 2026-10-20 | Updated the structural result to `b875093`: the NIXL path requires equal PREFILL and DECODE attention TP, and the trace labels per-component bytes as transfer bytes. |
| 2026-10-20 | Recorded the structural result at `07bf337` on `feat/step5-pd-structural`; native-ordering acceptance boxes stay open. |
| 2026-10-20 | Released the structural part (Step PD handoff with dummy timing) under decisions D2 and D3. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved complete single-request PD ticket. |

**What to build:** Complete one controlled-workflow PD request through component capacity accounting, target reservation, READ, readiness, first decode, source release, and request completion inside existing owners.
Use controlled durations to prove the complete lifecycle without requiring the communication campaign or claiming measured communication accuracy.

**Blocked by:** structural part: `15.controlled_structural_path` and `06.simulation_scope` (decisions D2 and D3). Native-matched lifetime ordering and timing criteria: `08`. Trained-predictor acceptance: `09.full_frontier_path[GLM]`.

**External prerequisites:** Verified external09.full_frontier_path[GLM], complete 08/15 and a Step D-PD disposition are required. Evidence-based reuse of existing C03 semantics may proceed; a necessary material refactor waits for approval. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
Record distinguishing native evidence and reuse sufficiency in the existing D-* record. Routine incremental changes proceed under standing execution authority; only a demonstrated material core refactor or materially unresolved semantics/acceptance requires explicit approval.

**Status:** in-progress (D20 measured PD CPU implementation); native-matched and trained-predictor acceptance blocked

**Current D26/D27:** user approved complete P-scoped lifecycle/ownership/transfer
and separate Step live-SWA payload correction. D26 delivered e8d9c6c/d56f2dd/
68ab45c with fresh v2fixture: canonicalP6128+D16, semantic/client6144/1024;
READ/source-release preserved. D27 parent191 checks and independent No findings
confirm32 live SWA pages/139887104B per rank. Parent integration, new combined
fixture and strict77 remain pending. Physical allocation is not inferred from
wire bytes. Actual06 timeout remains historical failed fullCLI acceptance;
1024 outputs and final lifecycle still require completion within2400s.

**Current D20 assignment:** `step5-measured-pd` exclusively owns scheduler/transfer events and necessary Request/Batch changes. Implement supported measured-mode use of the existing lifecycle rather than deleting the NIXL guard without replacement or adding an alternate engine. Coordinate computation-provider interfaces with16/22's owner. CPU workflow tests may use labelled fixture costs; they do not establish real NIXL timing or native tail/readiness acceptance.

**Decisions D5 and D7 — 2026-10-20:** The native-matched part first assesses the error contribution of the two D-PD mismatches found by R1, each with its own verdict ([requirements.md](../requirements.md) decision D5 and [spec §8](../spec.md)). T1: native client TTFT ends at D's first output token after the KV read, while Frontier's `ttft` ends at P prefill completion; Frontier already simulates the transfer and the first D step. T2: native D recomputes the prompt tail (25 of 4105 tokens), while Frontier D computes one token. A term that changes the Frontier mean client TTFT and mean E2E by at most 1 percentage point of the native value in every assessed cell, and changes no verdict, is recorded and not fixed for now; a larger term is aligned within the existing transfer, D first-step and metric owners. The first two acceptance boxes close either with the alignment or with the recorded small-term measurement. Thinking Mode stays off for this task (D7); the Step structural `ValueError` for Thinking Mode is the intended harness boundary.

**Structural release — 2026-10-20:** Under the user's 2026-10-20 mainline decisions ([requirements](../requirements.md), [spec §6, §9 and §10](../spec.md)), the structural part starts now: Step5 on `pd-disaggregation` with P32/D32, each role TP8/DP4/EP32, dummy timing, reusing the shared GLM C16 PD commit `2da11c8` and DeepSeek C17 EP commit `dc7f382`. It covers component capacity accounting for SWA window pages, sparse raw KV, closed summaries and the open tail, reservation before READ, readiness, first decode and source release. Criteria that compare with native ordering wait for `08`.

**Structural result — 2026-10-20:** The structural part is implemented on `feat/step5-pd-structural` at `b875093` (not pushed, not merged). Step5 PD with dummy timing completes the single 4105/9 case and a 12-prompt boundary case, offline and online. It separates per-rank, per-lane, logical-unique and resident bytes. The op trace reports per-component transfer bytes as `transfer_bytes_per_rank` and only the total D residency. It reuses the C16 transfer events, the V1 allocator, the `(replica_id, dp_id)` lanes and the C17 EP waves. It rejects PD-AF, PP>1, prefix caching, MTP, Thinking Mode, illegal layouts and, on the NIXL path, unequal PREFILL and DECODE attention TP. The 77-case legacy matrix and the structural cases are identical to `c14e642`: 14 at `bf121c0`, and the 5 step5 and glm53_pd cases again at `b875093`. Evidence: [test report](../test_report_2026-10-20_step5_pd_structural.md). The acceptance boxes below stay open. Most of them require native-matched ordering, the native 25-token D recompute tail or native wire bytes, which wait for `08`. The 50-scenario fidelity and budget box is met for this structural commit, but the final native-matched change must meet it again.

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S06.
**Shared owners:** C03 and the existing allocator, transfer, scheduler, and request-event owners.

## Review disposition — 2026-10-18

NOT STARTED. C03 owns one component-capacity and pending/transfer/release implementation. Distinguish network completion, target readiness and source release only when native evidence affects scheduling or memory lifetime; no separate Step allocator/transfer manager.

## Acceptance

- [ ] A single request completes prefill, handoff, first D step, remaining decode, and request completion with observed ordering matched to the approved native lifetime contract.
- [ ] Capacity tracks raw sparse KV, SWA pages, closed summaries, open-region state and counters, and relevant scratch separately; logical unique, resident, and wire bytes remain distinct.
- [ ] Source retention has one pending owner, target reservation precedes READ where required, and differing source/target page identities retain correct component ownership.
- [ ] READ submission, network completion, GPU-visible readiness, and D admission are distinct where required; early network completion cannot admit a request with unreadable state.
- [ ] Region/page boundary and partial-tail cases preserve raw KV and summary integrity expectations, open-region counters, matched tokens, and the next decode crossing a region boundary.
- [ ] Approved tail recomputation is scheduled and charged before the appropriate D progress; D-first-step and P-free boundaries match evidence without freeing source pages prematurely.
- [ ] Completion leaves the expected allocator and pending-owner balances; missing summaries, divided raw-KV capacity, incorrect readiness, and premature release are rejected by meaningful negative checks.
- [ ] The implementation reuses existing lifecycle ownership without a second allocator or event system; recorded expected/actual controlled timings are labeled controlled workflow replay, not communication or predictive accuracy.
- [ ] At least 50 baseline-versus-candidate scenarios preserve unchanged fidelity under specification §10; preregistered wall-time/memory budgets are measured, with explanation and disposition for every excess.

Specification references: §§4–5, 9–10.

## D5 T1 simulator endpoint — 2026-10-20

Completed and committed as **6db2cd9** on `feat/step5-pd-structural`. PD-only `client_ttft` ends at existing first D output (`first_decode_token_completed_at`), not P completion and not the historically named second-output `decode_first_token_completed_at`. Historical prefill `ttft`, TPOT, E2E, event timing, allocation and token accounting are unchanged. Nine new regressions pass, including one output and the actual NIXL transfer-end/admission path: no D scheduling is possible before transfer completion. The final affected suite is **764 passed / 8 Torch skips**; full unit returns **5,581 passed / 70 skipped / 103 failed / 29 collection errors**, the exact known **132** failing IDs, no new/missing IDs. Step 4 cases/26 requests, GLM 2 cases/10 requests and legacy DeepSeek-v3 offline/online 2 requests preserve old metrics and verify the first D endpoint. All **77** legacy original outputs match `8585748` exactly; only the approved extra PD column differs in 17 cases. No production fallback, ordering guard or second lifecycle owner was added; one incomplete existing MLA fixture now supplies required `sys_arch`. This closes only T1's simulator endpoint code/test line and advances, rather than closes, aligned TTFT/native acceptance. Native lifetime/wire/content qualification, client/proxy terms and whole-ticket criteria stay open. T2 requires ticket 10 measured attention/MoE profile pricing of the observed 16–31-token tail and ticket 24 cells. Exact commands and limitations are in `/data/ycfeng/Frontier/.worktrees/step5-pd-structural/task_memory/task_2026-10-20_pd_client_ttft/test_report_2026-10-20_pd_client_ttft.md`; no GPU operation or native predictive claim.
