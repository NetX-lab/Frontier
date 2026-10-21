# 08: G3 native PD content and lifetime qualification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Verify the canonical capture/abort partial slice and select existing-D independent full-local execution within the locked native reference. |
| 2026-10-21 | Assign case-aligned capture/reference implementation in step5-canonical-g3, reusing corrected R2 mechanisms. |
| 2026-10-21 | D21: case-aligned native evidence serves P0 H80032 first and the other independent tracks; obsolete R2 capture stays paused, no G3 waiver. |
| 2026-10-21 | R2 P1 review closed; old2048-length capture paused because canonical6144/1024 and independent referenceG3 remain uncovered. |
| 2026-10-21 | D18 resume: judged sealed H800 R2 as diagnostic FAIL despite platform success; retained raw verdict and identified observer/validator defects for correction. |
| 2026-10-21 | D17 defers SharedGPT short-prompt follow-up; canonical observation targets 6144-token prefill and 1024-token completion. Corrected the unsupported extrapolation below 16 tokens. |
| 2026-10-21 | Recorded that 122 of the 256 canonical-workload prompts are under 32 tokens, outside the observed NIXL tail-rule range; short-prompt P/D behavior is unmeasured. |
| 2026-10-20 | Prepared R2 code/stage02 after 560 harness passes and 77 identical fidelity cases; native execution and existing acceptance checkboxes remain open. |
| 2026-10-20 | Recorded the H800 R1-with-warm-requests result (`exp-1020-195546-730923`): W1 nine of nine PASS, and the D5 T1 and T2 evidence. |
| 2026-10-20 | Recorded decision D5: ticket 08 runs supply the first native estimate of the first-token and tail terms. |
| 2026-10-20 | Merged the native P/D harness and R1 tooling into `feat/step5-preview` at `bd58493`; published with PR #50 at `7266981`. Next: H800 R1 with `tests/integration/step5_pd/r1_run.json`. |
| 2026-10-20 | Recorded run R1 on H200 (`exp-1020-164750-497812`): first native P/D request over NIXL completed, 9/9 W1 conditions PASS; H200 evidence only, H800 R1 still required. |
| 2026-10-20 | Changed the native prerequisite to `06.simulation_scope` (decision D2) and released the ticket for mainline native PD work. |
| 2026-10-19 | Recorded completed source preparation and four acquisition gaps; native G3 still blocked by complete06. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |

**What to build:** An S03.G3 native PD qualification package that proves raw KV, side-summary, and tail integrity from source retention through target readiness and release, and supplies evidence to the existing shared decisions.

**Blocked by:** `06.simulation_scope` (decision D2). The complete original 06 census is not required.

**External prerequisites:** Accepted `06.simulation_scope` evidence and the locked native reference are required. Native PD evidence may precede D-PD/D-ASYNC/D-DP-PD dispositions and has no GLM simulator barrier. A different native functional reference requires the applicable conditional decision. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.

**Status:** in progress (R1/D5 evidence complete; R2 collector P1 correction/review closed; canonical content/readiness/lifetime and independent reference qualification remain open).

**Latest implementation:** capture/abort source7d46abe and recordse14e7df pass658 parent CPU checks on an immutable snapshot. Existing-D full-local6144/1024 execution without transfer metadata is the selected independent functional path; this retains the locked image/model/connector and does not activate conditionalD-S5-REF. Full resident identity, actual state/output comparison, reset/reuse and explicit capacity-pressure/component/alias conservation are still being implemented. No complete stage, native G3 or formal admission is claimed. See the implementation disposition in requirements and `../../task_2026-10-21_step5_priority_runs/canonical_g3_parent/`.

**Current D21 scope:** qualify content/readiness/retention/lifetime and independent native reference for6144/1024, **P0 H80032 first**.32 total=P16/D16, each TP8/DP2/EP16/PP1;64=P32/D32 TP8/DP4/EP32/PP1. Bind distinct hardware/scale evidence and assess existing observations before repeating. G3 remains required before formal work; basic64 runtime may proceed now. H80064 formal waits for own readiness and H80032 four clean errors<=30%; H20064 waits for H20032, which is independent of H800. Final10% remains. Follow `../plan.md` D21 policy: H800 non-preemptible codesign; all personal nonterminal H200 non-preemptible reservations across sessions<=32, every additional job preemptible, H20064 wholly preemptible. Lower-priority work cannot starve P0. Dated records below do not restore old caps or complete06 waiting;06.simulation_scope is the live prerequisite.

**Current D20 disposition — 2026-10-21:** corrections throughcb09a27 pass636 recorded checks and independent P1 follow-up; preserved old-run verdict04 isUNVERIFIED, notG3PASS. Offline stage02 at`/data/ycfeng/tmp/step5_d18_r2_h200_20261021/stage02` has39 matching manifest files/38 tool pins, but max_model_len2048 and prompts255–1537/output17/128/256. It cannot exercise canonical6144/1024, and reference_census is only intra-run consistency. Parent pauses this old capture under D20. Remaining evidence is the existing canonical content/retention/readiness and independent-reference criterion; none is waived or replaced by CPU transfer input. See R2 correction worktree`task_memory/task_2026-10-21_step5_r2_correction/test_report_2026-10-21_r2_h200_preparation.md` and integration records.

**R2 judgment — 2026-10-21:** `exp-1021-000521-476638` succeeded on four tasks. The immutable stage02 checker reports runtime, placement/request association and planned cancellation PASS; lifetime6 failures and tail-attribution60 failures, while content/mapping/readiness remain UNVERIFIED. All32 workers hit the same complete-row capture exception;69 registered layer views lack a direct region-name match. The checker additionally rejects null remote identities on local allocation records and applies tail subtraction to an already reduced P prefix. Preserve the raw FAIL and correct the observer/validator from native source; these findings do not establish actual native corruption or G3 acceptance. Evidence: `../scratch_evidence/d18_resume_20261021/r2_verdict02.json` and `r2_check_summary.json`; next correction must not mutate the submitted stage or manufacture missing content.

**D17 workload disposition — 2026-10-21:** SharedGPT and its 122 short-prompt requests are deferred. Native behavior below 32 tokens remains unmeasured; the earlier extrapolation claiming zero transferred tokens below 16 was invalid (the formula becomes negative). Do not add that campaign as a prerequisite of the fixed-length case. Canonical requests now have 6144 prompt and 1024 output tokens. The observed-range rule `16 × floor((prompt − 16) / 16)` predicts 6128 loaded tokens and a 16-token local tail at this length; verify actual KV/summary content, transfer count, D recomputation and release lifetime on the native path. Existing R2 evidence remains relevant and its verdict is still required.

**H800 R1 result — 2026-10-20:** `exp-1020-195546-730923` (job tag `r1wh80002`) ran on four H800 nodes and the platform reported `Succeeded`. The nine W1 conditions all pass on H800: placement, startup, prefill response, decode output, NIXL counters, decode prompt sources, release, ordering and exit. The sixteen warm requests gave the D5 evidence: T1 (from P prefill completion to the client's first token) is 6.7% to 62.0% of client TTFT with a median of 20.5%, which is large, so the PD client TTFT endpoint must move to the D first token; T2's tail is `prompt − 16 × floor((prompt − 16)/16)`, that is 16 to 31 tokens, for prompts from 128 to 7681 tokens. Details: [preparation and result report](../test_report_2026-10-20_native_pd_h800_r1w_preparation.md), section "Execution result on H800". Remaining: R2 content and lifetime checks, and concurrent placement evidence.

**Decision D5 — 2026-10-20:** Ticket 08 runs record, per request, the client first-token time, the P response time at the proxy, and D's externally loaded and locally computed prompt tokens. Warm requests (after the first request of a run) give the first native estimate of the T1 and T2 terms ([requirements.md](../requirements.md) decision D5 and [spec §8](../spec.md)). R1's single cold request is not used for this estimate.

**Mainline scope — 2026-10-20:** Under the user's 2026-10-20 mainline decisions ([requirements](../requirements.md), [spec §6, §9 and §10](../spec.md)), this ticket starts once `06.simulation_scope` closes. First deliver one complete native P/D request over NIXL on the smallest legal roles, with the request completing on D and the transfer path and timestamps recorded; then extend to the remaining content and lifecycle criteria. Reuse the existing native drivers instead of adding new observation tooling.

Implementation is active under `../requirements.md`. Source/test preparation proceeds independently; native G3 uses06.simulation_scope and case-aligned evidence, not the complete on-demand census. Existing owners remain. The maxlen2048 R2 stage stays paused under D20/D21; no artifact approves a new architecture or replaces native qualification.

**Source workstreams:** S03, S06.
**Shared owners:** C03, C05; existing D-PD, D-ASYNC, D-DP-PD evidence consumers.

## R2 preparation — 2026-10-20

Code/staging is completed at `1696916` and `90559d8` on `feat/step5-native-pd`. Final package is `/data/ycfeng/tmp/step5_native_pd_r2_h800/stage02`, job tag `r2h80002`; stage01 is superseded. The 20-request/15-wave diagnostic workload captures complete transferred KV/summary rows on twelve requests and native no-fence readiness on eight, including six concurrent requests. Independent scheduler logical mapping, native per-request immediate/deferred free counts, GPU-step release guards, local-clock release intervals, boundary/decode, placement/load and decode-cancel/reuse evidence have offline consumers.

All 560 harness/fault tests pass, 77 fidelity cases and clean-cache filenames match, and stage02's 35 tool files, 27 original source pins, ten hook pins and four schema pins verify. The same phase reviewer confirmed both P1 fixes. No GPU/platform/cloud operation occurred. [Report and parent-only commands](../test_report_2026-10-20_native_pd_r2_preparation.md) records evidence and limits. This is preparation, not G3 acceptance: native/reference continuity, actual pressure/lane/overlap/reuse and pending-READ abort outcomes remain unverified where not observed. No checkbox below is changed.

## Historical review disposition — 2026-10-18

NOT STARTED; blocked on06. Prepare state-content and ownership checks from existing sources, then qualify the actual connector path. Generic NIXL environment success, candidate connector allowlists and byte counts do not prove Step raw/summary/open-region/tail transfer.

## Acceptance

- [ ] Deliver S03.G3 with nested G0–G2 evidence, independently checkable content/lifecycle case results, measured placement, and same-decision evidence cards under §§5, 6, and 9; own the minimum observations needed without waiting for observer ticket `07` or full profiling.
- [ ] Exercise complete native text P/D with dummy weights and required graph/chunk behavior; identify the actual CUDA-buffer NIXL path, descriptors, rank mapping, and raw-KV/side-summary registration. Neither a generic connector guard nor candidate-source claims establish support.
- [ ] Compare raw KV and side-summary contents, matched tokens, and D's first scheduled tokens; establish open-tail transfer, retention, or recomputation from observations. Known-state checks are separate from performance evidence, and equal seeds alone do not prove P/D parameter/buffer equivalence.
- [ ] Cover locked region/page boundaries before/on/after, partial tails, the next decode crossing a region, differing source/target local page IDs, and actual SWA transferable ranges; preserve full-history sparse KV rather than dividing it by region size.
- [ ] Verify target reservation precedes READ where required, network completion is distinguished from GPU-visible readiness, all required state is ready before D proceeds, and source pages remain occupied until the evidenced release boundary.
- [ ] Demonstrate page reset/reuse and target-capacity pressure with explicit component/alias ownership, resident versus wire bytes, one source pending owner, and no leaked or double-released resources.
- [ ] Extend a complete single-request handoff to concurrent transfers, reuse races, cancellation, and abort conservation; account for every request, token, transfer, and retained/released component.
- [ ] Record actual endpoint/DP-lane placement, P first-token emission or forwarding, D tail work, and ready/free ordering; submit distinguishing evidence to the same D-PD/D-ASYNC/D-DP-PD decisions without treating observations as approval of new Frontier semantics.
- [ ] Reject missing summaries, wrong content/page mapping, premature readiness/free, incomplete request/rank records, and byte-count-only or exit-only success. Native defects produce BLOCKED_ON_RUNTIME with a minimal reproducer, never a Frontier fix or weakened native requirement.
- [ ] Qualification-tooling changes meet §10's ≥50-scenario fidelity checks and preregistered wall-time/memory budgets with over-budget dispositions; at ≥32 GPUs auxiliary validator errors leave evidence unverified and notify the user without stopping healthy workloads solely for harness failure.


## Source preparation disposition — 2026-10-19

[Native PD source preparation](../native_pd_source_preparation_20261019.md) records the actual model guard, mean-only side-storage view, partial-region continuity, MRv2 connector call boundaries and shared C16 reservation/readiness/release owners. Parent corroborated51/51 ledger hashes and56/56 local links before task records changed; citations identify that read snapshot. Complete08 remainsblocked on06 with0/10criteria accepted. Generic allowed names, byte counts and the old buffered P2pNccl proxy do not establish native NIXL, open-tail continuity or streaming clientTTFT.

Four pending acquisition units reuse the existing locked-image collection owner: (1) MRv2 connector wrapper and pre/finish/post-forward definitions, (2) reachable configured factory/config/Nixl/MultiConnector and side-storage consumers, (3) scheduler reservation/free/reset and handoff metadata, and (4) native PD entry/proxy/client output. Resolve actual bodies/hashes before instrumentation. Historical namespace-filtered member lists do not establish installed-package absence. Shared C03/C05/C06 remain canonical; no duplicate connector, normalizer, gate or workflow. Preparation may proceed during06; native calibration and unresolved D-PD/D-ASYNC/D-DP-PD decisions retain their gates.

## Run R1 on H200 — 2026-10-20

The first deliverable ran on H200 because H800/codesign had no whole free node
([hardware decision](../native_pd_20261020/hardware_decision_r1.md)). Job
`exp-1020-164750-497812` (4 × 8 H200, step_main, non-preemptible) ran P and D each as
TP8/DP2/EP16 on the locked image with dummy weights and ended `Succeeded` after 9 min 20 s,
at most 4.98 GPU-hours. One 4105-token request was prefilled on P lane 1 and pulled over NIXL
by D lane 1 on another node; D loaded 4080 prompt tokens from the remote KV, computed 25 and
returned 9 tokens. NIXL moved 817,631,232 bytes in 2,944 descriptors with no failed transfer,
failed notification, expiry or late notification, and every engine's KV usage returned to 0.
All 9 W1 conditions of design01 §3.6 PASS; the two-request W2 also matched its expected
prompt split (6128 loaded, 41 computed). Evidence and limits:
[test report](../test_report_2026-10-20_native_pd_r1.md), stage
`native_pd_20261020/r1_h200_01/`, harness commits `4a55513` and `d77e155` on
`feat/step5-native-pd`.

This is H200 evidence only (D4). Acceptance criteria remain 0/10: R1 covers the request path
and vLLM counters, not KV page or side-summary content, boundaries, readiness versus network
completion or the release boundary. The UCX transport was not recorded. Next: R1 on H800 when
four whole nodes are free (same stage builder with `r1_run.json`), then the R2 content and
lifetime design.
