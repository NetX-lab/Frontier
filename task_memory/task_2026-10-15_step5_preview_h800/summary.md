# Phase summaries: Step-5-Preview planning and execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Update the current status: H200 64-GPU native done. |
| 2026-10-26 | Update the current status: H200 32-GPU native done, H200 profiles merged, step E running. |
| 2026-10-25 | Add the current status and document index after the D31 mainline reset. |
| 2026-10-22 | D25 full H80032/64 workloads complete, both432/432 verified; p90/throughput and remaining decode instability archived. |
| 2026-10-22 | Consolidated clean145a2f5 passes77/77 exact fidelity; P0 G3 capture finishesSucceeded4/4, native judgment pending. Four-track8-request diagnostic p90/throughput independently recomputed. |
| 2026-10-22 | Integrated G3/reference1171ab6 and MRv2 producer82f4452 at145a2f5; parent merged suites685/465PASS. Submitted32-H800 capture exp-1021-235955-532667; native judgment and consumer alignment pending. |
| 2026-10-21 | Record bc8ae27 integration,323/197 parent checks, final-source fidelity limits and native generation03 per-lane FULL replay. |
| 2026-10-21 | D21 code/native execution and D22 strict native PASS recorded:274 compute and204 native parent checks;P0 basic PARTIAL;full actual-main idle-DP binding still active. |
| 2026-10-21 | D21-DOCS: archive four-track priority/dependency update, global H200 non-preemptible cap32, terminal receipts and superseded package intent; final acceptance remains open. |
| 2026-10-21 | Bounded communication startup correction6a9cedc integrated asb68ce4d;101 parent tests and189 package pins pass. Old H20001 fails141 before measurement; unchanged32-GPU case02 exp-1021-143708-523626 isPending under96/130 reservations. Exact commands/evidence: test_report_2026-10-21_d20_comm_startup.md and scratch_evidence/d20_comm_startup/. Native quality remains unverified; measured compute/PD implementation continues. |
| 2026-10-21 | D20-CLIENT executable comparison accepted at e2be122: parent183/148 passing suites, Step/GLM CLI arithmetic and shipped-matrix evidence rejection; reviewP3 corrected. Full16/17/22/28 and real native calibration remain open. See test_report_2026-10-21_d20_client_integration.md. |
| 2026-10-21 | D20 approved: CPU implementation released from GLM waiting; compute/bundle, measured PD and client-metric code streams assigned. Pinned collective-sim builds and executes locally; formal native/profile acceptance remains open. |
| 2026-10-21 | Mainline checkpoint completed; CPU22/28 preparation assigned with existing gates intact. One-cell timing probe exp-1021-130839-908503 completed at exit0; causal judgment pending. See `checkpoint_2026-10-21_mainline.md`. |
| 2026-10-21 | D19: raised H200 cap130; qualification exp-1021-125354-923219 queued at96 total reserved GPUs, identity/mount/RDMA verified; formal admission open. Evidence: `test_report_2026-10-21_h200_qualification_submission.md`; independent mainline checkpoint delegated. |
| 2026-10-21 | Record native BF16 fix and six remaining SWA median failures; next measurement is a causal single-shape diagnostic. |
| 2026-10-21 | Native db5c99f verified585 PASS;64-GPU submission waits for32-GPU reservation; R2 retirement finding requires correction. |
| 2026-10-21 | Archive reviewed profiling integration and completed H200 probe; retain native stage02 and R2 dependencies. |
| 2026-10-21 | Integrated D18 communication source and submitted the first-prefill H200 diagnostic; R2 correction, compute and native consumer work remain active. |
| 2026-10-21 | D18: adopted common68f891b as958d81a; artifact agreement complete, native/profile/communication execution active. |
| 2026-10-21 | D17: active workload changed to fixed 6144/1024 tokens, 128 formal requests at QPS 1.0; SharedGPT and short-prompt follow-up deferred, package verification and handoff recorded. |
| 2026-10-21 | Recorded D16: shared workload adopted (`cb8090b`), retiming = historical linear rescale, no seed. |
| 2026-10-21 | Recorded D15: canonical workload shared with the GLM/V4 session — 256 requests, prompt-plus-output THPT, trace SHA-256, retiming rule open for one decision. |
| 2026-10-21 | Recorded the calibration scope reduction (D9–D14): one canonical 64-H200 SharedGPT case with 128 requests at QPS 4.0, TP8/DP4/EP32 per role, MTP off, measured backend, four metrics gated at ≤10%; tickets 24–27 retired, 09 retargeted to H200-64, 23 and 28 rewritten. |
| 2026-10-21 | Archived S1/S2 read-only judging, candidate byte/ms curves/DP4 payloads and 535 CPU passes; selected queue cells fail and stable/isolated/held-out/admission gaps remain. No GPU control or production change. |
| 2026-10-21 | Archived UTC October 20 R2 code/stage02 preparation, 560 tests, 77 identical fidelity cases and parent-only execution handoff; native/reference gates remain open. |
| 2026-10-20 | Recorded decisions D5–D8, PR #50 at `8585748`, and the D8 cleanup with the new evidence location `scratch_evidence/`; no user decision is open. |
| 2026-10-20 | Summarized the mainline day: ticket 06 simulation scope closed, shared-branch merge, structural PD, native P/D R1 on H200, MoE profiler pilot, stream consolidation and PR #50 at `7266981`. |
| 2026-10-20 | Published Draft PR50 at9ebf4cf; verified local/remote/PR head identity and unchanged201native inputs. |
| 2026-10-20 | Recorded corrected _04 freeze and dated H800 observations plus PR-only authority; implementation and qualification remain open. |
| 2026-10-19 | Accepted FIFO/None source and CPU correction after independent review fixes; retained H800 Failed evidence, 626 distinct affected checks and unchanged native gates. |
| 2026-10-19 | Recorded terminal H200 timeout and accepted failure archive source/CPU correction after 664 affected passes; native gates unchanged. |
| 2026-10-19 | Accepted node-create/child-env harness correction after613affectedpasses; preserved Failed generation and native gates. |
| 2026-10-19 | Recorded quota-only H800 stopped generation and accepted minimal H200 resource extension after451CPU passes; native acceptance unchanged. |
| 2026-10-19 | Accepted minimum writer/FP8 scale source bindings after608CPU passes; native gates unchanged. |
| 2026-10-19 | Recorded allocator source/CPU acceptance and minimum native writer/scale continuation; native ticket gates unchanged. |
| 2026-10-19 | Accepted capture-node source/CPU tooling (461 tests) and separate H800 API smoke; native06 remains1/10 and whole15 remains0/8. |
| 2026-10-19 | Accepted resident-parameter candidate01 source/CPU tooling with657affectedtests,111CPUAPI and1554applicability checks;native06/whole15 unchanged. |
| 2026-10-19 | Accepted profiler candidate02 source/CPU tooling:498 affected passes, locked CPU API04 and parent two-axis reviews; native06/whole15 and formal gates unchanged. |
| 2026-10-19 | Accepted native state candidate01 source/CPU tooling:497 joint and140 affected caller passes overlap; parent Standards and independent Spec No findings; native06/whole15 unchanged. |
| 2026-10-19 | Accepted native census candidate02 source/CPU slice:419 affected passes, two-axis findings resolved, complete encoded record cap and unchanged retained simulator evidence; native06/whole15 unchanged. |
| 2026-10-19 | Synchronized final independent lifecycle review: 35 scoped checks and final artifact identities; initial 33-check bytes retained, acceptance unchanged. |
| 2026-10-19 | Archived completed request_device_20261019_01: 26,917 device checks and 35 scoped lifecycle checks PASS; current missing work is precision/physical/cache-state census, with06 1/10 and whole15 0/8 unchanged. |
| 2026-10-19 | Accepted candidate03 device/request/completion source and CPU tooling after final Spec follow-up, 357 affected passes and 234 source checks; native gates unchanged. |
| 2026-10-19 | Archived reviewed frontend source/CPU tooling, 275 affected passes and unchanged whole-ticket/native gates; native worker/device witness path remains active. |
| 2026-10-19 | Archived completed sole request run, 24,682 checks and five separate observed outcomes; accepted only06criterion7, preserved forced native shutdown/clean-performance limits and completed07/14/29 preparation. |
| 2026-10-19 | Froze and submitted the sole H800 request generation; preserved exact job/resources and continuing native evidence audit without full-G2 credit. |
| 2026-10-19 | Accepted the separate reviewed CPU request caller capability with63ownerPASS; native06 remains0/10 and its newgeneration is still unexecuted. |
| 2026-10-19 | Archived generation05 controlled structural acceptance: 646 tests, three CLI requests, 77 legacy cases and eight CPU metrics WITHIN; native and formal model/hardware gates remain open. |
| 2026-10-19 | Archived canonical shared merge generation03,1142checks/77fidelity and explicit dense-init/cache limits; restored H800-first structural continuation. |
| 2026-10-18 | Archived accepted Ticket15 representation,738checks/77fidelity, preserved three CPU overruns and official cache limits; implementation remains active. |
| 2026-10-18 | Archived the documentation-review deliverables, evidence-backed status, main-path recovery and next-session ownership; no implementation or campaign was started. |
| 2026-10-18 | Archived completed CPU evidence checkpoint, exact fidelity/UNVERIFIED performance disposition, verified commits and requested execution pause with resume handoff. |
| 2026-10-18 | Archived Attention raw audit, TP16 actual capacity RCA and verified UUID comparison correction e10e030; native qualification and clean measurement remain open. |
| 2026-10-18 | Archived completed FFN raw/weight/math evidence and four independent review findings per initial driver, with parallel CPU corrections in progress. |
| 2026-10-18 | Recorded restored native execution, successful original FFN generation02 and independently owned Attention/TP16 integration in progress. |
| 2026-10-18 | Archived primary05/06 source-design closure and model-lock failure disposition; native integration/execution remain open. |
| 2026-10-18 | Archived independent original-request RCA: final INVALID_REFERENCE isolated to retired Pi env whitelist, with no native/platform execution or input change. |
| 2026-10-18 | Archived the public review-access inspection: receipt schema excludes actionable worker prose, no public review endpoint, frozen input consistency preserved. |
| 2026-10-18 | Archived dependency-based parallel CPU tooling corrections, fresh simulator verification and explicit unresolved native/baseline-drift limits. |
| 2026-10-17 | Archived failed capacity02 after native all-rank model loading and corrected independent FFN preparation. |
| 2026-10-17 | Archived complete G1 acceptance after new resource/identity verification;05/06 ready and preparation continues. |
| 2026-10-17 | Archived nine accepted G1 criteria, completed native-content/sentinel analyses and the remaining mandatory-recording blocker. |
| 2026-10-17 | Archived successful SWA lifetime verification, stage references/fresh fidelity, seven accepted G1 criteria and binding-only FP8 observation. |
| 2026-10-17 | Archived04F DSA execution/storage/selection evidence and unresolved SWA replay anomaly; isolated metadata ownership verification is active. |
| 2026-10-17 | Reconciled accepted04E, committed worker/checker generations and explicitly historical status paragraphs. |
| 2026-10-17 | Archived three execution-owner audits, preserved partial-source failure, and froze native FFN execution subset before launch. |
| 2026-10-17 | Accepted standalone runtime binding and collected fourteen native FFN/dummy owner sources. |
| 2026-10-17 | Archived reviewed sparse input fixes and preserved original worker; first runtime-binding diagnostic failed in harness before native imports. |
| 2026-10-17 | Accepted 04D after independent native-output audit and collected installed runtime owners; G1 remains incomplete. |
| 2026-10-17 | AcceptedH80004C72exactcases/216localgraphs/eighthostnegatives; subsequentH200worktemporarilydeferred. |
| 2026-10-17 | Recorded concurrent H800 RDMA recovery/H200 two-node submission and confirmed H200 quota blocker; execution remains incomplete. |
| 2026-10-16 | Archived04B64-case exact native results and two-node Socket communication; RDMA remains unqualified. |
| 2026-10-16 | Archived accepted 04A native results and platform recovery; retained original failed checker verdict and next executor boundary. |
| 2026-10-16 | Recorded the exact G1 platform certificate/volume-mount blocker after inspecting replica events. |
| 2026-10-16 | Recorded accepted multi-GPU diagnostic content and G1 tooling correction/submission, with native acceptance pending. |
| 2026-10-15 | Archived the completed, user-approved planning package and verified local tickets. |
| 2026-10-16 | Archived tickets01–02 acceptance, explicit-proxy result and preserved UNVERIFIED timing outcomes;03prepared, not dispatched. |
| 2026-10-16 | Recorded ticket03 returned inventory and actual26-file source reconciliation; primary phase acceptance is in progress. |
| 2026-10-16 | Closed03after reviewer acceptance of corrected source/design semantics;04ready, native qualification unstarted. |
| 2026-10-16 | Recorded actual CUDA initialization failure, user-deferred driver work and independent04CPU/source preparation. |
| 2026-10-16 | Added82planned G1 CPU/source reference cases and local mathematical/source verification; phase review is pending. |
| 2026-10-16 | Archived the passing CUDA forward-compatibility rerun and the process-start launch requirement for native work. |


## Current status and document index — 2026-10-25 (D31)

This directory, on branch `feat/step5-preview`, is the main document index for the Step-5-Preview
task. Read these first:

- `requirements.md`: decisions, newest first. The current one is D31.
- `plan.md`: the top section "Current plan — D31 mainline reset" is the active plan.
- `review_2026-10-25_mainline_reset.md`: why the plan was reset, and the review of the completed
  tickets.
- `progress.md`: the newest entries are at the top.

### Status in plain words

Status as of 2026-10-26:

- **Native reference results.**
  - H200 32 GPUs (step C, routing image): TTFT 1,083.7 ms, TPOT 61.7 ms, E2E 64.25 s, total
    throughput 5,214.6 tok/s (`evidence/d31_h200_32_native_metrics.json`).
  - H200 64 GPUs (step F): TTFT 1,381.3 ms, TPOT 69.6 ms, E2E 72.6 s, total throughput
    5,005.5 tok/s (`evidence/d31_h200_64_native_metrics.json`,
    `test_report_2026-10-26_h200_64_native_workload.md`). NIXL READ cost 56.55 ms
    (`evidence/d31_h200_64_nixl_read_cost.json`).
  - The D25 H800 runs (`../task_2026-10-22_step5_full_workload/`) used the original router, so
    they are not comparison targets.
- **Frontier.** The full dummy-timing workload completes in 49 minutes (step B). Step E runs the
  `h200_32` workload with the measured H200 tables.
- **Operator profiles.** The H200 profile tables for both roles are measured and merged (run 03,
  `test_report_2026-10-26_h200_profiles.md`). The first runs on them found five defects in the
  profiler and predictor; all are fixed (`ae4c9312`, `39a82403`, `dd1f37dc`, `6790f110`,
  `d034202c`).
- **Next.**
  1. Record the step E error table against native.
  2. Calibrate in order: workflow, then operators, then CPU overhead.
  3. Run the Frontier `h200_64` workload (step G) after step E is within 30%.

### Sub-task directories (`task_memory/`)

| Directory | Content | State |
| --- | --- | --- |
| `task_2026-10-22_step5_full_workload` | D25 native H800 32/64 workload runs and metrics | Done (original router) |
| `task_2026-10-22_step5_position_routing` | D29 `position_cyclic` in Frontier, collective pricing, main09 attempt | Parked by D31; the merged code stays |
| `task_2026-10-22_step5_d29_native_balanced` | D29 native patch and derived image `f037d958` | Image published, never run on a GPU |
| `task_2026-10-22_step5_finish_sync` | D30 finish synchronization | Merged |
| `task_2026-10-22_step5_prefill_target` | D26: P computes 6128 tokens, D computes the 16-token tail | Merged |
| `task_2026-10-22_step5_swa_pages` | D27 effective SWA transfer pages | Merged |
| `task_2026-10-22_step5_workflow_fidelity` | Legacy-scenario fidelity after the Step changes | Done |
| `task_2026-10-22_step5_mrv2_{forward,inputs,load,main,perf}` | D23 first-prediction agreement and actual-main diagnosis | Merged or archived |
| `task_2026-10-22_step5_root_metrics` | D23 root metrics | Done |
| `task_2026-10-22_step5_reference_values`, `task_2026-10-22_step5_d24_stage`, `task_2026-10-22_step5_d28_d24_pressure_stage` | D24 and D28 native state diagnostics | Parked by D31.4 |
| `task_2026-10-22_step5_http_capture`, `task_2026-10-21_step5_client_metrics`, `task_2026-10-20_pd_client_ttft` | Client-side metrics and TTFT definition | Done |
| `task_2026-10-22_exact_prefill_cap` | Exact prefill engine cap | Done |
| `task_2026-10-21_step5_priority_runs` | D21 stager for the four hardware and scale cases | Done; reused by steps C and F |
| `task_2026-10-21_step5_measured_compute`, `task_2026-10-21_step5_measured_pd`, `task_2026-10-21_step5_d20_integration` | D20 measured compute and PD consumer code, with CPU fixtures | Merged |
| `task_2026-10-21_step5_profile_integration`, `task_2026-10-21_step5_p0_profiles`, `task_2026-10-21_step5_d22_timing` | Profiler code and profiling admission attempts | Code merged; reused by step D |

### Unmerged worktree branches

There are 101 Step-5-related worktrees under `/data/ycfeng/Frontier/.worktrees/`. All except 30
are merged into `feat/step5-preview`. Of those 30:

- 28 hold D29 or D30 records, older candidates of code that was merged later, or verification
  records. Examples are `records/d29-*`, `verify-step5-pricing-0[1-6]`, `fix/d29-plan-key` and
  `step5-d29-named-query`. They are parked and are not needed for the D31 plan.
- 2 are detached `fidelity-candidate-*` worktrees from an older line, about 300 commits each, with
  merge base `71cde5eb`.
- `feat/step5-pressure-capacity` and `feat/step5-r2-correction` change only test-integration
  helpers for the parked D28 and R2 work.

No worktree is deleted without the user's approval (D31.8).

## D25 full H800 workload completion — 2026-10-22

Both requested H80032/64 workloads completed successfully: three128-request measured windows after16 warmups per round,432/432 requests per size. Pooled p90 TTFT1.760660/1.911773s, TPOT87.792540/93.557086ms, E2E91.639790/97.515587s; three-window mean total throughput5101.423/4999.053tok/s. All measured TTFT values are below5s. Native/accuracy qualification remains unverified; in-measurement JIT and first-round decode slowdown prevent a fully steady-state claim. Complete evidence, commands, definitions and warmup/historical TTFT analysis: [D25 summary](../task_2026-10-22_step5_full_workload/summary.md).

## Current D21 documentation delivery — 2026-10-21

**Parent execution update:** measured code is integrated at `bc8ae27`, with323 combined and197 profiling/timing checks passing. Legacy execution completes77 cases and preserves297 non-schema artifact pairs byte-for-byte; strict config/metadata additions and documentation-dirty provenance remain recorded in `d20_fidelity_schema_audit02.json`. Full offline/online measured request acceptance remains active. P0 generation03 (`exp-1021-205523-285491`) Succeeded, completes24 exact6144/1024 requests and both idle boundaries, and now records FULL runtime on both decode lanes (2224/2152 observed forwards). This closes the missing replay observation only; G3 and complete G4 remain open. See `../task_2026-10-21_step5_priority_runs/native_replay/generation03_receipt.json`.

**Overview:** D21 replaces the single-H20064 campaign with four hardware/scale milestones in existing tickets. P0 H80032 is highest priority; P1 H80064 basic runtime is eligible now but formal calibration requires own readiness and H80032 all-four clean errors<=30%. Equal-priority P2 H20032 is independent of H800; P2 H20064 basic runtime is eligible now, formal work requires own readiness and H20032 all-four clean errors<=30%. Every final track still needs all four errors<=10% in each of three clean complete-population repetitions;30% is a release condition, never final acceptance or an average.

**Deliverables:** `spec.md`, `plan.md`, `progress.md`, `summary.md`; existing tickets06/08/09/10/12/14/15/16/22/23/28/29/30; public `docs/general/step5-preview.md`; `d21_campaign_checkpoint.md` and `test_report_2026-10-21_d21_docs.md` in this task. Requirements are parent-owned and not edited. No source/test/config/frozen-package/README/other-worktree edits, GPU/network/auth/push or new tickets. Parent owns final acceptance/publication.

**Preserved contract:** frozen68f891b/958d81a workload,128 requests of6144/1024 at QPS1,16 separate warmups per repetition,three clean repeats, existing measured backend/collective_sim/runtime/metrics.32 total=P16/D16 TP8/DP2/EP16/PP1;64=P32/D32 TP8/DP4/EP32/PP1. Hardware/scale-specific native/profile/bundle evidence is mandatory; no timing scaling or cross-track PASS. Retired24–27/W1–W6 remain retired. D20 executable compute/bundle/PD/client CPU implementation continues P0-first, with native/predictive gates unchanged.

**Resource and live-state reconciliation:** H800 uses non-preemptible codesign without starving ready P0. H200 uses step_main; all personal nonterminal reservations across sessions including queued count toward non-preemptible total<=32, every additional job is preemptible even below obsolete80/130, and H20064 is wholly preemptible. Parent's complete09:02UTC personal list had0 live jobs. Prior64 qualificationexp-1021-125354-923219 Succeeded08:20:59UTC and32 communicationexp-1021-143708-523626 Succeeded08:15:05UTC; business/runtime judgment remains pending. No incompatible job needed stopping.

**Superseded intent:** old non-preemptible64-H200 `canonical02` queued-to-formal plan is not current-policy launchable. Retain qualification02/canonical02 frozen packages and assess ended64 basic evidence before repeating. Future64-H200 generations are wholly preemptible; formal waits for H20032 release30 plus own readiness. Obsolete maxlen2048 R2 stage remains paused because6144/1024 and independent-reference G3 are absent. Launch configuration code belongs to another owner.

**Validation:** commands, link/current-reference assertions, whitespace checks and limits are in `test_report_2026-10-21_d21_docs.md`. This verifies documentation consistency, not runtime readiness or calibration. Local doc commit is reported to the parent; no push occurs in this slice.

**Open dependencies:** integrated D20 measured CPU execution/fidelity/performance dispositions; case-aligned G3/content/readiness/lifetime and independent reference; admitted hardware/scale compute/communication inputs; each track's actual runtime and measurement freeze; each32 clean comparison/release30; both64 formal comparisons; all final10% rows and same-HEAD29/30. The phase records below are historical snapshots; their old caps, statuses and superseded decisions are not current instructions.

## Historical D18 common-workload adoption — 2026-10-21

The user confirmed adoption of GLM/V4's `68f891b` workload. Step cherry-pick `958d81a` preserves all seven source/package files. Active artifact: `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/`; manifest SHA-256 `5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3`. The previous independent Step package and SharedGPT remain historical.

Observed checks: 4/4 checksums; 6 existing fixed/SharedGPT tests passed; actual Frontier reader preserves128 formal requests,6144/1024 lengths and0..127 arrivals; builder reproduces every frozen byte. `test_report_2026-10-21_common_workload_adoption.md` records commands and limits. Requirements/spec/plan/tickets09/23/28, the handoff receipt and public status now agree; reciprocal artifact adoption is closed.

Publication completed:958d81a plus public-doc commit00ed846 pushed by fast-forward to the Step branch backing Draft PR50. Detailed task records follow the existing local ignore policy.

D18-COMM source b1f032d has482 CPU passes/70 Torch skips and77/77 identical fidelity; its32-H200 diagnostic remains queued at04:27UTC. Profiling5fa1926 is integrated as f0ab2eb/3cdc996 and published through6353137 after787 CPU passes/77-case fidelity. Native H200 probe confirms the BF16 dispatch fix and22/22 attention functional checks, but six SWA median comparisons fail the unchanged5% criterion; linear timing is also unqualified. A single-shape event-boundary/order/clock diagnostic is being prepared. Native consumer f3f407e is merged as db5c99f: all review findings closed,585 parent tests pass, both stage02 packages verify unchanged; Engine NNN logs supply graph-lane context. Its64-GPU qualification waits because32+64 exceeds70. R2 b97af03 remains unmerged pending retirement-safety correction. Canonical runtime/R2/timing/admission remain open. See dated integration/review records and `d18_profile_20261021/native_judgment01/`.

Three execution streams are active: Step native consumer/runtime, fixed-case compute profiling with linear-failure correction, and H200 TP8/DP4 communication preparation. Parent recovered the linear traceback and judged R2: platformSucceeded but diagnosticFAIL, with observer/validator issues and missing content/readiness evidence. See `progress.md` for task IDs and `scratch_evidence/d18_resume_20261021/` for retained results. Deliveries, R2 remediation, H200 qualification and four-metric calibration remain unfinished; no new GPU submission is claimed here.

## Ticket 12 S1/S2 judging and candidate costs — 2026-10-21

Completed offline analysis/export source is committed as `8ab12ccbf30eb3a4191df505ede1bf9bae8b5b49` on `feat/step5-comm`, with no push. Changed files are `tests/integration/step5_comm/{analyze_result,export_costs}.py`, `tests/unit/test_step5_comm_{analysis,cost_export}.py` and `docs/profiling/step5-collectives.md`. Measured workers and production predictor/registry/scheduler behavior are unchanged.

The parent S1/S2 jobs Succeeded with all four/two containers exit 0/restart 0, correct image/mount/181 source pins and H800 placement. All 1500/1380 measurement/summary and 500/460 kernel keys, groups/rails/idle/bytes, 64/32 native rank-layer matches and 256/128 completed collectives pass. Every node has active verbs ports and actual GDRDMA channels, with no selected Socket channels. Individual DP-communicator transport attribution remains unverified. Selected queue timing cells fail (9/1350, 15/1242); all auxiliary verdicts remain non-blocking and unverified rather than claiming cost admission. The foreign job was not inspected; cluster co-tenancy is parent-reported and its causal effect unknown.

Deliverables are `step5_comm_20261020/{s1_20261020_01,s2_20261020_01}/` (raw-copy/platform/execution receipts and `analysis/`/`costs/`), `step5_comm_20261020/candidate_cost_table_2026-10-21.md`, `step5_comm_20261020/candidate_cost_table_selected.json`, and `step5_comm_20261020/final_judging_verification.json`. S1 has 1500 observations/300 descriptive curves; S2 has 1380/276. Byte/ms tables preserve samples, failures, CV, ranges, pass spread and graph/backend/rail/vector context; no row is training-selected or production-consumable. DP4 per-step facts are 85 FP8 dispatches at 4288 B/row plus three BF16 at 8256 B/row, and 88 combines at 8192 B/row, across eight rails without multiplying request owners by TP/top-k.

Validation passes 535 image-source CPU cases or 465 with 70 optional skips. Final raw hash verification passes 99/55 small files plus compressed/uncompressed measurements; source-tool hashes and all three passes match the generated candidates. Per-job verdicts, uncertainties, exact commands and limits are in `test_report_2026-10-20_step5_comm_preparation.md`, section `S1 and S2 execution results`. Ticket 12 remains incomplete: stable/isolated curves, exact DP attribution, prospectively held-out qualification, capture-limit reconciliation and cost admission are open. First do CPU rejection/consumption design and attempt recovery of existing communicator/PID log ownership; then consider a focused 32-H800/20-minute S1R (10.67 GPU-hours) for independent host/held-out evidence. No new observer, stage, GPU operation, remote write or cost admission was performed; no DP2-only repeat is recommended.

## Ticket 08 R2 preparation — UTC 2026-10-20

Source/staging is complete on `feat/step5-native-pd` at `1696916` and `90559d8`, after the authorized integration merge `62bbe90`. All 560 harness/fault tests pass; all 77 simulator fidelity cases and fresh-cache filenames are identical. No simulator core or frozen environment source changed, and no GPU/platform/cloud action or push occurred.

The final diagnostic package is `/data/ycfeng/tmp/step5_native_pd_r2_h800/stage02`, tag `r2h80002`; stage01 is superseded. Offline verification passes 35 tool files, 27 original native pins, ten hook pins and four schema pins. It contains twenty requests in fifteen waves, twelve content-selected and eight native-only/no-fence, including six concurrent requests, cancellation and reuse. P/D each use TP8/DP2/EP16 on four H800 nodes. Exact parent commands and limitations are in `test_report_2026-10-20_native_pd_r2_preparation.md`; design/review/diff and sixteen selected evidence files are under `native_pd_20261020/` and `native_pd_20261020/r2_preparation02/`.

The same reviewer confirmed both implementation P1 fixes. Native content/readiness/lifetime/placement and reference continuity remain pending parent execution and acceptance. This preparation closes no ticket08 checkbox. Ticket09 still needs clean P32/D32 TP8/DP4/EP32 roles, at least three identical repetitions in one job with actual idle checks, and streaming client TTFT for every measured request; no R2 content capture belongs in clean windows.

## Corrected native generation and PR authorization — 2026-10-20

Corrected `request_physical_20261019_04` is frozen at source `94d154e`, **201 inputs / 16,439,703 bytes**, receipt SHA-256 `2c1b94dde76a2f7e69692f2df1cd7ca8fa93bfceba6f5d82aef041046729fada`; retained preparation records report **122 native pins, 24 original inert SDK checks and 543 readiness checks PASS**. The platform observed `exp-1020-000928-505270` **Running** at **2026-10-20 00:11:49 HKT / 2026-10-19T16:11:49Z**, H800 `codesign` 2×8, non-preemptible. Later records retain `gpu-h800-0530`, `gpu-h800-0157` and 16 distinct GPU UUIDs. Exact artifacts and original budgets are linked in [Ticket06](tickets/06-native-single-role-qualification.md).

These observations qualify no fresh GPU measurement or G2 criterion. **Ticket06 remains PARTIAL 1/10, criterion 7 from prior evidence only; whole Ticket15 remains PARTIAL 0/8.** Implementation remains active along complete06 ->15.native_reconciliation and06 ->{native07,native08}->09 H80064, with single owners, GLM16–22 prerequisites and final common branch unchanged. The [latest request](requirements.md) authorizes ordinary local commits and this branch's remote push/PR only; parent will record actual results and owns the ongoing native run/report and `progress.md`. Previous failures, CPU scopes and historical instructions remain intact.

## Physical H200 terminal outcome and failure archive correction — 2026-10-19

The original `request_physical_h200_20261019_02` / `exp-1019-184312-272826` ended **Failed** at **2026-10-19T11:20:08Z / October 19, 2026 19:20:08 HKT**. Both nodes exhausted the unchanged 2010-second workload deadline. Node 0 records raw SIGTERM and failed residual cleanup; node 1 records raw exit 0 with workload failure and passing residual cleanup. The original cloud archive was not created because pending native trace validation returned before its existing callback. The native timeout cause remains **UNKNOWN**; missing worker-local logs were not recovered. Executed HEAD `649b50450b52d54cf2eeec6456301e619c25308b`, receipt SHA-256 `f26f138596d15f073d7090d8130af70e307475fd5d23b8ffbfc44dad35fb03de` and all 62 frozen inputs remain immutable. Exact terminal evidence is in `native_request_physical_audit_20261019_02/terminal_disposition01.json`.

The minimum existing-harness correction is accepted as source/CPU capability only: 4 specific RED cases, 28 focused GREEN cases and **664 final affected passes**; counts overlap. All worker identities, source bindings and eight-worker coverage are checked before a pending-trace error supplies its exact reservation charge. Budget-compliant failed logs use the original archive; pending trace remains `UNVERIFIED`, native failure remains `FAIL`, and invalid or over-budget reservations still reject archival. No Frontier core, native runtime behavior, metric, reference or budget is changed. Parent Standards and Spec reviews have no remaining delta findings; no independent review is claimed. Source/identity checks pass, including all 62 original frozen inputs. Evidence: `native_physical_20261019_01/failure_archive_01/acceptance01.json` and [the dated report](test_report_2026-10-19_native_failure_archive.md).

**Ticket06 remains PARTIAL 1/10, criterion 7 from prior evidence only; whole Ticket15 remains PARTIAL 0/8.** The next step is the exact four-file commit and identity check, then one new generation to retain and localize the missing native failure evidence within the original workload/archive/resource budgets. H800/codesign is first; H200/step_main is permitted only after actual quota shortage, with at most 64 total testing GPUs and all RJobs non-preemptible. Do not repeat ended graph API, capacity, FFN or old physical generation IDs. Complete06 -> 15.native_reconciliation; 06 -> {native07, native08} -> 09 H80064. Shared core, C06 normalizer and native observer/caller retain single owners; GLM16–22 full-path prerequisites and the final common branch remain.

## Native node record correction — 2026-10-19

The sole H200 physical diagnostic exp-1019-181811-095305 failed before native spawn at2026-10-19T10:21:22Z;all41original inputs and failure records remain immutable. The existing caller/node constructor correction fixes duplicate exclusive creation and missing outer child profiler environment. Candidate01 passes4regression cases and613affectedtests(includes4),with753unchanged simulator source applicability checks. Final Standards and Spec are parent reviews with0remaining findings after both independent429failures. Source/CPU capability only:06PARTIAL1/10criterion7;whole15PARTIAL0/8. Evidence:[node-record report](test_report_2026-10-19_native_profiler_node_record.md) and native_physical_20261019_01/node_record_01/acceptance01.json. Next:exact3filecommit/identity ->distinct frozen H800-first native witness;H200 only on actual quota shortage,<=64total/non-preemptible. Then complete06 ->15.native_reconciliation and06 ->{native07,native08}->09H80064;shared/C06/native singleowners,GLM16–22gate and final common branch remain.


## Capture-node source/API milestone — 2026-10-19

Source commit: `1c3d29b23f8a504fcbb4de5d6c8da7ed74486bf5`. `native_physical_20261019_01/graph_nodes_review_01/post_commit.json` verifies the exact six changed paths, parent438424c and Git=current=frozen bytes. Tracked worktree is clean; unrelated `.pi/` is preserved, with no push or ignored-record addition. Native06 remains PARTIAL1/10 and whole15 remains PARTIAL0/8. Next is remaining transient-scale/native-write/global-occupancy and complete readiness, not another graph API repeat.

The [capture-node report](test_report_2026-10-19_graph_nodes.md) accepts the frozen six-file source/CPU slice:461 affected tests PASS/35.40s with unchanged before/after hashes, 1,045 retained/source checks PASS, and parent Standards/Spec reviews with no remaining findings. The separate one-H800 API job `exp-1019-153656-254478` is Succeeded/Completed/exit0/restart0; actual capture/replay IDs correspond, but the smoke contains no model call metadata. The post-run change is detached parser code only. No native G2 criterion is newly accepted: **06 PARTIAL1/10 (criterion7 only); whole15 PARTIAL0/8**. Complete remaining transient-scale/native-write/allocator-occupancy witnesses and full readiness before a distinct missing-witness generation. Do not repeat the ended API job or reopen this API investigation without a concrete failure.

## Resident-parameter source/CPU milestone — 2026-10-19

`native_parameters_20261019_01/acceptance_01.json` and [the report](test_report_2026-10-19_native_parameters.md) accept candidate01 source/CPU tooling only:657affectedCPUtests,111locked-imageCPUAPIchecks and1554source-applicabilitychecks. Canonical registered tensors, complete per-view finite counters, labelled raw samples and alias validation reuse the existing reservation/stream/event/drain with the original deadline. Two source defects are corrected; both final reviews are parent reviews because independent429requests produced no reports. Historical simulator evidence is reused only for unchanged source paths;raw benchmark artifacts were not all rehashed, and current observer overhead/CUDA/pinning/headroom remainsUNVERIFIED. **06 stays PARTIAL1/10,criterion7only;whole15 stays PARTIAL0/8.** Next:exact six-file commit ->remaining transient-scale/write/occupancy and complete readiness ->distinct native missing-witness generation ->06 ->15.native_reconciliation. Shared/C06/native singleowners,GLM16–22gate,final commonbranch andH800first/H200<=64non-preemptible policy remain.


## Native profiler source/CPU milestone — 2026-10-19

`native_physical_20261019_01/acceptance_02.json` and [the report](test_report_2026-10-19_native_profiler.md) accept candidate02 source and CPU tooling only. The existing native caller/observer uses the original Torch wrapper, capture/request windows, compact annotations, one tensor/trace reservation and detached `graph_launch_chains` analysis. Capture binding requires an actual positive executable identity and valid continuity; auxiliary response failure preserves original native capture and exceptions. No Frontier core or native model/router/sampling/graph-policy change occurred.

Final seven affected suites498PASS/33.05s;114profiler-only passes are included in498. Locked-image API04 passes original CPU wrapper windows and a31-byte cap with CUDA uninitialized. Current source/API/applicability810checks and retained4669checks pass within their scopes;77scenario/902hash/451pair/24CPUattempt/eight-budget evidence is reused only for unchanged simulator paths. Standards and Spec are both parent reviews with0remaining delta findings after the independent requests failed429; no independent reviewPASS is claimed. All RED/failure/timeout records and candidate01 remain immutable.

**06 stays PARTIAL1/10, criterion7 only; whole15 stays PARTIAL0/8.** Replay launch correlation and executable association are not graph-node/layer attribution. Full parameter/scale consumption, transient replay content, cache write/occupancy, current native compatibility and observer overhead remain unverified. Next: exact eight-file commit/source check, remaining full-parameter/transient-scale/write/occupancy readiness, then complete source/API/byte/deadline readiness before a distinct missing-witness generation. H800/codesign first; actual insufficient quota permits H200/step_main<=64total GPUs, all non-preemptible. Shared core/C06 and native observer/caller retain single owners;16–22 retain GLM09.full_frontier_path and final branch remains `feat/glm53-dsv4-vllm028`.

## Native state source/CPU milestone — 2026-10-19

`native_state_20261019_01/acceptance_01.json` and [the report](test_report_2026-10-19_native_state.md) accept candidate01 source/CPU tooling only. Actual per-group finalized tables, persistent-index active counts, original assignment/removal/reset/release/copy seams, known-owner storage/aliases and candidate SWA/DSA state plus resident-scale content use the existing observer/caller, single reservation, stream, events and drain. Missing assignments, replaced owners, nonfinite content and wrong device joins remain explicit; finite bytes are not native write or scale-consumption proof. All18 census source identities enter the existing future preflight ledger:11 historical matches plus7 newly bound retained bodies, with no conflicts. Old native generations and inputs remain immutable.

Verification: attempt09 six suites497PASS/31.20s, followed by changed caller attempt11 140PASS/22.04s; these counts overlap and do not describe637distinct tests or a final506-test joint run. All11 attempts and four original fixture/assertion failures are retained;01–08 exact command receipts and09 per-command source snapshot were unavailable and not reconstructed. Source checks80/80 and42parent identity/log checks pass. Standards is parent review with0findings after delegated429retry exhaustion; independent Wegener Spec has0actionable findings. Retained77scenario/902artifact/451pair/eight-budget applicability remains limited to unchanged simulator paths, with official provenance/cache/drift UNVERIFIED. The9lockedTorch checks establish primitive CPU applicability only; the complete current child/CUDA path and observer overhead remain unexecuted/unverified.

**Ticket06 PARTIAL1/10, criterion7 only; whole15 PARTIAL0/8.** Next: scoped source commit and Git=current=frozen -> focused existing native-profiler capture/replay physical correlation -> complete source/API/bytes/deadline readiness -> distinct missing-witness generation -> complete06 ->15.native_reconciliation. No ended request/capacity/FFN/Attention/source-recovery campaign is repeated. Parent owns native observer/caller/platform/audit; C06 and each shared core module retain one canonical owner.16–22 retain GLM09.full_frontier_path; final branch is `feat/glm53-dsv4-vllm028`. H800/codesign first; actual quota shortage permits H200/step_main<=64total GPUs, all non-preemptible. Only an evidenced material core refactor or materially unresolved semantics/acceptance needs explicit approval.

## Current completed native device diagnostic — 2026-10-19

The distinct `request_device_20261019_01` / `exp-1019-075927-372034` completed Succeeded at **2026-10-19T00:08:32Z / October 19, 2026 08:08:32 HKT**. Source HEAD `7962f5fe8199a6884f6c1628f7a9a1bf8faeb609` and receipt SHA `f2d2aa4f3427a8265e4d0b4acfa55b4add7091930ba2044dd40efcc33610e049` are frozen. [The new report](test_report_2026-10-19_native_request_device.md) records **26,917 retained device checks PASS**, **35 scoped source/lifecycle checks PASS**, original public→internal→worker→device association on all16 ranks, 544 real device stages and256 actual FULL replay calls, 539,099,136 valid hidden finite elements, and ready observer-owned copy events. Full precision/scale consumption, physical kernel and cache-state census remain missing. **06 is PARTIAL1/10, criterion7 only; whole15 is PARTIAL0/8.**

Both final nodes/peers/archive/residual checks PASS under the original allowances.86 cloud members/160,876,320 bytes and70 selected local members/103,333,166 bytes are identified; final analyzer source and first FAIL/counter RCA retained. Source/platform review is scoped; complete precision/scales/kernel/cache-state, transport and clean timings remain unqualified. Actual node0 EngineCore force-kill and node1 eight-worker native SIGKILL plus other warnings remain recorded. Current requirements/plan, report/disposition and before-images are updated; no core refactor, new benchmark or GPU run follows this archive.

1. Reuse this completed generation, final analysis source and immutable archive. Both request jobs and their launchers ended; no poll, retry or resubmission. The latest complete source/device archive is `native_request_device_audit_20261019_01/`; its final analysis source is `analysis_source_02.py`. General conservative receipt labels remain unchanged.
2. Parent implements the remaining all-layer/dispatch/precision binding, followed by actual tables/lifecycle/known-storage and valid-state content coverage in the existing native observer/caller. Use `device_request_witness_20261019_01/remaining_native_census_plan.md` and the existing state-range report;28 parent source/hash/AST checks verify selected claims only. Eight retained source bodies absent from the executed113-ID ledger need future binding; recover only selected missing allocator/prepare/embedding bodies if necessary. No repeated broad recovery, zero-scale theory or isolated calibration.
3. Add only missing native capture/kernel correlation through the existing profiler after observer/API/byte/deadline readiness. CPU/capture calls cannot prove every FULL-replay kernel or transient scale. Apply affected checks, inherited fidelity and unchanged budget dispositions before a separately frozen missing-witness generation. Preserve native model/router/sampling/precision/graph policy.
4. Complete06 then15.native_reconciliation;06 also releases native07 and08, followed by09 H80064. C06 may concurrently extend the existing normalizer/gate using detached records. Other shared modules keep one canonical owner;16–22 retain the external GLM full-path gate. H800/codesign first; actual insufficient quota permits H200/step_main≤64 total GPUs, all non-preemptible. Only a demonstrated material core refactor or materially unresolved semantics/acceptance requires explicit approval.

## Historical first native request observations and active continuation — 2026-10-19

The sole H800/codesign TP16 job `exp-1019-045056-405672` completed Succeeded at **2026-10-18T21:03:06Z /2026-10-19 05:03:06HKT**. Both replicas/containers completed with exit0/restart0; the original starter and launcher ended with return0. The155-file frozen receiptSHA7c6fddf2d70832576e06c601367a54574eddbd10ed9b846379642ffef888bbd0 andcd7be9f8 source remain unchanged. No new submission, poll, platform stop or repeated campaign is needed for this completed run.

[The report](test_report_2026-10-19_native_request.md), `native_request_audit_20261019_01/native_request_analysis_01.json` and `ticket06_disposition_01.json` retain24,682localartifactchecks/0failures/PASS, two unchanged native4105prompt/9outputpubliccompletions,34actualCPUrequeststages/rank,256originalFULLreplaycalls,688source-boundscopes andactualrequestterminal/archive/lifecycle.67cloudfiles/64,605,257bytes and51locallyselectedfiles/7,056,230bytesreconcile.16cloud-onlycapacityJSONLhashesremainpriorparentreads,notworkerhistoricalattestation.

OnlyTicket06criterion7 is accepted for fixed-lengthpublicnativecompletion;whole06PARTIAL**1/10**,whole15**0/8**. Public/internaladmissionassociation, devicecontent/finite/completion, executedprecision/scales andcompletephysical/planning/statecensusremainmissing. Nativeforce-kill/SIGKILLonbothnodes, shutdownTCPStore/NCCL/broken-pipewarnings andinferenceJITwarningsaredisclosed. Thisdiagnosticdoesnotqualifycleanperformance,currenttransport,fullG2,HTTPonline,PD,H80064,H200,predictionorfinalcommonHEAD. Historicalnativefailed/unadmittedandofficialcache/performanceevidenceretainstheiroriginalverdicts.

07/14/29source/designpreparationdocumentsarecomplete. Parentchecks02verifies133hashes,79HEADblobs,24nativemanifestmembers,70PDlinks,56PDfunctionranges,20independentintervals and5errorexamples;otherfunction-rangeclaimsarenotindependentlycertified. InitialcheckerfalseFAILispreserved. Currenttaskdocsareupdatedinplacewithpre-editsnapshotsin `native_request_audit_20261019_01/documentation_before/`;preparationdocumentskeepthesource/docgenerationtheyoriginallyinspected. Curieindependentauditwaspendingatparentdisposition;noneisinferred.

Nextimplementation:originalfrontendregistrationassociation ->boundeddevice/stream/state/physicalwitnesses ->distinctpurpose-boundnativequalification ->complete06 ->15.native_reconciliation. ParentownsStepnativehooks/callers,platform/auditandlocalcoreintegration;C06ownsoneexistingnormalizer/gate,othercommonmodulesretainsingleowners. H800/codesignfirst,H200quota fallback<=64total/allnon-preemptible. Do notrepeatendedFFN02/Attention01/capacity03/source recovery/gen04/gen05/CPUcampaigns. Earliersectionsbelowarephasearchives,notcurrentactive-jobinstructions.

## Controlled structural request milestone — 2026-10-19

The primary accepts15.controlled_structural_path only. Generation05 contains44changed paths/61frozen files at parent1bde1999. The three-file timing-catalog correction preserves public mapping isolation, validation, ordering and ownership. Final reviews:James Standards0violations/0smells,Plato Spec0findings.646affected tests PASS,three actual controlled CLI requests PASS,77legacy cases numerically identical with451verified artifact pairs.24CPUattempts and6380evidence checks pass;all eight metrics meet unchanged budgets. Gen04MoE overruns/analyzer exit1 remain historical.

Exact deliverables: `step5_structural_20261019_05/`, `step5_structural_reviews_20261019_05.json`, `step5_structural_acceptance_20261019.json`, `step5_structural_cli_audit_20261019_05.json`, `step5_structural_performance_20261019_05.json`, and `test_report_2026-10-19_step5_structural.md`. Current requirements/spec/plan/ticket15/review/progress are synchronized.

Each controlled request completes17stages/1564layer executions/1496FFNmerges,4105prefill+8decodeinputs/9outputs and35cacheevents. Offline6571.25ms/online6617.25ms are synthetic source_structural expectations,not native ground truth. Original official modified-source/cache gate and initialization drift remain UNVERIFIED. Whole15all8criteria,060/10,GLMfullserving,trainedpredictions,PD,H80064,formalH200 and finalcommonthree-modelHEAD remain open. NoGPU/API/Docker/native operation or checkpoint download in archival.

Next:44-path sourcecommit/61Gitblob check;separate two-file06caller freeze/commit;one purpose-bound native request generation with publicoutputs/CPUinputs/replaycallee. H800/codesign first;quota fallbackH200 at most64total/allnon-preemptible. Sharedcore retains one parent owner locally and one final integration owner. Completed tests/profiles/FFN02/Attention01/capacity03/sourceacquisition are not repeated.

## Historical canonical shared integration checkpoint — 2026-10-19

The primary accepts only the reviewed canonical merge fromb60d64d andba48220:41changed paths/54frozen files,52suites/1142PASS,77/77legacy fidelity,451artifactpairs and24sentinelattempts. Source review has0Standardsviolations/0smells and0Specfindings. Performance evidence passes6319checks, but dense initialization remainsOVER(+2.1905560017% versus1.9413370940%). Primary disposition isUNVERIFIED_UNRESOLVED_MEASUREMENT_CONDITIONS under the existingbudgetitem3. Officialmodified-source/cachequalification remainsUNVERIFIED. Exact records: `test_report_2026-10-19_shared_merge.md`, `step5_shared_merge_acceptance_20261019.json`, `step5_shared_merge_20261019_03_analysis.json` and currentpointerfiles.

Implementation is active. Complete the scoped mergecommit/blobcheck, then Step one-poolcapacity, fixedper-sequence scratch, sourceparameterstorage, canonicalFFNmerge/timing and complete92-layerrequest. Native06may advance independently on actualkwargs/replay/state/census; source recovery is alreadycomplete. Whole15/06,GLMtrainedfullserving,H80064,H200and finalthree-modelHEADareunfinished. Resourceorder:H800/codesignfirst,quota fallbackH200/step_main<=64total/allnon-preemptible. NoGPUoperationinthischeckpoint. EarlierH200-only/commit-next paragraphs below are historical phase archives.

## Ticket15 representation checkpoint — 2026-10-18

Implementation remains active after the confirmed review handoff. The primary accepts only `15.model_representation`: the frozen 23-file configuration/layer/precision/operator/component-description contracts. Step execution remains guarded; complete structural request execution and native reconciliation are unfinished.

Deliverables: `test_report_2026-10-18_step5_representation.md`, `step5_representation_acceptance_20261018.json`, final freeze `step5_representation_mtpfix_20261018/`, and updated `review.md`, `plan.md`, `progress.md`, `tickets/15-mixed-model-request-path.md`. Manifest SHA c5036292dc13e470ca6c889787261a4d0923c63927153deb1fa7a117751bca3c. 35 suites / 738 PASS / 18.83 s; 77/77 legacy numeric/intermediate cases and 451 artifact pairs verified. The initial 76/77 MTP memory failure is retained and corrected with actual PP2 regressions. Final Standards 0 breaches/0 meaningful smells and Spec 0 open findings are inherited completed reviewer reports; the parent verified bytes and corrections during archival continuation.

The 24 CPU attempts and 6,064 validation checks completed: 5/8 metrics within budget; MoE initialization/run/total remain OVER by 2.544307%, 3.656461%, 3.404594%. Diagnostic evidence shows source-compilation asymmetry and additional shared C04 bookkeeping, but does not isolate all causes. Scoped acceptance uses `UNVERIFIED_UNRESOLVED_MEASUREMENT_CONDITIONS` under the unchanged budget policy. The official comparator still failed modified-candidate provenance and does not qualify clean-cache/commit fidelity. Frozen-working-tree fidelity PASS is limited.

Next: commit 23 frozen files and verify Git blobs, complete Step native grouped-binding inspection and committed shared integration review, then implement the full 92-layer controlled request path through the approved canonical grouped KV pool. Ticket06 remains 0/10; GLM full serving, three-model common HEAD and formal H80064/H200 gates remain open. Latest GPU authority: H200/step_main, total <=64, all non-preemptible; no GPU action in archival continuation.

## Documentation review archive — 2026-10-18

The documentation audit and corrections restore model representation and complete structural execution before detailed calibration. Step predictive E2E support remains unimplemented and unvalidated. Tickets 01–04 are DONE within their accepted scope; 05/06 are PARTIAL with 0/10 accepted criteria each. All later native, trained-predictor and formal acceptance obligations remain open. This phase made documentation-only changes and did not launch implementation, project tests, CPU/GPU benchmarks, profiling, native collection or platform operations.

Deliverables, relative to `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/`:

| Deliverable | Exact path and outcome |
| --- | --- |
| Current-state and main-path assessment | [review.md](review.md): all 30 ticket rows, shared progress, evidence limits, local-investigation drift, minimum recovery path and conditional architecture alternatives |
| Corrected engineering contract | [spec.md](spec.md), [requirements.md](requirements.md), existing `tickets/01-*.md` through `tickets/30-*.md`: representation milestones, outer FP8 preservation, accurate runtime identity, evidence gates and current authority |
| Dependency-aware execution plan | [plan.md](plan.md), [dependencies.json](documentation_review_20261018/dependencies.json): canonical graph, prerequisites, owners and acceptance gates |
| Reference workflow audit | [reference_workflow_review_2026-10-18.md](reference_workflow_review_2026-10-18.md): retained reference facts and explicit PARTIAL/UNVERIFIED fresh online verification |
| Resume handoff | [progress.md](progress.md#resume-handoff-for-the-next-execution-session): exact next tickets, parallel preparation and single-owner shared files |
| Validation and provenance | [test_report_2026-10-18_documentation_review.md](test_report_2026-10-18_documentation_review.md), `documentation_review_20261018/`: before copies, source/repository observations and document-only validation results |
| Supporting dispositions | [design.md](design.md), [issues.md](issues.md), [capacity_proposal.md](capacity_proposal.md): current outcomes separated from historical proposals |
| Glossary rename | `/data/ycfeng/Frontier/.worktrees/step5-preview/GLOSSARY.md`: byte-identical rename of CONTEXT.md; no new domain or architecture decision |

Substantive changes are limited to sequencing, accurate evidence/authority and the model normalization contract. Ticket 15 starts representation after 03 and controlled structural execution next; it still needs 06 native reconciliation to close. Ticket 10 uses the documented 05 coverage disposition, while whole 05 remains required at final 30. Preparation for 07/14/29 starts independently of native/final acceptance. Integration tickets 16–22 retain the shared GLM full-path gate. Formal per-cell error thresholds, independent holdouts, required actual H200 validation, original DeepSeek variant and the final common branch `feat/glm53-dsv4-vllm028` are unchanged.

Document validation completed: both passes returned 14/14 PASS. The final pass retained all 40/278 acceptance marks, verified the 37-node/64-edge acyclic graph and all mandatory final dependencies, checked 148 local links and 16 inherited paths, and matched 40 before copies, five immutable sources and all 1,198 tracked-file fingerprints except the byte-identical glossary rename. Three independent review finding groups were corrected; no project tests were run. Retained historical results are not fresh validation: FFN02 0/6 and Attention01 0/60 rows remain unadmitted; TP16 startup is not G2; the dense-init performance overrun remains UNVERIFIED. Public retrieval supplied no readable bodies; bounded fallbacks failed with a size limit and timeout. Large tensor hashes, native runs, image identity and live job state were not rechecked. The shared worktree is active candidate work, not an accepted common-source integration result.

Next execution starts with 15 representation, parallel targeted 06 evidence recovery and 07/14 preparation, then 29 contract preparation. C02/C04 common files remain with the shared integration lead; C03/C05/C01/C06 retain one owner each. Existing jobs and accepted shared commits must be inspected before dependent work. Only a necessary material core refactor or materially unresolved semantics/acceptance requires explicit approval. This review ends without starting implementation; overall model support remains pending.

The following archives retain their original observation times and acceptance limits. Their earlier next actions do not override this review's requirements and plan.

## Historical CPU checkpoint completion and execution pause — 2026-10-18

The latest user request is complete: finish the two already-started private CPU capabilities, their reviews/corrections, prepared joint verification, verified code commits and current task records, then pause. **Active agent execution is paused.** No new caller/native entry, service generation, GPU submission or further monitoring window was started.

Deliverables are `tests/integration/step5_environment/local_attention_evidence.py`/`test_local_attention_evidence.py` (timed outputs after drain/outside annotations, bracketing resources, retained-sample cap with separate uncapped raw staging) and `role_execution_observation.py`/`test_role_execution_observation.py` (one transparent native execute, copied prepared CPU metadata before sampling, separate bounded diagnostics). New native/caller/graph/input/content/continuous identity witnesses remain absent. CPU frontiers are optimistic upper bounds, not device progress; partial failure-record interruption may remain in-memory only.

Current-byte reviews60fc59f3 and final5c8b3c5a returned **No findings**. Four AttentionP2defects have actual regression evidence:12RED/19PASS->41GREEN and4RED->258jointPASS/2.46s. Fresh corrected fidelity02 completed77/77scenarios,902/902artifact size/SHA checks,803requests, no numeric/intermediate/source endpoint differences,145.178698s and observed30GiB. Initial fidelity01 overallFAIL remains preserved because its auxiliary observer assumed cgroupv2 on an actualv1host; only scratch observation was corrected before the fresh run.70dummy/7existingCSVcases do not qualify native Step accuracy, and retained caches do not prove clean-cache training equivalence.

All24sequential sentinels completed with exact outerwaitexit0/220.873476s and source/input/runner/package checks. Seven/eight paired medians meet unchanged frozen budgets. Dense initialization4.198711431->4.299223691s is+0.100512260s/+2.393883%, over1.941337%; rawoverall`REVIEW_REQUIRED_OVER_BUDGET` is not changed. Runtime/runner/inputbytes match01fb2ce and do not import the new modules, excluding their direct execution as the cause. Actual initialization asymmetry/environment cause remains unresolved; originalbaselineinitdrift+135.547411%dense/+134.886987%MoE remainsUNVERIFIED. Under`performance_budgets.md`§3, accept the CPU capabilities with **UNVERIFIED_UNRESOLVED_MEASUREMENT_CONDITIONS** disposition, not complete performance PASS, native-helper timing or environment recovery. No budget widening/sample drop/sentinel rerun.

Verified code commits: Attention`a129a6bd2a9300159427249a197418f8a150e05a`, terminal-blank-line correction`c4f2c6b8d31699e2eb9136741fc200669b8a550e`, request metadata/finalHEAD`ac9efb1013c687f98cbde4d0678cbfad12987e50`. The initial staged whitespace check failed but shell continued to commit; no history rewrite. Final cleanup removed one terminal newline only, exact AST/location equality established, affected258tests passed again/2.77s. Final311-lineAttentionSHA`19fa025bf7844c9f99b1d68d90943256e4cb9933c6c15f5ecd3a39c3b5fffe91`; the other three bytes equal reviewed hashes. Fourcurrent=HEAD and7frozenpairs/runtime/input/runnerunchanged. Matrix/sentinel provenance remains originalreviewed312-lineAttention/precommit829dc685, not retroactively changed. Only inherited`.pi/` is untracked; no ignored task-record staging, push, merge or native/SDK/reference mutation.

Exact completion evidence in this task: `test_report_2026-10-18_parallel_evidence_joint_cpu.md`, two scopedCPUreports, `parallel_evidence_verification_20261018_01.json`(SHA25de9d94fdd5f9b24c9448fa33afe9b98ae13f8884a35008aaff4e73dec3ee7a), `parallel_evidence_post_commit_20261018_01.json`(SHA4c7c224f35c3475c25b09d4251dc4d6b341bbda30f0d3ef0082e58c3c092e24d), `parallel_evidence_final_static_20261018_01.json` (closing static checks after document synchronization). Requirements/plan/progress/review/tickets05/06/proposal are synchronized. All earlier native/CPU outcomes below are historical snapshots, not current scheduling authority.

Tickets01–04 remain done;05/06 each remain0/10in-progress;07–30 blocked, including externalGLMfull-path for production15–22. H200actualretest is deferred, not waived. Sole source-onlyexp-1018-173210-527470 lastactualreceipt10:50:26.465640Z remainedPending/H800quota0/replicasempty; watcherended10:52:10Z/originalSDKwaitersliveatlastinspection. Job/waiters untouched; they can later execute already-approved source-only work despite agentpause. Currentterminal/actual18bodies remainunknown; no newquery/window.

Resume only on user continuation:1.Readlatestrequirements/plan/progress and inspect existing actual job/artifacts without uncertain duplicate submission.2.Complete source-grounded caller/entry and aggregate generation storage/time bounds; actualinput/graph/content/precision/allrank/lifecycle proofs are still missing.3.Review/verify/freeze a distinct purposeful native generation then audit it; do not repeat unchangedFFN02/Attention01/capacity03/TP8 or release downstream/GLM/H200gates from fake evidence.

## Historical native evidence and UUID correction outcome — 2026-10-18

Tickets01–04 are complete;05/06 each remain0/10. Approved single submissions completed: Attention `exp-1018-155755-050374` Succeeded/exit0/restart0 with60rawobservations; TP16 `exp-1018-160003-695910` Failed after nativecapacityreturned and16RPCsnapshots persisted. No automatic repeats/platformstop/native/reference/SDKchanges. FFNgeneration02 used its soleapprovedrun andis not repeated. H200deferrednotwaived;GLMproductionbarrierunchanged.

Attention187files/930,313,005bytes verified; independenttraceaudit confirms300kernelowner-repeat pairs but finds300nativeDSADtoDcopies omitted fromkernel-onlydurations. Rawvaluesunchanged,admission0/60; all300CPUwholeforward/1650ownerpartwindowslackperiodicsamples. Parent60savedwarmupfinite/shape/indexchecks are nottimedoutput/weight/numericalacceptance. Reports: `test_report_2026-10-18_native_submissions.md`, `test_report_2026-10-18_local_attention_native_audit.md`, `local_attention_native_audit_20261018_01.json`, `local_attention_saved_content_20261018_01.json`.

TP16actualRCAconfirmsbareTorchvsGPU-prefixedinventoryrepresentation,notmissingGPU. All16actualvisiblepermutations/full92layer88routed352expertmetadatareconciled;44files58,477,091bytes matchretainedhashes. BothnativeexecutorsusedSIGKILLfallback; node0actualterminalexit/postexchangefinalstatus remainUNVERIFIED. Fakeplatforminterruptionschedule demonstratesreachabilityonly. OriginalFailedreceiptand7G2UNMEASUREDclassesunchanged. Reports: `test_report_2026-10-18_tp16_native_rca.md`, `role_tp16_native_rca_20261018_01.json`.

Cause-based comparison-only correction is committed **e10e030d24881fef7e1f6116bd7347a294a3d1d3**. Fourfiles: `tests/integration/step5_environment/check_role_tp16.py`, `test_role_tp16_driver.py`, `tests/fixtures/step5_environment/tp16_capacity03_uuid_census.json`, exactstaticfixture`.gitignore`exception. Prefixnormalizedonlyforcomparisons/rawunchanged;8/node16totalphysicaluniqueness/legalpermutations/fullbindingspreserved. Actualregression4RED/18PASS;affected258fakePASS;independentreview29163823-b2b2-41aNo findings. Fresh77/77fidelity902/902artifacthashes andall24sequentialattempts/eightoriginalbudgetmediansPASS. Originalinitdrift+136.532432%dense/+133.537082%MoEUNVERIFIED;acceptpairedsimulatornonregressiononly. Finalfourfilecurrent=reviewed=verified=HEADand7frozenpairsPASS,onlyuntracked.pi,noignoredrecords/push/merge. Exactreports: `test_report_2026-10-18_tp16_uuid_correction.md`, `tp16_uuid_performance_20261018_01.json`, `tp16_uuid_post_commit_20261018_01.json` (SHA-256ee23808497c217ce4c084a9df9b8fe86af3f95a96c02b69dccf1cad764c68500).

ParentprematureperformancecomparatorFileNotFoundErrorisretained; samepersistentseriescontinued/no restart orsampledrop. WorkloadprovenanceretainsactualprecommitHEAD/dirtysource,notreattributed. Nextis distinctsource-groundedG2service/replay/contentmeasurementproposal,notcapacityaloneoroldmeasurementrerun. Nativeaccuracy/cleanprofiling/fullticket/H200/GLM remainopen.

## Historical approved native integration frontier — 2026-10-18

User approves Attention/TP16 integration/GPU execution and restores original FFN generation02; routine in-scope progression no longer requires repetitive permission questions. Frozen reference/image/model/data/evidence rules, H200 deferral, GLM barrier and ticket edges remain unchanged. Requirements/plan section12 supersede historical no-execution permission statements.

FFN generation02 used its one actual submission, job `exp-1018-135843-630406`: Succeeded/Completed/exit0/restart0, three children exit0. Exactly22 files/3,220,900 bytes passed size/SHA audit. Post-timing weight comparison is complete:2,122 tensors/25 bounded read-only slices, `PASS_POST_TIMING_WEIGHT_BYTES_ONLY`; six warmup-content mathematical diagnostics completed with finite reported errors. Independent raw audit49f79fbd reproduced six rows/30calls/220charged kernels plus24prime kernels. Raw-evidence archive PASS; clean admission **0/6**, tickets05/06 each **0/10**. Timed output content, actual unquantized apply branch, continuous weight/GPU/NIC identity and complete compiled-build/FP8 activation/native routing binding remain unresolved. No FFN repeat or historical backfill.

Initial Attention integration passed133fake checks, but complete independent reviewa26363aa found fourP2 defects in total archive cap, early exception retention, stdout failure isolation and prior-child evidence retention after later startup failure. Initial TP16 integration passed174fake checks, but complete review55bc7b32 found twoP1 and twoP2 defects: no archival reserve within2400s; completion/status PASS before archive verdict; failed census raw RPC payload loss; binding faults discarding known real worker pairs. These are readiness failures, not missing user authority. Correction agents0ec402f3(Attention pair) and22290dad(TP16 caller/supervisor pairs) are running with the parent's configured model/xhigh and fake-only scope. Corrections/current hashes/new-code simulator checks are not yet accepted.

No actual Attention/TP16 job has been submitted. After corrected source reviews and new jointly applicable77-case fidelity/24 frozen-budget sentinels, the primary will separately freeze their generation specs. Proposed explicit limits are Attention4GiB generation evidence and TP16 archival reserve300s within unchanged2400s; they are not yet frozen/verified runtime inputs. Capacity-only is not G2. The owned FFN launcher only unregisters its own SDK exit-stop callback; frozenworker/native submit/wait unchanged, no platform stop/retry/healthy-job action. Exact delivered records: `local_ffn_generation02_weight_analysis_20261018_01/receipt.json`, `local_ffn_generation02_math_20261018_01/statuses.json`, `local_ffn_generation02_audit_20261018_01.json`, `test_report_2026-10-18_local_ffn_generation02_audit.md`, initial Attention/TP16 CPU reports and `progress.md`'s current correction checkpoint. H200 and GLM/downstream barriers remain.

## Historical read-only RCA outcome — 2026-10-18

The Pi access blocker is resolved. Current policy restricted_provider:null leaves interception, mandatory delegation/internal-ID routing and receipt filtering dormant for every provider/model. The user reported runtime reload; this task uses original authorized Python/local-record access without repairing or invoking the retired intermediary.

The primary independently read the final generation02 session and actual local events/facts/record. inspect_local_launcher returned INVALID_REFERENCE at2026-10-17T14:16:13.131Z, then needs_input14:16:25.322Z. A deterministic AST-extracted pure-validator reproduction at2026-10-18T04:30:45Z rejects the original plan; removing only STEPMIND_BACKEND in an in-memory plan or allowing only that key in a copied policy both accept. The cause is the old Pi env whitelist, not native FFN failure, resource insufficiency, missing IDs or earlier429/upstream errors. No frozen/native/backend change was made.

Original local records lack eight actual platform/preflight artifacts and any execute event, supporting that this archived worker stopped before launcher execution. No authentic platform job ID is retained for a scoped API query; no live platform absence claim is made. All30retained path hashes and7/7source/snapshot/f01312a pairs stayed unchanged; HEAD9ecce37/tracked-stagedclean/only.piuntracked. No launcher, SDK/auth, platform/cloud operation, submission/retry/control, accepted CPU tests/fidelity/sentinels or native workload repeated.

Deliverables: `test_report_2026-10-18_generation02_rca.md`, exact `generation02_rca_20261018_01.json` (SHA d5be1886f8a2b78ba243b946653326dcd6901ab78daa081d0a1a66441cf891ed). Existing filtered/raw receipts and all old errors remain preserved. This closes finding access and diagnosis, not generation02 native execution or ticket05/06 acceptance. Current read-only authority explicitly prohibits resumption/redispatch; later execution needs an original-generation execution instruction, not resource reapproval or Pi repair. Independent05attention coverage and06authentic-worker/shutdown source-design inspection is complete in `design.md` and `source_design_20261018_01.json`, without launcher integration. The primary independently read the sources after two Explore attempts used a locked DeepSeek model; those completed/aborted outputs are unaccepted, not running or credited. The attention ledger predicts60raw observations from30shapes/3heldouts and records remaining admission gaps. Worker identity requires live PID/start-time capture, not direct-child identity; native executor SIGKILL fallback is distinct from the added supervisor's single-SIGTERM policy. Attention/TP16 GPU execution unapproved; H200 deferred; capacity02 KV failure and GLM production barrier unchanged.

Source-design static closure: `test_report_2026-10-18_source_design_closure.md` and `source_design_closure_20261018_03.json`. Final static verifier checks5native source-manifest identities,13stable sources,30preserved RCA paths,7frozen pairs,4accepted code/Git pairs and12documents. Two failed static iterations are preserved: incorrect no-index exit expectation, then extra terminal blank lines in two documents. Only document EOF cleanup followed; no RCA/test/native/platform rerun. Final archival additions are checked separately in `source_design_final_records_20261018_01.json`. These artifacts complete this read-only/source-design slice, not either native ticket.

## Parallel CPU preparation outcome — 2026-10-18

The user's confirmed dependency-based continuation was executed as04(done)->{05,06}, without GPU/platform queries or native execution. Attention preparation now selects30frozen eager BF16 single-request cases (including3heldouts), not the earlier31documentation count. Current source/test preserve native forward ownership, restore prefix state, report the selected conventional batch-invariant GEMM and persist all observations before final diagnostic attestation. Independent complete review found missing module-global json; primary corrected it and added actual fake-orchestration persistence regressions. Final correction-only source review: No findings. Native timing/content/admission conditions remainunverified.

Supervisor review established five defects, corrected in only its source/test pair: bounded cleanup/report attempts survive evidence faults, native exit9 is retained, timeout bytes serialize, late terminal/exit cannot bypass deadlines, exact publisher roles are bound, and missing worker PID/start-time inventory remainsUNVERIFIED. The existing private marker transport records the node0supervisor's spawned child PID; PASS terminal comes from that child, supervisor fallback is explicitly FAIL-only. Independent source review: No findings. No launcher integration, real child/query/signals or resource change.

Primary final8CPU suites:228PASS/4.96s (75attention+82supervisor+71unchangedseams). All four final files ASTPASS and<=2000lines. All7generation02source/snapshot/f01312apairs and preparation SHA b7fb9c86… unchanged. Actual fresh77simulator scenarios matched retained baseline/902artifactsize-hashchecks; separate24sentinelattempts completed with exactouterwaitexit0 and alleightpairedmedians within unchanged full-precision budgets. Runtime/runner/inputs did not drift and no simulator test executed the native helpers. The contemporary baseline initialization is140.18%/135.26%above original observations; cause remainsUNVERIFIED, not erased by paired results. Original-environment/cache/native/performance-recovery claims withheld.

Durable deliverables under this task directory: `test_report_2026-10-18_parallel_cpu_tooling.md`, `test_report_2026-10-18_parallel_fidelity.md`, `parallel_fidelity_20261018_01.json`, `test_report_2026-10-18_parallel_performance.md`, `parallel_performance_20261018_01.json`, `test_report_2026-10-18_parallel_cpu_evidence.md`, `parallel_cpu_evidence_audit_20261018_01.json`. Original failures, RED controls, review interruptions and earlier evidence gaps remain preserved. Task records remain ignored/local; only the two source/test pairs are selected for separate code commits, no push.

Historical pre-maintenance status checkpoint: after this CPU checkpoint, the user selected a read-only originalgeneration02requestcheck. OnepublicStepMindstatus at2026-10-18T03:43:45.002044Z still returnedSPEC_REVIEW_REQUIRED/needs_input/jobs[], with nospecificconditionorreadableapprovedreviewlocator. Exactfilteredreceipt `local_ffn_generation_02_status_20261018_01.json`. No actualplatformAPI/privatefile/remoteaccess, no redispatch/stop/newID. The originalreviewfinding remainsUNKNOWN; need a sanctionedfilteredactionablefinding/interface, not internalauthorizationID. Emptyjobs is not independently verified no-attemptproof.

Final code commits: `46813da4dc37ba4f9d171e1433d0161b2101be4f` attention source/tests and `9ecce3782e5e4baa0f19815395643dd3805c614b` supervisor source/tests. A final staged whitespace check rejected one extra terminal newline in the attention test; only that newline was removed, final228testsPASS/4.90s and exactcurrent=Gitbytes verified. Final test914aed15…/586lines; other hashes unchanged. Only`.pi/` remainsuntracked; task records ignored/local; no push. Actualfidelity/sentinelHEAD attribution remainsf01312a, not retroactively changed to these commits.

Tickets05/06 remain0/10accepted;07–30retain full-ticket barriers, production15–22retainGLMfull-path prerequisite. Generation02 has no new live query/submission, frozen same-ID/no-redispatch rule remains. Attention/TP16GPUexecution unapproved, H200 deferred but final retest required. Inherited CONTEXT.md/.pi artifacts remain untouched.

## Historical generation02 public review-access outcome — 2026-10-18

The pending-finding and access restrictions below describe the pre-maintenance investigation only. They are superseded by the current independently verified RCA above, not new prerequisites for continuation.

The user asked whether they must supply specific information or whether the primary can review/read logs autonomously. The primary inspected installed public extension/broker/receipt-source interfaces and the frozen public workload, without reading private files or performing another platform operation. The tool exposes only dispatch/status; public broker handlers have no review/read endpoint; the status projector explicitly excludes worker prose, raw errors, logs and paths. MCP reports0servers/0tools. Seven source/snapshot/f01312a pairs and preparation/retained receipt hashes remain unchanged. Public mem_gb32 matches32000Mi under the documented conversion; it is not an established worker objection.

Deliverable: `task_memory/task_2026-10-15_step5_preview_h800/test_report_2026-10-18_generation02_review_access.md`. Public code/spec review is possible independently but does not identify the actual private finding. The missing prerequisite is a sanctioned sanitized actionable reviewer conclusion or read-only review interface, not new model parameters, internal-ID registration, resource reapproval or user-performed raw-log analysis. The provider-specific delegation rule is Claude-only; the task's newer filtered-read-only/no-substitute-route restriction still governs. No tool/configuration change, direct API/auth/SSH, status requery, dispatch, native run or ticket acceptance occurred. Exact private objection and live platform history remain UNKNOWN/UNVERIFIED.

## Historical tickets05–06 outcome

Tickets01–04 remain complete,40accepted/238unchecked/278.05/06 each remain in-progress with0accepted criteria. Four source generations38files/1,867,559bytes verified; third frontend collection permits ordinary initialization of existing locked tokenizer assets, not a native repair or multimodal feature. Fourth generation settles native headless startup sources, not a TP16 capacity execution/approval.

Capacity01 failed at tokenizer prerequisite before workers. Capacity02 `exp-1017-121747-859071` loads unchanged full wrapper/model on all8H800 ranks, then fails cache admission:−4,787,900,968bytes native KV allowance/utilization0.92.48rank records/127parent samples/9,801,368metadata bytes rehashed/recomputed; primary142checks and independent review agree. Native capture during memory determination is not service or changed-input replay. Allocator warnings remain but model loads returned; no host OOM inferred. No resource/EP/backend/utilization/model change or retry. Detailed deliverables: `test_report_2026-10-17_role_capacity.md`, `role_capacity_analysis_02.json`, `role_capacity_metadata_audit_02.json`, `role_capacity_metadata_02/`, immutable capacity01/02 snapshots/platform/raw logs. Generation01parentcloudmetadata later retrieved read-only:20samples/18,818bytes, originalSHA in `role_capacity_metadata_01/read_manifest.json`; no worker/model evidence inferred.

05 proceeds independently. New FFN consumer fixes actual root ownership by invoking inherited `_forward_ffn`, validates native and actual TP1 groups, and records both dense GEMM dispatches. Initial3findings/followupNo findings preserved. Driverstatus/RDMAunit defects corrected;69CPUtestsPASS0.35s. Nativeexp-1017-132150-162582 completed3children/sixrows/30rootcalls/Succeededexit0restart0;6traces6finitecontent/19files2,056,731bytes rehashed. Primary24rawintegrity/budgetchecksPASS and independentauditconsistent, but newweights/config/topology/currentbuild/continuousGPU-NICconditionsunverified; no admittedclean dataset. SixconditionalFP64callsusinghistoricalweights completed, not newcontentacceptance. Exactdeliverables `test_report_2026-10-17_local_ffn.md`, `local_ffn_evidence_01/`, `local_ffn_integrity_audit_01.json`, `local_ffn_conditional_math_01/`. Separatecommitsa477f50/df9d95f/7d4f479/ad4d7c8(no push), coreunchanged. Nativeheadlesssourcegeneration04successful; sourceTP16capacityproposalawaitscontractreview/newapproval, notautomaticretry. Fresh77exact and sentinel02limits/H200originaljob/GLMbarrierunchanged.

## Outcome

Current G1 outcome: all ten ticket04 criteria are accepted by the primary in `g1_package_acceptance_01.json`. The distinct resource generation exp-1017-083634-286962 produced168verified phase records/271parent-process-physical GPU samples/sixfreshchildrenexit0. All4,358 actualnewtensorfiles (41,462,170,614bytes) individually rehashed in35boundedCPUcalls and exactly matchoriginal04E. Quantizer generation02 exp-1017-083634-393146 capturedactual stable-libtorch binary and DeepGEMM Python wrapperidentities, ninefiles verified; zero-floorresult remainsPARTIAL. BothjobsSucceeded/Completed/exit0/restart0/launchersended, lockedimage/currentNFS/personalcreator verified. Independentresource/identityreviewsNo findingswithlimits; primarycompletedactualrawintegrity/packageacceptance. No historicalvalues reconstructedorrequirement waived.05/06are ready; H200deferrednotwaived/GLMbarrier unchanged. Exactdeliverables: `test_report_2026-10-17_corrective_resources.md`, `precision_resources_analysis_01.json`, `precision_resources_integrity_01.json`, `native_g1_precision_resources/observed/`, `native_quantization_binding/observed_generation2/`, `quantization_binding_generation2_analysis_01.json`.

The original G1 checkpoints and their acceptance snapshots below remain historical. DSA generation02 retained22frames through4110; SWA generation03 `exp-1017-070246-633403` completed successfully after retaining captured metadata. The three replay errors fell from0.01096–0.01332 toabout3e-5, with unchanged eager/capture results. This supports a harness lifetime diagnosis, not a native defect or general numerical guarantee. ActualFA3/full/AOT/tileflags, pointers/lengths, lifecycle resources and terminal identities are retained.

The stage reference analysis completed29read-onlycalls/390comparisons, including12exact native-conditioned DSA prefill gate checks. Separate score/stage reviewers returnedNo findings within source/localJSON/CPUlimits. Freshfidelity77/77exactmatches baseline, while dirty-tree attribution, clean-cache withholding and non-performance limits remain explicit. New rejection index records actual retained-record checkerrejects for the named negativecategories. Exactpaths: `test_report_2026-10-17_attention_state.md`, `attention_stages_analysis_01/`, `attention_state_swa_analysis_03/`, `attention_fidelity_02.json`, `g1_criterion_rejections_01.json` and frozenworker/comparator snapshots.

Binding-only FP8 observation `exp-1017-075002-951161` completed in a newprocess: full92/95/352/1406configuration/noexpertallocation/nooverrides, nativehelperUE8M0true, threefinite BF16→E4M3FN/FP32scale frames,163377rawbytes. Creator/codesign/image/NFS/H800/CPU8/32000Mi/Succeeded/Completed/exit0/restart0 verified. CPU byte/scale/layout analysis05 is complete:49152exactbytes,256exactnonzero scale equations,384finite scale formats and9retainedlayoutchecks. Zero scale differs from the Triton equation in128elements; PARTIAL/floor equivalenceUNVERIFIED is preserved. No matching image-build proof or historical04Emode/PrepareFinalize claim. The revised recorder was subsequently executed as distinct generation02, capturing its naturally loaded identities without recovering original ones. Fresh24CPU sentinels/eightwithinbudget medians and163artifactarchive are verified, with historicalnonPASS/dirty/environment/cache/outerlauncherreceipt limitations intact. Original04E per-phase/parent records remain missing. The new resource generation supplies its own fully verified observations, not historical values; G1 acceptance references both generations explicitly. H200 remains deferred, not waived; GLM production barrier remains.

Latest preparation: six-owner job `exp-1017-044926-195003` copied6/6/89778bytes; eight-owner `exp-1017-045416-346332` copied7/8/208788bytes andFailed/exit1 onmissingrequested `vllm/utils/workspace.py`. The unchanged installed modular kernel actually imports `vllm/v1/worker/workspace.py`; independenttwoownerjob `exp-1017-045732-533373` copied2/2/80818bytes andSucceeded/exit0. ReviewerNo findingsfor15fileintegrity/worker/env/ledger, originalmissingrecordpreserved. Exactcreator/image/currentNFS/resources/terminalverified. Report `test_report_2026-10-17_execution_sources.md`; threewrappersunder `tests/integration/step5_environment/inspect_{ffn_execution,execution,workspace}_sources.py`. Source-onlyevidence, notnativeforward/fit.

04E completed in `exp-1017-051804-058598`: six native modules,full352-expert FP8/BF16 MoE,18 input frames, actualTriton/finite/zero/precision/rawretention checks passed. Creator/image/currentNFS/actualcodesign/platformSucceeded/containerexit0/restart0 verified,312s<2400s. Raw38.614655485GiB independentlyhash/size/shape/dtypechecked and completeCPUmathematicalerrors retained. Report `test_report_2026-10-17_precision_ffn.md`; rawcloud `/mnt/codesign-exp/ycfeng/step5-preview/g1_precision_ffn_20261017_01`. Per-phase/parentmemoryevidencegapand general numerical/FP8modebounds remainUNVERIFIED, not waived. NoGPUrepeat/fullG1closure. Next actualstatefulSWA/prefill/DSA.

`checkpoint_04e.md` freezes sixnativeclasses/full352experts/config/exclusions/realgroups/nativeDummyModelLoader+reload+postprocessing, threefixedinputfamilies, oneH800/CPU8/32000Mi/2400s/rawcloud64GiB. `check_precision_ffn.py` implementedandpreflightpassed; P2outputconfinementfixed/3regressionsPASS, followupreviewNo findings. Nativejob `exp-1017-051804-058598` submitted05:18:04HKT,launcherPID3120148; at submission, actual FFN forward and terminal acceptance were pending; the later accepted execution/content-retention result is recorded above. Preservefullpre/postweights/routing/outputforofflinecontentanalysis. GeneralnumericalboundsremainUNVERIFIED; exactzero/finitechecksareonlysubset. StepSiLU-then-clampcorrectedfromsource. NoG1criteria/dependentrelease/H200operation.

Follow-up: two late P2 findings in 04D's input validation/path handling were corrected without changing fixtures, oracle or native execution. Three CPU regression tests and unchanged26/208/7selftestPASS; same independent reviewer returned No findings. Latest combined regressions (including RDMA) five PASS in0.81s. Original worker SHA remains pinned in accepted summary and snapshot `worker_snapshots/check_sparse_decode_a002989.py`, identical to commit `a002989`; revised harness is not relabeled as native-executed. The source-only collector and its18files/958520bytes/four prior hashes/worker/terminal were reverified on CPU. Full commands: `test_report_2026-10-17_runtime_sources_input_locks.md`.

First runtime-binding diagnostic `exp-1017-041443-460476` failed before native imports at `KeyError: 'source_size_bytes'` in an incomplete harness manifest conversion. Original worker/log/terminal preserved; actual local verifier reproduced then passed after field correction, rejecting two altered hash/size cases. No native/config/backend/model fit evidence resulted. Source-backed interface corrections retained native92backbone/95MTP-inclusive arrays and explicitly scoped omitted model cache hooks. Generation2 `exp-1017-042926-324906` succeeded, exit0/restart0: native defaultB16/cacheauto→BF16, SWA[511,0], two196608byteviews/summaryaliases, full92/95/352metadata/Fp8Config1406exclusions. Independent reviewer No findings and primary raw audit PASS; accept standalonebinding only. Collection17.96823467500508s/RSS1277160KiB/CUDA143021056bytes. Report `test_report_2026-10-17_runtime_binding.md`. No native repair/budgetchange/G1closure.

Historical04D result (2026-10-17): accepted checkpoint04D as a subset only. H800/codesign job `exp-1017-034334-792896` completed 26 native paged sparse-decode cases/104 exact BF16 frames/78 local graph replays/eight host rejections. Independent post-run reviewer `9e7a4744-2ce2-4b5`: No findings, 638,976 exact output values and 624 input hashes checked; worker/config/source/environment and Succeeded/exit0/restart0 matched. Address/storage evidence consists of inspected worker assertions, not retained raw pointers. Formal pre-submission reviewer response was unavailable; no such approval is claimed. Collection 20.819661525078118s, host peak 1346160KiB and CUDA peak 33602560bytes were within allocation. CPU fixture ValueError and verdict-write PermissionError are retained; no native/reference change or duplicate GPU run. Deliverables: `tests/integration/step5_environment/check_sparse_decode.py`, `checkpoint_04d.md`, `test_report_2026-10-17_g1_sparse_decode.md`, `native_g1_sparse_decode/observed/`, `native_g1_sparse_decode/verdicts.json`, `04d_platform_terminal.json`. Complete ticket04 remains 0/10;05/06 not released.

A separate read-only source job `exp-1017-035922-615759` copied 18 installed runtime/config/cache/precision/dummy owners (958520bytes, four prior hashes matched) from the locked image on gpu-h800-0017, Succeeded/exit0/restart0. This is source evidence, not actual runtime binding or precision/FFN qualification. Next scope is `g1_runtime_precision.md`. H200 is still deferred; no further H200 operations performed.

Historical04C execution (2026-10-17; H200 deferral remains current): user directed subsequent work to H800/codesign and temporarily deferred H200. Completed ticket04checkpoint04C on job `exp-1017-023052-337126`, nodegpu-h800-0435:72exactnative norm/residual/gate cases,216isolated operator graph replays,eighthostrejectionsPASS. Full text config parsing preserves92backbone/95declared/352experts and FP32norm/residual settings; no model construction. Independent reviewer accepted all288frames/384tensor records against Decimal70/BF16 expectations and verified source/image/device/terminal identity. PlatformSucceeded/containerexit0/restart0,7.854scollection,1512356KiBhostpeak/541696CUDAbytes. Three preflight defects were fixed before GPU run; subsequent CPU-only verdict-writePermissionError was handled bysudo writing only a new verdict, no native rerun/rawreplacement. Deliverables: `checkpoint_04c.md`, `test_report_2026-10-17_g1_norm_gate.md`, `native_g1_norm_gate/observed/`, `native_g1_norm_gate/verdicts.json`, `04c_platform_terminal.json`, and `tests/integration/step5_environment/check_norm_gate.py`. Accept04Csubset only, ticket04still0/10completecriteria. Harness committed at `43b03650b40226ea08f6a306c9946538cd6b1afc`, trackedworktreeclean/Frontierunchanged/no push. Next 04D source-backed scope is in `checkpoint_04d.md`; execution is now complete as recorded above. OriginalH200jobPending/position2/H2000at02:13HKT withlauncherPID2238524alive; watcher ended, nojobstop/duplicate. H200finalacceptance is notwaived.

Previous environment checkpoint (2026-10-17): H800 `exp-1017-013329-532329` and H200 `exp-1017-013658-791756` were submitted concurrently with both documented RDMA resource types restored through the existing ResourceConfig defaults. Same image, source and numerical collective implementation are preserved. H800 completed on two distinct hosts with17verbs devices each,240exact checks/48graph replays and actual NET/IB0–7/GDRDMA/no Socket. Both platform/replicas Succeeded and containers exit0/restart0. Missing MLX5_1.25symbols still appear without blocking the tested path; no package repair was performed. H200 is Pending due explicit insufficient quota, remaining H2000, position5 at01:52HKT; its local launcher PID2238524 remains alive. H800 communication is accepted; no H200 execution PASS or full-model/PD qualification is claimed. Reports: `environment_h800/test_report_2026-10-17_h800_rdma.md` and `environment_h200/test_report_2026-10-17_h200_two_node.md`. Shared probe is `tests/integration/step5_environment/inspect_rdma.py`, with two passing fault-injection regression tests. An independent read-only review found an auxiliary command-timeout propagation defect; narrow correction records diagnostic timeout rather than preventing NCCL execution, and parent rerun passed2tests. Full native/model and formal H200 ticket dependencies are unchanged. Verified tracked probe/regression files are committed at `66887b698b80cb57886fe5ada005d94d0395429f`; final2testsPASSin0.69s, worktree clean and simulator/core bytes unchanged from01fb2ce. Task records and diagnostic launchers remain in their existing ignored local task directory; no force-add or push.

Previous result: original G1 job `exp-1016-160330-448213` recovered from the platform mount incident and succeeded. Checkpoint04A has 82 PASS case verdicts, four host rejection controls PASS and one no-debug observation. See `test_report_2026-10-16_g1_native_results.md`, `native_g1/observed/raw_actuals.json` and `native_g1/verdicts_reviewed.json`. The initial 78 PASS / 4 FAIL verdict remains intact: the checker incorrectly expected RuntimeError where this locked build raised AcceleratorError with the correct native assertion. Six focused rejection tests validate the correction. No native code or comparison tolerance changed.

The user then authorized primary04B and parallel multi-node verification.04B successful job `exp-1016-193813-683851` has64exact native cases and272typed snapshots; its first generation's final summary Path/string error was fixed without changing native code or mathematical references. All raw tensor frames match between generations. Deliverables: `native_g1_joint/observed_generation2/`, `native_g1_joint/verdicts.json`, `checkpoint_04b.md`, `test_report_2026-10-16_g1_joint.md`.

Two-node job `exp-1016-182002-277566` succeeded on gpu-h800-0398 and gpu-h800-0044:240correctness checks and48graph replays PASS over Socket. RDMA was not available; logs show no IB device and missingMLX5_1.25symbols. Direct loaded NCCL version2.30.7/hash matches across16ranks, while PyTorch reports2.29.7. Evidence: `test_report_2026-10-16_multinode_nccl.md` and `cuda_forward_compat_multinode/platform_terminal.json`. Ticket04 remains0/10 complete criteria because model construction, full precision/runtime-bound shapes and broader native paths remain unqualified. Primary accepted these subsets; independent review was unavailable due to model-service budget exhaustion. No jobs remain active from these checkpoints; later dependencies are unchanged.

The planning phase and tickets01–03 are complete. Ticket03's source/design package remains accepted. The first actual CUDA diagnostic failed in initialization with host driver570.124.06 versus Torchcu130. At the user's request, a rerun turned on the locked image's CUDA forward-compatibility libraries (`cuda-compat-13-0 580.82.07`) and passed all five compute checks on one H800, provided the compatibility path is set at process start. Ticket04remains in-progress: initial independent CPU/source reference preparation is accepted, the CUDA initialization blocker is removed, and native04A–04E and standalone binding subsets are complete, but full G1 qualification remains incomplete. New implementation dispatch retains the chosen executor/invocation boundary; model/kernel/graph/PD qualification and later campaigns remain open.

## Workspace and deliverables

- Worktree: `/data/ycfeng/Frontier/.worktrees/step5-preview`.
- Branch: `feat/step5-preview`, based on `feat/glm53-dsv4-vllm028@01fb2ce5984f1d26135ea9bceef96f8e39be0401`.
- Exact task directory: `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/`.
- Original read-only input directory: `/data/ycfeng/Frontier/.local-draft/step5-preview/`.

The following artifact paths are relative to the exact task directory above:

| Artifact | Delivered content |
| --- | --- |
| `requirements.md` | Original requests, five-file SHA-256 manifest, and all eight recorded user decisions |
| `spec.md` | Confirmed English model/workflow/measurement/acceptance specification and source deltas |
| `plan.md` | Approved 30-file linked index, 47-edge dependency graph, inherited owners, external gates, and verification strategy |
| `tickets/01-baseline-reuse.md` through `tickets/30-evidence-support-acceptance.md` | Thirty independently verifiable, individually stored slices; exact filenames are indexed in `plan.md` |
| `source_spec_s5/01_Step5_Spec.md` | Byte-identical original specification snapshot |
| `source_spec_s5/02_Step5_Tickets.md` | Byte-identical original workstream snapshot |
| `source_spec_s5/03_Shared_Integration.md` | Byte-identical original integration-contract snapshot |
| `source_spec_s5/04_Source_Evidence.md` | Byte-identical source-ledger snapshot supplied during the interview |
| `source_spec_s5/05_Campaign.yaml` | Byte-identical planning-template snapshot, with unknown values preserved |
| `progress.md` | Execution history, decisions, review remediation, and future execution prerequisites |
| `review.md` | Phase reviews and resolved findings |
| `test_report_2026-10-15_source_documents.md` | Source hashes, inherited identity checks, and source consistency findings |
| `test_report_2026-10-15_planning_draft.md` | Draft review and whitespace verification |
| `test_report_2026-10-15_ticket_proposal.md` | Approved-proposal graph checks and exact command |
| `test_report_2026-10-15_ticket_package.md` | Final actual-file verification, resolved content finding, and final PASS |

## User-confirmed scope

- Adapt the full pure-text LLM only. Vision, video, and all multimodal support are excluded from this task and its future roadmap. Native residency remains measured honestly; no automatic module stripping is assumed.
- Complete the H80064 primary path, then actual H200 retesting. H800-only evidence is a phase result, not overall task completion; H20064 is not promised.
- Require independent per-cell mean metric errors <=10% and eligible p95 client TTFT/ITL errors <=15%, with sampling and stability criteria frozen before formal comparisons.
- Allow native qualification, inventories, and independent measurements before the GLM simulator milestone. Step simulator tickets 15–22 wait for the verified GLM full-path milestone, then may proceed alongside DS under single shared owners.
- Keep C01–C06 and the existing D-* authorities shared. Define actual correction tickets only after diagnosis, and make them explicit blockers of revalidation.
- Keep text MTP, prefix caching, 1M context, and extra topologies outside primary scope; no speculative extension tickets were created.

## Observed validation

Final document-package verdict: **PASS**.

- At planning close278criteria were unchecked. Currently tickets01–04 have40accepted;238remainunchecked.05/06 ready withzeroaccepted.
- 47 approved dependency edges, no cycles, and all 29 preceding tickets required transitively by ticket 30.
- 30 matching index rows, 37 valid local document links, and 16 existing inherited T/D paths.
- Eight explicit GLM implementation gates; native work retains its independent path.
- Five originals, snapshots, and recorded hashes match; all snapshot modes are `0444`.
- All 35 ticket/core documents checked for whitespace had no diagnostics. Targeted ticket 02 content revalidation also passed.
- Earlier source verification matched three inherited-input hashes and six reviewed Frontier Git blobs.
- The ticket reviewer checked all 30 files. Routing scope and the inline CLI fragment were corrected; no planning finding remains open.

Environment and exact commands are recorded in the test reports. These are document and provenance results, not runtime correctness or numerical-accuracy evidence.

## Planning-close snapshot

At planning close, ticket01 was ready-for-agent and02–30were blocked, with zero implementation tickets done and no Step-specific execution approval. Those are historical facts, superseded by the subsequent explicit execution/resource request and the phase results below.

Unknown image/runtime identities, native capacity/PD/graph behavior, shared decisions, campaign sample/stability settings, H200 minimum matrix, and future correction IDs are assigned to the approved tickets. They are not filled with guesses or treated as measured results.

The original drafts and inherited GLM task files were preserved. The existing glossary/ADR were reused; no new agreed term or irreversible architecture decision warranted another file. Task documents remain local, ignored by the existing repository rule, and uncommitted. No implementation, GPU run, external publication or Git commit was performed during planning.

## Tickets01–02 phase outcome, 2026-10-16

Both tickets are **done**. Ticket01 delivered the baseline/reuse audit, preserved failure dispositions, accepted77case capture and frozen CPU review budgets. Ticket02 delivered a locatable Step reference, not native model support. The explicit proxy preserves the locked image/config/layers and completed its one-H800 identity job `exp-1016-013143-191657`; original identity job `exp-1016-003706-065435` is also terminal Succeeded. Actual metadata, packages and593source/library identities were checked, including30Step5-related paths. Build SHA remains unknown and public source remains SOURCE_CANDIDATE.

Deliverables relative to this task directory:

| Artifact | Observed result / boundary |
| --- | --- |
| `tickets/01-baseline-reuse.md`, `performance_budgets.md`, `cpu_performance_observations.json` | All10criteria accepted, scoped frozen budgets, historical failures retained |
| `reference_lock.json`, `reference_lock_2026-10-16_observed.json`, `reference_inputs/` | Preserved initial and pre-acceptance observed generations; exact three-file revision-pinned metadata, no weights/snapshot |
| `native_reference_inspection/` selected observations and identity receipts | Successful proxy/original jobs, immutable platform identity, exact config bytes, actual packages and source/library ledger |
| `test_report_2026-10-16_explicit_proxy.md` | Primary verified8evidence hashes,3metadata files,593cross-job identities and5existing launcher rejection controls |
| `bytecode_control_observations.json`, `test_report_2026-10-16_bytecode_control.md` |24retained results; controlled6PASS/2OVER_BUDGET; primary accepts UNVERIFIED disposition, not performance PASS |
| `review.md`, `tickets/02-native-reference-lock.md` | Reviewer ses_ebf40acefffesyjkUcmfaqIO5R confirms scoped acceptance; current status authority supersedes immutable snapshot's pending status |
| `grok_batch_03_inventory.md`, `tickets/03-text-model-inventory.md` | Next checkpoint prepared; dependency released and standing resource approval retained; no03execution |

The original bytecode-cache mismatch explains the large initial compilation discrepancy. One later diagnostic MoE profile pair had no extra Frontier calls and did not reproduce the positive initialization penalty; the two controlled residual causes remain unknown. No threshold widened, failed observation overwritten, simulator fix invented, fidelity cost accepted or later code-step check waived. Missing shell receipt for sequence23 and delayed one-time sequence24 recovery remain explicit limits.

Local publication was canceled by the user. The earlier interrupted launcher had already completed a retained local pull but no tag/push or new job. No replay/prune followed. Source HEAD remains01fb2ce, tracked core unchanged, no commit/push. Native G1–G4, full-model seed/state identity, graph/memory/PD support, production GLM full-path barrier, campaigns and H200 remain pending.

The next explicit invocation subsequently resumed the same Step executor for03. That historical boundary does not grant unlimited future coding dispatch; ordinary continuation supports primary verification/corrections/acceptance and cannot bypass a new checkpoint's invocation requirement. Resources do not need approval again.

## Ticket03 source-evidence checkpoint, 2026-10-16

Inventory JSON/Markdown, observation design and test report returned from Tasks25–27; source collector/launcher preparation returned from Tasks28–29. The primary corrected unsupported KV inference and a missing page-token dimension, strengthened static checks, reviewed source containment, and used StepMind for one read-only H800 source-copy job. All26files/363805bytes match manifest/allowlist/original ledger; source manifest SHA `4bcd69c5838bc9a52576e0fd518b47852d97181627127b4ea5fefa41903fdc46`. Job/source/GPU/terminal facts are in `native_source_export/execution.json`.

The actual source establishes4logicalKVheads, localKV1/provider replication2 atTP8, localindex-Q4 and indexK/Zreplication8; weighted-ReLU scoring with retained/unconsumed ssmax_s; separate rawKV and FP8summary/FP32tail storage; and explicit fused/alternative physical scopes. Actual dispatch, buffer identities, page allocation, wrapper residency, graph replay and PD integrity remain future observations. No model/kernels ran. The inventory records92backbone/69SWA/23sparse/88MoE/4dense and all1406exclusions; all10native scenarios remain unexecuted.

Exact checks, source citations,10-criterion evidence mapping and current artifact hashes are in `test_report_2026-10-16_text_inventory.md`. Reviewer `ses_ebef214a6ffe6MEe0fZwObVdHW` accepted the package after correction of historical-topK-plus-current cardinality, partial/open-region means in the transfer view, and stale live status. Primary accepts all ten03criteria and releases04ready-for-agent; no native G1 execution has started. No production source, locked inputs, old timing observations or Git state changed. Next action is an explicit `/grok-implement` continuation for the original Step session; resource approval remains standing.

Final document verification passed:12maintained records plus3tooling files,01–03done/04ready/05–30blocked,11source citation ranges,3current artifact hashes,5read-only original snapshots,unchanged core againstHEADandbaseline and clean tracked status. The source/design phase is complete; runtimequalification and the remaining27tickets are open.

## CUDA diagnostic and user-directed deferral

`test_report_2026-10-16_cuda_compatibility.md` records the unchanged-image oneH800 job `exp-1016-103931-795883`: terminalFailed at `torch.cuda.init()` with "driver too old (found version12080)". The loaded driver library was570.124.06 and CUDA runtime was13; four dependent computation checks were SKIPPED. The worker output and operator receipt are retained under `cuda_compatibility/`. Container startup/source export is not compute qualification. OS cuda-compat package presence remains unknown; the actual570library selection is proven.

The user explicitly requested waiting for the named session's solution. Continue only independent CPU/source G1 cases and reference preparation. Ticket04stays open; no native qualification or dependent-ticket release follows. The failed generation and all immutable references are preserved.

## CUDA forward-compatibility rerun

The user later selected the image's forward-compatibility switch, `VLLM_ENABLE_CUDA_COMPATIBILITY=1`. Job `exp-1016-132434-466356` (one H800, same locked image, Succeeded) wrote its outputs to `cuda_forward_compatibility/observed/`. Details are in `test_report_2026-10-16_cuda_forward_compatibility.md`.

- The locked image contains `cuda-compat-13-0 580.82.07-0ubuntu1` in `/usr/local/cuda-13.0/compat`, so the first report's open package-presence question is resolved.
- With that directory first in `LD_LIBRARY_PATH` at process start, the process loaded `libcuda.so.580.82.07`, and the driver and runtime APIs both reported 13000. Allocation, BF16 cuBLAS, CUDA graph replay and the unchanged Step5 Triton gate passed with zero error. Device attributes for `cuMemMap` support are both 1.
- Turning the switch on inside an already-running process (vLLM edits the variable on `import vllm`) still loaded the host 570 library and failed. Native workers must therefore set the path before Python starts; `plan.md` records this launch requirement.
- Limitations: one GPU only, with no NCCL or peer-to-peer check. Timings come from a mixed stack (kernel module 570.124.06 with user-mode driver 580.82.07). Five small checks are not G1 qualification.

## Independent G1 reference preparation

Deliverables: `g1_preparation.json` (82planned cases), `g1_preparation.md` (source contracts, mathematics and runtime unknowns), and `test_report_2026-10-16_g1_preparation.md` (commands/results). They cover33uniform summary,16max-shift,16reset/reuse,4copy-continuation,4padding,6selector and3conditional-window cases. All native results are absent. The missing-tail negative fixture explicitly requires TRITON_DEBUG=1 in a separate future native process; no default-build assertion behavior is inferred.

Local CPU checks passed: independent mathematical references, all26source hashes, six extracted-source compact-length calls, seven corrupted-reference rejections and three invalid-argument rejections. The reviewer identified two missing preconditions; explicit physical slots/fresh-cache state and finite FP32 scored-selector inputs now address them. Follow-up reviewer ses_ebd432d45ffe4zL0x5ecqabSnj accepted this corrected preparation subset, with no remaining material finding and no independent command/hash rerun claimed. Decimal60 comparisons disclose FP64 rounding observations without creating a GPU tolerance. Source/model inputs, diagnostic failure, original locks, old performance outcomes and production code are unchanged. Final current-state/integrity checks passed for30tickets,10record files/two scripts, report/source/snapshot hashes and unchanged core/Git. Initial preparation is complete; native04's ten criteria and05–30remain open/blocked as before. The next dependent native work is the explicitly invoked G1 implementation checkpoint, which follows the forward-compatibility launch requirement.

Representation commit: `b60d64d6073b126fd6439bdba5a13e3951ab758d`. Post-commit verification in `step5_representation_post_commit_20261018.json` confirms all 23 Git blobs equal current and frozen bytes. Only unrelated `.pi/` remains untracked. This identity check does not relabel the earlier official comparator or performance results.

Canonical merge commit: `1bde1999af07a41651bb025d92e95bf14ad9e1ca`. `step5_shared_merge_post_commit_20261019.json` verifies54/54Gitblobs=current=frozen and both expected parents. Tracked worktree is clean; unrelated `.pi/` remains. This identity result does not change earlier official or performance verdicts.

Structural commit: `03c7f823d71193277a0edae5fcab5374c2ed8831`. `step5_structural_post_commit_20261019.json` verifies61/61Gitblobs=current=frozen, exact44changedpaths and parent1bde1999. Only unrelated `.pi/` and the two separately reviewed nativecaller files remain untracked. This is source identity, not a relabeling of historical official/cache tests.


## Ticket06 request caller CPU archive — 2026-10-19

Accepted two-file source/CPU capability only: `request_caller_20261019_01/`, `request_caller_acceptance_20261019_01.json`, `test_report_2026-10-19_request_caller_cpu.md`. Owner63PASS/16.73s and final James Standards/Plato Spec No findings read and hash verified;neither tests nor native operations repeated. Adjacent capacity/supervisor/observer bytes remain unchanged. Caller excluded from44-pathstructuralacceptance; nativecompatibility,GPUcompletion/content/precision/physicalscopes/fullG2 unmeasured,whole060/10. Next:separatecallercommit/sourcecheck -> canonicalrequestpreflight -> finalsource/config/launcherfreeze -> oneH800nativepublicrequest/input/replayobservation.

Caller commit: `cd7be9f8b13923fb6f336b0fd85f277598348506`. `request_caller_post_commit_20261019_01.json` verifies bothGitblobs=current=frozen and61unchangedstructuralblobs;trackedworktreeclean,onlyunrelated.piuntracked.No nativeorclean-cachequalificationisclaimed.


## Ticket06 original frontend admission source/CPU acceptance — 2026-10-19

The existing source-pinned observer now detaches original n=1 pure-text admission before EngineCore dispatch. It retains full public/internal IDs, accepted prompt, resolved/submitted sampling and actual process/object identities through native mapping removal; the caller joins ordinary completed public output. No native method body, sampling/ID policy, transport or Frontier core contract is replaced.

Three changed source files in tests/integration/step5_environment: role_model_input_observation.py,check_role_execution.py,test_role_execution_driver.py. The unchanged model-input comparison test remains context. Final Standards/Spec source reviews report no findings after the P3 identity-name and P2 byte-cap corrections. Parent fault checks4FAIL/12PASS become16PASS; final four affected suites275PASS/33.88s,zeroJUnitfailures/errors/skips,source hashes unchanged.199sourcechecksPASS. Exact frozen files/hashes/reviews/commands and limits are in frontend_admission_20261019_01/acceptance_02.json and test_report_2026-10-19_frontend_admission.md.

Acceptance is source/CPU tooling only. Native06 remains1/10criterion7 andwhole15zeroofeight. Actual native frontend delivery, worker/device join, finite/completion, precision/fullphysical-statecensus, HTTPonline/PD,H80064,requiredH200,trainedprediction/commonHEAD remain unverified. Old endednative receipt/source snapshots,force-kill/JIT/transport warnings,officialdirty/cache/initdrift and historical77-case/eight-budgetrecords remain unchanged. No benchmark/native/platform operation in this step. Next:scoped sourcecommit/identitycheck,then bounded actual-worker/device/completion witnesses and a distinct frozen workload as needed.


Frontend admission source commit: `60ab52d7ac04493d1cd6ec781c80f0006896ed44`. `frontend_admission_20261019_01/post_commit_02.json` confirms all four frozen Git/current sources match, exact three changed paths and parentcd7be9f8; tracked worktree clean, unrelated `.pi/` untouched. No push or ignored task-record addition. This establishes source identity only; native06 remains1/10 andwhole15 remains0/8. Continue actual worker/device/completion witnesses.


## Device/request/completion source and CPU milestone — 2026-10-19

The device/request/completion source and CPU tooling slice is accepted in `device_request_witness_20261019_01/acceptance_03.json` and [its report](test_report_2026-10-19_device_request_witness.md). Candidate03 preserves original worker registration, finalized native inputs, owned asynchronous copies, complete valid hidden finite reduction and the existing post-generate event-query RPC. The parent verified 357 affected CPU passes, 234 retained source checks, 90 fresh archive/identity checks and six locked-image Torch CPU API checks. Standards candidate02 has zero documented violations or actionable smells; Spec candidate03 has no remaining delta finding after the 1–4-byte JSON response-cap correction. Acceptance is limited to source and CPU tooling. Actual CUDA/native frontend→worker→device delivery, precision/scale consumption, complete physical/cache-state census, full G2, PD, H80064, H200 and predictive/common-HEAD acceptance remain unverified. Ticket06 stays PARTIAL 1/10 (criterion7 only), and whole15 stays PARTIAL 0/8. The old executed request source/receipt is immutable historical evidence. Next: commit the six reviewed files, verify Git=current=frozen, then finalize and run the distinct `request_device_20261019_01` generation under the existing H800-first policy and budgets.

Deliverables: `device_request_witness_20261019_01/acceptance_03.json`, `review_candidate_03.json`, `spec_review_03.md`, `standards_review_02.md`, `source_checks_03.json`, `verification_applicability_03.json` and `test_report_2026-10-19_device_request_witness.md`. The six source/test files remain below2,000lines each. All original attempts, failed verdicts, source snapshots and native receipts remain unchanged.


Device witness source commit: `7962f5fe8199a6884f6c1628f7a9a1bf8faeb609`. `device_request_witness_20261019_01/post_commit_03.json` verifies all six Git/current/frozen source files, the exact six changed paths and parent `60ab52d7ac04493d1cd6ec781c80f0006896ed44`. The tracked worktree is clean; unrelated `.pi/` remains untouched. No push or ignored task-record addition occurred. Source identity establishes no CUDA/native acceptance. Next: final canonical preflight, inert SDK/launcher review and a distinct purpose-bound H800/codesign generation, retaining06 1/10 and whole15 0/8.


## Native census source/CPU checkpoint — 2026-10-19

Accepted fixed two-file candidate02 in `native_census_20261019_01/acceptance_02.json`; full English evidence is `test_report_2026-10-19_native_census.md`. The observer binds resident original call/precision paths, actual nested GEMM callers and FULL capture boundaries. Complete JSON envelope/rejection/summary caps and lazy return classification are verified. Final419CPU passes and separate Standards/Spec follow-ups close2/1priorP2findings. Retained77scenarios/902hashes/451exactpairs/eightCPUbudgets apply only to unchanged simulator paths; native overhead, original-env drift and official cache/commit qualification remain UNVERIFIED. No native/GPU/profiling campaign.06PARTIAL1/10andwhole15PARTIAL0/8 unchanged. Parent next owns actual tables/lifecycle/storage/state content and native physical correlation before a distinct source-frozen generation; shared core/C06 stay single-owner. Source commit identity follows in `native_census_20261019_01/post_commit_02.json`.


Native state source commit: `291ba7b713d09bc448924499df70adaab16e69dd`. `native_state_20261019_01/post_commit_01.json` verifies all nine Git/current/frozen files, exact changed paths and parent `3c9bff8332376cb9165af66649403beefd87c713`. Tracked worktree is clean; unrelated `.pi/` remains.83documentation/criterion/artifact checks pass. No push or force-add of ignored records occurred. Source/CPU-only acceptance leaves06PARTIAL1/10andwhole15PARTIAL0/8. Parent continues focused existing native-profiler capture/replay correlation and remaining qualification readiness.

Native profiler source commit: `8bf0315a6b483d0a51daed214012a58398fd1d60`. `native_physical_20261019_01/post_commit_02.json` verifies all eight Git/current/frozen sources, exactly eight changed paths and parent `291ba7b713d09bc448924499df70adaab16e69dd`. The tracked worktree is clean; unrelated `.pi/` is untouched. No push or ignored-record addition occurred. Source identity leaves Ticket06 PARTIAL 1/10 (criterion7 only) and whole15 PARTIAL 0/8. Remaining work is full parameter/transient scale/write/occupancy and complete readiness before a distinct native missing-witness generation.

Resident-parameter source commit: `438424c36ed7d0d933b5e90dd64e2a82a86de303`. `native_parameters_20261019_01/post_commit_01.json` verifies all six Git/current/frozen files, exactly six changed paths and parent8bf0315. Tracked worktree clean; unrelated .pi/ untouched, no push or ignored-record addition.66 documentation/source checks preserve acceptance text and1/10,0/8counts. Parent continues remaining transient-scale/write/occupancy and complete readiness; CPU/API source acceptance does not qualify native execution.


## Allocator observation source/CPU milestone — 2026-10-19

Accepted only the three-file source/CPU tooling in `native_physical_20261019_01/occupancy_01/acceptance01.json`; report: `test_report_2026-10-19_allocator_occupancy.md`. Final435affectedtests/31.23s,12lockedCPUAPIchecks and40source/artifactchecks pass; independent final Standards and Spec reviews have no findings. Two storage/statistics duplication findings were corrected. Original failed fixture attempts remain recorded. Native CUDA occupancy, attribution/history, overhead and headroom remain unverified. The seven-source writer/scale note narrows the next work to existing-table bindings plus combined native readiness.06PARTIAL1/10 andwhole15PARTIAL0/8 are unchanged; no GPU/benchmark/new reference campaign occurred.


## Minimum native writer and FP8 scale source milestone — 2026-10-19

`native_physical_20261019_01/writer_scale_01/acceptance01.json` accepts the three-file source/CPU slice from49482c6. Existing source and call tables now bind ordinary fused KV insertion, fused DSA preparation, separate KV update and both FP8 quantization producers; `quant_dtype` uses the existing dtype metadata conversion. Five retained owner signatures match exact source pins. Final seven suites608PASS/38.38s, exit0, with no source changes. Independent Standards has No findings; parent Spec has no remaining delta findings after the independent reviewer failed429 without a report. The first155-pass affected run is included in the final suite, not added. Report: `test_report_2026-10-19_writer_scale_bindings.md`.

Python parent/producer/consumer metadata, capture association and native exception preservation are source/CPU capabilities only. Transient scale contents, completed GPU writes, current native compatibility and observer overhead remain unverified. Ticket06 stays PARTIAL1/10 (criterion7 only); whole15 stays PARTIAL0/8. Finish this exact scoped commit, then combined existing layout/source/API/record/trace/deadline readiness and one distinct native missing-witness generation. No new collector or core abstraction is required. Shared/C06/native single owners, GLM16–22 gate, final common branch and H800-first/H200 quota fallback<=64 non-preemptible policy remain.

Source commit: `0bae7145ef8a1ac6ca3e59eeca55e341800bc465`. `native_physical_20261019_01/writer_scale_01/post_commit.json` verifies the exact three paths, parent49482c6 and Git=current=frozen. No push or ignored-record addition; unrelated.pi preserved. Continue combined readiness and a distinct native witness;06/15 formal status unchanged.


## H800 quota disposition and H200 source/CPU milestone — 2026-10-19

The sole H800 physical generation exp-1019-175110-826226 remainedPending/zero replicas with actual16requested/13remaining quota shortage, then was stopped through the approved original Python wrapper under the user's H200fallback instruction at2026-10-19T10:07:56Z. Its35original sources, frozenreceipt andthreeplatformsnapshots remain preserved.42quota/source/archive checks PASS; no H800nativeworkload executed. The three-file existingTP16driver extension accepts H800/codesign andH200/step_main, checks actual configured hardware and retains resource identity without changing native workload or budgets.451affectedCPUtests/41.92s,756sourceapplicabilitychecks,122nativepins and24inertSDKchecks PASS. IndependentStandardsNo findings;parentSpecNo remainingdeltafindings after independent429/no report. Evidence: native_physical_20261019_01/h200_resource_01/acceptance01.json and test_report_2026-10-19_h200_resource_extension.md.

Next:exactthree-filecommit/sourcecheck ->distinctrequest_physical_h200_20261019_01freeze ->oneH200/step_main16GPU/non-preemptiblenativediagnostic ->actualcriteriondisposition.<=64totaltestGPUs. H200evidence does not qualifyH800capacity/timing.06PARTIAL1/10criterion7only andwhole15PARTIAL0/8 unchanged; noFrontiercorechange,newarchitecturedecision,checkpointdownload orbenchmarkreinterpretation.

H200 resource source commit: `63921567cd7c57955f80288aa115393567c1d3cf`. `native_physical_20261019_01/h200_resource_01/post_commit.json` verifies exactthreepaths, parent0bae714 andGit=current=frozen; trackedworktreeclean, unrelated.pi preserved, no push or ignored-record addition. Continue distinctH200freeze andoneauthorizednative diagnostic; acceptance unchanged.

Node record source commit: `649b50450b52d54cf2eeec6456301e619c25308b`. `native_physical_20261019_01/node_record_01/post_commit01.json` verifies exactthreepaths,parent6392156 andGit=current=frozen;trackedworktreeclean,unrelated.pi preserved,no push or ignored-record addition. Native06 remains1/10criterion7only andwhole15 remains0/8. Next is one distinct H800-first frozen witness with unchanged budgets.

Failure archive source commit: `b5b2005ccb809252b6dc1e37ff087e5c231dc7a5`. `native_physical_20261019_01/failure_archive_01/post_commit01.json` verifies the exact four changed paths, parent649b504 and Git=current=frozen. Tracked worktree is clean; unrelated `.pi/` remains, no push or ignored-record addition. Continue one new purpose-bound diagnostic; native06/whole15 acceptance is unchanged.


## FIFO source/CPU completion and terminal disposition — 2026-10-19

Original H800 `request_physical_20261019_03` / `exp-1019-205955-721468` is terminal Failed at **2026-10-19T13:37:21Z / October 19, 2026 21:37:21 HKT**. The detached 3,660-check audit retains 90 files / 79,981,279 bytes and all 181 frozen inputs. Warmup return and public completion are absent; 16 capture traces are zero-byte pending records. Native remains FAIL / BLOCKED_ON_RUNTIME; pending traces and original live-stack cause remain UNVERIFIED.

The five-file FIFO/optional-preemption correction is accepted for source and CPU capability in `native_physical_20261019_01/fifo_export_01/acceptance03.json` and [the report](test_report_2026-10-19_native_fifo_export.md). The original exporter stays on its calling thread; the fresh CPU reader preserves error, deadline, inode and reservation semantics. Legal preemption None is retained. Independent Standards and Spec candidate03 reviews have no remaining findings. Distinct affected coverage is 626; the final 168-case test-only fixture rerun replaces existing cases. Locked CPU traces exceed the actual FIFO; CUDA stays uninitialized.

Deliverables: final source snapshot/reviews/test receipts/acceptance in `native_physical_20261019_01/fifo_export_01/`; dated `test_report_2026-10-19_native_fifo_export.md`; immutable retained audit in `native_request_physical_audit_20261019_03/`; source-only PD preparation `native_pd_source_preparation_20261019.md`. All REDs, review findings and old frozen inputs remain. No benchmark campaign or new native acceptance occurred in this correction.

Next: exact five-file commit and Git=current=frozen check -> distinct corrected missing-witness generation with the existing preflight/config/SDK/launcher and unchanged budgets. H800/codesign first; H200/step_main only on actual quota shortage, at most 64 total testing GPUs and all non-preemptible. Complete06 ->15.native_reconciliation;06 ->{native07,native08}->09 H80064. Shared core, C06 normalizer and native observer/caller each retain one owner. GLM16–22 full-path prerequisites and final common branch remain. Ticket06 stays PARTIAL1/10, criterion7 prior-only; whole15 stays PARTIAL0/8.

FIFO source commit: `94d154e1f787685032db0242fbc4b99d937bbf14`. `native_physical_20261019_01/fifo_export_01/post_commit03.json` verifies exact five changed paths, parentb5b2005 and Git=current=frozen. Tracked worktree is clean; unrelated `.pi/` remains; no push or ignored-record staging. Source identity grants no new native criterion. Continue the distinct corrected H800-first witness with unchanged budgets.


## GitHub publication — October 20, 2026

The user-requested branch publication is complete: `feat/step5-preview` pushed with upstream tracking to `NetX-lab/Frontier`, Draft [PR50](https://github.com/NetX-lab/Frontier/pull/50) opened against`main`. New documentation commit`9ebf4cf224410715cbb04d84dc94f8654627d4d8` adds`docs/general/step5-preview.md`; actual local,remote andPRheads match. Earlier verifiedcodecommits are included. Detailedrequirements/spec/tickets/progress remainupdatedlocally under the repository's explicit local-only task_memory policy; no forced raw artifacts or unrelated.pi were added. No merge,force-push or branch rewrite.

Current native generation remains bound to executedsource`94d154e` andfrozenreceipt`2c1b94dde76a2f7e69692f2df1cd7ca8fa93bfceba6f5d82aef041046729fada`;all201inputhashesremainunchangedafterdocumentationcommit. PRpublication grantsnoG2/nativeacceptance.06PARTIAL1/10priorcriterion7;whole15PARTIAL0/8. Original locallauncher/session8498 continues;nextactualnativeevidence/archive/terminalaudit ->complete06 ->15.native_reconciliation;06 ->{native07,native08}->09H80064.


## Mainline execution and consolidation — October 20, 2026

Overview: after the user adopted decisions D1–D4, the work moved to the main path in parallel streams and ended with one consolidated branch.

Deliverables:

- Ticket 06 simulation scope closed from retained TP16 runs: [disposition 01](native_simulation_scope_20261020/disposition01.md) and [disposition 02](native_simulation_scope_20261020/disposition02.md).
- Local merge of the shared tip `71cde5e` into `feat/step5-preview` (`c14e642`): [merge report](test_report_2026-10-20_shared_branch_merge.md) and [fidelity report](test_report_2026-10-20_merge_fidelity.md), 77/77 legacy cases identical against both parents.
- Ticket 17 structural part: Step sequential PD at P32/D32 with attention TP8×DP4 and MoE EP32, dummy timing, offline and online; 103,687,956 B per rank for a 4105-token prompt. [Report](test_report_2026-10-20_step5_pd_structural.md) and [review](review_2026-10-20_step5_pd_structural.md); the two Medium findings are fixed in `b875093`.
- Ticket 08 run R1 on H200 (`exp-1020-164750-497812`): one 4105-token request over NIXL, D loads 4080 tokens and recomputes 25, 817,631,232 B in 2,944 descriptors, all nine conditions pass. [Report](test_report_2026-10-20_native_pd_r1.md). H200 evidence only.
- Ticket 10 MoE slice and H800 pilot (`exp-1020-152802-841686`): DeepGEMM for block-FP8 layers 3–87, Triton for BF16 layers 88–90; details in `progress.md`.
- Ticket 12 phase 1 design: [design](step5_comm_20261020/design01.md).
- Consolidation: merges `4b3f662`, `bd58493`, `0d2ffda` and public status commit `7266981`, pushed to PR #50. [Consolidation report](test_report_2026-10-20_stream_consolidation.md): failing unit IDs unchanged (132, all pre-existing), 156 harness tests and 381 profiler tests pass, smoke matrix 12/12.

Open work: H800 R1 and R2 (ticket 08), the Step attention profiler and full grids (ticket 10), collective collection (ticket 12), `20.structural`, then ticket 09 on H800 and the GLM-gated tickets. The user decisions listed at that point were answered later the same day (next section). The handoff is at the end of [progress.md](progress.md).

## Decisions D5–D8 and cleanup — October 20, 2026

Overview: the user answered the four open decisions. The answers are applied in the task records and the public status, and the approved cleanup is done.

Deliverables:

- Decisions in [requirements.md](requirements.md):
  - D5: measure the D-PD first-token and tail terms before fixing. A term is small at most 1 percentage point of the native mean client TTFT and E2E, with no changed verdict. A small term is recorded; a large one is aligned in the existing owners.
  - D6: the ticket 09 repetitions may share one job and one startup.
  - D7: Thinking Mode stays off in every run.
  - D8: cleanup allowed.
- Public status commit `8585748` (`docs/general/step5-preview.md`), fast-forward pushed. PR #50 head is `858574873b4a61bcad66b0ab90dbe1c04788652d`, still Draft.
- Cleanup:
  - Removed: eight temporary worktrees (about 3.7 GB), and about 128 GB of this task's scratch data under `/data/ycfeng/tmp`. `/data` usage went from 567 GB to 440 GB.
  - Evidence that the reports cite: kept in `scratch_evidence/` in this task directory (75 MB), with copy manifests. A checksum dry run showed the copies match the originals.
  - Commits on no branch: the earlier candidate merge `0c9b0ca` and its follow-ups `1aebe86` and `d54e2e8` are saved as a verified Git bundle in `scratch_evidence/unreachable_commits/`.
  - Reports that cited the deleted paths now give the new locations.
  - Kept: `/data/ycfeng/tmp/step5_native_pd_r1/`, `/data/ycfeng/tmp/step5_profiling_20261020/`, older evidence, and other sessions' data.

Open work: unchanged from the previous section, plus the D5 assessment once warm native requests and the ticket 10 profiles exist. No user decision is open.

## Calibration scope reduction — 2026-10-21

The task's acceptance target is now one canonical case rather than a campaign: Step-5-Preview, text backbone, SharedGPT with **128 requests at QPS 4.0**, online P/D disaggregation on **64 H200** GPUs, P32/D32 with each role TP8/DP4/EP32/PP1 over four nodes, MTP off in both native and Frontier, chunked prefill and prefix caching disabled, CUDA Graph and quantization enabled, dummy weights, and `collective_sim` communication with **measured** operator times from the existing profiling-CSV admission path.

Formal contract: TTFT, TPOT, THPT (output-only tokens per second over the frozen window) and E2E, each independently `abs(frontier - vllm) / abs(vllm) <= 0.10`. Aggregation and the request set are frozen before comparison; no request is dropped and no metric is re-aggregated to reach the threshold.

What changed structurally: the H800-64 W1–W6 × three-load campaign, its independent holdout matrices and the H800→H200 generalization campaign are retired. Tickets 24, 25, 26 and 27 are RETIRED with their remaining content folded into `28`; ticket 09 is retargeted from H800-64 to the canonical 64-H200 qualification; ticket 23 is rewritten as the canonical manifest freeze (trace acquisition and hashing, resolved runtime settings, metric contract, windows and repeats); ticket 28 is rewritten as the H200 profiling plus native collection plus four-metric comparison and fit freeze.

Open items this reduction creates: the historical `sharegpt_req300_rate10_llama.jsonl` is not present locally or in the dev-vidur processed traces and must be sourced upstream and frozen (D13); H200 profiles and H200 communication measurements must be collected because H800 data cannot be rescaled; and the formal 64-GPU job must queue for eight idle H200 nodes, since 458 of 512 non-preemptible cards were in use at decision time (D14).

## Canonical workload shared with the GLM/V4 session (D15) — 2026-10-21

The canonical case now uses the first 256 requests of LLMServingSim `v1.0.0` `sharegpt_req300_rate10_llama.jsonl` (SHA-256 `fad744ff3b25bff1e164cc210af4f29176e8d57d1be14c4f981a10bc32ddd61a`), retimed from the trace's own 10.26 QPS to 4.0, and gates THPT as prompt-plus-output tokens per second (`token_throughput_tps`); output-only and request throughput are diagnostics. Workload and metric contract are shared with the GLM-5.3 / DeepSeek-V4-Flash session through one manifest; the communication backend intentionally differs, so cross-task communication error is not directly comparable. Open: one shared choice of retiming rule before ticket 23 freezes the manifest. Evidence: `scratch_evidence/canonical_workload_alignment_20261021/`.

## Shared retiming rule (D16) — 2026-10-21

Both sessions use the historical dev-vidur linear rescale (`scale = 2.590246128`, no seed) through the shared frozen files `data/processed_traces/canonical_sharegpt_req256_qps4/` (GLM/V4 commit `9ed979e`, cherry-picked here as `cb8090b`; checksums and pin test pass; independent 0-mismatch equivalence). Ticket 23's workload items are done; runtime, windows/repeats and metric-mapping items remain. Short prompts (122 of 256 under 32 tokens) need native P/D qualification in tickets 08/09.

## Fixed-length workload replacement (D17) — 2026-10-21

The current formal workload is **128 requests of 6144 prompt / 1024 output tokens at QPS 1.0**, with deterministic arrivals at seconds 0..127. Each of three clean repetitions first runs 16 separate same-shape warmups, then drains both roles before a fresh formal origin. SharedGPT use and short-prompt qualification are deferred; historical files and D16 evidence remain unchanged. No SharedGPT short-prompt transfer count is inferred by extrapolating the R1 formula.

The new replay package is `data/processed_traces/canonical_fixed_6k1k_req128_qps1/`, generated through the existing `tests/comparison/dp_placement_pp/make_trace.py`. It contains eight files covering formal and warmup inputs, CSVs, ID maps, a metadata-only manifest and checksums. Final manifest SHA-256: `43e1e3d0ffa0012bf98eeac6b1b74cf738297af473b77a4d601861e5b1b069a0`.

Validation: 7/7 checksums; 8/8 byte-identical generated-output comparisons in two independent roots; 144 deterministic and distinct native prompt sequences with exactly 6144 IDs; existing SharedGPT tests 2 passed; actual Frontier reader preserved all 128 formal plus 16 warmup Request objects. The manifest explicitly sets the Frontier trace max-token limit to 7168 and all scale factors to 1.0 to avoid the default 4096-token limit clipping prompts. No simulator or generator source change was necessary. Evidence: `test_report_2026-10-21_fixed_workload.md`.

Updated requirements D17, spec §0/§8, current plan, tickets 08/09/23/28 and the public status doc. The four mean/throughput gates, 64-H200 P32/D32 topology, measured backend and runtime feature switches continue to apply. THPT uses all 917504 formal tokens over first submission to last completion, including all elapsed time inside that window; all requests remain in the metric population.

Cross-session adoption remains pending: `fixed_workload_handoff_20261021.md` tells the GLM/V4 owner to adopt the package unchanged. Native long-output completion, concurrency-capable unchunked settings, actual graph replay and H200 performance remain unqualified. R2 judgment and profiler/communication admission work remain open. No new GPU job or other-session edit was performed for the workload redesign.

Tracked deliverables were committed as `0f90ad5` and fast-forward pushed to `feat/step5-preview` backing Draft PR #50. Task records remain local. Independent review closed all three documentation findings and found no critical final package discrepancy.
