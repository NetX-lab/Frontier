# 10: P0 H80032 calibration and P1 H80064 validation/calibration

Current route authority: D-DSV4-H800-MAINLINE-CORRECTION in `requirements.md`
(2026-10-25); it sets milestone order. Earlier authority: D-DSV4-ONLY-GLM-PAUSED plus D-CALIBRATION-PRIORITIES in
`requirements.md`. The two H800 priority tasks below are active for DSV4 only;
all GLM tiers/tests/new packages/calibration pause pending new explicit user
approval. Old36-cell,multi-QPS and holdout stay retired. Ticket17 binds DSV4,18 owns
native evidence and19 owns both the promotion receipt and final metric gate.

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Checked the five h4a_p0_diagnostic items (DIAGNOSTIC result in test_report_2026-10-25_h4a_p0_diagnostic.md); next is the YC checkpoint. |
| 2026-10-25 | D-DSV4-H800-MAINLINE-CORRECTION: added milestone h4a_p0_diagnostic as the next P0 step, pointed the status to the mainline handoff, and closed the P0 binding and native-reference items with native06 R4 evidence. |
| 2026-10-22 | W128-V4-S5-STATUS: relabelled obsolete execution snapshots and linked current main progress/05/18; H800 milestones and gates unchanged. |
| 2026-10-22 | Final checkpoint: GLM pause confirmed;warm-q1 corrected one-H800 job Running/business pending;native04 immutable sealed/unadmitted,05 unsealed pending versioned full-checker repair. |
| 2026-10-22 | Active H80032/64 acceptance now DSV4 only; GLM all tiers paused without paired-model dependency, exact frozen successor/TCP accepted and warm-q1 launch failure recorded. |
| 2026-10-22 | Recorded completed P0 native collections with UNVERIFIED admission, source/preparation milestones and pending successor freeze; official mean/four-error gates remain unchanged. |
| 2026-10-21 | Reactivated as P0-H800-32 and P1-H800-64 with separate basic/formal phases and same-model30% promotion, preserving10% final acceptance. |
| 2026-10-21 | Clarified that the retired matrix does not retire the current 3-clean-repetition rule; tickets 16–19 now own the fixed 128-request 6144/1024-token QPS-1.0 case and full-completion THPT window. |
| 2026-10-21 | Superseded by the canonical H200 calibration contract; formal scope moved to tickets 16–19. History retained. |
| 2026-10-18 | Documentation review: audited formal status without changing the 36-cell, repeat, holdout or metric criteria. |
| 2026-10-15 | Created in the plan phase. |
| 2026-10-15 | Review remediation: `thresholds_frozen` gets its own blockers and separate request E2E and request-throughput verdicts (S4); validated probe output is required before formal runs (S2); GLM-5.3 representative cells finish with recorded verdicts before the DeepSeek-V4-Flash cells start (S1, Q12). |
| 2026-10-15 | Re-review fix R3: each model's light-mode measurement is compared with the frozen limit before its formal runs. |

## Current priority milestones

### Current execution status

Use the [mainline handoff](../progress.md#current-session-handoff--2026-10-25-mainline-correction)
and the [current route](../plan.md#current-execution-route--dsv4-h800-mainline-2026-10-25).
The P0 step `h4a_p0_diagnostic` below is done; the next step is the YC checkpoint. The snapshots below are historical
evidence, not current execution status or P0 admission.

### h4a_p0_diagnostic (DSV4 only, done 2026-10-25)

First Frontier prediction of the P0 case, labelled DIAGNOSTIC. It does not count
toward the 30% promotion or the 10% final gate. Steps and criteria are in the
[plan H4a section](../plan.md#h4a--first-diagnostic-p0-prediction). Result: TTFT 17.35%,
TPOT 2.50%, E2E 2.10%, THPT 0.54% against the mean of native06 R4
([test report](../test_report_2026-10-25_h4a_p0_diagnostic.md)); substitutions
S1-S5 replace profile04 with profile02 and are listed in the report.
- [x] H4a-1: `design.md` section with cost slots, step-cost definition,
  profiler-perturbation check, out-of-range rule, KV capacity binding, file list
  and line estimate; inputs copied out of `/data/ycfeng/tmp`.
- [x] H4a-2: V4 PREFILL and unified DECODE execution with grouped NIXL, through
  existing boundaries (owned by tickets 05 and 07).
- [x] H4a-3: decode and prefill step-cost tables from the complete profile04
  outputs (owned by ticket 09).
- [x] H4a-4: P0 run of 16 warmups plus 128 formal requests per repetition; four
  errors from the `tests/comparison/e2e_gate/` client gate against each native06
  R4 round and their mean. Departure: one formal replay without warmups.
- [x] H4a-5: TTFT/TPOT error attribution, substitution list, test report and YC
  checkpoint.

### Historical execution snapshot — superseded (2026-10-22)

GLM06 PAUSED_USER_APPROVAL_REQUIRED handoff complete,no fresh06/tests/build/seal/
children. Corrected V4 warm-q1 exp-1022-003156-993689 observed Running,1H800/codesign/
non-preemptible,correct personal creator/local NFS/locked image;hardware business/
rows PENDING. First preflight failed BEFORE_SUBMISSION because aggregate32 rejected
Step32+requested1;new descriptor ONLY aggregate_H800_ceiling32->128,527 pins unchanged.
Native04 SEALED_NOT_ADMITTED_NOT_SUBMITTED after staged PYTHONPATH/checkout mismatch
and missing67th breakable_cudagraph.py pin. Candidate05 UNSEALED under same V4 owner;
VERSIONED_CHECKER_FIX ses_ea0af5ae9ffe20NcnuumIBrR5v. Old5efd/9116/candidate04 immutable,
no04 READY_TO_LAUNCH;final NEW freeze/seal only after
ALL_COMPUTABLE_FULL_CHECKER_PREDICATES_PASS and all writers' no-more-writes.
DSV4-only/mean/30%/10% unchanged;Running microbenchmark is not P0 native admission.

### Earlier execution snapshot — historical; see current status above

**Status:** P0 native collections are completed,not running;V4 candidate03
exp-1021-220603-144957 has384/384 formal,431/432 all (first idle invalid,rep2
warmup502,rep3 client-clean);GLM05_final exp-1021-221509-816489 has432/432 including
384 formal (rep1 first idle unverified,rep2/3 client-clean). Both global references
remain UNVERIFIED / REFERENCE_UNSTABLE;no admitted four-metric result exists.
P1 basic validation is eligible when its own prerequisites/resources
are available; P1 formal calibration is BLOCKED by the missing P0 promotion
receipt, not by a requirement that P0 first reach10%.

DSV4 is the sole active completion target. All GLM simulations/tests/calibration,
new packages and FULL KERNEL_ONLY proposal follow-up are parked pending new
explicit user approval;preserve completed source/data/UNVERIFIED metrics,no GLM
completion edge. V4 warm-q1 dc86fad/source d2878ed/seal7929afae… is retained;
latest launch attempt exits1 before business cause is established. Primary
diagnoses `/data/ycfeng/tmp/glm53_dsv4/v4_q1_warm_20261022_launcher.log`;no submitted/
Running/success/new timing claim despite legacy glm helper names. Cold-start
R1 APPROVE241PASS21SKIP merged7c1b888,TCP merged30db587,combined288PASS25SKIP.
All writers stopped;accepted frozen checker
5efd00ea6544d64440c535d06ad415b6a4d1a25b96ce8863cf4944568f37ec36 in root
`task_memory/task_2026-10-21_native_cold_start/FREEZE.json`. Native admission
PENDING;pin checks before/after seal,old9116ab7 immutable,no POST retry. DSV4
successor04 owner ses_ea0bffa17ffes1B4qUTxbDHVTT active;GLM06 stop handoff pending.
Source repair does not admit old failures. P90 TTFT/TPOT/E2E is recommendation only;
official means plus unchanged total THPT still gate until explicit approval.

### p0_h800_32 (DSV4 only, highest priority; GLM paused)
- [x] Bind the shared68f891b workload to H800/codesign32, P16+D16,
  TP8/DP2/EP16/PP1 per role, `preemptible=False`. Evidence: ticket 17 P0-scoped
  binding and `../canonical_case_20261021/p0_binding/root_verification.json`.
- [ ] Complete model/runtime, hardware-specific coverage and non-dummy
  prerequisites using matching01,05–09,17/18;no04/16/GLM predecessor. Reusable source work and native
  preparation proceed in parallel.
- [x] Collect and admit three clean native repetitions; each has16 independent
  warmups and128 exact6144/1024 formal requests at QPS1. Evidence: native06 R4,
  `../native06_acceptance_20261023/primary_acceptance_r4.json` and `client_report.md`.
- [ ] Compare client TTFT, TPOT, total-token THPT and request E2E separately.
  Publish the case-bound30% promotion verdict as soon as all four qualify;
  keep calibrating P0 until all four<=10% under ticket19.

### p1_h800_64.basic_validation (DSV4 only; GLM paused)
- [ ] Bind H800/codesign64, P32+D32, TP8/DP4/EP32/PP1 per role,
  `preemptible=False`, with purpose `basic_validation`.
- [ ] Verify all64 ranks/roles, real device/backend/runtime, initialization,
  representative forwards/output and terminal evidence. Reuse existing
  matching evidence only for what it actually proves.
- [ ] This phase may run before P0 reaches30%, subject to P0 resource priority.
  It neither collects an early formal campaign nor claims calibration accuracy.

### p1_h800_64.formal_calibration (DSV4 only; GLM paused)
- [ ] Require BOTH this model's basic validation and ticket19's admitted
  P0-H800-32 receipt: every four-metric error<=30% in every required repetition.
- [ ] Then collect the64-GPU formal reference and predictive comparison,
  preserving all common case/admission/coverage requirements.
- [ ] Reach final four-metric errors<=10%;30% is only the entry threshold.

Dependency: `{P0 admitted comparison<=30%, P1 basic validation} -> P1 formal`.
There is no edge from P0 final10% to P1 basic validation. Current DSV4 promotion
uses only DSV4 same-hardware32-GPU receipts;GLM is paused,not a predecessor.
P0 completion does not wait for P1 or H200.

**Acceptance:** DSV4's P0 result has all four<=10%; its P1 result was
started only after its recorded30% gate and then also reaches10%. Missing
results remain TODO/BLOCKED,never inherited from another hardware or scale.
GLM all-tier outcomes stay PAUSED_PENDING_USER_APPROVAL,not falsely completed
and not required for current DSV4 acceptance.

## Historical36-cell campaign — inactive; no current dependencies

**What to build (historical):** Frontier's predictive online E2E on topology A (P32+D32 on 64 H800) passes the frozen per-cell thresholds against clean dummy-weight native references for both models. That means 2 models × W1–W6 × 3 load levels = 36 main cells, each with at least 3 native repetitions, plus a same-topology holdout. Every failure has an RCA, and every run is in the manifest. GLM-5.3 representative cells run first. (Spec v2 T10, §9.3–§9.5, §10.)

**Blocked by:**
- `thresholds_frozen`: 02 `metric_normalization`, and the light-mode perturbation measurement of 02 `probe_validation[GLM]`.
- `representative_cells[GLM]`: `thresholds_frozen`, 01 `G0-D[GLM]`, 09 `full_frontier_path[GLM]`, 02 `probe_validation[GLM]`.
- `representative_cells[DSV4]`: `representative_cells[GLM]` finished with recorded verdicts (a PASS is not required), 01 `G0-D[DSV4]`, 09 `bundle[DSV4]`, 02 `probe_validation[DSV4]`, 15 `dsv4_agreement`.
- `main_36_cells`: both representative milestones, and all decisions closed (`12[GLM]`, `12[DSV4]`, 13 with `dsv4_agreement`, `14[GLM]`, `14[DSV4]`, 15 with `dsv4_agreement`).
- `independent_holdout`: `main_36_cells`.
- `rca_and_gate`: runs alongside every formal milestone and closes after `independent_holdout`.

It does not depend on the optional topology B.

**Status:** NOT STARTED; blocked on target native qualification, represented/trained paths and validated measurement tooling. Existing fixtures and 77 baseline comparisons do not close any formal cell.

## Milestones

### thresholds_frozen
Record every item in `requirements.md` before any formal run. None is relaxed after a failure.
- [ ] The acceptance thresholds and the metric definitions handed over by 02 `metric_normalization` (client TTFT, TPOT, ITL, output-length policy).
- [ ] Separate request E2E and request-throughput checks, judged on their own like every other gated metric. Reuse the existing gate contract in `tests/comparison/e2e_gate/README.md` (`request_e2e_time_ms` as the mean request E2E time of the formal requests, `request_throughput_rps` as formal requests over the formal window, the formal window from the first formal arrival to the last formal completion, and the existing per-metric relative-error limit), or record an explicit new-case contract that names the request population, formal window, statistics, and verdict rule. Either way, failed and unfinished requests are counted and reported, never dropped.
- [ ] The tail-metric eligibility rule (sample size and reference stability for p95 TTFT and ITL), the reference-stability check behind `REFERENCE_UNSTABLE`, and the reduced-sample policy for long-context workloads.
- [ ] The light-mode perturbation limit (proposed median ≤3%), checked against the measured value of 02 `probe_validation[GLM]`. A measured value above the limit leaves light mode unqualified for formal runs; the limit is not raised.
- [ ] The minimum completed requests per run (proposed ≥1,000).

### representative_cells (per model, GLM first)
- [ ] Before the model's first formal run, compare its 02 `probe_validation[m]` light-mode measurement with the frozen limit and record in `requirements.md` whether light mode is qualified for that model. If it is not, that model's formal runs use clean mode only.
- [ ] Per model, find the native sustainable capacity, then fix the absolute rates at about 25%, 60%, and 85% of it. Frontier and vLLM replay the same Poisson trace (seed `20261014`).
- [ ] Run 3 representative cells per model before expanding. The GLM-5.3 cells finish and their verdicts are recorded before the DeepSeek-V4-Flash cells start.

### main_36_cells
- [ ] Run all 36 cells with at least 3 clean repetitions each, warm-up isolated, and the cells registered in the E2E accuracy gate.
- [ ] Judge each cell on its own; never average errors across cells. Unstable references are reported as `REFERENCE_UNSTABLE` with the full repetition spread.

### independent_holdout
- [ ] Run a same-topology holdout with length and arrival traces not used in calibration. If the user approves topology B, it covers two load levels of W3 and W6.

### rca_and_gate
- [ ] RCA for every non-pass, following the ticket 09 order. Keep a GPU-hours ledger including initialization, compilation, warm-up, and failures.
- [ ] Write the initialization and routing stability report and the run manifest with every submitted run.

## Acceptance

- [ ] G0-D is complete, all D-* decisions are closed, and every required feature (online, real chunking, valid NIXL, graph replay) has execution evidence.
- [ ] Formal references come only from runs after the model's 02 `probe_validation` closed, in an observation mode qualified by `thresholds_frozen`.
- [ ] The frozen thresholds pass, including separate per-cell verdicts for request E2E and request throughput. A pass on client TTFT, TPOT, and output throughput does not hide a failure of either.
- [ ] No native future batch, host duration, transfer time, or expert histogram drives a predictive run.
