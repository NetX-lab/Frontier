# 11: P2 H20032 calibration and P2 H20064 environment/calibration

Current authority: D-H200-MAINLINE-CONTINUE, decision 46 and D-CODEX-DIRECT-RJOB. DSV4 only; GLM stays paused. Current status is recorded below.

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Q1 input qualification ledger complete (`f48457d2`): 27 inputs, 19 QUALIFIED / 6 OPEN / 2 NEEDS_DECISION; D-P2-Q1-CHOICES adopted the independent-pass attention check, kept the original table and started the 16-H200 preemptible nixl6144 measurement; payload route waits for YC; ticket stays OPEN. |
| 2026-11-01 | Completed F2 P2-64 with repeat1 TTFT FAIL, exact A8 coverage, and the full eight-reading F comparison; formal qualification remains open. |
| 2026-11-01 | F2 P2-32 completed with all-repeat numerical PASS and exact collective coverage; only F2 P2-64 remains active. |
| 2026-11-01 | Completed F1a at both scales with failed latency readings; both F2K18 replays and formal qualification remain open. |
| 2026-11-01 | Recorded both F1b terminal coverage failures; no incomplete-request metric gate or formal input change. |
| 2026-11-01 | Synchronized completed A6/F3 evidence and the active F1/F2 comparisons; acceptance criteria unchanged. |
| 2026-10-31 | Synchronized current Codex/H200 operator evidence and pending work; preserved historical snapshots and qualification requirements. |
| 2026-10-22 | W128-V4-S5-STATUS: relabelled obsolete execution snapshots and linked current main progress/05/18; H200 priorities and gates unchanged. |
| 2026-10-22 | Confirmed GLM stop handoff and corrected one-H800 V4 warm-q1 Running/business pending;native04 unadmitted/05 unsealed repair recorded without H200 admission or gate changes. |
| 2026-10-22 | DSV4-only H20032/64 active acceptance; GLM parked without completion dependency, no H800 accuracy edge added and unchanged30%/10%/personal32 preemptibility gates. |
| 2026-10-21 | Reactivated two P2 tasks: H20032 non-preemptible and H20064 wholly preemptible, with separate basic validation and30% formal promotion; personal non-preemptible cap32. |
| 2026-10-21 | Pointed delivery to the fixed 128-request 6144/1024-token QPS-1.0 contract under tickets 16–19; retained historical H200-extension content without reviving its blockers or claiming Step synchronization. |
| 2026-10-21 | Folded into the mainline: H200 became the formal calibration platform, so the separate H200 extension and its scope-freeze milestone are obsolete. Regression and delivery duties moved to ticket 19. |
| 2026-10-20 | Recorded review finding H2 (3e6a421 regression mislabelled as inherited) and the new GPU rule. |
| 2026-10-19 | Synchronized C21/C22 local evidence and remaining collector/metric-input/cache-zero/DP8 gates; no native/formal closure. |
| 2026-10-19 | Synchronized accepted C19/C20 commits, final source/CPU evidence, GLM/V4-only ownership and H800-first fallback; all native/formal limits remain explicit. |
| 2026-10-19 | Applied YC worktree ownership: GLM/V4 delivery here, Step implementation and eventual project-wide integration externally owned; no local Step merge. |
| 2026-10-19 | Recorded actual accepted Step source versus current shared-branch gap and canonical conflict owners; final qualification unchanged. |
| 2026-10-18 | Synchronized latest H800-first/H200-quota-fallback authority; distinguished authorized fallback tests from the later formal H200 scope freeze. |
| 2026-10-18 | Documentation review: made same-HEAD three-model delivery and identity regression explicit; preserved H200 deferral and explicit-only skill usage. |
| 2026-10-15 | Created in the plan phase. |
| 2026-10-15 | Review remediation: an `h200_scope_freeze` milestone, decided by the user, gives the H200 work an objective completion boundary (E2). |
| 2026-10-17 | User deferred new H200 work; the single existing ticket 01 H200 diagnostic may finish independently, and H800 remains primary. |


## Current Codex checkpoint — 2026-10-31

Use [current plan](../plan.md#current-execution-route--codex-cli-h200-operator-path-2026-10-31) and [current handoff](../progress.md#current-session-handoff--codex-cli-2026-10-31). D-H200-MAINLINE-CONTINUE and decision 46 apply; Codex uses personal StepMind Python RJobBackend directly. GLM stays paused. Dated snapshots below are historical where they conflict with this entry.

P2 clean native references are R4 PASS under their original authorization; operator comparisons now exist. B29 P2-32 is numerical PASS in all repeats; P2-64 remains FAIL (repeat 1 TTFT 12.25494%). Full formal acceptance remains OPEN for 05/09 qualification. Q1 (2026-11-01) built the input ledger and started the nixl6144 measurement; see ticket [19](19-comparison-and-acceptance.md). F1a is complete at both scales; P2-64 fails TTFT/TPOT/E2E in every reference repeat. See [F comparison report](../test_report_2026-10-31_f_variant_comparison.md). F2 GPU collection completed; K18 P2-32 is complete with all-repeat numericalPASS; P2-64 is complete with repeat01 TTFT12.792204607%FAIL and the other11 per-repeat metricsPASS. All39680 forward/stage ends match, zero stale events; A8 has9003672/9003672exact calls and zero violations. The complete comparison collector verifies8readings/128metrics/3unavailable; no replay or watcher remains active. Collectives-only03 completed on 16 non-preemptible H200, using 3.315556 observed container GPU-hours. Both F1b CPU replays terminated COVERAGE_MISS with the unchanged fitted parameters and compute tables; no complete E2E readings exist. Their failed G5h validation and exact coverage rejections remain recorded. Fresh A6 completed with qualification failures, and the complete F3 measurements did not yield an admissible affine fit. Preserve historical 64-GPU job authorization/admission; do not backfill a promotion receipt.

## Historical priority milestones

### Current execution status

Use the latest [main progress checkpoint](../progress.md#final-execution-checkpoint--current-2026-10-22)
and current execution sections in tickets [05](05-dsv4-flash-family.md) and
[18](18-native-ground-truth-h200.md). The snapshots below are historical evidence,
not current execution status or H200 admission.

### Historical execution snapshot — superseded (2026-10-22)

GLM06 returned PAUSED_USER_APPROVAL_REQUIRED,no fresh06/tests/build/seal/children;
old packages preserved and complete personal pagination found no GLM jobs to stop.
V4 warm-q1 exp-1022-003156-993689 observed Running,1H800/codesign/non-preemptible,
hardware business/rows PENDING,not H200 evidence. First preflight BEFORE_SUBMISSION;
new descriptor ONLY aggregate_H800_ceiling32->128,527 pins unchanged.
Native04 SEALED_NOT_ADMITTED_NOT_SUBMITTED after staged PYTHONPATH/checkout mismatch
and missing67th breakable_cudagraph.py pin. Candidate05 UNSEALED and
VERSIONED_CHECKER_FIX ses_ea0af5ae9ffe20NcnuumIBrR5v pending. Old5efd/9116/candidate04
immutable,no04 READY_TO_LAUNCH;final NEW freeze/seal requires
ALL_COMPUTABLE_FULL_CHECKER_PREDICATES_PASS and all writers' no-more-writes first.
H200 priorities/non-preemptible32/promotion30%/final10% unchanged,DSV4-only active.

### Earlier execution snapshot — historical; see current status above

**Status:** retained H200 operator/runtime evidence is available within its
recorded scope. Neither32-GPU formal comparison nor its30% promotion receipt
exists. H20064 basic validation is eligible independently; its formal campaign
is BLOCKED. Old incompatible H20064 packages are paused rather than submitted.
Only DSV4 advances now. GLM all tiers are PAUSED_PENDING_USER_APPROVAL;old GLM
source/data/UNVERIFIED references remain retained,not PASS or a DSV4 blocker.
GLM FULL KERNEL_ONLY proposal stays parked,not the current approval question.
Source successor/TCP merges7c1b888/30db587 and combined288PASS25SKIP are accepted
shared evidence;exact checker freeze is in root native-cold-start/FREEZE.json,
not H200 native admission. V4 warm-q1 launch exit1 remains primary diagnosis,
not a submitted/Running/success claim or H200 measured result.

### p2_h200_32 (DSV4 only; GLM paused)
- [ ] Bind H200/step_main32, P16+D16, TP8/DP2/EP16/PP1 per role,
  `preemptible=False`, with the shared68f891b workload.
- [ ] Before submission, page all personal active/queued jobs across sessions;
  total non-preemptible H200 reservations including the new job must be<=32.
  Another such32-GPU case waits for budget; do not silently change its class.
- [ ] Complete actual hardware/topology admission, non-dummy coverage and
  three clean repetitions with16 warmups/128 formal requests each.
- [ ] Record all four errors and the30% promotion receipt; final calibration
  completion retains the existing all-four<=10% requirement.
- [ ] This task has no H800 accuracy prerequisite. It runs at P2 priority,
  using resources not needed by ready higher-priority work.

### p2_h200_64.basic_environment (DSV4 only; GLM paused)
- [ ] Bind H200/step_main64, P32+D32, TP8/DP4/EP32/PP1 per role, with
  `ResourceConfig.preemptible=True` for the entire job and purpose
  `basic_validation`. Verify platform preemptible label `yes`.
- [ ] Prove actual64-rank/two-role environment, device/backend/runtime,
  initialization, representative outputs and terminal evidence. The retained
  single-role32-card G0-B is partial evidence, not proof of this full topology.
- [ ] This phase can run before H20032 reaches30%; no formal accuracy claim
  follows from it and it does not replace the32-GPU comparison.

### p2_h200_64.formal_calibration (DSV4 only; GLM paused)
- [ ] Require BOTH basic environment acceptance and an admitted same-model
  P2-H200-32 receipt with every metric<=30% in every required repetition.
- [ ] Submit the complete64-GPU formal job as preemptible, preserving all
  common workload, metric, coverage and admission rules.
- [ ] Keep interrupted attempts visible. Only complete new clean repetitions
  can count; restarting cannot splice samples or average away preemption.
- [ ] Reach final all-four<=10% and record the actual errors/receipts.

Dependency: `{H20032 admitted comparison<=30%, H20064 basic environment} ->
H20064 formal`. There is no H800-result prerequisite and no invented P3.
Every additional H200 job beyond the personal32 non-preemptible budget must
be preemptible; unchanged historical launch packages cannot bypass this rule.

**Acceptance:** separate case-bound32- and64-GPU DSV4 results,
proper promotion/resource evidence and all-four10% final gates. Regression and
shared-source delivery remain ticket19 duties; Step implementation stays with
its separate owner. Paused GLM completion,GLM/V4 A-B-A reruns and eventual
three-model delivery are not current DSV4 acceptance prerequisites. Retain prior
shared/legacy evidence;new simulation/testing/calibration work is DSV4 only.

## Historical recovery and delivery — no active priority/resource dependencies

### Mainline recovery — 2026-10-20 (historical)

- **H2 (defect):** `3e6a421` changed `_create_batch` to `_get_settled_kv_progress` and broke 7 PP decode-preemption tests and 1 DP-placement test that pass on `bfa82de`. The C35 "inherited failure" review used `da13911`, which already contains `3e6a421`; the same claim is in the PR #51 document `docs/general/glm53-dsv4-vllm028.md`. Fix it so both those tests and the 5 recompute-pricing tests from `3e6a421` pass, correct the records and public document, and add the preemption module to the pre-commit check set. Only `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` fails on `bfa82de` and is truly inherited.
- GPU rule: H800 first without count limit; H200 only for insufficient H800 quota, at most 70 GPUs, non-preemptible.

## Historical local delivery after C21/C22 — not current completion dependencies

Frontier C22 `b5120e6` contains one scoped test file; native C21 `a589f86bd` contains two recorder/test files on its existing development branch. Exact committed/frozen/live identities and source/CPU evidence are in the dated reports. Neither was pushed/merged/deployed or presented as formal model support. Existing inherited A2 test and `.pi/` remain untouched. Current worktree owns GLM/V4 only; Step delivery/integration remains external. H800-first, insufficient-quota H200 fallback, aggregate<=64/non-preemptible and all actual-hardware/formal requirements remain unchanged. [C21](../test_report_2026-10-19_recorder_lifecycle.md), [C22](../test_report_2026-10-19_v4_cache_lifecycle.md).


**What to build:** After the H800 result stands on its own, the H200 compute, communication, and host costs are measured again with dummy-weight native runs (no H800 scaling) and validated on a matrix whose scope the user freezes before the first formal ticket11 H200 campaign. Authorized fallback tests retain the separate resource decision below. Old scenarios and tools are regression-checked. The existing calibration workflow must cover v0.28.0; invoke the `frontier-calibration` skill only if explicitly requested, and a review package and user entry points are delivered. The user merges into `main`. (Spec v2 T11, §13.)

**Blocked by:**
- `h200_scope_freeze`: 10 (`done`). The agent may draft the proposal earlier; the user decides it after ticket 10.
- `h200_profiles_and_validation`: `h200_scope_freeze`.
- `legacy_regression`, `review_package`: 10 (`done`).
- `glm_dsv4_delivery`: 10 (`done`) plus the existing GLM/V4 and legacy acceptance below.
- `three_model_integration` is an external project coordination reference: Step qualified delivery and a reviewed integration HEAD belong to the separate Step/integration owner. Supply shared-interface evidence here without implementing Step or silently waiving project-wide gates.

The H800 evidence is reported without waiting for H200.

**Status:** BLOCKED for this ticket's formal H200 scope and final delivery on the dependencies above. The blanket 2026-10-17 H200 submission pause is superseded by D-H800-FIRST-H200-FALLBACK and D-H200-TEMPORARY-ROUTE: when approved GPU work is needed, check/submit H800 RJobs first; if H800 quota is insufficient, use H200 with at most 70 GPUs and non-preemptible RJobs (D-GPU-H800-UNLIMITED-H200-70, 2026-10-20). Routine fallback tests do not wait for this formal scope freeze, do not close its milestones and cannot qualify H800-specific cells. Record actual hardware/runtime/backend identities. The historical ticket01 job `exp-1017-015318-641429` finished Failed; no replacement is implied by this documentation update. Formal H200 criteria remain recorded, not waived or completed. H800 legacy-regression and delivery work may proceed when their ticket10 blockers close.

## Calibration workflow preparation

The inspected local skill still targets vLLM-BS v0.10.2, with historical site H200/helper defaults. Do not apply those defaults to this H800/v0.28.0 task. Preserve the earlier explicit-invocation preference as a user preference; current local `agents/openai.yaml` is absent, so no current loader-policy setting is asserted. Update only the verified target workflow after evidence exists, within the existing calibration harness; no skill files are changed by this review.

## Historical worktree delivery boundary — 2026-10-19

YC explicitly restricts this worktree to GLM-5.3 and DeepSeek-V4. Step-5-Preview remains under `/data/ycfeng/Frontier/.worktrees/step5-preview`; this ticket does not authorize a Step merge, implementation, native qualification or changes to Step task records. C16/C17 are accepted GLM/V4 structural slices at2da11c8/dc7f382. C19 local harness7cc3115 and C20 logical boundaries3e6a421 are also accepted as scoped source/CPU work; their native/formal/physical-cache limits and retained provenance remain explicit. Neither supplies Step support or completes this delivery ticket. The interrupted C18 Step integration is stopped and its24uncommitted changes were withdrawn after byte-identical evidence retention (17restored/7new removed); GLM/V4 accepted source and C19/A2/.pi were preserved. Deliver reviewed reusable changes, exact branch/commit identities and compatibility limitations to the separate integration owner. Historical read-only Step recovery remains in the [report](../three_model_readiness_20261019/report.md), not as a local execution schedule.

## Historical external three-model coordination — separate owner, not a DSV4 gate

### three_model_integration
- [ ] Provide the external integration owner with reviewed GLM-5.3/original DeepSeek-V4-Flash commits and evidence. That owner coordinates eventual three-model HEAD with the separate Step branch before the user-controlled main merge; this checkbox does not assign Step implementation or branch merging to this worktree.
- [ ] Document C01–C06 interface compatibility for the external Step/integration owner, retaining its separate native/image/production gates. Do not transfer measurements or native qualification across images, models or hardware.
- [ ] Run GLM/V4 A → B → A cold/warm identity regression, wrong-model/runtime/hardware/profile rejection and existing shared-path legacy checks here; hand these results to the external owner for the eventual three-model structural/formal regression. Use the existing prediction/gate tools; report BLOCKED models honestly instead of completing delivery.
- [ ] Record exact final HEAD, runtime/profile/artifact identities and any Step approval still outstanding. H800 evidence does not wait for the final H200 scope; that formal scope remains a later user decision. Fallback tests follow the current H800-first resource rule above.

## Milestones

### h200_scope_freeze
- [ ] Propose the H200 scope from the available H200 capacity: the models, the cells (workloads, load levels, and topology), the native repetitions per cell, the acceptance criteria and thresholds, the optional expansion cells in priority order, and the outcomes that close the ticket with H200 marked `BLOCKED` or deferred (for example, an unavailable topology or a `blocked-on-runtime` model).
- [ ] The user decides the scope; record it in `requirements.md`. The agent does not freeze it alone, and nothing is relaxed after a failure.

### h200_profiles_and_validation
- [ ] Measure H200 compute, communication, and host costs on the same runtime lock.
- [ ] Run every required cell of the frozen scope. Each cell ends with a recorded verdict: PASS or a non-pass label with its RCA. Optional expansion cells are reported when run but are not required for closure.
- [ ] Write the hardware and topology support matrix; no cross-hardware PASS transfer.

### legacy_regression
- [ ] Re-run the old scenario tools and logic regressions. Mark missing data explicitly. Run the simulator wall-time regression (≤10%).

### review_package
- [ ] Extend `skills/frontier-calibration` to the verified v0.28.0 workflow (Q4).
- [ ] Update user docs, add runnable examples for the new models, write the final calibration report and the tracked data index (locations, hashes, generation commands).
- [ ] Write `summary.md` and hand the branch to the user for the merge.

## Acceptance

- [ ] Every required cell of the frozen H200 scope has a recorded verdict, or the frozen scope's `BLOCKED` or deferred outcome is recorded with its evidence.
- [ ] No H200 accuracy claimed from H800 ratios.
- [ ] Every new native run used dummy weights.
- [ ] Every artifact is reproducible from tracked manifests.
- [ ] The merge is left to the user.
