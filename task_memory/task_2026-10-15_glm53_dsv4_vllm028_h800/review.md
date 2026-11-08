# Review: current specification and tickets

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-08 | Added section 12: lead acceptance of the codex-implement H200 mainline review, the disposition of its four findings, two new findings the lead added, and the remaining open items. |
| 2026-10-18 | Added section 11: current documentation-only audit, all-ticket evidence/status assessment, main-path recovery, official workflow checks, unresolved choices and execution handoff. Sections 1–10 remain historical. |
| 2026-10-15 | Reviewed the existing planning package against the binding sources, approved decisions, execution rules, and inspected repository code. |
| 2026-10-15 | Added §9: the item-by-item disposition of every finding, the remediation applied, and its verification. Sections 1–8 are kept as the original review. |
| 2026-10-15 | Added §9.4: independent re-review of the remediation and the five precision fixes R1–R5. |
| 2026-10-15 | Added §10: ticket 00 phase review, its findings, the fixes, and the primary spot checks. |

## Current review entry

Read **section 12** for the 2026-11-08 lead acceptance of the H200 mainline review, which carries the current
status of this task's completion question. Section 11 holds the 2026-10-18 resume assessment and corrected
execution path; sections 1–10 describe earlier snapshots, and their ticket states and original approval rules are
historical.

## 1. Verdict and scope

**Verdict: changes requested before treating this package as an execution-ready plan.** The overall scope and reuse strategy are sound. Two high-priority dependency defects remain: the GLM-first implementation order is weaker than the approved decision, and validated observation tooling is not an explicit prerequisite for formal calibration. There are also narrower sequencing, acceptance, and documentation issues below.

This is a **planning-phase document review**, not acceptance of model support or a branch implementation review. Ticket 00 remains `ready-for-agent`; tickets 01–15 remain blocked. The review request does not start their execution. Missing future implementations, native runs, profiles, and D-* decisions are not findings by themselves.

The review used the existing task directory. The requested deliverable is this report; recommendations below have not been applied to the spec, plan, or tickets.

### Reviewed snapshot

- Repository: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
- HEAD: `01fb2ce5984f1d26135ea9bceef96f8e39be0401`.
- HEAD parent / execution baseline: `bfa82de80413b5f4b38e4990a80b5108f9521f50`.
- Tracked diff from baseline: only `CONTEXT.md` and `docs/adr/0001-dummy-weight-native-ground-truth.md`, 67 inserted lines. There is no feature implementation diff to accept.
- Initial `git status --short`: empty. Task records are ignored by Git; a clean status does not inventory them.
- Reviewed documents: `requirements.md`, `spec.md`, `plan.md`, `progress.md`, all 16 tickets, all three `source_spec_v2/` files, the glossary, and ADR 0001.
- Code inspection was targeted to the claimed reuse boundaries and baseline facts, not an exhaustive runtime audit.

### Authority and citation convention

Approved Q1–Q11 decisions and explicit task deltas override the read-only v2 sources. Otherwise the original v2 spec and campaign YAML remain binding (`spec.md:10`, `requirements.md:42–68`). A missing detail in the synthesis does not revoke a binding source requirement; it can still leave a dangerous ambiguity in an executable ticket.

Paths below are relative to this task directory unless prefixed `repo:`. **V2** means `source_spec_v2/Frontier_GLM53_DeepSeekV4Flash_vLLM028_H80064_Spec_v2.md`; **YAML** means `source_spec_v2/Frontier_vLLM028_H80064_Campaign_Template_v2.yaml`. Line references identify the reviewed snapshot, before this review's additions to requirements/progress.

### Review participation

| Reviewer | Assigned scope | Session |
| --- | --- | --- |
| Spec reviewer | Binding-source and approved-decision fidelity; metric and acceptance coverage | `ses_ec1bded36ffeDh8jQee3crvgvo` |
| Standards/readiness reviewer | All tickets, dependencies, execution rules, objective completion and evidence records | `ses_ec1bdec9affeahEAVpoy0CztPV` |
| Code exploration reviewer | Baseline/code anchors, reuse seams, hashes and historical paths | `ses_ec1bdec7bffeidiFEWpVjpTxDz` |
| Primary agent | Integration and independent reinspection of decisive documents, code excerpts, Git identity, and source hashes | Current session |

Severity: **P1** means repair before dependent implementation or formal calibration; **P2** means a material planning/acceptance gap to resolve before the affected milestone; **P3** means a lower-impact record or precision issue. These are review severities, not measured runtime failures.

## 2. Spec findings

### S1 — P1: the dependency graph does not enforce the approved GLM-first full path

**Requirement.** Q2 says DSV4 native G0-A/B and raw kernel profiling may run in parallel, but its Frontier simulation work starts after the GLM path is complete (`requirements.md:43`). `spec.md:31` repeats “full path first.”

**Evidence.** `tickets/05-dsv4-flash-family.md:5–7` requires only `04.family_predictor_integration`, D-PD, and D-ROUTING for DSV4 integration. `plan.md:60` reproduces that edge. GLM's PD, workflow, and trained bundle remain downstream work in `plan.md:62–68`; wave 6 explicitly runs 05 integration alongside 07 and 08 integration (`plan.md:85–86`).

**Analysis.** The written blockers allow DSV4 implementation after GLM operator/family integration, while GLM may still lack working NIXL PD and v0.28 workflow integration. A family-level Replica path is not the complete PD simulation path promised by Q2. The sequencing defect exists even if every listed blocker is honored.

**Recommended correction.** Define an objectively verifiable `GLM.full_frontier_path` milestone covering the required GLM family, PD, workflow, and trained full-simulation checks. Make DSV4 integration depend on it. Keep DSV4 native qualification and raw profiling parallel. Give shared tickets 06–09 explicit per-model completion boundaries so a GLM-first milestone does not accidentally wait for all DSV4 work. Clarify whether representative predictive E2E is included; do not silently reinterpret the approved phrase as either “ticket 04 only” or “all GLM matrix cells.”

**Closure evidence.** The updated graph and ticket 05 must agree on the same named milestone and its acceptance checks. Also express ticket 10's stated GLM representative-cell precedence as an edge if it is intended to be strict (`tickets/10-h800-predictive-e2e.md:3,6–7`).

### S2 — P1: validated instrumentation/probe output is not a prerequisite for formal calibration

**Requirement.** V2 requires validated causal pairing, clock/scope handling, and clean/light perturbation evidence; T02 produces the adapters and perturbation report (V2:539–554,729–735). These are implemented as explicit work in `tickets/02-observer-instrumentation-gate-adaptation.md:16–25`.

**Evidence.** `plan.md:47` ends the T02 chain at `02.probe_validation` with no outgoing edge. Downstream work consumes `02.observer_schema`, while ticket 09 bundle blockers and ticket 10 formal-run blockers do not require validated adapters/probes (`plan.md:62–70`; tickets 09:5–8 and 10:5–8). Freezing a perturbation **limit** at ticket 10:17 is not the same as measuring and passing it.

**Analysis.** The blocker lists permit formal cost replay/E2E while log pairing, phase/graph normalization, or observer perturbation remain unverified. This can invalidate expensive measurements after collection. Current CPU code demonstrates why this is substantive: `repo:frontier/profiling/cpu_overhead/vllm_cpu_probe.py:283–313` pairs scheduled steps and forwards by position, `:323–329` subtracts the whole logged KV-load interval, and `:340–345` treats non-FULL modes as eager. A schema alone does not validate these semantics for asynchronous v0.28 execution.

**Recommended correction.** Add explicit, model/runtime-scoped validated-tooling prerequisites to formal consumers: cost replay that consumes the adapter output and ticket 10 formal reference collection/normalization. Include the measured clean/light result and frozen metric contract. Exploratory traces may still be collected earlier and labeled diagnostic/unverified. If metric normalization depends on D-ASYNC/D-DP-PD, separate the independently validated observer subset from the post-decision normalization subset; do not introduce a circular dependency by making all decision evidence wait for all of T02.

**Soft-gate distinction.** This is a pre-launch qualification/acceptance dependency. During a running allocation of 32 or more GPUs, an auxiliary validator failure must still remain non-blocking, preserve healthy work, and mark affected artifacts unverified (`requirements.md:73`).

**Closure evidence.** A dependency walk from formal replay/E2E reaches the relevant validation milestone. Recorded clean/light comparison and causal/clock negative tests are required when that milestone is later executed.

### S3 — P2: the wave schedule places DP4 decision evidence after the decision

**Requirement.** D-DP-PD observes two lanes and then the main topology's DP4 (V2:955–957). Ticket 15 explicitly uses `01.G0-D` for DP4 and includes that extension in its work (`tickets/15-decision-d-dp-pd.md:9,15`).

**Evidence.** `plan.md:85` puts decision evidence/cards in wave 5 and says user decisions gate the next wave. `plan.md:86` puts `01.G0-D` in wave 6. The detailed graph is closer to the intended order (`plan.md:58`), so the two execution representations disagree.

**Analysis.** Following the wave table literally creates a wait: the decision needs DP4 evidence, while obtaining the evidence is scheduled behind that decision. Alternatively, the decision could be prematurely closed using only two-lane evidence. This is a wave-order conflict; no cycle in the explicitly listed milestone DAG was demonstrated.

**Recommended correction.** Schedule native G0-D before the final DP4-backed D-DP-PD decision, or explicitly exempt native qualification from that wave's simulator-decision gate. If the decision is staged, define a two-lane observation milestone and a separate DP4-backed final decision. Native G0-D must not require Frontier placement implementation.

**Closure evidence.** The graph, wave table, and ticket 15 expose the same evidence-before-decision order, including the qualification ladder G0-B → G0-C → G0-D.

### S4 — P2: independent request E2E and request-throughput acceptance is under-specified

**Requirement.** V2:652 preserves request E2E/request-throughput request/window and failure accounting. V2:686 explicitly requires their independent checks under the existing or an explicit new-case contract, in addition to the client TTFT, TPOT, and output-throughput thresholds.

**Evidence.** `spec.md:215–222` lists request E2E among recorded metrics but omits its independent gate and request throughput. Ticket 10 freezes metrics/thresholds and then requires them to pass without naming these two checks (`tickets/10-h800-predictive-e2e.md:16–17,34–38`). The existing gate already has them: `repo:tests/comparison/e2e_gate/README.md:77–90` lists `request_e2e_time_ms` and `request_throughput_rps`, with each metric gated independently.

**Analysis.** The binding V2 requirement still applies, so this is a synthesis/ticket completeness defect rather than an authorized removal. An executor relying on the operational checklist could treat request E2E as reporting only and omit request throughput. This matters when handoff latency appears after a P-side first token rather than entirely in client TTFT (V2:654).

**Recommended correction.** Add both independent gates to `thresholds_frozen` and Acceptance. Name the reused contract or define the new-case contract before formal runs: request population, formal window, failed/unfinished accounting, statistics, and verdict rules. Freeze tail eligibility, reference-stability checks, and reduced-sample policy alongside them. Do not invent numeric thresholds in this review or relax them after failures.

**Closure evidence.** The acceptance checklist and versioned case contract require separate verdicts for both metrics; their failure cannot be hidden by passing TTFT/TPOT/output throughput.

## 3. Standards and execution-readiness findings

The readiness reviewer independently confirmed S1–S3. They are not counted again here. The following are additional findings on execution rules and task records.

### E1 — P2: ticket 00's remote-evidence instruction omits the required operator routing

**Evidence.** `tickets/00-baseline-audit.md:17` directly describes cloud-bundle inspection via `ssh shai-ycfeng`. Current OpenCode operating instructions require StepMind/intranet operations to be routed through the `stepmind` tool to `stepmind-helper`, with only filtered receipts returned. The general remote SSH allowance does not override that specific routing rule.

**Analysis.** This is an unsafe execution instruction under the configured harness, not evidence that anyone already performed an unauthorized operation. Local manifest/source inspection remains straightforward.

**Correction and closure.** Make ticket 00 explicitly distinguish local manifest inspection from operator-routed remote availability/hash checks. Record filtered receipts and cloud paths locally, leaving raw private evidence with the operator. No remote operation was needed for this review.

### E2 — P2: H200 has no objective completion boundary

**Evidence.** Ticket 11 calls for a “small matrix that grows as resources allow” and “Run a small H200 matrix, then expand” (`tickets/11-h200-regression-delivery.md:3,11–14`). Its Acceptance section requires correct provenance and restrained claims but does not specify which H200 results make the ticket done (`:24–29`). `plan.md:95` requires every acceptance criterion to have evidence.

**Analysis.** Neither the synthesis nor the binding source freezes a full H200 campaign here. That flexibility is legitimate, but the ticket has no explicit future scope-freeze or stopping condition. Both a minimal sample and indefinite expansion can satisfy the prose. This is a readiness judgment, not a violation of an existing numeric H200 matrix requirement.

**Correction and closure.** Add a pre-execution H200 scope-freeze milestone: required models/cells, repetitions, criteria, optional expansion, and blocked/deferred outcomes, with the decision owner identified. Preserve the independent H800 deliverable. A later execution must compare results against that frozen scope before marking ticket 11 done.

### E3 — P2: the completed planning verification claim lacks inspectable supporting records

**Evidence.** `progress.md:35–38` records 5 failed / 48 passed in each checkout, says failure lists were identical, and concludes “PASS: no new failure.” Before this review, the task directory contained no dated test report and no cited failure-list/output artifact. The task-record verification instructions require commands, environment, expected/actual values, and supporting logs.

**Analysis.** The historical result may be correct. Matching counts alone would not prove matching failures; the recorded assertion of identical lists is not independently auditable without node IDs/output. This review does not overturn the old result and did not rerun those tests.

**Correction and closure.** Recover retained logs and record exact checkout identities, environment, commands, and all failed node IDs in a historical test report. If no output survives, label the old comparison as not independently verified; a fresh comparison may establish a new result but cannot recreate the old run. This review's own verification report does not repair that historical evidence gap.

### E4 — P3: milestone names and aggregate completion rules are inconsistent

**Evidence.** The graph uses `03.inventories`, `03.layer`, `03.cache`, and `03.routing` (`plan.md:49–58`), while ticket 03 defines `layer_inventory`, `cache_inventory`, and `routing_inventory` (`tickets/03-layer-cache-routing-inventory.md:11–25`). Ticket 09's blockers use `bundle[GLM]`/`bundle[DSV4]`, but its milestone headings are `predictor_bundle`, `cost_replay`, and `coverage` (`tickets/09-predictor-bundle.md:5–21`).

**Analysis.** The intent is understandable, but the plan says edges bind named milestones. An executor cannot mechanically determine which per-model subchecks close an aggregate `bundle`. Whole-ticket `done` blockers in ticket 09 also need reconciliation with S1's per-model sequence.

**Correction and closure.** Use one set of milestone identifiers in tickets and graph. Define aggregate milestones as conjunctions of explicit per-model checks. Validate that every dependency target resolves and that no aggregate requires work intended to start after it.

### E5 — P3: maintained tickets lack the required modification history

**Evidence.** All 16 reviewed tickets go directly from title/description/status to milestones or work; none has a Modification History table. Examples: ticket 00:1–9 and ticket 15:1–13. The task-record instructions require a brief history at the top of maintained documents under `task_memory/`.

**Analysis and correction.** This is a low-impact documented-rule mismatch. Add short creation/update entries when revising the tickets. Do not modify the read-only source snapshots to impose this convention retroactively.

## 4. Code-grounded clarifications for ticket 00

These are source-map corrections or implementation cautions, not additional blockers on the whole plan. Ticket 00 already owns verification of source claims.

### C1: the binding source overstates the existing attention-layer resolver ownership

V2:99 proposes mapping GLM index reuse and V4 compressed layers through `frontier/execution_time_predictor/layer_contract_resolution.py`. The current `_resolve_typed_layer_contract` explicitly resolves a **typed FFN contract** (`repo:frontier/execution_time_predictor/layer_contract_resolution.py:66–75`); `LayerKind` describes dense/routed/shared FFN kinds (`repo:frontier/model_architectures.py:41–46`). Attention identity is already attached through `get_layer_attention_spec` in `_assemble_stage` (`repo:frontier/execution_time_predictor/base_execution_time_predictor.py:55–87`).

Ticket 00 should document this distinction in `source_map.md`, and the task spec can record a source-owner correction without modifying V2. Do not assume a file called “layer contract” owns the indexer/attention schedule. Final extension boundaries still follow the approved design process.

### C2: “preserve legacy layer-wise latency hiding” is not evidence of an existing overlap model

`spec.md:164–169` describes the P2P latency-hiding rule as retained in its old scope. Current config labels `enable_latency_hiding` as a future extension (`repo:frontier/config/kv_cache_transfer_config.py:50–54`). A repository search finds the capability getter and metadata fields, but no production caller implementing layer-wise overlap through this capability. `repo:frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py:112–113` only returns the flag.

Keep the intended NIXL rejection. Record the actual baseline behavior; do not implement new P2P overlap merely to satisfy the wording “preserve.” This inspection concerns that flag/capability path and is not a general proof that no other work overlaps in the simulator.

### C3: historical task locations and one config-class name need precision

- Ticket 00:18 lists three historical task names without their checkout root. The exploration reviewer verified their directories under `/data/ycfeng/Frontier/task_memory/`: `task_2026-09-21_oversized_module_split`, `task_2026-09-21_issue26_correctness_pr`, and `task_2026-09-27_pr_stack_vllm_e2e_accuracy_gate`. They are not local task directories in this worktree. Only directory existence was checked; their evidence was not accepted or audited.
- The stage-admission record uses the separate `/data/ycfeng/Frontier/.worktrees/stage-admission-ordering/task_memory/` root, as already noted in `spec.md:273`.
- `spec.md:287` names `ExecutionTimePredictorConfig.enable_dummy_mode`. The actual base class is `BaseExecutionTimePredictorConfig` in `repo:frontier/config/execution_time_predictor_config.py:13`, with `enable_dummy_mode` at `:240–245`.

Use exact locations/symbols when ticket 00 writes its source map; these corrections do not change the intended feature scope.

## 5. Verified strengths and baseline anchors

| Area | Evidence | Assessment |
| --- | --- | --- |
| Language-only scope | `requirements.md:68`; `spec.md:117,208,253`; ticket 01:14,24; ticket 03:15 | Explicitly excludes vision and other multimodal modeling/workloads. Text-required metadata exceptions do not authorize multimodal qualification. |
| Ground truth and runtime | `spec.md:29,112–118`; ADR 0001; ticket 01 | Exact vLLM SHA, dummy weights, full structure, and metadata-only downloads are retained. Dummy timing and DP dummy forwards are separate concepts. |
| Native routing and leakage prevention | `spec.md:190–194`; tickets 14 and 10:38 | No inherited forced-uniform routing; independent calibration and holdout separation are explicit. |
| Main campaign | `spec.md:101,209–221`; ticket 10:3,24–28 | 36 cells, at least 3 clean repeats per cell, per-cell verdicts, and independent same-topology holdout are present. This implies at least 108 clean main-cell runs, excluding qualification, pilots, holdout, and failures. |
| Decision authority | `requirements.md:44,56,74`; tickets 12–15 | Evidence and design cards precede user decisions. Pending D-* choices are intentional, not missing implementation. |
| Resource and publication authority | `requirements.md:45,48,73–75`; `plan.md:97–104` | CPU/GPU resources pre-authorized; external publication and interface/schema changes retain their separate approvals; large-run validators are soft gates. |
| Regression strategy | `spec.md:234–249`; `plan.md:88–95`; ticket 00:23–27 | Existing seams and at least 50 concrete scenarios are specified. Those future checks have not yet been run for this feature. |
| Frozen DSA | `repo:frontier/attention/families.py:222–235` | `operators=()`, `dsa_frozen=True`, `execution_enabled=False`; the stated implementation gap is real. |
| Connector and slowdown isolation | `repo:frontier/config/kv_cache_transfer_config.py:8,79–87` | Only `none`/`p2p_nccl` exist. Adding NIXL alone would not reject nonzero slowdown, so ticket 07's explicit negative case is necessary. |
| DP placement scope | `repo:frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py:25–52` | The guard checks MONOLITHIC, one Replica, vLLM V1, and a dense-DP restriction. Topology B also encounters the one-Replica restriction; merely removing the PDD check is insufficient. |
| Ordered layer reuse | `repo:frontier/execution_time_predictor/base_execution_time_predictor.py:55–87` | Family/variant identity and ordered layer records already exist. This does not itself implement index-reuse or compression-boundary timing. |
| Legacy metric meanings | `repo:tests/comparison/e2e_gate/README.md:74–90` | Internal TTFT and total-token throughput match the synthesis. New client/ITL/output-only/tail metrics require the planned extension. |
| CPU producer reuse | `repo:frontier/profiling/cpu_overhead/vllm_cpu_probe.py:283–345` | There is a concrete parser/measurement seam to extend; no existing versioned v0.28 adapter was assumed. |

The report does not treat a 64-GPU job's auxiliary validation failure as permission to cancel it. It also does not treat resource pre-authorization as approval of D-* decisions, shared-schema changes, external publication, or an unbounded code step. Oversized tickets are intentionally split when execution begins (`requirements.md:44`).

## 6. Ticket-by-ticket disposition

These are review dispositions, not changes to ticket status.

| Ticket | Current status | Review disposition and next required clarification |
| --- | --- | --- |
| 00 | ready-for-agent | Audit scope is concrete. Correct operator routing (E1); record baseline/source-map clarifications C1–C3 and inspectable test evidence. The review did not execute this ticket. |
| 01 | blocked | Qualification ladder, full structure, dummy initialization, and text-only checks are clear. Align G0-C/D observation prerequisites and wave placement with S3. |
| 02 | blocked | Strong migration/negative-test scope. Connect validated outputs to formal consumers (S2), preserving decision-dependent normalization boundaries. |
| 03 | blocked | Native layer/cache/routing inventories and design-card ownership are clear. Normalize milestone names (E4); verify attention versus FFN ownership (C1). |
| 04 | blocked | Full family chain and trained/structural checks are appropriate. Completion is not equivalent to the entire GLM PD path (S1). |
| 05 | blocked | Raw profiling versus Frontier integration is correctly distinguished in principle. Replace the insufficient GLM prerequisite (S1). |
| 06 | blocked | Raw communication measurement before a decided narrow model is appropriate. Define shared/per-model closure when repairing S1. |
| 07 | blocked | Decision and connector-schema approvals, conservation, and legacy cost isolation are appropriately explicit. Preserve actual baseline behavior (C2). |
| 08 | blocked | Active-diff analysis before approved integration is sound. Avoid deleting placement guards without the approved semantics. |
| 09 | blocked | Bundle, cost replay, and coverage are useful outputs. Define per-model aggregate completion and validated-tooling dependencies (S2/E4). |
| 10 | blocked | Matrix/holdout and no-future-input contracts are strong. Add formal tooling gates and explicit request E2E/request-throughput checks (S2/S4); encode GLM-first precedence. |
| 11 | blocked | Independent H200 evidence and final delivery ownership are sound. Add objective H200 scope-freeze/closure (E2). |
| 12 | blocked | D-PD correctly remains a user decision after native/cache/communication evidence. No decision is made by this review. |
| 13 | blocked | D-ASYNC correctly requires distinguishing experiments; keep insufficient-causal-evidence outcomes open. Coordinate normalization dependencies with S2. |
| 14 | blocked | Native routing, calibrated scope, and predictive leakage constraints are clear. No routing representation is selected by this review. |
| 15 | blocked | Native placement evidence and retained guard are correct. Resolve G0-D/DP4 ordering before final decision (S3). |

All tickets also have the low-impact history-table issue E5.

## 7. Recommended remediation order and acceptance of the review

1. **Repair sequencing and measurement dependencies:** S1, S2, S3, and E4 together. Define per-model milestones, validate reference names, and check the resulting graph for cycles. Confirm the interpretation of GLM full-path completion against Q2 rather than silently weakening it.
2. **Close acceptance contracts:** S4 and E2. Make metric and H200 scope freezes explicit tasks with objective closure. Preserve the required same-topology holdout independently of optional topology B.
3. **Repair operational and evidence records:** E1 and E3. Route remote checks through the operator; recover or accurately qualify historical test evidence.
4. **Update source-map precision and document history:** C1–C3 and E5. Keep immutable snapshots unchanged and record corrections in maintained task documents.
5. **Re-review the revised plan before dependent execution.** Check all 16 blocker lists against the graph/waves and the approved decisions. Later implementation acceptance still requires its own native qualification, numerical gates, and regression evidence.

The report itself is complete. Remediation is **open**, and no finding is marked fixed by writing this report. The identified sequencing/acceptance issues do not invalidate independent local source-audit work, but do prevent claiming that the full execution plan is closed.

## 8. Verification and limits

Local Git identity and the source snapshot hashes were checked in this review. All three hashes match `requirements.md:32–34`:

| Snapshot | SHA-256 |
| --- | --- |
| V2 spec | `9f65592c6718b849603ba080f70ad2cb4005b3c514e8f4fce684133d092d5c4f` |
| Campaign YAML | `f109a0e3990cf7ba12614cca13cddd9e736df1e4a32c06e7980ff3d315e8b447` |
| Revision notes | `40bc491ac2ae48bd9dfc6df3e08e0a6fe72f8705e4cf3c26cb9bedf82ca5f26d` |

Exact commands, expected/actual results, and evidence limitations are recorded in [the review verification report](test_report_2026-10-15_spec_ticket_review.md). This review did not rerun pytest, qualify vLLM v0.28, inspect GPU/platform/private evidence, download model metadata, or validate upstream kernel/model claims. H800 fit, RDMA support, real initialization, cache descriptors, and routing remain future native-evidence questions as the plan states. Historical test results are reported as historical claims, not new observations.

**Axis summary:** Spec — 4 findings (2 P1, 2 P2), highest severity is the GLM-first/tooling dependency gap. Standards/readiness — 5 additional findings (3 P2, 2 P3), highest severity concerns operator routing, H200 closure, and historical evidence. Shared dependency observations are counted only once; C1–C3 are separate source-map clarifications.

## 9. Disposition and remediation (2026-10-15)

The user asked for each finding to be re-examined and every adopted finding to be fixed. S1 needed a user decision on what "complete GLM path" means and how the decisions are staged; the user chose Q12 and Q13, recorded in `requirements.md`. No ticket status changed: ticket 00 stays `ready-for-agent` and tickets 01–15 stay `blocked`. No code, GPU, or source snapshot was touched.

### 9.1 Findings

| ID | Disposition | Remediation | Verification |
| --- | --- | --- | --- |
| S1 | Adopted, with the user's Q12 and Q13 | New milestone `09.full_frontier_path[GLM]`: 04 done, the GLM closures of 06/07/08, `09.bundle[GLM]`, and one `python -m frontier.main` PD run of GLM-5.3 on topology A with trained predictors. Every DeepSeek-V4-Flash simulator milestone (05 integration, 06/07/08/09 DSV4 closures) waits for it; DSV4 native qualification, inventories, raw profiling and measurements, and decision evidence do not. D-PD and D-ROUTING are staged `12[GLM]`/`12[DSV4]` and `14[GLM]`/`14[DSV4]`; D-ASYNC and D-DP-PD are decided once on GLM-5.3 evidence, with `13.dsv4_agreement` and `15.dsv4_agreement`. GLM representative cells finish with recorded verdicts before DSV4 ones start. Files: `requirements.md` (Q12, Q13), `plan.md`, `spec.md` (Overview, deltas 10–11), tickets 03–15. | Graph check: all DSV4 simulator milestones descend from `09.full_frontier_path[GLM]`; DSV4 native milestones do not; no GLM milestone waits for a DSV4 stage. |
| S2 | Adopted | Ticket 02 gains `probe_validation[m]` (per model, GLM first) with a Consumers section. `09.cost_replay[m]`, `10.thresholds_frozen`, `10.representative_cells[m]`, and `02.metric_normalization` wait for it. Earlier traces are labelled diagnostic and unverified; decisions may cite them with that label, formal replay and formal runs never use them. The pre-launch check does not change the soft-gate rule for runs of 32 or more GPUs. Files: tickets 02, 09, 10; `plan.md`; `spec.md` (qualification section). | Graph check S2 rows. |
| S3 | Adopted | Ticket 01 states that the ladder never waits for a simulator change or a decision and that `G0-D[GLM]` runs as soon as `G0-C[GLM]` closes. Ticket 15 splits the two-lane observation (may start after G0-C) from the card and decision (after `G0-D[GLM]`). Wave 5 runs `G0-D[GLM]` before the 15 card. Files: tickets 01, 15; `plan.md`; `spec.md`. | Graph check: 15 descends from `01.G0-D[GLM]`; no ladder milestone descends from a decision. |
| S4 | Adopted | Request E2E and request throughput are checked separately per cell under the existing gate contract (`tests/comparison/e2e_gate/README.md`: `request_e2e_time_ms`, `request_throughput_rps`, formal request set and window, per-metric relative-error limit) or an explicit new-case contract frozen in `10.thresholds_frozen`. Failed and unfinished requests are counted. No new numeric threshold was introduced. Files: `spec.md` acceptance list, ticket 10 `thresholds_frozen`, ticket 02 normalizer, `plan.md` verification strategy. | Text inspection. |
| E1 | Adopted | Ticket 00 separates local manifest and source inspection from remote availability and hash checks, which go through the `stepmind` tool to `stepmind-helper`; only filtered receipts and cloud paths are recorded locally. | Text inspection. |
| E2 | Adopted | Ticket 11 gains `h200_scope_freeze`, decided by the user before the first H200 run: models, cells, repetitions, criteria, optional expansion, and blocked or deferred outcomes. H800 delivery does not wait for H200. Files: ticket 11, `spec.md` delta 12, `plan.md`. | Text inspection. |
| E3 | Partly adopted | No plan-phase output survived (`/data/ycfeng/tmp/opencode` searched), so the old comparison is marked not independently verified in `progress.md`. A fresh comparison replaces it: [test_report_2026-10-15_review_remediation.md](test_report_2026-10-15_review_remediation.md) §1. | Worktree `01fb2ce` and baseline `bfa82de`: 5 failed / 48 passed each; failure lists identical node by node; all five failures come from `README.md` strings missing at the baseline. |
| E4 | Adopted | One naming form, `<ticket>.<milestone>[<model>]`, in the tickets and the graph; aggregates (`06[m]`, `07[m]`, `08[GLM]`, `08[DSV4]`, `09.bundle[m]`, all decisions closed) are defined as conjunctions in `plan.md`; ticket 09's blockers use the milestone names. | Graph check: no unresolved blocker, no cycle, 64 nodes. |
| E5 | Adopted | All 16 tickets carry a Modification History table. `source_spec_v2/` is unchanged. | `rg -c "^## Modification History" tickets/*.md`: 16. |
| C1 | Adopted | Ticket 00's source map and ticket 03's design cards start from `resolve_attention_topology` in `frontier/attention/model_binding.py`, which resolves per-layer attention identity; `_assemble_stage` and the sklearn attention pricing are recorded as consumers, and `layer_contract_resolution.py` as the FFN-contract owner (precision fix R4 in §9.4). Files: tickets 00, 03; `spec.md` (implementation decision, delta 9). | Code read at `bfa82de`. |
| C2 | Adopted | `spec.md` and ticket 07 record that `enable_latency_hiding` is a future-extension flag with no production overlap path; the behaviour is kept, no P2P overlap is added, and NIXL rejects an enabled flag. | Code read at `bfa82de`. |
| C3 | Adopted | Exact roots of the four historical task records and the class name `BaseExecutionTimePredictorConfig.enable_dummy_mode`. Files: ticket 00; `spec.md` (delta 2, code facts). | Text inspection. |

### 9.2 Additional corrections found during remediation

| ID | Finding | Correction |
| --- | --- | --- |
| A1 | `frontier/attention/model_binding.py` detects DSA only from DeepSeek-V3.2 model types and `dsa_*` keys, so GLM-5.3's `index_topk_freq`, `index_topk_pattern`, and `index_skip_topk_offset` would fall through to MLA. | Ticket 04 owns the binding and a fall-through negative test; `spec.md` records the fact. |
| A2 | `VllmLoadBalancingClusterScheduler` also rejects more than one Replica and non-V1 replica schedulers, and needs `attn_dp=1` for dense models; only the PDD check was listed. | Tickets 08 and 15 and `spec.md` list the full guard; ticket 15's card must state the change for each check, including topology B's three DECODE Replicas. |
| A3 | `kv_cache_transfer_end_event.py` delivers KV and releases the source in one event. | Ticket 12 lists it as a fact the D-PD experiments test. |
| A4 | The plan graph and the tickets disagreed in seven places: the `08.approved_workflow_integration[DSV4]` edge named `01.G0-C[DSV4]` instead of `13.dsv4_agreement`; `10.representative_cells[DSV4]` named agreements of 13 and 15 instead of `15.dsv4_agreement`; `10.rca_and_gate` and the `06.narrow_cost_model[GLM]` → `[DSV4]` edge were missing; `09.predictor_bundle[m]` lacked `14[m]`; `10.thresholds_frozen` and `02.metric_normalization` lacked `02.probe_validation[GLM]`. | `plan.md` now follows the tickets; ticket 02's `metric_normalization` blocker names `probe_validation[GLM]`. |

### 9.3 Remaining

- The review's step 5, an independent re-review, is done; see §9.4.
- The five `README.md`-based documentation test failures exist at the baseline. This task does not edit `README.md`; ticket 00 records them.
- Every native-evidence question the original review left to execution (H800 fit, RDMA, real initialization, cache descriptors, routing) remains open, as planned.

### 9.4 Independent re-review of the remediation

- **Reviewer:** OpenCode `reviewer` subagent, session `ses_ec18d30bfffetVE4eIdNuTBL0H`, read-only.
- **Inspected:** `requirements.md`, `plan.md`, `spec.md`, `progress.md`, this file, all 16 tickets, the remediation test report, the campaign YAML ticket block, the E3 failure lists, and the code anchors at `bfa82de`.
- **Verdict:** accepted with fixes. All 12 findings (S1–S4, E1–E5, C1–C3) closed; consistent with Q12 and Q13; no P1 or P2 defect; ticket statuses unchanged. A1–A4 match the code.
- **Precision fixes (all P3), applied the same day:**

| ID | Finding | Fix |
| --- | --- | --- |
| R1 | Ticket 09 said every blocker of `predictor_bundle[DSV4]` already waits for `full_frontier_path[GLM]`; `14[DSV4]` does not. | Ticket 09 now names the 05–08 closures as gated and `14[DSV4]` as a decision that may close earlier. |
| R2 | Tickets 12 and 14 listed `12[GLM]` / `14[GLM]` as blockers of the whole DSV4 stage while saying evidence may start early; the waves started `14[DSV4]` evidence before the `14[GLM]` decision. | Tickets 12 and 14 and `plan.md` split each DSV4 stage into evidence (DSV4 native milestones only) and the amendment card and decision (plus the GLM decision); wave 5 places the amendment cards after the matching GLM decision. |
| R3 | `thresholds_frozen` compared the light-mode limit only with the GLM-5.3 measurement. | Ticket 10 `representative_cells` compares each model's `probe_validation[m]` measurement with the frozen limit before its formal runs; an unqualified light mode means clean mode only. |
| R4 | Per-layer attention identity is resolved by `resolve_attention_topology` (`frontier/attention/model_binding.py:267`, called from `frontier/config/model_config.py:523`) and read through `bind_layer_attention` (`:255`); `_assemble_stage` and `sklearn_execution_time_predictor.py:7510` consume it. The remediation named the consumer as the owner. | Tickets 00 and 03, `spec.md`, and the C1 row above name `model_binding.py` as the owner and the others as consumers. |
| R5 | The remediation test report counted four plan/ticket mismatches; A4 lists seven. | Corrected. |

The short-form name `04.integration` in `plan.md` was also corrected to `04.family_predictor_integration`.

- **Verification after the fixes:** the graph check, rerun with the split DSV4 stages, reports 66 nodes, no unresolved blocker, no cycle, and PASS for every Q12, Q13, S2, S3, and R2 check (`test_report_2026-10-15_review_remediation.md` §2).
- **Not checked by the reviewer:** running the graph script (it compared the encoding by hand), the `source_spec_v2/` hashes, a pytest rerun, and the contents of the historical task directories.

## 10. Ticket 00 phase review (2026-10-15)

- **Component / phase:** ticket 00 (baseline audit and reuse inventory), acceptance review.
- **Reviewer:** read-only `reviewer` subagent, session `ses_ec037d65affeMd3oQ5dPDMukjv`. A first dispatch was cancelled before it returned anything. The reviewer session had read and search tools only and ran no shell command.
- **Inspected artifacts:** `tickets/00-baseline-audit.md`, `progress.md`, spec v2 §8.1–8.3 and T00, `source_map.md`, `reuse_inventory.md`, `baseline_audit.md`, the checkpoint 1–3 test reports, `test_report_2026-10-15_bundle_provenance_closure.md`, `cloud_archive_check/`, `bundle_provenance_local/`, and `tests/comparison/e2e_gate/artifacts.json`.
- **Verdict:** accept with fixes. The three acceptance items PASS; no P1.

| ID | Finding | Disposition and verification |
| --- | --- | --- |
| P2-1 | The 77-case fidelity baseline that later tickets compare against existed only under `/data/ycfeng/tmp`. | Adopted. Index copied to `baseline_record/`; full capture copied to `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t00/refactor_fidelity_baseline_bfa82de/` by job `exp-1015-215759-514413` (Succeeded, terminal verified): 530 files, 702,886,537 bytes, 451/451 artifact pins, listing digest equal to the local one. `baseline_audit.md` "Durable copy of the 77-case capture". |
| P3-1 | Runtime fields were read for one run only; `vllm_import` holds date-like values in 20 PD runs. | Adopted. All 60 read; the 20 local `vllm_import.txt` files show `VLLM_IMPORT 0.10.2` on line 2 after a vLLM INFO line, so the field is a parsing artifact (inference) and all 60 runs imported 0.10.2. `reuse_inventory.md` runtime paragraph. |
| P3-2 | Memory-trace location cited as `baseline_audit.md`. | Adopted. Points to `source_map.md` §6; `baseline_audit.md` has a pointer section. |
| P3-3 | Stale "`git grep` truncated" note on `VLLM_DP_SYNC_INTERVAL`. | Adopted. Replaced by the complete five-hit list from `git grep -n VLLM_DP_SYNC_INTERVAL bfa82de`. |
| P3-4 | Stale "not a verified cause" for the five mismatches. | Adopted. Cross-reference to "Provenance closure" added. |
| P3-5 | Missing spec §8.1 evidence-level label for the missing predictor caches. | Adopted. C1, C5, C6, C2' trained Frontier replay marked insufficient evidence, not reproduced. |
| P3-6 | Inventory title said "checkpoint 1". | Adopted. |
| P3-7 | Predictor-cache search lacked an exact command. | Adopted. Commands, exit codes, and the 897 `Permission denied` cause of exit 1 recorded in the closure report. |
| P3-8 | Stray `cloud_archive_check/__pycache__/*.pyc`. | Noted in the closure report; not deleted (removal needs approval). |

Primary spot checks run in place of the reviewer's shell checks:

| Check | Command (summary) | Result |
| --- | --- | --- |
| (a) archive comparison | recount from `worker_output/archive_hashes.json` against local evidence files | 60 present, 844 files, 844 equal; local-only 60 x {`run_manifest.json`, `host_state.log`, `host_state.jsonl`, `launcher_receipt.json`}; PASS |
| (b) mismatch cause | remove the added line from `e2e_c4_moe_pdd/inputs/fork_changed_files.txt` in memory and hash | `7a4996bb13a0962fa7dded10e4167b7264dd2ea7070cda186b6b0414d3948653`; PASS |
| (c) source pins | `git show <commit>:<path>` hash for 5 occurrences across harness, producer, and config pins | 5 of 5 match; PASS |
| (d) baseline pytest totals | tail of `/data/ycfeng/tmp/glm53_dsv4/t00/pytest/unit_integration.log` | `85 failed, 4181 passed, 73 skipped, 15 errors in 376.60s`; PASS |
| (e) tracked change | `git diff --stat bfa82de..HEAD`; `git status --short` | only `CONTEXT.md` and the ADR; clean; PASS |

Scratch output: `/data/ycfeng/tmp/opencode/t00_phase_review/`.

## 11. Documentation-only resume audit — 2026-10-18

### 11.1 Outcome, scope and evidence boundaries

**The task has drifted from the required broad-to-detailed sequence.** Environment/native diagnostics and observer work made real progress, but complete model representation and structural serving are still missing. The plan had made those early capabilities wait for full native qualification, routing/async/PD decisions and sometimes measured costs. This is a dependency problem, not proof that the DES architecture needs replacement. Corrected documents move existing-module model representation first and retain the independent native and numerical acceptance gates.

Review owner: primary Codex; no subagents or background work started. Reviewed artifacts: current requirements/spec/plan/design/progress and all 16 tickets; original v2 spec/plan/campaign; source/reuse/baseline inventory; retained GLM/DeepSeek startup, observer, selected-backend, cache, communication, workflow and A2 reports; Frontier and native source/commits; Step's shared integration plan, model spec/plan, relevant structural/delivery tickets and recorded progress; official model recipes. Targeted source inspection establishes the cited owner/guard facts; this is not an exhaustive correctness review of every existing simulator path.

- Frontier pre-review HEAD: `9ccd30cfa23c4cd8612c8cb1c6cc8e4afaa6124f`, branch `feat/glm53-dsv4-vllm028`, baseline `bfa82de80413b5f4b38e4990a80b5108f9521f50`. Relative to baseline, committed changes cover glossary/ADR, three diagnostic/bootstrap modules and their tests, and raw gate admission. **No change under `frontier/` or model assets under `data/`.** Uncommitted 65 A2 test lines and `.pi/` are inherited and preserved.
- Native observer branch: `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/frontier-observer-v028`, HEAD `6c523e85331b6fcaa1016bc97da8fdb3c1b3b9d2`, base `2cf0a6915ce544dc493a0990f2ea38d81601128a`. Local unit/AST evidence is narrower than deployed schema/causal/clock/perturbation qualification.
- Reference image remains `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`. Preserve H800 CUDA compatibility/approved dynamic UCX policy, original model metadata and no-checkpoint workflow. This review does not change any runtime lock.
- Step's existing worktree is a separate topic based on `01fb2ce`, not yet a common final integration HEAD. Its native/image/acceptance evidence remains separate. Shared contract source is `/data/ycfeng/Frontier/.local-draft/step5-preview/03_Shared_Integration.md`; Step records are under `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/`.
- Current-phase verification only reads Git/source/documents and retained artifacts, checks hashes/counts/links/dependencies and records documentation edits. No runtime tests, benchmarks, native inference, GPU inspection/control or new ground-truth collection was performed. All quoted historical test counts below are retained results, not reruns.

### 11.2 Ticket and area assessment

Statuses apply to the intended outcome, with narrower completed slices named explicitly. No whole-ticket DONE is inferred from preparation. Paths are relative to this task directory unless prefixed `repo:` or absolute.

| Ticket / area | Intended outcome | Actual status | Evidence | Deviation / risk | Required action |
| --- | --- | --- | --- | --- | --- |
| 00 baseline | Verified owners/reuse, old-feature baseline | **DONE**, baseline audit only | `source_map.md`, `reuse_inventory.md`, `baseline_record/`; retained 85 failed / 4,181 passed / 73 skipped / 15 errors | A captured baseline is not a green baseline; repeated exact old artifacts do not prove new support | Reuse accepted snapshot; distinguish historical failures from changes |
| 01 runtime/native | Full dummy-weight ladder through 64-H800 P32+D32 | **PARTIAL**; DeepSeek native **BLOCKED** | `runtime_lock/runtime_lock.json`; H800_03 `exp-1017-023422-134135`; accepted reduced GLM05 `exp-1017-114557-789609`; full GLM startup inventories and `g0b_startup_memory_01/`; DeepSeek `exp-1017-033620-885264` | Environment/reduced/full-startup scopes are different. Full routing, G0-C/D unqualified; missing mHC broadcast blocks original DeepSeek dummy workflow | Retain lock; use completed evidence; solve a focused supported native blocker without stalling static Frontier work |
| 02 observer/gate | Native causal traces, adapters, CPU costs, measured light-mode qualification | **PARTIAL**; schema/probes open | Native observer commits through `6c523e853`; A1 `9ccd30cf`; `ticket02_host_trace_gaps.md`, `ticket02_raw_admission_a2_checks.md`, A2 checkpoint | Raw envelope admission is incomplete-only; A2 remains unapplied, 40 PASS/1 FAIL historical. Null standalone dummy rows are not proof of serving-parent loss; no EngineCore JSONL retained | Target only upcoming native evidence/driver gaps; defer A2 as auxiliary until needed, preserve cap/contract and RED |
| 03 contracts/inventories | Complete per-layer/operator/cache/routing contract | **PARTIAL** static preparation; native inventories open | `ticket03_glm_layers.csv` 78 rows, 21 full/57 shared, 3 dense/75 MoE; `ticket03_cache_preparation.md` 99 components | All 78 execution fields remain unknown; allocations are not valid transfer bytes. Old full-G0-B prerequisite blocked useful representation | Close separate source `static_contract` and `operator_inventory`; retain native verification milestones |
| 04 GLM family | Represent and execute full GLM chain, then measured predictor | **NOT STARTED** implementation; preparation complete | `ticket04_dsa_preparation.md`, `ticket04_selected_backend.md`; repo `attention/model_binding.py`, `families.py`, model-config precision checks | Real GLM index fields fall through; DSA frozen. Existing importer does not cover selected sparse backend; block-FP8 mismatch unresolved | First register configuration-derived bindings/operator/memory contributions and structural full-model run; profile only represented paths |
| 05 DeepSeek family | Original 43-layer compression/mHC/routing path, then measured predictor | **NOT STARTED** Frontier; native **BLOCKED** | Pinned metadata and `vllm/models/deepseek_v4/nvidia/model.py` initialization/finalizer; missing broadcast MRE/native failure | Native blocker was incorrectly a barrier to all static model work; variant substitution would change scope | Parallel static/operator/structural family work using shared contracts; keep native/calibrated claims blocked |
| 06 communication | Native TP/EP/NIXL curves and measured narrow consumer | **PARTIAL** source audit; target measurements **NOT STARTED** | `ticket06_curve_preparation.md`, retained three existing P2P CPU checks | Existing/synthetic P2P tests are not target EP32/NIXL or contention evidence; risk measuring before workload exists | Prepare ownership/grids early; measure represented shapes; split lifecycle and contention dependencies |
| 07 NIXL PD | Correct payload, admission/readiness/source release and cost | **PARTIAL** baseline audit; implementation **NOT STARTED** | `ticket07_lifetime_baseline.md`; retained 33 PASS/one deliberate PD-AF deselection; repo transfer-end/pending/allocator owners | Current transfer-end delivers D and frees P; D allocation occurs later. Reserve-before-READ was written as if already established | Obtain native lifecycle evidence; structural extension through existing owners first, numerical integration after measurement; core refactor only if necessary |
| 08 workflow/placement | Relevant v0.28 async/graph/chunk/DP/PD behavior | **PARTIAL** baseline; target delta/integration **NOT STARTED** | `ticket08_workflow_baseline.md`; retained 81 unit PASS, placement 15 PASS/1 FAIL; current full constructor guards | Historical lane2 late-join failure unresolved; source comparison unnecessarily waited for unrelated inventories | Early pinned-source delta table; existing-owner structural behavior after required dispositions; retain guards |
| 09 bundle/E2E | Structural path, isolated trained bundle, decomposition and full trained GLM PD | **PARTIAL** identity audit; target path/bundle **NOT STARTED** | `ticket09_identity_preparation.md`, `ticket09_identity_source_check.md`, `ticket09_cache_identity_decision.md`; six synthetic fixtures | Shared prediction manager already supplies canonical chain. No trained target bundle; old definition hid structural progress until calibration | Establish structural E2E before cost tuning; reuse identity chain, then compare/calibrate and close trained GLM full path |
| 10 H800 acceptance | 36 cells, repeats, frozen metrics, independent holdout | **NOT STARTED** | `tickets/10-h800-predictive-e2e.md`; no accepted model G0-D or target bundle | Fixtures/imports/startup/old exact comparisons cannot close any predictive cell | Preserve matrix and per-metric/failure accounting; execute after represented, native-qualified and measured paths |
| 11 delivery/H200 | Regression, common branch, review package, later H200 | **BLOCKED**; H200 deferred | Current worktree identities; Step shared plan and Step ticket29; existing H200 failure record | No three-model common HEAD or final regression; H800 timing must not transfer to H200 | Integrate/reconcile Step at one HEAD after its gates; report H800 independently; resume H200 only when directed |
| 12 D-PD | Evidence-backed component/lifecycle and contention disposition | **BLOCKED** on native evidence, no selected core design | Cache preparation, PDD baseline, original experiment requirements | Old decision waited for all communication curves before first structural handoff; assumed reservation semantics | Split `lifecycle_disposition` from `contention_conclusion`; approve only a necessary core/material semantic change |
| 13 D-ASYNC | Existing queue/stage timing sufficiency, cross-model agreement | **PARTIAL** source preparation; disposition **BLOCKED** | `ticket13_async_evidence_boundaries.md`, 12 source pins | Host method return/issuance does not prove device completion/overlap; no distinguishing experiment | Use causal G0-C with required schema; extend existing boundary if sufficient, propose core refactor only with counterexample |
| 14 D-ROUTING | Predictive routing expression supported by native occupancy | **PARTIAL** expression audit; disposition **BLOCKED** | `ticket14_routing_preparation.md`, retained 33 materializer checks | Marginal assignment/ratio is not token routing/correlation/padding; strict routing content remains OFF | Keep static representation independent; collect approved-safe relevant occupancy later; no uniform balancing or per-cell future inputs |
| 15 D-DP-PD | Native P/D entry/lane/load policy and agreement | **BLOCKED** on G0-C/D | Existing source guard and ticket15 evidence checklist | No DP4 observation, native placement implementation or chosen alternative | Separate two-lane observation and DP4 disposition; do not remove guards or pass round-robin off as native |
| Step-5-Preview | Third-model consumer on shared Frontier with independent reference | **PARTIAL** early model/native work; final integration **NOT STARTED** | Shared integration C01–C06; Step spec/plan/progress, tickets15/29; recorded rank-local evidence | Step G1/rank-local results are not complete G2/64-H800 or cleaned timing evidence. Candidate source SHA is not an image-build SHA | Supply C02 contract fixtures now; reconcile its production barrier and native identity before integration; no claim of current live job state |
| Shared C01–C06 | One identity, layer, state, profiling, workflow and gate implementation | **NEEDS REVIEW** at next edit; no duplicate production infrastructure found in current core diff | Existing Frontier owners, shared integration plan, separate topic HEADs | The main issue is planned over-coupling and repeated preparations, not an already implemented generic runtime redesign | Assign one owner per shared file set; model work owns disjoint leaves/tests; final same-HEAD three-model regression |

### 11.3 Main-path findings and recovery

**P1 — Representation blocked behind measurement/decisions (03–05, 07–09).** The old graph allowed only report preparation while complete G0-B and D-* evidence remained open. Real source gaps are already identifiable: `attention/model_binding.py` recognizes DeepSeek-V3.2/`dsa_*` markers rather than GLM index fields, the general topology resolver repeats one homogeneous binding except the existing Qwen/GDN special case, `DSAAttentionFamily` is frozen, and block-FP8 loader metadata is not yet expressed correctly. None requires routing histograms to begin representation. Corrected 03 static/operator and 04/05 model/structural milestones are the next main work; native inventories stay independently gated.

**P1 — A local diagnostic/observer loop displaced model delivery (01/02, related 03/04/06/07/09 preparations).** Full-model runs01–05, CUTLASS call/layout/context observations, the controlled fixture and startup counters each established bounded facts, but the production model path remained unchanged. Those experiments were not intrinsically invalid; repeating them after their scoped question closed would have diminishing main-path value. The latest attempt expanded into raw admission A2 (448 source changes plus 65 test lines versus the 330 incremental cap; separate 308-line scratch incomplete), while no target model profile/binding existed. **Continue** completed evidence and the minimum upcoming reference instrumentation; **defer** A2 and optional full-model W31/cubin archaeology; **stop repeating** the completed primitive/startup or closed capture-state source investigation without a new distinguishing main-path question. The actual native original-DeepSeek failure still deserves focused blocker resolution.

**P1 — Structural serving was conflated with calibrated prediction (04/07/08/09).** `09.full_frontier_path[GLM]` deliberately uses trained timing, replay and coverage. It is a valid later GLM-first gate, but a poor first demonstration of model support. New structural milestones use the existing explicit dummy timing mode and reconcile operations, memory, communication and completion before training; unsupported paths still fail. Full online/PD/graph/chunk claims remain dependent on their actual implemented semantics. Numeric acceptance is unchanged.

**P2 — PD lifecycle and NIC contention were over-coupled (06/07/12).** Raw curves are needed for a contention conclusion and numerical transfer cost, not to determine every storage reservation/readiness/free event. Current `kv_cache_transfer_end_event.py` joins D arrival and P release; later scheduler allocation makes a reserve-before-READ assumption unsafe. Ticket12 now has a native-evidence lifecycle disposition and a later measured contention conclusion. Ticket07 can establish a structural handoff once the former is known and any necessary core change approved; it cannot claim measured correctness prematurely.

**P2 — Observer evidence and regression were easy to overstate (00/01/02/08/09).** Direct reads this review reproduced 16 hashed worker trace payloads, 192 startup rows, 1,152 footer missing-parent counters, 1,552 explicit null-parent rows, including 752 dummy dispatches and 400 `execute_dummy_batch` begin/end pairs. These standalone calls lack a native SchedulerOutput; no serving causal loss follows from those nulls. Missing EngineCore JSONL establishes missing retained coverage only; collection/persistence cause is UNKNOWN. The retained symptom checker asserts the known incomplete symptom then intentionally returns FAIL, so it is not a green-capable cause regression. A1 always returns incomplete/qualification false, and even a future raw-host PASS remains separate from schema/probe/native qualification. Repeated 77 old-case equalities with unchanged core source are regression evidence only; ticket00's baseline has known failures, including the workflow late-join case. Corrected docs preserve all failures instead of accepting them as new capability.

**P2 — Shared integration and approval boundaries needed operational detail (all, especially 03–09/11–15).** Shared source requirements already define C01–C06; there is no justification for a second layer registry, model cache, CPU publisher, sparse SDK, NIXL manager, gate, or EngineCore. No such duplicate core implementation was found in this branch's diff; this is a forward constraint. Model semantics differ: sparse MLA, compressed attention and Step sparse GQA do not share one `sparse=True` timing table. Corrected owner assignments and common-HEAD gate make reuse concrete. The latest user request authorizes routine extensions; tickets12–15 no longer automatically demand permission for every additive field, entry or experiment. An unresolved material semantic choice/core refactor still needs a concrete proposal and explicit approval.

**Continue / defer / simplify / revert:** continue representation and real native blockers; retain useful observer and controlled-numerics evidence; simplify dependencies and reuse existing producers/identity/events; defer auxiliary A2, isolated tuning and H200. **No implementation revert is justified by this documentation audit.** Current accepted observer/diagnostic commits are not shown incorrect by these findings. Preserve the unapplied candidate and RED test; do not delete/revert unrelated dirty work. Any subsequent code removal requires its own demonstrated cause and applicable authorization.

**Shortest safe route:** source contracts/operator ownership -> full GLM (and parallel DeepSeek family) representation -> full structural execution -> relevant native qualification/instrumentation and native-evidenced serving semantics -> structural online/PD/graph/chunk E2E -> measured workflow/operator comparison -> targeted correction/trained bundles -> frozen 64-H800 matrix/holdout -> same-HEAD three-model regression. Native readiness preparation runs alongside representation; a native blocker never becomes a fabricated PASS or a barrier to unrelated source work.

### 11.4 Official workflows and reference correctness

Official pages were read during this review, not executed:

| Primary source | Current documented detail | Consequence for this task |
| --- | --- | --- |
| [GLM-5.3 recipe](https://recipes.vllm.ai/zai-org/GLM-5.3) | Page updated 2026-10-15; badge says 0.29.0+, while prerequisite/install text still includes >=0.28 / ==0.28.0; Transformers >=5.15.0 is recommended. Recipe examples include MTP and newer GPU examples. | The page is internally inconsistent on version labeling; actual locked-source/run evidence remains authoritative. Do not auto-upgrade or add MTP. Official examples do not establish this task's H80064/dummy-weight qualification. |
| [DeepSeek-V4-Flash recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) | Updated 2026-09-29; newer default is `-0731`, with original Preview still available. Includes v0.28 NIXL/FP8-cache/block256 H200 PD example. | Preserve the selected original variant. This is useful native workflow input, not evidence that original dummy loading initializes mHC auxiliaries or runs H80064. |
| [Step-5-Preview recipe](https://recipes.vllm.ai/stepfun-ai/Step-5-Preview) | Uses the separate Step image and Hopper workflow, with backend/hardware-specific scope. | Keep independent runtime/source/instrumentation identity and measured profiles. Recipe support is not this task's native qualification. |

The local calibration skill still describes vLLM-BS v0.10.2, and its host binding describes H200/step_main plus a helper route from an earlier task. Those are not the target-runtime/H800/current-Codex operation contract. Ticket11 must update the verified target workflow after evidence exists, reusing the existing harness; this review neither applies that calibration workflow nor changes skill files. The earlier explicit-invocation preference is a retained user preference, not proof of a current `agents/openai.yaml` setting (that file is absent locally).

Pinned local vLLM source is used for exact model/loader/backend semantics; mutable recipe prose cannot supersede the image/source/model lock. The attempted version-specific NIXL documentation/raw-source URL fetches failed in the browsing tool; no missing-page content is assumed. Local source and the working official model recipes cover the facts used here.

The DeepSeek failure is specifically a dummy initialization path gap: `hc_attn_fn_broadcast` is initialized by its `load_weights` finalizer, while the dummy path initializes registered tensors without that finalizer; the native failure and source MRE agree. MegaMoE's alternative finalization is SM100-specific and does not establish an H800 remedy. A model/version/backend change or native functional workaround is not silently authorized by an instrumentation request.

### 11.5 Unresolved material decisions

No option below is finalized. These are downstream, evidence-bounded questions, not reasons to stop early model work.

| Area / tickets | Exact limitation and evidence still needed | Smallest recommended direction / alternative / cost | Approval condition |
| --- | --- | --- | --- |
| C02 per-layer binding (03–05, Step15) | Homogeneous topology resolver and GDN-located unique `LayerAttentionSpec`; actual downstream capacity/variant consumers need a reachability map | Extend existing config normalization/bindings/variant entries first. If a type relocation or consumer contract refactor is required, propose one canonical change; copying per-model layer registries multiplies regression cost and is rejected | Routine entries authorized. A material core contract refactor requires exact caller diff and approval before edits |
| C03 cache/lifetime (03/07/12, Step state) | Aggregate arrival/free differs from potential native reservation/readiness/release; 99 GLM component specs do not establish validity/READ descriptors | Prefer component resident/transfer summaries and existing events/pending owner if conservation holds; otherwise propose the minimum additional capacity or lifetime boundary. Aggregate-only duration adjustment cannot repair admission errors | Native G0-C lifecycle evidence first; approval for a necessary material semantic/refactor choice; no new generic NIXL engine |
| C05 async/placement (08/13/15) | Existing in-flight timing and full MONOLITHIC/single-Replica guards; native CPU waits/device completion and DP4 load policy unobserved | Test existing queue/stage/load-report sufficiency. Extend only active differences; copying EngineCore/IPC or a universal resource DAG has high maintenance/regression cost without current evidence | Approve only an evidenced necessary core redesign or materially different benchmark routing contract |
| Original DeepSeek reference (01/05) | Pinned original dummy path lacks required mHC auxiliary state | First demonstrate an upstream-supported unchanged-runtime dummy workflow from authoritative code/evidence. If none exists, retain BLOCKED and present version/source/workflow alternatives and cross-model impact | No switch from original variant, functional patch or reference lock change is approved here |
| Step integration (11, Step15/29) | Topic HEAD, independently locked image/native path and separate production barrier | Reconcile existing C01–C06 on the integration branch with one owner per file set; retain independent measurements. Do not create another infrastructure branch implementation | Routine merge/integration work follows session scope; any material shared redesign or Step's still-explicit production decision requires applicable approval |

### 11.6 Corrections, ownership and next execution order

Substantive in-place edits: `requirements.md` records latest scope and superseded portions of Q2/Q12/approval rules; `spec.md` states the five-stage path and distinct structural/numeric support; `plan.md` replaces the dependency graph/waves; all existing tickets retain their IDs and original final goals but correct dependencies, status/acceptance wording and shared reuse. Ticket11 adds common-HEAD three-model identity/regression requirements. No threshold, native scope, model variant or historical result is relaxed. Old documents are retained in `documentation_review_20261018/before/`; existing frozen campaign/approval archives remain unchanged. `CONTEXT.md` is renamed to `GLOSSARY.md` with byte-identical contents per the supplied AGENTS instruction; no ADR or speculative new architecture is created.

Use the plan's eight-wave table and milestone DAG. Start **03 static/operator contracts -> 04 GLM representation/structural execution**. Concurrent leaves: **05 DeepSeek static/operator/structural family work**, **01 original native blocker review**, **02 minimal reference collection/schema/driver preparation**, and **08 pinned-source workflow diff**. These are next-session assignments only; none was dispatched here.

Single-owner boundaries: C02 model normalization/layer contracts/bindings; C03 allocator/component/transfer/pending/release; C04 operator/physical-scope profiling/training identity; C05 scheduler/in-flight/DP/P-D placement; C01 runtime/profile identity and canonical predictor cache; C06 native adapter/normalizer/gate and CPU producer. One native instrumentation owner edits hooks; model contributors own disjoint configuration/operator leaves and model-specific tests. C labels are existing shared ownership labels, not new tickets/services. The execution lead assigns actual people/agents before shared edits.

Native 01 G0-B/C/D and 02 validation remain required for measured consumers, with D-* evidence before affected structural serving changes. Raw profiles/communication curves follow represented workloads. Workflow and operator comparisons explain E2E errors before corrections. GLM's trained full path and representative verdicts remain first; DeepSeek raw preparation can proceed independently. Formal 10 still requires 36 cells with >=3 clean repetitions, frozen central/tail/request metrics, counted failures and independent holdout. Step and old-feature regression must close on the same reviewed HEAD. H200 remains deferred.

### 11.7 Resume handoff and current-phase closure

**Review deliverables A–E are complete as documentation; model support is not complete.** Start from the current `requirements.md`, `plan.md`, `progress.md` and this section, not an old A2-first checkpoint. Preserve the inherited RED test, untracked `.pi/`, all frozen source/approval/evidence archives, runtime/model locks, routing OFF and no-checkpoint policy. Do not restart completed native diagnostics simply to recover confidence in existing evidence. Record exact source/image/instrument/config/topology/artifact identities for future campaigns and keep 32+ GPU auxiliary validators non-blocking during healthy work.

The next execution session may use standing routine resource/instrumentation/bug-fix authority without repeated questions. Pause only a dependent material core refactor or material unresolved semantics/acceptance choice with a concrete evidence-backed proposal; continue independent work. This review starts no implementation, tests, profiling or campaign. Verification and limitations are in `test_report_2026-10-18_documentation_review.md`.
## 12. H200 mainline completion review — lead acceptance (2026-11-08)

### 12.1 What was reviewed and by whom

Assignment: `review_20261108_h200_mainline/assignment.md`. Executor: codex-implement request
`4c000c1e-7ba4-4ef2-a77f-e341a5c8b551`, model `gpt-6.1-sol-sys-paygo`, effort `ultra`, session
`01a1ba95-8a5d-7c21-b128-2363134ee32a`, zero watchdog retries. Report:
`review_20261108_h200_mainline/codex-implement-4c000c1e-7ba4-4ef2-a77f-e341a5c8b551/implementation.md`
(174 lines) with the long form in `review_20261108_h200_mainline/report.md`.

The question put to the executor was whether the H200 mainline task, meaning DSV4 simulation support plus
accuracy calibration, is actually complete, and whether errors or omitted critical executions exist. Writes
were confined to `review_20261108_h200_mainline/`; no platform, RJob, or GPU operation was permitted, and git
access was read-only. The executor's self-report was `implemented`; acceptance is recorded here.

The lead confirmed the write scope held: `git status --short --untracked-files=no` is empty on the executor's
own record, and no file outside the review directory was modified by it.

### 12.2 Accepted verdict

**The two H200 numerical cases are complete and correct. The H200 mainline as a whole is not complete, because
one explicit acceptance criterion — actual GPU-hour accounting — is still open.** The lead accepts this verdict.

Confirmed as passing, each re-checked by the lead against the artifacts:

- P2-32 formal PASS, worst per-repeat relative error `0.05607176023817675` at `repeat03 / ttft_ms` (5.61%).
- P2-64 formal PASS with the generation08 reference, worst per-repeat relative error
  `0.03162330523324778` at `repeat02 / ttft_ms` (3.16%).
- All four metrics within 10% in every repeat, in both the all-request and warm readings, which are identical
  for generation08.
- The P2-32 to P2-64 promotion receipt (sha256 `d505eece...`, issued 2026-11-02T07:38:49Z, limit 0.3, 24
  checked errors, max 0.0561) existed before the 64-GPU package was built and submitted. Its digest is
  embedded in the generation08 `config.json`, `case/inputs/native_config.json`, and run manifest.
- No incorrect headline metric, no missing admitted native repeat, no retrospectively supplied receipt, no
  forbidden whole-step input, and no arbitrary multiplier on the inspected formal paths.
- The executor reproduced every per-repeat error from the row CSVs with its own `audit_metrics.py`, whose
  pooled throughput divides total tokens by the sum of per-repeat spans (line 61), matching this task's
  convention rather than reusing `headline_recompute.py`.

Independent audit totals, all with zero failures: `metrics_audit` 157/157, `evidence_audit` 861/861,
`qualification_audit` 906/906, `runtime_audit` 97/97, `runtime_supplement` 107/107, `handoff_audit` 28/28.

### 12.3 Disposition of the four findings

| Finding | Classification | Lead disposition |
| --- | --- | --- |
| **F1** Actual GPU-hour accounting remains an acceptance blocker | Blocking completion defect; the limitation is honestly recorded but cannot be used to call the mainline finished | **Confirmed, remains open.** Ticket 19 line 203's own wording keeps the box open until all actual clocks are added. The 35-job ledger totals 158.271667 container GPUh and excludes generations 05–08 (18.52 + 8.57 + 41.81 + 23.31 = 92.21 GPUh). Generation01 has one of eight exact finish clocks (4.053333 GPUh known); generation02 has `container_clocks_complete: false`; generations 03 and 04 retain null exact totals. No metric depends on these clocks, because the client intervals use dispatch/first-token/last-token monotonic timestamps. Ticket 19 stays `PARTIAL` and the accounting box stays unchecked. No new GPU run is required to close it. |
| **F2** Three heartbeat verification claims are supported only by prose | Verification record gap, separate from any numerical defect | **Confirmed and escalated.** The executor could not locate the stdout and correctly declined to equate "not located" with "not executed". The lead found the source: that evidence exists only in the unmerged worktree `/data/ycfeng/Frontier/.worktrees/p2-64-startup-harness`. Re-running it there today gives `12 failed, 6 passed in 85.73s` against the recorded `18 passed in 5.74s`, so the record does not reproduce. Recorded as **I-017**; see `test_report_2026-11-08_h200_mainline_review.md` sections 3 and 4. |
| **F3** The final report overcounts its own checked errors | Record-keeping arithmetic error, no verdict consequence | **Confirmed and corrected.** The actual quantities are 4 Frontier values, 12 per-repeat vLLM values, 4 pooled vLLM values, and 16 errors. `test_report_2026-11-07_p2_64_generation08_formal.md` prose was corrected. The executor also noted that the lead's own assignment brief repeated the same wrong counts, which is accurate. |
| **F4** Several current-state statements contradict the completion evidence | Documentation and status errors | **Confirmed and corrected,** with the executor's scope constraint respected: case-specific corrections only, no global H800 boxes checked, no dated historical failure erased. Five entries corrected in `plan.md` (two cells), `tickets/17-dsv4-canonical-case.md` (Status paragraph), and `frontier/model_architectures.py` (one docstring, one comment clause; no executable line touched). |

### 12.4 Two findings the lead added

- **I-016, a defect the review could not have found.** Four of 41 mainline unit tests in
  `tests/unit/test_native_pd_worker.py` failed on the declared Python 3.10 floor, because line 122 passed the
  3.11-only `-P` interpreter flag to a child process. Introduced on 2026-10-21 by `9c54f663`. The executor's
  environment had no `pytest` (`handoff_audit.log` ENVIRONMENT `"pytest": false`), so it audited records and
  committed diffs only; the defect was found by running the suite. Fixed with a portable `sys.path` prelude
  that preserves the staged-path assertion; verified `41 passed in 1.14s`, exit 0.
- **I-017, an unmerged deliverable.** `git merge-base --is-ancestor 57caf2f7 HEAD` is false, and
  `git branch -a --contains 57caf2f7` lists only `p2-64-startup-harness`. The mainline `native_pd_worker.py`
  contains no `heartbeat` occurrence against 28 in the harness copy. The peer heartbeat check recorded as the
  I-013 resolution under D-P2-64-SIXTH, which the successful generation08 run used, therefore lives only on
  that branch, against the standing rule that worktree code and documentation return to the main branch on
  completion. The merge is **recommended, not performed**: both copies of the worker and of its test module
  have diverged in opposite directions, the change touches the native measurement harness, and the harness
  branch's own tests currently fail on an un-root-caused isolation leak.

### 12.5 Open items after this review

1. **GPU-hour accounting (F1).** Obtain the missing historical container clocks or authoritative never-started
   evidence, reconcile each generation separately, and merge generations 05–08 into the ledger. Estimates do
   not satisfy the present wording. If the clocks cannot be recovered, this returns to YC as an explicit
   acceptance decision. Ticket 19 stays `PARTIAL`.
2. **Merge `p2-64-startup-harness` into mainline (I-017)**, and root-cause the eight order-dependent timeouts
   before accepting the merged tests.
3. **The remaining two heartbeat claims.** The lead re-ran only the 18-test module. The `101 passed in 19.50 s`
   suite and the `10.01 s / 1.01 s` old/new stopped-peer timing comparison are still unverified.
4. **H800 P0/P1** remains later scope. H200 results never qualify H800, and no formal four-metric or 30%
   promotion receipt exists for those cases.

### 12.6 Verification and limits

Lead checks, commands and actuals: `test_report_2026-11-08_h200_mainline_review.md`. No GPU or platform
execution was performed for this review or its verification. Sections 1 to 4 of that report are CPU unit
evidence; section 5 is a re-reading of existing artifacts. The executor's own search for heartbeat stdout was
explicitly bounded (2 logs in retry08, 232 logs under the H200 scratch root, raw downloads and private state
excluded) and is not an exhaustive search of every historical session archive.

