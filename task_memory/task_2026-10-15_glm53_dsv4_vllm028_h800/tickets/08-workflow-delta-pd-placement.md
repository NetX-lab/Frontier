# 08: v0.28.0 workflow deltas and P/D placement

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Synchronized current H200 workflow evidence and direct Codex execution; full acceptance remains open. |
| 2026-10-22 | Active DSV4 workflow/placement completion no longer waits for GLM08/full path; parked all new GLM work and retained native async/placement/coverage requirements. |
| 2026-10-20 | Recorded review finding M3 (async two-batch queue for PD/DP2) as deferred work. |
| 2026-10-19 | Synchronized actual full-GLM business PASS, 40-process census and C40 transfer-observation FAIL; recorded accepted handle repair and pressure02 submission without closing native or trained gates. |
| 2026-10-19 | Accepted complete C40 source/CPU and pressure/bootstrap reviews; synchronized frozen inputs and the submitted functional job while retaining native/formal gates. |
| 2026-10-19 | Synchronized C39 source/CPU acceptance and C40 native observation dependencies; native/formal criteria remain open. |
| 2026-10-19 | Synchronized C36 81 evidence PASS/3 threshold FAIL, C38 actual capability and full-V4 available-artifact reviews; C39 preparation does not close workflow or placement gates. |
| 2026-10-19 | Updated C36 to reviewed implementation under standing authority; source/CPU, native duration and perturbation remain separately gated. |
| 2026-10-19 | Accepted exact C35 source/CPU capacity slice and unchanged 77-case comparison, retaining seven inherited PP failures and all measured-workflow/decision gates. |
| 2026-10-19 | Recorded C33 policy/clean 563dfff matrix, C34 selected native inputs and authorized single-owner C35 ordinary-cache capacity implementation with verification pending. |
| 2026-10-19 | Accepted C29 queue source/CPU and C30 policy oracles, committed C31 bootstrap/inventory and submitted exact original-V4 diagnostic; native/formal outcomes remain separately gated. |
| 2026-10-19 | Accepted C25/C26/C28 source and CPU slices, retained completed one-H800 GLM graph evidence and C27 workflow deltas; native/formal gates remain open. |
| 2026-10-19 | Accepted scoped C23 cache-zero and C24 DP8 CPU evidence; synchronized remaining native inputs, ownership and formal limits. |
| 2026-10-19 | Distinguished accepted controlled C16/C17/C20 workflow execution from unimplemented target placement/native runtime-diff integration. |
| 2026-10-18 | Documentation review: unblocked source comparison and added structural workflow before measured integration; preserved placement guards. |
| 2026-10-15 | Created in the plan phase. |
| 2026-10-15 | Review remediation: per-model integration milestones, GLM-5.3 first; the DeepSeek-V4-Flash closure waits for 09 `full_frontier_path[GLM]` (S1, E4, Q12, Q13). Current placement-guard facts are listed. |
| 2026-10-17 | Completed current Frontier workflow/test preparation, preserving the unchanged historical late-join failure and native decision gates. |

**What to build:** Frontier's vLLM V1 scheduling follows v0.28.0 wherever the active loop differs in a way that changes admission, batch shape, resource use, or outputs. It does so by adjusting the existing batch queue, stage host/device timing, DP dummy forward and finish synchronization, and held outputs rather than adding a second engine. P/D entry and DP-lane placement follow the native routing contract decided in D-DP-PD. (Spec v2 T08, §7, §9.1.)

**Blocked by:**
- `active_runtime_diff`: 00 and 01 `runtime_lock`; source comparison starts before complete observer/cache/routing cards. Close each row as source-supported or observation-needed; executed behavior needs native evidence.
- `workflow_structure[GLM]`: `active_runtime_diff`, 12 GLM `lifecycle_disposition`, 13 and 15 dispositions.
- `approved_workflow_integration[GLM]`: `workflow_structure[GLM]` and relevant validated native workflow comparisons.
- `workflow_structure[DSV4]`: active DSV4 runtime diff,13 DSV4 disposition and12
  DSV4 lifecycle disposition;accepted shared implementation reusable,no GLM edge.
- `approved_workflow_integration[DSV4]`: matching structure, relevant validated comparison and 15 `dsv4_agreement` for native placement claims.

`08[m]` closes only after its source diff, structural serving and measured workflow integration. Do not claim a row is `SAME_CONTRACT` solely because method names match.

**Status:** PARTIAL. C27 source deltas, C33 explicit existing-balancer policy and C35 capacity admission are accepted within source/CPU scope; C34/C38 supply selected actual GLM queue/output and host boundaries for comparison. C16 accepts controlled GLM sequential PD, and C17/C20 controlled V4 co-location/logical boundaries. Target P/D placement, complete native async/graph qualification and measured workflow integration remain OPEN. C35's focused run retains seven inherited PP failures; the two earlier C33 placement failures also remain historical unresolved findings. Neither source slice declares those causes fixed.



## Current Codex checkpoint — 2026-10-31

B24 EP lane reuse, B26 DP dummy-forward run-ahead and B29 serving-path delays are implemented and verified on the H200 operator route. Decision 46 keeps measured CC formal; F2 compares K=19 with a one-GPU measurement. Complete workflow qualification remains OPEN. Use [current plan](../plan.md#current-execution-route--codex-cli-h200-operator-path-2026-10-31) and [current handoff](../progress.md#current-session-handoff--codex-cli-2026-10-31); old execution snapshots below are historical. Codex directly submits and monitors personal StepMind RJobs; GLM stays paused.

## Current scope override — D-DSV4-ONLY-GLM-PAUSED (2026-10-22)

Current simulation,tests,correction,calibration and acceptance are DSV4 only. All
GLM tiers/new packages/proposals are PAUSED_PENDING_USER_APPROVAL;retain prior
code/data/raw receipts/UNVERIFIED and do not rerun GLM tests. GLM-first and paired
completion directions below are historical. Active08[DSV4] reuses accepted shared
queue/placement work,requiring its own12/13/15 and native/observer qualifications;
missing V4 evidence remains blocking. No new runtime or material design approval.

## Historical mainline recovery — 2026-10-20

- **M3 (gap, deferred until W4 profiles run):** `max_concurrent_batches=2` is limited to MONOLITHIC PP1/DP1 (`vllm_v1_engine_replica_scheduler.py:366-382`), and in-flight decode requests are skipped (`:1010-1018`). vLLM v0.28 async scheduling re-schedules them with output placeholders (`scheduler.py:530-562`). The GLM target is PD with DP2 and `async_scheduling=true`, so the two-batch queue must extend to PD/DP2 with placeholder semantics before CPU-overhead modeling is enabled.

## Current C39/C40 continuation

Full original-GLM functional PD now supplies actual DP2 frontend evidence: 16 choices and 16 terminal releases pass native score/tie/inflight comparisons, and all four frontend inflight vectors end at [0,0]. Each role assigns four requests to each DP owner. Actual frontend client count is two per role; P clients handle 5/3 requests and D clients 0/8. This is diagnostic route/ownership evidence, separate from KV release. [Actual C40 review](../c39_glm_full_pd_20261019/functional_c40_review/test_report_2026-10-19_functional_c40_actual.md).

Final census is 40 processes, not the planned 38. C40 remains PARTIAL with D transfer-observation FAIL; all processes remain unverified and frontend EngineCore ownership is null. Qualified two-lane evidence, G0-C/schema and G0-D/DP4 are still needed before Ticket15 disposition. No P/D policy choice or constructor guard is changed.

Native repair `169309961` and pressure02 inputs are independently APPROVED (80 suite tests and 17 actual-body CPU cases). The new stage has 36/36 matching files; config SHA-256 is `9100e06faca15a6315b5b7b209f7e05f937a227738f1d367df3c7a1463fa53d1`. Pressure02 `exp-1019-232508-663845` was submitted at 15:25:08 UTC on 32 non-preemptible H800/codesign; the latest retained receipt at 15:33:25 UTC reports Starting with four pending replicas and no assigned nodes. The existing 8 MiB serving buffer preserves D17/P-profiled, workload and bootstrap 12-record/65,536-byte sub-budgets; serving startup_memory remains None. Root keeps the same job/launcher; native/terminal acceptance is pending. [Source/input acceptance](../c39_glm_full_pd_20261019/handle_serialization_repair/independent_review/receipt.json), [freeze](../c39_glm_full_pd_20261019/pressure_frozen_02/receipt.json), [15:33:25 UTC platform](../c39_glm_full_pd_20261019/pressure_platform_02.json).

## Current source slices, native evidence and C39 preparation

C33 `563dfffd1ee30b136ae76e03df306b26418e476c` adds explicit `policy_version="v0.28.0"` through the existing config/balancer, preserving legacy `v0.10.2`, default round-robin and all placement guards. Real Request terminal ownership, count-triggered snapshot KV pressure and capacity semantics pass 163 focused checks and three DES cases. The unchanged comparator passes 77/77 only for clean candidate `563dfff` against clean `bfa82de`; independent review verifies 902 artifact files and both 519-name cache inventories, not cache binary equality or later HEADs. Two inherited placement tests remain FAIL. [C33 evidence](../c33_dp_policy_20261019/report.md).

C34 [selected native audit](../c34_native_queue_20261019/actual_native_review_02/test_report_2026-10-19_glm_native_queue.md) passes 27 checks for actual capacity two/FIFO, original output-parent retention and graph host returns. It leaves native arrival, physical GPU overlap and observer qualification unresolved and retains shutdown-time API `EngineDeadError`.

C35 is accepted at exact integrated commit `8d479ef5b232f198035059441afc33bd88e9fc68`. Existing `max_concurrent_batches=None` preserves physical pipeline capacity and legacy serialization; explicit two admits only the exact V1 scheduler, MONOLITHIC PP1/DP1 ordinary cache without prefix caching, speculation or GDN. Stages remain serial; grouped/disaggregated/subclass guards remain. Root's real capacity/stage/preemption/release/graph selection records 135 PASS / 7 FAIL. The same seven IDs and exception headings, normalized only for printed object addresses, reproduce on clean `da13911`: parent 31 PASS / 7 FAIL, candidate 33 PASS / 7 FAIL in the preemption module. This establishes inheritance, not a root-cause fix.

The unchanged comparator passes 77/77 using clean source `8d479ef` and harness `da13911`, whose comparator files match the baseline byte-for-byte. Independent inspection verifies all 902 artifact files (451 per side) and 519 actual cache filenames per side. Equality follows existing run-path normalization; cache binary contents are not compared. [Exact commands, manifests, inherited failures and comparison receipts](../test_report_2026-10-19_queue_admission_full_v4.md). Protected `test_e2e_gate.py` and `.pi/` remain retained. No output placeholder, held-output or timing framework is added; neither C33 nor C35 settles Ticket15 or D-ASYNC.

C36 [independent review](../c36_glm_observer_20261019/paired_actual_review/test_report_2026-10-19_paired_observer.md) passes 81 evidence checks and fails all three frozen 3% perturbation thresholds; all ten cases remain eligible and retained. C38 [actual review](../c38_native_boundaries_20261019/actual_review/test_report_2026-10-19_native_boundaries.md) passes root46/independent51 for 22 positive-step boundary sets and verifies the same two actual host clocks. Absolute GPU endpoints and original launcher exit remain UNVERIFIED; startup13, teardown EngineDeadError and complete observer qualification remain unresolved.

Full V4 [available-artifact review](../v4_full_structure_20261019/actual_review/test_report_2026-10-19_v4_full_structure_actual.md) passes 415 checks for original structure, initialization, allocation and generation. G0-B remains OPEN: zero EngineCore traces, forced native teardown, 736 worker missing-parent scopes and strict routing OFF; original launcher exit is UNVERIFIED. None of these results supplies native P/D placement or closes D-ASYNC.

The earlier [9/9 readiness probe](../c39_glm_full_pd_20261019/test_report_2026-10-19_glm_full_pd_readiness.md) records historical deployment gaps. Those source gaps are now closed by accepted C39/C40, bootstrap and pressure-harness work; both stages are frozen. Root owns runtime evidence; c36_overhead retains input ownership. Native READ/ready/release, applicable perturbation qualification and measured workflow requirements remain necessary.

## Historical C29–C31 checkpoint — superseded above

C30 completes five unchanged-source-body policy cases,10 scored choices,one explicit-rank control,three async factory branches and native report suppression `[0,1,1]`. New scoring is selected by online internal `DPLBAsyncMPClient`, not by offline LLM's AsyncScheduler setting. The existing Frontier policy has no automatic v0.28 binding. A small legacy-default selector, canonical snapshot KV metadata and actual request-owner/finished-ID tracking in the existing balancer is the proposed next source extension; no production change or Ticket15 disposition is made here.

Retain count-triggered publication even when only KV usage changes, carry snapshot-local pressure through existing latched reports, and test total-versus-allocatable/null capacity mapping before claiming native equality. Ownership is released on real terminal output, not chunk/preemption/stale-drop. C29 source/CPU queue hooks are committed but native capacity2/settlement evidence remains pending. One owner each for scheduler/cache and balancer/publication; version-specific policy tests can proceed independently of native instrumentation. Existing timers, legacy tests, constructor guards and Ticket13/15 native prerequisites remain. [Detailed proposal](../c29_c31_native_progress_20261019/c30_policy/report.md); [root verification](../test_report_2026-10-19_native_queue_v4_g0a.md).

C31 actual original-V4 uses V2 Model Runner and MARLIN MXFP4; V1 runner internal hooks cannot establish its internal coverage. Native prefix caching resolves to the unchanged defaultTrue; Frontier V4 prefix caching remains unsupported. The first-forward blocker is crossed, but twelve scale rows/four quant auxiliary inventories remain invalid; do not label the Failed diagnostic as G0-A acceptance. Selected backend/runner evidence now narrows the next native/inventory work.

## Historical C25–C28 checkpoint

C27 source/evidence audit is DONE for the read-only comparison; workflow_structure and approved integration remain PARTIAL. Its57pins establish selected GLM AsyncScheduler/asyncTrue/PP1/mp/V1 on all16 historical ranks: native capacity2 versus Frontier1. Internal DP policy differs (client outstanding floor, KV pressure, rotating ties), and actual graph selection needs captured descriptors and applicable DP agreement. PP3 zero-output queue fill is conditional; cadence>1 is deferred unless selected; interval32 is source-matched. Scalar report keys need causal mapping evidence, not automatic replacement. C16 already reserves D before READ; refine its known D-ready/P-free approximation only from joined lifecycle evidence. C25 now provides three actual real-forward replay host returns in a reduced GLM diagnostic; it supplies no device timing, async-overlap or cross-DP proof. Prepare existing queue/frontier and version-bound balancer/graph oracles, preserving guards and legacy tests; no new EngineCore/registry/PD manager is justified. [Root verification](../test_report_2026-10-19_native_inputs_dummy_init.md); [source delta report](../c25_native_progress_20261019/c27_audit/report.md); [dependency and owner plan](../plan.md).

## Historical C23/C24 checkpoint

C23/C24 are accepted at `88c030fdfee25adf49fbf7718a7166cc09e37052` for source/CPU structural scope only: 1,292 isolated checks, source-backed actual 64-logical-device offline/online, allocation-to-zero execution and unchanged77artifact/cache equality. Ticket05 remains PARTIAL; graph/PD/native/trained/formal acceptance remains OPEN. Actual representative zero counters, source/command identities and limitations are recorded in the [verification report](../test_report_2026-10-19_dsv4_cache_zero_dp8.md). The earlier C21/C22 missing-consumer and DP8 guard descriptions below are historical checkpoints, superseded by this result.

Use the existing scheduler/allocator, operator registry, timing/trace, metrics and producer contracts. This removes the local cache-zero/DP8 CPU blockers; it does not supply native routing assignments, causal clocks, graph/READ behavior or measured latency. GLM-first trained integration and model-specific native prerequisites remain unchanged.


## Verified preparation — 2026-10-17

`../ticket08_workflow_baseline.md` records current admission, queue/output, DP dummy/sync, load publication, KV release and topology limits. Delegate unit81PASS; placement15PASS/1FAIL; focused sync/held2PASS. Primary independently reran81unitPASS and the same isolated historical lane2 `(0,False)` versus expected `(0,True)` failure, and checked40 retained summaries. Larger60s timeouts are not PASS. Baseline audit already records the exact failure; no new regression, nativeVERSION_DELTA or production repair is inferred. Verification: `../test_report_2026-10-17_parallel_preparation.md`. Constructor guards and D-ASYNC/D-DP-PD/shared-KV choices remain unchanged.

## Baseline code facts at `bfa82de`

- `VllmLoadBalancingClusterScheduler` (`frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py`) rejects any cluster that is not `MONOLITHIC`, has more than one Replica, or does not use the vLLM V1 replica scheduler; dense models also need `attn_dp=1`. Topology A needs PREFILL and DECODE roles, and topology B has three DECODE Replicas, so removing only the PDD check would not express either.
- `VLLM_DP_SYNC_INTERVAL = 32` is defined in `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`.

## Structural workflow milestone

### workflow_structure (per model)
- [ ] M3: async two-batch queue for PD roles with DP2, re-scheduling in-flight decode requests with vLLM output-placeholder semantics.
- [ ] Reuse existing admission/in-flight queue, stage host/device cost, DP synchronization and placement owners for observed target semantics, using structural timings while numerical evidence is incomplete.
- [ ] Demonstrate reachable online arrivals, multi-chunk prefill, graph/eager boundaries, idle-lane participation and PD handoff without silently discarding unsupported work.
- [ ] Preserve the full DP-placement constructor guard until the chosen roles, replica count, load reports and causal timing are actually implemented and tested. A new EngineCore/async framework is not an assumed requirement.
- [ ] Escalate only a demonstrated core limitation/material semantic choice; normal supported-module corrections follow current authorization. Structural execution is separate from a validated native workflow comparison.

## Milestones

### active_runtime_diff
- [ ] For each row of v2 §7.2 (iteration and admission, in-flight and outputs, CPU/GPU async, DP and EP, CUDA graph, PDD, network and route, terminal), classify the observation as `SAME_CONTRACT`, `VERSION_DELTA`, `INACTIVE_IN_SCOPE`, `MEASUREMENT_ONLY`, or `NEEDS_DISCUSSION`, with native evidence.
- [ ] Compare `VLLM_DP_SYNC_INTERVAL=32` with the v0.28.0 active loop.
- [ ] Check uncovered paths (PDD decode recompute, special memory pressure) by experiment, not by assumption.

### approved_workflow_integration (per model, GLM first)
- [ ] Make minimal changes only for active `VERSION_DELTA` items, each with a reachable failing test written first.
- [ ] Implement P/D placement as decided in D-DP-PD; never delete the guard as the implementation. Test placement and load-report timing and causality, not just that the constructor succeeds. GLM closure only; the DeepSeek-V4-Flash closure reuses it.
- [ ] DeepSeek-V4-Flash closure: classify its G0-C observations against the GLM-5.3 rows, change code only for a model-specific active `VERSION_DELTA`, and record source/native disagreements with dispositions 13 or 15; seek approval when they require a material semantic or core architecture change.
- [ ] Re-run the regressions of the already-merged fixes and the fidelity comparison.

## Acceptance

- [ ] Every new dependency and configuration has native evidence.
- [ ] No second scheduler or resource framework.
- [ ] No merged fix is redone or reverted.
