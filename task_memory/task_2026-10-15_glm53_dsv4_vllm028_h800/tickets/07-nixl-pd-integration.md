# 07: NIXL PD on the existing handoff owners

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D-DSV4-H800-MAINLINE-CORRECTION: added the current route block; grouped NIXL for V4 PREFILL/DECODE is part of H4a-2, and the GLM-first ordering no longer applies. |
| 2026-10-22 | DSV4-only active NIXL acceptance removes GLM07/full-path predecessors; retained accepted shared implementation/native gates and parked GLM simulation/tests/calibration. |
| 2026-10-21 | P/D path now binds four priority hardware/scale cases; retained B structural acceptance and stopped CUPTI investigation without relaxing native timing. |
| 2026-10-21 | Q11 CUDA-event FULL dummy producer merged after 150 CPU checks; kernel-only attribution and H200 execution remain open and the consumer coverage error is unchanged. |
| 2026-10-21 | Bound active P/D coverage to the fixed 128-request 6144/1024-token QPS-1.0 H200 case; full unchunked prompt transfer, longer-context capacity and per-repetition native completion remain required evidence. |
| 2026-10-21 | The canonical case requires this path to run non-dummy on 64 H200; the FULL dummy DSA coverage producer continues (Q11) because TTFT/TPOT cannot be produced without it. |
| 2026-10-20 | M2 remainder done (`6b9ce80`, merged `b67cb02`); recorded the skipped-waiting count and DECODE-only CLI findings for calibration. |
| 2026-10-20 | Handoff status: M1 done; M2 remainder approved, not implemented; K5 NIXL READ fit recorded. |
| 2026-10-20 | Recorded review findings M1 (D-side over-reservation) and M2 (D lane policy) as work items. |
| 2026-10-20 Asia/Hong_Kong | Accepted pressure02 business and host-only pressure/handle evidence from 355 artifact checks; preserved historical FAIL, incomplete observer qualification and trained/formal gates. |
| 2026-10-19 | Synchronized actual full-GLM business PASS, 40-process census and C40 transfer-observation FAIL; recorded accepted handle repair and pressure02 submission without closing native or trained gates. |
| 2026-10-19 | Accepted complete C40 source/CPU and pressure/bootstrap reviews; synchronized frozen inputs and the submitted functional job while retaining native/formal gates. |
| 2026-10-19 | Synchronized C39 source/CPU acceptance and C40 native observation dependencies; native/formal criteria remain open. |
| 2026-10-19 | Accepted C16 controlled structural slice at2da11c8; native lifecycle and formal milestones remain open. |
| 2026-10-19 | Recorded passing R2 structural/legacy evidence and final ledger-derived same-lane validator correction; no native acceptance. |
| 2026-10-19 | Recorded actual C16-R1 reservation/growth deadlock; structural acceptance requires the corrected real-consumer liveness regression. |
| 2026-10-19 | Separated source-derived GLM structural lifecycle contract from still-unqualified native lifecycle/contention evidence; existing owners remain canonical. |
| 2026-10-19 | Linked current GLM component/payload/lifecycle readiness and disposition; no standalone metadata-only implementation started. |
| 2026-10-18 | Accepted C11 dc3912c GLM PP1 unified-role static storage preparation;157isolatedPASS/77exact artifacts/cache, no NIXL milestone closure. |
| 2026-10-18 | Documentation review: removed assumed D reservation semantics and separated structural PD from measured timing/lifetime acceptance. |
| 2026-10-18 | Completed current PDD source baseline and33existingCPU checks with one deliberate PD-AF deselection; no native lifetime, conservation or approved NIXL integration closure. |
| 2026-10-15 | Created in the plan phase. |
| 2026-10-15 | Review remediation: per-model milestones, GLM-5.3 first; the DeepSeek-V4-Flash closure waits for 09 `full_frontier_path[GLM]` (S1, Q12, Q13). The legacy P2P regression keeps the actual baseline behaviour of `enable_latency_hiding` (C2). Current code facts for the NIXL checks are listed. |

**What to build:** Frontier simulates a PD deployment using NIXL pull. Each request is one logical KV handoff after the required prefill, with no invented layer-wise overlap. D storage reservation follows the observed lifecycle disposition; D never computes before all required data is usable. Pinned source requires reserve-before-READ; existing Frontier handoff still needs this incremental integration. Actual native timing and cross-process readiness/release order require measurement. P holds the source until the release condition approved in D-PD. Legacy P2P NCCL pricing is isolated, and incompatible configurations are rejected. The existing transfer events, pending-KV set, and release callback stay the single owners. (Spec v2 T07, §5.)

**Blocked by:**
- `structural_pd_integration[GLM]`: 04 `structural_execution`, 03 `cache_inventory[GLM]`, and 12 GLM source-derived `structural_contract`; native lifecycle qualification remains a measured/formal prerequisite.
- `approved_pd_integration[GLM]`: `structural_pd_integration[GLM]`, 02 `probe_validation[GLM]`, 06 `raw_measurements[GLM]`, and complete `12[GLM]`.
- `lifetime_regression[GLM]`: `approved_pd_integration[GLM]`.
- `structural_pd_integration[DSV4]`: accepted reusable transfer owners,03
  `cache_inventory[DSV4]` and12 DSV4 lifecycle disposition;no paused GLM predecessor.
- `approved_pd_integration[DSV4]`: matching structural integration, 02 `probe_validation[DSV4]`, 06 `raw_measurements[DSV4]`, and complete `12[DSV4]`.
- `lifetime_regression[DSV4]`: matching approved integration.

`07[m]` closes only after its structural, measured integration and lifetime-regression milestones. Structural durations are labeled dummy timing; measured scope/range and native lifetime acceptance remain required.

**Status:** PARTIAL baseline/source preparation. C11 `dc3912c` admits GLM PP1 full-model PREFILL/DECODE source storage with157isolatedPASS and77exact artifact/cache comparisons. C16 controlled NIXL structural implementation is accepted at2da11c8 on existing owners, including actual allocator/event tests and full78-layer offline/online structural execution. Measured integration is NOT STARTED. C15 records the source-derived structural contract; native lifecycle and native PD serving capability remain unqualified. See `../test_report_2026-10-18_glm_unified_role_storage.md`.


## Current route — DSV4 H800 mainline (2026-10-25)

Authority: D-DSV4-H800-MAINLINE-CORRECTION in [requirements](../requirements.md#decision-2026-10-25--d-dsv4-h800-mainline-correction-current-route-authority). Route: [plan.md](../plan.md#current-execution-route--dsv4-h800-mainline-2026-10-25). Handoff: [progress.md](../progress.md#current-session-handoff--2026-10-25-mainline-correction).

- Grouped NIXL transfer for V4 between PREFILL and unified DECODE is part of H4a-2, together with ticket 05 `pd_support`. Use the existing transfer and KV allocation owners (`frontier/events/kv_cache_transfer_start_event.py`, `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py`, `frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py`).
- `approved_pd_integration (per model, GLM first)` below: the GLM-first order no longer applies. DSV4 reuses the accepted shared implementation and adds its own component contracts.
- H4a uses the existing NIXL transfer curves under `data/profiling/network/h800_dgx/`. A missing curve for the V4 payload is listed as a substitution in the H4a report, not filled silently.

## Canonical P/D scope — 2026-10-21 (scope still valid; route order superseded above)

Scope override2026-10-22:D-DSV4-ONLY-GLM-PAUSED. Only DSV4 simulation/tests/
calibration/new packages are active;GLM all tiers/proposal follow-up are
PAUSED_PENDING_USER_APPROVAL. Reuse accepted shared implementations without new
GLM07 work. All older GLM-first/dual-model build and closure directions below are
retained history. Active07[DSV4] still requires own structure,qualified native
lifecycle/02/06/12 and lifetime evidence;no coverage waiver or false PASS.

The formal path feeds10/11 and16–19 on the selected H800/H200 priority case:
32 GPUs=P16/D16 with TP8/DP2/EP16/PP1 per role;64=P32/D32 with TP8/DP4/EP32/PP1.
Use128 requests at QPS1 with fixed one-second arrivals, exactly6144 input /
1024 output tokens each, no prefix caching/chunking/thinking, graph/quantization
on and native-default MTP off. NIXL payload and reservation follow the complete
6144-token prompt and the native cache layout; decode capacity covers total length
7168. Require `max_model_len >= 7168` and
`max_num_batched_tokens >= max_model_len`, but derive effective graph and memory
settings from ticket 18 rather than declaring them frozen here. Native 128/128
completion in each of 3 independently warmed clean repetitions and qualified
lifetime/timing are still open; earlier eight-request H800 host evidence does not
close them. Matching hardware/topology communication coverage remains required.

V4 grouped structural/tracing B is accepted at2a84c58/2ffd2cb, including actual
MetricsStore callback and1024-output release checks. It does not establish
native wire/lifetime/timing or full-serving admission. Active V4 work follows
the current priority plan;GLM work is paused,not a serial-calibration predecessor.

### Historical Q11 producer progress (CUPTI follow-up is now stopped)

`0096cb9`, merged as `7f64919`, adds `frontier/profiling/attention/dsa_full_dummy.py`: native reduced-model FULL capture/replay with capture-time external CUDA events, real dummy invalid-slot/padded-metadata witnesses, and both index variants. Primary reran the five focused DSA modules: **150 PASS in 3.98s**. Source/CPU acceptance only; no GPU measurement has occurred. The output is isolated `dsa_full_dummy.v1` JSON with `consumer_admission=false` and `kernel_only_qualified=false`; CUDA-event wall time cannot replace the kernel-only timing required by the graph consumer. The current `DSACoverageError` stays intact. Follow-up W128-GRAPH-MAP investigates the smallest supported graph-node/scope attribution mechanism before the remaining producer work. Implementation records are in `task_memory/task_2026-10-21_glm_full_dummy_h200/`.

## Historical mainline recovery — 2026-10-20

Review findings from the [mainline review](../review_report_2026-10-20_mainline_and_tickets.md) on the accepted C16 slice `2da11c8`:

- **M1:** vLLM reserves `ceil(prompt/block)` blocks for an async remote KV load (`num_new_tokens=0`, no lookahead; `vllm/v1/core/sched/scheduler.py:918-921,1007-1011`). Frontier's `reserve_kv_transfer_target` calls `_allocate_request(request, 1)` on a request with `prompt+1` processed tokens (`frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:176-178`), reserving `ceil((prompt+2)/block)`; `tests/unit/test_glm53_structural_pd.py:505,523` assert 3 and 2 where vLLM gives 2 and 1.
- **M2:** D lane selection exists only on `RoundRobinClusterScheduler.select_kv_transfer_target`, so the v0.28 DP load-balancer policy cannot place D requests; native D runs DP2 with async scheduling.

### Status at the 2026-10-20 handoff

- **M1 done:** `5ad40e0` + `0a8cc8e` (prompt 64→1, 65→2, 128→2, 192→3 blocks). A non-NIXL PDD D-side reservation question is recorded in `future.md`.
- **M2:** interface declared on the cluster scheduler base `7311c17`; the remainder is approved (D-M2-APPROVED: `select_kv_transfer_target(time, request)`, DECODE role only, structural PD driver identity plus new placement unit tests, matrix PD coverage gap recorded) and not implemented. Design: [w3_review_fixes_20261020/m2_m3_design.md](../w3_review_fixes_20261020/m2_m3_design.md).
- **NIXL value for K6:** K5 measured 126.2 Gbit/s + 0.0647 ms per read (ticket 06).

### Status after the OpenCode v2 resume (2026-10-20)

- **M2 remainder done:** `vllm_load_balancing_cluster_scheduler.py:97` DECODE selector, `kv_cache_transfer_start_event.py:156` passes `(time, request)`, `base_replica_scheduler.py:963` admission report; vLLM reference `core_client.py:1471-1522,1539-1542` at `2cf0a6915ce5`. Records `m2_decode_lane_20261020/`.
- **Open for ticket 10 calibration (on the GLM target path, not chased now):** vLLM's DP request-count publication counts skipped waiting requests, which include requests waiting for remote KV (`scheduler.py:2293-2295` `get_request_counts()`, `core.py:2120-2135` `_maybe_publish_request_counts()`), while Frontier's approved DECODE waiting count (`vllm_v1_iteration_policy.py:551-575`) excludes them. On the PD D role this can change lane choice under load. Settle it with the native D-side placement evidence in ticket 10; changing the count needs YC approval because it changes an approved definition.
- **Limitation:** the shared CLI cannot select a cluster scheduler policy for the DECODE role alone (`m2_decode_lane_20261020/notes.md`).
- M3 (two-batch async queue on DECODE) stays deferred.

## Current C39/C40 continuation

Full original-GLM functional PD passed eight HTTP requests, 2400 prompt/32 output tokens and all platform/native/container/launcher exits. Eight D reservation/wait/receive/application/promotion chains, eight P held/notification/free chains and 16 immediate physical frees are observed; physical returns total P40/D44 blocks. [Actual C40 review](../c39_glm_full_pd_20261019/functional_c40_review/test_report_2026-10-19_functional_c40_actual.md).

The original functional C40 result remains PARTIAL with READ handle observation FAIL: D serialization errors removed 320 positions, including successful prepare, submit, DONE and release. That functional case exercised no refusal, lease expiry or deferred free. The separate pressure result below now supplies actual capacity refusal/free/readmission and repaired handle host records. Full native lifecycle disposition, contention and measured integration remain OPEN.

Native repair `169309961` and pressure02 inputs retain their independent source/CPU acceptance (80 suite tests, 17 actual-body CPU cases, 36 frozen files). The actual pressure02 job `exp-1019-232508-663845` now passes business/terminal checks: eight client and sixteen upstream HTTP 200 responses, 7552 prompt/640 output tokens, four native-parent/container exits 0 with no restart. Platform finish is 2026-10-19T16:07:17Z (October 20 Asia/Hong_Kong); launcher exit 0 is 16:07:18.177688Z. Earlier Starting receipts and watcher time 16:10:19Z are historical observations, not the current state.

The finite pressure review passes 355/355 retained-artifact checks. It accepts five same-owner refusal → prior free → readmission witnesses across both D lanes (422 refusals/eight successful reservations) and 64 complete serializable prepare/submit/native-DONE/handle-release host chains. D17 includes null, leaving sixteen usable blocks; initial successful reservations own fifteen, and all sixteen are held at final free. Eight D scheduler chains, 64 P notification/completion pairs and sixteen frontend route/release chains are observed. These close the selected host pressure/handle evidence only and never rewrite the original functional transfer FAIL.

All forty pressure traces have contiguous sequences, zero recorded drops and no observer/footer errors, but all footers retain `run_verified=false`. P848/D5744 missing-parent counts and 64 unknown receive-report sources remain; 32 worker manifests are false and eight engine/frontend manifests omit qualification. Within-node clock metadata agrees, while distinct node boot IDs leave cross-node timing unqualified. Native DONE remains a host observation; absolute device endpoints/readiness, complete observer qualification and measured contention stay OPEN. Teardown EngineDeadError/SIGTERM warnings are retained. [Pressure actual review](../c39_glm_full_pd_20261019/pressure_actual_review/test_report_2026-10-19_pressure_actual.md), [355-check receipt](../c39_glm_full_pd_20261019/pressure_actual_review/verification_receipt.json).

## Accepted C16 structural slice — 2026-10-19

C16 is accepted at2da11c8. Root1124isolated/205independent checks, complete callback/FIFO/conservation oracles, both full78-layer main outputs, final APPROVE/zero unresolved and77identical legacy artifacts/zero cache filename differences establish the controlled dummy structural slice. Final15 staged/committed bytes match. Offline tail4 shares D(1,0) withrequest1; online withrequest0. Actual R2 execution receipts and separate R3 validator-only postvalidation remain distinct. Official comparator exit1 is solely modified-source provenance. Native lifecycle/timing/wire bytes, trained integration, graphs and formal acceptance remain open. See [report](../test_report_2026-10-19_glm_structural_pd.md).

Historical rejected-generation diagnosis: The R1 actual allocator/event counterexample uses a4-block D pool: old prompt127/decode20 owns2blocks at processed128; incoming prompt64/decode3 reserves2beforeREAD; old needs3next and preempts. After completion, FCFS waiting=[old,incoming], free2, incoming owns2; neither progresses in3actual phase2 attempts. C16-R1 is unaccepted despite1119root-isolated and200reviewer-focused passing tests. Require the smallest existing admission/scheduling correction, a persistent actual-consumer progress/conservation regression, final-byte review and new-generation verification. Preserve reservation-beforeREAD, D-ready gating, P lifetime, pending-transfer FIFO and all native/formal gates. Evidence: [oracle](../c16_glm_structural_pd_20261019/growth_deadlock_review_r1/growth_oracle.json). This concrete routine bug does not establish a need for a new core architecture.

## Current code facts at `bfa82de`

- `frontier/config/kv_cache_transfer_config.py` rejects a nonzero `kv_send_slowdown_per_layer_ms` only when `kv_connector == "none"`. Adding a NIXL connector name alone would accept a nonzero slowdown, so the rejection is new work.
- `enable_latency_hiding` is marked as a future extension in the same file. `supports_latency_hiding()` in `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py` only returns the flag, and no production path reads it to overlap transfer with prefill.
- `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` prices the per-layer KV extract only for `kv_connector == "p2p_nccl"`.
- `frontier/events/kv_cache_transfer_end_event.py` delivers the KV to the target cluster (`on_kv_cache_arrival`) and releases the source (`complete_kv_transfer_for_requests`) in the same event. Whether this stays valid under NIXL is decided in `12[GLM]`, not assumed.

## Current PDD preparation — 2026-10-18

`../ticket07_lifetime_baseline.md` records current source order: transfer end delivers to the D queue and frees P, while actual D admission/allocation occurs later in the replica scheduler. This is not the requested NIXL reservation-before-READ model and selects no D-PD abstraction. Three unchanged CPU commands produced33PASS/1deliberate PD-AF stage-hook deselection/0FAIL. Primary verified34source pins, all author/test receipts and three full logs, and separately bound the later-read role-schedules file. Source order and callback-free stubs do not prove real P/D conservation, native readiness, near-full D behavior, late/duplicate completion or OOM handling. No test rerun or production edit occurred. Evidence: `../test_report_2026-10-18_concurrent_preparation.md`, core76-entry index and separate56-entry `completion_20261018/manifest.json` under `../parallel_preparation_20261018/concurrent_wave_20261018/`. The original core's pending-at-creation flags remain historical. Formal milestones/checklists remain unchanged.

## Structural serving milestone

### structural_pd_integration (per model)
- [x] M1: D-side reservation equals vLLM `ceil(prompt/block)`; later decode growth allocates through the normal path; test values follow vLLM.
- [x] M2: D-side lane selection goes through the configured cluster scheduler policy, including the v0.28 DP load balancer. Evidence: `6b9ce80` (merged `b67cb02`), `test_report_2026-10-20_m2_decode_lane.md`: default-policy PD driver artifacts byte-identical to `bb4aeff`, 77/77 matrix identical, asymmetric snapshot test picks lane 1 where round robin picks lane 0.
- [ ] Extend existing connector capability checks, pending-KV owner and transfer events with native-evidenced NIXL identity/payload semantics. Reuse current duration interfaces in labeled structural mode; no measured curve prerequisite for structural execution.
- [ ] Use component resident bytes separately from valid transfer bytes; record required data readiness, D admission/reservation and P release. Test single/two requests and reachable near-full D cases for premature execution/free, leak and duplicate charge.
- [ ] If existing capacity or event ownership cannot express the observed lifecycle, document exact code/counterexample and seek approval for the smallest core refactor before that change. Keep unrelated representation/operator work moving; do not invent a temporary PD path.

Read-only GLM07/12 readiness and primary disposition are retained in `../c14_attention_workload_20261019/glm_pd_readiness/` and `../ticket07_glm_structural_readiness.md`. The existing component/allocator/request-event/release owners remain reusable; no new manager is demonstrated. Connect metadata only as required by an actual transfer payload consumer. Logical content, resident capacity and native READ descriptors remain distinct; lifecycle evidence gates and scalar transfer guards are unchanged.

## Milestones

### approved_pd_integration (per model, GLM first)
- [ ] Reuse the NIXL connector identity/capability extension delivered structurally; GLM owns the canonical implementation and DeepSeek adds component contracts. Routine identity/config entries are authorized; a redesigned schema requires approval.
- [ ] Payload from the cache inventory's groups and descriptors, without dense `tokens × layers × head_dim` scaling.
- [ ] D admission and reservation, ready, and source-free boundaries exactly as the model's D-PD stage decided.
- [ ] Reject a nonzero `kv_send_slowdown_per_layer_ms` and an enabled `enable_latency_hiding` under NIXL. No P2P extract charge unless native NIXL gathers.

### lifetime_regression (per model)
- [ ] Run controlled traces for a single request, two concurrent requests of different lengths, D cache near full, and overlapping EP+NIXL on the same NIC.
- [ ] Conservation tests: normal, receiver out of memory, duplicate or late completion, abort. No double free and no leak.
- [ ] Legacy P2P regressions, including `test_kv_cache_extract_pricing.py`, stay unchanged, and the fidelity comparison passes. The P2P baseline has no layer-wise overlap path today; this ticket keeps that behaviour and does not add P2P overlap.

## Acceptance

- [ ] No false layer-wise overlap, no premature D execution or free, no double charge or double release.
- [ ] No copy of the connector's background protocol.

## Structural consumer handoff — 2026-10-19

`structural_contract[GLM]` is SOURCE-REVIEWED, separate from the native `lifecycle_disposition[GLM]` acceptance below. Pinned vLLM reserves D blocks before READ, keeps D unschedulable until all handles plus receive post-processing are usable, and releases P after P observes all expected consumer notifications. D readiness and P notification completion are distinct cross-process boundaries; source alone does not establish which becomes scheduler-visible first or their measured delay. Retained observer fixtures prove bounded identity/host-body order only, not a distributed GLM PD run.

A labeled dummy structural implementation may reuse existing target fields, D allocator, transfer start/end, P pending set and release callback. Its deterministic D-ready then P-release transitions at one simulated timestamp are an explicit structural approximation, not a native ordering or zero-delay claim. Preserve these as distinct transitions so later measurement can refine timing. No new manager, per-page events or generic resource graph is demonstrated. Actual code must preserve pre-READ reservation, no D execution before readiness, allocator conservation, backpressure and supported error/duplicate behavior. A demonstrated need for a material core change still requires the smallest reviewed proposal and approval.

Connect the existing AnalyticalKVCacheTransferPredictor to component-aware whole-page structural payloads checked against D reservations; keep native READ bytes UNKNOWN. This source contract unblocks only development and CPU structural verification for07[GLM]. Native lifecycle qualification, distinguishing workloads, contention, measured payload/timing and formal gates stay open under their existing criteria. Evidence: `../c15_source_execution_20261019/glm_pd_evidence/`; primary disposition: `../c15_source_execution_20261019/glm_pd_evidence/primary_disposition.json`.
