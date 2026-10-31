# Spec: active DeepSeek-V4-Flash calibration on H800/H200; GLM paused

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded completed A6 repeatability and F1a diagnostic outcomes; formal inputs and acceptance criteria unchanged. |
| 2026-11-01 | Recorded F1b held-out and terminal coverage failures without changing the measured-input or acceptance contract. |
| 2026-11-01 | Updated A6 from pending to completed with qualification failures; recorded rejected F3 fit. |
| 2026-10-31 | Synchronized Codex execution, active H200 route, measured formal inputs, B29 delays and per-repeat qualification limits. |
| 2026-10-22 | W128-V4-S5-STATUS: separated historical execution snapshots from current scope/rules and linked main progress; no contract changes. |
| 2026-10-22 | Final execution snapshot: GLM pause confirmed,warm-q1 corrected job Running/business pending,native04 sealed/unadmitted and05 unsealed pending versioned full-checker repair;no new scope/gate approval. |
| 2026-10-22 | D-DSV4-ONLY-GLM-PAUSED overrides dual-model execution/completion: current testing/correction/calibration/acceptance DSV4 only, all GLM tiers/new packages/proposals paused until new explicit user approval. |
| 2026-10-21 | Replaced64-H200-first with P0 H80032, P1 H80064 and two P2 H200 tasks; separated basic validation from30% promotion and capped personal H200 non-preemptible reservations at32. |
| 2026-10-21 | YC raised aggregate H200 usage to130; the formal per-model64-H200 case is unchanged. |
| 2026-10-21 | Verified reciprocal Step-5 adoption958d81a with all five68f891b package files identical; runtime qualification remains separate. |
| 2026-10-21 | Bound the common workload to exact68f891b bytes for GLM/V4 and future Step-5 adoption; recorded the two approved Q10 exceptions. |
| 2026-10-21 Asia/Hong_Kong | Distinguished returned token length from computed KV and froze source-derived successful NIXL tail-query endpoints without changing workload or metric gates. |
| 2026-10-21 Asia/Hong_Kong | Replaced the formal SharedGPT workload with 128 fixed-length requests (6144 input / 1024 output tokens), QPS 1.0 at one request per second, independent warmup for each clean repetition, and an explicit full-completion THPT window. SharedGPT artifacts and prior freezes remain historical; native serving, graph and memory evidence remain open. |
| 2026-10-21 Asia/Hong_Kong | Third round: retiming fixed as the historical linear rescale (identical to the dev-vidur generator, frozen at `9ed979e`); arrival-span figures corrected (256 requests span 63.75 s at QPS 4.0, not 24.6 s). |
| 2026-10-21 Asia/Hong_Kong | Adopted the canonical H200 calibration contract (`calibration.md`): platform moves from 64 H800 to 64 H200, scope reduces to one formal E2E case per model and four independently gated metrics (TTFT, TPOT, THPT, E2E), H800 tables retained as history, multi-workload/multi-QPS/multi-layout campaigns superseded. Owner overrides recorded: 256 formal requests instead of head64 (Q3=C), measured communication backend retained for the formal case (Q5=B). |
| 2026-10-20 Asia/Hong_Kong | Adopted the mainline review: routing removed from G0-B, observer qualification frozen, review fixes and profiling producers prioritized, H800-unlimited/H200<=70 GPU rule and loop cap. |
| 2026-10-20 Asia/Hong_Kong | Accepted pressure02 business and host-only pressure/handle evidence from 355 artifact checks; preserved historical FAIL, incomplete observer qualification and trained/formal gates. |
| 2026-10-19 | Synchronized actual full-GLM business PASS, 40-process census and C40 transfer-observation FAIL; recorded accepted handle repair and pressure02 submission without closing native or trained gates. |
| 2026-10-19 | Accepted complete C40 source/CPU and pressure/bootstrap reviews; synchronized frozen inputs and the submitted functional job while retaining native/formal gates. |
| 2026-10-19 | Accepted C39 source/CPU commits and156 tests; recorded single-owner C40 lifecycle/route work before final full-GLM deployment input freeze. |
| 2026-10-19 | Synchronized completed C36 evidence81 PASS/threshold3 FAIL, C38 root46/independent51 actual PASS and full-V4 available-artifact415 PASS; qualification gates remain OPEN and C39 preparation has no GPU run. |
| 2026-10-19 | Recorded C36 49-check reduced native acceptance, frozen five-pair perturbation run and accepted C37 clock-metadata source slice; full qualification remains open. |
| 2026-10-19 | Updated C36 to reviewed implementation under standing authority; source/CPU, native duration and perturbation remain separately gated. |
| 2026-10-19 | Recorded accepted C35 and reduced V4 G0-A while preserving full-model, observer and formal boundaries; full V4 is queued at the retained snapshot and C36 is proposal-only. |
| 2026-10-19 | Added C32–C34 evidence boundaries and limited C35 implementation scope; C33 comparison remains pinned to 563dfff and all qualification gates stay open. |
| 2026-10-19 | Accepted C29 queue source/CPU and C30 policy oracles, committed C31 bootstrap/inventory and submitted exact original-V4 diagnostic; native/formal outcomes remain separately gated. |
| 2026-10-19 | Accepted C25/C26/C28 source and CPU slices, retained completed one-H800 GLM graph evidence and C27 workflow deltas; native/formal gates remain open. |
| 2026-10-19 | Accepted scoped C23 cache-zero and C24 DP8 CPU evidence; synchronized remaining native inputs, ownership and formal limits. |
| 2026-10-19 | Recorded accepted C21/C22 scoped commits and tests, cache-zero/DP8/native/CPU-input gates and external Step ownership; corrected formal H200 freeze versus authorized quota-fallback tests. |
| 2026-10-19 | Synchronized accepted C19/C20 commits, final source/CPU evidence, GLM/V4-only ownership and H800-first fallback; all native/formal limits remain explicit. |
| 2026-10-19 | YC clarified this worktree owns only GLM-5.3 and DeepSeek-V4; stopped C18 Step integration, preserving its uncommitted evidence for scoped withdrawal. Step development belongs to its separate worktree; cross-model reuse is coordination only. |
| 2026-10-19 | Accepted C17 dc7f382 after exact26-file identity,1230 focused tests, actual43-layer offline/online conservation, independent APPROVE and77 identical legacy artifacts; began single-owner accepted Step representation integration. |
| 2026-10-19 | Accepted C16 at2da11c8 after final15-file identity/review and actual structural/legacy evidence; C17 validation remains open. |
| 2026-10-19 | Synchronized C16-R2 legacy comparison, C17 request-quota review finding and bounded native prerequisite audit; acceptance remains pending. |
| 2026-10-19 | Recorded rejected C16-R1 liveness counterexample, required new-generation verification, explicit C17 idle-DP source branch and accepted Step integration dependencies. |
| 2026-10-19 | Accepted C15 source-rank component boundary and GLM source-structural/native lifecycle milestone separation; formal gates unchanged. |
| 2026-10-19 | Started C15 source-rank SP/mHC/shared execution consumer with one owner; corrected padding geometry versus selected routing-mask evidence. |
| 2026-10-19 | Recorded accepted C14 596451a eager attention component; full V4 workflow/native/formal gates remain open. |
| 2026-10-19 | Read back accepted C13 boundary and latest resource rule; full model/structural/native/formal acceptance unchanged. |
| 2026-10-18 | Recorded accepted C13 source TP-SP payload/RS dispatch scope; preserved full V4 execution, native and formal gates. |
| 2026-10-18 | Recorded accepted C12 static source tensor dimensions; dynamic workload/execution and native/formal gates remain unchanged. |
| 2026-10-18 | YC restored H800-first GPU submission; H200 is the insufficient-quota fallback, aggregate<=64GPUs and non-preemptible. |
| 2026-10-18 | Synchronized accepted C07-C09 source/cache and C11 unified-role storage boundaries; V4 operator/precision inventory does not establish execution or measured support. |
| 2026-10-18 | Corrected historical GLM binding limitation after accepted C02/C04; retained measured/native-chain requirements and all formal gates. |
| 2026-10-18 | Reconciled approved cache scope and temporary resource route; representation/structural/native/calibrated evidence remain separate. |
| 2026-10-18 | Recorded temporary H200 tests<=64GPUs/non-preemptible and approved existing-manager grouped cache extension; H800 formal objectives and native qualification remain unchanged. |
| 2026-10-18 | Applied the documentation review: representation before measurement, structural versus measured milestones, three-model ownership and current approval policy; formal accuracy gates and reference locks retained. |
| 2026-10-15 | Created from spec v2, the campaign YAML, and grilling rounds 1–2. |
| 2026-10-15 | Made the language-only scope explicit (Q11): vision and other multimodal paths are excluded. |
| 2026-10-15 | Review remediation: GLM-5.3 full-path boundary (Q12) and per-model decision stages (Q13); validated probe output before formal use (S2); separate request E2E and request-throughput checks (S4); actual baseline of P2P latency hiding (C2); owner and symbol corrections (C1, C3). |
| 2026-10-15 | Re-review fix R4: per-layer attention identity is resolved in `model_binding.py`; `_assemble_stage` consumes it. |

Binding source: `source_spec_v2/Frontier_GLM53_DeepSeekV4Flash_vLLM028_H80064_Spec_v2.md` (cited below as "v2 §N") and `source_spec_v2/Frontier_vLLM028_H80064_Campaign_Template_v2.yaml`. This file is the English synthesis; where they differ, the deltas in [Deltas against spec v2](#deltas-against-spec-v2) and the decisions in `requirements.md` win. Terms follow the root `GLOSSARY.md`; the ground-truth choice is recorded in `docs/adr/0001-dummy-weight-native-ground-truth.md`.



## Current scope override — D-DSV4-ONLY-GLM-PAUSED (2026-10-22)

Current execution status is in the latest [main progress checkpoint](progress.md#final-execution-checkpoint--current-2026-10-22)
and current execution sections in tickets [05](tickets/05-dsv4-flash-family.md)
and [18](tickets/18-native-ground-truth-h200.md). Historical execution paragraphs
below do not replace those records or alter the current scope and rules.

### Historical execution snapshot — superseded (2026-10-22)

Final execution state supersedes the older source/package snapshot below. GLM06
returned PAUSED_USER_APPROVAL_REQUIRED,no fresh06/tests/build/seal/child processes;
old packages preserved and personal pagination found no GLM jobs needing stop.
V4 warm-q1 exp-1022-003156-993689 is observed Running,1H800/codesign/non-preemptible,
identity accepted but hardware business/rows PENDING. Its first preflight failed
BEFORE_SUBMISSION because aggregate ceiling32 rejected Step32+requested1;new
descriptor ONLY aggregate_H800_ceiling32->128,preserving527 pins.
Native candidate04 SEALED_NOT_ADMITTED_NOT_SUBMITTED after full-checker staged
PYTHONPATH/checkout mismatch and missing67th breakable_cudagraph.py pin. Candidate05
UNSEALED and VERSIONED_CHECKER_FIX ses_ea0af5ae9ffe20NcnuumIBrR5v remain pending.
Old5efd/9116/candidate04 immutable;no04 READY_TO_LAUNCH. Final NEW freeze/seal only
after ALL_COMPUTABLE_FULL_CHECKER_PREDICATES_PASS and all writers' no-more-writes.
No native/business/calibration PASS is inferred;DSV4-only/mean/30%/10% unchanged.

### Current scope and acceptance rules

Current correction,simulation,testing,calibration and acceptance cover DSV4 only.
All GLM tiers' simulation/tests/calibration,new package preparation and FULL
KERNEL_ONLY proposal follow-up are PAUSED_PENDING_USER_APPROVAL until a new
explicit user approval. Preserve prior code/data/receipts/UNVERIFIED metrics;
no deletion,false PASS or retrospective native admission. The GLM measurement
proposal stays parked,not a current next approval question. Dual-model and
GLM-first execution/completion directions below are retained scope/history,not
current dependencies or authority. DSV417 directly reuses accepted68f891b;
paused04/16/GLM bundle/full path never blocks05/09/10/11/17/18/19 acceptance.

For DSV4,the technical workload/runtime/coverage contract is unchanged:128
formal6144/1024 requests,QPS1,16 independent warmups and3 clean repeats. P0
H80032,P1 H80064,P2 H20032/H20064 retain hardware-specific inputs;formal64 needs
own basic validation and same-model/hardware32-GPU every error<=30% in every
admitted repetition. Finalall-four<=10%. Personal H200 non-preemptible aggregate
<=32;H20064/additional H200 preemptible. Mean TTFT/TPOT/E2E and total THPT remain
official;P90 is recommendation only,not approved implementation.

### Historical source/package execution snapshot — superseded (2026-10-22)

Source freeze7c1b888/checker5efd00ea…f37ec36 and TCP30db587/combined288PASS25SKIP
are accepted software evidence,not native admission. DSV4 successor04 preparation
and primary's failed warm-q1 launch diagnosis continue;GLM06 was told stop with
handoff pending. Keep historical9116ab7 immutable and require exact new checker/
source pin checks before/after seal. No POST retry or new runtime approval here.

## Current execution and input contract — 2026-10-31

The latest requirements decisions supersede older execution snapshots: D-H200-MAINLINE-CONTINUE keeps DSV4 H200 P2-32/P2-64 on the mainline with no twelve-hour end time; D-CODEX-DIRECT-RJOB uses direct Codex StepMind Python RJobBackend. H800 remains subsequent hardware scope and GLM is paused. The current workspace rule limits new personal non-preemptible H200 reservations to 32, including queued jobs.

H3o costs are composed from operator measurements and communication costs; whole-step tables remain diagnostic only. Decision 46 permits measured communication, which remains the formal input. F1–F4 are diagnostic comparisons; promotion of a variant to a formal input requires a new YC decision.

Ingress option A is implemented (B29, a69a5213): 10.3 ms delays engine arrival while preserving client arrival; 11.4 ms delays the KV handoff start. Both code defaults are zero. Three-repeat client TTFT/TPOT/E2E and total THPT gates are unchanged. P2-32 B29 is a numerical PASS; P2-64 repeat 1 is FAIL at 12.25494% TTFT even though pooled TTFT is 7.67691%. Fresh A6 completed on November 1 but failed qualification (four attention scopes and seven of eight uniform collective curves). Full operator qualification and hardware-specific NIXL/memory coverage remain open. The selected A6 repeatability probe completed128 valid steps/640 rows and showed nonpersistent CSA contrasts, without qualifying a new measurement generation. F1a is complete at both scales; P2-64 fails TTFT/TPOT/E2E against all three clean repeats. These diagnostics do not replace formal inputs; see [repeatability](test_report_2026-11-01_a6_repeatability.md) and [F comparisons](test_report_2026-10-31_f_variant_comparison.md). F1b held-out validation fails; both replays also terminate on compute COVERAGE_MISS and provide no complete E2E reading. Completed F3 NIXL measurements have an invalid affine fit; they provide no replacement input. See [A6 recovery](test_report_2026-11-01_a6_report_recovery.md) and [F3 measurements](test_report_2026-11-01_f3_h200_nixl.md).

The priority table and dated source snapshots below are historical where they disagree with this section. The common fixed workload, metric definitions, three-repeat admission, hardware identity and no-step-cost contract remain binding. See [current plan](plan.md#current-execution-route--codex-cli-h200-operator-path-2026-10-31) and [Codex test report](test_report_2026-10-31_codex_resume.md).

## Retained prioritized technical contract — hardware priority superseded by the current route

Authority: the latest D-CALIBRATION-PRIORITIES request in `requirements.md`. It supersedes the hardware/scale/priority portions of the earlier `/data/ycfeng/Frontier/.local-draft/new-models/calibration.md` adoption. The fixed workload and unchanged acceptance definitions remain binding. Historical inputs and results are retained, not rewritten.

### Priority tasks and dependencies

Each row is tracked separately for GLM-5.3 and DeepSeek-V4-Flash. Tickets10/11 own the priority milestones;16/17 bind model cases,18 collects native evidence and19 evaluates promotion/final acceptance.

| Task | Priority | Total / P+D | Per-role TP/DP/EP/PP | Resource class | What may start now | Formal calibration gate |
| --- | --- | --- | --- | --- | --- | --- |
| P0-H800-32 | P0 |32 /16+16 |8/2/16/1 | H800/codesign, non-preemptible | All ready prerequisites and calibration | Own admission/coverage; goal all four<=10% |
| P1-H800-64 | P1 |64 /32+32 |8/4/32/1 | H800/codesign, non-preemptible | Basic running validation and preparation | Basic validation plus same-model P0 all four<=30% |
| P2-H200-32 | P2 |32 /16+16 |8/2/16/1 | H200/step_main, non-preemptible | Own prerequisites/calibration within personal32 budget | Own admission/coverage; no H800 accuracy edge |
| P2-H200-64 | P2 |64 /32+32 |8/4/32/1 | H200/step_main, entire job preemptible | Basic environment validation and preparation | Basic validation plus same-model H20032 all four<=30% |

For each model independently:

```text
P0-H800-32 admitted comparison (all four <=30%) ─┐
P1-H800-64 basic validation ────────────────────┴─> P1 formal calibration
P2-H200-32 admitted comparison (all four <=30%) ┐
P2-H200-64 basic environment validation ────────┴─> P2-H200-64 formal calibration
```

The basic validations and preparation branches can run in parallel before the
30% thresholds. Resource priority is P0>P1>P2; it is not an invented
P0-complete→all-H200 dependency. A higher-priority ready task takes shared
resources first. Both H200 tasks remain priority P2.

The30% receipt is case-bound and admission-valid, with all four finite errors
individually<=0.30 in each of the existing three clean repetitions. One passing
metric, an average across metrics, a mock fixture, missing coverage or another
hardware/model does not unlock it. Final acceptance stays<=10% for every metric;
a32-GPU result in(10%,30%] may unlock64 GPUs without closing the32-GPU task.
The64-GPU formal native reference campaign itself is gated; early environment
checks must retain `basic_validation` purpose and cannot be relabeled formal.

### Common case contract

- **Objective.** One fixed-workload case per model for each of the four priority tasks, using only the two specified scales and hardware. Every completed calibration has four independent<=10% errors. No additional workload/QPS sweep or historical36-cell/holdout matrix is revived.
- **Platform.** Use the task row above. P/D disaggregation, online serving, thinking/prefix/chunking disabled, native CUDA Graph and quantization enabled, and dummy weights with no checkpoint download remain unchanged. Existing H800/H200 evidence is reusable only within its actual hardware, topology, runtime and measured scope.
- **Aggregate resources.** Across all sessions, i-fengyicheng may reserve at most32 non-preemptible H200 GPUs, including queued jobs. All additional H200 submissions must be preemptible; every H20064 job is wholly preemptible. P2-H200-32 jobs wait for non-preemptible budget when needed. H800 jobs in both priority rows are non-preemptible. Check actual quota and full requested reservations before submission; do not use the historical70/80/130-card clauses. Retain interrupted preemptible attempts and begin fresh complete repetitions instead of stitching partial runs.
- **Workload.** Exactly **128 formal requests per model and per repetition**, each with **6144 input tokens and 1024 output tokens** (the working token-count interpretation of 6k/1k is `k=1024`). Arrivals are fixed and open-loop: request ordinal `i=0..127` is scheduled at `i` seconds, QPS **1.0**, so the arrival span is **127 s**. There is no Poisson generation, SharedGPT selection, trace-gap retiming or target-tokenizer re-count. Prompt token IDs must be deterministic and valid for each model; the native request must preserve the exact counts without chat-template expansion, truncation or early output termination. Warmups, repeated clean runs and diagnostics are not additional formal cases.
- **Artifact identity.** The common workload is the exact **68f891b** package at `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/`, with `trace.csv`, `request_ids.json`, `warmup_pool.csv`, `manifest.json` and `SHA256SUMS`. YC also selects it for future Step-5 adoption (D-COMMON-WORKLOAD-68F891B). Manifest SHA256=`5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3`; SHA256SUMS-file SHA256=`594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9`. Every formal package must stage all five files byte-for-byte in the existing runner's trace directory; matching counts alone is insufficient. Formal and warmup identities are disjoint; **16 same-shape warmup requests at QPS 1.0** precede each repetition and are excluded from formal totals. Stable template IDs determine prompt content; repetition-qualified transport IDs must not change inputs. Formal totals per repetition are786432 input,131072 output and **917504 total tokens**. Artifact/consumer verification does not establish native completion or all-lane graph coverage.
- **Superseded SharedGPT history.** SharedGPT is deferred, not a second formal case. The old `data/processed_traces/canonical_sharegpt_req256_qps4/` freeze at `9ed979e`, source trace SHA-256 `fad744ff3b25bff1e164cc210af4f29176e8d57d1be14c4f981a10bc32ddd61a`, 256-request/QPS-4.0 linear rescale `2.590246128`, distribution reports and reproduction evidence remain unchanged. They do not satisfy the new workload contract. The earlier head64/head128 alternatives and claims of workload identity with Step-5-Preview are superseded here; this worktree neither edits nor asserts synchronization of that separate worktree.
- **Topology.** Both models use the corresponding32- or64-GPU row. The32-GPU mapping reuses the already-approved P16+D16, TP8/DP2/EP16/PP1 layout. A native refusal is a typed blocker with evidence; do not silently downgrade a named priority task.
- **Metrics and the four independent gates.** TTFT is client-observed (request submission to first output token); internal TTFT (P-side prefill completion) is diagnostic. TPOT is the mean of each request's `(E2E − TTFT) / (output_tokens − 1)`. E2E is client submission to that request's last output token, not workload makespan. THPT is formal prompt-plus-output tokens divided by elapsed time **from the first formal submission through the last formal last-token completion**: `917504 / (last_completion − first_submission)` for a complete repetition. The 127 s arrival span is **not** the THPT denominator; warmup, startup and inter-repetition idle time are excluded. All 128 requests must complete with exact lengths; failed/unfinished requests stay recorded and invalidate the reference rather than being dropped. Client latency means and per-window THPT are compared with the identically aggregated native values under four independent `abs(frontier − vllm) / abs(vllm) ≤ 0.10` gates. p50/p95, output-only throughput and request/s do not gate.
- **MTP follows the native default, which is disabled for both models.** Both configs declare `num_nextn_predict_layers: 1` (GLM-5.3 also sets `index_share_for_mtp_iteration: true`), but neither declaration enables speculation: `EngineArgs.speculative_config` defaults to `None` and `create_speculative_config` returns `None` unless `--speculative-config` is given or the HF config is a `SpeculatorsConfig` (`vllm/engine/arg_utils.py:1047-1085` in the locked runtime). Frontier keeps MTP disabled for both models, and no MTP implementation, profiling or calibration work is created. The same principle governs the effective CUDA Graph mode and quantization scheme.
- **Communication.** The measured backend stays in use. Every touched collective/NIXL shape needs a qualified curve for the case's actual hardware, topology and lane widths; missing coverage remains a typed failure. No H800/H200 scaling or cross-hardware table relabeling. `collective_sim` remains available but is not substituted for the agreed backend.
- **Profiling is limited to touched shapes.** Reuse already qualified hardware-specific rows first. GLM needs both DSA variants plus MoE/linear/collectives/NIXL; V4 needs ratios0/4/128 plus those other domains. Cover P6144/C0 and successful full-hit Dq1 cached6143..7166/attention6144..7167, compression/page/top-k boundaries and the actual graph buckets, separately for DP2/EP16 and DP4/EP32 where shape differs. Returned7168 is distinct from computed KV7167. First-tail, compiled and physical-TP qualification are not waived. With chunking off, max_model_len>=7168 and max_num_batched_tokens>=max_model_len; effective limits/memory still require native evidence.
- **Non-KV overhead.** Bind actual weight, activation, graph and non-Torch terms to each hardware/topology. H200140GB observations do not freeze H80080GB capacity, and changing DP/EP requires its own binding. No zero-byte default or arbitrary scaling factor.
- **Delivery order.** Reusable support, profiling and native preparation proceed in parallel. Execute the priority/dependency graph above; within each eligible case, bind native admission and non-dummy execution before comparison/final acceptance. Report every(model, hardware, size, purpose) separately.
- **Session coordination.** Step-5-Preview adopted the same68f891b artifact as958d81a. Primary read-only inspection verified all five package files byte-identical; its native consumer/runtime qualification remains separately owned. Shared changes require Q10 approval; D-Q10-CONTEXT and D-Q10-CLIENT-METRICS explicitly approve the bounded existing-frontier context correction and both-sided client-TTFT binding. Other shared changes remain subject to Q10.
- **Repetitions and coordination.** For each model, run **3 clean repetitions inside one job**, each with its own independent warmup and formal measurement window, all 128 formal requests completed, and both roles back at idle between repetitions (no running/waiting requests, KV usage back to zero). Retain per-repetition boundaries, counts and four metrics; no successful subset substitutes for a failed repetition. Step-5-Preview remains separately owned; shared artifact identity does not prove native runtime equivalence. The measured communication backend here remains the deliberate contract choice.

## Mainline recovery decisions — 2026-10-20 (superseded where they conflict with the canonical contract above)

Source: YC adopted [the mainline review](review_report_2026-10-20_mainline_and_tickets.md); decisions are recorded verbatim in `requirements.md` (2026-10-20 entry).

- **Main path (superseded 2026-10-21 by the canonical contract).** The objective was predictive online PD for both original models on 64 H800, GLM first through the trained path, with `04/05.primitive_profiles` and `06.raw_measurements` on H800 as the next milestones. The platform and milestone set are now H200; the sequencing principle (work that does not move a dependency-graph milestone is deferred, `D-MAINLINE-LOOP-CAP`) still holds.
- **G0-B.** G0-B proves that the full structure runs on this runtime. It no longer includes the native routing occupancy sample, which moves to `03.routing_inventory` and ticket 14 (`D-G0B-ROUTING-DECOUPLE`). Existing evidence closes the remaining four items for both models ([reconciliation](g0b_reconciliation_20261020/report.md)).
- **Observer qualification.** C36 light-mode perturbation stays FAIL at the unchanged 3% limit. Observer qualification is frozen (`D-OBSERVER-FREEZE`); it gates formal cost replay and formal E2E references (07 approved PD integration, 09 cost replay, 10), not profiling, predictor training or structural/trained simulation.
- **Profiling producers are required work.** The selected native backends are known: GLM uses `FlashAttnMLASparseBackend` plus the DSA indexer; V4 uses FlashMLA sparse/SWA plus compressor/indexer. Their profiling producers do not exist yet and must call the v0.28.0 classes picked by the resolved backend, reusing the existing producer/importer/trainer contract.
- **Sequence-parallel MoE storage and shapes.** With vLLM v0.28 defaults (`all2all_backend=allgather_reducescatter`), EP on, TP>1 and DP>1 enable sequence-parallel MoE: shared experts are replicated on every rank and run on token-sharded input. Both model profiles must represent this through the existing `replicate_shared_expert_in_sequence_parallel` contract; parameter memory and shared-FFN profiling shapes follow it.
- **Fidelity of unchanged features.** A regression in an existing feature is fixed, not relabelled as inherited. An "inherited failure" claim requires reproduction on the original base `bfa82de`.
- **Resources (historical).** The earlier70-H200 ceiling and H800-first fallback rule are superseded for current H200 scheduling by D-H200-LIMIT-130:130 aggregate,64 per formal model case. Historical job allocations and receipts retain their original values.

## Historical original-GLM PD and C40 evidence boundaries (H800; current contract above controls)

The original objective remains predictive online PD for both original models on 64 H800, with GLM first for the complete trained path. Reduced G0-A[DSV4] is PASS from C32's seven native checks and MARLIN02's two exact positives/six rejected controls with successful terminal records. This closes only the original diagnostic criteria; [the disposition](v4_marlin_control_20261019/test_report_2026-10-19_v4_marlin_control.md) preserves full-model G0-B/C/D, observer and trained/formal gates. Historical C31/MARLIN01 FAIL is unchanged.

C33's explicit v0.28 policy is implemented through the existing balancer with the v0.10.2 default preserved. Its unchanged 77/77 legacy comparator PASS applies only to clean candidate `563dfff`, not later HEADs; two inherited placement-test failures remain. This is source/CPU evidence, not native P/D placement or Ticket15 disposition. [Verification](c33_dp_policy_20261019/report.md).

C34 [GLM attempt02 review](c34_native_queue_20261019/actual_native_review_02/test_report_2026-10-19_glm_native_queue.md) passes 27 selected checks for capacity two, FIFO, original output-parent retention and graph host returns. It preserves the shutdown-time API `EngineDeadError`, incomplete startup worker evidence, separate clock domains, and unobserved native arrival/device overlap. It does not qualify observer tooling, settle D-ASYNC, or supply numerical producer inputs. GLM attempt01 remains FAIL for missing traces.

C35 `8d479ef5b232f198035059441afc33bd88e9fc68` is source/CPU accepted through the existing V1 scheduler. `max_concurrent_batches=None` preserves physical pipeline capacity and legacy serialization; explicit two is restricted to the exact V1 scheduler, MONOLITHIC PP1/DP1 ordinary cache without prefix caching, speculation or GDN. Physical stages stay serial and grouped/disaggregated/subclass guards remain. Root validation is 135 PASS / 7 inherited FAIL, with matching failed IDs and normalized headings on clean `da13911`; the unchanged comparator passes 77/77 for clean source `8d479ef` and the byte-identical baseline comparator in harness `da13911`. The 902 artifact files and 519 actual cache filenames per side were independently inspected; this is normalized comparator equality, not cache binary equality. [Evidence](test_report_2026-10-19_queue_admission_full_v4.md).

Full original V4 `exp-1019-172330-858927` completed on 16 H800 with both containers exit 0/restarts0. Actual inventory covers all 16 ranks at 43 layers/256 experts; four prompts generate four tokens each with finite selected logprobs. Available inventory/generation checks pass, while logs retain EngineCore force-kill/worker SIGTERM and only 16 worker traces with no EngineCore trace. Complete observer/G0-B remains OPEN; strict routing stays OFF. C36 five-pair `exp-1019-181346-562466` has ten eligible cases and no integrity errors, but perturbation FAILS the frozen 3% criterion: all 3.1641677477%, long 3.3022346216%, short 8.3335401304%. No samples, pairs or threshold changed. Finite CPU diagnosis does not isolate the cause or justify a recorder rewrite or another reduced GPU attempt. C38 native `6a8a5a09acd93a07ff3bb9d3797cdb1162bd0520` is source/CPU accepted (67 root/owner tests and five rejected source mutations), and actual reduced H800 validation passes 46 root/51 independent checks: 22 complete boundary sets, ten FULL replays and six HTTP requests. Actual EngineCore/worker boot, time namespace and CLOCK_MONOTONIC match; host-clock comparability is accepted only for those two processes. Absolute GPU endpoints, perturbation and complete CPU pricing remain unqualified. Startup13 and teardown EngineDeadError remain. Original C38/full-V4 launcher exit codes are unavailable after interruption; platform/container success is separately verified. [Full checkpoint, commands, jobs and limits](test_report_2026-10-19_c36_c38_full_v4.md). The original 78-layer GLM functional PD run now supplies the separate current evidence below; complete G0-B/C/D, trained GLM PD, D-ASYNC/D-DP-PD and formal delivery remain open.

C39 deployment source/CPU remains accepted at Frontier `9069538`, with 156 root tests and independent reviews. Complete C40 lifecycle/frontend source is accepted at native `000702d340a7508b926e43e9f8c1bfea21e686f1`: corrected A1, independent A2/A3/B reviews, final owner 78-test suite and root 320-comparison frontend differential. Bootstrap `89290df` and pressure harness `6c3acc0` are separately accepted. Both final original-GLM inputs passed independent readiness review and were frozen with 36 exact files each; the old `6a8a5a09` check-only subset remains historical. The freeze receipt records no GPU submission; root owns subsequent runtime evidence. Functional totals 2400 prompt/32 output tokens; pressure totals 7552/640 with only D overridden to 17 native blocks.

Functional job `exp-1019-222513-665277` completed on four eight-H800 nodes with the original 78-layer/256-expert GLM, P16+D16 and TP8/DP2/EP16/PP1 per role. All eight requests returned HTTP 200 with 2400 prompt and 32 output tokens. Platform Succeeded at 2026-10-19T14:34:37Z; four containers and four native parents exited 0 with no restart, and the launcher exited 0 at 14:34:39.648555Z. The functional job and the separate pressure02 job are both terminal. Their business results are accepted separately from full native qualification.

Post-shutdown collection contains 40 matching files: four frontends, four EngineCores and 32 workers. Actual `api_server_count=2` per role supersedes the planned 38-process count. The C40 review is `PARTIAL_C40_HOST_EVIDENCE_TRANSFER_OBSERVATION_FAIL`: eight D reservation/wait/receive/application/promotion chains, eight P notification/free chains, 16 immediate physical frees and 16 frontend choices/releases are observed. All 16 D workers lack successful prepare/submit/DONE/release records because opaque `nixl_xfer_handle` values failed JSON serialization (320 missing positions across 95 intervals). All 40 processes retain `run_verified=false`; no full transfer or observer qualification is claimed.

The minimal handle repair at native `169309961db88821211149ff3fafca275f29f816` is independently APPROVED for source/CPU and pressure02 inputs: the 80-test suite and 17 actual-body CPU cases pass, with the generic JSON serializer unchanged. Pressure02 is frozen with 36/36 exact files and config SHA-256 `9100e06faca15a6315b5b7b209f7e05f937a227738f1d367df3c7a1463fa53d1`; the real SDK payload check passes. Its eight requests/7552 prompt/640 output tokens, D17 including null, P profiled/default capacity, model and runtime stay unchanged. The existing serving buffer input moves from 4 MiB to 8 MiB, based on the retained 5,572,856-byte estimate if all 640 outputs use one lane; this is not a guaranteed bound. Bootstrap startup sub-budgets remain 12 records/65,536 bytes, and serving startup_memory is None both before and after. The new pressure run now supplies the actual drops/gaps and handle host-lifecycle checks below; source/input acceptance alone did not establish them. Old functional01/pressure01 inputs and historical FAIL remain immutable.

Pressure02 job `exp-1019-232508-663845` completed on four eight-H800 nodes with the original GLM, P16+D16 and TP8/DP2/EP16/PP1 per role. All eight client responses and sixteen P/D upstream responses are HTTP 200, with 7,552 prompt and 640 output tokens. Platform finish is **2026-10-19T16:07:17Z** (October 20, 00:07:17 Asia/Hong_Kong); four native parents and containers exit 0 without restarts, and the launcher exits 0 at **16:07:18.177688Z**. The watcher timestamp 16:10:19.282313Z is later observation, not completion. The initial Starting receipts are historical; no launcher or pending allocation remains for this run.

The independent pressure review passes **355/355 retained-artifact checks**, not source tests. It accepts five same-owner refusal → prior physical free → readmission witnesses across both D lanes, with 422 native allocation refusals and eight successful reservations. D capacity is 17 blocks including null: each admitted prompt reserves fifteen of sixteen usable blocks (93.75%), and each request owns all sixteen at final free. All 64 worker transfer incarnations contain ordered serializable prepare/submit/native-DONE/handle-release host records; eight D scheduler chains, 64 P notification/completion pairs and sixteen frontend route/release chains are retained. These new observations close the pressure and repaired-handle **host-evidence slices only**; the original functional transfer-observation FAIL remains immutable.

Post-stop pressure collection covers 40 processes with contiguous sequences, zero recorded drops or observer/footer errors, and no declared pending records. All forty footers retain `run_verified=false`; all 32 worker manifests are false, while eight frontend/EngineCore manifests omit the field. P848/D5744 missing-parent counts, `host_evidence_incomplete=true`, and 64 `receive_report` rows with unknown source/null source step are preserved. Clock metadata agrees within each of four nodes, whose boot IDs differ; cross-node subtraction and absolute device readiness remain unqualified. Native logs retain teardown EngineDeadError and worker SIGTERM warnings. Raw cloud traces remain in place; the selected local snapshot is not a new per-layer weight audit. [Actual pressure review](c39_glm_full_pd_20261019/pressure_actual_review/test_report_2026-10-19_pressure_actual.md), [355-check receipt](c39_glm_full_pd_20261019/pressure_actual_review/verification_receipt.json).

[Independent acceptance](c39_glm_full_pd_20261019/handle_serialization_repair/independent_review/receipt.json), [new frozen inputs](c39_glm_full_pd_20261019/pressure_frozen_02/receipt.json), [real SDK payload](c39_glm_full_pd_20261019/pressure_payload_02.json), [initial platform receipt](c39_glm_full_pd_20261019/pressure_platform_01.json), [15:33:25 UTC platform receipt](c39_glm_full_pd_20261019/pressure_platform_02.json), [launcher start](c39_glm_full_pd_20261019/pressure_launcher_started.json).

[Actual C40 review](c39_glm_full_pd_20261019/functional_c40_review/test_report_2026-10-19_functional_c40_actual.md), [business/platform review](c39_glm_full_pd_20261019/functional_actual_review/test_report_2026-10-19_glm_full_pd_actual.md), [repair evidence](c39_glm_full_pd_20261019/handle_serialization_repair/test_report_2026-10-19_nixl_handle_serialization.md), [pressure02 capacity](c39_glm_full_pd_20261019/pressure_02_capacity.json).

C36 perturbation remains FAIL at the frozen 3% threshold; full G0-B/C/D, schema/probe qualification, D-PD/D-ASYNC/D-DP-PD, trained GLM PD and formal delivery remain OPEN. Reservation refusal, lease expiry and deferred free were not exercised by the functional run. No cross-process elapsed-time, device-completion or calibrated-cost claim follows from the valid host rows. [Acceptance and dependencies](test_report_2026-10-19_c39_c40_glm_pd.md).

## Historical C29–C31 evidence and planning corrections — superseded above

C29 native `09c802c1a` adds actual batch-queue occupancy/capacity/FIFO IDs and host Future wait boundaries through the existing recorder; root46source/CPU tests pass. This establishes neither GPU completion nor native overlap. Carry actual parent identity on an async output object before observing return from its existing ready-event synchronization; do not use a later caller thread's ContextVar as a parent. Native queue comparison precedes removing the grouped-cache single-flight guard or changing scheduler/frontier semantics.

C30 source-body oracles distinguish online internal `DPLBAsyncMPClient` from offline `LLM`/`SyncMPClient`. A legacy-default explicit policy selection is the smallest proposed existing-balancer extension; no second router or automatic model-name binding is justified. Native load reports sample KV pressure only when waiting/running counts change; snapshots retain that value. Map null/capacity meaning explicitly rather than subtracting one from every existing capacity. Real Request ID ownership survives stale reports, chunks and preemption, and releases once on actual terminal output. This source evidence does not decide Ticket15 P/D placement or close native G0-C/D.

C31 Frontier `587ec80`/`005d5cf` extend the existing exact overlay and diagnostic inspector. Original-V4 mHC/hash coverage is non-vacuous; unimplemented MXFP4 packed/scale coverage stays invalid even if parameters disappear during post-load. The selected seven-file C26 deployment excludes C29. One original reduced V4 diagnostic on1H800 completes generation/shutdown but fails two inventory checks; complete G0-A remains OPEN. All resource, variant, strict-routing-OFF and formal gates are unchanged. [Exact current verification](test_report_2026-10-19_native_queue_v4_g0a.md).

C31 actual original-V4 uses V2 Model Runner and MARLIN MXFP4; V1 runner internal hooks cannot establish its internal coverage. Native prefix caching resolves to the unchanged defaultTrue; Frontier V4 prefix caching remains unsupported. The first-forward blocker is crossed, but twelve scale rows/four quant auxiliary inventories remain invalid; do not label the Failed diagnostic as G0-A acceptance. Selected backend/runner evidence now narrows the next native/inventory work.

## Historical C25–C28 evidence and planning corrections

The current source/CPU and reduced native checkpoint is recorded in [test_report_2026-10-19_native_inputs_dummy_init.md](test_report_2026-10-19_native_inputs_dummy_init.md). C25 observes three actual real-forward FULL replay host returns with parent/scope identity on one H800; it does not provide GPU completion/duration, startup capture JSONL, cross-DP agreement, clean/light perturbation or complete model/probe qualification. Keep the diagnostic worker's incomplete flags and the historical missing-EngineCore cause UNKNOWN. C26 repairs original-V4 dummy auxiliary derivation through its existing post-load hook; native V4 G0-A remains OPEN. C28 extends the existing exact deployment allowlist to six files rather than creating a second deployment path. The GLM diagnostic installs the C25 six-file subset only, excluding C26.

The [C27 source comparison](c25_native_progress_20261019/c27_audit/report.md) corrects planning assumptions: selected native GLM PP1 async permits two queued batches, while Frontier currently permits one; v0.28 internal DP selection includes client outstanding work, KV pressure and rotating ties instead of the legacy policy. Existing queue/output/settled-frontier and balancer owners are the first extension candidates. Retain grouped in-flight and placement guards until their replacement behavior is verified. Do not silently approve a new core framework or replace historical policy assertions.

Graph execution requires resolved captured descriptors and applicable DP agreement; a graph flag or dispatch intent is insufficient. PP3 zero-token queue fill is conditional and not an immediate PP1 blocker; cadence>1 is deferred unless selected. Interval32 is source-matched. Scalar wave/report mapping needs evidence, not a presumed representation error. C16 already represents D reservation before READ; D-ready then P-release at one timestamp remains a labeled structural approximation until Ticket12 lifecycle evidence. Ticket13/15 native decisions and formal gates stay open. The next priority is qualified native queue/device/output/clock/READ evidence and original-V4 G0-A, ahead of numerical calibration. This worktree owns GLM/V4 only; resource and dummy-weight requirements remain unchanged.

## Historical execution checkpoint — C23/C24 accepted source and CPU structural slices

Commit `88c030fdfee25adf49fbf7718a7166cc09e37052` adds actual grouped-allocation assignments -> immutable Batch metadata -> one synthetic before-forward cache-zero scope, and source-backed TP8/DP8/EP64 admission through the existing parameter and structural-driver guards. Root isolated suite: 1,292 tests across 64 modules passed in 23.40 seconds. Full 43-layer DP8 offline and online each complete 8/8 requests on 64 logical devices; DP2 online prompt/chunk/decode 2051/2050/3 (pool 1024) completes 2/2. Actual allocation/handoff/trace, stages and EP phases conserve. Unchanged 77-case artifacts and predictor-cache filenames match; comparator exit 1 is solely modified-source provenance at parent b5120e6, not clean-commit/full-comparator PASS. [Exact verification](test_report_2026-10-19_dsv4_cache_zero_dp8.md), [final source manifest](c23_cache_zero_execution_20261019/combined_candidate_manifest_r2.json), [acceptance receipt](c23_cache_zero_execution_20261019/acceptance_receipt.json).

Cache-zero IDs retain raw group-major occurrences, including assigned-then-preempted requests; predictor queries do not drain allocator state. Layer 0 is an existing timing anchor for worker/step work, not a claim of transformer-layer ownership. Counter values in actual reports describe representative stage/worker anchors, not aggregate TP kernel launches. Distinct physical reservation extent is not native bytes zeroed, stores, H2D traffic or measured bandwidth. Dummy duration is one invocation without byte/layer scaling. Normal V4 measured/profiling/runtime guards remain closed; grouped PP1 zero-only outputs, complete idle-attention traces, native cache contents, graphs/PD, trained and formal support remain open. The empty-step source argument is conditional, not execution proof.

Disjoint actual-recorder-to-existing-collector scratch verification passes 7/7 CPU cases, all with qualification false. It shows no collector defect requiring another pipeline. Historical EngineCore absence staysUNKNOWN and real native lifecycle/collection coverage remainsOPEN. Required device-completion, actual graph and READ-ready causal inputs are absent/unqualified; native input hooks precede one adapter in the existing CPU producer. Routine native instrumentation/tests are already authorized; these are technical prerequisites, not new blanket approval gates. [Collector/input report](c23_cache_zero_execution_20261019/collector_readiness/verification_report.md).

This worktree owns GLM-5.3 and original DeepSeek-V4-Flash only. Step development/integration remains external. H800 first; insufficient-quota H200 fallback, aggregate task GPUs <=64, non-preemptible. No new GPU/native/profiling/calibration campaign, checkpoint download, push or main merge occurred. Ticket05 remains PARTIAL; no native/formal model support is claimed.


## Historical C21/C22 checkpoint — superseded by C23/C24 above

C21 is locally accepted on the native development branch at `a589f86bd6f080e28064b83df3d91d2a26a4061d`, parent6c523e853. The existing recorder reports selected path/init/close/write diagnostics; optional INFO faults cannot change the original recorder/native result. R1 was rejected after four logging counterexamples; final R2 passes33root tests and independent behavioral review with zero open findings. Its24matrix scenarios contain20reached logging faults and4existing-file controls. Reviewer disposition remains COMMENT because LSP/static tools were unavailable; no such tool or upstream PR approval is claimed. Actual collector coverage stays OPEN and historical EngineCore cause UNKNOWN. The native commit is not a deployed or qualified reference campaign. [C21 verification](test_report_2026-10-19_recorder_lifecycle.md).

C22 is accepted at `b5120e6b39f44a79245b04de71e0ed95d8bb34a6` for one existing test module only. Real V4 scheduler/Request callbacks observe physical-ID live counts79→27→27→23→0, reserved null, exact retired prefixes/refcounts, no in-flight retirement, completion and repeated reuse. Root205affected tests and independent73tests pass; two deliberate bad-frontier/reclamation mutations fail as expected. Actual configured decision events expose zero-required counts79/0/0/0 twice. This is a lane0 scheduler fixture with two real lane constructors, not full DES/native execution or C20 trajectory readback. All479nominated production/config/driver files are unchanged; no new77-case matrix is claimed. [C22 verification](test_report_2026-10-19_v4_cache_lifecycle.md).

The audit identifies one missing execution scope: newly assigned cache-block IDs reach allocation/debug records but have no before-forward cache-zero operator/cost consumer. Attention-output zeroing is different work. Full structural operator coverage remains OPEN until that consumer is represented once per step/physical worker, with bytes from the canonical layout and explicit synthetic-versus-native timing. DP8/64logical is still rejected by both driver and parameter-memory guards; EP64 fixtures are insufficient. Source-backed storage/constructor checks precede any range extension.

The existing CPU producer remains unchanged: [input readiness](c21_c22_continuation_20261019/cpu_input_readiness.md) records the absent device-completion, actual graph and READ-ready inputs required for a versioned numerical adapter. Do not invent a schema, infer cross-process offsets or publish host-only overhead rows. This does not block independent GLM native preparation, cache representation or workflow audit. Original V4 dummy-init and strict routing-OFF prerequisites, H800-first insufficient-quota H200 fallback (aggregate<=64/non-preemptible), GLM/V4-only ownership and all native/formal gates remain unchanged.

C16 is accepted at `2da11c8b6d319eca989fd3c9b5c997824667ff63`:15 exact staged/committed files implement GLM PP1 eager dummy sequential PDD through the existing allocator/transfer/scheduler/metrics owners. Root1124isolatedPASS, independent205PASS plus actual no-manual-free completion/FIFO/conservation/negative oracles, final APPROVE/zero unresolved, and unchanged77-case clean-cache comparison establish this scoped acceptance. All77 artifacts and cache filenames match; official comparator exit1 remains solely frozen modified-source provenance and is not a clean-commit/full-comparator PASS. [Verification report](test_report_2026-10-19_glm_structural_pd.md), [final review](c16_glm_structural_pd_20261019/c16_final_review_r3_20261019/report.md), [commit identity](c16_glm_structural_pd_20261019/r3_final_candidate/commit_verification.json).

Both full78-layer offline/online CPU runs complete5/5requests,5transfers,53,353,728B,24stage rows and1,587,581op rows. P32+D32 means64 logical simulated devices, not physical GPUs. Ledger-derived actual same-lane owners are request1 offline and request0 online for tail4, with capacity waits27.38851072/27.45851072ms. R2 executed source and old driver617eca are preserved; final R3 driverdb02e74e only revalidates the immutable outputs and adds wrong-lane/too-early rejection. All production bytes are unchanged between executed R2 and final R3. The earlier R1 liveness/preflight failures remain retained diagnosis. Whole-page bytes are per-worker/lane resident payload, not native wire/rank aggregate; same-time D-ready then P-release remains an explicit simulator approximation. Native NIXL lifecycle/timing, trained integration, graphs and formal full_frontier_path remain unqualified.

C17 is accepted at `dc7f382700218280980b923ba7fe9128091aaf1a` for the controlled CPU dummy structural slice. All26 staged/committed blobs match the final R3 manifest. Root62modules passed1230tests in26.33s; independent final-source review returned APPROVE with no unresolved material findings. Actual full43-layer offline and online each completed4/4requests,12real stages,481,136operator trace rows and1,376EP waves over32logical devices. Maximum stage conservation error is1.00000097e-9ms and maximum phase error6.94e-18ms. The unchanged77-case clean-cache matrix completed in432.822s:77identical artifacts, zero missing/mismatched cases and zero cache filename differences. Official comparator exit1 remains solely the recorded modified-working-tree provenance at91d1697; this is not a clean-commit capture or full-comparator PASS.

R2 executed37-file combined source includes exact final C16 plus final C17 production/driver bytes. R3 adds only three corrected preexisting tests. Root proved485tracked production/config files and both integration drivers byte-identical before reusing the R2 main/matrix evidence; no R3 simulator or matrix rerun is claimed. Initial1227PASS/3FAIL and earlier main_red_06 remain diagnostic history. Source identity is resolved before trace expansion quota; idle sources consume no real-request quota, while synthetic destination IDs remain separately labeled and native assignments UNKNOWN. Existing idle attention emits no request/stage metrics, so complete physical idle-attention trace coverage is not claimed. Complete compression/window boundaries, DP8/64logical, graph/PD, trained/native and full ticket05 acceptance remain open. [Final report](test_report_2026-10-19_dsv4_structural_execution.md), [root execution/comparison/identity evidence](c17_v4_structural_execution_20261019/final_acceptance/), [independent final review](c17_v4_structural_execution_20261019/independent_review/final_r3/review_report.md).

**D-WORKTREE-SCOPE-GLM-DSV4 (latest YC clarification):** this worktree implements and validates only GLM-5.3 and the original DeepSeek-V4-Flash. Step-5-Preview development belongs to `/data/ycfeng/Frontier/.worktrees/step5-preview` and its separate branch/owner. C18 Step representation integration and its execution-gap map are stopped; root archived and withdrew only this turn's uncommitted C18 changes:17tracked files restored,7new files retained then removed, all26nominated/dependency paths equal accepted C17dc7f382. [Withdrawal receipt](c18_scope_withdrawal_20261019/withdrawal_receipt.json). No C18 commit or Step support acceptance exists here. Retain shared-interface compatibility and external integration handoffs without importing Step source, changing Step task records or taking over its implementation. This scope clarification supersedes earlier local Step-merge instructions; existing GLM/V4 commits and native/formal acceptance criteria remain intact.

Bounded native recovery is complete: [report](post_structural_native_readiness_20261019/report.md), [34-artifact manifest](post_structural_native_readiness_20261019/manifest.json). GLM full78-layer startup/generation exists but complete G0-B/routing/observer/PD/graph qualification does not. Original V4 remains blocked by dummy-loader/finalizer initialization; no documented qualified same-model/runtime remedy was found. Strict routing capture stays OFF. Reuse existing drivers/observer evidence rather than restarting CUTLASS/startup investigation. Native preparation can proceed independently of simulator changes. GPU work retains H800-first, insufficient-quota H200 fallback, aggregate<=64 and non-preemptible requirements; no GPU/native/profile/calibration campaign occurred in these structural slices. A subsequent [12-row remaining-evidence map](glm_native_remaining_evidence_map_20261019/report.md) binds42read-time inputs and11native blobs to observer6c523e853. The later29-entry/zero-error recursive census already found no EngineCore file; no repeat census is planned. EngineCore init/path/close/write outcome remains UNKNOWN.

C19 is accepted at `7cc31154a5f90a2cc6dce886c34fd88c8956e2b3` for the existing local v0.28 DP/PD harness source/CPU slice only. Root final py312 checks119PASS including9actual local HTTP cases; py310119PASS with those9explicitly deselected and covered in py312. Independent APPROVE has zero unresolved material findings; observer-directory aliases and either containment direction are rejected before launch after a preserved RED. Only five harness/test files changed. Ticket02 native schema/probe/metric/recorder/clock and full model qualification remain open. [C19 report](test_report_2026-10-19_native_harness.md), [commit receipt](c19_native_harness_20261019/commit_receipt.json).

C20 is accepted at `3e6a421401a97ce11b7232facf3fa2028e56938f` for full43-layer CPU source-logical compression/window boundaries. Actual offline128/127/3 and online2051/2050/3 both complete2/2requests and8real stages on16logical devices; source calls, shapes, logical writes and stage/EP conservation pass. A real final-prefill/decode offset is corrected only by consuming existing `_get_settled_kv_progress` in the V1 decode snapshot. Root1254tests, independent APPROVE and complete actual-artifact readback pass. The preserved77-case matrix has77identical artifacts and zero cache filename differences; comparator exit1 is solely frozen modified-source provenance at C17, not clean-commit/full-comparator PASS. R1/R2 failures remain diagnostics. Physical cache retirement, complete idle-attention trace, configuration/64logical coverage, graph/PD, native/measured and full ticket05 gates remain open. [C20 report](test_report_2026-10-19_dsv4_boundaries.md), [commit receipt](c20_boundary_execution_20261019/commit_receipt.json).

Current work follows representation -> complete structural execution -> qualified native observation -> discrepancy localization -> calibration/formal validation. Do not reopen Step integration, repeat EngineCore census/CUTLASS/startup loops, calibrate isolated operators before their native workflow is qualified, or treat either CPU slice as native support. The [19-document scope audit](scope_readback_20261019/report.md) and [final remediation readback](scope_readback_20261019/remediation_final/report.md) resolve the one local ownership contradiction without changing model/native gates. C21 local recorder diagnostics are complete. Existing collector coverage and native device/graph/READ inputs are next; the CPU producer causal/clock adapter follows those required inputs. Each shared module has one owner.

## Historical GLM07/12 preparation — superseded by accepted C16

C15 retained-evidence/source recovery verifies26 identities and supplies a source-derived structural contract using existing target, allocator, request-event, P pending and release owners. This unblocks labeled CPU structural implementation of a concrete component-aware payload consumer with D reservation and backpressure; no new implementation is claimed yet. D readiness and P-side consumer-notification completion remain distinct; native cross-process order and delay are UNKNOWN. Native distinguishing workloads, actual READ bytes, failure timing and formal qualification remain open. Ticket12 now separates this structural contract from native lifecycle acceptance, correcting the previous dependency that made structural development wait for native qualification. Evidence/disposition: `c15_source_execution_20261019/glm_pd_evidence/`. One canonical owner must control shared transfer/allocator changes; no standalone metadata inventory or new manager.

## Problem Statement

A performance engineer wants Frontier to predict the online serving behaviour of GLM-5.3 and DeepSeek-V4-Flash on a 64-GPU H200 PD deployment: prefill and decode on separate Replicas, NIXL pull KV transfer, and CUDA graph replay. The predictions must meet the four-metric accuracy gate of the canonical contract against vLLM v0.28.0. The following gaps were established at the task baseline; accepted implementation progress is recorded in the current tickets and `progress.md`:

- The DSA attention family is frozen: it has no operators and execution is disabled. DeepSeek-V4-Flash's compressed and sliding-window attention, sparse indexer, mHC, and hash-routed MoE have no representation.
- The KV connector set holds only `none` and `p2p_nccl`. The existing per-layer extract cost and the measured send-slowdown correction belong to the P2P NCCL connector and must not leak into NIXL.
- The vLLM DP load-balancing cluster scheduler accepts only one co-location Replica, so the P/D entry and DP-lane placement of the target topology have no supported model.
- The calibration tooling (E2E accuracy gate, CPU-overhead producer, native drivers, `frontier-calibration` skill) is built around vLLM-BS, an instrumented fork of vLLM v0.10.2. Its pairing, clock, phase, and connector rules are not proven for v0.28.0, multi-node runs, or asynchronous scheduling.

## Solution

The calibration platform is 64 H200 (`D-H200-CALIBRATION-PLATFORM`, 2026-10-21), at most 70 non-preemptible. Exploratory Stage A/B work follows the earlier `D-GPU-H800-UNLIMITED-H200-70` rule (H800 first, H200 only on insufficient H800 quota). Record actual hardware and keep timing identities separate: H800 measurements never qualify an H200 cell, and no H200 result blends H800 tables. The `D-DSV4-GROUPED-CACHE` approval authorizes Option B in `ticket05_component_cache_proposal.md`: extend existing KVCacheManager/BlockPool with one shared pool/group tables under the explicit PP1 co-location restrictions. It does not enable the V4 family before its remaining representation gates pass.

Extend Frontier along its existing owners instead of building parallel machinery:

- model binding, operator and attention-family registries, profiling producers, typed predictor identity, ordered per-layer execution time;
- the vLLM V1 engine replica scheduler, the KV-transfer configuration, predictor, and events;
- the CPU-overhead producer and the E2E accuracy gate.

Historical C07–C14 component boundaries are explicit in the tickets: C07-C09 implement V4 source parameter storage, five-group cache capacity and the approved PP1 co-location allocator semantics. C11 extends GLM full-model static storage to PREFILL/DECODE without changing transfer or serving semantics. The per-layer V4 operator/precision inventory and accepted C12 static unpartitioned constructor dimensions supply source metadata through existing modules/CSV while execution and generic profiling remain guarded. Accepted C13 `d3d4229` declares the three source TP-SP payloads in COMM_FAMILY and exposes the existing backend reduce-scatter through canonical dispatch. All remain source-only; stage placement, ordered timing and endpoint frequency are not implemented by those declarations. Accepted C14 `596451a` derives per-request/per-layer eager attention shapes/branches from Batch snapshots and consumes them through existing synthetic AttentionOperatorTimes. This PP1/PCP1/DCP1 source component has833root-isolatedPASS, independent APPROVE and77exact artifact/cache comparisons; modified-source provenance is retained. It does not bind a full ExecutionTime/stage/trace or enable normal V4 serving. C15/C17 subsequently connected SP valid/padded source-rank tokens, mHC/shared/synthetic routed FFN, selected EP backend and endpoints to canonical controlled stage/trace/serving. Full compression/window/configuration boundaries, native/measured validation and external compatibility coordination remain open. A manifest, static byte count or legacy regression cannot satisfy the native/reference or formal acceptance gates below.

Accepted C15 `91d1697` is an explicitly partial source-rank SP/mHC/shared/endpoints component consumer, verified by877isolated tests, independent APPROVE and77exact artifact/cache comparisons. It does not satisfy full stage/trace/serving admission. Geometric valid rows and the padding-skip environment flag do not prove native routed assignments: selected score/hash router masking and EP backend behavior require their own evidence. Missing routed work must stay explicit rather than be replaced by balanced synthetic assignments.

The ground truth is dummy-weight native runs on vLLM v0.28.0 at SHA `2cf0a6915ce544dc493a0990f2ea38d81601128a` (ADR 0001). Observer-only instrumentation lives on a vLLM-BS branch cut from that SHA.

The implementation order follows the 2026-10-18 resume instruction:

1. **Representation and operator coverage:** normalize each pinned model configuration at the existing boundary; bind per-layer attention and FFN/precision contracts; register semantic operators and physical owners; describe memory and state through existing interfaces. GLM-5.3 is the first complete tracer. DeepSeek static representation may proceed independently, even while its native dummy path is blocked. Step-5-Preview supplies shared contract requirements from its separate task. No numerical fit or complete G0-B is a prerequisite for this work.
2. **Structural E2E:** connect complete layers to the existing request, stage, communication and memory workflow, first co-location and then the required online/chunked/graph/PD path as its semantics become supported. Controlled timing proves counts, ordering and conservation only. Unsupported operators, states or configurations fail explicitly; no frozen guard is simply disabled. A PD slice needing a core refactor remains blocked on the smallest evidenced proposal, without blocking unrelated model work.
3. **Reference instrumentation and measurement:** reuse the locked native runtimes and existing benchmark/profiling/gate owners. Instrument actual execution order, local CPU preparation/launch/synchronization, GPU paths, communication, KV handoff and memory. Early source audits and instrument preparation may run in parallel with stages 1–2; formal collection must have a represented workload and a runnable native path. Local initialization diagnostics remain distinct from calibration.
4. **E2E comparison and localization:** compare workflow ordering, admission, asynchronous CPU/GPU boundaries, graph/chunk behavior, PD coordination and memory lifetimes alongside operator coverage, shapes, precision, durations and communication payloads. Establish how an operator discrepancy changes the complete workflow before adding a correction.
5. **Calibration and validation:** apply only measured corrections, freeze fit data and holdout separation, satisfy the existing per-cell gates, and regress GLM/V4 plus existing shared infrastructure here; supply the separate integration owner with evidence for eventual three-model regression.

GLM remains first for the complete trained PD path, followed by DeepSeek measured integration; `09.full_frontier_path[GLM]` retains its trained-predictor, cost-replay and coverage conditions. It is **not** the first structural simulation gate. `04.structural_execution` and `09.structural_frontier_path[GLM]` expose earlier executable progress. This supersedes the earlier Q2/Q12 blanket barrier for DeepSeek representation only. Step production milestones still follow their own approved task; this review does not silently release them.

Cross-worktree shared contracts are described in `/data/ycfeng/Frontier/.local-draft/step5-preview/03_Shared_Integration.md`. Under the latest YC clarification, `feat/glm53-dsv4-vllm028` owns GLM/V4 work only; Step remains in its separate worktree/branch and does not merge here as a local implementation stage. C01 runtime/profile identity, C02 layer contracts, C03 cache/transfer lifetime, C04 physical-cost measurement, C05 workflow/placement and C06 gate/normalization each retain one owner for changes made here. Provide reviewed reusable GLM/V4 changes and an interface/regression handoff to the separate Step/integration owner. Eventual project-wide three-model acceptance remains an external coordination result; it does not authorize implementing Step in this worktree. Timing tables require matching runtime, hardware, precision and physical scope; Step's separate image/runtime qualification cannot transfer to these models.

Tickets 12–15 retain their IDs and per-model evidence: D-PD and D-ROUTING have GLM and DeepSeek stages; D-ASYNC and D-DP-PD are checked across model runtimes. They are no longer automatic approval requests for ordinary extensions. Source-backed reuse of existing semantics is recorded and verified by the owner. Any material Frontier core refactor, unresolved materially different simulation semantics, or change to acceptance goes to the user with alternatives; this review approves none. Static fields and operator declarations do not wait for routing histograms, NIC contention curves or an async redesign.

## User Stories

### Simulator users

1. As a performance engineer, I want to run a GLM-5.3 PD simulation from one command with a checked-in model configuration, so that I can predict serving latency without a GPU.
2. As a performance engineer, I want to run a DeepSeek-V4-Flash PD simulation the same way, so that both target models share one workflow.
3. As a performance engineer, I want the P32+D32 topology (each role TP8 × DP4 × EP32 × PP1) expressed as one PREFILL Replica and one DECODE Replica with four DP lanes each, so that request ownership matches vLLM.
4. As a performance engineer, I want a diagnostic-only alternative P/D ratio (Topology B: one 16-GPU PREFILL Replica and three 16-GPU DECODE Replicas, each TP8 × DP2 × EP16) kept expressible, so that layout questions can be explored without it counting as formal acceptance scope.
5. As a performance engineer, I want online open-loop arrivals with CUDA graph replay on the canonical case, so that the simulated path matches the deployed one.
6. As a performance engineer, I want client TTFT, internal TTFT, PD KV-ready time, TPOT, ITL, output-token throughput, total-token throughput, and request E2E reported under distinct names, so that no metric is silently redefined.
7. As a performance engineer, I want an input outside the measured shape, world size, or concurrency range to return an explicit coverage miss, so that I never get a silently extrapolated prediction.
8. As a performance engineer, I want a configuration that combines NIXL with the P2P-only per-layer slowdown or layer-wise latency hiding to be rejected, so that legacy corrections cannot be applied to the wrong connector.
9. As a performance engineer, I want the simulator to stay CPU-only and sequential, so that it does not gain a CUDA or vLLM runtime dependency.

### Model structure

10. As a Frontier maintainer, I want GLM-5.3's 78 backbone layers bound from its pinned configuration: the first 3 dense FFN, the rest MoE with 256 experts and top-8, and the indexer schedule resolved as vLLM v0.28.0 resolves it. That way the per-layer table exactly matches the native module inventory.
11. As a Frontier maintainer, I want GLM-5.3 full-indexer layers and shared-index layers to have distinct per-layer attention/indexer variants within the existing layer contract. A shared-index layer still runs its own sparse MLA attention, so index reuse is not mistaken for attention or KV reuse.
12. As a Frontier maintainer, I want DeepSeek-V4-Flash's 43 backbone layers bound with their uncompressed, CSA, or HCA attention variant from the compression-ratio schedule, with no next-token or MTP entries executed as extra layers.
13. As a Frontier maintainer, I want the DeepSeek-V4-Flash compressor, partial state, sliding-window plus compressed attention, sparse indexer, mHC, and low-rank output projection each owned by exactly one operator or measured composite, so that no cost is missing or double-charged.
14. As a Frontier maintainer, I want the first three DeepSeek-V4-Flash layers to use hash routing on input token ids, and the remaining layers score-based routing with shared experts, so that hash layers never borrow a gate-based distribution.
15. As a Frontier maintainer, I want compression-boundary work modelled, so that two single-token decode steps on either side of a 4-token, 128-token, or KV-page boundary can differ in cost.
16. As a Frontier maintainer, I want a new family to reject non-dummy timing until every required link exists: model binding, operator specs, profile features, producer, training target, predictor, execution-time composition, trace and metrics, tests. The existing explicit structural dummy timing mode may exercise represented operations before measurement; unsupported structural components still fail explicitly, and old tables never fill a gap.

### Profiling and predictors

17. As a calibration engineer, I want profiling producers to call the real v0.28.0 classes chosen by the resolved backend, so that the measured kernels are the ones the full model runs.
18. As a calibration engineer, I want every profiling table to carry its runtime fingerprint, configuration revision, `weight_source=dummy`, device, schema version, physical scope, backend, dtype and quantization signature, parallel groups, and graph mode, so that tables from different runtimes never mix.
19. As a calibration engineer, I want training and loading to fail on missing features, schema mismatch, FP8/BF16 mismatch, H200 data fed to H800, v0.10.2 data fed to v0.28.0, or mixed graph and eager data. That way the wrong table can never train a predictor.
20. As a calibration engineer, I want new runtime tables placed beside existing ones without overwriting them, so that v0.10.2 data and its regressions survive.
21. As a calibration engineer, I want the shape grid seeded from short native traces and then widened at the boundaries: top-k thresholds, compression ratios, capture buckets, pages, and uneven lengths. That way the predictor covers the shapes that actually occur.
22. As a calibration engineer, I want warm-up, autotune, and graph capture separated from steady-state measurement, and multi-rank results reported by their critical-path maximum, so that measurements reflect serving.

### PD and communication

23. As a calibration engineer, I want the NIXL handoff to be one request-level logical transfer that reuses the existing transfer events, pending-KV set, and source-release owner. That way the KV lifetime has one owner.
24. As a calibration engineer, I want NIXL pull to earn no layer-wise prefill overlap, so that the simulator does not invent a benefit that native NIXL does not provide.
25. As a calibration engineer, I want the NIXL payload built from the real KV cache groups and transfer descriptors, so that CSA and HCA state is not approximated by scaling a dense KV size.
26. As a calibration engineer, I want TP collectives, MoE dispatch and combine, and NIXL transfers measured on the libraries, world sizes, payloads, and concurrency actually used, so that costs come from evidence rather than bandwidth constants.
27. As a calibration engineer, I want KV transfer and EP traffic that share a NIC checked for measurable contention, so that a single-stream curve is not presented as the 64-GPU concurrent cost.
28. As a calibration engineer, I want normal completion, receiver memory shortage, duplicate or late completion, and abort tested for release conservation, so that no KV block is freed twice or leaked.

### Workflow and placement

29. As a Frontier maintainer, I want every v0.28.0 scheduler observation classified as same contract, version delta, inactive in scope, measurement only, or needs discussion. Only active version deltas that change results get runtime code.
30. As a Frontier maintainer, I want the existing batch queue, scheduled and completed frontiers, DP dummy forward, finish synchronization, and held outputs reused, so that no second engine model appears.
31. As a Frontier maintainer, I want the PDD guard on the vLLM DP load balancer kept until D-DP-PD is decided, so that default round-robin never silently stands in for the native policy.

### Calibration tooling

32. As a calibration engineer, I want the E2E accuracy gate extended with v0.28.0 cases and metrics as the single acceptance entry, so that there is no second gate.
33. As a calibration engineer, I want the CPU-overhead producer to pair engine steps and forwards by explicit causal identifiers rather than position. Durations should come from one clock domain, and NIXL READ submission should be measured separately from network wait. That way asynchronous and multi-node traces are not mis-joined.
34. As a calibration engineer, I want observer-only instrumentation in three modes (clean, light buffered, diagnostic sampled), with light-mode median perturbation frozen at or below 3% before formal runs (a measured value above the limit leaves light mode unqualified; the limit is not raised), so that instrumentation does not distort the reference.
35. As a calibration engineer, I want each old calibration rule (the `pd_s33` endpoint offset, the layer-wise `send_span` admission, the DP/PP mode pairs, forced uniform routing, the k1d/k1e/X3 CPU tables) judged per item. Old cases keep their rules, and new cases inherit nothing automatically.
36. As a reviewer, I want every native run recorded with its source, harness, instrumentation, runtime, initialization, configuration, topology, rank map, workload seed, profile and predictor hashes, observation mode, clock domain, rank completeness, GPU-hours, and gate status. Failed and timed-out runs stay in the manifest, so that results are reproducible and nothing is dropped.

### Acceptance and delivery

37. As a reviewer, I want both models qualified as full-structure dummy-weight native runs on one exact vLLM stack, through G0-A to G0-D, with the 64-H200 P32+D32 run as the last step and the fixed 128-request, 6144/1024-token, QPS-1.0 case as the only formal workload.
38. As a reviewer, I want exactly two formal E2E results — one per model — each carrying client TTFT, TPOT, THPT and E2E with their relative errors, evidence references and PASS/FAIL verdicts, so that acceptance is one decision per model rather than a cell matrix.
39. As a reviewer, I want results labelled `FAIL`, `INSUFFICIENT_EVIDENCE`, `COVERAGE_MISS`, `REFERENCE_UNSTABLE`, `BLOCKED`, or `BLOCKED_ON_DECISION` whenever they do not pass, so that no shortfall reads as success.
40. As a reviewer, I want old features, old data, and simulator wall time regression-checked against the pre-change baseline, so that existing users see no unexplained change.
41. As a performance engineer, I want every H200 claim to stand on H200 measured data, with H800 tables retained beside them and never blended into an H200 result, so that each hardware claim keeps its own evidence.
42. As a calibration engineer, I want the `frontier-calibration` skill extended to the v0.28.0 workflow once it is verified, so that later agents follow the new procedure.

## Implementation Decisions

### Baseline, runtime, and models

- Execution base: `main@bfa82de` on branch `feat/glm53-dsv4-vllm028`. PRs #34, #35, #38, #40–#44 are already merged and are reused, not re-ported. The merged fixes for online DP rotation, skipped waiting, time limit, in-flight phase, preemption, and graph padding count as already present, with regressions to run.
- Truth runtime: vLLM v0.28.0, `2cf0a6915ce544dc493a0990f2ea38d81601128a`, with no automatic version fallback. A unified version change or upstream fix needs the user's decision and moves both models together.
- Every new native launch uses `--load-format dummy`. Only allowlisted metadata is fetched: configuration, tokenizer files, chat template, generation configuration, and small required source or index files, each with revision and SHA-256. Checkpoint weights and unconstrained snapshots are never downloaded.
- The runtime lock records the image digest, Python, Torch, CUDA runtime, driver, transformers, Triton, FlashInfer, FlashMLA, DeepGEMM, TileLang, NCCL, NIXL, UCX, and the EP backend. Unused entries are recorded as `not_used`.
- The model lock records the repository, variant, configuration and tokenizer hashes, `weight_source=dummy`, native seed, initializer revision, resolved architecture and quantization, module and auxiliary-buffer inventory, attention, MoE, and connector backends, and the rank map.
- The model variants are fixed: `zai-org/GLM-5.3` (`GlmMoeDsaForCausalLM`) and the original `deepseek-ai/DeepSeek-V4-Flash` (`DeepseekV4ForCausalLM`). The `-0731`, DSpark, and V4.1 variants are out of scope.
- Scope is language-only. Only the text-only causal-LM paths of these two architectures are qualified, profiled, and modelled. If a resolved configuration, processor, or native module inventory contains a vision or other multimodal component, ticket 01 records it, the native run uses only the text path, and Frontier never models that component.
- No layer, expert, or quantization is removed to make a run fit. Reduced structures serve diagnostics only.

### Native qualification ladder

| Stage | Content | What it proves |
| --- | --- | --- |
| G0-A | dependencies, imports, dummy initialization, operator and small-configuration tests | diagnostics only |
| G0-B | full structure in one serving Replica (16 or 32 GPUs as capacity requires), real quantization and kernels; resolved configuration/backends/inventories; text-only path; memory ledger. Native routing occupancy is not part of G0-B (2026-10-20) | the model runs on this runtime |
| G0-C | native P/D with unchunked prefill for the current canonical case, online arrivals, graph replay, same-structure layout; earlier chunked diagnostics remain history | NIXL payload, state handoff, token counts |
| G0-D | one P32+D32 run per model on 64 H200 | all ranks, cross-node EP and NIXL, real routing, stable drain |

Observer preparation and hooks may start after the runtime lock; formal model-specific trace validation still requires the relevant native execution. The canonical formal case waits for G0-D (2026-10-21: the former 36-cell matrix is superseded). D-DP-PD's DP4 evidence comes from G0-D, so G0-D runs before that decision; native qualification never waits for a Frontier simulator change.

Formal cost replay and formal E2E references use only traces collected after the observer tooling is validated for that model: causal pairing, clock domains, NIXL READ separation, graph rules, and the measured clean-versus-light perturbation. Earlier traces are labelled diagnostic and unverified. This is checked before launch; during a run with 32 or more GPUs, a validator failure only marks artifacts unverified.

### Model structure and registries

- GLM-5.3 reuses the `dsa_attention` family with a verified variant. Its indexer schedule follows vLLM v0.28.0's own resolution (`index_topk_freq`, `index_topk_pattern`, `index_skip_topk_offset`), not the HF `indexer_types` text. At the original baseline, model binding detected DSA only from DeepSeek-V3.2 model types and `dsa_*` keys. Accepted C02/C04 now bind the pinned GLM-5.3 schedule through these existing contracts; measured timing, native execution correspondence and the remaining ticket04 chain still require their stated evidence.
- Per-layer attention identity is resolved by `resolve_attention_topology` in `frontier/attention/model_binding.py` and consumed where stages are assembled (`_assemble_stage`) and where attention is priced. `layer_contract_resolution.py` resolves the typed FFN contract. The attention and indexer extensions start from the former, although v2 §1.2 names the latter (C1).
- DeepSeek-V4-Flash uses per-layer variants that distinguish uncompressed, CSA, and HCA attention.
- mHC first extends the existing backbone or operator expression. A new family needs a stated gap and a minimal interface, and mHC is never folded into dense attention as a fixed overhead.
- Profiling and runtime both emit a per-layer table (`global_layer_id`, `layer_kind`, `attention_variant`, `index_producer_layer`, `indexer_executed`, `ffn_kind`, `op_owners`) that exactly matches the native module inventory.
- Each measured time belongs to one physical scope:
  - A fused kernel's sub-steps are not charged again.
  - A fused MoE that contains communication is either one measured composite or observable segments, never a mix of the two.
  - Whole-layer timings only check decomposition coverage; no layer correction factor is used.
- Ordered per-layer execution-time records already represent layers that differ; no model hierarchy mirror is added.

### Profiling data identity

- The existing `model_architecture_profile` and `typed_operator_contracts` are extended with:
  - runtime fingerprint, configuration revision, `weight_source=dummy`, initializer identity, device fingerprint, schema version;
  - physical scope, semantic owners, and included sub-scopes;
  - backend, model, activation, and cache dtype, and quantization signature;
  - phase, parallel groups, and rank aggregation;
  - graph mode, capture size, actual and padded tokens, and mixed-batch composition;
  - query, context, and selected-context lengths;
  - MoE routed rows, occupancy, padding, and route kind;
  - compression ratio, boundary flags, and cache group;
  - measurement family, warm-up, repetitions, and spread.
- Only fields that change estimator inputs or outputs enter the training identity.
- The existing identity, loader, and measurement-path resolution chain locates runtime-specific files. Call sites never assemble version directories themselves.
- Extend the existing typed identity only for fields that affect admission, interpretation or lookup. Record the contract and regress historical inputs; routine additive metadata does not need repeated approval. A material core schema redesign needs the source-backed refactor proposal.

### PD transfer

- NIXL pull receives no layer-wise transfer benefit.
- Cost isolation against the P2P NCCL connector:

  | Cost or rule | P2P NCCL (legacy) | NIXL pull |
  | --- | --- | --- |
  | per-layer `attn_kv_cache_extract` | priced as today | not inherited; only a gather or repack that native NIXL actually runs is priced |
  | `kv_send_slowdown_per_layer_ms` | kept within its measured scope | must be 0; a nonzero value is rejected |
  | layer-wise latency hiding | today only the `enable_latency_hiding` field, marked as a future extension, with no production overlap path; this behaviour is kept and no P2P overlap is added | an enabled flag is rejected |
  | submission, transfer, visibility on D | existing contract | measured in the native NIXL scope, neither omitted nor duplicated in host probes |
  | source hold and free | existing owner | same owner; final release condition verified |

- Causal order that must hold: P finishes the required prefill → remote metadata is available → D has reserved storage and can post READ → all required data is usable on D → the request may enter its first D step. P keeps the source KV until it is safe to free. P and D keep executing other ready work meanwhile.
- The NIXL connector identity is added at the existing configuration capability check, under a name chosen by the current configuration conventions. Add the identity through the existing capability check and document its contract; approval is required if expressing its semantics needs a material core refactor.
- Payload and capacity come from the native `KVCacheConfig` and NIXL descriptors, not from one `tokens × layers × head_dim` formula. Group capacity in the allocator is part of D-PD; vLLM's hybrid KV manager is not ported.
- Whether target receipt and source release stay merged at transfer end is decided in D-PD, not assumed.

### Communication

- Measure first: the world sizes actually used (8, 16, 32), payload quantiles and boundaries, and the valid concurrency levels among 1, 2, 4, 8, and 16. The prefill throughput path and the decode latency path are measured separately.
- Use the existing CC backend and KV-transfer predictor registries. A narrow measured backend is added under the existing registry only if needed.
- No UCX or DeepEP protocol state machines, no packet-level RDMA simulator, and no NIXL control threads. Whether to model shared-NIC queueing is decided in D-PD.

### Workflow and placement

- Every v0.28.0 observation is classified as `SAME_CONTRACT`, `VERSION_DELTA`, `INACTIVE_IN_SCOPE`, `MEASUREMENT_ONLY`, or `NEEDS_DISCUSSION`. Only active `VERSION_DELTA` items that change admission, batch shape, dependencies, resources, cost, or output metrics get runtime code.
- The existing `VLLM_DP_SYNC_INTERVAL=32` is compared against the v0.28.0 active loop and is not treated as a bug merely because it is a constant.
- No CPU+GPU+network serial sum, no plain maximum, and no DES event per kernel or poll. Runtime granularity stays at batch, layer, and EP wave.
- When two implementations fit the evidence but differ in admission or latency, the user picks; the agent does not pick the one closer to a single E2E result.

### Routing

- Native routing on dummy weights stays unmodified. `moe_uniform_routing` is not inherited, and dummy weights are not assumed to give a balanced load.
- Native routing occupancy evidence belongs to `03.routing_inventory` and ticket 14, not G0-B. Primitive MoE profiles sweep tokens-per-expert and do not wait for it. Any routing collection must respect the strict capture-safety rule; ticket 14 proposes the concrete method when it becomes the next blocking dependency.
- Routing for predictive runs comes from independent calibration runs with a declared shape, seed, and phase scope. Holdout runs' future expert histograms are never an input.
- For DeepSeek-V4-Flash hash layers, known prompts may be routed from the input and verified auxiliary state. Decode-token effects are estimated statistically.

### Instrumentation and CPU-overhead producer

- Observer-only commits go on a new vLLM-BS branch from `2cf0a69`, with no functional router, scheduler, or completion changes. Pushing that branch needs the user's approval at that time.
- The old vLLM-BS diff is classified as observer-only, harness, functional router change, runtime fix, or obsolete. Only observer and harness parts migrate.
- Traces carry run, request, engine, node, rank, group, forward, and step identifiers, the clock domain, actual and padded tokens, phase, graph mode, transfer identity, and KV group.
- The existing CPU-overhead producer gains a versioned source adapter for the new fields; no second CSV publisher is added. Historical logs keep their producer SHA.
- Timestamps from different hosts are never subtracted unless the clock offset is verified.

### Topology and workloads

Superseded on 2026-10-21 by the canonical contract: the multi-workload family, the multi-QPS levels, the ≥1,000-request cells, Topology B as a holdout, and the large E2E scenario matrix are no longer formal acceptance scope. They remain available as diagnostic tooling and their historical results are retained. What follows is the frozen formal configuration.

- **One formal case per model.** 128 requests at QPS 1.0, fixed one per second over 127 s, each exactly 6144 input / 1024 output tokens. Use the replacement manifest specified above; SharedGPT and target-tokenizer re-count are not dependencies.
- **Topology.** GLM-5.3 uses Topology A: nodes N0–N3 serve PREFILL and N4–N7 serve DECODE, each role TP8 × DP4 × EP32 × PP1, regenerated from the verified inventory if the node layout differs. DeepSeek-V4-Flash is evaluated at the same layout and drops one step only if the locked runtime cannot support it.
- **Serving features.** P/D disaggregation, online serving, thinking disabled, prefix caching disabled, chunked prefill disabled, CUDA Graph enabled, quantization enabled, MTP as the native default requires, dummy weights.
- **Native admission.** Record utilization and in-flight depth for the fixed workload, but do not change the owner-selected count, rate or lengths to improve utilization. Each of 3 clean repetitions has independent warmup and completes all 128 requests; failed or unfinished requests remain explicit and cannot be filtered out.
- All requests carry text token inputs only; no image, video, or audio inputs.

### Metrics and acceptance

- **Four independent gates:** mean client TTFT, mean request TPOT, total-token THPT over first formal submission to last formal last-token completion, and mean per-request E2E. Use the same definitions and repetition aggregation in Frontier and native; the 127 s arrival span is not the throughput window. Internal TTFT, ITL, PD KV-ready time, output-token throughput and request throughput are diagnostics. `abs(frontier − vllm) / abs(vllm) ≤ 0.10` per metric; all four must pass.
- **Legacy fields keep their meaning.** `ttft_ms` stays internal TTFT and `token_throughput_tps` stays prompt-plus-output tokens; the gated client TTFT and the gated THPT are reported under distinct names, never by redefining those columns.
- **Admission of the reference.** The native ground-truth run must clear its own evidence checks: all ranks present, clean shutdown, stable repetitions, matching request population and endpoints. A reference that fails admission leaves the case `REFERENCE_UNSTABLE` rather than being averaged.
- **Correctness tests are separate from formal cases.** Operator, attention, MoE, configuration, memory-accounting, P/D scheduling, KV-transfer and CUDA-graph checks stay required; they are not formal E2E cells and do not multiply the acceptance matrix.
- **Anti-overfitting without matrix growth.** Fresh clean repetitions and targeted correctness checks guard the published result; no additional workload cells are added to serve as holdouts.
- Memory: the existing parameter-memory policy is kept and documented. No blanket 2-byte policy and no exact quantized-capacity framework. Non-KV overhead is taken from H200 native observation at the canonical topology. A measured cache capacity may isolate scheduling error only when recorded as `capacity_source=measured` with its validity scope; it is not presented as memory prediction.

### Approval and resources

- All GPU and CPU resource use for this task is authorized in advance (Q7). The October 20 Asia/Hong_Kong request explicitly authorizes commit/push/PR publication of the current Frontier branch; separate native-repository publication and main merge remain outside that authority.
- Apply the 2026-10-18 approval policy in `requirements.md`: routine model/operator/profiling/native-observer/test extensions are authorized for the subsequent execution phase. Record D-* reuse dispositions with evidence; no agent marks an unresolved core refactor or materially ambiguous acceptance semantics approved.
- Before a material Frontier core refactor, identify the exact failing boundary and show why an incremental extension is insufficient; present the smallest change and alternatives and wait for explicit approval. A registry entry or routine binding is not such a refactor.
- GPU placement: the canonical formal case uses 64 H200, at most 70 non-preemptible; exploratory work retains the earlier H800-first rule. Hardware-specific evidence never transfers across devices.
- Runs with 32 or more GPUs use non-blocking validators.

## Testing Decisions

- A good test checks external behaviour at the highest seam: simulator outputs, rejected configurations, predictor identity failures, gate verdicts. It does not check private structure or source strings. A new negative test must fail on the unfixed code or on a targeted mutant.
- The five seams, with no new one (Q9):
  1. **Acceptance:** the E2E accuracy gate, extended with v0.28.0 cases, metrics, and manifests. This is the only acceptance entry.
  2. **Simulator:** a full `python -m frontier.main` run with the new model configuration. Dummy timing mode is used for structure (layer coverage, PD events, conservation) and trained predictors for numbers.
  3. **Predictor:** the boundary from profiling CSV with typed contracts to training and lookup, where identity and coverage failures must be raised.
  4. **Regression:** existing unit and integration tests, the refactor-fidelity harness, and the dummy smoke matrix, run as a baseline-vs-candidate comparison over at least 50 concrete scenarios for every code-changing ticket.
  5. **Native:** the existing PD and DP/PP native drivers producing raw bundles that feed the gate normalizer.
- Required test areas (v2 §12):
  - **Dummy and initialization:** no weight download; non-parameter state is valid; dummy-weight runs, DP dummy forward, and dummy timing mode stay distinct.
  - **Model and operators:** binding failures; GLM index relations; V4 backbone versus MTP counts, compression boundaries, and hash routing; no double charging of fused or shared experts.
  - **PD:** legacy P2P regressions; NIXL rejects legacy slowdown and hiding; no premature READ or ready; release conservation under late, duplicate, and abort cases.
  - **Async, DP, routing:** existing queue, frontier, held-output, and DP dummy coverage; witnesses for new deltas; the PDD guard holds until D-DP-PD; native skew distinguishable from balanced.
  - **Probe and gate:** out-of-order forwards, mismatched clock domains, READ submission versus wait, unknown graph modes, TTFT and throughput naming; missing cells, ranks, or requests and NaN values never pass.
  - **Regression:** CPU-only import, predictor cold and warm cache, numeric outputs, simulator overhead.
- Prior art: `tests/unit/test_kv_cache_extract_pricing.py`, the kernel-gap pricing and profiler tests, the preemption and time-limit tests, `tests/e2e/refactor_fidelity/`, `tests/unit/test_e2e_gate.py`, `examples/architecture/run_dummy_smoke_matrix.sh`.
- Every run of a reused test records its source, harness, configuration, and data identity. The baseline fail, skip, and collection-error set is recorded afresh in the current environment.

## Out of Scope

- Vision and all other multimodal modelling: vision encoders and projectors, image and video preprocessing, multimodal embeddings and their caches, multimodal position encodings, multimodal serving options, and image or video workloads. Frontier models the language backbone only.
- MTP and DSpark, EPLB, DBO, PCP and DCP, prefix caching, KV CPU offload, HiSparse offload, PD-AF, and Thinking Mode for the new models. Native-default MTP is disabled for both models (source check, 2026-10-21); final canonical command/runtime parity remains to be recorded.
- Parallel cluster processing; the sequential DES executor stays.
- Real checkpoints, output quality, benchmark accuracy, and routing claims for production checkpoints.
- An exact quantized-weight capacity framework, or a new blanket parameter-byte policy.
- Copying vLLM's EngineCore, worker, or connector thread and IPC structure; a generic CPU/GPU/NIC event framework; a packet-level network simulator; NIXL retry, TTL, or heartbeat protocols.
- A parallel `tests/comparison/vllm_028/` harness, a second gate, or a second CPU CSV publisher.
- Model variants other than the two named, and any version other than vLLM v0.28.0 unless the user decides.
- Repository-wide refactors. Modules above 2,000 lines get only related, documented cleanup or splits.

## Further Notes

### Deltas against spec v2

1. The execution base is `main@bfa82de`, 20 commits after the reviewed snapshot `9a650fb`. The changes that matter here:
   - PR #45 (`4fbb298`) keys `kernel_gap.csv` rows by attention TP size through a new `num_tensor_parallel_workers` column and updates the gate artifact hashes. The kernel-gap and graph-family owner row in v2 §1.2 must use this layout.
   - PR #46 adds H200 profiling and CPU-overhead tables.
   - PR #47 removes the `outputs/` and `logs/` run outputs, which no code reads.
   - PR #48 adds the in-repo `skills/frontier-calibration` skill, scoped to vLLM-BS v0.10.2.
   - PR #49 and the release bump refresh docs and guard version strings only.
2. The historical clue `task_2026-09-22_stage_admission_ordering` lives in `/data/ycfeng/Frontier/.worktrees/stage-admission-ordering/task_memory/`, not in the main checkout's `task_memory/`. The other three historical task records named in ticket 00 (`task_2026-09-21_oversized_module_split`, `task_2026-09-21_issue26_correctness_pr`, `task_2026-09-27_pr_stack_vllm_e2e_accuracy_gate`) live in the main checkout's `/data/ycfeng/Frontier/task_memory/`, not in this worktree.
3. GLM-5.3 remains first for trained full-path integration (Q2/Q12); the 2026-10-18 instruction permits parallel early model representation. See the staged milestones rather than treating whole tickets as serial.
4. D-* IDs 12–15 remain the shared evidence/decision records (Q3), with the narrower 2026-10-18 approval boundary.
5. Instrumentation goes on a vLLM-BS branch from `2cf0a69` (Q4), and the skill is extended in T11.
6. Resource use needs no per-run approval (Q7).
7. Scope is explicitly language-only (Q11). Spec v2 implied this through its causal-LM architectures and text workloads but never stated it.
8. Task records under `task_memory/` are ignored by Git. Working reports stay there, and tracked deliverables go through existing owners (gate cases and manifests, docs), as v2 §13 already intends.
9. v2 §1.2 assigns GLM index reuse and V4 compressed-layer mapping to `layer_contract_resolution.py`, which resolves FFN contracts; attention identity is resolved in `model_binding.py` and consumed in `_assemble_stage` (C1). v2 §5 keeps the P2P latency-hiding rule "in its old scope"; at `bfa82de` that scope is a future-extension flag with no production overlap path (C2).
10. The "complete GLM path" of Q2 is the full simulator path without the formal E2E case (formerly `64-H800`, now the 64-H200 canonical case of ticket 19), closed by `09.full_frontier_path[GLM]` (Q12; platform updated 2026-10-21). v2 lets T04 and T05 start from the same blockers; this task orders them.
11. D-PD and D-ROUTING are staged per model; D-ASYNC and D-DP-PD are decided once on GLM-5.3 evidence and checked against DeepSeek-V4-Flash (Q13).
12. v2 lists one H200 matrix that grows with resources. That is now the calibration platform itself: the 2026-10-21 canonical contract fixes 64 H200 (at most 70 non-preemptible) and one formal case per model, so ticket 11's former "H200 scope freeze" is obsolete and folded into tickets 16–19. Actual-hardware evidence is still required for every claim.

### Verified facts at `bfa82de` (2026-10-15)

- `DSA_ATTENTION_FAMILY` has `operators=()`, `execution_enabled=False`, `dsa_frozen=True`.
- `KV_CONNECTORS = ("none", "p2p_nccl")`.
- `VllmLoadBalancingClusterScheduler` rejects any non-`MONOLITHIC` cluster, any cluster with more than one Replica, and any replica scheduler other than vLLM V1; dense models also need `attn_dp=1`.
- `H800DeviceSKUConfig`, `data/profiling/compute/h800/`, `data/config/models/deepseek-v3.json`, and the `latent_mla_attention` family exist.
- `DummyForwardBatch` and `BaseExecutionTimePredictorConfig.enable_dummy_mode` (`frontier/config/execution_time_predictor_config.py`) exist under those names.
- The DSA marker check in `frontier/attention/model_binding.py` recognises DeepSeek-V3.2 model types and `dsa_*` keys only.
- A nonzero `kv_send_slowdown_per_layer_ms` is rejected only when `kv_connector == "none"`.

### Risks to settle in ticket 01 (inferences, not yet measured)

- GLM-5.3 with dummy weights still allocates its full weight memory and may not fit one 8-GPU H800 node, so G0-B likely needs 16 or more GPUs.
- DeepSeek-V4-Flash's quantized expert kernels on Hopper may take a different path than the H200 recipe.
- Historical 2026-10-15 risk, now resolved at environment scope: H800_03 verified the locked image/CUDA compatibility, NCCL and NIXL READ environment. This is not full-model PD or EP32 qualification.
- The 64-GPU runs consume large GPU time. The canonical formal case per model keeps a GPU-hours ledger that includes initialization, compilation, warm-up, and failures; the superseded 36-cell matrix is not costed further (2026-10-21).
