# 09: Per-track basic runtime and G4 clean native qualification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-21 | Reconcile completed basic runs, verified P0 useful-token totals and the remaining MRv2 dummy/rank evidence. |
| 2026-10-21 | P0 H80032 finished; H20032 and H80064 basic validation launched. Fix staged-package shadowing before generation02;204+38 parent checks retain formal gates. |
| 2026-10-21 | D21: own all four native tracks in this existing ticket, distinguish basic readiness from formal clean qualification, and supersede H20064 non-preemptible staging. |
| 2026-10-21 | D19 cap130: qualification02 submitted as exp-1021-125354-923219;96 total reserved GPUs, Pending with identity/mount checks complete. |
| 2026-10-21 | Native stage02 merged db5c99f with585 parent passes; source review closed and64-GPU qualification waits for32-GPU reservation. |
| 2026-10-21 | D18: adopt common 68f891b artifacts; begin Step native consumer/runtime preparation while R2 qualification remains open. |
| 2026-10-21 | D17: canonical run uses 128 fixed 6144/1024-token requests at QPS 1.0, 16 separate warmups and three clean repetitions; short-prompt qualification deferred. |
| 2026-10-21 | D16: the case uses the frozen shared files with the historical linear-rescale arrivals; added the short-prompt P/D recording requirement (122 of 256 prompts under 32 tokens). |
| 2026-10-21 | Decision D15: the frozen case is 256 SharedGPT requests (first 256 in trace order) retimed to QPS 4.0, shared with the GLM/V4 session. |
| 2026-10-21 | Retargeted from H800-64 to the canonical 64-H200 case (decision D12): SharedGPT 128 requests at QPS 4.0, P32/D32 each TP8/DP4/EP32/PP1, MTP off, chunked prefill and prefix caching off. |
| 2026-10-20 | R1 passed on H800; the remaining prerequisite is ticket 08 R2 (G3 content and lifetime). H800 R1 stays as diagnostic evidence, not as formal-case evidence. |
| 2026-10-20 | Recorded decision D6: the clean repetitions may share one job and one startup. |
| 2026-10-20 | Recorded the D4 resource plan: H800 first with no GPU-count limit, H200 fallback of at most 70 GPUs. Superseded by D9–D14. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |

**What to build:** Separate basic-runtime and S03.G4 clean-native eligibility records for each D21 track in `../spec.md` §0. Clean qualification requires three repetitions, actual graph/NIXL execution and complete rank/request accounting; basic startup alone is not G4 or calibration.

**Blocked by:** Basic runtime uses `06.simulation_scope` and its own runtime package; it does **not** wait for another track's accuracy. Formal clean collection requires own readiness in `../plan.md` D21 graph, including case-aligned `08` G3. H80064 additionally waits for `28.release30(H80032)`; H20064 waits for `28.release30(H20032)`.

**External prerequisites:** D21 policy below; original personal local StepMind Python RJobBackend and current host/storage rules. Basic native validation has no GLM simulator barrier; predictive acceptance retains existing prerequisites. G3, actual capacity/placement and required runtime are not waived. TP16/DP1 startup does not qualify TP8/DP2/EP16 or TP8/DP4/EP32 roles. Each scale is its own target, never a reduced substitute for another.

**Status:** all four formal track records remain open. P0 H80032 generation03 `exp-1021-205523-285491` Succeeded;24 exact requests, both idle boundaries and actual FULL runtime rows on both D lanes are verified. H20032 `exp-1021-201544-965435` and H80064 `exp-1021-202325-579469` also Succeeded with24 exact requests and every lane idle, but their old multi-API logs lack runtime replay rows. Historical H20064 `exp-1021-125354-923219` provides retained basic startup/capacity/completion evidence with its original REFERENCE_UNSTABLE result. Every full basic/G4 result remains PARTIAL or open. No basic result establishes four-metric accuracy or a30% release.

**D31 status (2026-10-25):** Retargeted by D31.2. The active tracks are H200 32-GPU (plan steps C and E) and then H200 64-GPU (plan steps F and G), all non-preemptible. H800 tracks wait for quota and need a new native collection with the routing image.

**G4 useful-token checkpoint:** P0 generation03 logs account for147072 P tokens and24936 D tokens, exactly24x6128 and24x(16+1023). This closes aggregate useful-token conservation for the basic run, not request/rank/forward or dummy coverage. MRv2 only emits these graph statistics for non-dummy runs. Use the existing observer for the remaining real dummy/rank evidence; do not infer it from lane-count differences. Native FULL_AND_PIECEWISE and actual MRv2 dummy metadata/descriptor semantics must also bind the producer/consumer. Report: `../../task_2026-10-21_step5_priority_runs/native_replay/test_report_2026-10-21_g4_useful_accounting.md`.

## Current D21 milestones

| Priority / track | Job policy | Roles | Basic runtime | Formal clean collection prerequisite |
| --- | --- | --- | --- | --- |
| P0 H80032 | `codesign`, non-preemptible | P16/D16; each TP8/DP2/EP16/PP1 | Eligible now; readiness open | Own ready(H80032) |
| P1 H80064 | `codesign`, non-preemptible | P32/D32; each TP8/DP4/EP32/PP1 | Eligible now in parallel, no P0 starvation | Own ready(H80064) AND all four H80032 clean errors<=30% |
| P2 H20032 | `step_main`, non-preemptible within cap | P16/D16; each TP8/DP2/EP16/PP1 | Eligible now; readiness open | Own ready(H20032), independent of H800 |
| P2 H20064 | `step_main`, wholly preemptible | P32/D32; each TP8/DP4/EP32/PP1 | Eligible now; assess prior run first | Own ready(H20064) AND all four H20032 clean errors<=30% |

The two P2 labels are equal priority. Across all sessions, count every personal nonterminal reservation including queued jobs: H200 non-preemptible total<=32; every additional H200 job is preemptible even below obsolete80/130. Lower-priority H800 work cannot starve P0. `release30` requires four errors<=30% in each of three clean complete-population repetitions, with no averaging. It permits only the corresponding64 formal work; all four final metrics still require<=10%.

**Superseded package:** the old H20064 non-preemptible `/data/ycfeng/tmp/step5_d18_native_20261021/canonical02/` queued-to-formal plan is not launch-eligible under D21. Historical qualification02 already ended. Preserve both frozen stages; any needed H20064 run uses a new wholly preemptible generation, and formal collection waits for H20032 release30 plus own readiness. Stage01 remains prohibited. No blind qualification repeat, no source/native waiver.

**Execution record:** `../d18_native_20261021/stage_handoff.md` and `../test_report_2026-10-21_d18_native_integration.md`. Configured7168 context,8192 token budget,128 sequences, graph ceiling256, Step native utilization0.92; effective runtime still needs observation. Preserve complete process-labelled logs including native **Engine NNN** before graph tables. FULL runtime rows plus healthy completion support replay; full rank/forward accounting remains required without adding64 independent replay-counter requirements. Stage01 is prohibited.

**Decision D6 — 2026-10-20:** The three or more clean repetitions may run in one job with one engine startup. Each repetition replays the same frozen workload with its own warmup and measurement windows; the run records the window boundaries and checks that both roles are idle between repetitions (no running or waiting requests, KV cache usage back to 0). Startup-dependent variation is then sampled once per job, and the record states this.

**Preserved D17/D18 workload:** every track uses128 requests of6144/1024 at QPS1 over127 seconds,16 separate warmups and a drain before each clean repetition. MTP/chunking/prefix caching/Thinking Mode off; graph and quantization on; measured operators and `collective_sim`. Resolve per-track runtime/capacity rather than reusing R1's512-token/one-sequence settings. SharedGPT stays deferred.

**Source workstreams:** S03.
**Shared owners:** C01, C03, C05, C06 for identity, lifecycle, placement, and evidence.

## Acceptance

- [ ] Deliver S03.G4 with nested G0–G3 evidence, immutable clean-run manifests, all-rank/request accounting, capacity/topology observations, and explicit eligibility verdicts under §§0, 2, 6, 8, and 10; this is native qualification, not Frontier accuracy.
- [ ] Record basic and formal eligibility separately for each of the four tracks, including priority, hardware/scale, actual job preemptibility and dependency evidence. Neither64 formal track starts before its matching32 release30 and own readiness; basic validation may proceed now.
- [ ] Qualify each32 track on four8-GPU nodes as P16/D16 TP8/DP2/EP16/PP1 and each64 track on eight8-GPU nodes as P32/D32 TP8/DP4/EP32/PP1. Record node/NIC/NUMA mapping, role EP span and any concrete capacity/placement blocker; EP is not an extra multiplier. Evidence is not transferable without matching qualification.
- [ ] Execute the complete locked text backbone and expert set using dummy weights, unchanged routing, actual mixed precision/exclusions, MTP off, chunked prefill off and prefix caching off; preserve full state and measured capacity rather than shrinking the model to fit.
- [ ] Run shared `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/` from `68f891b` unchanged: 128 formal requests, 6144/1024 tokens, open-loop QPS 1.0 at offsets 0..127 seconds. Record hashes, producer identity, actual submission lateness and accepted/completed token counts. Stable template IDs seed the prompt; repeat-qualified transport IDs preserve cross-repeat prompt equality. Common-artifact adoption is complete under D18; Step consumer execution remains to be qualified.
- [ ] Verify canonical native transfer/tail behavior per track: R1 suggests6128 transferred prompt tokens and16 locally recomputed tokens; actual content, counts and release must be observed. Record occupancy, queues and batching on all two/four DP lanes per role, as applicable; QPS/request count is not utilization proof.
- [ ] Demonstrate online arrivals, **unchunked** prefill, actual decode graph replay, and qualified CUDA-buffer NIXL transfers on both roles; record native prefill captured pieces/eager breaks and resolved backend dispatch.
- [ ] Account for every role, rank, request-owner DP lane, useful forward, and required DP dummy forward; idle lanes must retain native synchronization even when useful payload is zero.
- [ ] Complete at least three clean native repetitions with identical declared workload/configuration identities, completed requests/tokens, terminal status, and separately retained observations of variation and native errors.
- [ ] Keep clean, observed, and detailed diagnostic evidence separate; exclude GPU/NIC profiling interference from clean repetitions and mark contaminated or incomplete runs ineligible rather than borrowing their timing as clean truth.
- [ ] Reject missing ranks/requests, fewer than three clean repetitions, disabled sparse/graph paths, nonfinite state, unsupported placement, or incomplete PD readiness/release evidence; successful process exit alone cannot qualify G4.
- [ ] For this ≥32-GPU workload, auxiliary validators are non-blocking soft gates: isolate errors, retain logs, notify the user, and mark affected evidence unverified without stopping healthy work solely for a harness error. Native failures retain normal handling and invalid evidence never becomes PASS.
- [ ] Leave the metric contract and the comparison itself to tickets `23` and `28`; G4 qualifies the reference only.

Specification references: §§0, 6, 8, 10.
