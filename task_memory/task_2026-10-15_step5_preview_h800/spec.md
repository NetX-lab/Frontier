# Specification: Step-5-Preview text inference simulation and calibration

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Apply D29 temporary balanced routing and workflow/GPU-before-CPU order. |
| 2026-10-21 | D22: attention agreement uses independently executed clean-window references; thresholds and historical verdicts remain unchanged. |
| 2026-10-21 | D21: four prioritized hardware/scale tracks, independent 32-to-64 formal release gates, basic/formal separation and global H200 non-preemptible cap32. |
| 2026-10-21 | D20 releases measured-path CPU implementation while retaining native/profile/predictive acceptance conditions. |
| 2026-10-21 | D18: adopt the byte-identical GLM/V4 workload from 68f891b (Step cherry-pick 958d81a); close reciprocal artifact adoption, retain separate native/runtime qualification. |
| 2026-10-21 | D17: SharedGPT deferred; canonical workload is 128 requests of 6144/1024 tokens, evenly spaced at QPS 1.0, with 16 separate same-shape warmups; runtime and cross-session adoption remain to be verified. |
| 2026-10-21 | Decision D16: arrival retiming is the historical dev-vidur linear rescale (scale 2.590246128, no seed), adopted with the shared frozen workload `data/processed_traces/canonical_sharegpt_req256_qps4/`; removed the open seed/distribution option; corrected prefix spans (first 128 = 31.0 s at 4.096 QPS); dropped the historical priority column; recorded that 122 of 256 prompts are under 32 tokens. |
| 2026-10-21 | Unified the canonical workload with the GLM-5.3/DeepSeek-V4-Flash session (decision D15): 256 requests (was 128), THPT gated as prompt-plus-output tokens per second with output-only and request throughput as diagnostics, shared trace identity and SHA-256, explicit arrival retiming with the rule left open for one decision, cross-task communication caveat, MTP-off source citation from the Step image. |
| 2026-10-21 | Applied the calibration scope reduction (decisions D9–D14): one canonical 64-H200 SharedGPT P/D-disaggregated online case with 128 requests at QPS 4.0, per-role TP8/DP4/EP32 over 4 nodes, MTP off in both native and Frontier, four metrics each gated at ≤10%, measured backend; retired the H800 W1–W6 campaign tickets 23–28 and retargeted 09 to H200-64. |
| 2026-10-20 | Applied the user's decisions D5–D7: assess the D-PD first-token and tail mismatch before aligning it (§8), allow the ticket 09 clean repetitions to share one job (§8), and keep Thinking Mode off (§2). |
| 2026-10-20 | Aligned the G2 simulation-scope finite criterion with the approved D1 draft: finite hidden outputs on all ranks. |
| 2026-10-20 | Applied the user's mainline review decisions: added the G2 simulation scope (§6), let downstream tickets and the Step structural PD/DP/EP work use it (§9), and replaced the 64-GPU limit with H800-first and an H200 fallback of at most 70 GPUs (§10). |
| 2026-10-20 | Linked corrected _04 readiness and dated resource observations; branch PR authority grants no new native or measurement qualification. |
| 2026-10-19 | Accepted FIFO/None source and CPU correction after independent review fixes; retained H800 Failed evidence, 626 distinct affected checks and unchanged native gates. |
| 2026-10-19 | Accepted resident-parameter candidate01 source/CPU tooling with657affectedtests,111CPUAPI and1554applicability checks;native06/whole15 unchanged. |
| 2026-10-19 | Accepted profiler candidate02 source/CPU tooling:498 affected passes, locked CPU API04 and parent two-axis reviews; native06/whole15 and formal gates unchanged. |
| 2026-10-19 | Accepted native state candidate01 source/CPU tooling:497 joint and140 affected caller passes overlap; parent Standards and independent Spec No findings; native06/whole15 unchanged. |
| 2026-10-19 | Synchronized final independent lifecycle review: 35 scoped checks and final artifact identities; initial 33-check bytes retained, acceptance unchanged. |
| 2026-10-19 | Archived completed request_device_20261019_01: 26,917 device checks and 35 scoped lifecycle checks PASS; current missing work is precision/physical/cache-state census, with06 1/10 and whole15 0/8 unchanged. |
| 2026-10-19 | Accepted candidate03 device/request/completion source and CPU tooling after final Spec follow-up, 357 affected passes and 234 source checks; native gates unchanged. |
| 2026-10-19 | Recorded source/CPU frontend admission acceptance and strict budget/failure guards; no native, accuracy, architecture or hardware acceptance changes. |
| 2026-10-19 | Recorded completed TP16 diagnostic requests/replay observations and sole06criterion7 acceptance; no fullG2, cleanperformance or predictive/hardware acceptance. |
| 2026-10-19 | Accepted generation05 controlled structural request evidence: 646 affected tests, 77 legacy cases, three CLI passes and eight CPU metrics within frozen budgets; native and whole-ticket gates remain open. |
| 2026-10-19 | Corrected current representation/shared-integration status and H800-first quota fallback; retained all native, accuracy and material-refactor gates. |
| 2026-10-18 | Updated accepted C04 integration and temporary H200 test resources; retained PRIMARY H80064 and all evidence thresholds. |
| 2026-10-18 | Recorded user confirmation and active implementation phase; preserved all formal acceptance and material-refactor gates. |
| 2026-10-18 | Corrected current status/reference identity, representation-first sequencing, conditional architecture approval, measurement dependencies and direct Python operations; formal gates unchanged. |
| 2026-10-15 | Created the plan-phase synthesis of five source documents, S5-Q1–S5-Q6, verified source discrepancies, and inherited task dependencies. |
| 2026-10-15 | Restored preregistered simulator wall-time/memory budgets and required explanations for over-budget regressions after planning review. |
| 2026-10-15 | User confirmed the specification and invoked To Tickets; detailed slicing is in progress. |
| 2026-10-15 | Completed the approved 30-ticket planning package and document verification. |
| 2026-10-16 | Linked current execution authority and accepted G0 identity while retaining the specification's unmeasured model/runtime boundaries. |

**Current contract:** [requirements.md](requirements.md) D21 and §0 below replace the single-H20064 restriction and every earlier GPU cap/order. D20 measured compute/bundle, PD and client-metric CPU implementation continues **P0-first**, before the external GLM implementation milestone; native/profile/predictive acceptance is not waived. The dated milestone sections before §0 are historical evidence, not current scheduling or resource instructions.

## Historical corrected native generation and publication scope — 2026-10-20

[Ticket06](tickets/06-native-single-role-qualification.md) records corrected `request_physical_20261019_04` at source `94d154e`, frozen **201 inputs / 16,439,703 bytes** and **122 native pins / 24 inert SDK checks / 543 readiness checks PASS**. Its platform **Running** observation is dated **2026-10-20 00:11:49 HKT / 2026-10-19T16:11:49Z**, for `exp-1020-000928-505270`, H800 `codesign` 2×8; later worker records show two distinct nodes and 16 distinct GPU UUIDs. Original budgets and all measurement gates remain. This generation gives no G2 or fresh qualified GPU measurement credit: **06 PARTIAL 1/10, criterion 7 prior-only; whole15 PARTIAL 0/8**.

Implementation remains active. [The latest original request](requirements.md) authorizes ordinary local commits and this branch's remote push/PR only; parent records actual publication results. Complete06 remains required for `15.native_reconciliation` and native07/08 ->09 H80064; single owners, GLM16–22 prerequisites and the final common branch are unchanged. Previous failures and source/CPU evidence retain their recorded scopes.

**Status: corrected specification accepted; implementation resumed on 2026-10-18 after `confirmed，continue`.** Use the standing execution authority in [requirements.md](requirements.md), the corrected staged graph in [plan.md](plan.md), and the recorded review evidence without promoting its unvalidated claims.

Tickets01–04 are DONE within their accepted scopes;04 is rank-local G1 only and05 remains PARTIAL0/10. Represented contracts, canonical shared merge1bde1999 and controlled structural03c7f823 are accepted only within their recorded limits. Current native diagnostic evidence and remaining gates follow below; formal predictive/PD/hardware/common-HEAD support remains open.

Historical source milestone before the completed device run: the reviewed frontend admission source/CPU slice is accepted in `frontend_admission_20261019_01/acceptance_02.json` and [its report](test_report_2026-10-19_frontend_admission.md): 275 affected CPU passes, 199 identity/applicability checks and Standards/Spec follow-up No findings. It observes original n=1 registration before dispatch and retains public/internal IDs, accepted prompt and resolved sampling; this is tooling capability only. Actual native frontend delivery, frontend-to-worker join, device/finite/completion, precision and complete physical/state evidence remain unqualified. Ticket06 stays1/10 and whole15 stays0/8. Active caller/observer now differ from the ended run; its immutable receipt/source snapshots remain historical. Frontend sources are committed at `60ab52d7ac04493d1cd6ec781c80f0006896ed44`; their native qualification remains open.

Historical source milestone before the completed device run: the device/request/completion source and CPU tooling slice is accepted in `device_request_witness_20261019_01/acceptance_03.json` and [its report](test_report_2026-10-19_device_request_witness.md). Candidate03 preserves original worker registration, finalized native inputs, owned asynchronous copies, complete valid hidden finite reduction and the existing post-generate event-query RPC. The parent verified 357 affected CPU passes, 234 retained source checks, 90 fresh archive/identity checks and six locked-image Torch CPU API checks. Standards candidate02 has zero documented violations or actionable smells; Spec candidate03 has no remaining delta finding after the 1–4-byte JSON response-cap correction. Acceptance is limited to source and CPU tooling. Actual CUDA/native frontend→worker→device delivery, precision/scale consumption, complete physical/cache-state census, full G2, PD, H80064, H200 and predictive/common-HEAD acceptance remain unverified. Ticket06 stays PARTIAL 1/10 (criterion7 only), and whole15 stays PARTIAL 0/8. The old executed request source/receipt is immutable historical evidence. Next: commit the six reviewed files, verify Git=current=frozen, then finalize and run the distinct `request_device_20261019_01` generation under the existing H800-first policy and budgets.

The distinct `request_device_20261019_01` / `exp-1019-075927-372034` completed Succeeded at **2026-10-19T00:08:32Z / October 19, 2026 08:08:32 HKT**. Source HEAD `7962f5fe8199a6884f6c1628f7a9a1bf8faeb609` and receipt SHA `f2d2aa4f3427a8265e4d0b4acfa55b4add7091930ba2044dd40efcc33610e049` are frozen. [The new report](test_report_2026-10-19_native_request_device.md) records **26,917 retained device checks PASS**, **35 scoped source/lifecycle checks PASS**, original public→internal→worker→device association on all16 ranks, 544 real device stages and256 actual FULL replay calls, 539,099,136 valid hidden finite elements, and ready observer-owned copy events. Full precision/scale consumption, physical kernel and cache-state census remain missing. **06 is PARTIAL1/10, criterion7 only; whole15 is PARTIAL0/8.**

The executed same-thread/copy diagnostics do not establish complete per-layer scale consumption, kernel census, cache occupancy, clean timings, native HTTP online, PD or primary/H200 acceptance. Final nodes/archive/residuals pass within existing bounds, with native force-kill/SIGKILL and JIT/shutdown warnings retained. The remaining census plan is an incremental native instrumentation proposal, not a new Frontier architecture. Source/CPU and old-run paragraphs above are historical milestones; the completed device report supersedes their pending-device next actions.


## Resident-parameter source/CPU milestone — 2026-10-19

`native_parameters_20261019_01/acceptance_01.json` and [the report](test_report_2026-10-19_native_parameters.md) accept candidate01 source/CPU tooling only:657affectedCPUtests,111locked-imageCPUAPIchecks and1554source-applicabilitychecks. Canonical registered tensors, complete per-view finite counters, labelled raw samples and alias validation reuse the existing reservation/stream/event/drain with the original deadline. Two source defects are corrected; both final reviews are parent reviews because independent429requests produced no reports. Historical simulator evidence is reused only for unchanged source paths;raw benchmark artifacts were not all rehashed, and current observer overhead/CUDA/pinning/headroom remainsUNVERIFIED. **06 stays PARTIAL1/10,criterion7only;whole15 stays PARTIAL0/8.** Next:exact six-file commit ->remaining transient-scale/write/occupancy and complete readiness ->distinct native missing-witness generation ->06 ->15.native_reconciliation. Shared/C06/native singleowners,GLM16–22gate,final commonbranch andH800first/H200<=64non-preemptible policy remain.


## Native profiler source/CPU milestone — 2026-10-19

`native_physical_20261019_01/acceptance_02.json` and [the report](test_report_2026-10-19_native_profiler.md) accept candidate02 source and CPU tooling only. The existing native caller/observer uses the original Torch wrapper, capture/request windows, compact annotations, one tensor/trace reservation and detached `graph_launch_chains` analysis. Capture binding requires an actual positive executable identity and valid continuity; auxiliary response failure preserves original native capture and exceptions. No Frontier core or native model/router/sampling/graph-policy change occurred.

Final seven affected suites498PASS/33.05s;114profiler-only passes are included in498. Locked-image API04 passes original CPU wrapper windows and a31-byte cap with CUDA uninitialized. Current source/API/applicability810checks and retained4669checks pass within their scopes;77scenario/902hash/451pair/24CPUattempt/eight-budget evidence is reused only for unchanged simulator paths. Standards and Spec are both parent reviews with0remaining delta findings after the independent requests failed429; no independent reviewPASS is claimed. All RED/failure/timeout records and candidate01 remain immutable.

**06 stays PARTIAL1/10, criterion7 only; whole15 stays PARTIAL0/8.** Replay launch correlation and executable association are not graph-node/layer attribution. Full parameter/scale consumption, transient replay content, cache write/occupancy, current native compatibility and observer overhead remain unverified. Next: exact eight-file commit/source check, remaining full-parameter/transient-scale/write/occupancy readiness, then complete source/API/byte/deadline readiness before a distinct missing-witness generation. H800/codesign first; actual insufficient quota permits H200/step_main<=64total GPUs, all non-preemptible. Shared core/C06 and native observer/caller retain single owners;16–22 retain GLM09.full_frontier_path and final branch remains `feat/glm53-dsv4-vllm028`.

Native profiler source commit: `8bf0315a6b483d0a51daed214012a58398fd1d60`. `native_physical_20261019_01/post_commit_02.json` verifies all8Git/current/frozen sources, exact8changedpaths and parent `291ba7b713d09bc448924499df70adaab16e69dd`. Tracked worktree clean; unrelated `.pi/` untouched. No push or ignored-record addition. This is source identity only;06PARTIAL1/10criterion7 andwhole15PARTIAL0/8 remain. Current work is remaining full-parameter/transient-scale/write/occupancy and complete readiness before a distinct native missing-witness generation.

## Native state source/CPU milestone — 2026-10-19

`native_state_20261019_01/acceptance_01.json` and [the report](test_report_2026-10-19_native_state.md) accept candidate01 source/CPU tooling only. Actual per-group finalized tables, persistent-index active counts, original assignment/removal/reset/release/copy seams, known-owner storage/aliases and candidate SWA/DSA state plus resident-scale content use the existing observer/caller, single reservation, stream, events and drain. Missing assignments, replaced owners, nonfinite content and wrong device joins remain explicit; finite bytes are not native write or scale-consumption proof. All18 census source identities enter the existing future preflight ledger:11 historical matches plus7 newly bound retained bodies, with no conflicts. Old native generations and inputs remain immutable.

Verification: attempt09 six suites497PASS/31.20s, followed by changed caller attempt11 140PASS/22.04s; these counts overlap and do not describe637distinct tests or a final506-test joint run. All11 attempts and four original fixture/assertion failures are retained;01–08 exact command receipts and09 per-command source snapshot were unavailable and not reconstructed. Source checks80/80 and42parent identity/log checks pass. Standards is parent review with0findings after delegated429retry exhaustion; independent Wegener Spec has0actionable findings. Retained77scenario/902artifact/451pair/eight-budget applicability remains limited to unchanged simulator paths, with official provenance/cache/drift UNVERIFIED. The9lockedTorch checks establish primitive CPU applicability only; the complete current child/CUDA path and observer overhead remain unexecuted/unverified.

**Ticket06 PARTIAL1/10, criterion7 only; whole15 PARTIAL0/8.** Next: scoped source commit and Git=current=frozen -> focused existing native-profiler capture/replay physical correlation -> complete source/API/bytes/deadline readiness -> distinct missing-witness generation -> complete06 ->15.native_reconciliation. No ended request/capacity/FFN/Attention/source-recovery campaign is repeated. Parent owns native observer/caller/platform/audit; C06 and each shared core module retain one canonical owner.16–22 retain GLM09.full_frontier_path; final branch is `feat/glm53-dsv4-vllm028`. H800/codesign first; actual quota shortage permits H200/step_main<=64total GPUs, all non-preemptible. Only an evidenced material core refactor or materially unresolved semantics/acceptance needs explicit approval.


Native state source commit: `291ba7b713d09bc448924499df70adaab16e69dd`. `native_state_20261019_01/post_commit_01.json` verifies all nine Git/current/frozen files, exact changed paths and parent `3c9bff8332376cb9165af66649403beefd87c713`. Tracked worktree is clean; unrelated `.pi/` remains.83documentation/criterion/artifact checks pass. No push or force-add of ignored records occurred. Source/CPU-only acceptance leaves06PARTIAL1/10andwhole15PARTIAL0/8. Parent continues focused existing native-profiler capture/replay correlation and remaining qualification readiness.

## 0. Current calibration tracks (decision D21, 2026-10-21)

One frozen workload is evaluated on **four hardware/scale tracks**, as milestones in existing tickets 09/23/28/30, not new tickets or a revived workload sweep. D21 supersedes D9–D14's single-H20064 restriction and D19's130-GPU cap. D17 lengths/arrivals, D18 common files, D15 metrics and existing native/profile admission continue. This section is normative; prior decisions remain historical evidence where superseded.

| Priority / track | Hardware / job policy | Topology | Basic runtime validation | Formal calibration release | Final gate |
| --- | --- | --- | --- | --- | --- |
| P0 H80032 | `codesign`, 32 H800, non-preemptible; highest priority | P16/D16, each TP8/DP2/EP16/PP1, 2 nodes per role | Eligible now under native prerequisites | Own readiness; no other track's accuracy gate | All four errors <=10% |
| P1 H80064 | `codesign`, 64 H800, non-preemptible | P32/D32, each TP8/DP4/EP32/PP1, 4 nodes per role | May proceed now in parallel without starving P0 | Own readiness **and all four P0 H80032 clean errors <=30%** | All four errors <=10% |
| P2 H20032 | `step_main`, 32 H200, non-preemptible within global cap | P16/D16, each TP8/DP2/EP16/PP1, 2 nodes per role | Eligible now under native prerequisites | Own readiness; **independent of H800 accuracy** | All four errors <=10% |
| P2 H20064 | `step_main`, 64 H200, whole job preemptible | P32/D32, each TP8/DP4/EP32/PP1, 4 nodes per role | May proceed now in parallel | Own readiness **and all four P2 H20032 clean errors <=30%** | All four errors <=10% |

Both P2 labels have equal user priority. Basic validation establishes startup, capacity, placement, required runtime/graph/NIXL behavior and fixed-length completion; it is not formal clean collection, fitting or a predictive accuracy claim. Own readiness includes qualified G3/runtime, frozen case, admitted hardware/scale-compatible profiles and independent measured predictor inputs, plus the existing acceptance prerequisites. The 64-GPU release checks apply before formal calibration/clean collection starts, not only before publishing a verdict. Three clean repetitions with complete populations establish the 32-GPU release evidence: **each of the four errors in each repetition must be <=30%**. Equality passes. No average across metrics, hardware or repetitions may hide a failure. The 30% release threshold does not replace final10%, release H200 from H800, or establish final support.

All personal nonterminal H200 reservations across **all sessions**, including queued jobs, count toward **non-preemptible total <=32**. Every additional H200 job is preemptible even below the obsolete80/130 thresholds. A future H20064 run is wholly preemptible, never split. H800 work uses non-preemptible `codesign`; lower-priority work must not consume resources needed by ready P0 work. Distinct hardware/scale manifests, native evidence, profile/bundle identities and outputs are required. No timing scaling or cross-track accuracy substitution is permitted.

**Initial D21 reconciliation:** the parent recorded a complete personal list with0 nonterminal jobs at09:02UTC. H20064 qualificationexp-1021-125354-923219 Succeeded08:20:59UTC and H20032 communicationexp-1021-143708-523626 Succeeded08:15:05UTC; no incompatible live job needed stopping. Business judgments remain pending. Preserve historical receipts/stages; the old non-preemptible H20064 `canonical02` queued-to-formal plan is superseded, not resubmitted as-is. Assess reusable64 basic evidence first; any needed future64-H200 generation is wholly preemptible and formal work waits for H20032 release30 plus own readiness. Obsolete maxlen2048 R2 remains paused; case-aligned G3 is not waived. Exact package paths and owners are in [plan.md](plan.md).

| Dimension | Frozen requirement |
| --- | --- |
| Model | Step-5-Preview, text backbone only |
| Ground truth | Image tag `vllm/vllm-openai:step-5-preview`, resolved to an immutable digest before formal collection; the currently locked linux/amd64 manifest `sha256:69308454…6a7f20` is re-verified against the tag, not assumed to match |
| Backend | `collective_sim` for communication; **measured** operator times through the existing profiling-CSV admission and trained-predictor path |
| Workload | **128 formal requests**, each **6144 prompt / 1024 output tokens**; **QPS 1.0**, open-loop evenly spaced arrivals `arrived_at_i = i` seconds for `i = 0..127`, no arrival seed. Files: `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/`, frozen by `68f891b`, adopted as `958d81a` (D18). SharedGPT is deferred (D17) |
| Hardware | Per the four-track priority table above |
| Architecture | P/D disaggregation, online serving |
| Topology | 32 total: P16/D16, each TP8/DP2/EP16/PP1; 64 total: P32/D32, each TP8/DP4/EP32/PP1. EP spans the role's TP×DP ranks, not extra GPUs |
| Weights | Dummy load format |
| Thinking mode | Disabled (D7) |
| Prefix caching | Disabled |
| Chunked prefill | Disabled |
| CUDA Graph | Enabled, actual decode replay |
| Quantization | Enabled, native FP8/E4M3 with the recorded exclusion list |
| MTP | **Off in both native and Frontier** (D11) |
| Metrics | Client TTFT, TPOT, THPT (prompt + output tokens per second), per-request E2E — each independently `abs(frontier - vllm) / abs(vllm) <= 0.10` (TTFT, TPOT, E2E on their means over the formal requests; THPT over the frozen window) |

**Four tracks, one workload, no sweep.** Retired W1–W6 campaigns, QPS sweeps, additional topology families and independent E2E holdout matrices stay retired. Repeated measurements and targeted diagnostics inside the four tracks remain allowed. Freeze aggregation and the full request set before comparison; never discard a difficult request or adjust aggregation to reach either threshold.

**Metric contract.** Client TTFT is client submission to the first client-visible token on one client clock; it is not the internal prefill-completion latency (`ttft_ms` remains that internal meaning, per D5 and ticket 14). TPOT is mean output-token interval after the first token. **THPT is total token throughput: prompt plus output tokens per second over the frozen measurement window** (decision D15) — the existing `token_throughput_tps` of `tests/comparison/e2e_gate/README.md`, with the window running from the first actual formal client submission to the last formal completion on each system's corresponding clock. All elapsed time inside that window remains included. Output-only tokens per second and requests per second (`request_throughput_rps`) are recorded and reported on the same window as **non-gating diagnostics**. E2E is per-request submission-to-completion latency, not makespan. `SPOT` in the earlier material is read as TPOT; no separate fifth metric exists. Clock normalization and evidence admission reuse `tests/comparison/e2e_gate` rather than a second gate.

**Fixed-length workload and rationale (D17).** Here 6k/1k means 6144/1024 tokens, not characters or words. The agent selected 128 formal requests at QPS 1.0 under the user's delegated authority. The 127-second arrival span is about twice the previous 63.75-second span, with 786,432 prompt tokens, 131,072 output tokens and 917,504 total formal tokens per repetition. Offered input is 6144 token/s, about 6.73 times the old SharedGPT case's `4 × 228.289`; offered output demand is 1024 token/s, close to its `4 × 279.715 = 1118.86`. Reducing QPS avoids multiplying both input and output pressure when replacing the short prompts. Fixed spacing isolates length-dependent compute, sparse attention and transfer costs from burst variation. This is a controlled calibration workload, not a production-distribution claim or proof of saturation. The earlier R1 timings use chunked prefill and TP8/DP2/EP16; they cannot predict unchunked TP8/DP4/EP32 throughput. Actual lane occupancy, queue growth, batching, completion time and repeat variation are recorded on H200, with every formal request retained.

**Replay and exact token counts.** Use the existing shared builder `tests/integration/canonical_case/build_fixed_manifest.py` and its frozen `trace.csv`, `request_ids.json`, `warmup_pool.csv`, `manifest.json` and `SHA256SUMS`. Native `/v1/completions` receives deterministic token-ID prompts from the existing prompt producer, seeded by the stable template request ID, with 6144 IDs, no chat template or extra special tokens, `min_tokens = max_tokens = 1024`, `ignore_eos = true`, and streaming enabled. Transport IDs follow `{request_id}-repeat{repeat:02d}` and never change prompt content. Freeze the prompt producer and model vocabulary identity before native execution and check actual accepted prompt and final output counts; no truncation, early EOS or output-chunk counting may substitute for exact lengths. The shared files fix lengths, IDs and arrival times; model vocabulary mappings remain explicit. Replay is open-loop, never paced by prior request completion, and dispatch lateness is recorded.

**Warmup, windows and required serving capacity.** Each of three clean repetitions gets 16 same-shape warmup requests at QPS 1.0, using IDs disjoint from the 128 formal IDs. After warmup both P and D return to idle (running/waiting zero and KV usage zero); only then establish a fresh formal time origin. Repeat the same schedule and inputs with separate output directories, and drain before the next repetition. Warmup count does not prove every lane or graph shape is warm; record lane participation and actual decode replay. Include all 128 formal requests in all four metrics; TPOT is the mean of each request's `(E2E − client TTFT) / 1023`. THPT uses all 917,504 tokens divided by the interval from the first formal client submission to the last formal completion on each system's corresponding clock; the 127-second arrival span alone is not its denominator. Native unchunked configuration must satisfy `max_num_batched_tokens >= max_model_len >= 7168`; freeze a concurrency-capable `max_num_seqs` and graph capture policy in ticket 23. The R1 values (512 batched tokens, one sequence) are diagnostic settings and cannot be reused as canonical capacity qualification. Frontier must explicitly use trace replay with `trace_request_generator_config_max_tokens=7168` and all three scale factors equal to 1.0; its default 4096-token limit would silently shorten the prompt. Verify the generated Request objects retain the full lengths and schedule before simulation.

**Deferred SharedGPT record.** Preserve `data/processed_traces/canonical_sharegpt_req256_qps4/`, source pins, D16's historical linear rescale and its successful equivalence checks unchanged. Its length distribution and retiming are no longer active requirements. The 122 short prompts, including 73 below 16 tokens, remain an unmeasured future coverage item rather than a prerequisite of this fixed-length case. The earlier claim that the extrapolated NIXL formula yields zero below 16 tokens was incorrect: it can become negative outside its observed range and must not be extrapolated. For 6144 tokens, the observed-range rule suggests 6128 transferred and a 16-token D tail; tickets 08/09 must verify this on the canonical native path, and ticket 10 prices the tail for D5.

**Shared workload with the GLM-5.3 / DeepSeek-V4-Flash session (D18).** Both sessions now reference the package from `68f891b`; Step cherry-picked it as `958d81a`. Manifest SHA-256: `5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3`; SHA256SUMS-file SHA-256: `594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9`. The reciprocal adoption request is closed; Step consumer integration and native qualification remain separate work. The independent D17 package `canonical_fixed_6k1k_req128_qps1/` is retained unchanged as historical evidence and is not an active input. This session does not edit the other worktree or its scratch files. The communication backend intentionally differs: this task uses `collective_sim`, the GLM/V4 session uses its measured CC backend. **Communication error is therefore not directly comparable between the two published results**; every place that compares results across the two tasks states this.

**Backend and hardware caveat.** Each track requires its own actual hardware/scale-qualified compute, communication and native evidence. Existing H800/H200 collections may be assessed and reused only when identity, physical scope, topology and quality match; raw or unadmitted data is not upgraded by D21. A qualified 32-GPU result is formal evidence for that32 track, never for64. Existing64-H200 basic-runtime evidence may be assessed rather than blindly repeated; it cannot establish formal calibration or the H20032 release condition.

**Resolved native runtime questions.** Whether the image actually runs MTP, which connector is active, effective `--max-model-len`, graph policy and quantization behavior are resolved from the image and its serving command and recorded with provenance before formal collection; a tag alone is not evidence (calibration §4). The MTP default is resolved at source level for the locked Step image (vLLM `0.1.dev22912+g8224bb82f`): `vllm/engine/arg_utils.py:707` defaults `speculative_config` to `None`; `create_speculative_config` (`:2048–2087`) returns `None` unless `--speculative-config`, `--spec-method`, `--spec-model` or `--spec-tokens` is given; the only automatic path (`:2217–2226` → `vllm/transformers_utils/config.py:756–791`) requires `speculators_config` in the model `config.json`, which the locked reference `config.json` (SHA-256 `e535c5c9…24bb63`) does not contain. The GLM/V4 session found the same in vLLM v0.28.0 (`vllm/engine/arg_utils.py:1047-1085`; line numbers differ between the two builds). The frozen serving command must therefore carry none of those four flags. If an essential part of that runtime cannot be executed, the recorded outcome is a concrete compatibility blocker plus the smallest viable resolution — never a silent substitution of a different implementation or a reduced-feature configuration. of that runtime cannot be executed, the recorded outcome is a concrete compatibility blocker plus the smallest viable resolution — never a silent substitution of a different implementation or a reduced-feature configuration.



[requirements.md](requirements.md), including decisions S5-Q1–S5-Q8, overrides source proposals. This specification preserves the source drafts unchanged.
Canonical terms come from repository-root [GLOSSARY.md](../../GLOSSARY.md). `docs/adr/0001-dummy-weight-native-ground-truth.md` explains the inherited dummy-weight trade-off; its recorded GLM/DS scope is not silently rewritten as Step qualification.
Use **Replica**, **DP lane**, **dummy-weight native run**, **DP dummy forward**, **dummy timing mode**, **internal TTFT**, and **client TTFT** with those meanings.

| Read-only input under `source_spec_s5/` | Role |
| --- | --- |
| `01_Step5_Spec.md` | Proposed model, state, profiling, campaign, and acceptance contracts |
| `02_Step5_Tickets.md` | Proposed S00–S10 workstreams and source-only optional scope |
| `03_Shared_Integration.md` | C01–C06 single-owner integration responsibilities |
| `04_Source_Evidence.md` | M01–M09 / F01–F07 source ledger and unresolved reference identity |
| `05_Campaign.yaml` | Planning template; not an executable CLI configuration |

[test_report_2026-10-15_source_documents.md](test_report_2026-10-15_source_documents.md) records five byte-identical source/snapshot pairs, three matching inherited-input hashes, and six matching Git blobs at the reviewed commit. The exact SHA-256 manifest remains in `requirements.md`.
Those checks establish document provenance, not the truth of every source interpretation or a native measurement. All five documents are supplied. Ticket02 subsequently locked metadata and observed image/environment identity; see `reference_lock.json`, its preserved `reference_lock_2026-10-16_observed.json` supplement and ticket02 acceptance. G1 is accepted as rank-local evidence in `g1_package_acceptance_01.json`. G2 is partial; G3/G4 and predictive model support remain unqualified.

| Identity | Recorded value and boundary |
| --- | --- |
| Selected Step branch/base, S5-Q1 | `feat/step5-preview`, based on `feat/glm53-dsv4-vllm028@01fb2ce5984f1d26135ea9bceef96f8e39be0401` |
| Step worktree | `/data/ycfeng/Frontier/.worktrees/step5-preview` |
| Source-draft reviewed main | `7df62a298d489dbff5b2381ca1a86d7c388fac3d`; different from the selected base |
| Historical inherited review | `9a650fba342a0da835e162d6f030f056e7bc9211`; not the current execution identity |
| GLM/DS native reference | vLLM `v0.28.0@2cf0a6915ce544dc493a0990f2ea38d81601128a`, unchanged by this task |
| Step reference | Installed vLLM `0.1.dev22912+g8224bb82f` (not the GLM/DS v0.28.0 reference). Official discovery tag `vllm/vllm-openai:step-5-preview`; linux/amd64 manifest `sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20`; observed package/source identities in the lock supplement, full build SHA unknown |
| Locked metadata revision | `stepfun-ai/Step-5-Preview@364b3b03fed2477aa0953fe855a2c4c8c48ee3cc`; three-file acquisition manifest and exact hashes in `reference_lock.json` |
| Public source candidate | `jeejeelee/vllm@02e28ab2a3c4a87bc51ece1b092c290297897884`, `SOURCE_CANDIDATE`; image correspondence unknown |

The stacked branch does not change the GLM branch owner or imply that implementation commits were merged. Future integration must record actual source HEAD, dirty paths, shared owners, and accepted changes; the final three-model regression must identify one common source HEAD.

## 2. Objective and scope

PRIMARY is **P0 H80032** of §0. All four tracks require full text-backbone predictive online E2E, P/D disaggregation, PP1, chunked prefill **disabled**, actual decode CUDA Graph replay, D29-matched balanced token-to-expert routing and the official FP8 configuration's resolved mixed precision. Historical dummy-weight runs retain their original routing identity. D21 restores H800 formal calibration without reviving the retired W1–W6 campaign.
Frontier remains CPU-only. GPU profiling and native serving run in their respective qualified image processes; simulator imports must not acquire a vLLM or torch dependency.
The full text model, expert count, quantization exclusions, and required state must not be reduced to obtain a passing run. Dummy weights replace checkpoint acquisition, never execution time, kernels, communication, or allocation.

### Scope exclusions and claim boundaries

- Vision, video, and all multimodal inputs, encoders, projectors, preprocessing, embeddings, caches, and serving support are excluded **now and from this task's future roadmap**. Source X01 is removed, not deferred; no corresponding work item or future ticket exists in this plan.
- Native text-serving residency must be measured truthfully, including any wrapper-associated resident modules. This accounting does not implement vision. No loader switch is assumed, no native module stripping is authorized, and no memory saving is inferred from text-only scope.
- Text MTP, prefix caching, 1M context, and extra parallel/topology support remain outside PRIMARY. They require later explicit scope approval; no future tickets are created now. Declared MTP layers are inventoried separately from executed backbone layers.
- DBO/ubatching, PCP/DCP, EPLB, offload, PD-AF, and B300-specific MegaMoE optimization are outside PRIMARY. Native capability restrictions cannot be removed to claim support.
- Thinking Mode is off in every run of this task, native and Frontier (decision D7). Step structural execution rejects it with `ValueError` (`frontier/config/config.py:596`); this is the run-harness boundary, not a gap to close.
- Support wording must state text-only, MTP off, prefix caching off, Thinking Mode off, measured hardware/topology, and validated serving-context range. No checkpoint-quality or production-routing claim follows from random weights.

Actual native/profile/predictive validation is required separately for all four tracks. H20032 proceeds independently of H800; each64 formal track waits only for its corresponding32 release evidence plus its own readiness. S10 and overall support remain open until all required track and common-source gates pass. A basic-runtime or smaller-track result cannot close another track's acceptance.

## 3. Full text-model and precision contract

The following are **source-derived fixture expectations**, not native measurements or permanent runtime constants. S01 must lock raw metadata; S02 records source inventory; S03.G2 reconciles the resolved model against it.

| Domain | Expected fixture | Required treatment |
| --- | --- | --- |
| Configuration boundary | Outer `Step5ForConditionalGeneration` / `step5_vl`; nested `Step5ForCausalLM` / `step5` | Unwrap `text_config` once at the existing model boundary while retaining the outer FP8 policy and complete exclusions; do not bind Step3/MFA or generic dense GQA |
| Layer range | 92 backbone layers; 3 declared next-token prediction layers | Use the declared backbone range, not raw array length; MTP remains off |
| Attention geometry | Hidden 4096; 64 Q heads; 4 KV groups; explicit head dimension 192 | Q projection width 12288 is a fixture check; `hidden/Q_heads` is not the head dimension |
| Attention schedule | 69 SWA and 23 sparse-GQA layers; three SWA then one configured full-attention layer repeatedly | Resolve sparse semantics from configuration and qualified implementation |
| Window | 512 tokens | Separate visible history, allocated pages, and transferable pages |
| FFN schedule | 88 MoE layers, IDs 3–90; dense IDs 0, 1, 2, 91 | Preserve the final dense layer; reuse dense/routed/shared contracts |
| FFN geometry | Dense width 13824; routed/shared width 1536; 352 experts, top-8 | Respect per-layer and per-rank placement, padding, precision, and sharing |
| Selector | Proxy dimension 256; 16 index heads, 1 index KV head; 4 provider groups; top-512 regions of 8 tokens | Region, token, and KV-page units remain distinct; provider groups do not multiply GPU count |
| Precision | Dynamic FP8/E4M3, 128×128 blocks, complete `modules_to_not_convert` list | Preserve native exclusions, including routed experts in layers 88–90, and actual parameter-name mapping |
| Norm/residual/gate | FP32 norm/residual configuration and headwise gate | Inventory actual dtypes, casts, and fused physical scopes independently of weight precision |
| Configured context limit | 1,048,576 | Keep model position limit separate from the smaller, jointly frozen serving limit |

All varying model/hardware/workload values must come from validated configuration, existing registries, or independent measurements. Missing required information fails explicitly; names remain labels, not hot-path classification logic.
At TP8, four logical KV groups do not become 0.5 local KV heads. Verify KV/provider replication and projection shapes against native placement. `352/32=11` is only an expert-placement arithmetic check, not a measured capacity result.
Keep logical unique bytes, per-rank and aggregate resident bytes, and actual PD wire bytes separate. Replicas and DP lanes retain existing request ownership; TP/EP ranks are not additional schedulers.

## 4. Existing architecture and minimum extensions

The previous Frontier implementation was built around vLLM 0.10.2. The target runtime differences require model/operator bindings and workflow corrections where semantics differ; they do not require copying native modules or replacing the DES architecture. Reuse existing semantics where the inspected source and native evidence support them, and isolate model-specific differences in small extensions.

Use one model normalization boundary, global layer IDs, existing layer bindings, ordered `StageExecutionTime`, profiling/training dispatch, and existing scheduler/event/allocator ownership.
Attention identity belongs to `frontier/attention/model_binding.py`; `LayerContractSpec` and `frontier/execution_time_predictor/layer_contract_resolution.py` are FFN-domain contracts, not an alternative attention identity owner.
`LayerAttentionSpec` currently resides in `frontier/attention/gdn/config.py` and remains the sole type. Extend it only for a demonstrated shared requirement; a move, compatibility-export layer or generic graph is not a prerequisite. A necessary material refactor follows §9.
Attention specifications are in `frontier/attention/ops.py`; families are selected through `frontier/attention/families.py`; common operator definitions/registration are in `frontier/operators/spec.py` and `frontier/operators/registry.py`.
The earlier review HEAD used homogeneous attention outside Qwen3.5. Accepted C02/C03 now supplies model-owned per-layer bindings and GLM component capacity; structural C04 is integrated at Step merge `ff3960e4dbaf99d6ab3959cea62dd102d699b513`. The bounded pinned-source audit shows the canonical one-pool/group-table contract can express Step raw aliases and sparse sidepages for the equal-raw-page initial scope. Generation05 connects source-derived TP selection, fixed scratch, legal block binding and parameter bytes through existing interfaces for the controlled MONOLITHIC/PP1 scope. Native allocation, transfer and measured timing remain separate gates. Reviewed shared ba48220 integration is separately recorded in step5_shared_merge_acceptance_20261019.json; no later shared dirty change is implicitly accepted. Neither a first/majority-layer substitute nor applying GLM token-index reuse to Step region selection is valid.

| Responsibility | Existing boundary and intended extension |
| --- | --- |
| Model/FFN/precision normalization | `frontier/model_architectures.py`, model config, FFN resolution; add Step metadata mapping and per-layer precision inputs |
| Attention family and workload | Attention binding/families/ops; minimal sparse-GQA and SWA features, not a generic model DAG |
| Cache descriptions | `frontier/attention/memory.py` and existing KV interfaces; component-specific resident/wire calculations |
| Physical measurement | Existing attention/linear/MoE wrappers and `frontier/profiling/common/vllm_compat.py`; call qualified native kernels |
| Timing and execution | Existing predictors, stage timing, queues, held outputs, EP work, and request-level events |
| Publication and acceptance | Existing dataframe loaders, family trainers, model identity, publisher, and `tests/comparison/e2e_gate/run_gate.py` |

Potential Step config/workload leaves or a native attention wrapper are design candidates, not mandatory new files. Inspect the then-current shared tree before assigning paths. No second allocator, NIXL manager, event queue, scheduler, predictor cache, CSV publisher, benchmark framework, or verdict engine is permitted.
Interface reuse, semantic reuse, wrapper reuse, and timing-table reuse are four different claims; each needs its own compatibility evidence. Step region summaries are not DS compressed KV, Step region selection is not GLM token-index reuse, and the headwise sigmoid gate is not DS mHC.

### Physical cost ownership

Cover packed QKV/indexer/gate projection; norm/RoPE/residual/casts; raw KV writes; summary updates; selector scoring/selection; sparse attention and SWA; staging/gather/packing; output projection; dense/routed/shared FFN; communication; and host preparation/graph launch.
Each physical scope contributes duration and kernel count exactly once, even when it covers several semantic operators. Fused projection must not add a second indexer GEMM; fused decode gate/summary work must not be charged again.
Preserve distinct costs through existing registry mappings where necessary; do not overwrite another family's timing fields or add a fusion-graph interpreter.
Predictive features may include local Q/KV/provider geometry, valid/padded rows, window visibility, region remainder, physical score/table extent, selected-region bounds, KV layout/dtype, staging path, fusion, and graph identity.
Do not store full Q/K/V, exact per-token top-k lists, or native `AttentionMetadata` in the simulator. Kernel-internal tiles remain cost-model details, not DES events.

## 5. Cache, PD, and execution semantics

| Component | Resident accounting | PD treatment to establish from G3 |
| --- | --- | --- |
| Sparse-GQA raw K/V | Native full-history page pool, local heads, precision, replication | Actual descriptors and rank mapping; never divide raw KV by region size |
| SWA K/V | Native window/page allocation and reuse rules | Actual transferable range; neither all history nor exactly 512 tokens is assumed |
| Closed-region summaries | Native FP8 side storage per applicable layer | Confirm actual connector registration and transfer view |
| Open-region state | Accumulators, counts/positions, last-slot state | Determine transfer, retain, or recompute disposition from native evidence |
| Padding scratch/workspaces | Graph padding, logits, pack/gather, prefill staging and peak reuse | Capacity/cost only unless actual protocol requires otherwise; no valid-token credit |
| MTP-specific state | Account for any measured residency separately | No MTP execution or journal workload is assumed in PRIMARY |

Selecting 512 eight-token regions can reduce attention reads; it does not compress the retained raw sparse KV to one eighth. Candidate summary-budget formulas and workspace thresholds are implementation-specific, awaiting image matching; they are not general runtime formulas.
Component identity, alias ownership, resident bytes, wire bytes, disposition, and readiness conditions belong in the shared C03 contract. Model leaves describe state; the existing transfer owner controls submission, readiness, and release; the measured backend supplies cost.
Maintain one source pending owner. Target reservation must precede READ where required; network completion must not falsely imply GPU-visible readiness; source pages stay occupied until the approved release boundary. New D-ready/P-free boundaries require distinguishing native evidence. Record routine reuse in D-PD; request approval only if the required change is a material core refactor or materially unresolved semantic/acceptance choice.
G3 must check region/page boundaries before/on/after, partial tails, the next decode crossing a region boundary, differing source/target page IDs, page reuse/reset, target capacity pressure, concurrency, and cancellation/abort conservation.
Verify raw KV and side-summary content, matched tokens, D's first scheduled tokens, tail recomputation, and release order. Exit success, byte counts, or a permissive guard alone cannot prove state integrity.
The candidate generic connector guard returns false while a model-specific CUDA-buffer connector path is described. Qualification must trace the image's actual path, including side storage; neither statement alone proves PD support.

Actual decode graph replay is mandatory. Preserve supported native prefill captured pieces and eager breaks; whole-prefill FULL capture is not assumed. Supported native dispatch is distinct from silently falling back to an incorrect simulator family or disabling sparse/graph requirements.
Record resolved policy, replay/capture/fallback counts, dispatch, padded shape, and host planning. Charge planning once per actual step/metadata group when reused; layer-specific staging, summary, and kernels remain layer costs.
Use existing queues, held-output semantics, launch/drain, and kernel-gap accounting. Long-prefill/short-arrival, idle-DP-lane, and first-D-tail experiments determine whether a new scheduling-visible boundary is necessary. Never copy native threads/IPC/MRv2 or create per-kernel/page events.
Padding may consume compute and scratch but must not advance live request tokens, regions, or KV frontiers. DP dummy forwards retain required synchronization even with zero useful payload.

## 6. Native reference and qualification

**D22 attention timing contract.** Candidate and independent reference measurements use disjoint cohorts of primed outer-only, normalized clean-replay windows through the existing timing path, with matching replay counts and minimum-window rules. Candidate mean versus reference-window median must remain within5%, and candidate CV must remain<=5%. Boundary-instrumented single replays are exported as diagnostics only. No fixed overhead subtraction, quality-dependent retry or historical FAIL reclassification is permitted. Scoped-operator and linear protocols are unchanged. Fresh admitted measurements must still match each D21 hardware/scale/runtime contract.

S01/G0 must capture immutable target-platform image digest, image ID/RepoDigests, architecture/entrypoint, actual package versions, relevant source/library hashes, and any available provenance. Missing build SHA stays unknown; candidate code must not be relabeled as image-build code.
Retrieve exact revision-pinned metadata through a file allowlist. No checkpoint weights or whole-repository snapshot may be downloaded. Additional small sidecar metadata is admitted only if the native text loader actually requires it and its content/purpose is recorded.
All new native runs use `--load-format dummy`, complete text structure, native quantization/exclusions, and unchanged router. Record initialization seed, parameter shapes, auxiliary buffers/scales, finite checks, and backend choice.
Equal seeds do not establish P/D parameter equivalence across layouts. Assert output equality only with matching parameter/buffer identity; known-state PD diagnostics are separate from performance evidence. Fix output lengths through supported native sampling options and verify completions without changing sampling or fabricating tokens.

| Milestone | Future evidence required | Claim limit |
| --- | --- | --- |
| S01.G0 | Image/metadata/source identity and exact metadata-only acquisition | Locatable reference; no execution claim |
| S03.G1 | Real rank-local kernel/state diagnostics on legal small shapes, typically 1–2 GPUs | Operator/state checks, not full-model E2E |
| S03.G2 | Capacity-qualified complete single role; prefill, subsequent chunk, decode, graph; all-rank census | Full-model path and resolved inventory; candidate 16/32-H800 groups require capacity evidence |
| S03.G3 | Complete native P/D, CUDA-buffer NIXL path, summary/tail content and lifecycle checks | PD functional qualification, not latency accuracy |
| S03.G4 | Per-track32/64 online PD with graph, complete rank/role activity and at least three clean repeats;64 formal work follows the matching32 release gate | Eligible reference for that hardware/scale, not Frontier accuracy |

**G2 simulation scope (D1).** Downstream simulator work needs only consumed native facts: (1) legal full-role startup, all ranks loaded and measured KV capacity; (2) complete locked text backbone/expert set, exclusions and unchanged router reconciled with §3; (3) initial/later-chunk prefill, decode and actual decode graph replay; (4) fixed-length completion; (5) finite hidden outputs on all ranks; (6) native failures recorded as BLOCKED_ON_RUNTIME with a reproducer. Finite resident/cache content belongs to on-demand census. A validator failure after native success does not reopen a criterion closed by other evidence. Complete precision/scale, physical/fusion, host planning and occupancy census runs only to localize an observed E2E error; its current owner is28, not retired24. G2 simulation scope is not track-specific G3/G4 or predictive acceptance.

G2 capacity includes actual mixed-precision parameters/replicas, initialization peaks, graph, workspace, raw cache, summaries, and scratch. An 8-H200 recipe proves neither 8-H800 fit nor 64-GPU PD support. Hopper backend selection must be observed; B300 MegaMoE assumptions do not transfer.
Native failure yields `BLOCKED_ON_RUNTIME` with a minimal reproducer; missing semantic approval yields `BLOCKED_ON_DECISION`. Unaffected evidence work may continue. Do not patch native functionality, disable required paths, or encode a native defect as simulator behavior to force qualification.
`D-S5-REF` is a conditional user choice only if a runtime failure requires a different image, upstream functional repair, or connector. It is not a default approval or unconditional dependency. A reference change requires a new identity and affected requalification.

## 7. Profiling, communication, and predictive independence

Observer/source preparation may proceed from accepted03. Native capability qualification remains independent of Step simulator implementation. New full-path compute, routing and TP/EP measurement work follows `15.controlled_structural_path` and its respective G2/observer prerequisites; PD measurements additionally require G3. G1 local measurements in05 remain diagnostic until admitted and should be extended only to supply a concrete missing full-path scope. Ticket10 consumes the05 coverage disposition, not complete 05 acceptance, and must independently close every required measurement gap.
Use separate clean, light-observer, detailed trace, and Frontier outputs. Validate observer correlation/perturbation before using its records for formal cost replay. Earlier records stay diagnostic and unverified.
Observer code must not add hot-path synchronization or change dispatch, routing, token work, or graph breaks. Diagnostic synchronization stays outside clean timing. Use request/step/rank/group/scope IDs and causal relationships; do not subtract unrelated cross-node clock readings or inherit nth-forward pairing blindly.

Profile boundaries come from the locked inputs: region `r-1,r,r+1,2r±1`, page `B-1,B,B+1`, window `W-1,W,W+1`, selector `K*r-1,K*r,K*r+1`, plus measured long-context coverage.
Include whole-prefill shapes as the canonical case dispatches them (chunked prefill is disabled — decisions D9–D14), long-prefix short-prompt cases, short-selector bypass, actual staging/paged/ragged paths, graph-bucket edges, valid/padded rows, physical table extent, closed/open regions, reachable precision classes, and legal KV/provider replication. Chunk-boundary and subsequent-chunk coverage is required only if a later approved case enables chunked prefill.
Keep duration, kernel count, device gaps, and host preparation separate. Independent sampling and held-out shapes remain separate. Shared physical scopes that already include communication/activation/gating cannot be charged by another owner.
Every profile must identify image/runtime, model/metadata, hardware, precision/exclusions, role, TP/DP/EP/PP, phase, graph policy, shape/layout, backend, and physical-scope/fusion coverage as applicable. Declare family-specific required features through existing contracts; reject mismatches and coverage gaps.
Publish through the existing producer/trainer/cache identity boundary. Preserve legacy table meanings and cold/warm behavior; check GLM → Step → GLM and sequences including DS against isolated loads.

For current calibration D29 requires balanced token-to-expert routing in both
Frontier and native execution, with effective assignment, DP/EP, padding and
integer-remainder semantics verified. Preserve historical routing identities;
random-uniform sampling and expert-placement balancing alone do not establish
the required policy. Ticket11.routing_policy owns the prerequisite check and
proposal. A needed native implementation/reference change requires its explicit
decision and a fresh generation. Independently measure remaining occupancy,
padding and selector locality across relevant seeds/content/layers/phases;
held-out routing must never become predictive input.
Measure TP collectives, EP dispatch/combine/reductions, and NIXL with bytes, participants, topology, precision, useful/padded tokens, backend, and concurrency. Separate EP-only, NIXL-only, and shared-network cases. Do not replace measurements with peak bandwidth or a fixed 64-GPU latency.
Reuse C04/T06 workload and communication interfaces. New locality or contention
state stays in D-ROUTING/D-PD. D29's controlled balanced regime does not establish
production-routing representativeness. Align workflow first, GPU operators and
op gaps next, and CPU overhead/gap last; defer large CPU-gap analysis, sampling
and suites until then. Minimal simulator Python execution enablement is separate.
Do not inherit old `p2p_nccl` extract/slowdown, forced-uniform routing, `pd_s33` endpoint subtraction, DP×PP mode selection, or same-host pairing without reference-specific evidence and approval. Preserve those legacy references' behavior within their original scope.

Fixture replay, trace-conditioned cost replay, controlled workflow replay, and predictive online E2E remain separate evidence tiers. Final prediction receives only preregistered workload/configuration and independently measured predictors/statistics.
No future batch composition, exact top-k, expert histogram, transfer duration, completion time, or other data from the compared native run may drive its prediction. Trace-conditioned replay is diagnostic and cannot close the predictive acceptance gate.

## 8. Campaign and metric acceptance

**Current campaign: §0, D21.** Four prioritized hardware/scale tracks share the frozen128-request6144/1024 QPS1 workload and existing runtime/backend/metric rules. Tickets09/23/28/30 own track milestones; tickets24–27 and the W1–W6 × three-load18-cell campaign stay retired. The workload table and18-cell arithmetic below are historical, not additional obligations.

The32-GPU roles use P16/D16, each TP8/DP2/EP16/PP1;64 uses P32/D32, each TP8/DP4/EP32/PP1 (Frontier MoE TP1/EP16 or EP32). EP is not an extra GPU multiplier. Capacity, placement and execution require track-specific evidence; TP16/DP1/EP-off capacity03 qualifies neither topology.
The serving-context candidate 131072 is not a resolved default. Freeze it consistently for native and Frontier after inventory; retain the original model position configuration.

| Historical candidate workload (retired) | Prompt / output tokens | Purpose |
| --- | --- | --- |
| W1 | 512 / 128 | Short selector, SWA, host/graph cost |
| W2 | 2048 / 256 | Region accumulation |
| W3 | 8192 / 256 | Active sparse selection beyond `K*r` |
| W4 | 32768 / 128 | Long chunked prefill and PD payload |
| W5 | 4096 / 4096 | Long decode crossing state boundaries |
| W6 | Fixed mixed trace: 1K–64K prompt, 32–2048 output | Online placement, queues, and varying shapes |

The retired source proposal ran six workloads × approximately 25%/60%/85% of independently piloted sustainable capacity: 18 primary cells, at least three clean native repeats per cell. None of it is frozen QPS or qualified execution, and none of it is required under the canonical case.

**The canonical case replaces the matrix.** Its formal request set is the fixed-length selection of §0 (D17): 128 requests of 6144/1024 tokens, one every second. Exact input hashes, request IDs, arrival schedule, warmup and measurement windows, repeat aggregation, completion and stop rules, resolved serving context, graph and quantization settings are frozen in ticket 23 before formal comparison, not chosen afterwards. Overload is recorded separately and incomplete requests are never hidden.

**Repetitions in one job (decision D6, 2026-10-20).** Clean repetitions may share one job and one engine startup. Each repetition replays the same frozen workload with its own warmup and measurement windows, and the run records the window boundaries. Both roles must be idle between repetitions (no running or waiting requests, KV cache usage back to 0). Startup-dependent variation (graph capture, kernel selection, placement) is then sampled once per job, and the qualification record states this limit.
Targeted diagnostics inside the four tracks remain allowed. Additional workload families, QPS/topology sweeps and independent E2E holdout matrices remain out of scope. D21 keeps the frozen workload; it grants no new load adjustment.
Extra topology is not implied: P16/D48 is not automatically EP48 for 352 experts; heterogeneous TP and extra D Replicas require separate scope, placement, and connector qualification.

### Frozen threshold decisions; remaining measurement rules

**The final contract for each D21 track is TTFT, TPOT, THPT and E2E, each independently `abs(frontier - vllm) / abs(vllm) <= 0.10` in each clean repetition.** The30% threshold in §0 releases only corresponding64 formal work. The supporting internal/p95 tiers remain diagnostics, not additional final gates.

| Metric or contract | Required verdict |
| --- | --- |
| Canonical case TTFT, TPOT, THPT, E2E | Each independently <=10% relative error; all four must pass, and no metric may be dropped or re-aggregated to reach the threshold |
| Applicable internal-stage metrics (diagnostic) | Each independently <=10% when reported; applicability and exact intervals frozen before formal comparison |
| p95 client TTFT and p95 ITL (diagnostic) | Each <=15% when reported, conditional on sample sufficiency and reference-repeat stability rules frozen before formal comparison |
| Structural correctness | Request/token/layer conservation, required state content, valid execution order, no loss/double charging or premature release |
| Scope/communication holdouts | Preregister independent shape/phase/communication budgets; source suggestion of median <=10% is not an additional approved universal threshold |

Client TTFT is client submission to first client-visible token on one client clock. Preserve old `ttft_ms` as internal prefill-completion latency and old `token_throughput_tps` as prompt-plus-output throughput; add explicit client/output-only/ITL mappings through the same gate, the output-only one as a diagnostic.
For the canonical case, **THPT is total token throughput — prompt plus output tokens per second over the frozen measurement window — that is `token_throughput_tps`** (decision D15). Output-only tokens per second and requests per second are non-gating diagnostics. TPOT is the mean post-first-token output interval. `SPOT` in the earlier material is read as TPOT; no separate fifth metric exists. Reuse `tests/comparison/e2e_gate` for normalization and admission instead of introducing a second gate.
Define P-prefill completion, KV-ready, D-first-step, request completion, and inter-token intervals separately. Client/proxy/CPU boundary terms require evidence and the relevant shared decision; an internal-only result cannot pass client TTFT.

**First token and recomputed tail (decision D5, 2026-10-20).** Ticket 08 run R1 established that native P computes the prompt minus a tail (4080 of 4105 tokens), D reads those 4080 tokens over NIXL, computes the 25-token tail locally and produces the first output token. Frontier ends `ttft` at P prefill completion and lets D compute one token. The error contribution is assessed before any change, as two separate terms: T1, the first-token endpoint (KV read plus the first D step, which Frontier already simulates but does not map to TTFT), and T2, the moved tail work. A term is small when, in every assessed cell, it changes the Frontier mean client TTFT and mean request E2E by at most 1 percentage point of the native value (one tenth of the 10% budget) and changes no per-cell verdict; this threshold is the parent agent's working rule and the user may change it. A small term is recorded with its measured size and not fixed for now. A large term is aligned within the existing owners: T1 through the existing request timestamps and the ticket 14 metric mapping, T2 through the existing PD transfer and D first-step path. The method and data sources are in [requirements.md](requirements.md), decision D5.
For prediction `p` and actual `a`, absolute error is `abs(p-a)`; relative error is `abs(p-a)/abs(a)` when `a != 0`. When actual is zero, relative error is **undefined**. Never invent an epsilon; preregister an applicable absolute budget if needed before formal comparison.
Insufficient samples or missing requests/ranks/pairs yield `INSUFFICIENT_EVIDENCE`; unstable repeats yield `REFERENCE_UNSTABLE`. Neither is PASS. Averages across cells, models, or hardware cannot replace any per-cell verdict.
Freeze p95 eligibility rather than selecting it after seeing errors; an ineligible p95 result stays explicitly unverified. Unknown parameters remain unknown until the freeze record supplies evidence.

## 9. Cross-task implementation and decision gates

The current request narrows the historical S5-Q6 barrier: ticket15 starts with model/operator representation from03, then a complete controlled structural request path, and finally native reconciliation after06. All existing acceptance criteria remain required; static config, fixtures and controlled durations do not prove native or predictive support. The early scope includes model normalization, per-layer attention/FFN bindings, operator and profiling registration, precision/shape metadata and existing memory/execution interfaces. It does not authorize a generic runtime redesign.

**Mainline decisions D2/D3, amended by D20/D21.** Tickets07/08/10/11/12/17/20 and15.native_reconciliation consume06.simulation_scope, not the entire on-demand census. Structural17/20 reused C16/C17 before the GLM milestone. D20 additionally releases measured-compute/bundle, PD and client-metric CPU implementation now; D21 serves P0 first. Trained/predictive acceptance still needs the external GLM milestone and real native/profile inputs. Neither a second path nor another decision framework is introduced.

Integration tickets16–22 retain external `09.full_frontier_path[GLM]` for trained/predictive **acceptance**, not D20-authorized CPU implementation. D20 explicitly permits measured compute/bundle, PD and client-metric code and CPU regression now; D21 prioritizes P0. In the shared plan the GLM gate is `09.bundle[GLM]` plus `09.structural_frontier_path[GLM]`, including one complete trained-predictor `python -m frontier.main` GLM PD run on topology A. It does not require shared10's formal64-H800 accuracy matrix, but includes native G0-D via workflow/placement. No gate closure is inferred from D20/D21 or an import/fixture.

The common final branch remains `feat/glm53-dsv4-vllm028`. Shared C01–C06 responsibilities and exact inherited ticket paths are in [plan.md](plan.md). The shared execution lead currently owns C02/C04 model/config/binding/family integration. Step contributors supply separate contracts, leaf changes and evidence; no two agents edit the same common abstraction. DeepSeek static representation can proceed independently, while inherited GLM-before-DeepSeek trained integration and representative-verdict sequencing remain intact.

| Existing decision | Evidence and minimum Step disposition |
| --- | --- |
| D-PD / shared12 | Raw/summary/open-state disposition, component capacity, reservation, READ/visibility and source release; contention is added only if relevant to the measured path |
| D-ASYNC / shared13 | Actual planning reuse, held outputs, launch/drain, chunk/graph dispatch and client-visible boundaries |
| D-ROUTING / shared14 | Independent native occupancy and selector locality; additional predictive state only if existing features are insufficient |
| D-DP-PD / shared15 | Actual P/D entry, endpoint/DP4 load reports and placement; the existing co-location-only guard cannot simply be removed |

These records retain one shared decision and implementation per responsibility. Evidence collection, operator registration, small config/model extensions and instrumentation need no renewed approval in the execution phase. If existing semantics suffice, record the evidence and proceed. If a required capability cannot be expressed incrementally, identify the exact code limitation, propose the smallest core refactor, compare alternatives and regression costs, and obtain explicit approval before that refactor. Materially unresolved simulation semantics or acceptance criteria also require a decision. A C/D label or source difference alone is not a blocker or permission for architectural work.

No such core refactor is approved by this review. [review.md](review.md) records candidate limitations and alternatives without choosing a new allocator, runtime DAG, transfer manager or parallel registry.

## 10. Verification, completion, and operating limits

Reuse existing unit/integration checks, full `frontier.main`, typed CSV-to-trained-predictor admission, the native drivers, and the single E2E gate. Code-changing steps require baseline-versus-candidate fidelity comparisons across at least 50 concrete dense/MoE, offline/online, length/count/QPS, feature, and model scenarios.
Required negative cases include wrong text binding/head/layer/precision, fractional local KV heads, region-as-token errors, raw-KV division, missing summaries, duplicate fused cost, padded-token credit, premature readiness/free, wrong identity/backend, missing coverage, and cross-model cache contamination.
Separate pure moves from semantic changes; preserve unchanged numeric results and key intermediates. Apply the 2,000-line soft module limit, cleanup-first analysis, and recorded split rationale to touched oversized modules. Do not make a repository-wide refactor a prerequisite.
Before running baseline-versus-candidate performance comparisons, freeze the simulator wall-time and memory regression budgets. Record measured changes; every over-budget regression requires an explanation and a recorded disposition before the affected step is accepted. Budget values are not invented during this planning phase.
Dummy timing smoke is structural only. Native G0–G4, trained-predictor simulations, per-cell predictive gates, independent holdouts, and fresh H200 measurement supply their distinct evidence. No runtime tests are required for this document-writing phase.

**GPU resources (D21).** Follow §0: P0 H80032 highest, P1 H80064 non-preemptible `codesign`; equal-priority P2 H20032 non-preemptible and H20064 wholly preemptible on `step_main`. Across all personal nonterminal reservations in every session, H200 non-preemptible total<=32; every additional job is preemptible. Check fresh complete reservations before submission. Lower-priority H800 work must not starve ready P0 work. Earlier64/70/80/130 allowances do not authorize current jobs.

**Mainline execution rule (2026-10-20).** Advance only model support, the PD/DP/EP structural path, measured profiles and predictors, native PD and E2E ground truth, comparison and calibration. `tests/integration/step5_environment/` is frozen except for fixes that a mainline run needs. A sub-goal that goes through two rounds of "tool failure, tool fix, rerun" without a new acceptance criterion is stopped and reported.

Ticket23 freezes the common workload and each track's hardware, topology, runtime and measurement contract before formal collection. Tickets09/28 require matching native/profile evidence and the64 release conditions. A32 track is not a reduced substitute for64; it is an independently approved formal target.
Overall completion requires mandatory S00–S10 outcomes, **all four tracks** independently passing all four metrics<=10%, and same-head legacy/GLM/DS/Step contracts, profile isolation and applicable qualified predictive regressions. The retired workload matrix/holdouts do not return. Missing, unstable or unfinished track evidence cannot borrow another PASS;30% release is not final acceptance.
Final evidence must trace source SHA and dirty state, inherited shared-change identities, Step image/metadata locks, GLM/DS reference locks, instrumentation, workload/campaign, inventories, physical-scope coverage, immutable bundles, native/Frontier manifests, per-cell actual/predicted/error values, holdouts, H200 results, regression records, and measured GPU-hours.
Keep previous raw evidence/readings immutable; corrections create new generations and rerun affected checks. Use the existing calibration workflow: semantic alignment, workflow/operator/communication/KV diagnosis, then approved host corrections. No arbitrary factor or tolerance relaxation may absorb missing work.

The documentation review is complete and implementation resumed after confirmed，continue. Ordinary Step implementation, instrumentation, test/resource use, profiling and E2E collection are authorized by the current request. Commit completed scoped code steps after required verification; publishing, destructive integration/history changes, native functional substitution and reference replacement retain their applicable authorization requirements. The historical review-only boundary is superseded by the recorded execution confirmation; no whole-ticket acceptance follows from it.
For authorized operations, use the original personal local StepMind Python RJobBackend and current host runbook. The old Pi exclusive-helper and filtered-receipt restrictions are superseded. Read original API outputs and retained evidence directly; preserve existing job identities and never duplicate a submission because status is unknown.
Future approved jobs follow platform/storage rules, using durable cloud artifacts only under `/mnt/codesign-exp/ycfeng`; local records retain exact permitted paths and identities. Avoid measurement interference between concurrent GPU/NIC allocations.
For runs using >=32 GPUs, auxiliary agent validators are non-blocking soft gates: log failures, notify the user, and mark affected evidence unverified; never stop healthy workloads solely for a harness error. Native failures retain normal handling, and final evidence verdicts still require valid records.

## 11. Explicit deltas and factual anchor corrections

- S5-Q1 fixes the stacked base; the source's reviewed main is not the worktree HEAD. No GLM ownership transfer or completed integration is asserted.
- S5-Q2 closes the missing-companion-document issue. Source hashes/byte checks do not close the absent raw-metadata/image/native-evidence issues; historical report notes about stale requirements are superseded by current requirements.
- S5-Q3 removes source X01 from current and future scope. Remaining text extensions are unapproved scope choices, not PRIMARY work or newly generated tickets.
- S5-Q4 makes actual H200 retesting explicit in final acceptance, correcting the campaign's incomplete final checklist without inventing H200 counts/topology.
- S5-Q5 confirms the 10%/15% thresholds and restores explicit p95 client TTFT/ITL eligibility missing from YAML; sample/stability/window rules still await the pilot freeze.
- S5-Q6's historical blanket barrier is narrowed by the current user request: ticket15 representation/structural work starts before GLM trained closure;16–22 retain their verified GLM and native prerequisites. Early preparation never closes a whole ticket.
- Source 02's blanket “all NOT_STARTED” conflicts with its explicit optional `NEEDS_DISCUSSION` statuses. G0/G1 and partial05/06 work now exist, while Step representation is accepted for contracts and predictive support remains unimplemented; optional text scope is not promoted by that blanket sentence.
- Correct module anchors are `frontier/attention/ops.py` and `frontier/operators/{registry,spec}.py`; FFN layer contracts do not own attention identity, and `LayerAttentionSpec` is currently in `attention/gdn/config.py`.
- “No fallback” forbids silent semantic substitution, not qualified native prefill breaks. Candidate code is navigation evidence, not image-build identity; recipes and fixture arithmetic are not measured qualification.

Remaining unknowns include full image-build provenance (digest/package/HF metadata are locked), per-track role capacity/runtime/graph evidence, PD content/readiness/lifetimes, admitted profile coverage, required shared dispositions, and frozen measurement/stability parameters. The H200 minimum-matrix study is retired, not an unknown to investigate. Historical public-documentation limits remain in `reference_workflow_review_2026-10-18.md`; no unknown is filled with a default or used to stop unrelated authorized CPU implementation.

Resident-parameter source commit: `438424c36ed7d0d933b5e90dd64e2a82a86de303`. `native_parameters_20261019_01/post_commit_01.json` verifies all six Git/current/frozen files, exactly six changed paths and parent8bf0315. Tracked worktree clean; unrelated .pi/ untouched, no push or ignored-record addition.66 documentation/source checks preserve acceptance text and1/10,0/8counts. Parent continues remaining transient-scale/write/occupancy and complete readiness; CPU/API source acceptance does not qualify native execution.


## FIFO source/CPU completion and terminal disposition — 2026-10-19

The existing one-reservation and native-error/deadline contracts are preserved by the accepted source/CPU FIFO correction; no simulation semantics, reference runtime, metric or acceptance threshold changes. See [the report](test_report_2026-10-19_native_fifo_export.md) and `native_physical_20261019_01/fifo_export_01/acceptance03.json`. The original H800 generation remains failed, and source/CPU checks grant no new G2 credit.06 remains PARTIAL1/10, prior criterion7 only; whole15 remains PARTIAL0/8. All hardware, shared-branch and downstream gates remain.
