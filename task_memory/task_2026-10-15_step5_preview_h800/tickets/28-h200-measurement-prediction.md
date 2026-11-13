# 28: Four-track native comparison, release30 and final10 acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-21 | D21: four existing-ticket track milestones;32 clean comparisons release matching64 formal calibration at30%, with final10% unchanged. |
| 2026-10-21 | D20 authorizes executable Step client-normalizer/gate integration alongside measured compute/PD code; real comparison remains evidence-blocked. |
| 2026-10-21 | D19 assigns CPU Frontier/gate command and native-input mapping preparation; actual comparison retains22/23/native prerequisites. |
| 2026-10-21 | D18: pin formal comparison inputs to shared 68f891b package, adopted as 958d81a; keep runtime, profiles and four-metric acceptance separate. |
| 2026-10-21 | D17: replace SharedGPT with 128 fixed 6144/1024-token requests at QPS 1.0 and scope H200 profiles to the reached prefill/context/batch shapes. |
| 2026-10-21 | Decision D15: native collection uses 256 requests retimed to QPS 4.0; THPT in the comparison is prompt-plus-output tokens per second, with output-only and request throughput reported as diagnostics; cross-task comparisons with the GLM/V4 session carry the communication-backend caveat. |
| 2026-10-21 | Rewritten by decision D12 into the single canonical-case deliverable: H200 profiling for the measured backend, native clean collection, the Frontier-versus-native four-metric comparison, targeted diagnosis and the fit freeze (absorbing retired 24/25/27). |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved fresh H200 profiling and prediction ticket. |

**What to build:** Four separately qualified D21 results using one frozen workload: matching measured profiles, clean native ground truth, equivalent Frontier simulation, four independent metric verdicts and targeted attribution. This existing ticket owns32 release30 and every track's final10, plus diagnosis/remeasurement/fit-freeze inherited from retired24/25/27. It does not revive those tickets or W1–W6.

**Blocked by:** Each track's own ready(t) in `../plan.md`:22 bundle,23 freeze,14 native mapping, admitted profiles and case-aligned08/09 native readiness plus existing acceptance prerequisites. H80064 formal work/clean collection additionally requires `28.release30(H80032)`; H20064 requires `28.release30(H20032)`. Basic64 validation is not blocked by30%.

**External prerequisites:** Actual hardware/scale-specific native/profile identities, resolved official image and qualified inputs; no hardware timing scaling or cross-track substitution. Use personal local StepMind Python RJobBackend/current host rules and the existing producer/publisher/`tests/comparison/e2e_gate`. No second comparator, release framework or tolerance change is authorized.

**Status:** in-progress (D20 comparison-code implementation); formal comparison blocked

**D31 status (2026-10-25):** Active as plan steps E and G: the H200 32-GPU and 64-GPU comparisons.

## Current D21 track milestones

| Priority / track | Resource/topology | Formal start | Current evidence / final status |
| --- | --- | --- | --- |
| P0 H80032 | `codesign`, non-preemptible; P16/D16 TP8/DP2/EP16/PP1 | Own readiness | Clean four-metric result pending; highest priority; final four<=10% |
| P1 H80064 | `codesign`, non-preemptible; P32/D32 TP8/DP4/EP32/PP1 | Own readiness AND H80032 release30 | Basic may proceed now; formal blocked; final four<=10% |
| P2 H20032 | `step_main`, non-preemptible within cap; P16/D16 TP8/DP2/EP16/PP1 | Own readiness, independent of H800 | Clean result pending; final four<=10% |
| P2 H20064 | `step_main`, wholly preemptible; P32/D32 TP8/DP4/EP32/PP1 | Own readiness AND H20032 release30 | Assess prior64 basic evidence; formal blocked; final four<=10% |

The two P2 labels are equal priority. `28.release30` means **all four errors<=30% in each of three clean complete-population repetitions**. Equality passes; no averaging across metrics/hardware/repeats hides a failure. Release unlocks only the corresponding64 formal calibration, not basic runtime, not final support and not H800-to-H200 work. All final gates stay<=10%. Distinct manifests, profiles, bundles, native/Frontier identities and outputs bind each track. Existing64-H200 platform success is not a clean calibration result or32 release evidence.

Resource policy counts all personal nonterminal reservations including queued jobs/other sessions: H200 non-preemptible total<=32; every additional job is preemptible even below obsolete80/130. H800 is non-preemptible `codesign`; lower-priority work cannot starve P0. D20 CPU compute/bundle/PD/client integration continues P0-first.

**Current D20 override:** `step5-client-metrics` owns shared client-metric reuse and truthful Step schema/runtime adaptation, including actual normalize/gate CLI execution and complete population/provenance rejection tests. The user explicitly approved this Q10 scope extension. D19 command-only preparation below is completed; it is not the current deliverable. The existing four10% metrics and native/profile requirements remain unchanged.

**D19 preparation:** `../d19_pipeline_preparation/` prepares the current Frontier runner and existinge2e_gate command/input mapping alongside22, including all128 formal requests and frozen windows. This work provides no accuracy evidence and does not bypass native/R2 or profile admission.

## Work

1. **Per-track measured profiles.** Admit actual hardware/scale-compatible compute and communication inputs:6144 unchunked prefill, context through7168, reached decode batches and native D tail. Assess existing data before repeating collection, but do not scale timing, waive quality or relabel raw diagnostics. SharedGPT-only shapes stay deferred. P0 inputs lead the work.
2. **Native clean collection.** Run `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/` from `68f891b` on the native reference: 128 requests of 6144/1024 tokens, open-loop QPS 1.0 over 127 seconds, online P/D, MTP off, chunked prefill and prefix caching off, graph and quantization on. Use three clean repetitions with 16 separate same-shape warmups each and both-role idle checks, per D6/D17/D18. Stable template IDs preserve prompt content; transport IDs are repeat-qualified. The independent earlier Step package is historical.
3. **Frontier run.** Run the same frozen workload and equivalent configuration with the measured backend and `collective_sim`.
4. **Four-metric comparison.** Report client TTFT, TPOT, THPT and per-request E2E, each independently `abs(frontier - vllm) / abs(vllm) <= 0.10`, with actual and predicted values, absolute and relative errors. THPT is prompt-plus-output tokens per second (`token_throughput_tps`) over the frozen window (decision D15); output-only tokens per second and requests per second are reported beside it as non-gating diagnostics. Where actual is zero, relative error is undefined; preregister an absolute budget instead of inventing an epsilon. Any statement that compares this result with the GLM-5.3 / DeepSeek-V4-Flash result says that the two tasks use different communication backends (`collective_sim` here, the measured CC backend there), so their communication error is not directly comparable.
5. **Targeted diagnosis.** Only where an error affects a formal metric or indicates a correctness problem, localize it across scheduler/request lifecycle, sparse attention and region selection, summary updates and KV behavior, MoE operators and communication, CUDA graph and host/device overlap, and P/D transfer synchronization. Report the D5 T1/T2 terms separately; a term above 1 percentage point of the native mean becomes a correction under ticket `17` or `14`.
6. **Fit freeze.** After any correction is applied and re-measured, freeze the predictor and statistical fits and record the freeze before stating the final verdict.

## Acceptance

- [ ] Every track has qualified compatible profiles for all measured costs, with hardware/scale/runtime/precision identity and coverage; no hardware scaling or cross-track accuracy substitution.
- [ ] Both32 release30 records contain all four errors<=30% in every one of three clean complete-population repetitions. Each64 formal start records own readiness and only its matching32 release; H20032 has no H800 accuracy dependency and basic64 validation needs no30% gate.
- [ ] The native collection uses the frozen manifest exactly — 128 fixed 6144/1024-token requests at QPS 1.0, the new package hashes, MTP off, chunked prefill and prefix caching off, graph and quantization on — with three clean repetitions, 16 separate warmups each, recorded window boundaries and both-role idle checks. Keep all formal requests; throughput divides 917,504 total tokens by the first-formal-submission to last-formal-completion window, not the 127-second injection span.
- [ ] The Frontier run uses the same workload and equivalent configuration with `collective_sim` and the measured backend, and completes every request.
- [ ] Each of the four tracks reports all four metrics with actual/predicted/absolute/relative errors and independently passes<=10% under the frozen three-repeat rules. THPT is prompt-plus-output; output-only/request throughput are diagnostics. No omitted metric, request, re-aggregation or pooled repetition hides a failure;30% release alone never closes this ticket.
- [ ] Aggregation method, request set, clock normalization and admission rules are exactly those frozen in `23`; any deviation invalidates the verdict rather than being absorbed by tolerance.
- [ ] Any targeted diagnosis names the responsible component and ticket, and any correction is applied, re-measured and re-verified rather than papered over with a numeric factor.
- [ ] The predictor/statistical fit freeze is recorded with its evidence before the final verdict, and the verdict states plainly PASS or FAIL per metric.
- [ ] Evidence identifies the image digest, metadata revision, workload hash, topology, hardware, precision and graph policy, and remains immutable; corrections create new generations.
- [ ] At ≥32 GPUs auxiliary validator errors are logged, marked unverified and reported without stopping healthy work; native failures retain normal handling and invalid evidence never becomes PASS.

Specification references: §§0, 4, 8, 10.
