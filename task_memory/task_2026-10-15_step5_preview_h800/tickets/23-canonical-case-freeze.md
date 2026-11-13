# 23: Canonical case manifest and freeze

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | D31 status note added. |
| 2026-10-21 | D21: preserve common workload bytes and freeze four distinct hardware/scale manifests, basic/formal eligibility and paired32 release conditions. |
| 2026-10-21 | Native consumer/stage02 integrated and verified585 PASS; configured runtime/metric windows recorded, effective native qualification remains open. |
| 2026-10-21 | D18: lock GLM/V4 package 68f891b via Step 958d81a, close common-artifact adoption, and start Step consumer/runtime preparation. |
| 2026-10-21 | D17: freeze the new fixed-length workload (128 requests, 6144/1024 tokens, QPS 1.0); defer SharedGPT; retain its historical hashes, with new runtime qualification and GLM/V4 adoption pending. |
| 2026-10-21 | Decision D16: adopted the shared frozen workload `data/processed_traces/canonical_sharegpt_req256_qps4/` (commit `9ed979e`, cherry-picked as `cb8090b`) with its checksums; retiming is the historical linear rescale (scale 2.590246128, no seed), independently reproduced with 0 mismatches; removed the seed/distribution option; corrected the 128-prefix span to 31.0 s; resolved the priority column; workload acceptance items checked. |
| 2026-10-21 | Decision D15 (shared with the GLM-5.3/DeepSeek-V4-Flash session): 256 requests (first 256 in trace order) instead of 128; THPT is prompt-plus-output tokens per second with output-only and request throughput as diagnostics; manifest must carry the trace identity and SHA-256, an explicit retiming rule and the selected-set distribution report; one shared manifest for both sessions. |
| 2026-10-21 | Rewritten from the retired H800 pilot/freeze (decision D12) into the single canonical-case manifest and freeze: SharedGPT 128 requests at QPS 4.0 on 64 H200, trace acquisition and hashing, resolved runtime settings, four-metric contract. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved independent H800 pilot and freeze ticket. |

**What to build:** One unchanged common workload plus distinct hardware/scale runtime and evidence manifests for the four D21 tracks in `../spec.md` §0, frozen before formal collection/comparison. This ticket freezes inputs; it does not measure accuracy or run the comparison.

**Blocked by:** `14` for the metric mapping. The manifest may be frozen before `09` completes; the formal comparison in `28` still requires `09`.

**External prerequisites:** Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed by freezing the manifest. Use existing shared contracts and record their sufficiency.

**Status:** in progress — common68f891b artifacts adopted as958d81a; native consumer integrated as db5c99f with585 parent passes and both stage02 packages independently verified. Manifest SHA-256 `5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3`; SHA256SUMS-file `594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9`. Configured7168/8192/128, graph ceiling256, Step utilization0.92 and actual MRv2 source bindings are frozen in `../d18_native_20261021/stage_handoff.md`; effective native runtime/occupancy/replay remain unqualified. The real HTTP tests cover all432 requests, three repeats/idle boundaries, complete-population metrics and both nongate diagnostics. Final metric admission/mapping still depends on14. D17's independent package remains historical.

**D31 status (2026-10-25):** The frozen fixed workload is used unchanged for the H200 runs.

## Current D21 track freeze

| Priority / track | Policy | Topology per role | Formal release recorded in manifest |
| --- | --- | --- | --- |
| P0 H80032 | `codesign`, non-preemptible | P16/D16, TP8/DP2/EP16/PP1 | Own readiness |
| P1 H80064 | `codesign`, non-preemptible | P32/D32, TP8/DP4/EP32/PP1 | Own readiness AND H80032 all four clean errors<=30% |
| P2 H20032 | `step_main`, non-preemptible within global cap | P16/D16, TP8/DP2/EP16/PP1 | Own readiness, no H800 accuracy dependency |
| P2 H20064 | `step_main`, whole job preemptible | P32/D32, TP8/DP4/EP32/PP1 | Own readiness AND H20032 all four clean errors<=30% |

Both P2 labels are equal priority. Basic runtime may proceed now for either64 track without the30% condition; its outputs are labelled basic, never formal calibration. `../plan.md` defines readiness. Release evidence uses each of three clean repetitions with all four errors<=30%, not an average; final10% remains unchanged. Freeze hardware/scale-specific profile/bundle/native identities and separate outputs; no timing scaling or cross-track PASS substitution. H200 non-preemptible reservations across all sessions including queued total<=32; every additional job is preemptible even below obsolete80/130. Lower-priority H800 work cannot starve P0.

Old H20064 non-preemptible `canonical02` launch intent is superseded; preserve immutable stages and assess ended qualification02 basic-runtime evidence. New generation/runtime implementation is owned separately, not delivered by this document. Retired24–27/W1–W6 do not return.

## Deferred SharedGPT evidence

**Historical workload (D13/D15/D16, deferred by D17).** Preserve `data/processed_traces/canonical_sharegpt_req256_qps4/`, frozen by the GLM/V4 session in commit `9ed979e` and cherry-picked as `cb8090b`, unchanged. These files are not the active formal workload. Historical SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `trace.csv` | `47d5fb39f2a753e9f96fee2c98f2c77037a703113abe127697ab23969e46eddc` |
| `request_ids.json` | `7fb03729ecb270abaa03a7c442838313a017a1b8e9b6f5f6acbdb40ba20e03fa` |
| `warmup_pool.csv` | `3a86896b31132f7ea1fdb52508cbe64b97b50fd39e3a6d5c590ff91ceb14df16` |
| `manifest.json` | `8df7bd894c08e03e9dcae69394767c6536c4f665d120d2e730a32cb4a2f6944f` |

Historical builder `tests/integration/canonical_case/build_sharegpt_manifest.py`; pin test `tests/unit/test_canonical_sharegpt_manifest.py` (2 passed on `feat/step5-preview`, frontier-py310). Acquisition, 0-mismatch equivalence with dev-vidur's historical linear rescale and the selected/full distribution report remain recorded in D16 and `scratch_evidence/canonical_workload_alignment_20261021/`. Short-prompt follow-up is deferred. No historical bytes are replaced by the new case.

## Work — active D17 numerical case with D18 shared artifacts

1. **Package and producer — done.** Adopt `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/` unchanged from `68f891b`: `trace.csv`, `request_ids.json`, `warmup_pool.csv`, `manifest.json`, `SHA256SUMS`. Reuse `tests/integration/canonical_case/build_fixed_manifest.py`; verify its source hash and byte-identical regeneration. Bind the existing Frontier/native consumers to these files. The independent D17 package is historical and is not regenerated or selected for new runs.
2. **Request set.** Exactly 128 formal requests, each 6144 input and 1024 output tokens. Totals are 786,432 input, 131,072 output and 917,504 tokens. Sixteen same-shape warmups have disjoint IDs and a separate timeline. The rationale is D17 in `requirements.md`; all formal requests stay in every metric population.
3. **Arrivals.** Open-loop deterministic spacing: `arrived_at_i = i / 1.0` seconds, `i=0..127`, span 127 seconds; no arrival RNG or rescaling of SharedGPT. Each warmup trace independently spans 0..15 seconds at QPS 1.0. Actual submission lateness is recorded, never hidden by converting to a closed-loop client.
4. **Exact native lengths.** Freeze the existing deterministic token-ID prompt producer and each model's vocabulary identity. Stable template IDs seed identical prompts across repetitions; transport IDs use `{request_id}-repeat{repeat:02d}`. POST `/v1/completions` with exactly 6144 IDs, no chat template or extra special tokens, streaming, `min_tokens=max_tokens=1024`, `ignore_eos=true`. Check engine-accepted prompt counts and final usage. A 7168-token context is required; no truncation or early EOS is eligible.
5. **Resolved runtime settings.** From the image and its canonical serving command, record the image digest, the active KV-transfer connector, resolved `--max-model-len`, graph policy, quantization behavior and the exclusion list. MTP: the source-level default is resolved off (spec §0: `vllm/engine/arg_utils.py:707,2048–2087,2217–2226` and `vllm/transformers_utils/config.py:756–791` in the locked image; no `speculators_config` in the locked `config.json`); the manifest records that the serving command carries none of `--speculative-config`, `--spec-method`, `--spec-model`, `--spec-tokens`. A tag alone is not evidence; if a required part cannot be executed, record the blocker and the smallest viable resolution.
6. **Topology (D21).** Freeze four track manifests per the table:32 total=P16/D16 TP8/DP2/EP16/PP1;64 total=P32/D32 TP8/DP4/EP32/PP1. Record hardware, node mapping, EP span and job policy separately.
7. **Metric contract.** TTFT, TPOT, THPT and E2E, each independently `abs(frontier - vllm) / abs(vllm) <= 0.10`; client-level TTFT, **THPT as prompt-plus-output tokens per second (`token_throughput_tps`) over the frozen measurement window** (decision D15), per-request E2E (not makespan), aggregation method, formal request set, clock normalization and evidence admission rules. Output-only tokens per second and requests per second are recorded and reported as non-gating diagnostics. Reuse `tests/comparison/e2e_gate`; do not add a second gate.
8. **Windows and repeats.** Three clean repetitions in one job, with 16 same-shape warmups per repetition, both roles idle after warmup and between repetitions (running/waiting zero, KV usage zero), then a fresh formal origin. Use the interval from first formal client submission to last formal completion for THPT, with all 917,504 tokens; compute per-request TPOT as `(E2E-client TTFT)/1023` and average over 128 requests. Record repeat variation without discarding a request or choosing a favorable repetition. Native timeout/completion rules and reference-stability disposition are frozen with the runtime before launch.
9. **One shared manifest (D18) — done.** Step adopted GLM/V4's existing `68f891b` package with matching hashes. No reciprocal adoption action remains. Consumer readiness and Step runtime evidence are still required before formal execution. The communication backend intentionally differs (`collective_sim` here, the measured CC backend there), so any cross-task comparison states that communication error is not directly comparable. Other-session files remain owned by that session.
10. **Runtime capacity and profiling reach.** Freeze and qualify native `max_num_batched_tokens >= max_model_len >= 7168` with chunking disabled, concurrent `max_num_seqs` and reached graph capture sizes **per track**. Frontier uses `trace_request_generator_config_max_tokens=7168` and all scales1.0; verify full requests. R1's512-token/one-sequence configuration is insufficient. Each hardware/scale's profiles cover6144 prefill, context through7168, reached batches and D recomputation. Reuse only identity/quality-compatible evidence, never scaled timing.

## Acceptance

- [x] The shared package matches the frozen source and passes its checksums and builder/pin tests; the prior independent Step and SharedGPT packages remain unchanged. Evidence: D18 adoption report.
- [x] Exact formal count, 6144/1024 lengths, QPS 1.0 spacing, 127-second span, IDs, disjoint 16-request warmups and token totals agree across shared files and manifest. Step native prompt production is verified separately through the consumer.
- [x] Both sessions reference the same package from `68f891b`; Step cherry-pick `958d81a` preserves its bytes and hashes.
- [ ] All resolved runtime settings are recorded with image/source provenance, including the MTP-off source citation and a serving command without speculative flags, the connector, `--max-model-len`, graph policy and quantization behavior.
- [ ] Four distinct track manifests record D21 priority, hardware/scale, topology, preemptibility, basic/formal status and their exact readiness/release dependencies; the two P2 tracks are equal priority, H20032 is independent of H800, and global personal H200 non-preemptible reservations<=32.
- [ ] The four-metric contract is written down with definitions — THPT as prompt-plus-output tokens per second, output-only and request throughput as diagnostics — aggregation, clock normalization and admission rules, each metric independently gated at ≤10%, and no metric may be dropped or re-aggregated to reach the threshold.
- [ ] D17's warmup/measurement windows and three repetitions are implemented with actual idle checks; runtime capacity, stability and completion/stop rules are frozen before launch.
- [ ] The freeze record contains no Frontier output and no comparison result; choosing or adjusting a setting after seeing errors is not permitted.

Specification references: §§0, 4, 5, 6, 8.
