## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Second review accepted a smaller implementation using existing event adapters and source-local completion helpers. |
| 2026-09-08 | Designed shared forward synchronization, source-local attention, and request-level completion for D017. |

# Shared forward synchronization design

## Status and scope

Second review ACCEPT; YC authorized sequential implementation and validation. The reviewed simplification is recorded in review_shared_forward_sync.md. This document owns the design; analysis/mixed_phase_forward_stall_rca.md owns the observed failure and reproductions.

Target: current MONOLITHIC MoE co-location path, including pure-prefill, pure-decode, and locally mixed source batches. Acceptance case remains Qwen3 TP4/DP2/PP1/EP8, H200 step_main, 4096/1024, eager, prefix OFF, chunked prefill OFF, uniform routing, collective_sim/htsim with nvlink_analytic. CPU overhead remains disabled. Preserve existing PDD/PD-AF paths and their release guards. Do not change DP selection, request admission policy, predictor feature schemas, profiling scales, or official TTFT boundaries in this sub-step.

## Source findings

| Inspected implementation | Consequence for this design |
| --- | --- |
| replica_stage_schduler.py binds the live batch through StageExecutionContext.bind_forward_group | The admitted group is authoritative; lane-local batch IDs cannot identify a shared forward. |
| ForwardSyncState partitions bindings by prefill/decode; sync_entry.py partitions waiting rooms likewise | Remove local phase from MONOLITHIC group matching and completion ownership. |
| ep_wave_inputs.py retains source_batches and creates a virtual aggregate | Retain source batch objects; use the aggregate only for EP workload prediction. |
| forward_step_admission.py promotes lane tickets to one EP ticket and restores owners | Reuse these operations exactly once per wave. |
| prefill_collective.py and decode_collective.py predict some continuation/final timing using one sample_batch | Every live source needs its own attention and final timing prediction, including same-phase sources with different shapes. |
| sklearn_execution_time_predictor.py has prefill features, decode KV features, and attn_decode_in_mixed | Pass the intact local batch to the existing predictor; do not split it into independently predicted mini-batches. |
| Batch.on_batch_end and Request.on_batch_end already apply request-specific scheduled/committed tokens | Retain the established terminal event chain and its execution/mutation signatures. |
| Request.on_batch_end grants the first generated token at final MONOLITHIC prefill completion | Preserve this convention; do not append an extra first-token decode iteration. |

## 1. Identity, membership, and lifecycle

The shared layer key is `(cluster context, replica_id, stage_id, admitted_forward_group_id, layer_id)`. The scheduler instance provides cluster context. The synchronization point, such as pre_moe/post_moe, is lifecycle state within this key. Local phase, request IDs, local batch IDs, and token count do not partition the key.

Keep the existing monotonic resolved step ID for event/EP diagnostic compatibility, but resolve it once from the common key. The same admitted group retains its identity through every layer. Different group IDs, stages, replicas, or layers must never merge even when timestamps match.

Authoritative membership comes from admitted stage owners. For each attention-DP lane, there is either one real source batch or one idle participant. Attention-DP2 means two source lanes; EP8 means eight expert participants, not eight copies of each request. Group sealing continues to use StageExecutionContext: arrivals after sealing wait for the next forward, and an idle lane cannot accept a late real batch halfway through a model forward.

Lifecycle:

`admitted -> local_attention -> waiting_at_layer -> one_EP_wave -> source_continuations -> ... -> source_stage_ends -> group_released`

At a layer rendezvous:

1. Validate the batch schedule epoch, owning lane, admitted group, and expected layer.
2. Register its attention-ready time and original source batch.
3. Wait for real peers that are already busy in this group. Never replace a busy real peer with a dummy.
4. Use the existing idle eligibility/materialization rule for a genuinely idle peer; preserve its existing EP workload accounting. Idle participation has no user request or request completion callback.
5. When the expected lane set is present, seal/consume the rendezvous and schedule exactly one existing EP wave at the latest participant-ready time.
6. On wave completion, consume the wave state once, restore all real source owners once, and continue each real source independently.

Duplicate real participation or a mismatched group/layer is an invariant failure. Preserve the existing treatment of demonstrably stale idle events as no-ops; they must not recreate a closed wave. Close per-layer state after consumption and release group state when its stage owners finish. Do not introduce an unbounded completed-event history.

## 2. Source request and attention contract

Keep `source_batches[lane_id] -> original Batch` throughout the group. Do not replace a source with the virtual EP aggregate, change request order, or synthesize request ownership from an expert partition.

Retain the original Batch reference and its existing schedule epoch, stage start, layer cursor, request/token metadata, and execution/mutation signatures. Keep explicit component timing on the existing batch alongside the prefill ledger. Do not introduce a separate source-context object or a second request-progress store.

The model forward does not commit request tokens or mutate KV progress between layers. Existing request state remains authoritative until its terminal callback. Validate this invariant; avoid cloning Request objects or adding a broad predictor input schema.

| Local request work | Inputs preserved for existing prediction |
| --- | --- |
| Prefill | Scheduled query-token length per request; already processed prefix/context; existing KV occupancy and position conventions. |
| Decode | Decode request count; each request's context as resolved by the current decode-context helper; scheduled query width for the supported runtime. |
| Mixed | Both subsets plus total batch size, total prefill tokens, composition ratio, and the existing true-mixed feature mapping. |

For every live lane and layer, call `predict_stage_execution_time(source_batch, stage_id, cluster_type, num_layers=1, layer_id=layer, include_ffn=False)` and consume `get_single_layer_attention_scope_time()`. The intact local mixed batch allows `attn_decode_in_mixed` to observe its co-resident prefill workload. Separately predicting a prefill-only and decode-only artificial batch would lose that interaction and can double-count shared projection/cache operations.

Reuse current feature extraction, KV rounding, and profile selection. This preserves available shape/KV information but does not claim a more precise predictor than the current aggregate/rounded features. Missing required true-mixed profile coverage remains an explicit error; no pure-decode fallback or old profiling data is allowed.

The EP aggregate combines real token populations across sources once. For YC's example, lane0 has request0/decode1 plus request1/prefill4096, lane1 has request3/prefill4096: real total=8193, prefill total=8192, decode total=1. Top-k assignments and dummy contributions are separate from this real-token count. The aggregate is never an input to local attention or request completion.

## 3. Timing and ownership

Let `R[g,l,d]` be the ready time after lane d's local attention. Then:

`J[g,l] = max_d R[g,l,d]`

`C[g,l] = existing_EP_wave_plan(J[g,l], all_source_batches).wave_end_time`

`R[g,l+1,d] = C[g,l] + local_attention_seconds(source[d], l+1)`

The EP planner already accounts for dispatch, expert compute, combine, and its defined post-combine work; consume its result once. Do not replace it with a new sum/max formula or multiply it by the number of source batches. Each source experiences the common wave duration as latency, while the resource execution is one shared wave.

Per-lane join wait is `J-R[d]`. It contributes to elapsed stage/request latency, but must not be relabeled CPU overhead or added again as operator compute. Maintain explicit local attention components, shared-wave critical-path components, and final local components in the source context. Stage wall time derives from timestamps. Distinguish source service latency from an additive GPU-resource-work total.

For the final layer, predict final local work from that source batch, then emit one BatchStageEndEvent per live source at its own completion time. Current PP1/CPU-disabled settings remove those optional tails, but the code must not borrow another lane's final timing. Reuse the existing timing helpers and endpoint conventions where applicable; do not re-add modeled attention or FFN as final overhead.

Ticket lifecycle remains FULL_STAGE_WORLD owners -> one EP_WAVE ticket -> restored FULL_STAGE_WORLD owners. Promotion and restoration occur around the entire source set, never once per phase. Release each final source ticket through the existing BatchStageEndEvent. The next forward cannot consume a stage still owned by the current group.

For a dense layer inside a MoE model, preserve the existing full-stage FFN path and its per-source completion; do not invent an EP collective for that layer. Continue with the same admitted group and join at the next actual MoE layer. Existing PP stage bounds and speculative completion metadata must remain delegated to their established helpers. Their compatibility is a regression obligation, while trained numerical acceptance stays limited to the frozen PP1/non-speculative case. If those reachable paths require a broader Request or predictor contract change, report the concrete dependency before expanding implementation.

## 4. Request progression and metrics

Shared EP events never call request.on_batch_end and never finish the aggregate batch. They only advance the layer protocol. For requests that were already decoding at admission, advance existing decode layer bookkeeping once per completed model layer, including those inside a mixed source. Prefill requests retain the prefill progression convention. Do not increment every request based on a group-wide or lane-wide phase label.

The terminal MONOLITHIC chain remains:

`BatchStageEndEvent -> ClusterBatchEndEvent -> GlobalBatchEndEvent -> original Batch.on_batch_end -> each Request.on_batch_end`

Retain the existing schedule-epoch/execution/mutation guards and request-ID deduplication. The shared handler emits one stage end per original source, with no additional per-request terminal event. Cross-lane duplicate real request ownership fails validation rather than relying on local deduplication to hide it.

| Request at forward start | At intermediate layers | At terminal source completion |
| --- | --- | --- |
| Incomplete prefill | No prompt-token commit; no TTFT completion | Commit its scheduled prompt tokens; mark prefill only if the prompt is complete; record the prefill boundary once. |
| Final prefill in current MONOLITHIC case | Same as above | Preserve existing first-generated-token credit exactly once. |
| Already decoding | Advance layer bookkeeping once per layer | Advance its scheduled/committed decode count once, reset the layer counter, and preserve its previous prefill timestamp/TTFT. |
| Idle participant | Participate in required shared communication | No Request, TTFT, generated token, or completion row. |

For the example, request0 advances one decode token; request1 and request3 complete their respective 4096-token prefills and receive their existing first-token credit. Request0's old prefill completion time is unchanged. A 4096/1024 request must finish with 1024 output tokens total, not 1025.

Keep Frontier's existing internal prefill endpoint and official vLLM server TTFT comparison contract. The protocol fix adds actual simulated EP wait to the critical path; it does not justify attributing a remaining server TTFT difference to CPU overhead.

## 5. Module design and implementation boundary

Reuse the existing synchronization entry operations, normalized to one MONOLITHIC forward identity and waiting room. Add one small shared completion helper that consumes the common wave, restores source ownership once, and invokes the existing source-local completion helpers. Batch already owns request/token metadata and completion signatures; avoid an additional source-context state hierarchy.

Keep the existing prefill/decode events as adapters to the shared MONOLITHIC identity and completion mechanism. Their names describe the entry source, not the phase of every collective participant. Source membership is authoritative. This avoids new event types and preserves existing DES ordering. PDD/PD-AF keep their phase-specific semantics. Every MONOLITHIC source combination uses the common synchronization mechanism.

| Touch point | Planned responsibility |
| --- | --- |
| New scheduler/utils/forward_collective.py | Consume shared completion, restore all owners once, and delegate source-local continuation. |
| forward_sync_state.py and sync_state.py | Common MONOLITHIC identity/room ownership; preserve other cluster-specific contracts. |
| replica_stage_schedule_event.py | Initialize the decode component ledger alongside existing phase metadata. |
| base_cluster_scheduler.py | Thin delegation at the existing scheduler seam; avoid adding lifecycle implementation to this large module. |
| ep_wave_schedule.py | Separate common EP planning/ticket operations from phase-specific completion adaptation; shared route consumes the same planner. |
| prefill_collective.py / decode_collective.py / sync_entry.py | Remove obsolete MONOLITHIC routing or expose existing reusable local operations only as necessary; retain disaggregated behavior. |
| ep_wave_inputs.py | Reject duplicate request ownership across real source lanes before prediction. |

Reviewed scope is approximately8–10production files plus focused tests, reduced from the previous10–13estimate by retaining event adapters. Request and predictor base modules need no behavior change; an observed need to change those contracts requires a separately evidenced scope decision. No new feature flags, metric schema, model category, network backend, config schema, or EventType entry is proposed.

## 6. Verification and acceptance

Design inspection and implementation are complete. Focused73checks and the trained three-request reproduction PASS; full100request replay is running. The table defines the acceptance checks; concrete observed results and limits are recorded in test_report_2026-09-08_shared_forward_sync.md, rather than treating every row as independently proven by source inspection.

| Gate | Specific failure detected | Passing condition |
| --- | --- | --- |
| Phase combinations | Phase-dependent join or duplicate EP dispatch | P/P, D/D, P/D, mixed/P, mixed/D, mixed/mixed all produce one wave per key; exercise both lane orders. |
| Identity/lifecycle | Wrong-group merge, repeated completion, late admission | Different group/layer/stage keys remain isolated; exact real duplicates fail; stale idle events do not reopen a wave; sealed groups reject late membership. |
| Idle and ownership | Replacing a busy real peer or ticket leak | Real+idle completes; busy real peer is awaited; one promotion/restoration per wave; final rooms/tickets/queues empty. |
| Source-specific attention | Reuse of sample_batch timing or cross-lane KV | Predictor spy receives each original batch and its different context; deterministic unequal durations produce the expected max join time and individual next-layer ready times. |
| Local mixed attention | Lost mixed profile selection or repeated shared operators | Original local batch reaches the real feature builder; prefill subset/context and attn_decode_in_mixed features match request metadata. |
| Request completion | Double token credit, changed old TTFT, missed layer reset | Decode in mixed increments once per layer and once per forward; completed prefill timestamp unchanged for old requests; new prefill timestamp set once; final output count exactly1024. |
| Timing accounting | Counted collective per lane/phase or wait counted as CPU | One resource wave; per-source wall time equals event interval; explicit modeled components plus waits/tails reconcile without double counting. |
| Existing behavior | Regressions in same-phase, PDD, and event ordering | Relevant existing sync/admission/EP materialization and sequential PDD tests pass; equal-time event controls remain deterministic. |
| Real three-request reproduction | Fix passes fabricated timing but still stalls with trained predictor | Original4096/1024 first3arrivals complete3/3 with trained current-task profiles and empty final scheduler state. |
| Full calibration case | Incomplete numerical output or malformed source batch ledger | Fresh full100request4096/1024 run completes; request IDs/output lengths verified; first/all-prefill membership comparison produced. |

Extend existing tests/unit/test_forward_sync_state.py, test_shared_forward_group_admission.py, and EP materialization tests where they exercise the public scheduler seam; add one focused MONOLITHIC mixed-forward regression file when needed. Use a bounded predictor spy for deterministic timing/feature assertions, then the actual trained predictor reproduction. Do not substitute dummy-timing PASS for numerical validation.

After the functional gates, commit the completed code sub-step. Run the full calibration with new output and fresh predictor/collective caches on CPU master. Use only this current-main task's freshly collected H200 profiles and validated replay-02 reference, never the old task's numerical data. Record predicted/actual/absolute/relative errors for official-server-TTFT comparison and secondary TPOT/E2E/throughput. A <=10% mean TTFT gate belongs to calibration; it is not guaranteed by synchronization correctness. If still outside the gate, continue source-grounded RCA without a residual CPU constant.

Dependency sequence:

`design_review -> shared_lifecycle_and_source_timing -> focused_regressions -> trained_3request_replay -> code_commit -> fresh_100request_replay -> {batch_comparison, metric_comparison} -> next_RCA_or_acceptance`

The two final comparisons are independent after valid full artifacts. No additional H200 run is necessary merely to design or execute the CPU synchronization repair; rerun groundtruth only if its controlled inputs or required instrumentation change.

## Review result and limits

All three YC requirements have explicit mechanisms and acceptance checks. Inspection identified and addressed the source-specific attention issue and first-token credit convention in this design. The reviewed implementation is committed as8ba22b49 after73focusedchecks and a successful original three-request trained reproduction; full-case numerical validation remains pending. Existing vLLM local mixed-batch evidence does not establish exact cross-DP diagnostic round pairing. Exact scheduling and final error after repair remain measured outcomes, not design assumptions.
