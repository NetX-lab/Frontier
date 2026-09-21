## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Independently reviewed the communication proposal and narrowed it to the reached single-node runtime and existing five-phase execution model. |

# D019 communication design review

## Verdict and recommendation

**REVISE the original proposal to the bounded design below, then submit that material configuration/physical-workload decision to YC.** The source and algebra support replacing the ideal protocol with explicit `vllm_naive`. A comm-duration-only replacement is insufficient: the same physical DP populations must drive MoE routing and message sizes, and top-k/shuffle must move after multicast. These changes fit the existing forward cohort and five MoE phases; a new event hierarchy and multi-node broadcast implementation are unnecessary for this case.

The recommended bounded revision is coherent: one explicit protocol selector with the existing ideal default; a physical source-population descriptor at the EP-wave input seam; native broadcast prediction for wholly intra-node NVLink groups; and reuse of existing five-phase timing with protocol-specific placement. Keep collective_sim/htsim as the selected backend. Only the newly supported native intra-node broadcast bypasses an inapplicable network runner; existing collectives retain their current path. Multi-node/native-broadcast combinations remain explicitly unsupported.

Reviewer: `/root/repair_review`. Read-only production review; only this report was written. Source target: current Frontier worktree HEAD `fc205071f4a04f78591ec2a7ef2912fa91d75912`, C's `analysis/d019-communication.md`, and pinned diagnostic vLLM `all2all.py`/`fused_moe/layer.py`. GPU measurements and production implementation remain pending.

## Independently established facts

| Fact | Source | Implication |
| --- | --- | --- |
| `NaiveAll2AllManager` copies a local tensor into its global-row buffer and broadcasts each source-DP segment; it does this for hidden states and router logits. | vLLM `distributed/device_communicators/all2all.py:28–55` | Dispatch payload depends on each source DP's physical token count, hidden width and expert-logit width, not local routed-assignment count. |
| Combine performs DP allreduce, retains local rows, then `reduce_output` applies the attention TP reduction. | Same file:57–66; `model_executor/layers/fused_moe/layer.py:1852–1861` | Replace the ideal EP return with this hierarchy. Adding TP reduction on top of the complete ideal return duplicates the modeled expert aggregation. |
| Current EP phase prediction returns only the two ideal A2A durations from local EP routed-token payloads. | Frontier `sklearn_moe_execution_time_predictor.py:1577–1605`; `operators/families.py:376–390,550–580` | Neither an algorithm string nor a topology parameter currently selects the naive runtime. |
| `EPWaveInputs` retains source DP batches but filters idle batches before building the aggregate token count. | `scheduler/utils/ep_wave_inputs.py:10–70` | It is the appropriate narrow seam for physical source populations. Today an idle lane contributes zero to the MoE aggregate. |
| Idle lanes are empty `Batch(requests=[], num_tokens=[], is_idle=True)` objects. | `scheduler/utils/sync_entry.py:88–98` | A dummy physical token must not be represented by a fabricated request or added to request progress. |
| `LayerEPWorkload` is a pure aggregate routing/ownership descriptor, without source-DP lifecycle semantics. | `moe_ep_workload.py:1–6,78–91` | Do not make it the owner of request/source maps. Supply its existing aggregate routing-token input from the new physical descriptor. |
| Current pre-dispatch timing includes both gating components and shuffle. | `entities/execution_time.py:802–835` | Naive protocol must place local router before dispatch and global top-k/shuffle after it. Merely replacing dispatch/combine scalar values misrepresents phase order and input width. |
| Existing five-phase accessors and wave plan already model pre, dispatch, routed, combine and post. | `scheduler/utils/expert_parallel.py:79–181,188–238` | Reuse these phases and existing completion callbacks. No new synchronization/event family is needed for the bounded correction. |
| `predict_broadcast` exists but raises; `CommOperatorSpec` and predictor dispatch lack a broadcast alias. | `cc_backend/backends/collective_sim_cc_backend.py:789–799`; `operators/spec.py:187`; `sklearn_execution_time_predictor.py:6185–6245` | Add broadcast through the current communication mechanism, not a reinterpretation of allgather/P2P. |
| Collective-sim calls htsim before evaluating NVLink time and before discarding surrogate intra-node network time. | Submodule `python/collective_sim_core/predictor.py:21–34,91–99` | There is no existing native intra-only fast path. The bounded broadcast path must deliberately bypass this runner when its physical group is wholly intra-node and NVLink analytic is active. |

## Algebra and no-double-count contract

For token x, let z[d,t](x) denote its weighted contribution from the expert shard on physical rank (d,t). The DP reduction computes u[t](x) = z[0,t](x) + z[1,t](x). After each rank slices to its source-DP rows, the TP reduction computes y[d](x) = sum_t u[t](x), which includes every EP shard exactly once. This agrees with the final aggregation represented by the existing routed EP return, despite different message traffic and staging.

The post-combine TP group is the attention TP group. It must not be mapped to MoE TP=1 or appended after an already complete ideal EP return. Local `moe_sum` is a different operation: it reduces a rank's top-k local contributions before the DP/TP global aggregation and is owned by the corrected compute profile.

## Bounded design and exact responsibility

1. **One runtime choice.** Add a validated declarative MoE communication-runtime field to the existing replica config, preserving the current ideal protocol as default and explicitly selecting `vllm_naive` for this task. Keep runtime selection separate from CC backend and top-k implementation. Two protocol entries should own their behavior in one small module/table. No inference from model names, TP/EP values, uniform routing, or scheduler names.
2. **One physical input descriptor.** Extend the existing wave-input preparation seam with per-source-DP real and physical token counts. The vLLM runtime policy supplies the reached one-token idle forward contribution; it does not mutate idle Batch request counts or completion semantics. Derive global physical MoE width and naive message sizes from the same descriptor. Feed 4097 into existing aggregate routing materialization; preserve 4096 real prefill tokens for request accounting and attention/KV semantics. The descriptor must also accept existing pure/mixed source batches rather than special-case request 0.
3. **Existing five phases.** The protocol owner assigns local norm/router/staging to pre-dispatch, hidden/logit broadcasts to dispatch, global top-k/shuffle/experts/local sum to post-dispatch compute, DP global reduction to combine, and attention TP local reduction to post-combine. Preserve each scope's measurement-family and ownership; staging is compute/memory work and must not be charged again through inclusive dispatch timing. Keep current phase conservation checks.
4. **Native intra-node broadcast capability.** Implement the existing backend broadcast API and declarative alias support. Give collective_sim a native broadcast kind with tensor bytes and group semantics. Before launching htsim, accept this kind only when its actual participant ranks belong to one server, `nvlink_analytic` is active, and intra-server traffic is excluded from the network simulation. Calculate and report native intra time and zero *applicable* inter-node work. Do not fabricate an htsim result. Multi-node, legacy-fabric and otherwise unsupported native-broadcast paths fail explicitly. Other collective kinds continue through their existing htsim integration.
5. **Bounded physical model.** Implement one documented broadcast algorithm/critical-path contract appropriate to the observed runtime and NVLink measurement. The two-rank case needs one root-to-peer transfer. The public operation remains broadcast; do not make a hidden `DP==2 -> send_recv` alias. Larger wholly intra-node groups may be modeled only under the same explicit algorithm contract, with their calibration coverage declared. Do not claim their NCCL performance is validated by the two-rank measurement.
6. **Preserve shared completion.** Use current phase times/aggregate barriers and request callbacks. Do not revive retired DP gather/scatter fields. Separate protocol primitive diagnostics from complete phase totals to avoid parent/child duplication.

This design needs ordinary updates across config construction, a small protocol owner, EP-wave input plumbing, operator dispatch, phase timing, collective_sim adapter and its native intra estimator/predictor plus focused tests. It is a material shared contract change even though each substep can remain small. It avoids multi-node flow generation, multiple broadcast algorithms, new scheduling events, and a general traced-routing import feature.

## Options and practical costs for YC

| Option | Deliverable and actual cost | Calibration consequence |
| --- | --- | --- |
| **Recommended: explicit runtime + physical inputs + native intra-node broadcast** | One shared configuration/descriptor decision; several small existing-seam edits and focused numerical/phase/backend checks. Preserve ideal default and current backend. Multi-node native broadcast remains unsupported. | Can represent the reached naive first-forward work without deliberate payload or phase mismatch. Fresh measurements still required. |
| Defer protocol change and calibrate only supported allreduces | No new protocol schema now; can independently evaluate launch/transfer discrepancy while A/B continue. | Ideal EP versus naive traffic and the missing physical dummy remain open; full first-forward op correction cannot be called complete. |
| General native broadcast including multi-node htsim flows | Additional root/topology/algorithm contracts, network traffic generation and cross-node validation. | Extends coverage beyond the current single-H200-node case; not required to begin or close this bounded target. |

The recommended path has the smallest coherent implementation scope that addresses the demonstrated current protocol discrepancy. It is more work than a scalar comm correction because the evidence proves message ownership and compute ordering differ, not merely their timing constants.

## Remaining implementation choices and evidence gates

- **Runtime selection and support envelope** are the high-value user decision. Recommend one approval covering the bounded design above, including same-node-only native broadcast and preservation of the ideal default.
- **Physical dummy policy** must be bound to the pinned vLLM forward semantics/recorded metadata. A common descriptor is essential; a magic global `+1` is not acceptable. First-forward source metadata and exact-routing probes establish the initial contract. A full dummy scheduling lifecycle beyond the agreed forward cohorts is not silently included.
- **Broadcast algorithm and launch/transfer parameters** must follow fresh primitive/in-context evidence. The identified 50-us-per-ring-step contribution explains model structure but does not determine a replacement value. C's GPU bench remains the next measurement; no timing parameter change is endorsed by this review.
- **Phase and group timing** retain the current global barrier abstraction. Real DP pair and TP group overlap/arrival skew may cause residual error; record that practical limit and test whether it is material before proposing finer synchronization.
- **Diagnostic representation** may use named primitive children under existing dispatch/combine boundaries. Register growing operation categories centrally and ensure complete phase totals do not also add their children. No generic workflow framework is required.

## Acceptance and review limits

1. Default ideal protocol produces unchanged semantic results in its focused existing tests.
2. For source populations [4096 real, idle], real request/KV progress stays 4096; physical populations are [4096,1]; aggregate routing is 4097; exactly 32776 top-k assignments conserve across eight EP ranks.
3. Mixed/local-decode source batches retain request composition and per-request progress while physical payloads derive from their actual total scheduled tokens.
4. Dispatch hidden/router payloads, DP combine width, post-TP local widths and phase ordering match the pinned runtime. Algebra check reaches one sum over eight ranks with no duplicated ideal return.
5. Native broadcast uses the same-node NVLink capability, reports its actual execution path and rejects multi-node requests. Existing htsim-backed collectives continue using htsim; no silent P2P/allgather substitution.
6. Fresh eight-rank primitive evidence, matching in-context scopes and complete first-forward P/S/V/critical-path comparison precede numerical acceptance.

This review inspected source and the written proposal only. Initial guessed source paths produced file-not-found errors; subsequent `rg --files` located the actual modules cited above. No GPU, benchmark, production edit or timing fit was performed. Root may continue independently authorized A/B GPU work while YC reviews the shared design decision.
