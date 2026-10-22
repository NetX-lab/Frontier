# Balanced routing execution identity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Completed source-level mismatch analysis; proposed explicit per-case runtime selection and existing native registration. |

Analysis status: PASS (source question resolved). Correction state: approved under D-V4-BALANCED-RUNTIME-SELECTOR. Native/kernel/four-metric qualification: UNVERIFIED.

## Evidence and gain

- `frontier/moe_routing_runtime.py:48-62` derives `standard_fused_topk` from balanced load ratios. The model profile may override it globally, but no existing per-case runtime selector exists. Four production call sites resolve this identity: layer_contract_resolution, prediction_family_trainers, sklearn_moe_execution_time_predictor and moe_routing_workload.
- Pinned native router_factory selects RoutingSimulatorRouter when its existing simulation environment variable is set. The existing RoutingSimulator registry has uniform_random/normal_routing, not deterministic balanced routing. Its unchanged register_strategy API can add the required policy without changing native/C26 source.
- The native balanced tensor operation is a different operator implementation from standard_fused_topk and from the existing Frontier uniform_topk loop/clone. Relabeling either existing measurement as the new runtime would be invalid.
- A per-case selector delivers exact operator identity and protects old measurements/caches/default configurations. The existing distribution field cannot supply it: load distribution and executed routing kernel are distinct properties. A global V4 profile change would also change unrelated/default cases.

The latency/fidelity percentage of the new runtime has not been measured. The requested configuration exception is for selecting/identifying the actual operation, not fitting latency or adding a small tuning factor. Its measured cost remains missing until qualified evidence exists.

## Supported native strategy

Keep source2cf and the accepted single C26 overlay unchanged. Register balanced_topk through an explicitly called, standalone module, staged through the existing worker-extension and isolated-profile bootstrap paths. Do not mutate environment on module import or silently select another policy.

For TP/SP chunks in DP-major EP order, use the prefix sum of existing native CPU local_sizes as the local token base B. Route token t, choice k to `((B+t)*K+k) mod E`, with K/E from actual inputs. `T_ROWS=sum(TP*ceil(padded_width_dp/TP))`, `A=T_ROWS*K`; each expert receives floor(A/E), then the first A% E expert IDs receive one more assignment. This matches the existing Frontier integerization. Capture-time equal widths make FULL replay offsets fixed; eager unequal widths use the actual prefix, not rank*n. Padding participates. CPU controls must verify actual native context/rank conventions rather than assume them.

The simulation router overrides native hash routing as well as ordinary top-k. That is the requested temporary balanced policy and must be explicit in run identity. Actual selected backend/kernel and graph behavior remain fresh-run acceptance items.

## Proposed shared-interface extension

Add optional per-case `moe_gating_routing_runtime_path` to ReplicaConfig, defaulting to existing resolution. The central resolver handles explicit values through the existing runtime-path registry; register balanced_topk once and thread the selection through the four callers. Both producer and consumer retain the actual `routing_runtime_path` identity. Existing metadata/filter/hash ownership must observe the new selection; default resolution and unrelated cases must stay unchanged.

Use the same balanced tensor function when profiling that operation. A missing balanced_topk timing row must produce the existing typed coverage failure; never use standard_fused_topk/uniform_topk data or zero CPU costs as a replacement.

Expected maintained-code boundary: ReplicaConfig, central runtime registry/resolver, four call sites, the standalone strategy, profiling selection/implementation, existing bootstrap/binding logic and focused tests. Request an upper bound of14 source/test/binding files and500 changed lines for this routing feature, including the independently authorized strategy. Stop and report if the concrete diff exceeds it. Generated sealed snapshots remain byte-bound copies, not places for additional logic. No extra GPU allocation is requested; previously approved prefill/native budgets apply after readiness.

## Verification and dependency

1. Existing-registration strategy/CPU conservation and shared config/caller implementation are now approved under D-V4-BALANCED-RUNTIME-SELECTOR; the aggregate routing bound is14 maintained files/500 changed lines.
2. Verify TP1, equal and unequal DP/SP widths, padding, global expert counts/remainder, dtype and unique top-k; compare the real Frontier materializer.
3. Verify selected runtime reaches actual producer/consumer/cache paths; default behavior remains unchanged, and absent/wrong timing provenance rejects.
4. Review and bind the same module/configuration into fresh packages. Verify actual worker policy/backend/source origins and native/GPU execution in the already approved runs.
5. Align workflow and GPU/op gaps first. CPU-overhead measurement is deferred to the final calibration phase.
