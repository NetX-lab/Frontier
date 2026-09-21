## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Prepared explicit runtime selection for the user-requested uniform routing case. |

# D012: uniform routing runtime selection

## Established cause

D011 enables existing VLLM_MOE_UNIFORM_ROUTING=1. vLLM fused_topk calls uniform_topk; token i chooses (i*8+k)%128 for k0..7, weights1/8. For N tokens, expert e receives floor(N*8/128)+[e<(N*8)%128], identical to Frontier balanced Hamilton allocation for the same N. Different source/dispatch token populations remain a separate workflow issue.

Frontier moe_routing_runtime.resolve_moe_gating_routing_runtime_path currently maps balanced/skewed/zipf to standard_fused_topk and random to uniform_topk. There is no explicit runtime selector in config. Switching Frontier to random changes the user's required expert distribution and is not a valid workaround. Relabeling standard profiles would misrepresent the measured implementation.

## Proposed bounded change, pending YC design agreement

Add optional ReplicaConfig.moe_gating_routing_runtime_path, empty by default. Reuse the existing supported runtime-path validation and extend the existing resolver to accept an explicit override. Unset retains existing mapping. Propagate the field through existing ReplicaConfig construction/copy paths. Pass the override at both shared model manager call sites and predictor runtime resolution. Keep expert-ratio generation untouched.

Expected production ownership: frontier/config/config.py; frontier/moe_routing_runtime.py; frontier/execution_time_predictor/shared_prediction_model_manager.py; frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py. Estimated40–70added/changed lines plus focused tests; actual diff must be checked. Existing >2000line modules get narrow call-site changes; runtime selection stays in the small existing routing-runtime module. No new allocator, trace reader, or category inferred from strings outside the resolver.

## Verification and next-member check

1. Default balanced remains standard_fused_topk; explicit balanced+uniform_topk resolvesuniform and keeps1/128 weights.
2. Unsupported explicit runtime fails; unavailable matching profile rows fail without fallback.
3. Shared training and predictor use the same explicit path; CLI field reaches the effective ReplicaConfig.
4. Fresh uniform profiles in both existing contexts, fresh cache training, and actual CPU-master case complete100requests.

Supported runtime membership remains owned by the existing canonical set and validator; callers pass a selected value. A future supported runtime extends that central contract and its profiling implementation rather than adding per-caller classification.

D011 authorizes the uniform vLLM experiment and new profile collection independently. D012 is needed for this shared config/interface extension, not a repeated approval of the experiment. No core edit applied before agreement.

## Model-sharing review update

Read-only review found that shared-manager FFN dedup signatures and typed/legacy in-memory model registries omit routing runtime identity. Disk data hashes already include the filtered routing_runtime_path column. A public per-Replica override must carry runtime through relevant existing in-memory sharing/projection identities as well, so heterogeneous supported cluster configurations cannot silently select a model for the wrong runtime. This is part of the same explicit-selector correction; the initial40–70line estimate is not yet a verified implementation bound. Detailed review: analysis/cpu-master-runtime-01/runtime-selection-review.md (when complete). No core mutation or standalone numerical run is authorized by this analysis.
