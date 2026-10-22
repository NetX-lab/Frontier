# CORE/SOURCE interface receipt

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Publish first canonical count, source tensor and provenance interfaces. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. These interfaces are CPU/source plumbing, not native/GPU qualification.

- `frontier.moe_routing_policy.resolve_routing_count_policy(policy_id)` returns the registry-owned frozen `RoutingCountPolicy(runtime_path, requires_source, count_experts)`. The callable takes keyword-only `routing_ratios, routing_source, routing_token_count, router_topk, total_expert_num`; only `materialize_layer_ep_workload` invokes it in production. Ratio calls still delegate to the unchanged Hamilton owner.
- Frozen `TokenInterval(start,length)`, `SourceLaneRouting(lane_id, source_batch_id, source_global_id, forward_group_id, graph_mode, gating_runtime_context, physical_rows, live_rows, intervals)` and `WaveRoutingSource(policy_id, source_lanes, gathered_physical_rows)`. `source_global_id=-1` is the existing ungrouped Batch value; it is copied, not invented. DP order is contiguous; physical gathered width equals the supplied source-width sum. `WaveRoutingSource.live_rows` sums only owned intervals. Cache keys may directly contain the complete immutable source.
- `build_wave_routing_source(source_batches, *, policy_id, cluster_type, gathered_physical_rows, model_config=None)` reads actual bound Batch query/context arrays, skips dummy live intervals, and freezes the existing hot-gating predicate. No virtual/EP-lane inference. `LayerEPWorkload` and each canonical `EPLaneWorkload` carry `routing_source`; physical `routing_token_count`, `live_routing_token_count` and `padding_row_count` remain separate.
- `frontier.profiling.moe.inputs.MoESourceFrame(source_lane_id, query_lens, context_lens, physical_rows, gating_runtime_context)`; `source_frame_tensors(frame, *, device) -> (positions, is_padding)` produces signed int64 positions and bool masks once before timing. Padding tensor positions are inert storage. Both native dispatch probe and adapter must use this one function. `MoERoutingProfileInput`, `MoERoutingProfileCell`, `routing_profile_cell` and `read_routing_profile_inputs` own the one-dict/explicit-input schema `moe.position_cyclic.inputs.v1`.
- `frontier.profiling.common.runtime_identity.verify_step5_profile_runtime(manifest_path, *, callables=()) -> {shared, imports, callables}` verifies exact installed module file/spec/hash and actual Python class/bound function/code paths against the sealed full installed-source receipt. Gate/norm/oracle/kernel builders pass their actual callables, including bound methods. JIT wrappers expose their underlying `fn`; uninspectable compiled entries are not claimed as Python proof. No worker-self-hash fallback.

## Verifier artifact contract

`manifest.json` has `image`, `sha256` (stage-relative sealed files), and `native_binding`:
`native_runtime_adapter`, `native_reference`, `profile_generation`, `installed_package`, `installed_source_hashes`, `serving_pin_count=53`, `model_config_revision`, `model_config_sha256`, `native_source_receipt`, `native_derivation_manifest`.

The sealed registry lives at `tests/comparison/dp_placement_pp/vllm_replay.py`, with its existing dependency `tests/comparison/stage_admission_pp/vllm_burst_driver.py`. The full receipt schema is `native_installed_sources.v1`, as already specified in profile_source_design.md: actual `image/base_image/installed_package/python_version/derivation_manifest_sha256/sources`. The derivation receipt supplies `base_image` and package-relative `derived_source_sha256`. The final image owner emits these actual artifacts after build; image SHA is never fabricated by this API. The derived reference must differ from base and effective routing must be `position_cyclic`, padding skip enabled. Shared identity matches across producers; imports/callables legitimately differ. Unbound legacy RuntimeIdentity entirely omits native_facts and preserves its old fingerprint.

## Dependencies requested from root

PRICING: exact delivered API/signature for `get_moe_profiling_operators(context)`, native quant/layout facts and prepared-source/gathered inclusive replay callables; selected schemas/plan config. NATIVE: actual factory/set_routing_positions contract plus finalized derivation receipt keys. Root alone authorizes peer commit import/merge. H04 must remove the zero-live shortcut only for applicable source-required physical scopes and use paired exact duration/count lookup; no new-mode shuffle charge.
