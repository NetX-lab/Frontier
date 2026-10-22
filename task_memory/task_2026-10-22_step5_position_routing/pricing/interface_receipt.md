# Pricing interface receipt

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Publish first selected-operator and immutable plan interfaces. |

TaskID: D29-PRICING-IMPLEMENTATION. Baseline: 1aee0af.

## Selected MoE operators

`frontier/operators/families.py:get_moe_profiling_operators(context=None)` returns
ordered OperatorSpec objects from MOE_FAMILY. Context is a mapping containing
`model_architecture_profile`, `routing_runtime_path` and, for Step position mode,
actual native `quant_dtype` (None or `float8_e4m3fn`) and `defer_input_quant` (bool).
The supplied context must come from native selection; weight FP8 alone is not
prepare evidence. No-context preserves the four legacy targets in their order.

- BF16/deferred prepare: gate linear, routing topk, inclusive grouped GEMM.
- Reached nondeferred FP8: gate linear, routing topk, input quantize, inclusive GEMM.
- No separately charged shuffle in position mode.

`moe_input_quantize`: MOE_FAMILY / ROUTED, RESHAPE, MEMORY, MOE_TP,
EP-agnostic, execution attr `moe_input_quantize_time`. Actual model resolver
at layer3/TP1/EP16 succeeds. Registry inventory includes it without requiring
it as a legacy profiling target.

`step5_operator_binding(..., context=...)` binds FP32 gate/router and FP8 prepare
for this selected path; context is part of fixed binding identity. Legacy
no-context binding remains unchanged. CORE must use the same context for target
selection, exact duration/count pairing and model/cache identities.

## Plan and backend capability

`frontier/cc_backend/collective_plan.py` defines immutable `CollectiveTensor`,
`CollectivePlan`, `CollectivePlanCost`. Plan includes ordered exact physical DP2
sizes, tensor geometry/dtypes/strides, all eight DP pairs or both TP8 groups,
actual graph mode, native/context and buffer receipts. `stages`, `to_dict()`,
`cache_key()`, `endpoint_bytes(rank)` preserve size/root/tensor order; no averaging.
Cost requires sixteen actual paired native kernel counts and a count receipt,
separate from algorithmic stages; duration must equal N+max(local)+remainder.
CPU fixtures remain explicitly CPU evidence.

CommOperatorSpec supports optional `plan_builder`; scalar payload/alias plus plan
is rejected. BaseCCBackend.predict_collective_plan raises explicit unsupported
capability; no other backend silently approximates a plan.

## Shared native-call planner

`forward_collectives(..., native_call_names=None)` is the sole selected planner.
Explicit Step subset must be nonempty/distinct/reached and retains native order.
Only explicit selection changes combine descriptor to
`reduce_scatterv_into_output` / `AgRsAll2AllManager.combine_into_output`.
No-option legacy list and endpoint bytes are unchanged.

## CORE-owned required numerical facts

Source gate/router/prepare/shared and post-combine TP widths are physical ni,
even with live0. Inclusive expert frame is gathered M=sum(ni), with local live
assignment counts separately retained. Every applicable physical scope needs
paired exact duration/count evidence; zero-live is not an early-return rule.
Native inclusive experts own permutation/GEMMs/intermediate work/unpermute and
weight finalization, excluding source prepare and all communication.

Native DEEPGEMM uses actual alignment/capacity/layout evidence, not Triton padded
all-expert sum. CPU-count rounded M and native physical-capacity bound remain
distinct. Actual native auto selection (including E8M0 OR shape/layout criteria)
must be retained rather than forcing TRITON.

## Dependencies not duplicated

CORE/SOURCE must publish immutable WaveRoutingSource/SourceLaneRouting,
MoERoutingProfileInput parser, `source_frame_tensors(frame, *, device)` and
`verify_step5_profile_runtime(manifest_path, *, callables=())`. Native owns the
real `set_routing_positions` setter/factory and derived adapter. Pricing does not
provide replacement source/mask/provenance APIs.

## Validation

Thread-capped frontier-py310, no bytecode/cacheprovider: selected pricing,
comm-family and Step binding-reuse suites: **60 passed** in10.81s. This establishes
CPU contracts, not measured GPU parameters/native-runtime qualification.
