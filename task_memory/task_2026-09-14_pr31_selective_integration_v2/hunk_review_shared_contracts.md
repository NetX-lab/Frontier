## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Reviewed all original workload/operator/type hunks and connected callers. |

# Shared contract review

Reviewer: root implementation agent; direct source review against pinned main `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`. This is source/caller evidence; the final validation manifest records final-source execution separately.

| Original IDs | Disposition | Inspected boundary and evidence |
| --- | --- | --- |
| H247–H253 | RETAIN | `moe_ep_workload.py`: `generate_moe_routing_ratios` preserves layer seed and normalized expert order; both MoE and disaggregation predictors call it. Workload constructor validates contiguous participant IDs, per-expert ownership, per-EP counts and conservation before freezing mappings and creating immutable lane descriptors once. `lane(ep_id)` then returns that canonical descriptor; batch builders and expert-parallel helpers consume it. Existing shared-routing and typed EP/conservation suites exercise the contract. Caching changes neither integerization nor routing identity. |
| H254–H256 | RETAIN | `operators/binding.py`: architecture linear declaration renamed to `attention_linear_ops`, consistently consumed through architecture profile. The registry owns sharded/replicated classification; the operator caller does not branch on model names. No second classification mechanism introduced. |
| H257–H260 | RETAIN | `operators/typed_contracts.py`: runtime attention resolution routes required shape/topology through the model-owned contract; architecture-specific operator names derive from registry profiles. Producer metadata validation remains distinct from numerical predictor admission. Existing operator-family and typed metadata tests cover selectors and unsupported operators. |
| H466 | RETAIN | `DeviceSKUType.MI355X=9` extends existing enum without renumbering; config registry supplies platform/capacity/topology. No caller-side name classification is required. CPU discovery tests establish schema/admission only. |
| H467 | RETAIN | `MeasurementType.DEVICE_EVENT` is an explicit enum member; manager/predictor registry and training paths keep it separate from CUDA_EVENT/KERNEL_ONLY. W05 timer and W06 per-field path tests cover active selection; no native parity inferred. |
| H468 | RETAIN | `NodeSKUType.MI355X_UBB=10` extends the existing node registry without modifying existing values or standard scheduling policies. |

No production edits were required by this bounded review. All files are below the critical-module size threshold except other independently reviewed modules; no arbitrary split or cosmetic rewrite was proposed. Native observations remain separately SKIP/NOT RUN.
