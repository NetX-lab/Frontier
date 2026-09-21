## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded scoped module cleanup and split boundaries before D012. |

# D012 module boundaries

The large config/predictor/shared-manager modules are touched only at runtime configuration/identity boundaries. Predictor constructor stores _moe_gating_routing_runtime_path without any reader; remove this redundant assignment and resolve through its existing helper, which reads ReplicaConfig after base initialization. Global routing allocations remain unchanged. Config gains one declarative optional field and existing copy/flatten propagation; no alternative configuration path is added.

Runtime validation/default selection stays in the existing small frontier/moe_routing_runtime.py module. A later large-module split should extract model identity/storage/projection as one boundary, keeping training and estimator lifecycle in the manager; D012 does not perform that unrelated refactor. Shared-manager cleanup is documented in its focused implementation report. The new runtime uses the existing canonical supported-path validator and no new fingerprint or parallel registry.
