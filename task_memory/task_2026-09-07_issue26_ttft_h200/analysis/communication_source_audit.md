## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded current runtime communication source paths pending measured RCA. |

# Communication source audit

Status: incomplete numerical RCA; no repair authorization or timing attribution.

The actual Frontier per-layer path is `_predict_expert_parallel_phase_operator_times` -> `_predict_comm_operator` -> the `CommOperatorSpec` entries in frontier/operators/families.py. Both dispatch and combine entries use `collective_alias=alltoall`, EP domain, EP participant count and the EP payload builder. The older `_get_expert_parallel_communication_time` function alone is insufficient to establish the executed path.

In the selected vLLM Qwen3 FusedMoE path, DP2 and explicit naive backend activate `do_naive_dispatch_combine`. `NaiveAll2AllManager.dispatch` calls `naive_multicast` separately for hidden states and router logits; it broadcasts each DP segment over the DP group. Combine all-reduces global hidden states across DP2, then slices the local segment. `reduce_results=True` additionally calls tensor-model-parallel all-reduce; its scope label is `expert_parallel_allreduce` for EP>1 even though the actual communicator is TP4. Scope names alone do not identify the collective or domain.

Sources: Frontier frontier/operators/families.py:551, sklearn_moe_execution_time_predictor.py:1583, sklearn_execution_time_predictor.py:6149; vLLM vllm/distributed/device_communicators/all2all.py:17, fused_moe/layer.py:1790 and1597, models/qwen3_moe.py:146.

Next evidence: actual identified per-worker diagnostic dispatch/combine/reduction scopes; logical batch token sizes across both DP lanes; routing vectors and payload ownership. This candidate semantic difference must be quantified before selecting a correction. Retain collective_sim/nvlink_analytic; bandwidth fitting cannot settle a primitive/domain mismatch.
