## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Captured actual first-stage prediction branches, feature tuples, filtered CSV rows and the existing-cache P values. |

# D019: actual predictor-query receipt

Nine of the eleven nonzero first-forward compute models return measured exact values. Only `moe_shuffling` and `moe_grouped_gemm` miss their exact feature keys and actually invoke `RandomForestRegressor.predict`. Each of those two results is then reused through the runtime cache. This establishes the prediction path; it does not establish the size of predictive error without matched S measurements.

The diagnostic stopped before handling the first `BatchStageEndEvent`, at **65.79007690374297 ms**, with request0 and batch0. This exactly reproduces the prior first-prefill boundary. It did not run request completion, decode, clean E2E or estimator training. The old current-task caches are used only to attribute the existing P. They are not fresh corrected-profile evidence.

## Actual model selection and values

All costs below are milliseconds per model layer. Physical CSV lines include the header. The parent trace has48 layers; internal prediction-call counts are not operator execution counts and must not be summed as durations.

| Frontier operation / selected model | P ms per layer | Actual return path | Matching filtered CSV physical lines |
| --- | ---: | --- | --- |
| `attn_pre_proj` | 0.137167997658253 | measured exact | linear51 |
| `attn_rope` | 0.032336000353098 | measured exact | linear51 |
| `attn_post_proj` | 0.046784000471234 | measured exact | linear51 |
| `input_layernorm` | 0.022608000785112 | measured exact | linear22 |
| `post_attention_layernorm` | 0.022111999802291 | measured exact | linear22 |
| `attn_prefill` | 0.146383993327618 | measured exact | attention68,82 |
| `attn_kv_cache_save` | 0.016815999522805 | measured exact | attention68,82 |
| `moe_gating_linear__prefill_hot` | 0.013779555550880 | measured exact | MoE237,240,241,247,260,273,274,309,318 |
| `moe_gating_routing_topk__prefill_hot` | 0.092485333896346 | measured exact | same nine MoE rows |
| `moe_shuffling` | 0.052944906490655 | RF once, runtime cache383 times | none |
| `moe_grouped_gemm` | 0.325747836525242 | RF once, runtime cache383 times | none |

`add`, `add_attn_residual`, and `add_ffn_residual` are zero fused aliases, while `moe_gating` repeats the routing-topk label. These do not introduce additional independent compute prediction queries. Embedding and final norm remain coverage questions from the full-op audit; an existing profile/model does not make them present in the first-forward ledger. Communication is owned by the separate communication audit.

The measured-exact branches return at `sklearn_execution_time_predictor.py:4565` or `:4838`; RF fallback returns at `:4886`, and the subsequent on-demand runtime cache returns at `:4859`. Each raw record contains the actual executed source line, method, model type, features and return value. The delegated prefill wrapper is retained in the raw receipt but excluded from the eleven-model table.

## Feature and source contracts

Linear source: `runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/linear_op.csv`. Projection/RoPE use the filteredTP4 slice; both layer norms useTP1. Their selected feature key is `(num_tokens=4096)`. The captured filtered training frames contain41 rows each.

Attention source: the same directory's `attention_combined.csv`. Prefill uses `(kv_cache_size=0, prefill_chunk_size_squared=16777216)`, selecting two rows from81 standard-prefill rows. KV save uses `(batch_size=1, kv_cache_size=0, total_tokens=4096)`, selecting those two rows from2026 rows. These actual filtered matches confirm the prior row arithmetic and do not invoke RF.

MoE source: `supplements/moe-uniform-01/moe.csv`. Both gating models select the `__prefill_hot` pseudo-model, with `(num_tokens=4096)`, from387 prefill-hot rows. All nine matched rows have `gating_runtime_context=prefill_hot` and `routing_runtime_path=uniform_topk`. They span three load-distribution labels and three seeds; gating's model feature schema is only `num_tokens`. Their mean is the actual exact lookup value. Restricting the comparison to the three rows labeled `uniform` would not reproduce the configured gating predictor.

Shuffling and grouped GEMM use the same fourteen-feature runtime key:

| Feature | Actual value |
| --- | ---: |
| `num_experts_per_device` | 16 |
| `router_topk` | 8 |
| `hidden_dim` | 2048 |
| `expert_hidden_dim` | 768 |
| `total_routed_tokens` | 4096 |
| `tokens_per_expert_avg` | 256 |
| `tokens_to_experts_ratio` | 256 |
| `model_expansion_ratio` | 0.375 |
| `load_imbalance_cv` | 0 |
| `max_load_ratio` | 1 |
| `min_load_ratio` | 1 |
| `expert_utilization` | 1 |
| `load_entropy` | 4.000000000017834 |
| `load_gini_coefficient` | 0 |

There are no matching filtered rows for this key. In particular, nominal4096 with sampled local totals4081/4058/4073 is not this key. Exact-vector supplements should use the existing feature producer, including its entropy convention; writing an idealized literal4 would create a different exact key.

## Interpretation and limits

The broad claim that the first-forward discrepancy is caused by RF lacking4096 samples is unsupported. QKV, attention, KV save and gating already reproduce measured anchors. Their P/S/V investigation must examine measured stability, execution context, kernel/scope mapping and implementation. More identical-size rows alone would not remove an RF error in these nine operators, because RF did not produce their current costs.

The shuffling/GEMM feature miss is proven and is a concrete target for exact-layout profiling. It is not yet proof that RF explains their observed P/V discrepancy. The prior GEMM profiler implementation omitted gated SiLU and final reduction; the repaired implementation changes the measured operation. Report old-implementation P/S attribution separately from corrected-implementation measurements rather than calling that entire change an ML error.

The diagnostic captured2592 query returns and14 unique query records. Attention queries can be evaluated by more than one accounting path, and all eight EP lanes query the MoE models across48 layers. These counts are execution evidence for branch selection, not a duration aggregation rule. The per-layer P table, multiplied once by48, agrees with the previously rounded operation totals within the documented logging precision.

## Artifacts and verification

- `query_receipt.json`: raw executed return branches, normalized feature tuples, matching filtered rows, full input settings, Python version, branch HEAD and stop boundary.
- `validation.json`: all eleven P-to-log comparisons with predicted, previous logged value, absolute and relative differences; nine exact-row means and two actual RF branches checked.
- `execution.log`: bounded diagnostic output.
- `../../test_report_2026-09-08_d019_predictor_query.md`: exact execution and validation commands, observed result and initial serialization-precision failure.

Pending numerical work: matched S/V measurement and scoped correction, followed by fresh CUDA-span and clean E2E validation. No additional predictor-production change is proposed by this audit.
