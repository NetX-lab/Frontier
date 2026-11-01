# V4 attention A5/A6 held-out result

4945 layer rows in 989 complete workload units; K=5, seed=1234.

Kernel time gates: median absolute relative error <= 5%, p90 <= 10% (design A6). PASS applies to evaluable rows; NOT_EVALUABLE rows cannot establish accuracy.

| Scope | Held out | Evaluable | Joint rejected | Old rejected | Median | p90 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dsv4_fused_qkv_projection | 4945 | 4825 | 120 | 35 | 0.799% | 2.775% | PASS |
| dsv4_fused_qkv_rmsnorm | 4945 | 4825 | 120 | 35 | 2.419% | 6.440% | PASS |
| dsv4_q_projection | 4945 | 4825 | 120 | 35 | 0.977% | 6.340% | PASS |
| dsv4_query_padding | 0 | 0 | 0 | 0 | n/a | n/a | LEGALLY_ZERO |
| dsv4_attention_output_zero | 0 | 0 | 0 | 0 | n/a | n/a | LEGALLY_ZERO |
| dsv4_output_inv_rope_quant | 4945 | 4825 | 120 | 35 | 2.791% | 8.213% | PASS |
| dsv4_grouped_output_projection | 4945 | 4825 | 120 | 35 | 0.469% | 1.344% | PASS |
| dsv4_output_projection | 4945 | 4825 | 120 | 35 | 0.759% | 2.033% | PASS |
| dsv4_compressor_projection | 3956 | 3860 | 96 | 28 | 0.815% | 2.177% | PASS |
| dsv4_index_weights_projection | 1978 | 1930 | 48 | 14 | 1.327% | 3.738% | PASS |
| dsv4_index_compressor_projection | 1978 | 1930 | 48 | 14 | 0.890% | 2.699% | PASS |
| dsv4_index_query_projection | 1976 | 1930 | 46 | 12 | 1.089% | 3.059% | PASS |
| dsv4_index_fused_query_prepare | 1976 | 1930 | 46 | 12 | 3.083% | 8.435% | PASS |
| dsv4_index_short_context_fill | 2 | 0 | 2 | 2 | n/a | n/a | NOT_EVALUABLE |
| dsv4_fused_q_kv_prepare | 4945 | 4815 | 130 | 35 | 2.186% | 5.559% | PASS |
| dsv4_compressor_save_partial_states | 3956 | 3852 | 104 | 28 | 2.880% | 7.132% | PASS |
| dsv4_index_compressor_save_partial_states | 1978 | 1926 | 52 | 14 | 2.632% | 6.637% | PASS |
| dsv4_compressor_fused_compress_store | 3956 | 3846 | 110 | 28 | 2.032% | 7.297% | PASS |
| dsv4_index_compressor_fused_compress_store | 1978 | 1926 | 52 | 14 | 2.500% | 13.863% | FAIL |
| dsv4_index_scores | 1974 | 1878 | 96 | 16 | 1.269% | 3.604% | PASS |
| dsv4_index_topk | 1976 | 1878 | 98 | 16 | 1.427% | 3.896% | PASS |
| dsv4_sparse_attention | 4945 | 4787 | 158 | 35 | 0.469% | 1.411% | PASS |

Legally zero: 11876 absent-scope contributions checked; 0 runtime selection mismatches (PASS).
Shared admission: 65438 predictions; 0 value changes.

Uniform measured collectives, one payload point left out at a time (D-A6-COLLECTIVE-UNIT):

| Curve | Points | Held out | Median | p90 | Max | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ('all_gatherv', 'ep', 16) | 5 | 3 | 12.263% | 12.835% | 12.978% | FAIL |
| ('all_gatherv', 'ep', 32) | 5 | 3 | 9.012% | 11.010% | 11.509% | FAIL |
| ('all_reduce', 'tp', 8) | 8 | 6 | 8.828% | 18.793% | 20.233% | FAIL |
| ('logits_all_gather', 'tp', 8) | 24 | 22 | 5.396% | 9.022% | 12.869% | FAIL |
| ('reduce_scatterv', 'ep', 16) | 5 | 3 | 2.140% | 7.220% | 8.490% | PASS |
| ('reduce_scatterv', 'ep', 32) | 5 | 3 | 30.644% | 39.670% | 41.927% | FAIL |
| ('sp_all_gather', 'tp', 8) | 9 | 7 | 3.527% | 22.978% | 26.420% | FAIL |
| ('sp_reduce_scatter', 'tp', 8) | 8 | 6 | 5.251% | 8.929% | 9.644% | FAIL |

All model-key statistics, signed-gap errors, combined-time errors, rejection columns/examples, fold assignments, data hashes and trainer fields are in result.json. No timestamps are emitted.
