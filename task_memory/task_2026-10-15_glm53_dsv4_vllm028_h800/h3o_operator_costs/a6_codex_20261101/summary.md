# V4 attention A5/A6 held-out result

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Generated the completed five-fold evaluation summary. |

7105 layer rows in 1421 complete workload units; K=5, seed=1234.

Kernel time gates: median absolute relative error <= 5%, p90 <= 10% (design A6). PASS applies to evaluable rows; NOT_EVALUABLE rows cannot establish accuracy.

| Scope | Held out | Evaluable | Joint rejected | Old rejected | Median | p90 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dsv4_fused_qkv_projection | 7105 | 6995 | 110 | 35 | 1.340% | 5.516% | PASS |
| dsv4_fused_qkv_rmsnorm | 7105 | 6995 | 110 | 35 | 3.570% | 8.722% | PASS |
| dsv4_q_projection | 7105 | 6995 | 110 | 35 | 1.696% | 10.225% | FAIL |
| dsv4_query_padding | 0 | 0 | 0 | 0 | n/a | n/a | LEGALLY_ZERO |
| dsv4_attention_output_zero | 0 | 0 | 0 | 0 | n/a | n/a | LEGALLY_ZERO |
| dsv4_output_inv_rope_quant | 7105 | 6995 | 110 | 35 | 3.168% | 9.750% | PASS |
| dsv4_grouped_output_projection | 7105 | 6995 | 110 | 35 | 0.677% | 1.770% | PASS |
| dsv4_output_projection | 7105 | 6995 | 110 | 35 | 1.032% | 2.777% | PASS |
| dsv4_compressor_projection | 5684 | 5596 | 88 | 28 | 1.232% | 3.334% | PASS |
| dsv4_index_weights_projection | 2842 | 2798 | 44 | 14 | 3.006% | 7.948% | PASS |
| dsv4_index_compressor_projection | 2842 | 2798 | 44 | 14 | 1.412% | 4.579% | PASS |
| dsv4_index_query_projection | 2840 | 2798 | 42 | 12 | 2.452% | 5.768% | PASS |
| dsv4_index_fused_query_prepare | 2840 | 2798 | 42 | 12 | 4.647% | 12.267% | FAIL |
| dsv4_index_short_context_fill | 2 | 0 | 2 | 2 | n/a | n/a | NOT_EVALUABLE |
| dsv4_fused_q_kv_prepare | 7105 | 6985 | 120 | 35 | 2.793% | 7.147% | PASS |
| dsv4_compressor_save_partial_states | 5684 | 5588 | 96 | 28 | 4.004% | 9.483% | PASS |
| dsv4_index_compressor_save_partial_states | 2842 | 2794 | 48 | 14 | 3.713% | 9.471% | PASS |
| dsv4_compressor_fused_compress_store | 5684 | 5576 | 108 | 28 | 3.239% | 13.992% | FAIL |
| dsv4_index_compressor_fused_compress_store | 2842 | 2794 | 48 | 14 | 2.933% | 17.506% | FAIL |
| dsv4_index_scores | 2838 | 2758 | 80 | 12 | 1.847% | 5.449% | PASS |
| dsv4_index_topk | 2840 | 2758 | 82 | 12 | 2.135% | 5.813% | PASS |
| dsv4_sparse_attention | 7105 | 6971 | 134 | 35 | 0.743% | 2.114% | PASS |

Legally zero: 17060 absent-scope contributions checked; 0 runtime selection mismatches (PASS).
Shared admission: 94982 predictions; 0 value changes.

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
