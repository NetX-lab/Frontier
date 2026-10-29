# V4 predictor families read by P2 PD

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Lead: replaced the blocked collective part with the held-out result under D-A6-COLLECTIVE-UNIT (requirements item 36). |
| 2026-10-29 | Created by B19 (Codex `70321f85`). |

PD uses `Simulator` → `ExecutionTimePredictionModelManager` → `train_dsv4_for_manager` (`frontier/simulator.py:155`, `shared_prediction_model_manager.py:432`, `dsv4_training.py:48`). The V4 registration returns before the generic attention, MoE and CPU-overhead trainers.

| Family | Runtime admission and prediction | Evidence | A6 status |
| --- | --- | --- | --- |
| Standalone operators | Exact row-axis/TP lookup; missing time/count coordinate raises `DeepSeekV4CoverageError` | `dsv4_operator_predictor.py:309,353` | Exact only; no held-out estimator run required |
| MoE gating | Exact SP-local rows | `dsv4_operator_predictor.py:356` | Exact only |
| Routed MoE | Exact EP/population/rank-class tuple, with balanced-lane equality checked | `dsv4_operator_predictor.py:359-374` | Exact only |
| Runner input preparation | Exact active geometry; absent coordinates must stay absent | `dsv4_operator_predictor.py:318-335`, `dsv4_runner.py:59` | Exact only |
| Sampling | Exact logits-processing branch and batch rows | `dsv4_operator_predictor.py:318`, `dsv4_runner.py:53` | Exact only |
| Block zeroing/writes | Exact block/write geometry, with exact zero for no work and checked new-row entry lengths | `dsv4_operator_predictor.py:318-351` | Exact only or no-op |
| Attention signed gaps | Same key, features and joint admission as kernel time; exact lookup then fitted interpolation | `dsv4_predictor.py:259-291,314-357` | Included in result.json, absolute microseconds and combined relative error; not gated |
| Non-attention kernel gaps | One measured segment/TP constant multiplied by the exact activity count returned with the operator; collectives use registry counts | `kernel_gap.py:52-68`, `dsv4_runtime.py:31-40`, `deepseek_v4.py:210-212,365-372` | No interpolating estimator |
| Unequal EP combine | Exact ordered SP vector and payload bytes; missing rows raise `MeasuredCCOutOfRangeError` | `measured_cc_backend.py:253-259,271-280` | Exact only |
| Uniform measured collectives | Exact pooled median at measured bytes; log-log interpolation between measured sizes; typed error outside range | `measured_cc_backend.py:77-95,250-263` | One payload point left out at a time (D-A6-COLLECTIVE-UNIT): 7 of 8 P2 curves FAIL, see below |

Standalone operators: dsv4_embedding, dsv4_final_norm, dsv4_lm_head, dsv4_mhc_attn_post_ffn_pre, dsv4_mhc_ffn_post_attn_pre, dsv4_mhc_final_post, dsv4_mhc_first_pre, dsv4_mhc_head, share_expert_act, share_expert_down_proj, share_expert_up_proj.

Collective operators: dsv4_sp_attention_allgather, dsv4_sp_attention_reduce_scatter, dsv4_sp_final_mhc_allgather, dsv4_ep_dispatch_allgather, dsv4_ep_output_reduce_scatter, dsv4_embedding_allreduce, dsv4_logits_gather.

Attention scopes whose sole feature is `padded_rows` use exact lookup at every admitted coordinate. Other attention feature families can interpolate within the measured row group. This classification does not permit interpolation at a new row count.

Row-only attention scopes: dsv4_fused_qkv_projection, dsv4_fused_qkv_rmsnorm, dsv4_q_projection, dsv4_query_padding, dsv4_attention_output_zero, dsv4_output_inv_rope_quant, dsv4_grouped_output_projection, dsv4_output_projection, dsv4_compressor_projection, dsv4_index_weights_projection, dsv4_index_compressor_projection, dsv4_index_query_projection, dsv4_index_fused_query_prepare, dsv4_index_short_context_fill.

Attention scopes with additional features: dsv4_fused_q_kv_prepare, dsv4_compressor_save_partial_states, dsv4_index_compressor_save_partial_states, dsv4_compressor_fused_compress_store, dsv4_index_compressor_fused_compress_store, dsv4_index_scores, dsv4_index_topk, dsv4_sparse_attention.

Attention kernel-count regressors are also fitted by the paired-count trainer (`kernel_gap.py:118-147`), but the V4 attention runtime consumes `scope_times` costs and signed gaps (`deepseek_v4.py:105-116`). It does not query those count regressors; `_operator_gaps` skips scopes with their own signed gap (`dsv4_runtime.py:31-38`). Non-attention counts come from the exact operator lookup or the collective registry.

## Uniform collective held-out

B19 stopped this part: the collective table (`data/profiling/network/h200_dgx/summary.csv`, SHA-256 `f4d52911770cc39939dddc0fa98ce30b530b1cd60ea2bd94070d293e200ca559`) has payload, group and SP geometry, but none of the seven columns of the attention unit, and the runtime fits log-log curves instead of the PD sklearn trainer. The lead set the unit (requirements item 36, D-A6-COLLECTIVE-UNIT):

- A curve is one uniform measured collective key `(op, group, group_size)` as `MeasuredCCBackend` loads it. A unit is one measured payload point of the curve with all its samples, pooled over modes and repetitions as the runtime pools them.
- Each interior point is left out in turn and predicted by log-log interpolation from its two neighbours. This is the existing `_Curve.describe` leave-one-out (`measured_cc_backend.py:97`); the tool reads it through `describe_curves()` and writes no second evaluator. End points are not left out, because the runtime raises outside the measured range instead of extrapolating.
- The A6 thresholds (median at or below 5%, p90 at or below 10%) apply per curve. A left-out point is interpolated across two measured intervals where the runtime interpolates across one, so the errors overstate the runtime error.
- The backend is built as `Cluster._initialize_cc_backend` builds it for the P2 DECODE cluster.
- The backend's other diagnostic, `describe_holdouts` (`measured_cc_backend.py:228`), compares single-active and unequal-token rows with the uniform curves. Those rows are a different workload from the curve, so it does not measure the curve's own held-out error and is not used here.

| Curve | Points | Held out | Byte range | Median | p90 | Max | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| all_gatherv, ep, 16 | 5 | 3 | 8,264-6,346,752 | 11.404% | 12.091% | 12.263% | FAIL |
| all_gatherv, ep, 32 | 5 | 3 | 8,264-6,346,752 | 9.012% | 11.218% | 11.769% | FAIL |
| all_reduce, tp, 8 | 8 | 6 | 8,192-50,331,648 | 8.828% | 18.793% | 20.233% | FAIL |
| logits_all_gather, tp, 8 | 24 | 22 | 32,320-775,680 | 5.396% | 9.022% | 12.869% | FAIL |
| reduce_scatterv, ep, 16 | 5 | 3 | 131,072-100,663,296 | 0.640% | 6.920% | 8.490% | PASS |
| reduce_scatterv, ep, 32 | 5 | 3 | 262,144-201,326,592 | 30.644% | 39.754% | 42.032% | FAIL |
| sp_all_gather, tp, 8 | 9 | 7 | 8,192-25,165,824 | 3.527% | 22.978% | 26.420% | FAIL |
| sp_reduce_scatter, tp, 8 | 8 | 6 | 8,192-50,331,648 | 5.251% | 8.929% | 9.644% | FAIL |

Reading: the curves with five points span three orders of magnitude of payload, so one left-out point leaves an interval of about 28 times in bytes (each measured interval is about 5.3 times). Whether a run interpolates at all depends on the payloads it requests; B21 records, for each measured collective call of the A10 runs, whether it hits a measured point or interpolates and between which points. The backend also reports a non-monotone `sp_reduce_scatter` curve between 16,384 and 65,536 bytes (10.63, 8.89, 8.10 µs).
