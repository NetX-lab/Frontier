## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Parsed all three fresh first-forward compute groups; first-forward before/after reference bracket validated. |

# D019 fresh compute reference — provisional results

Collection passes the first-request scope/identity checks. Numerical calibration remains INCOMPLETE. The current Frontier column is the existing fresh-task baseline P, not a repaired or retrained result. Groups are independent runs; their values are not added into an invented full-forward total. Communication retains the user-approved ideal abstraction; optional naive-protocol modeling is deferred.

## Boundary and perturbation

Before reference: first request0 batch4157 DP0 TP0 =79.307357788 ms; peer physical counts `[4096,1]`. Other TP ranks=79.550849915,79.559837341,79.498626709 ms. New attention batch4666 TP0=92.264129639 ms: +12.956771851 ms (+16.337%). New MoE batch4602 TP0=92.876190186 ms: +13.568832397 ms (+17.109%). Both expose material probe/run sensitivity despite bounded selection. The after reference was subsequently collected below; do not fit a production timing correction directly from these probes alone.

The earlier separate B reference was80.335617065 ms. The new before reference is1.028259277 ms lower (-1.280%), substantially smaller than the12.96/13.57ms probe increases. This supports retaining a probe perturbation concern; it does not isolate its cause.

## Complete current computation mapping — all three groups collected

All durations are48-layer totals, except model-level embedding/final normalization. Signed gap is `(P/V-1)*100`; missing values remain missing. TP0 is a reported rank, not a chosen pure-kernel estimator. The CSV retains signed/absolute gaps, rank ranges, scope provenance and semantic qualifications.

| Frontier operation | Baseline P ms | vLLM scope | TP0 V ms | TP min–max ms | Signed gap | Status |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| input_layernorm | 1.085184 | input_layernorm | 0.969440 | 0.968192–0.976512 | +11.939% | DIAGNOSTIC_ONLY |
| attn_pre_proj | 6.584064 | attn_pre_proj | 3.870016 | 3.773472–3.870016 | +70.130% | DIAGNOSTIC_ONLY |
| attn_rope | 1.552128 | attn_rope | 0.939648 | 0.939648–0.953536 | +65.182% | DIAGNOSTIC_ONLY |
| attn_kv_cache_save | 0.807168 | attn_kv_cache_save | 0.565504 | 0.564128–0.645856 | +42.734% | DIAGNOSTIC_ONLY |
| attn_prefill | 7.026432 | attn_prefill | 5.734944 | 5.636000–8.420704 | +22.520% | DIAGNOSTIC_ONLY |
| attn_post_proj | 2.245632 | row_parallel_gemm | 1.454816 | 1.440832–1.546720 | +54.358% | DIAGNOSTIC_ONLY |
| post_attention_layernorm | 1.061376 | post_attention_layernorm | 0.952000 | 0.946624–0.960672 | +11.489% | DIAGNOSTIC_ONLY |
| moe_gating_linear | 0.661440 | moe_gating_linear | 0.535040 | 0.535040–0.624512 | +23.624% | DIAGNOSTIC_ONLY |
| moe_gating_routing_topk | 4.439280 | moe_gating_routing_topk | 4.364160 | 4.304224–14.499360 | +1.721% | DIAGNOSTIC_ONLY |
| moe_shuffling | 2.541360 | moe_shuffling | 1.333568 | 1.333568–3.605472 | +90.568% | DIAGNOSTIC_ONLY |
| moe_grouped_gemm | 15.635904 | moe_grouped_gemm | 16.927232 | 16.918816–21.803104 | -7.629% | DIAGNOSTIC_ONLY |
| embedding | missing | embedding_compute | 0.246208 | 0.242496–0.262976 | missing | MISSING_FRONTIER_ACCOUNTING |
| attention_output_initialization | missing | attn_output_init | 0.374848 | 0.374848–0.765920 | missing | MISSING_FRONTIER_ACCOUNTING |
| moe_output_reduction | missing | moe_sum | 2.790784 | 2.790784–2.859296 | missing | MISSING_FRONTIER_ACCOUNTING |
| final_layernorm | missing | final_layernorm | 0.020064 | 0.020000–0.020160 | missing | MISSING_FRONTIER_ACCOUNTING |
| add | 0.000000 | included in norm parents | missing | nonadditive | missing | NONADDITIVE |
| add_attn_residual | 0.000000 | included in norm parents | missing | nonadditive | missing | NONADDITIVE |
| add_ffn_residual | 0.000000 | included in norm parents | missing | nonadditive | missing | NONADDITIVE |
| moe_gating | 4.439280 | moe_gating_routing_topk alias | missing | nonadditive | missing | NONADDITIVE |

## Rank effects and limits

Attention TP2 prefill totals8.420704 ms versus5.636000–5.734944 on its three peers. This is distributed across many layers: its five longest layer scopes are0.225888/0.188992/0.184864/0.184256/0.176256ms; no one giant event explains the difference. Event-only rows do not establish whether these differences arise from device work, host submission gaps or another runtime effect.

MoE TP0/1/2/3 grouped-GEMM parent plus sibling sum =19.718016/22.636704/19.732512/24.662400ms. The parent itself=16.927232/19.797248/16.918816/21.803104ms; sum=2.790784/2.839456/2.813696/2.859296ms. Topk totals4.364160/7.457920/4.304224/14.499360ms show a particularly large rank-dependent effect. Neither averaging all ranks nor taking a minimum proves clean operation cost.

The baseline MoE profile is semantically incomplete: W1/copy/W2 omits gated SiLU and expert sum; its routing producer also used sampled counts. Therefore a baseline-P versus new-V gap is descriptive, not proof of RF interpolation error. Compare repaired S at the same global4097/expert layout before assigning causality. New V grouped-GEMM does not include sum: compare complete repaired profiles to same-rank/same-run parent+sum.

## Execution and next steps

Exact parser commands executed from active Frontier root:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_compute.py --run attention=task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/compute_attention/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/attention
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_compute.py --run moe=task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/compute_moe/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/moe
```

CPU Python3.13.13/conda dev-vidur-v03-hopper-e2e; GPU mode manifest pins Python3.10.16 and vLLM4bc1bc026c91dff78bd7cf5ba6411f14d15e043d, eager, uniform routing, prefixOFF, FLASHINFER, per_scope/default CUDA events, selected first formal local batch only. Attention and before-reference client files have400rows. MoE had300warmup completion rows at first extraction; its selected first-forward scope records were complete, and full-run completion remains a separate check.

Pending: communication-calibrated Frontier P and causal closure of remaining profiling/runtime-context gaps. Detail first-forward scope collection is complete; full-run completion remains a separate check. No new design issue is introduced. Existing limitations: probe perturbation, dummy participant timings absent, ideal communication abstraction retained, CUDA/official-TTFT gates still open.

## P / refreshed S / new V separation

Fresh S is the mean of three current-task profiling medians, scaled by48layers. It has matching local shape and source-profile identity; its standalone timing context differs from in-engine V. These comparisons cannot attribute the remaining S-V gap to a specific timer mechanism without an A/B. CPU profiling overhead is not added to any value.

| Operator | P ms | Fresh S ms | New V TP0 ms | P−S ms | S−V ms | S−V percent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attn_pre_proj | 6.584064 | 7.013888 | 3.870016 | -0.429824 | 3.143872 | 81.236669 |
| attn_rope | 1.552128 | 1.596928 | 0.939648 | -0.044800 | 0.657280 | 69.949597 |
| attn_post_proj | 2.245632 | 2.285312 | 1.454816 | -0.039680 | 0.830496 | 57.085980 |
| input_layernorm | 1.085184 | 1.116672 | 0.969440 | -0.031488 | 0.147232 | 15.187325 |
| post_attention_layernorm | 1.061376 | 1.073152 | 0.952000 | -0.011776 | 0.121152 | 12.726050 |
| attn_prefill | 7.026432 | 6.955008 | 5.734944 | 0.071424 | 1.220064 | 21.274211 |
| attn_kv_cache_save | 0.807168 | 0.686592 | 0.565504 | 0.120576 | 0.121088 | 21.412403 |

The seven existing exact-feature anchors reproduce fresh profiling relatively closely; their aggregate change is+0.365569ms across48layers. QKV/RoPE/output-projection differences remain between S and V, even with same F.linear primitive and inference_mode enabled. Predictor lane identified timer end-event construction and neighboring synthetic operations as candidates, not proven causes. KV-save S medians have43.4% relative spread; its small aggregate mean must not be treated as a precise constant. Source and per-sample receipts are retained in analysis/d019-linear-attention-comparison.json.

## Detail group and newly quantified coverage

Detail batch has290valid scopes per real TP worker. Its TP0 forward event is96.684608459ms, +17.377250671ms (+21.911%) versus before reference. Counts verify one embedding, one final norm,48output initializations,48input norms,48post norms and48each W1/activation/W2.

| vLLM component | TP0 ms | TP1 ms | TP2 ms | TP3 ms | Frontier mapping |
| --- | ---: | ---: | ---: | ---: | --- |
| embedding_compute | 0.246208 | 0.242496 | 0.260000 | 0.262976 | missing model-level term |
| input_layernorm | 0.969440 | 0.968192 | 0.976512 | 0.971648 | input_layernorm |
| post_attention_layernorm | 0.952000 | 0.951328 | 0.960672 | 0.946624 | post_attention_layernorm |
| attn_output_init | 0.374848 | 0.385408 | 0.765920 | 0.384672 | missing per-layer initialization |
| moe_grouped_gemm_w1 | 6.955360 | 6.901216 | 11.259584 | 7.017312 | inside grouped-GEMM parent |
| moe_activation | 6.548864 | 6.555808 | 6.560480 | 6.548640 | missing gated activation in old profile |
| moe_grouped_gemm_w2 | 4.046496 | 4.001440 | 4.046976 | 4.037760 | inside grouped-GEMM parent |
| final_layernorm | 0.020064 | 0.020160 | 0.020000 | 0.020000 | missing model-level term |

The6.548640–6.560480ms activation is consistent across TP ranks; the previous MoE group separately measured2.790784–2.859296ms expert-result sum. These establish meaningful omitted work, but they are not an additive estimate of the net repair: the old profiler also executes a replacement contiguous copy and differs in routing/layout. W1 on TP2 is11.259584ms against6.901216–7.017312ms on peers; event-only evidence cannot identify its extra interval as GPU computation.

Embedding's new event scope is0.242496–0.262976ms, much longer than the earlier kernel-activity inventory's0.026112ms. The quantities are from different timing families/runs, so this is an instrumentation/context concern, not a numerically established CPU subtraction. Attention output initialization is0.374848–0.765920ms across48layers, verifying the previously unscoped zeros/fill boundary.

For exact4097/EP0 repaired MoE profiling, S CSV full=.463360ms/layer and hot full=.399632ms/layer; V TP0 parent+sum=.410792ms/layer. CSV S exceeds V by12.797%, while hot S is2.717% lower. Actual assignments4104/padded4608 are established in analysis/d019-moe-normalized/validation.json. This supports the repaired compute implementation's relevance and preserves the remaining measurement-context distinction. Current Frontier4096 query/data are not silently substituted for that4097shape.

Detail parser command (executed, PASS_COLLECTION_DIAGNOSTIC_ONLY):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_compute.py --run detail=task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-d019-compute-01/runtime/compute_detail/operators --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-compute-results/detail
```

## MoE-repaired Frontier P checkpoint — communication not yet corrected

Bounded first-forward predictor receipt `/data/ycfeng/tmp/issue26-d019-moe-integration-query-02/query_receipt.json` reports endpoint72.327535217ms at commit882ac05b82458a55a7a6bab90e0cffe4039c6def. This is a scoped MoE repair checkpoint. The unchanged17.899440ms attention-AR prediction is known to overcount the observed implementation and remains to be calibrated. A total now within10% of the before reference does not close CUDA-span correction by cancellation. F still models global4096 while V executes4097including the peer's token.

| Frontier operation | Baseline P ms | After-MoE P ms | Matching new V TP0 ms | After-MoE gap | Revised boundary |
| --- | ---: | ---: | ---: | ---: | --- |
| input_layernorm | 1.085184 | 1.085184 | 0.969440 | +11.939% | input_layernorm |
| attn_pre_proj | 6.584064 | 6.584064 | 3.870016 | +70.130% | attn_pre_proj |
| attn_rope | 1.552128 | 1.552128 | 0.939648 | +65.182% | attn_rope |
| attn_kv_cache_save | 0.807168 | 0.807168 | 0.565504 | +42.734% | attn_kv_cache_save |
| attn_prefill | 7.026432 | 7.026432 | 5.734944 | +22.520% | attn_prefill |
| attn_post_proj | 2.245632 | 2.245632 | 1.454816 | +54.358% | row_parallel_gemm |
| post_attention_layernorm | 1.061376 | 1.061376 | 0.952000 | +11.489% | post_attention_layernorm |
| moe_gating_linear | 0.661440 | 0.661419 | 0.535040 | +23.620% | moe_gating_linear |
| moe_gating_routing_topk | 4.439280 | 4.439296 | 4.364160 | +1.722% | moe_gating_routing_topk |
| moe_shuffling | 2.541360 | 2.541356 | 1.333568 | +90.568% | moe_shuffling |
| moe_grouped_gemm | 15.635904 | 22.173354 | 19.718016 | +12.452% | moe_grouped_gemm + moe_sum (same-run siblings) |
| embedding | missing | missing | 0.246208 | nonadditive/missing | embedding_compute |
| attention_output_initialization | missing | missing | 0.374848 | nonadditive/missing | attn_output_init |
| moe_output_reduction | missing | missing | missing | nonadditive/missing | moe_sum; same value already included in repaired parent comparison |
| final_layernorm | missing | missing | 0.020064 | nonadditive/missing | final_layernorm |
| add | 0.000000 | 0.000000 | missing | nonadditive/missing | included in norm parents |
| add_attn_residual | 0.000000 | 0.000000 | missing | nonadditive/missing | included in norm parents |
| add_ffn_residual | 0.000000 | 0.000000 | missing | nonadditive/missing | included in norm parents |
| moe_gating | 4.439280 | 4.439296 | missing | nonadditive/missing | moe_gating_routing_topk alias |

The repaired grouped-GEMM prediction22.173354467ms includes activation and sum and compares to V19.718015995ms (+12.451%). It replaces the old incomplete15.635904ms prediction; the old and new rows deliberately use different V boundaries. Expert sum is therefore covered by the repaired profile and is not added again as a missing term. Embedding, output initialization and final norm remain named missing model terms. New arithmetic precision differences in the unchanged inputs are negligible; the primary endpoint increase is6.537458313ms from the corrected expert path.

## Completed first-forward reference bracket

Both unselected first-forward records are validated across fourTP ranks: request0 only,4096prefill/0decode,physical[4096,1],positive CUDA-event span and op_profile_selected=false. The after mode has begun its formal run; full400completion remains separate.

| Reference | TP0 ms | TP1 ms | TP2 ms | TP3 ms |
| --- | ---: | ---: | ---: | ---: |
| batch_before | 79.307358 | 79.550850 | 79.559837 | 79.498627 |
| batch_after | 78.118782 | 78.112770 | 78.021698 | 78.157280 |

| Probe group | TP0 event ms | Increase vs before | Increase vs after |
| --- | ---: | ---: | ---: |
| attention | 92.264130 | 12.956772ms / 16.337% | 14.145348ms / 18.107% |
| moe | 92.876190 | 13.568832ms / 17.109% | 14.757408ms / 18.891% |
| detail | 96.684608 | 17.377251ms / 21.911% | 18.565826ms / 23.766% |

Before→after TP0 drift is−1.188576ms (−1.499%); all three probes lie well above both bookends. The48-layer MoE repair endpoint72.327535ms differs from the after78.118782ms by−5.791247ms (−7.413%). This does not establish correction completion because the unchanged attention-AR overprediction and remaining compute/context gaps can cancel. Optional naive-protocol modeling stays deferred under the approved ideal abstraction.

## Full five-mode completion and identity validation

All five completed modes pass the existing full-case batch identity analyzer, using400client records per mode. Independently checked three warmup rounds of100requests each,100formal requests,400unique response IDs,4096prompt/1024observed output tokens per request,valid first-token/completion ordering and eightworker identities. Each formal request has exactly one4096prefill on each TP rank of exactly one DP lane.

| Mode | Completed client records | Warmups r0/r1/r2 | Formal requests | Workers | Selected real op batches |
| --- | ---: | --- | ---: | ---: | ---: |
| batch_before | 400 | 100/100/100 | 100 | 8 | 0 |
| compute_attention | 400 | 100/100/100 | 100 | 8 | 8 |
| compute_moe | 400 | 100/100/100 | 100 | 8 | 8 |
| compute_detail | 400 | 100/100/100 | 100 | 8 | 8 |
| batch_after | 400 | 100/100/100 | 100 | 8 | 0 |

Total2000completed requests =1500warmup+500formal across five independently restarted diagnostic modes. This supersedes earlier running-mode caveats in this chronological report. `full-suite/summary.json` and the five per-mode reports retain source paths and checks. No new GPU collection was performed for this closeout. Root observed job inactive/ExecMainStatus0; this lane establishes completion from the resulting request/batch artifacts.

## Integrated existing communication correction

Root executed the reviewed existing AR parameter with the identical fresh compute predictor cache. Actual first-forward59.190354384ms; every compute query in the above MoE-repaired table remains exactly unchanged. AR48-layer17.899443200→4.762262367ms accounts for the entire DES delta. Compared with batch-only before79.307357788 and after78.118782043ms, signed errors are−25.365872682% and−24.230315891%; the CUDA gate remainsFAIL. See `../d019-first-forward-integrated/report.md` and its actual query/validation receipts. Ideal EP abstraction and CPU-disabled workflow remain in effect.
