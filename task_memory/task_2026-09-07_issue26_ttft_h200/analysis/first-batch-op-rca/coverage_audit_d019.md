## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Read-only D019 audit: enumerate all first-forward prediction names and recorded comp/memory kernel families, separate communication, and propose unexecuted coverage work. |

# D019: first-forward operator coverage audit and proposed completion plan

The previous operator table is **partial**, not a completed all-operator comparison. This audit enumerates every current Frontier first-forward prediction name and all 38 `(scope, kernel-name)` families in the complete DP0/TP0 first-forward trace. It supplies existing numbers while explicitly preserving missing measurements and mappings. No GPU execution, profiling, production edit or predictor correction is authorized by this document. The proposed work below awaits YC review.

## Fixed identities and duration families

- **F**: fresh Frontier first req0, 4096 prefill, 48 layers, endpoint 65.790076904 ms; `frontier_summary.json`, first-prefill source trace and command in this directory. Predictions come from CUDA-event microprofiles and collective_sim/htsim + nvlink_analytic, not CUPTI kernel-only profiles.
- **E**: all-op CUDA-event run `../h200-rca-01/runtime/operators`, first DP0/TP0 batch4706, req `cmpl-pf4096_dc1024:0-0`, 4096 tokens; batch interval 116.210144 ms. Default per-scope event durations include submission gaps and rank waits. Source `operators/summary.json` plus normalized raw rows.
- **K**: record-function run `../h200-rca-01/runtime/kernels`, first DP0/TP0 batch4734, same request/token membership; 153.403107 ms batch interval. CUDA kernel/device durations from `kernels/first_batch_kernel_families.json`; every one of its 3077 kernel launches has a matching device activity. DP0 four-rank first-batch coverage is established; later trace coverage fails. This is not an unperturbed production run.
- **C**: communication-only event run `../h200-rca-comm-01/runtime/operators`, first DP0/TP0 batch4250, same request/tokens; 86.412033 ms batch interval. 48 attention AR + 48 extra MoE TP AR + one embedding AR; metadata/identity/sequence/positive checks passed on 24 selected rank-batches.
- **B**: independent batch-only run, first DP0/TP0 forward 80.335617 ms. It supplies no fine operator rows. Clean official TTFT 121.484280 ms is a separate run and boundary; never subtract E/K/C operator values from it to infer CPU cost.

Signed gap is **100 × (F − vLLM) / vLLM**. A positive number means Frontier is longer. E columns are CUDA-event diagnostic scope comparisons. K active durations are a separate coverage inventory and are never compared with F CUDA-event predictions to compute a gap. Missing independent prediction or scope stays `missing`, never zero. No E minus C or E minus K calculation is used to manufacture projection time. All times below are ms summed over the specified 48 layers on one rank; parallel ranks are never summed.

**Shape A**: local request batch 1×4096, zero prior KV, block16, BF16, H2048, attention TP4; Q8/KV1 heads per TP, head128. Projection input [4096,2048], QKV local width1280; output projection [4096,1024]→[4096,2048]. Norm/residual hidden [4096,2048]. Final norm and embedding occur once, unlike layer ops.

**Shape L**: router GEMM before naive DP dispatch, local [4096,2048]→[4096,128], replicated within TP. **Shape M**: post-dispatch global MoE inputs: vLLM source predicts M4097 (4096 real + one eager dummy token), while Frontier aggregate filters idle and has M4096. Both global experts128/topk8, EP8/local experts16, expert hidden768. First real request membership is observed; exact raw global routing vector and padding are not captured in these metadata-disabled op logs. Uniform vLLM assignment is deterministic round-robin, while selected MoE GEMM profile rows use sampled uniform load features; M/routing/padding equality is not established.

## All 11 nonzero Frontier computation/memory predictions

Only F and E event-scope numbers are compared here; semantic conflicts remain flagged. K kernel activity is inventoried separately below and supplies no F/K gap. Nested children are not additive to their parents. All active Frontier computation/memory path names, including aliases and fused-zero names, are retained.

| Frontier prediction | F ms | E scope ms | F vs E signed gap | Mapping / shape |
| --- | ---: | ---: | ---: | --- |
| input_layernorm | 1.085184 | 1.266720 | -14.331% | 1:many: first RMSNorm + 47 fused add/norm; A; children K04,K36 |
| attn_pre_proj | 6.584064 | 3.927296 | +67.649% | 1:many: GEMM + 96 copy + 96 Q/K norm kernels; A; children K05,K06,K07 |
| attn_rope | 1.552128 | 1.029024 | +50.835% | 1:1; A; children K08 |
| attn_prefill | 7.026432 | 6.047232 | +16.193% | 1:many: indices + attention + multiply; A; children K11,K12,K13 |
| attn_kv_cache_save | 0.807168 | 0.721312 | +11.903% | 1:1 memory write; A; children K10 |
| attn_post_proj | 2.245632 | missing | missing | 1:1 GEMM; event parent inseparable from TP AR; A; children K14 |
| post_attention_layernorm | 1.061376 | 1.244416 | -14.709% | 1:1 fused add/norm; A; children K16 |
| moe_gating_linear | 0.661440 | 0.723584 | -8.588% | 1:1 GEMM; even moe_gating scope sequence; L; children K17 |
| moe_gating_routing_topk | 4.439280 | 6.306880 | -29.612% | 1:many; odd moe_gating sequence; global M differs; M; children K20,K21,K22,K23,K24,K25,K26 |
| moe_shuffling | 2.541360 | 1.889632 | +34.490% | 1:many; global M/expert map differs; M; children K27,K28,K29,K30 |
| moe_grouped_gemm | 15.635904 | 18.180160 | -13.995% | partial 1:many; profile omits SiLU and uses copy; M; children K31,K32 |
| add | 0.000000 (fused accounting) | missing independent residual scope | missing | Fused into norm parents; E nested add is non-additive. |
| add_attn_residual | 0.000000 (fused accounting) | missing independent residual scope | missing | Fused into norm parents; E nested add is non-additive. |
| add_ffn_residual | 0.000000 (fused accounting) | missing independent residual scope | missing | Fused into norm parents; E nested add is non-additive. |
| moe_gating | 4.439280 (alias only) | 6.306880 (routing scope alias) | missing: duplicate alias | Same routing value already compared above; not a second additive operation. |

`attn_post_proj` has no pure projection E measurement: the 17.231616 ms parent includes attention AR. Its available diagnostic event-level comparison is the **mixed** aggregate F projection+AR=20.145072 ms against E parent=17.231616 ms, gap +16.908%; it is not placed in the pure comp table. K14 supplies actual projection kernel duration only in the separate K coverage inventory, with no cross-run subtraction and no F/K gap.

`moe_grouped_gemm` is not semantically matched: Frontier `_run_fused_moe_iteration` executes W1, a contiguous slice/copy, then W2; vLLM executes W1, gated SiLU, W2. K31 and K32 separate observed GEMMs and activation, but Frontier has no independent predicted W1/W2/copy/SiLU costs. A whole-parent gap is diagnostic only. `moe_sum` is outside this parent on vLLM and absent from Frontier's current profile iteration.

## Complete recorded computation/memory inventory

Every positive-duration first-forward comp/memory kernel family is listed below. Full recorded names, scope names, counts, duration family and source are retained in `coverage_audit_d019_kernel_inventory.csv`. A parent name means the logical operation contains that child, not that an independent child prediction exists. The child-level gap is `missing` unless independent matching costs exist; assigning the whole parent prediction to each child would double-count.

| ID | Actual computation/memory step | Count | K ms | Frontier mapping | Independent child gap |
| --- | --- | ---: | ---: | --- | --- |
| K00 | Embedding input index/mask | 1 | 0.001664 | missing; Frontier prediction missing | missing |
| K01 | Embedding gather | 1 | 0.005632 | missing; Frontier prediction missing | missing |
| K02 | Embedding masked fill | 1 | 0.018816 | missing; Frontier prediction missing | missing |
| K04 | First input RMSNorm | 1 | 0.034464 | input_layernorm; parent-only; child prediction missing | missing |
| K05 | QKV GEMM | 48 | 1.469919 | attn_pre_proj; parent-only; child prediction missing | missing |
| K06 | Q/K contiguous copies | 96 | 0.621920 | attn_pre_proj; parent-only; child prediction missing | missing |
| K07 | Q/K RMSNorm | 96 | 1.359910 | attn_pre_proj; parent-only; child prediction missing | missing |
| K08 | RoPE | 48 | 0.776836 | attn_rope; 1:1 logical operation; timing-family mismatch | missing |
| K09 | Attention external zeros/fill | 48 | 0.181282 | missing; missing; aten::zeros parent observed, source allocation owner unresolved | missing |
| K10 | KV cache write | 48 | 0.367809 | attn_kv_cache_save; 1:1 logical operation; timing-family mismatch | missing |
| K11 | FlashInfer block-to-vector indices | 48 | 0.258246 | attn_prefill; parent-only; child prediction missing | missing |
| K12 | FlashInfer paged prefill | 48 | 4.819984 | attn_prefill; parent-only; child prediction missing | missing |
| K13 | FlashInfer output multiply | 48 | 0.203623 | attn_prefill; parent-only; child prediction missing | missing |
| K14 | Attention output GEMM | 48 | 1.227588 | attn_post_proj; 1:1 logical operation; timing-family mismatch | missing |
| K16 | Post-attention fused add+RMSNorm | 48 | 0.778755 | post_attention_layernorm; 1:1 logical operation; timing-family mismatch | missing |
| K17 | Router GEMM | 48 | 0.336101 | moe_gating_linear; 1:1 logical operation; timing-family mismatch | missing |
| K20 | Uniform route weights fill | 48 | 0.059007 | moe_gating_routing_topk; parent-only; child prediction missing | missing |
| K21 | Uniform route arange | 48 | 0.057853 | moe_gating_routing_topk; parent-only; child prediction missing | missing |
| K22 | Uniform route multiply | 384 | 0.495072 | moe_gating_routing_topk; parent-only; child prediction missing | missing |
| K23 | Uniform route add | 384 | 0.494020 | moe_gating_routing_topk; parent-only; child prediction missing | missing |
| K24 | Uniform route remainder | 384 | 0.766824 | moe_gating_routing_topk; parent-only; child prediction missing | missing |
| K25 | Uniform route index copies | 384 | 1.026315 | moe_gating_routing_topk; parent-only; child prediction missing | missing |
| K26 | Uniform route DtoD | 48 | 0.064639 | moe_gating_routing_topk; parent-only; child prediction missing | missing |
| K27 | MoE block alignment | 48 | 0.541701 | moe_shuffling; parent-only; child prediction missing | missing |
| K28 | MoE expert count/sort | 48 | 0.144835 | moe_shuffling; parent-only; child prediction missing | missing |
| K29 | MoE metadata copy | 48 | 0.112031 | moe_shuffling; parent-only; child prediction missing | missing |
| K30 | MoE expert-map indexing | 48 | 0.239780 | moe_shuffling; parent-only; child prediction missing | missing |
| K31 | MoE W1+W2 GEMM | 96 | 10.270075 | moe_grouped_gemm; partial 1:many; Frontier W1+contiguous-copy+W2 versus runtime W1+SiLU+W2 | missing |
| K32 | Gated SiLU | 48 | 7.067903 | moe_grouped_gemm; missing activation implementation in current Frontier profile | missing |
| K33 | MoE expert-result sum | 48 | 2.709863 | missing; Frontier prediction missing | missing |
| K36 | Later input fused add+RMSNorm | 47 | 0.768258 | input_layernorm; parent-only; child prediction missing | missing |
| K37 | Final fused add+RMSNorm | 1 | 0.015968 | missing; Frontier prediction missing | missing |

K00–K02 map by pinned `vocab_parallel_embedding.py:433–451` to input mask/index transform, embedding lookup, and output masking. K09 has CPU parents `aten::zeros → zero_ → fill_` and 48 observed kernels; the exact allocation source is unresolved and must not be renamed a proven attention-buffer contract. K33 maps to `ops.moe_sum` after the grouped-GEMM scope (`fused_moe.py:1797`). K37 is the final model fused add/RMSNorm. These four logical steps (embedding with three children, zeros/fill, MoE sum, final norm) have no independent Frontier first-forward prediction. SiLU K32 additionally has an implementation mismatch inside an otherwise named parent.

The inventory has 32 comp/memory families and six communication/staging families. All K device activities are accounted for: 3077 kernels + 144 DtoD memcpy activities =3221. No gpu_memset event exists in this first trace. This is exhaustive for the recorded forward, not for the official TTFT path: preprocessing H2D copies, CPU metadata preparation, logits, sampling and output-bookkeeping lie outside the current profiler/model boundary and remain missing. Zero-copy views/reshapes may have CPU work without a device kernel, so no device row does not prove zero host cost.

## Communication table, independent of computation/memory

The dispatch staging copies below belong to the communication protocol; they are disclosed separately so that memory work is not hidden, and not duplicated in the comp table. F is topology-aware modeled communication; E/C are event spans; K is device activity. Each table row labels its own comparison family.

| Frontier term | F ms | vLLM operation / mapping | E ms | C TP0 ms | K TP0 ms | Signed gap and qualification |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| attn_tp_allreduce | 17.899440 | 48 attention TP4 AR; 1:1 logical collective, backend/group algorithm must match | missing | 5.537248 | 29.569227 | F/E missing; F/C +223.255%; K is inventory only, no F/K gap |
| EP dispatch | 2.125344 | 1:many: 96 DtoD staging + 192 DP broadcast kernels; ideal EP8 all-to-all differs | 6.981152 | missing | 5.412876 | F/E -69.556%; F/C missing; K is inventory only, no F/K gap |
| EP combine | 2.125344 | 1:1 broad phase, actual DP2 reduction+slice differs from ideal EP8 exchange | 4.196768 | missing | 3.735306 | F/E -49.358%; F/C missing; K is inventory only, no F/K gap |
| missing | missing | 48 post-combine TP4 AR (expert_parallel_allreduce); independent Frontier term missing | 37.563168 | 4.792544 | 65.823747 | F/E missing; F/C missing; K is inventory only, no F/K gap |
| missing | missing | one embedding TP4 AR | 0.222336 | 0.218336 | 1.261573 | F/E missing; F/C missing; K is inventory only, no F/K gap |

Dispatch K staging is 0.424730 ms; broadcast kernels 4.988146 ms. They sum to K dispatch 5.412876 ms. E dispatch/combine scopes appear twice per layer (outer MoE layer, inner parallel-state); only 48 outer even-sequence scopes are counted, never all 96. `attn_post_proj_tp_allreduce` is absent from E default scopes; its C measurement is not subtracted from E parent projection.

C attention AR across TP0/1/2/3 is 5.537248/5.649952/5.708768/5.644864 ms. C extra MoE TP AR is 4.792544/25.384416/28.413280/27.393568 ms. These parallel spans are not additive and include rank waiting. K similarly shows large collective waiting and cumulative host submission gaps. No row here establishes pure wire latency.

## Frontier aliases, zeros and absent operations

- `moe_gating`=4.439280 ms in OP-TRACE is a routing alias of `moe_gating_routing_topk`, not linear+routing and not an extra operation. The execution-time property named gating may sum both terms; trace labels must not be treated as that property. The raw trace logger explicitly prints routing for this alias (`sklearn_moe_execution_time_predictor.py:3377`).
- `add`, `add_attn_residual`, `add_ffn_residual` are recorded 0 in this fused-add-norm configuration. This is an actual fused accounting decision, not missing data. E has 95 nested `add` scopes totaling1.899008 ms; they belong inside norm parents and cannot be added again. Frontier retains no independent add component to compare with them; independent residual gap is missing.
- `moe_tp_allreduce` and `share_expert_tp_allreduce` are recorded0 because configured MoE TP=1 and no shared expert. They cannot be used as zero-cost matches for vLLM's extra **attention TP group** post-combine reduction, despite the similar label.
- `attn_decode` is inactive for this pure-prefill local first batch, so not missing coverage for this batch. Dense MLP/shared-expert/model-extra ops are likewise inactive under this model. Dummy DP participant comp/memory remains outside these first-real-batch traces.
- `attn_input_reshape` and `attn_output_reshape` appear in attention profile CSVs but have no independent Frontier first-forward ledger term. They time contiguous/view/reshape preparation in the profiling wrapper; vLLM scope decomposition differs. Any claimed runtime memory omission requires checking actual strides, copies and zero-copy views. Existing K Q/K copies are already mapped inside pre-projection and must not be added a second time.
- The all-forward inventory does not include logits, sampler, token D2H/CPU bookkeeping and output delivery. These must be separately inventoried if the requested scope includes official TTFT, rather than silently calling the forward table E2E-complete.

## CSV sparsity versus semantic mismatch

1. **Structural sparse columns, not missing norm samples:** `linear_op.csv` has82 rows. At4096, CSV line22 TP1 contains embedding median0.0244960003, input norm0.0226080008, post norm0.0221119998 ms. Line51 TP4 contains QKV0.1371679977, RoPE0.0323360004, output projection0.0467840005 ms; its norm columns are empty because the profiling plan stores replicated norm/memory separately. Predictor norm features use only `num_tokens`, so TP4-row blanks do not prove absent norm coverage. The embedding sample exists but its operation is absent from this ordinary Frontier forward ledger: a simulation accounting issue, not CSV shortage.
2. **Attention rows exist:** `attention_combined.csv` has the 4096, zero-prior-KV, single-prefill, TP4 FLASHINFER shape (recorded examples lines68/82), with cache-write, attention and reshape columns. Runtime kernel/memory scope boundaries and input strides still require matching; adding more identical rows does not resolve a boundary mismatch.
3. **MoE data is not simply sparse:** `moe.csv` has774 rows and18 rows at4096: three seeds × three sampled load distributions × two gating contexts. Uniform prefill_hot examples lines237/240/241 have gating-linear medians0.013872/0.013584/0.013472 ms and routing0.089296/0.091392/0.093424 ms; standalone_legacy rows624/632/647 are a different context. The predictor has separate prefill_hot models. Choosing legacy rows to judge hot-path predictions would create a false coverage claim.
4. **True unmeasured workload:** no exact M4097 profile and no matching deterministic full expert load/padding vector are established. Current sampled-uniform GEMM rows differ from deterministic balanced/round-robin routing, and eager dummy participation changes global M. This needs exact workload/profile evidence, not general denser sampling.
5. **Implementation mismatch:** current vLLM-fused profile substitutes slice/contiguous for SiLU and omits final MoE sum. No number of samples from that same implementation makes it semantically equivalent. The user-deferred SiLU repair remains deferred. Separate runtime-vs-profile component measurements are required to quantify its net effect because the profile's extra copy also costs time.
6. **Timing/context mismatch:** F microprofiles use CUDA events, E uses intrusive per-op events, K uses intrusive CUPTI profiling, C uses fewer event scopes. Large measured overhead does not establish a predictor defect. Full profiling row selection/feature proof is still required before declaring a specific ML regression or interpolation error.

## Proposed work awaiting YC review

1. **Freeze accounting boundaries and mapping acceptance.** Use the 11 Frontier comp/memory parents, 32 recorded comp/memory families, six comm/staging families and inactive/alias register above as the checklist. Decide explicitly whether the deliverable is first model forward only or includes the remaining official-TTFT preprocessing/logits/sample/output stages. Preserve existing official TTFT endpoints either way.
2. **Close source/shape identity first.** Record exact local/global M, dtype, tensor dimensions/strides, prefill KV metadata, expert routing counts/padding and comm group/message shape for the first real forward plus dummy peer. Resolve K09's allocation source. This is a narrow diagnostic contract; do not enable currently broken global metadata logging indiscriminately.
3. **Supplement computation and memory by missing boundary.** Keep an independent batch-only baseline. Collect compatible low-density scope groups for pure projection/QKV-norm, routing, shuffle, GEMM W1/W2, activation, sum, embedding/final norm and actual copies. Capture profile-side component timings under the same operations/shapes and timing family. Use complete per-rank raw kernel activity to audit coverage, while fixing the known later-profile capture loss at its cause before claiming complete three-batch traces. Do not infer a pure projection event by subtracting an AR from a different run.
4. **Supplement communication separately.** Retain collective_sim/htsim+nvlink_analytic. Measure the actual naive DP broadcast/staging, DP reduction/combine and TP reductions with exact messages/groups; preserve rank spans and critical-path waiting separately from wire/model cost. Evaluate the attention AR overprediction and missing post-combine TP term together.
5. **Compute defensible gap tables.** Every active Frontier term must appear once or be a documented alias/fused zero. Every runtime comp/memory/comm stage must map once to a parent or explicit missing row. Produce signed/absolute gaps and percentages only for matching boundaries and positive durations; retain semantic/timing conflicts rather than forcing matches. A single first-forward task can be marked complete against that approved checklist without falsely passing the skill's separate three-logical-batch gate.
6. **Review RCA before changes.** Distinguish exact-shape sampling gaps, runtime/profile implementation gaps, predictor mapping errors, communication protocol errors and remaining host/wait intervals. Present the concrete repair proposal to YC. Do not enable SiLU repair, import routing, add constants or launch supplementary runs until this plan is reviewed.

Dependencies: boundary decision → source/shape identity → {comp/memory supplements, communication supplements} → complete mapping/gap audit → reviewed correction proposal. The two supplement families may run independently after identity is frozen. This document performs only the read-only audit; every proposed supplement and repair remains unexecuted.


## Read-only verification and limits

Execution used `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python` (Python 3.13.13, conda `dev-vidur-v03-hopper-e2e`) with `PYTHONDONTWRITEBYTECODE=1`. Existing JSON/JSONL/CSV artifacts were read and joined locally; no simulator, GPU process or profiler was launched. Only this audit and its two same-prefix CSV files were written. The inventory was checked against the existing complete first-batch family JSON: 38 total families, 32 computation/memory and six communication/staging families, 3221 device activities. The Frontier computation register covers all 15 non-communication OP-TRACE labels: 11 nonzero parents, three fused-zero residual names and one routing alias. The remaining three trace names are communication (`attn_tp_allreduce`, `moe_tp_allreduce`, `share_expert_tp_allreduce`) and are covered separately. EP dispatch/combine are separate wave records.

PASS means inventory/accounting coverage of existing artifacts. It does not mean every operation has a compatible vLLM event scope, exact runtime/profile workload, or a completed numerical accuracy gate. Proposed supplemental execution and repair status: **not started; awaiting YC review**.
