## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Read-only D019 audit of first-forward profile coverage and a proposed P/S/V experiment plan. No measurements, training, simulations, or production edits performed. |

# D019: profile coverage and predictor attribution plan

Status: PROPOSED, awaiting YC's review. Numerical correction and first-forward CUDA-span RCA remain INCOMPLETE.

## Finding

The hypothesis that profile coverage contributes to the gap is plausible for some MoE features, but insufficient CSV sampling has not been established as the cause of the approximately 19.67% clean TTFT gap. Several first-forward compute predictions already equal measured exact-feature rows. More samples of those same inputs would test measurement stability and runtime context, not repair an RF interpolation error that has already been demonstrated.

The current predictor explicitly prefers exact measured feature keys before regression. `frontier/execution_time_predictor/sklearn_execution_time_predictor.py:146` builds keys by averaging target values across the same features; the prediction path at line 4532 returns these values before calling the estimator. The on-demand path also checks exact keys at line 4812. “RF predictor” therefore does not mean that every reported P value came from RF inference.

Historical `first-batch-op-rca/profile_coverage_exact_rows.json` supplied row anchors; it did not publish a complete row-versus-prediction receipt or establish the source path for every runtime estimate. The arithmetic reconciliation below is possible from those rows and the existing P log. A complete audit still needs the selected feature vector, CSV filtering/context, exact-key hit or RF fallback, and all contributing source rows for each operation. No estimator was loaded or invoked during this review.

## Current inputs and coverage

Paths below are relative to the active task directory.

| Family | Current source | Verified coverage | Remaining issue |
| --- | --- | --- | --- |
| Linear | `runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/linear_op.csv` | 82 rows, 41 token sizes; two 4096 rows covering TP4 attention projections and TP1 embedding/LN. | No nominal 4097 row. DP0 attention projection remains local 4096 in the observed first forward; global MoE 4097 alone does not make its linear row missing. |
| Attention | Same directory, `attention_combined.csv` | 2026 rows, including 1890 true-mixed rows. Pure BS1 prefill4096/KV0/TP4 rows at physical lines68 and82. Local mixed total4097 rows exist, including prefill4096 plus one decode token with KV4096. | Audit the actual selected phase, sequence and KV features, not a nonexistent generic num_tokens column. A local mixed4097 attention row does not cover global4097 MoE dispatch. |
| MoE | `supplements/moe-uniform-01/moe.csv` | 774 rows, 43 nominal token sizes; 18 nominal4096 rows, six labeled uniform across three seeds and two gating contexts. EP8, 16 local experts, topk8. | No nominal4097 row. “Uniform” load samples are random, not the deterministic balanced runtime vector. Exact expert assignment and padding coverage remain missing. |

Physical line numbers include the CSV header. Counts establish present rows, not complete matching of all implementation and execution-context fields.

The MoE prefill_hot uniform4096 samples at lines237/240/241 have local routed totals4081/4058/4073 and load CV approximately0.06541/0.04875/0.04951. Current Frontier first-forward EP lanes instead use4096 assignments each and perfectly balanced local counts. The runtime model uses routed total, tokens per expert, load CV, min/max load ratio, entropy and Gini, among other features (`sklearn_moe_execution_time_predictor.py:1123,1780`). A nominal-token match is consequently insufficient to prove an exact MoE feature hit.

This finding uses actual CSV fields, not an assumption from a default setting. Those same rows declare `routing_runtime_path=uniform_topk`, `routing_assignment_policy=round_robin_uniform`, `routing_weight_policy=uniform_1_over_topk`, and `routing_uses_router_logits=False`. The current producer source explains the apparent conflict: `MoEWrapper.profile_all` at line804 first calls `_prepare_routing_inputs` for shuffling/GEMM, then separately calls `profile_gating`. `_prepare_routing_inputs` at line296 uses `generate_expert_routing` over128 global experts; its uniform branch generates random scores and selects topk. EP0's expert map keeps experts0–15, and local counts become the GEMM feature row. `profile_gating` executes the configured uniform_topk implementation independently. The combined output adds routing metadata and the separately generated GEMM load features to one row. Therefore the row's uniform_topk label does not establish deterministic round-robin shuffling/GEMM inputs.

The nominal4096 is the global source tensor's row count for this standalone MoE profile. With topk8 it creates32768 global assignments; EP8 gives an expected4096 assignments to each rank, not an exact count. The observed4081/4058/4073 are actual sampled EP0 local totals, not global totals and not DP-local source tokens. The current runtime Frontier perfect-balanced local vector has16 entries of256. These definitions explain why equal nominal4096 rows can still miss the runtime feature key.

The raw prefill_hot producer CSV at `runs/h200-uniform-moe-profiles-01/runtime/prefill_hot/compute/h200/qwen3-a3b-30b-moe/moe.csv` has these same seed/total/CV/GEMM values at the same physical lines237/240/241 (only decimal serialization precision differs after merge). This excludes the supplement merge inventing the unequal loads.

Provenance anchors are the supplement's `moe.summary.json` base/supplement paths and profiling command, and `runs/h200-uniform-moe-profiles-01/runtime/environment.log:247,249`, which record executed uniform_topk, load distributions and three-seed commands in both contexts. The environment records Python3.10.16 in `vllm-bs-0.10.2`, Torch2.8.0+cu128 and FlashInfer0.3.0. The actual rows independently establish unequal local loads. This bounded review has not recovered a complete immutable Frontier producer source receipt, so it does not claim a historical source hash from the current wrapper alone. That receipt belongs in the P/S provenance audit before producer changes are accepted.

The actual vLLM first-forward contract contains DP token counts[4096,1], global hidden states[4097,2048] and logits[4097,128]. Deterministic uniform routing gives EP0 eight experts257 assignments and eight experts256:4104 local assignments; other EP ranks retain4096. The padded kernel workload must be recorded separately. The extra logical token is not a bound on latency: an expert can cross a kernel block boundary. See `first-forward-ep-contract.md`.

## Existing P versus CSV arithmetic

All values are milliseconds per model layer. P is the existing48-layer total divided by48 from `first-batch-op-rca/frontier_summary.json`. The table is a read-only arithmetic comparison, not a new prediction run.

| Operation | Existing P | Existing exact-row median or mean of medians | Interpretation |
| --- | ---: | ---: | --- |
| QKV / attn_pre_proj | 0.137168 | 0.137167998 | Equal to log precision; linear CSV line51. |
| RoPE | 0.032336 | 0.032336000 | Equal to log precision; line51. |
| Attention output projection, compute only | 0.046784 | 0.046784000 | Equal to log precision; line51. |
| Input layernorm | 0.022608 | 0.022608001 | Equal to log precision; line22. |
| Post-attention layernorm | 0.022112 | 0.022112000 | Equal to log precision; line22. |
| Prefill attention | 0.146384 | (0.146911994 + 0.145855993)/2 = 0.146383993 | Equal to mean of matching pure-prefill rows68/82. |
| KV cache save | 0.016816 | (0.023167999 + 0.010464000)/2 = 0.016816000 | Equal to mean of rows68/82; variability/context still needs explanation. |
| MoE grouped GEMM | 0.325748 | prefill_hot uniform samples:0.331584007/0.314976007/0.315935999 | Nominal4096 only; differing routed feature vectors prevent a direct exact-row attribution. |

For example, QKV P totals6.584064ms over48 layers and its matching CSV median gives the same sum, while the heavily instrumented vLLM QKV scope totals3.927295990ms. That discrepancy cannot be explained simply by RF interpolating between sparse token sizes: the observed P already reproduces its measured anchor. Possible explanations include profiling runtime context, selected kernel and scope/shape mapping. These require S/V checks.

Do not average prefill_hot and standalone_legacy MoE gating rows indiscriminately. Their distinct measurements were intentionally collected under distinct contexts; the selected pseudo-model/context is part of the P receipt.

Explicitly UNVERIFIED: the selected exact-hit versus RF-fallback path for `moe_gating_linear` (P0.013780), `moe_gating_routing_topk` (P0.092485), `moe_shuffling` (P0.052945), and `moe_grouped_gemm` (P0.325748). Gating uses different feature selection from grouped GEMM, so the latter's sampled load mismatch cannot establish that gating missed its exact key. The table's linear row51 has tokens4096/TP4/Q32/KV4/hidden2048/BF16; LN row22 usesTP1; attention rows68/82 have pure BS1, chunk4096, KV0, TP4, FLASHINFER, BF16 and CUDA_EVENT. Final P receipts must retain all loader-filter fields as well as these shape anchors.

## Minimal experiment separation

Define P as Frontier's frozen predicted cost, S as a fresh measurement using the same implementation and exact feature/shape contract as that prediction, and V as a measurement of the corresponding actual vLLM operation. S and V must be compared only when their scopes, shapes and execution contexts match; otherwise publish separate rows and the missing condition.

| Arm | Smallest deliverable after plan approval | What it can establish |
| --- | --- | --- |
| P: CPU audit | For each first-forward operation: configured CSV, relevant filtered rows, feature key, selected gating context, exact lookup versus RF, emitted P, aggregation count. Include per-EP MoE feature vectors. Reuse existing P traces first; use bounded CPU diagnostics only where a source decision is missing. | Whether an estimate is an exact measured anchor, a regression result at an uncovered feature, or a wrong mapping/context. |
| S: targeted H200 same-implementation profiling | Re-measure only observed first-forward inputs. Start with QKV4096, pure prefill4096/KV0, the known-variable KV-save, and MoE4096 balanced versus4097 with actual EP0 assignments/padding. Record kernel configuration and per-sample values under the current image, dtype and eager mode. | P−S isolates predictive/measurement-anchor error only while the implementation and input contract remain fixed. Comparing existing anchors with fresh S separately tests stability/context. |
| V: matched vLLM scope measurements | Retain the formal request identity and a common forward/layer/DP/TP/EP identity; record source shapes and real/dummy token vectors. Use bounded compute/memory families with CUDA events and targeted kernel identities, and a separate communication family. | S−V isolates implementation/runtime-context differences when scopes match. It cannot be assigned to ML sample coverage. |
| Integration | Resolve proven mapping or implementation issues first, then supplement only missing exact features; regenerate the affected model artifacts and rerun the fresh same-case Frontier and isolated vLLM diagnostics. Finally perform a clean E2E run. | Whether a scoped correction reduces the actual CUDA span and clean TTFT gap. Training fit alone is not acceptance. |

For truly uncovered MoE features, compare frozen P with exact S before inserting S into training. Adding that row and then observing exact lookup return it is a useful integration check, but by itself is not independent evidence of model generalization. Keep original P and holdout measurement visible.

The existing sampled uniform MoE CLI cannot request the deterministic expert vector merely by setting `--num_tokens_list 4097`: `load_distribution.py:72` samples random top-k IDs. `MoEWrapper._prepare_routing_inputs` verifies supplied expert counts against its generated routing; counts alone cannot override routing. However, `profile_shuffling` and `profile_grouped_gemm` already accept a shared `routing_inputs` dictionary containing topk IDs/weights, local counts, global expert count and expert map. A bounded diagnostic script can use this existing seam to supply the observed deterministic mapping and verify EP0/other-rank vectors without adding a general traced-routing import feature. It must also record the actual block-alignment output; the current sample CSV lacks that exact evidence. Do not change the production routing interface as part of this diagnostic.

## Components to reuse

- `tests/performance/issue26_h200_fresh_profiles_worker.sh`: authoritative current linear/attention environment and entrypoint arguments. Its token grid has4096 and increments of16, not4097. Reuse bounded calls to `frontier.profiling.linear_op.main` and `frontier.profiling.attention.main`; do not rerun the whole grid to answer one shape question.
- `tests/performance/issue26_h200_uniform_moe_profiles_worker.sh`: current uniform_topk and gating-context configuration. `frontier.profiling.moe.main --num_tokens_list` supports targeted nominal tokens; the existing wrapper routing-input seam is needed for exact deterministic expert layouts.
- `tests/e2e/issue26_first_batch_op_rca_frontier.py`: existing P log reduction.
- `tests/e2e/issue26_first_batch_op_rca_vllm.py`, `issue26_first_batch_op_rca_selection.py`, `issue26_first_batch_op_rca_kernels.py`, and `issue26_first_batch_op_rca_communication.py`: existing V identity, operator, kernel and communication reducers. Extend only missing evidence, keeping non-overlapping operator scopes explicit.
- `tests/e2e/issue26_h200_diagnostics_worker.sh`: isolated vLLM diagnostic launch, with the existing case and warmup contract.
- `tests/e2e/issue26_cpu_frontier_worker.py`: final fresh Frontier execution after reviewed correction, not needed for this plan audit.

All script paths above are relative to the active worktree. GPU execution must retain H200/step_main, approved image and runtime, company HTTP proxy for external/company network operations, prefix caching OFF, uniform routing, and collective_sim/htsim with nvlink_analytic.

## Causes that must remain separate

1. **Data sparsity or predictor error:** established only by the actual feature hit/miss and P−S for the same implementation. Existing compute anchors disprove a blanket “all first-forward ops lack4096 data” claim.
2. **Kernel implementation/context:** the known gated-SiLU omission and different fused-kernel paths cannot be repaired by more samples of the same incomplete implementation. Gated-SiLU repair remains deferred by YC; list its unmatched scope rather than silently implementing it. Runtime prefix and kernel block configuration also belong here.
3. **Mapping/accounting:** local4096 attention versus global4097 MoE, real versus dummy work, per-expert padding, and inclusive scopes must be aligned before numerical attribution. A post-projection scope containing TP reduction is not a pure GEMM comparator.
4. **Communication configuration/protocol:** attention TP allreduce P totals17.89944ms, whereas the separate communication diagnostic measured about5.54–5.71ms across48 layers. This is a backend/protocol/runtime-wait question, not compute CSV sparsity. Post-MoE actual TP4 reduction and EP dispatch/combine must remain separate; do not append an overlapping inclusive scope. Rank collective time can include arrival skew.
5. **CPU/workflow timing:** keep the D019 ordering: complete CUDA-span RCA/correction before introducing measured profiling CPU overhead into workflow. CPU delay is not a free parameter to offset a compute or communication error.

The full-op first-forward CUDA span116.210144ms came from a more intrusive mode than the batch-only80.335617ms. These different executions are diagnostic comparisons, not interchangeable clean values or an additive decomposition of the clean TTFT gap. CUDA-event spans also include stream idleness when host launch gaps fall between the recorded events; kernel activity and gaps should reconcile within the same measurement when making that distinction.

## Proposed parallel responsibilities and dependencies

`freeze first-forward identities/scopes -> {P CPU audit, S/V compute-memory measurement, communication protocol audit} -> joint coverage/accounting review -> scoped correction -> fresh CUDA-span validation -> clean E2E -> later CPU profiling/workflow phase`.

- **Lane P, CPU only:** own the prediction/CSV provenance table and missing-feature list; avoid training until P is frozen. Can run alongside GPU collection.
- **Lane compute/memory, H200:** own exact S profiling and V compute scopes, sharing routing/layout receipts. Use GPU time serially per measured workload to avoid concurrent interference. Code review and trace reduction can overlap other lanes.
- **Lane communication:** own group identity, payload, collective implementation and wait accounting. It can inspect source/config in parallel; its timed GPU jobs must be scheduled separately from S/V compute runs.
- **Integrator/reviewer:** reconcile exclusive operator coverage against the same-run CUDA span, publish unmatched/overlapping/gap buckets, and identify the smallest repair. Keep scheduling/arrival-boundary proposals separate from predictor correction.

Required plan decision: approve the bounded exact-layout diagnostic and isolated S/V/communication sequence. A later change to a shared profiling schema or production routing/predictor interface needs a concrete separate design decision if the existing seams prove insufficient. No arbitrary39ms/3ms host constants, multiplicative op factors, bulk CSV fill, or old-version numerical reuse are proposed.

Read-only review completed; new tests and measurements in this document are proposals, not executed results.
