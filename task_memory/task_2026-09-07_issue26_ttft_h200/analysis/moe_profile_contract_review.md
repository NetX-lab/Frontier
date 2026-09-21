## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-08 | Added an independent review of the fresh MoE profiling and runtime operator contract. |

# Independent MoE Profile Contract Review

## Verdict

**ACCEPT**, limited to the operator-contract claim. The fresh profile path is proven to time W1 followed by a contiguous first-half channel slice and W2, while the corresponding vLLM `moe_grouped_gemm` scope includes SwiGLU `silu_and_mul` between W1 and W2. Frontier has no separate routed-expert activation term that compensates for this difference. A second boundary gap is also proven: vLLM performs local `moe_sum` outside that scope, while the current Frontier MoE profile/predictor contract has no `moe_sum` term. The timing impact, sign, and contribution to official TTFT remain **Unknown** and are not accepted as the primary TTFT RCA.

## Scope and Inputs

The frozen claim was:

> The task's fresh MoE CSV reaches `frontier/profiling/moe/moe_vllm_kernel.py::_run_fused_moe_iteration`; after W1 it uses a contiguous channel slice rather than SiLU multiplied by the gate channel, while the current vLLM fused-MoE scope contains SiLU multiplied by the gate channel. The scope/operator contract may therefore be mismatched.

The review is bounded to the single H200 `pf4096_dc1024` calibration case. It does not decide the full TTFT RCA or authorize a shared operator-schema change.

Inspected inputs and revisions:

- Frontier worktree HEAD: `74b1800e9562d5ddfab3a0c091ca300b4dcb9473`.
- Fresh profiling manifest: `runs/h200-fresh-profiles-02/run_manifest.json`; the run recorded source commit `9d756e05357531c0c4a24e40f6b119401764c743`, observed worktree commit `48e2f9d2f332d287e19aec7726c98db2848c932b`, and an empty profiling-core diff between them.
- Fresh profile worker snapshot: `runs/h200-fresh-profiles-02/worker.sh`.
- Fresh profile artifact: `runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/moe.csv`, 387 data rows plus one header, `CUDA_EVENT`, `BF16`, `vllm_fused`.
- Diagnostic vLLM checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, clean at `361d941c97fcec52e544f74b7ab91c54192de9c9`.
- Runtime corroboration: `runs/h200-diagnostics-03/runtime/operators/server.ops.dp0.tp0.pp0.jsonl` contains emitted `moe_grouped_gemm` scopes for 4096-token prefill batches. These rows prove that the named scope was active; source code defines its internal boundary.

Missing inputs:

- No controlled before/after H200 profile using identical tensors, routing, kernel configuration, and warmup with only the activation/reduction boundary changed.
- No separately instrumented `silu_and_mul` or `moe_sum` duration for this case.
- No completed Frontier result or stage-level decomposition that attributes the official TTFT residual to this contract gap was available to this review.

## Independent Evidence Ledger

| Claim | Label | Direct evidence | Strength | Gap / next check |
| ----- | ----- | --------------- | -------- | ---------------- |
| The fresh worker selected the vLLM fused profiling backend. | Evidence | `runs/h200-fresh-profiles-02/worker.sh:31-40` passes `--enable_load_imbalance`; `frontier/profiling/moe/main.py:348-351` maps that flag to `use_vllm_kernel`; `main.py:768-775` forwards it to `MoEWrapper`; fresh CSV rows record `moe_grouped_gemm_backend=vllm_fused`. | Corroborated | None for route selection. |
| The actual CLI reaches `_run_fused_moe_iteration`. | Evidence | `frontier/profiling/moe/main.py:105-164` constructs `MoEWrapper` and calls `profile`; `frontier/profiling/moe/moe_wrapper.py:783-828` calls `profile_grouped_gemm`; `moe_wrapper.py:583-590,620-653` calls `profile_fused_moe_kernel`; `frontier/profiling/moe/moe_vllm_kernel.py:543-564` calls `_run_fused_moe_iteration`. | Direct | None. |
| The profiler uses W1, a contiguous first-half slice, then W2, without routed-expert activation. | Evidence | `moe_vllm_kernel.py:290-306` invokes W1; `moe_vllm_kernel.py:308-317` takes `[:, :expert_hidden_dim_per_partition].contiguous()` and optionally quantizes it; `moe_vllm_kernel.py:319-335` sends that tensor to W2. No activation call occurs in this function. | Direct | None for current source behavior. |
| This is not an incidental non-gated configuration: the case is a gated SiLU MoE. | Evidence | `data/config/models/qwen3-a3b-30b-moe.json:11-23` specifies `hidden_act=silu`, `moe_intermediate_size=768`, 128 experts, and top-k 8; `moe_wrapper.py:107-114` resolves and stores `use_gated`; the fresh CSV records `use_gated=True`. | Corroborated | None for this case. |
| vLLM's corresponding `moe_grouped_gemm` scope contains SwiGLU activation between W1 and W2. | Evidence | `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe/fused_moe.py:1721-1795` places W1, `torch.ops._C.silu_and_mul`, and W2 inside the outer `moe_grouped_gemm` scope. The modular path has the same boundary at `fused_moe.py:2082-2140`. `vllm/model_executor/layers/activation.py:58-88` defines `SiluAndMul` as `silu(x[:d]) * x[d:]`. | Corroborated | A matched controlled timing is still required for magnitude. |
| Frontier separately compensates the missing routed-expert activation. | Evidence | **No compensation exists in the inspected contract.** `frontier/operators/families.py:77-133` defines only gating linear, routing top-k, shuffling, and grouped GEMM for routed MoE. `frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py:182-209` builds exactly those routed-MoE operator times. `frontier/entities/time_components.py:488-547` sums no routed-expert activation term. The only `share_expert_act` term belongs to the distinct shared-expert family and is conditionally included. | Direct negative evidence across registry, predictor, and accounting | Recheck only if the operator schema changes. |
| vLLM's local top-k reduction is included in its `moe_grouped_gemm` scope. | Evidence | It is **excluded**: `fused_moe.py:1797-1798` and `fused_moe.py:2142` call `ops.moe_sum` after the outer scope closes. | Direct | None for the source boundary. |
| Frontier accounts for `moe_sum` elsewhere. | Evidence | It does **not** appear in `MOE_FAMILY`, `_build_moe_operator_times`, or `MoETime.total_time()` at the lines cited above. The residual `add` operator is the later transformer residual add and is not a top-k expert reduction. | Corroborated negative evidence | A contract decision is required: explicit operator versus consistently broadened aggregate. |
| The missing activation/reduction causes a particular error direction or explains the TTFT residual. | Unknown | The fresh 4096/uniform/seed-0 profile row has `moe_grouped_gemm.median=0.367071986 ms`, while a diagnostic runtime file contains per-layer scopes with different values and routing/warmup conditions. These are not a matched A/B and cannot isolate activation or reduction. | Insufficient | Run the controlled H200 comparison and then reconcile per-layer contributions with the full stage trace. |
| This contract gap is the primary TTFT RCA. | Unknown | No completed Frontier-vLLM TTFT component sum or controlled repair rerun was available. | Insufficient | Preserve as a candidate until workflow, communication, routing, and post-forward alternatives are independently compared. |

## Reconstructed RCA

**Symptom.** The same public operator label, `moe_grouped_gemm`, refers to different executed work in fresh Frontier profiling and instrumented vLLM.

**Trigger/input.** The fresh H200 worker profiles Qwen3-30B-A3B with `use_gated=True`, BF16, TP1/EP8, `standard_fused_topk`, and the vLLM fused backend. The 4096-token rows are part of this artifact.

**Mechanism.** The profiler materializes a W1 output of width `2 * expert_hidden_dim_per_partition`, but consumes only its first half as W2 input. vLLM instead transforms both halves using `silu(first_half) * second_half` before W2. vLLM then locally reduces top-k expert outputs with `moe_sum` outside its named grouped-GEMM scope. Frontier neither performs the activation in the profiled step nor models activation or local reduction as another routed-MoE operator.

**Impact.** The operator contract is structurally non-equivalent, so direct profile-to-runtime comparison under the shared name is invalid. The numerical effect is unknown. Kernel configuration, routing allocation, cache reuse, process warmup, and asynchronous timing can offset or dominate the missing work in an unmatched observation, so source omission alone does not establish underprediction or TTFT causality.

**Credible alternative.** If `moe_grouped_gemm` were intentionally defined as only two GEMMs, omitting activation could be a naming choice. That alternative fails the current parity use: vLLM's same diagnostic scope expressly includes activation. It would require distinct names or documented aggregation before the two values could be compared.

**Counterexample to numeric attribution.** A single fresh profile row and a diagnostic runtime scope can be numerically close or ordered opposite to the expected activation cost because they do not fix routing, kernel config, worker warmup, or input tensors. Such agreement would be accidental and would not falsify the structural mismatch.

Confidence is **high** for the contract mismatch and absence of compensation, and **low/unknown** for its performance magnitude and TTFT share.

## RCA Corrections

1. Replace “the contract may be mismatched” with “the current profile/runtime operator boundaries are mismatched for gated SiLU MoE.” This is directly supported by the call chains.
2. Do not replace it with “the missing activation explains the TTFT gap.” That causal link has no controlled measurement or completed stage decomposition.
3. Extend the contract finding to include local `moe_sum`. Activation belongs inside the vLLM named scope; `moe_sum` is outside that scope but is also absent from Frontier's routed-MoE accounting.
4. Do not call `share_expert_act`, residual `add`, communication, or CPU overhead a compensation. They represent distinct work and do not execute the missing SwiGLU transform or local top-k reduction.

## Solution / Implementation Review

No implementation diff was proposed in the frozen claim, so this review does not approve a patch. The smallest root-cause-reaching correction for activation is:

1. Thread explicit `use_gated` and `activation` values from `ModelConfig` through `MoEWrapper._profile_with_vllm_kernel` into `profile_fused_moe_kernel`; do not infer behavior from a model-name string.
2. For this gated SiLU case, allocate an `N`-wide activation buffer and apply the same SwiGLU contract as vLLM, `silu(first_half) * second_half`, between W1 and W2. Remove the raw first-half slice as W2 input.
3. Preserve the FP8 order: W1 produces its output, activation consumes both dequantized/base output halves, and only the resulting activated tensor is quantized with the existing activation-quantization helper before W2. A branch that slices on FP8 would retain the defect.
4. Preserve supported non-gated behavior through an explicit activation path and fail fast for unsupported `use_gated`/activation combinations. Do not use a silent slice fallback.

This activation correction reaches the proven mechanism and keeps current routing/load-imbalance kernel selection intact. It still does not close the full expert-compute contract until `moe_sum` ownership is made explicit. Adding `moe_sum` to `MOE_FAMILY`, profiling, execution-time storage, and vLLM instrumentation is a shared operator-contract change and should pass the workspace approval gate. The alternative is to broaden one existing aggregate consistently on both sides and rename/document it so its boundary is auditable. Silently folding reduction into only the profiler or only the predictor would create another comparison mismatch.

No hard-coded timing coefficient, model-name branch, exception suppression, or historical numeric reuse is justified.

## Corrected Plan

1. **Activation contract unit check.** Add a pure tensor-level test with sentinel first and second halves. Acceptance: gated SiLU output equals `silu(first_half) * second_half`; changing either half changes W2 input; the test fails with the current first-half slice.
2. **Backend call-order check.** Mock or instrument the profiling step. Acceptance: W1 -> activation -> optional activation quantization -> W2, with W2 receiving an `N`-wide activated tensor. Test BF16 and FP8 ordering; test explicit failure for unsupported activation.
3. **Matched H200 A/B.** With identical weights, input, top-k weights/IDs, local expert map, kernel config, warmup, active-step count, and CUDA-event boundaries, compare the corrected profile scope with the vLLM scope for the 4096-token case. Acceptance: record raw per-iteration values and a declared tolerance before deciding numerical parity; no post-hoc scale factor.
4. **Resolve `moe_sum` ownership.** Choose an explicit registry operator or a consistently redefined aggregate. Acceptance: the profiler target list, vLLM diagnostic scopes, predictor inputs, and `MoETime.total_time()` cover local reduction exactly once. This is a shared-contract decision and remains gated.
5. **Only then test causal impact.** Regenerate fresh H200 MoE rows and predictor cache, rerun the same Frontier case, and compare stage/operator sums with official vLLM TTFT. Acceptance: predicted, actual, absolute error, relative error, and component delta are reported; the claim becomes a TTFT RCA only if the controlled component delta explains the observed change.

Explicitly deferred: other request lengths, TPOT/throughput repair, arbitrary calibration coefficients, CPU-overhead attribution, communication repair, and any conclusion that this is the dominant TTFT cause.

## Verification Record

Read-only checks performed from the active worktree:

```bash
git rev-parse HEAD
git diff 9d756e05357531c0c4a24e40f6b119401764c743..48e2f9d2f332d287e19aec7726c98db2848c932b -- \
  frontier/profiling/moe \
  frontier/execution_time_predictor/sklearn_moe_execution_time_predictor.py \
  frontier/operators/families.py

rg -n "_run_fused_moe_iteration|moe_grouped_gemm|silu_and_mul|moe_sum" \
  frontier/profiling/moe \
  frontier/execution_time_predictor \
  frontier/operators \
  frontier/entities \
  /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908/vllm/model_executor/layers/fused_moe

wc -l task_memory/task_2026-09-07_issue26_ttft_h200/runs/h200-fresh-profiles-02/runtime/profiles/compute/h200/qwen3-a3b-30b-moe/moe.csv
```

Observed results:

- The profiling-core diff across the run's prelaunch and observed commits was empty.
- The fresh CSV contained 388 lines: one header and 387 data rows.
- A selected 4096/uniform/seed-0 row recorded BF16, CUDA_EVENT, `vllm_fused`, TP1/EP8, `use_gated=True`, `total_routed_tokens=4081`, and `moe_grouped_gemm.median=0.3670719861984253 ms`. This value proves artifact population, not parity or causality.
- Diagnostic output emitted 4096-token `moe_grouped_gemm` rows from all-worker instrumentation; the source boundary shows what each scope contains.
- Repository tests contain predictor/schema checks for `moe_grouped_gemm`, but the bounded search found no regression test requiring profiler SwiGLU activation or local `moe_sum` accounting.
- LSP diagnostics were requested for the five inspected Frontier files. `frontier/operators/families.py` and `frontier/entities/time_components.py` were clean. The profiling/predictor modules reported unresolved GPU/runtime imports in the local analysis interpreter plus existing type diagnostics. There is no implementation diff under review, so these diagnostics neither validate nor invalidate a proposed repair.

Residual unknowns are the isolated activation cost, isolated reduction cost, exact corrected-vs-runtime tolerance, and any contribution to official TTFT. No GPU command or source modification was performed during this review.

## D011/D012 provenance refresh

At committed Frontier9e3d1874, the active uniform supplement still reaches the same `_run_fused_moe_iteration` W1 -> first-half contiguous slice -> W2 path. The relevant profiler source was inspected again during CPU generation01 training. Current clean vLLM46f7b179f still contains `silu_and_mul` inside grouped GEMM and `moe_sum` outside. D012 changes routing runtime selection and shared identity only.

Current fresh row: supplements/moe-uniform-01/moe.csv, num_tokens4096/load_distributionuniform/seed0/contextprefill_hot; BF16, gated, vllm_fused, uniform_topk. Grouped-GEMM median0.3315840065479278ms, routing-topk median0.0892960019409656ms, shuffling median0.0443679988384246ms. These are artifact values, not an aligned operator comparison. The row's local routed assignments4081 describe the controlled load-imbalance training sample; they are not a measured global runtime count or proof that the real batch has4081tokens. This refresh introduces no historical numeric inputs into the new run and does not establish the activation gap's timing magnitude or direction.
