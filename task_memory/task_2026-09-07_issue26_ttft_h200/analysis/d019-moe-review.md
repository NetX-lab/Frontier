## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Independently reviewed the gated SiLU/reduction repair against pinned vLLM before GPU validation. |

# D019 MoE activation and reduction repair review

## Verdict

**ACCEPT for the source repair and targeted GPU validation.** The BF16 expert computation now follows the pinned vLLM sequence, including the previously missing activation and output reduction. This verdict does not establish GPU numerical parity, a calibrated operator latency, or TTFT acceptance; those remain pending.

Reviewer: `/root/repair_review`, independent read-only source review using `double-checking-rca`. Production and test files were not modified by this reviewer.

## Scope and inputs

- Frontier branch `task/issue26-ttft-h200-20260907`, HEAD `fc205071f4a04f78591ec2a7ef2912fa91d75912`, working diff of `frontier/profiling/moe/moe_vllm_kernel.py` and new `tests/unit/test_moe_fused_expert_numerical_parity.py`.
- Reference `/data/ycfeng/tmp/vLLM-BS`, pinned `46f7b179fd3bf42b9616dc4670cba419afdb2085`, `vllm/model_executor/layers/fused_moe/fused_moe.py`.
- Task `requirements.md` D019 execution approval, `plan.md` CUDA-first sequence and frozen BF16 Qwen3 case, and `analysis/first-batch-op-rca/coverage_audit_d019.md`.
- Current review is scoped to the local fused expert path. DP collectives, routing-input producer corrections, CPU accounting, and fresh GPU timing are independently owned work.

## Independent evidence ledger

| Claim | Label | Direct evidence | Strength | Gap / next check |
| --- | --- | --- | --- | --- |
| Old profile omitted gated activation and reduction | Evidence | Diff removes W1 output slicing/copy; adds `silu_and_mul` at Frontier lines 310–312 and `moe_sum` at 340. Reference uses these at lines 1747–1748 and 1797–1798. | Direct | Fresh profile quantifies the effect. |
| BF16 operation order and weight placement match | Evidence | Frontier lines 292–340: W1 without routed weight, gated SiLU, W2 with routed weight and `top_k=1`, sum. vLLM lines 1724–1798 use the same defaults. | Direct | Four GPU numerical cases remain pending. |
| Shared W1/W2 output workspace is safe by construction | Evidence | Frontier 534–549 mirrors reference 1652–1662: activation has separate storage; W2 overwrites W1 only after activation consumes W1. | Direct | GPU test exercises aliased storage. |
| EP non-local outputs are initialized to zero rather than summed from uninitialized memory | Evidence | Pinned Triton kernel lines 380–388 writes zeros for `expert_id == -1`; both W1 and W2 use the same aligned expert mapping. | Direct | Rank 0 and rank 1 numerical checks exercise local/non-local entries. |
| Current BF16 kernel config selection agrees with reference | Evidence | Frontier 515–522 passes BF16 config dtype and actual M; vLLM 1627–1649 selects the same values. Actual 4096/4097 fit below default chunk size 32768 (`envs.py:652–653`). | Direct | Capture active runtime chunk/config values with GPU evidence. |
| New Frontier grouped-GEMM timing matches vLLM's same-name scope alone | Evidence | False: Frontier line 340 is inside its timed iteration, while vLLM line 1797 is outside `moe_grouped_gemm`. | Direct | Use vLLM grouped-GEMM plus independent reduction, without nested duplication. |
| Full FP8 numerical equivalence is established | Unknown | Existing Frontier FP8 compute type is forced FP16 at 232–233 despite BF16 output buffers; config lookup at 515 does not pass `use_fp8_w8a8`; reference derives compute type from hidden dtype and passes FP8 config flag. Both discrepancies predate this diff. | Direct boundary, unmeasured impact | Keep FP8 equivalence unclaimed; separate scoped follow-up if required. |
| The repair closes the aggregate CUDA/TTFT gap | Unknown | No fresh GPU/profile/predictor replay result belongs to this review. | Missing | P/S/V and complete forward rerun. |

## Reconstructed RCA

The frozen case reaches a gated expert FFN. The old profiling iteration performed W1, selected the first half of its output with a contiguous copy, then W2. Actual vLLM instead applies gated SiLU to both halves, executes W2 with the routed weights, and reduces the top-k outputs. The source mismatch directly proves missing work and incorrect arithmetic in the profiling producer. The replacement acts at that producer rather than fitting a latency scale.

An alternative explanation based solely on sparse RF training cannot account for these omitted operations. It may still explain other P/S discrepancies. Conversely, adding activation and sum does not imply that their historical kernel-active durations can be added directly to the old CUDA-event profile: the repair removes a copy, changes memory reuse, and executes a different sequence, so fresh S is necessary.

The potential counterexample of EP-local kernels leaving remote assignments uninitialized is addressed by the pinned kernel's explicit zero writes. Another counterexample is FP8: the pre-existing quantization/config path differs from reference, so the BF16 source verdict must not expand into a general FP8 parity claim.

## RCA corrections and measurement boundary

The new Frontier `moe_grouped_gemm` profile covers **W1 + gated SiLU + W2 + top-k output reduction**. The old vLLM same-name event value (18.180160 ms for the previous diagnostic) excludes reduction and therefore cannot be reused as a complete counterpart. The numerical mapping must use the inclusive expert-compute scope or the non-overlapping sum of grouped-GEMM and reduction from the same aligned measurement family/run. Existing shuffle/alignment remains separately profiled and is not added again inside this patch.

No independent reduction predictor term was found in the current Frontier execution-time sum; including it once in the expert profile does not create a simulator-side duplicate. Keep this coverage meaning in the analysis/profile receipt; renaming a shared operator or adding a new registry family is unnecessary for this bounded repair.

## Implementation review

- Root-cause reach: direct; the missing arithmetic is restored with pinned vLLM primitives.
- Shape/layout: full hidden width, doubled W1 width, halved activated width, global routed token order, EP map, and shared workspace agree for this case.
- Scope/minimality: one production helper signature and its single production caller plus one focused GPU test. No new model-name branch, timing factor, exception fallback, or parallel implementation.
- Failure semantics: vLLM kernel failures remain visible. Exact numerical comparison uses nonzero random inputs and weights, so a missing activation or reduction cannot pass as an all-zero fixture.
- Tests: 4096/4097 global-token sizes × EP ranks 0/1 exercise the extra-token layout and both local expert groups, with `rtol=0, atol=0` against the pinned `fused_experts`. The test is appropriate for this case; it does not establish every dtype, activation, chunked workload, or profiling measurement context.
- Existing boundaries: FP8 config/compute-type differences and lack of vLLM-style large-M chunking predate this patch. The wrapper already always materializes a gated W1 and exposes no activation selector. Do not broaden this repair into unsupported activation-generalization work.

## Corrected next steps

1. Execute the four existing GPU numerical tests in the approved H200 image with pinned vLLM and save the exact runtime/config receipt. Acceptance: all four pass, no skips, bitwise equality.
2. Collect fresh exact-routing profiles with this corrected producer and verify actual expert assignment/padding. Preserve new grouped-GEMM coverage semantics.
3. Compare P, fresh S and corresponding vLLM grouped-GEMM + reduction using aligned event boundaries. Do not compare CUDA-event predictions against kernel-active sums.
4. Re-run the full first-forward gate after integrating independent communication and other op corrections. CPU integration remains downstream.

Source blockers: **none for the approved BF16 case**. Pending gates: GPU numerical parity, fresh operator timings, predictor consumption, and end-to-end validation. Deferred: existing FP8/chunking generalization.

## Verification record

Executed from the Frontier worktree:

```bash
git diff -- frontier/profiling/moe/moe_vllm_kernel.py tests/unit/test_moe_fused_expert_numerical_parity.py
git diff --check -- frontier/profiling/moe/moe_vllm_kernel.py tests/unit/test_moe_fused_expert_numerical_parity.py
```

Observed: diff inspection completed; `git diff --check` exited 0 without output. This detects tracked patch whitespace errors only; the new untracked test was read directly. No Python or GPU test was executed by this reviewer. GPU outcomes and timing error values are pending, not inferred from the successful source check.
