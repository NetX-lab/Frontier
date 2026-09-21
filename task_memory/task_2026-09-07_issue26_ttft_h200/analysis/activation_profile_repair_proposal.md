## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Prepared the next scoped profiling decision without modifying the active run. |

# D013 proposal: correct the routed activation profiling boundary

Status: PROPOSED, not approved. Current D012 CPU simulation remains an independent execution step on committed9e3d1874. No D013 code changes or GPU measurements have started.

## Observed cause and limits

The currently selected fresh uniform profile reaches `frontier/profiling/moe/moe_vllm_kernel.py::_run_fused_moe_iteration`. Its W2 input is the contiguous first half of W1 output (lines308-309). The case is gated SiLU (`ModelConfig.activation`, `use_gated_mlp`; wrapper lines107-114), and vLLM46f7b179f executes `torch.ops._C.silu_and_mul` inside `moe_grouped_gemm` between W1 and W2 (fused_moe.py1721-1795). Frontier has no separate routed-expert activation accounting. This is a proven operator-contract defect; its timing magnitude, direction and contribution to clean TTFT remain unmeasured. Evidence: moe_profile_contract_review.md, refreshed with current uniform rows.

A separate proven gap is local `moe_sum`, outside vLLM's grouped scope. That requires an explicit operator ownership decision and is not silently included in this proposal.

## Recommended scope

Reuse the current ModelConfig activation/gated classification, existing activation implementation and vLLM custom kernel; thread the declared behavior through `MoEWrapper._profile_with_vllm_kernel` and `profile_fused_moe_kernel`. Current path: W1 -> first-half contiguous copy -> W2. Corrected path for this case: W1 -> gated SiLU over both halves -> W2, with the activation output buffer allocated outside the timed iteration as in vLLM. Existing `SiluAndMul` already uses the correct native kernel, but its forward allocates output; prefer a small extension permitting supplied output if needed, not a replacement activation module. Preserve activation-before-FP8-quantization order. Respect supported non-gated model settings through the existing enum/activation registry; no model-name cases or silent slice fallback.

Add narrowly scoped regression checks for both-half dependence and W1/activation/quantization/W2 order. On H200 step_main, perform a controlled same4096-case A/B with identical tensors, topk IDs/weights, EP map, kernel config, warmup and measurement boundary; save raw iterations and match corrected output with a reference gated expression before using latency values. Regenerate affected fresh MoE input rows and retrain/replay the same CPU Frontier case. Retain collective_sim/nvlink_analytic, prefix OFF, uniform runtime, current metric definition and no historical numeric inputs. Report component delta and clean mean error without claiming that this defect explains the entire gap.

Benefits: removes a proven operation omission inside an already-defined scope, while preserving the vLLM comparison boundary. Cost: shared profiling/activation interface edits, focused tests, new H200 profile execution and another fresh CPU replay. Likely production ownership spans moe_wrapper.py, moe_vllm_kernel.py and, only if needed, common/layers/activation.py; tests and the controlled measurement worker live under tests/. Exact patch remains subject to the agreed scope, not an unbounded model refactor.

## Alternatives considered

- Add a separate activation predictor: rejected because the existing vLLM grouped scope already owns activation; it would fragment an existing boundary without benefit.
- Fold activation and local reduction into the existing grouped timing: rejected for this sub-step because vLLM's current grouped scope excludes reduction. A consistently broadened scope would require a separate cross-side operator-contract decision.
- Fix activation and introduce a new explicit moe_sum family operator together: viable broader follow-up, but not recommended before the first controlled activation measurement; it changes predictor/time-component/instrumentation ownership in addition to the proven inner-scope fix.
- Add a timing constant or call the TTFT residual CPU overhead: rejected; neither executes the missing operation or establishes causality.
- Hold this repair and let YC redirect: valid alternative to the recommended scoped activation correction.

## Approval basis

The task's AGENTS.md Approval Gate covers modifications of shared interfaces/data contracts. D012 authorized routing runtime selection and model identities, not this profiling interface. YC also explicitly requested grill-me for high-value steps. This proposal asks once for the new scope; it does not require reapproval of the current CPU run. D013 source changes wait for an explicit answer and completion of the active simulation.
