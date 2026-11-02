# G3h native gate and routed-ID qualification

Updated for B12b (assignment dated 2026-10-28). The native reference root throughout this note is
`/data/ycfeng/tmp/opencode/vllm-v0.28.0`, revision
`2cf0a6915ce544dc493a0990f2ea38d81601128a`. Producer references are relative to the
Frontier working root. These are source comparisons and CPU boundary checks;
no GPU kernel census or package build was performed for B12b.

## Gate construction and dispatch

Native V4 constructs `GateLinear(input_size=config.hidden_size,
output_size=config.n_routed_experts, bias=False, out_dtype=torch.float32)` in
`vllm/models/deepseek_v4/nvidia/model.py:571-577` and requests float32 router
logits at line 706. Its parameter dtype defaults to the current model dtype
through `vllm/model_executor/layers/linear.py:251-253`, under the model loader's
`set_default_torch_dtype(model_config.dtype)` context
(`vllm/model_executor/model_loader/base_loader.py:53`).

The producer now shares the existing grouped-sigmoid `GateLinear` constructor
with balanced routing in `frontier/profiling/moe/moe_impl.py:156-180`. It supplies
`router_config.dtype` before construction computes dispatch eligibility and
keeps `out_dtype=torch.float32`. Native `GateLinear` defaults `bias=False`
(`vllm/model_executor/layers/fused_moe/router/gate_linear.py:49-58`); omitting the
explicit default preserves the existing grouped-sigmoid constructor arguments.
The backend is `vllm_gate_linear`.

Wrapper parameter dtype comes from the model at `moe_wrapper.py:155`, and the
gate is moved to that same dtype at line 218. Its FP32 output-dtype attribute is
not a tensor and is preserved by this move. Input allocation at
`moe_wrapper.py:271-278` uses the model dtype on the requested L rows. Dummy
weight initialization at `moe_wrapper.py:245-250` is retained. V4 therefore
uses BF16 inputs and weights with FP32 logits. No new wrapper dtype conversion
is needed.

The producer calls the native gate directly (`moe_impl.py:219-223`), as native
`runner/moe_runner.py:863` does. With the worker vLLM matching the pinned reference, dispatch uses the same
implementation in `router/gate_linear.py:183-232`:

- lines 183-189: cuteDSL `ll_bf16_gemm` for BF16 inputs and at most 16 rows,
  when construction marked the backend available;
- lines 192-198: the DSV3 specialized GEMM;
- lines 203-210: the FP32-weight specialized router dispatch;
- lines 213-219: opt-in Blackwell BF16x3;
- lines 222-224: BF16 x BF16 to FP32 `torch.mm`;
- lines 227-232: native replicated-linear fallback and FP32 output conversion.

For the V4-Flash BF16 / hidden-size 4096 / 256-expert gate on H200, source
eligibility implies that L=1,2,3,4 can select cuteDSL when available, otherwise
BF16-to-FP32 cuBLAS; L=768 selects cuBLAS. The DSV3 shapes and FP32-weight /
Blackwell tiers do not apply (`gate_linear.py:39-46, 83-113, 123-145`). This is
an inference from the pinned dispatch, not an observed H200 kernel trace.

Remaining qualifications: gate inputs and weights are synthetic; timings are
isolated eager calls rather than the native integrated forward or graph replay.
Worker platform and cuteDSL availability determine the actual tier. The
builder snapshots native sources as references, not a worker overlay
(`prepare_package.py:197-199`); the worker image still needs source/API
qualification against the pin. Native
balanced routing uses an EP-rank prefix offset (`balanced_routing.py:51-70`),
whereas the isolated gate uses offset 0 (`moe_impl.py:232-235`). Only the IDs
change; the gathered routed sample uses native offset-0 global IDs. No gate
implementation or dtype mismatch remains in this source comparison.

## Routed IDs and every measured consumer

Naive DP/EP prepare requests no indices override
(`prepare_finalize/naive_dp_ep.py:103-104`). Native simulated routing passes that
value through (`router/routing_simulator_router.py:325-340`). The frozen
`frontier/profiling/moe/balanced_routing.py:27-29` returns `torch.long` when it is
None. `moe_impl.py:313-324` is the producer's single dtype policy: balanced is
int64, grouped sigmoid / standard fused / uniform are int32. The policy is not
a CSV field; existing `routing_runtime_path` identifies it.

`load_distribution.py:86-89` now calls frozen `balanced_topk(token_offset=0)`
for the untimed balanced sample. `_prepare_routing_inputs` selects the path's
dtype at `moe_wrapper.py:391`; the shuffler selects it at `moe_impl.py:392`; the
functional path selects it at `moe_vllm_kernel.py:949-952`. The wrapper passes
the selected routing path to both measured scopes. Generated samples carry
that path as an optional internal field; externally precomputed samples without
the field retain the original standard/int32 functional default. All other paths retain
their prior int32 execution dtype. `uniform_topk` retains its exact HEAD body.

| Consumer | Dtype-dependent kernel or argument | Source evidence and native census comparison |
| --- | --- | --- |
| Standalone shuffling and grouped alignment | `vllm::moe::moe_align_block_size_kernel<long>` and `vllm::moe::count_and_sort_expert_tokens_kernel<long>` read int64 routed IDs. | `moe_impl.py:398-404`; native `moe_align_block_size.py:81-107`; `experts/marlin_moe.py:340-346`; integral dispatch at `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu:655-656`, alignment launch at 695, sort launch at 716. The alignment `<long>` name is observed in `evidence/runtime_census/p2_32.txt:228`. The sort kernel is source-required; the census's truncated top-kernel list does not display it. |
| Small alignment alternative | `moe_align_block_size_small_batch_expert_kernel<long, 256>` also depends on the routed-ID dtype. | `moe_align_sum_kernels.cu:658-686`; requires global E <= 64, so this branch is unreachable for the G3h global E=256 tasks and is not the census kernel. |
| Marlin workspace sizing | ID shape determines M and top-k; ID dtype does not affect workspace dtype. | `mxfp4_marlin_experts.py:53-79`; native `experts/marlin_moe.py:670-691`. |
| Both Marlin GEMMs | The GEMMs consume int32 `sorted_token_ids`, `expert_ids` and padded-count tensors plus FP32 top-k weights, not the original routed IDs. Their numeric `...l` template values do not encode routed-ID dtype. | Native `experts/marlin_moe.py:135-160, 202-225`; `moe_align_block_size.py:81-88`; `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu:398-402, 533-538`. The same `marlin_moe_wna16::Marlin<...>` family is observed in `p2_32.txt:223`. |
| SwiGLU clamp and invalid-slot masking | `_swiglu_limit_pad_aware_kernel` has an int64 `topk_ids_ptr` specialization; its displayed name has no `<long>` suffix. | Native `experts/marlin_moe.py:169-185`; `activation.py:187-190, 231-234`; `utils.py:484-513, 553-564`. The source loads each routed ID and then indexes the int32 expert map. Its name is observed in `p2_32.txt:226`. |
| EP-aware reduction | `vllm::moe::moe_sum_vec_kernel<c10::BFloat16, long, 6, true>` receives int64 routed IDs and the int32 expert map. | Adapter `mxfp4_marlin_experts.py:97-112`; native `experts/marlin_moe.py:759-781, 911-921`; `moe_align_sum_kernels.cu:799-802, 819-841`. The exact `<BFloat16, long, 6, true>` name is observed in `p2_32.txt:227`. |
| Reduction alternatives | `moe_sum_vec_dynamic_kernel<scalar_t, long, true>` and `moe_sum_scalar_kernel<scalar_t, long, true>` also consume int64 IDs. | `moe_align_sum_kernels.cu:849-863`; G3h uses top-k 6 and aligned contiguous BF16 hidden rows, so the fixed vector branch above is selected by source conditions. |
| Post-alignment shuffling gather | Uses the int32 sorted token IDs, not routed IDs. | `moe_impl.py:406-408`; no additional routed-ID template dependency. |

Remaining routed-scope qualifications: standalone shuffling keeps its existing
block-size selection and native alignment call with default
`ignore_invalid_experts=False`; grouped Marlin uses its own block-size rule and
`ignore_invalid_experts=True` (`moe_impl.py:390-404` versus native
`experts/marlin_moe.py:326-346`). This auxiliary shuffling row is not a second
native cost to charge outside the grouped envelope. Int32 expert maps and
alignment outputs are intentional native representations, not a remaining
routed-ID mismatch. Native EP dispatch/combine collectives and graph execution
are not part of these isolated local scopes. Zero-local-class launches remain
a source finding; they require the lead's later GPU check.

## Builder handoff

`package_plan.json.source_revision` remains the B12 value
`5083d0369a5f9b86c333b30fce78154a3364225f`. The lead replaces it after committing
B12 and B12b. The builder includes the native model/gate source references,
cuteDSL source and this note in a future package. The unchanged expected grid
has 13 EP16 and 20 EP32 tasks per method, or 66 rows across both methods.
Only `prepare_package.py --check-plan` is authorized here. Building, worker
image qualification, GPU kernel-name checks, graph equivalence, submission and
final acceptance remain with the lead.
