## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Read-only source review of linear profiling context seams and bounded repair options. |
| 2026-09-08 | Confirmed and selected the existing-input-path diagnostic option for isolated first-forward sensitivity; no shared contract is needed for that experiment. |

# D019 linear profiling context proposal

Status: READ-ONLY PROPOSAL. No production code or profiling CSV was changed by this review. The current 4096/1024 case, eager CUDA_EVENT boundary, H200 step_main, and ideal communication contract remain in force.

## Finding and recommended next step

The existing linear profiler has no runtime-context dimension. The existing MoE `prefill_hot` contract provides a useful design pattern, but it is not an existing switch for linear operations. Keep the measured hot linear samples as diagnostic evidence until the independent kernel/instrumentation review establishes which execution context should be represented. A task-local, separate candidate CSV and bounded first-forward query can quantify the consequence within D019; it must not become the formal replacement merely because its values approach vLLM.

Parent-reported controlled experiment: the same wrapper produces approximately QKV/RoPE/output-projection = 78/19.2/29.8 us with the hot prefix, versus 130/32/47 us in its synthetic context; vLLM is approximately 80.6/19.6/30.3 us. Preconstructing Event objects changes only about 2–5%. These results demonstrate sensitivity to surrounding work. They do not independently establish that a fixed 20-repeat synthetic FFN prefix represents all real prefill operations, or separate device clocks, queue occupancy and cache state. Kernel identity and instrumentation perturbation remain assigned to the independent review.

## Inspected source seams

| Seam | Existing behavior | Consequence for a correction |
| --- | --- | --- |
| `frontier/moe_gating_runtime.py` | Defines `gating_runtime_context`, `gating_runtime_context_impl`, `standalone_legacy`, `prefill_hot`, and exact implementation identity `ffn_like_prefix_20x`; filters rows by context; uses a `__prefill_hot` model suffix. Runtime selection is limited to the existing Qwen3 MoE capability check and any batch with prefill tokens, including mixed batches. | Reuse the separation of implementation identity, row selection and runtime selection. Do not copy its model-name fallback or treat its gating-specific columns as an already-general linear schema. |
| `frontier/profiling/moe/moe_wrapper.py:185–235, 376–419` | Allocates FFN-shaped prefix weights from model dimensions. Every hot sample first runs 20 up/activation/down iterations, then invokes gating on the original hidden states. The prefix return is discarded; it changes execution context rather than the profiled operator's inputs. Prefix kernels are outside gating's operator timers. | The prefix is a conditioning procedure, not another simulated model layer or CPU-overhead charge. Reuse its implementation only through an explicit shared helper if a formal linear producer adopts it. Calling MoE private methods on a synthetic object is appropriate for the bounded diagnostic, not the production interface. |
| `frontier/profiling/linear_op/linear_op_wrapper.py:40–88, 118–180, 181–249` | Constructor has no context argument. CUDA-event profiling uses one synthetic GPT layer, three warmups, synchronization, then 20 active samples. Result metadata records shapes, architecture and typed operator contracts, but no execution context. | Add producer conditioning and context/implementation metadata here only after deciding their semantics. Preserve operator timers and original input tensors. |
| `frontier/profiling/linear_op/main.py` | Owns public profiling arguments and propagation through local/Ray wrapper creation. | A public context option must reach all existing execution paths; a wrapper-only argument leaves the supported CLI incomplete. |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py:1169, 3215` | Loads linear CSV by model/TP/typed contract, trains linear models with `num_tokens` only. There is no phase/context filter. | Combining synthetic and hot rows at the same key would mix their targets, including exact-lookup aggregation. A distinct context must be filtered before training and exact metadata generation, and selected when querying a batch. Changing only a 4096 row does not give it prefill semantics. |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py:1856` | Separately trains input norm and attention projections from the same linear CSV. MoE context handling exists elsewhere in this file, not in the linear path. | Independent and shared training must use the same selection helper. Updating only on-demand training leaves a supported path inconsistent. |
| `frontier/training/attention_trainer.py`, `frontier/training/linear_op_trainer.py` | Public standalone trainers also consume linear compute rows. | A production context contract must cover these consumers or explicitly reject unsupported mixed-context inputs. They cannot silently aggregate contexts. |

The existing on-demand model cache key already includes model name, selected dataframe content and measurement family (`_get_model_hash`, line 2948). Separate selected slices/model identities can use that mechanism. This investigation does not justify a new fingerprint layer.

## Why profile_method/CUDA Graph is not an equivalent shortcut

`frontier/profiling/utils/__init__.py` normalizes `cuda`/`cuda_event` to CUDA_EVENT, and `kernel_only`/`record_function` to KERNEL_ONLY. The latter changes the measured quantity to kernel durations; it is not an alternate hot-context CUDA-event mode. The linear wrapper's record-function path profiles a multi-layer synthetic model; despite its historical comment about a captured graph, this method contains no CUDA Graph capture/replay call.

For co-location, `_select_measurement_type_for_batch` in the base predictor returns CUDA_EVENT whenever a batch has prefill tokens. KERNEL_ONLY is selected for eligible decode graph execution. Enabling a decode graph flag therefore does not solve this prefill context mismatch, and relabeling kernel-only numbers as CUDA_EVENT would violate the current comparison boundary. Kernel-only profiling can provide independent diagnostic evidence, but cannot replace the accepted eager prefill metric.

## Bounded options and scope

1. **Continue the existing diagnostic and, if useful, evaluate a separate candidate dataset.** Keep each context's raw samples and producer receipt separate. Use the existing linear input-file override and a fresh predictor cache with the existing bounded first-forward audit. Preserve the actual 4096 scenario and all unrelated targets; verify the selected exact rows and changed contributions. This requires no production interface change and can remain within D019's approved measurement/correction investigation. Its result is a sensitivity/candidate result, not general context support or formal CUDA closure. Do not include prefix time in simulated work. Candidate serialization, if needed, belongs in the existing task diagnostic under `tests/`, with a manifest stating its intentionally limited first-forward use.

2. **Introduce explicit linear prefill context as a supported production feature, after review.** The minimum runtime implementation touches at least five production files: a shared context-definition/filter helper, linear wrapper, linear CLI, base predictor, and shared manager. Supporting existing public standalone training makes at least seven; extracting and sharing the existing MoE prefix adds the MoE wrapper as another adapter. Tests and documentation are additional. This changes producer/consumer metadata and batch-selection contracts, and reaches the 8,315-line base predictor and 4,606-line shared manager, so the project's cleanup/split requirements must be considered. This is not a one-file local timer repair. Use a centralized declarative operation/capability mapping; do not spread per-model or per-op name conditions through callers.

Do not globally replace the linear producer's context by silently inserting the prefix. That would also change decode and other model profiling without an established applicability rule. Do not silently blend old and hot rows. A supported production decision must say whether all linear operations or only reviewed projections are eligible, how pure prefill and local mixed batches select context, how decode retains its existing context, and what missing-context data means. Those choices have materially different runtime behavior and are not settled by the current one-case measurement.

Recommendation: finish the independent kernel/context review and present its evidence together with option 2's concrete contract to YC if formal adoption is needed. The existing D019 authorization permits option 1 and ordinary evidence-backed operator fixes; it does not resolve these new shared-contract choices. Ask a focused grill decision before implementing the production context feature. No new GPU run, model-selection branch, CSV replacement or timing-family change is required merely to complete this source review.

### Follow-up: the bounded option is sufficient now

The parent selected option 1, and actual source/input inspection confirms it is feasible without introducing the general feature. The frozen case has `batch_size_cap=1024`, no decode CUDA Graph, and the observed first-forward linear feature is M4096/TP4. Ordinary single-token decode cannot produce M4096 with that cap; the hard stop additionally ensures only the observed pure-prefill first forward is evaluated. This is a case constraint, not a universal inference that any M4096 row means prefill.

The explicit current-task linear CSV contains 82 rows and exactly one M4096/TP4 row. Its three projection typed contracts exactly equal the profiles04 producer receipt. Export all 40 existing event-only/original-timer/prefill-hot samples per reviewed op with the fixed pooled-median rule and ordinary population standard deviation; retain all other rows, columns and typed metadata. The context and raw source are bound in a candidate receipt, without adding a shared CSV selector. Existing content-derived cache keys and the actual query audit establish which rows are selected. This bounded experiment is the recommended immediate step; formal shared-context support can be considered separately if its general runtime behavior is needed.

## Verification performed and limits

Read-only `rg`/`sed` inspection covered all `runtime_context` occurrences under `frontier/`, the producer loops, both runtime trainers, public standalone trainer entry points, cache identity, and measurement-family dispatch. The search found only MoE runtime-context support. No simulation or GPU measurement was run for this proposal; numeric experiment values above are explicitly parent-reported, while interface and control-flow findings are directly inspected. No source file was modified.
