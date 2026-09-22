# W6 — Legacy fused-MoE expert arithmetic

Date: 2026-09-22. Branch `fix/issue26-correctness-pr`, worktree
`/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`.

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created: reachability check, magnitude estimate, source repair, CPU validation. Native GPU validation NOT_RUN. |

## 1. Is the defect reachable

Yes, on the supported profiling environment, and only there.

`frontier/profiling/moe/moe_vllm_kernel.py` selects one of two expert paths at
import time. When vLLM exposes the low-level API (`fused_moe_kernel`,
`invoke_fused_moe_kernel`, `moe_align_block_size`, `try_get_optimal_moe_config`,
`get_config_dtype_str`) it sets `VLLM_API_VERSION = "0.10.x"` and profiles
through Frontier's own `_run_fused_moe_iteration`. Otherwise it falls back to
`VLLM_API_VERSION = "functional_fused_experts"` and calls vLLM's complete
`fused_experts`, which is correct by construction.

All five names resolve in the pinned reference vLLM v0.10.2
(`/data/ycfeng/Frontier/.real-engine/vLLM-BS` @ `ea95f571`, verified by
`grep` against `vllm/model_executor/layers/fused_moe/fused_moe.py`), and
`environment_profiling.yml:25` pins `vllm>=0.10,<0.11`. The documented profiling
environment therefore takes the legacy path.

Observed on this host, for the record: neither locally available Torch
environment reproduces it. `openmopd-py312` carries vLLM 0.11.0 and
`oneshot-opd-py312` carries vLLM 0.28.0; both report
`VLLM_API_VERSION = functional_fused_experts`. No environment on this machine
selects the legacy path, which is why native validation cannot run here.

## 2. What was wrong

vLLM's own `fused_experts_impl` runs:

```text
GEMM1 -> gated activation -> (activation quantization) -> GEMM2 with routing weights -> local top-k reduction
```

`_run_fused_moe_iteration` ran GEMM1, then took `intermediate_cache1[:, :E]` —
the gate half of the `gate | up` projection, unactivated — as the operand of
GEMM2, and never reduced the per-expert outputs. Two operations were missing:
`silu_and_mul` and `moe_sum`.

The second GEMM's `mul_routed_weight=True, top_k=1` already matched the
reference, as did the FP8 activation quantization; only the operand was wrong.

## 3. Magnitude, estimated before implementing

The two missing kernels are memory-bound elementwise work. Their cost was
estimated as `bytes / (peak HBM x 0.80)` and compared against the measured
`moe_grouped_gemm` median in the checked-in datasets:

```text
silu_and_mul : read 2E, write E, per routed row  -> 2 * 3E * M * top_k bytes
moe_sum      : read top_k * H, write H, per token -> 2 * H * M * (top_k + 1) bytes
```

| Dataset | shape | at the largest profiled token count | over all rows (median / max) |
| --- | --- | --- | --- |
| `a800/qwen3-a3b-30b-moe` | H=2048, E=768, top_k=8 | 4096 tokens: measured 0.934 ms, estimated missing 0.185 ms → **16.5%** | 6.8% / 26.3% |
| `h800/Qwen3-30B-A3B-tiny` | H=2048, E=768, top_k=8 | 64 tokens: 0.130 ms, 0.0018 ms → 1.4% | 0.2% / 1.4% |
| `h800/step-moe-noquant-small` | H=7168, E=5120, top_k=3 | 64 tokens: 0.337 ms, 0.0036 ms → 1.1% | 0.1% / 2.2% |

The error scales with token count: negligible at decode batch sizes,
material at prefill and chunked-prefill sizes. The two tiny h800 datasets only
reach 64 tokens, which is why their share stays small.

This is an analytical estimate with a stated bandwidth assumption, not a
measurement. It was used to decide whether the repair was worth making, not to
claim a corrected number.

## 4. The repair

`frontier/profiling/moe/moe_vllm_kernel.py`:

- The low-level import block also imports `vllm._custom_ops` as
  `_vllm_custom_ops`, so a build without it selects the functional path instead
  of running an incomplete computation. `_vllm_custom_ops = None` is declared at
  module level beside the existing `_functional_fused_experts = None`.
- `_run_fused_moe_iteration` now runs `torch.ops._C.silu_and_mul` into a
  dedicated activation buffer, feeds that buffer to GEMM2 (and to the FP8
  quantizer when enabled), and finishes with
  `_vllm_custom_ops.moe_sum(intermediate_cache3, out_hidden_states)`.
- `profile_fused_moe_kernel` allocates the two additional buffers once, outside
  the timed step: `intermediate_cache2` as `(M * top_k, E)` and
  `out_hidden_states` as `(M, H)`. The former GEMM2 output buffer is renamed
  `intermediate_cache3`, matching the reference's naming.

**No gated/non-gated branch was added.** `profile_fused_moe_kernel` always
materializes `w1` with `2 * expert_hidden_dim_per_partition` rows
(`moe_vllm_kernel.py`, weight construction), and `plan_mxfp4_weight_layout` is
called with `use_gated=True`. The gated layout is the only one this entry point
can produce, so a conditional would be unreachable code. Profiling a non-gated
expert would first require a different weight layout, which this function does
not support today.

## 5. Measurement ownership

The corrected `moe_grouped_gemm` target is the **complete local expert
computation**: GEMM1, gated activation, optional activation quantization, GEMM2,
and the local top-k reduction. Both profile methods time the whole step —
`_collect_cuda_event_stats` brackets `step_fn`, and
`_collect_record_function_stats` wraps it in one `vidur_moe_grouped_gemm` scope.

This makes the legacy path measure the same scope the functional path already
measured: vLLM's `fused_experts` returns reduced hidden states, so `moe_sum` was
always inside the functional measurement. The repair removes a disagreement
between the two backends rather than creating one.

Frontier's `MOE_FAMILY` (`frontier/operators/families.py:77`) has exactly four
operators — `moe_gating_linear`, `moe_gating_routing_topk`, `moe_shuffling`,
`moe_grouped_gemm` — and none of them represents the reduction. Counting it
inside `moe_grouped_gemm` counts it exactly once without adding a fifth
operator, a new profiling column, or a new trained model.

Noted for the record, because it differs: the Frontier-instrumented reference
vLLM puts `moe_sum` *outside* its `record_function("moe_grouped_gemm")` scope
(added by reference commit `139cba30b`). That split is finer-grained
diagnostics; Frontier's operator of the same name means the whole local expert
computation.

The local reduction is a sum over one token's own top-k expert outputs. It is
not a collective, and no DP/TP/EP communication cost was added or removed.

## 6. CPU validation

Environment: `/data/ycfeng/envs/openmopd-py312/bin/python` (Python 3.12, Torch
2.8.0+cu128, vLLM 0.11.0), `PYTHONPATH=$PWD`, from the worktree root. This
environment is used because the profiling-boundary tests import Torch; no GPU
kernel executes.

New file `tests/unit/test_moe_fused_expert_arithmetic.py`, 7 tests, all passing.
It replaces `_invoke_kernel`, `torch.ops._C.silu_and_mul` and
`_vllm_custom_ops.moe_sum` with plain-Torch references, so it validates the
composition, not native numerics. The file's docstring says so.

| Test | What it settles |
| --- | --- |
| `test_the_iteration_computes_the_gated_expert_output` | The full path equals a directly written gated-SwiGLU MoE reference. |
| `test_a_gated_activation_is_not_the_first_half_of_the_projection` | The repair is observable: the old slice-only arithmetic gives a different answer, so the test above cannot pass against the unrepaired path. |
| `test_the_second_gemm_consumes_the_activation_buffer` | Call order `GEMM1 -> silu_and_mul -> GEMM2 -> moe_sum`; GEMM1 has `mul_routed_weight=False, top_k=top_k`; the activation reads `(M*top_k, 2E)` and writes the buffer GEMM2 reads; GEMM2 has `mul_routed_weight=True, top_k=1`; the reduction reads `cache3` and writes `out`. |
| `test_the_local_reduction_sums_the_top_k_expert_outputs` | The reduction is local and per token; output shapes `(M, top_k, H) -> (M, H)`. |
| `test_the_activation_buffer_is_quantized_rather_than_the_raw_projection` | Under FP8 the quantizer receives the gated activation, with the block-derived group size. |
| `test_repeated_iterations_do_not_leak_a_previous_result` | Workspace reuse across profiling steps does not carry a stale output; the second result is finite and matches its own reference. |
| `test_the_profiler_allocates_the_four_buffers_the_computation_needs` | Allocation-site dimensions for all four buffers under TP=2, and one workspace shared by every profiled step. |

### Regression comparison

Same command, same environment, against a detached worktree at `HEAD`
(`bbbfcaa`), removed afterwards:

| Selection | At `HEAD` | With the repair |
| --- | --- | --- |
| `test_moe_mxfp4_increment10.py`, `test_moe_fused_event_contract.py`, `test_moe_native_admission.py`, `test_native_profiling_model_type_policy.py` | 1 failed, 136 passed | 1 failed, 143 passed (7 new) |

The single failure is the same on both sides:
`test_functional_vllm_kernel_exposes_mxfp4_switch_without_importing_vllm`
asserts `check_vllm_available() is False`, which needs an environment with Torch
but *without* vLLM. Neither local environment provides that combination. It is
environment-dependent, not a regression.

Default environment (`/data/ycfeng/envs/frontier-py310/bin/python`, no Torch),
`pytest tests/unit --continue-on-collection-errors`: **84 failed, 3778 passed,
49 skipped, 11 errors**, matching the W4 baseline of 84 failures and 3778
passing. Collection errors go from 10 to 11: the new file imports Torch at
module level, exactly as the seven existing profiling test files in that list do
(`test_moe_fused_event_contract.py`, `test_moe_native_admission.py`,
`test_native_profiling_model_type_policy.py`, `test_mla_native_profiling_wrapper.py`,
`test_moe_gating_constructor_boundary.py`, `test_moe_load_distribution_contract.py`,
`test_moe_routing_input_contract.py`). Verified by re-collecting with the new
file ignored: 10 errors.

## 7. Simulator fidelity

No fidelity matrix run. The change is confined to
`frontier/profiling/moe/moe_vllm_kernel.py`, which the simulator cannot import:
it requires Torch, and the simulator runs in an environment without it. The
evidence is the unchanged default-environment suite result above — the
simulator's own 3778 passing tests are byte-identical in outcome, and the
matrix's inputs are checked-in CSVs, not freshly profiled data.

Stated plainly: this repair changes what a *future* profiling run measures. It
changes nothing about a simulation run from existing data.

## 8. Not done

- **Native numerical validation: NOT_RUN.** The plan requires a GPU local-expert
  parity test against the pinned vLLM reference, including more than one local
  expert partition and a boundary case with uneven expert occupancy. It needs a
  worker with `vllm>=0.10,<0.11`, because that is the only configuration that
  selects the repaired path. No such environment exists on this host. Pending a
  decision on the GPU run.
- **Artifact identity: OPEN, raised for decision.** Existing rows record
  `moe_grouped_gemm_backend='vllm_fused'` for both the legacy and the functional
  path (`resolve_grouped_gemm_backend`, `moe_wrapper.py:50-59`), so the column
  cannot distinguish an incomplete legacy measurement from a complete one.
  `profiling_patch_tag` exists in one checked-in CSV but appears nowhere in the
  source, so it is not a live mechanism. No consumer in `frontier/` reads
  `moe_grouped_gemm_backend`; only `tests/unit/test_moe_native_admission.py:93`
  asserts the mxfp4 label. Options and a recommendation are in `review.md`.
- **FP8 path**: the repair changes what the FP8 quantizer receives, so the FP8
  path is an affected native path. It is not proven by the CPU shape test and is
  included in the pending GPU matrix.
