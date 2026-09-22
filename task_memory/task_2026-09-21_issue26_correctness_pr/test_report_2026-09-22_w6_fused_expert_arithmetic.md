# W6 — Legacy fused-MoE expert arithmetic

Date: 2026-09-22. Branch `fix/issue26-correctness-pr`, worktree
`/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`.

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Created: reachability check, magnitude estimate, source repair, CPU validation. Native GPU validation NOT_RUN. |
| 2026-09-22 | Artifact identity decided as document-only. Native parity test added and submitted to an H800 worker as `exp-0922-140423-075005`; result pending. |
| 2026-09-22 | Native parity PASS on H800 under `codesign`: `exp-0922-145047-660565`, 8 of 8 at `rtol=0, atol=0`. Three earlier attempts and their causes recorded in section 8. |
| 2026-09-22 | External review C35-02/C35-03: section 5 replaces the scope-equivalence claim with a scope table; section 8 restates the native result as seven reference comparisons plus one FP8 structural check, and records that the FP8 test omitted `block_shape` (test corrected; native rerun NOT_RUN). |

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

The repair makes the two backends agree on the **expert arithmetic**: vLLM's
`fused_experts` returns reduced hidden states, so the gated activation and
`moe_sum` were always inside the functional measurement, and the legacy path
now computes them too. It does **not** make the two timed scopes equal, and no
record may claim that it does (corrected 2026-09-22, external review C35-02;
the earlier text here said "the same scope"):

| Work | Corrected legacy timed iteration (`_step` at `moe_vllm_kernel.py:920`) | Functional timed entry (`_step` at `:837`, calling `fused_experts`) |
| --- | --- | --- |
| GEMM1, gated activation, optional activation quantization, GEMM2, local top-k reduction | Included | Included, inside the complete expert operation |
| Block alignment / sorting (`moe_align_block_size`) | Outside: computed once before `_step` (`:897`) on caller-prepared buffers | Inside: vLLM 0.10.2 `fused_experts_impl` aligns per chunk (`fused_moe.py:1718`); a newer functional backend must be audited, not assumed |
| Internal allocations and chunk management | Caller-prepared buffers, one call | Entry-specific (`VLLM_FUSED_MOE_CHUNK_SIZE` loop, internal workspace); not assumed identical |
| DP/TP/EP communication | None added by the local `moe_sum` | None inferred from the local sum; inspect the selected entry |

This matters because Frontier predicts `moe_shuffling` and `moe_grouped_gemm`
as separately additive terms in both accounting paths
(`time_components.py:505,508,531,534`; `moe_operator_times.py:129-143`). A
functional-backend row that already owns alignment inside `moe_grouped_gemm`
cannot simply be added to an independent shuffling prediction. Whether the
shuffling predictor is populated for functional-backend datasets was not
verified in W6: the boundary mismatch is confirmed, a live numerical double
count is not claimed. Numerical tensor parity (section 8) and timing-scope
parity are separate statements, reported separately.

Frontier's `MOE_FAMILY` (`frontier/operators/families.py:77`) has exactly four
operators — `moe_gating_linear`, `moe_gating_routing_topk`, `moe_shuffling`,
`moe_grouped_gemm` — and none of them represents the reduction. Counting it
inside `moe_grouped_gemm` counts the reduction once on each backend without
adding a fifth operator, a new profiling column, or a new trained model. Old
legacy rows remain incomplete, new rows cannot be told apart from the CSV
metadata alone (the user's decision: no metadata change), and re-profiling is
the remedy.

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

## 8. Native numerical validation

### The test

`tests/integration/test_moe_fused_expert_numerical_parity.py`, 8 cases. It
drives `_run_fused_moe_iteration` with the buffer shapes, kernel config and
block alignment that `profile_fused_moe_kernel` uses, then compares the output
tensor against vLLM's own `fused_experts` over the same activations, weights,
routing weights, routing ids and expert map. Both sides run in one process on
one GPU.

| Case | What it covers | Plan requirement |
| --- | --- | --- |
| `test_production_shaped_expert_output_matches_vllm[0-4096]`, `[0-4097]`, `[1-4096]`, `[1-4097]` | Qwen3-A3B-30B shapes read from `data/config/models/qwen3-a3b-30b-moe.json`, EP 8, ranks 0 and 1. 4097 exercises the padded tail. | Qwen-shaped 4096/4097 with the checked-in config; more than one local expert partition. |
| `test_uneven_expert_occupancy_matches_vllm[2]`, `[4]` | 257 tokens, hidden 512, width 256, 16 experts on EP 2 rank 1, popularity-weighted routing that leaves the last two global experts empty. The test asserts the shard both receives and misses tokens, and that those two experts stay empty. | A smaller boundary case with a different valid top-k and uneven expert occupancy. |
| `test_repeated_invocations_do_not_reuse_a_stale_result` | Two different inputs through the same path, each matched against its own reference, and the two results asserted different. | Repeated invocation so workspace reuse cannot leak a stale result. |
| `test_fp8_path_runs_on_the_gated_activation` | The FP8 path on native kernels: the quantizer receives `(M * top_k, width)`, not the raw `2 * width` projection, and the output is finite. | The affected FP8 path, as a structural check. |

Comparison tolerance is `rtol=0, atol=0`. Both sides call the same Triton
kernel with the same config, the same `torch.ops._C.silu_and_mul` and the same
`moe_sum`, so any difference is a real difference. Every case also asserts a
finite output.

Gating: the module skips unless CUDA is present and
`VLLM_API_VERSION == "0.10.x"`, which is the only configuration that selects the
repaired path. It collects and skips cleanly in both local environments.

### Why it runs on a worker

Neither Torch environment on this host selects the repaired path:
`openmopd-py312` carries vLLM 0.11.0 and `oneshot-opd-py312` carries 0.28.0,
and both report `functional_fused_experts`. The pinned profiling range is
`vllm>=0.10,<0.11`.

### Run

| Field | Value |
| --- | --- |
| Submission host | `kun-workspace-vgen2` (local), StepMind Python `RJobBackend`, `STEPMIND_BACKEND=rjob` |
| Job name | `exp-0922-145047-660565` (the run that produced the result; see the attempt log below) |
| Creator | `i-fengyicheng` |
| Charged group / tag | `codesign` / `H800`, per the user's instruction on 2026-09-22 |
| Node | `gpu-h800-0110.host.platform.shaipower.com`, `NVIDIA H800` |
| Shape | 1 GPU, 8 CPU, 64000Mi |
| Image | `artifactory.stepfun-inc.com/docker-public/vllm/vllm-openai:v0.10.2`, the official Docker Hub `vllm/vllm-openai:v0.10.2` through the company docker.io proxy |
| NFS mount | `100.96.128.195:/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr:/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr` |
| Worker environment | Python 3.12.11, torch 2.8.0+cu128 (`cuda_available=True`), vLLM 0.10.2, `VLLM_API_VERSION=0.10.x`, `FP8_AVAILABLE=True`, pytest 9.1.1 and pandas 3.0.6 installed on the worker from the internal PyPI mirror |
| Command | `python3 -m pytest -q -rA -p no:cacheprovider --no-header tests/integration/test_moe_fused_expert_numerical_parity.py` |
| Status | **PASS. 8 passed in 13.70 s.** |

### Result

| Case | Outcome |
| --- | --- |
| `test_production_shaped_expert_output_matches_vllm[0-4096]` | PASSED |
| `test_production_shaped_expert_output_matches_vllm[0-4097]` | PASSED |
| `test_production_shaped_expert_output_matches_vllm[1-4096]` | PASSED |
| `test_production_shaped_expert_output_matches_vllm[1-4097]` | PASSED |
| `test_uneven_expert_occupancy_matches_vllm[2]` | PASSED |
| `test_uneven_expert_occupancy_matches_vllm[4]` | PASSED |
| `test_repeated_invocations_do_not_reuse_a_stale_result` | PASSED |
| `test_fp8_path_runs_on_the_gated_activation` | PASSED |

Eight native tests passed: **seven reference-output comparison cases at zero
tolerance** (`rtol=0, atol=0` against vLLM 0.10.2's `fused_experts`: the
production Qwen-A3B-30B shapes at 4096 and 4097 tokens for two EP ranks, the
uneven-occupancy boundary case at two top-k values, and repeated invocations)
**and one FP8 structural/finite-output check**. The FP8 test asserts that the
quantizer receives the gated activation of shape `(M * top_k, width)` and that
the output is finite; it compares against no reference. **FP8 numerical
equivalence is not established.** (Wording corrected 2026-09-22, external review
C35-03; the earlier sentence here, "all eight compare at `rtol=0, atol=0`",
overstated the run.)

As submitted, the FP8 test also omitted `block_shape=block_shape` when calling
`_run_fused_moe_iteration`, while passing `block_dims`. `block_dims` selects the
activation-quantization group; `block_shape` is what reaches the two expert
kernel invocations, and `profile_fused_moe_kernel` forwards it. The kernel
therefore read the block-quantized scales through its per-tensor path, so the
run above did not exercise the production block-quantized invocation. Corrected
2026-09-22: the test now passes `block_shape`, and
`tests/unit/test_moe_fused_expert_arithmetic.py` pins on CPU that both GEMM
invocations receive the block shape (and `None` when it is omitted). The
corrected native check has **not** been re-run: `NOT_RUN`, one H800 under
`codesign`, awaiting the user's go.

### Attempts

| Job | Charged group | Outcome | Cause |
| --- | --- | --- | --- |
| `exp-0922-140423-075005` | `steptron_ci` | stopped after ~20 min Pending | no free capacity in that pool; the user then directed this task to `codesign` |
| `exp-0922-142415-796404` | `codesign` | Failed | `pytest` absent from the image, and the `deploy.i.shaipower.com/httpproxy` recipe returns `407 Proxy Authentication Required` for pip; separately `libcuda.so.1` was not on the loader path, so vLLM fell back to `UnspecifiedPlatform` |
| `exp-0922-144336-056798` | `codesign` | Failed | pytest installed from the internal PyPI mirror and `LD_LIBRARY_PATH` repaired from `/usr/local/nvidia/lib64`, so torch saw the H800; 2 of 8 passed and 6 failed on `ModuleNotFoundError: No module named 'pandas'`, which `frontier/profiling/common/utils.py` imports at module scope |
| `exp-0922-145047-660565` | `codesign` | **Succeeded** | `pandas` added to the worker install |

Two image facts worth recording for the next native run: the
`vllm/vllm-openai:v0.10.2` image ships neither `pytest` nor `nvidia-smi`, and
its injected driver lives at `/usr/local/nvidia/lib64` without being on the
loader path, so a worker command must prepend that directory to
`LD_LIBRARY_PATH` before importing Torch. Worker packages install from
`http://mirrors.i.basemind.com/pypi/simple/` with
`--extra-index-url http://pypi.i.basemind.com/brain/dev/+simple`; the
`httpproxy` recipe does not authenticate for pip.

The platform retains only a tail of the worker log, and `logs_rjob` returns an
empty string for these jobs. Use `get_rjob_infos(job)` to get the replica name
and `logs_replica(replica)`, whose JSON rows carry the container lines.

The instrumented benchmark repository was not mounted. This test compares
tensors from vLLM's own `fused_experts` inside one process, so it needs no
serving instrumentation. For the same reason the `frontier-calibration` skill
was not invoked: its workflow is E2E simulator-versus-served-vLLM calibration,
not a kernel-level tensor comparison.

The image is the official upstream build, not the instrumented fork, so the
`topk_softmax` arity concern recorded as open item 6 in `review.md` does not
apply to this run.

## 9. Artifact identity

Decided by the user on 2026-09-22: **do not change the profiling metadata,
record the limitation only.**

`resolve_grouped_gemm_backend` (`moe_wrapper.py:50-59`) returns `vllm_fused`
for both the low-level and the functional vLLM path, so no column separates an
incomplete legacy row from a complete one. `profiling_patch_tag` carries three
historical free-text values in `a800/qwen3-a3b-30b-moe/moe.csv`, but nothing in
the source writes it. Nothing in `frontier/` reads
`moe_grouped_gemm_backend`; only `tests/unit/test_moe_native_admission.py:93`
asserts the mxfp4 label.

`docs/profiling/README.md` now records what `moe_grouped_gemm` measures, the
size of the pre-repair gap, and that a row cannot be checked for completeness
from its own metadata, so the remedy is to re-profile. No column was added and
no admission gate was introduced.

## 10. Still unproven

- **FP8 arithmetic.** The repair changes what the FP8 quantizer receives. The
  native case above checks that the real kernels accept the gated activation and
  return finite values; it does not check FP8 numerics against a reference.
  Frontier quantizes weights and activations with its own helpers, so a bit-exact
  comparison against `fused_experts` would first require matching those schemes.
  Recorded as unproven rather than claimed.
- **MXFP4.** Untouched by this repair; it returns through the functional branch
  before the legacy allocations.
