## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Added independent source review of every changed production profiling hunk, caller boundaries, W05/W07 findings, and targeted acceptance gaps. |

# W00 Independent Profiling Source Review

- Reviewer: `/root/w00_review_inventory`.
- Inspected candidate: `7b59d1fd7ad41f1301625f0b7c54578b5762f486`; no production edits were present when inspection began.
- Pinned comparison: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Scope: changed production Python files under `frontier/profiling`, their immediate real callers and relevant existing tests. The changed `gdn/README.md` is documentation, excluded from the production hunk ledger.
- Authority: `Frontier_PR33_New_Execution_Plan_2026-09-16_EN.md`, especially W00, W05, W07 and N05/N06/N09, plus supplied workspace instructions and the checked-in AGENTS.md.
- Implementation: source-reviewed only; no code changed by this reviewer.
- Verification: no pytest, Simulator, native GPU, benchmark, platform job, or remote publication executed. All conclusions below are source findings or explicit verification gaps; RETAIN is not native PASS.
- Readability/ownership: bounded fixes and existing ownership seams identified below. Source cleanup remains outstanding for BUGFIX/REFACTOR rows.

## Findings requiring implementation or focused reproduction

### P01 — N05: campaign preflight occurs after native initialization

**Source-confirmed.** `gdn/main.py:76` imports and constructs `VllmQwen35GDNWrapper` before calling `build_profile_inputs(args)` at line 94. The optional unsupported mixed input is appended after valid inputs. An invalid later descriptor can follow native construction and multiple successful measurements. `profile_iterations`/`warmup_iterations` are checked only in `profile()`. Standard writer DEVICE_EVENT admission is checked after all profiles and dataframe construction. A RECORD_FUNCTION campaign reaches native work before eventual writer rejection.

**Caller boundary:** CLI `_parse_args -> main -> build_profile_inputs -> wrapper.profile -> output dataframe`. Existing `test_mixed_input_is_rejected_before_gpu_wrapper_work` bypasses the constructor with `object.__new__`; it proves only the per-input phase guard, not the campaign gate.

**Bounded correction:** construct the whole descriptor list first; validate phase, positive configured capacities, logical/physical capacity, context + query limit, all-zero or all-positive carried-state policy, exact iteration types/counts, and standard measurement method before importing the native producer. Reuse `GDNProfileInput` and normalized method helpers. Preserve one-token continuation as prefill.

**Additional source facts:** `_build_metadata` checks logical batch capacity but not physical capacity; `max_model_len` is passed to vLLM but not retained for per-workload validation. Physical batch metadata can report padding even though hidden states/metadata execute only the logical requests. Either validate the producer's actually supported physical representation or implement real padded execution only within authorized scope; do not emit invented physical work.

**Targeted evidence:** invoke real `main()` with a guard producer that fails if imported/constructed and an existing output sentinel. Late mixed descriptor, invalid counts, oversized logical/physical batch, context + query overflow, and wrong writer method must reject with zero constructor/profile calls and unchanged output. Then valid cold/continuation/decode campaigns must reach the producer.

### P02 — N06: unnamed standalone timers poison the global owner

**Source-confirmed.** `DeviceTimer.__init__` reads `Singleton._instances` directly. With no store, `DeviceTimer(None)` creates `TimerStatsStore(..., disabled=True)`; a subsequent named timer reuses it and stays disabled. `Singleton.__call__` ignores later constructor arguments. Explicit `profile_method` also silently loses to any existing store, whether or not the caller intended that precedence.

**Caller boundary:** attention `BaseAttentionWrapper.get_timer`, common layernorm/rotary and linear-op timers, MoE wrappers, collectives wrapper, and GDN wrapper all share `TimerStatsStore`. Most wrappers initialize the store first, so the standalone failure is conditional, not proof every profiling run is disabled.

**Bounded correction:** narrow existing-store accessor or explicit owner injection, a documented conflict/precedence contract, and no global-store creation for a disabled unnamed standalone timer. Preserve CudaTimer's import boundary and all five methods. Add exception-path checks so record-function/profiler contexts close correctly. No production inspection of metaclass internals outside the store owner.

**Targeted evidence:** first unnamed then named; explicit disabled/enabled existing stores; matching/conflicting requested method; nested and exceptional exit; CUDA_EVENT, DEVICE_EVENT, PERF_COUNTER, RECORD_FUNCTION, KINETO. Existing tests cover mocked device-event and reuse of a KINETO store only.

### P03 — N09: absent active timing samples become fabricated zeros

**Source-confirmed.** `gdn/vllm_wrapper.py:561-568` reads whatever samples exist and fills every absent family operator with `_zero_stats(profile_iterations)`. Neither active operator presence/count nor diagnostic `gdn_layer_e2e` samples are required. A disabled or broken timer can therefore produce structurally complete all-zero operator records.

**Caller boundary:** timed `_run_e2e` and `_run_decomposed -> TimerStatsStore.get_times -> rank aggregation -> statistics -> canonical gdn.csv -> training`. The existing GDN family operator specs already declare phases; use them as active/inactive ownership truth.

**Bounded correction:** before aggregation/export require the three active physical operator names plus diagnostic E2E, each with exactly `profile_iterations` finite, nonnegative samples. Fill only the inactive core phase. Permit measured zero samples; missing samples are the defect. TP ranks must agree on sample-key/count schema before numeric reduction; same-sized but differently named rank matrices must not silently aggregate unrelated operators.

**Targeted evidence:** absent active op, absent E2E, empty store, inconsistent counts, negative/NaN/Inf samples, and valid measured zero. Cover inactive-phase zero records separately and include an asymmetric rank schema failure test where practical.

### P04 — native ownership and metadata admission remain incomplete

**Source-confirmed ownership gap; native impact unexecuted.** GDN construction can initialize distributed/model-parallel state before later layer/cache/layout failure, but `with Wrapper(...)` never calls `__exit__` when `__init__` raises. `close()` destroys model parallel only under `_owns_distributed`; if distributed exists externally but this wrapper initializes model parallel, the owned model-parallel resource leaks. Conversely, independently initialized resources require independent ownership accounting.

`model_dtype` reports `frontier_model_config.dtype` while the native producer, hidden states, and synthetic weights explicitly use BF16. The wrapper checks state shapes/bytes, but does not establish the full reported model dtype/shape identity against the instantiated layer/checkpoint. Direct construction defaults to `cuda_event` even though the implementation explicitly requires ROCm and subsequently rejects that method.

**Correction/tests:** reuse an ExitStack or existing cleanup owner with separately registered resource callbacks; inject failure after each acquired resource and verify only owned resources are released. Normalize the default to the supported standard producer method. Validate BF16 and checkpoint/model shape agreement at the producer boundary, derive emitted runtime metadata from actual values, and retain numerical tolerance `rtol=atol=2e-2`.

**Native verification gaps:** current decomposed/native comparison checks outputs plus recurrent states, but independently generated prefix priming does not prove full-prefix vs prefix/current continuation equivalence. Add collected native fixtures using the same prefix/current tensors and check final output and both state components. CPU capability SKIPs must state missing AMD/MI355X/runtime requirements. NVIDIA timer results cannot close this lane.

### P05 — standard ROCm MoE DEVICE_EVENT cannot reach the fused kernel

**Source-confirmed, reachable interface mismatch.** `moe/main.py` advertises DEVICE_EVENT via `EXPORTABLE_PROFILE_METHOD_CHOICES`; `MoEWrapper` normalizes and passes the method unchanged through `_profile_with_vllm_kernel`; `moe_vllm_kernel.py:675` accepts only `cuda_event` and `record_function`. Thus the standard ROCm fused/MXFP4 path rejects the intended DEVICE_EVENT request even with a valid native runtime. CPU layout tests do not traverse this boundary.

**Correction/tests:** share method/platform admission and allow the existing event collector to execute DEVICE_EVENT without relabeling artifacts as CUDA_EVENT. Test actual wrapper-to-kernel argument flow and wrong-platform admission before allocations; separately collect a native AMD event lane. Preserve legacy CUDA and record-function behavior.

### P06 — routed SGLang sorting validation assumes one block per expert

**Source-confirmed internal arithmetic mismatch.** `experimental/sglang/moe.py:203-226` compares `sorted_experts[:sorted_token_blocks]` against one entry per active expert and iterates only those entries. A count greater than `block_size_m=32` requires repeated block owners. Example: size 64, top_k 1, counts `(64, 0)` implies two sorted token blocks but one expected owner. Counting uses assignment rather than accumulation, so even a repeated-owner loop needs histogram accumulation.

**Correction/tests:** derive ordered expected block owners by repeating each expert `ceil(count / 32)` times, validate all blocks, and accumulate per-expert valid lanes. Test loads 0/1/31/32/33/64, mixed multi-block owners, padding and invalid owner/weight/slot entries. Native sorting behavior remains NOT RUN.

### P07 — routed graph replay claims unchecked correctness and trace provenance

**Source-confirmed.** `routed_moe_replay.profile_routed_graph` obtains `expected` but never compares eager/captured/replayed outputs. It bypasses the existing `_make_replay_call` reset/check boundary, ignores mutable buffers returned by expert builders, omits the established synchronized warmup/reset behavior, then emits `correctness_checked=True`. `trace=True` only sets a string; it produces no trace callback/probe or kernel evidence. The expert builder's `validate` argument validates sorting metadata, not its fused expert output.

**Caller boundary:** callable-only public experimental entrypoint; repository `rg` found no standard simulator/training caller. Existing tests exercise only invalid arguments for this entrypoint. Therefore this is an experimental API defect, not demonstrated standard runtime corruption.

**Correction/tests:** route these builders through existing replay normalization/orchestration, preserving standard replay's already fixed return-shape normalization. Force incorrect output and mutable-state drift in a CPU simulated graph; the call must fail. Check actual representative trace evidence before setting its provenance. Keep HIP_GRAPH_REPLAY and JSON artifacts isolated from canonical training discovery.

### P08 — lower-priority boundary cleanup / focused reproductions

- `moe_impl.MoEGatingNetwork` catches every AssertionError/RuntimeError from ReplicatedLinear construction and silently selects `nn.Linear`. Narrow or replace this with explicit supported standalone capability selection; unrelated construction failures must propagate. Numerical equivalence does not alone preserve claimed kernel provenance.
- `moe_vllm_kernel._get_functional_mxfp4_state` hardcodes `hf_config.model_type='qwen3_5_moe_text'` in a generic profiling entrypoint. Its backend is extracted but `MoEWrapper` labels every MXFP4 run `vllm_aiter_mxfp4` solely from the requested boolean; NVIDIA or another physical backend could be mislabeled. Validate the pinned supported model/platform/backend, and verify actual packed tensors against the layout plan rather than treating the CPU plan as native layout proof.
- `collectives/main.py` duplicates the precision tuple from CollectivesInput and duplicates the dtype resolver from benchmark_runner. Consolidate at the existing collectives boundary. Local worker teardown is correctly under finally; preserve it. Local and Ray paths publish rank-zero timing, not rank maxima; do not claim rank-max aggregation. Local discovery returns physical IDs but worker selection consumes logical rank; verify inherited visibility mappings with nonzero IDs before changing behavior.
- `experimental/sglang/graph_replay.validate_replay_plan` does not repeat primitive/logical/context checks performed by build_replay_plan. A mutated serialized plan can pass write admission. `local_rank_for_visibility` does not reject rank >= world_size when visibility is populated. Reuse the construction validation at the external serialized boundary and cover negative plans without adding a second schema engine.
- `experimental/sglang/moe.moe_routed_spec` floors hidden-size packed dimensions and validates shape equality without checking the hidden dimension's required pack/group divisibility. The current checked-in model is aligned; malformed-model negative coverage remains useful.
- `experimental/sglang/moe.make_moe_routing_primitive` intentionally allows native top-k ordering during eager expert-set/probability checks, then returns native weights against torch.topk-ordered reference weights to generic replay check. Reproduce order differences before selecting a repair; preserve an order-independent oracle outside timing.
- `accelerator._active_visibility` treats an explicitly empty visibility variable as absent. Determine whether empty visibility is an intentional no-device contract in supported launchers before any behavior change. Existing `-1` rejection and native ROCm single-variable binding should remain.
- `common/model_config.py`'s family/layer-cache and GDN getter changes participate in W01/N01. Coordinate with the runtime owner; retain serialization and constructor inputs while removing the family split and repeated GDN shape normalization there.

## Existing tests and acceptance limits

The following inspected suites can be extended rather than replaced:

| Existing suite | Current source coverage | Missing gate |
| --- | --- | --- |
| `tests/unit/test_gdn_profiler_cpu_increment7.py` | Input phase/schema, one-token continuation, mixed per-call guard, CLI list shape, import/output naming | Full campaign before native side effects; actual sample admission; constructor resource failures |
| `tests/unit/test_device_timer_contract.py` | Mock DEVICE_EVENT, existing KINETO owner, measurement identities/paths | Standalone disabled-store poisoning, method conflict, all methods/exception cleanup, collected native timing |
| `tests/unit/test_profiling_timing_stats_contract.py` | Counts and RecordFunctionTracer priming conventions | GDN active/E2E expected counts and invalid numerics |
| `tests/unit/test_vllm_rocm_attention_wrapper_increment9.py` | CPU sequence/block plan and explicit lazy selection | Real dense prefill/decode/native result, mixed failure before GPU allocation |
| `tests/unit/test_moe_mxfp4_increment10.py` | Quantization mode and CPU packed-shape plan; a manually incomplete wrapper | DEVICE_EVENT real adapter boundary, native dtype/layout/backend identity |
| `tests/unit/test_collectives_rocm_runner.py` | Valid topology grid and exposed precision choices | Dtype-aware actual bytes, nonzero visibility/rank mapping, worker failure teardown, native RCCL/NCCL |
| `tests/unit/test_sglang_graph_replay_orchestration.py` | CPU simulated main replay/reset/check/trace; routed invalid count | Routed valid replay, corrupt outputs, mutable buffers, truthful trace provenance |
| `tests/unit/test_sglang_experimental_increment13.py` | Shape/routing/input/provenance and synthetic trace extraction | Multi-block sorting, native AITER/HIP graph correctness, serialized-plan tampering |

No test outcomes are inferred from reading these files. In particular, tests named CPU-safe still import torch/triton-dependent modules in some paths; collection in the actual minimal environment must be measured rather than assumed.

## Per-hunk disposition ledger

Each following row is one hunk from `git diff --unified=3 <pinned-baseline> -- frontier/profiling`; new modules each have one creation hunk. Coordinates describe the reviewed candidate before remediation. A mixed new-file hunk carries the most actionable disposition, with retained responsibilities and defects distinguished in its explanation. RETAIN means the inspected implementation serves its caller and no definite defect was found in that hunk; native evidence remains separate. Reconcile later edits against this snapshot.

Inspected inventory: **39 production Python files; 100 changed hunks**. One changed README is excluded as documentation.

| File under `frontier/profiling/` | Hunk | Disposition | Caller boundary / reason |
| --- | --- | --- | --- |
| `attention/backends/__init__.py` | `@@ -8,6 +8,7 @@ class AttentionBackend(Enum):` | RETAIN | Explicit enum/dispatch/lazy-export extension -> attention CLI and wrapper selection; no automatic backend selection. |
| `attention/backends/__init__.py` | `@@ -50,6 +51,12 @@ def get_attention_wrapper():` | RETAIN | Explicit enum/dispatch/lazy-export extension -> attention CLI and wrapper selection; no automatic backend selection. |
| `attention/backends/__init__.py` | `@@ -73,6 +80,12 @@ def __getattr__(name: str):` | RETAIN | Explicit enum/dispatch/lazy-export extension -> attention CLI and wrapper selection; no automatic backend selection. |
| `attention/backends/__init__.py` | `@@ -86,6 +99,7 @@ __all__ = [` | RETAIN | Explicit enum/dispatch/lazy-export extension -> attention CLI and wrapper selection; no automatic backend selection. |
| `attention/backends/vllm_rocm_attention_wrapper.py` | `@@ -0,0 +1,329 @@` | RETAIN | CPU sequence plan and explicit ROCm dense backend -> AttentionWrapper/BaseAttentionWrapper; native metadata ABI/results remain T12/T14 NOT RUN. |
| `collectives/benchmark_runner.py` | `@@ -8,23 +8,40 @@ import torch` | REFACTOR | Ray BenchmarkRunner -> shared accelerator visibility and CollectiveWrapper(dtype); retain actual dtype, consolidate duplicated precision rules (P08). |
| `collectives/benchmark_runner.py` | `@@ -33,6 +50,7 @@ class BenchmarkRunner:` | REFACTOR | Ray BenchmarkRunner -> shared accelerator visibility and CollectiveWrapper(dtype); retain actual dtype, consolidate duplicated precision rules (P08). |
| `collectives/benchmark_runner.py` | `@@ -71,6 +89,7 @@ class BenchmarkRunner:` | REFACTOR | Ray BenchmarkRunner -> shared accelerator visibility and CollectiveWrapper(dtype); retain actual dtype, consolidate duplicated precision rules (P08). |
| `collectives/benchmark_runner.py` | `@@ -83,7 +102,7 @@ class BenchmarkRunner:` | REFACTOR | Ray BenchmarkRunner -> shared accelerator visibility and CollectiveWrapper(dtype); retain actual dtype, consolidate duplicated precision rules (P08). |
| `collectives/collectives_impl.py` | `@@ -16,6 +16,7 @@ class GraphedCollective:` | RETAIN | CollectiveWrapper -> GraphedCollective.element_size; byte accounting uses actual allocated dtype. |
| `collectives/collectives_impl.py` | `@@ -101,3 +102,7 @@ class GraphedCollective:` | RETAIN | CollectiveWrapper -> GraphedCollective.element_size; byte accounting uses actual allocated dtype. |
| `collectives/collectives_input.py` | `@@ -1,6 +1,9 @@` | RETAIN | get_collectives_inputs -> precision/topology admission; exact worker divisibility avoids partial-node layouts. |
| `collectives/collectives_input.py` | `@@ -10,11 +13,19 @@ class CollectivesInput:` | RETAIN | get_collectives_inputs -> precision/topology admission; exact worker divisibility avoids partial-node layouts. |
| `collectives/collectives_input.py` | `@@ -32,10 +43,10 @@ class CollectivesInput:` | RETAIN | get_collectives_inputs -> precision/topology admission; exact worker divisibility avoids partial-node layouts. |
| `collectives/collectives_wrapper.py` | `@@ -21,6 +21,7 @@ class CollectiveWrapper:` | RETAIN | Local/Ray runners -> dtype-aware allocation and bytes; preserve existing KINETO ownership and verify via P02/T12. |
| `collectives/collectives_wrapper.py` | `@@ -31,7 +32,11 @@ class CollectiveWrapper:` | RETAIN | Local/Ray runners -> dtype-aware allocation and bytes; preserve existing KINETO ownership and verify via P02/T12. |
| `collectives/collectives_wrapper.py` | `@@ -61,7 +66,7 @@ class CollectiveWrapper:` | RETAIN | Local/Ray runners -> dtype-aware allocation and bytes; preserve existing KINETO ownership and verify via P02/T12. |
| `collectives/main.py` | `@@ -1,21 +1,47 @@` | REFACTOR | CLI -> Ray/local worker -> finally teardown -> writer; consolidate precision rules and inspect visibility/rank evidence per P08. |
| `collectives/main.py` | `@@ -44,96 +70,229 @@ def parse_args():` | REFACTOR | CLI -> Ray/local worker -> finally teardown -> writer; consolidate precision rules and inspect visibility/rank evidence per P08. |
| `common/accelerator.py` | `@@ -0,0 +1,232 @@` | RETAIN | linear/MoE/collective entrypoints -> one lazy platform/visibility owner; P08 empty-visibility policy needs reproduction. |
| `common/constants.py` | `@@ -30,6 +30,10 @@ class OperationMetrics(enum.Enum):` | RETAIN | GDN operator enum labels align family operator names; consumed by profiling timers. |
| `common/cuda_timer.py` | `@@ -1,110 +1,15 @@` | RETAIN | Existing layer/collective callers preserve CudaTimer import while sharing DeviceTimer; P02 fixes belong to the owner. |
| `common/device_timer.py` | `@@ -0,0 +1,129 @@` | BUGFIX | All wrapper timers -> singleton store; P02 unnamed global disable and explicit-method precedence. |
| `common/layers/layernorm.py` | `@@ -6,7 +6,9 @@ import torch` | RETAIN | Linear-op norm builders -> legacy/current vLLM RMSNorm plus config context; timer ownership is P02, native ABI is T12. |
| `common/layers/layernorm.py` | `@@ -16,8 +18,16 @@ try:` | RETAIN | Linear-op norm builders -> legacy/current vLLM RMSNorm plus config context; timer ownership is P02, native ABI is T12. |
| `common/layers/layernorm.py` | `@@ -37,10 +47,22 @@ class RMSNorm(nn.Module):` | RETAIN | Linear-op norm builders -> legacy/current vLLM RMSNorm plus config context; timer ownership is P02, native ABI is T12. |
| `common/layers/layernorm.py` | `@@ -51,6 +73,8 @@ class RMSNorm(nn.Module):` | RETAIN | Linear-op norm builders -> legacy/current vLLM RMSNorm plus config context; timer ownership is P02, native ABI is T12. |
| `common/layers/layernorm.py` | `@@ -75,7 +99,8 @@ class GemmaRMSNorm(nn.Module):` | RETAIN | Linear-op norm builders -> legacy/current vLLM RMSNorm plus config context; timer ownership is P02, native ABI is T12. |
| `common/layers/rotary_embedding.py` | `@@ -23,12 +23,14 @@` | RETAIN | get_rope -> one legacy/current keyword normalization boundary; validate real-version signatures without test-driven production exceptions. |
| `common/layers/rotary_embedding.py` | `@@ -97,6 +99,56 @@ def _load_vllm_get_rope():` | RETAIN | get_rope -> one legacy/current keyword normalization boundary; validate real-version signatures without test-driven production exceptions. |
| `common/layers/rotary_embedding.py` | `@@ -576,7 +628,8 @@ def get_rope(` | RETAIN | get_rope -> one legacy/current keyword normalization boundary; validate real-version signatures without test-driven production exceptions. |
| `common/model_config.py` | `@@ -3,7 +3,15 @@ from __future__ import annotations` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -51,6 +59,7 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -65,6 +74,14 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -121,6 +138,11 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -144,6 +166,26 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -238,9 +280,49 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -306,6 +388,8 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -350,6 +434,9 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -377,9 +464,19 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -554,6 +651,7 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/model_config.py` | `@@ -568,6 +666,14 @@ class ModelConfig:` | REFACTOR | from_model_name/JSON -> topology specs/GDN shapes -> profiling dispatch/plan; coordinate W01/N01, preserve serialization. |
| `common/timer_stats_store.py` | `@@ -20,14 +20,27 @@ class TimerStatsStore(metaclass=Singleton):` | REFACTOR | DeviceTimer -> scalar/event materialization/statistics; add narrow owner accessor, keep GDN-specific admission at P03 boundary. |
| `common/timer_stats_store.py` | `@@ -36,5 +49,7 @@ class TimerStatsStore(metaclass=Singleton):` | REFACTOR | DeviceTimer -> scalar/event materialization/statistics; add narrow owner accessor, keep GDN-specific admission at P03 boundary. |
| `common/vllm_compat.py` | `@@ -0,0 +1,34 @@` | RETAIN | Norm/RoPE/MoE adapters -> lazily scoped current-vLLM config; no eager GPU import. |
| `experimental/__init__.py` | `@@ -0,0 +1 @@` | RETAIN | Package boundary keeps experimental adapters isolated. |
| `experimental/sglang/__init__.py` | `@@ -0,0 +1,8 @@` | RETAIN | CPU-safe package export boundary; native imports remain in builders. |
| `experimental/sglang/attention.py` | `@@ -0,0 +1,277 @@` | RETAIN | make_primitive -> pinned shape/workload/native/reference adapter; preserve int32 indptr, mutable reset metadata and backend provenance. |
| `experimental/sglang/dense.py` | `@@ -0,0 +1,144 @@` | RETAIN | make_primitive -> model-derived BF16 projections and gated norm; native output correctness remains hardware gated. |
| `experimental/sglang/gdn.py` | `@@ -0,0 +1,178 @@` | RETAIN | make_primitive -> packed decode and explicit mutable state; retain independent FP32 oracle and physical/logical contract. |
| `experimental/sglang/gdn_trace.py` | `@@ -0,0 +1,318 @@` | RETAIN | Trace CLI -> pinned anchors/count checks -> experimental summary names; standard discovery isolation retained, native trace evidence NOT RUN. |
| `experimental/sglang/graph_replay.py` | `@@ -0,0 +1,494 @@` | BUGFIX | Direct replay retains fixed builder normalization/reset/check; serialized-plan admission/rank validation requires P08 repair. |
| `experimental/sglang/moe.py` | `@@ -0,0 +1,334 @@` | BUGFIX | Replay builders -> routing/sorting/experts; P06 multi-block sorting, P08 packed layout/order boundary checks. |
| `experimental/sglang/routed_moe_replay.py` | `@@ -0,0 +1,305 @@` | BUGFIX | Shared Frontier routing helper retained; public routed replay bypasses reset/check and fabricates evidence flags (P07). |
| `gdn/__init__.py` | `@@ -0,0 +1,20 @@` | RETAIN | CPU-safe GDN input exports; stale early-stage wording is cosmetic only. |
| `gdn/inputs.py` | `@@ -0,0 +1,255 @@` | BUGFIX | CLI/direct callers -> typed logical phase/state descriptors; retain one-token semantics and extend supported campaign capacity admission (P01). |
| `gdn/main.py` | `@@ -0,0 +1,136 @@` | BUGFIX | Campaign -> native producer -> canonical writer; full preflight must move before native import/construction (P01). |
| `gdn/vllm_wrapper.py` | `@@ -0,0 +1,639 @@` | BUGFIX | Native producer -> state priming/decomposition/timer/export; P01/P03/P04 capacity, active samples, ownership and actual metadata. |
| `linear_op/linear_op_impl.py` | `@@ -23,7 +23,7 @@ from frontier.profiling.common.model_config import ModelConfig` | RETAIN | Architecture profile -> existing attention-linear-op implementations; terminology migration and Qwen3.5 Gemma norm selection retained. |
| `linear_op/linear_op_impl.py` | `@@ -63,7 +63,8 @@ def _supports_share_expert(config: ModelConfig) -> bool:` | RETAIN | Architecture profile -> existing attention-linear-op implementations; terminology migration and Qwen3.5 Gemma norm selection retained. |
| `linear_op/linear_op_impl.py` | `@@ -699,23 +700,23 @@ def build_linear_op_attention_module(` | RETAIN | Architecture profile -> existing attention-linear-op implementations; terminology migration and Qwen3.5 Gemma norm selection retained. |
| `linear_op/linear_op_wrapper.py` | `@@ -102,7 +102,7 @@ class LinearOpWrapper:` | RETAIN | Expected timing names -> renamed profile attention_linear_ops field; no new operator ownership copy. |
| `linear_op/main.py` | `@@ -64,6 +64,10 @@ except ImportError:` | RETAIN | CLI discovery/sequential worker -> common accelerator helper; preserves parent process no-runtime discovery. |
| `linear_op/main.py` | `@@ -91,56 +95,8 @@ def _ensure_torch_available():` | RETAIN | CLI discovery/sequential worker -> common accelerator helper; preserves parent process no-runtime discovery. |
| `linear_op/main.py` | `@@ -787,8 +743,8 @@ def profile_model(` | RETAIN | CLI discovery/sequential worker -> common accelerator helper; preserves parent process no-runtime discovery. |
| `linear_op/profiling_plan.py` | `@@ -4,7 +4,7 @@ from __future__ import annotations` | RETAIN | Architecture/family -> typed operator plan; W01 must keep resolver and topology identities aligned. |
| `linear_op/profiling_plan.py` | `@@ -247,17 +247,17 @@ def _typed_operator_contracts(` | RETAIN | Architecture/family -> typed operator plan; W01 must keep resolver and topology identities aligned. |
| `linear_op/profiling_plan.py` | `@@ -427,7 +427,7 @@ def build_profiling_plan(` | RETAIN | Architecture/family -> typed operator plan; W01 must keep resolver and topology identities aligned. |
| `linear_op/profiling_plan.py` | `@@ -437,7 +437,7 @@ def build_profiling_plan(` | RETAIN | Architecture/family -> typed operator plan; W01 must keep resolver and topology identities aligned. |
| `linear_op/profiling_plan.py` | `@@ -461,7 +461,7 @@ def build_profiling_plan(` | RETAIN | Architecture/family -> typed operator plan; W01 must keep resolver and topology identities aligned. |
| `linear_op/profiling_plan.py` | `@@ -473,7 +473,7 @@ def build_profiling_plan(` | RETAIN | Architecture/family -> typed operator plan; W01 must keep resolver and topology identities aligned. |
| `moe/main.py` | `@@ -57,6 +57,10 @@ except ImportError:` | RETAIN | CLI multiprocessing/sequential discovery -> common accelerator binding; connected DEVICE_EVENT kernel mismatch is P05. |
| `moe/main.py` | `@@ -93,7 +97,7 @@ def _get_moe_wrapper_class():` | RETAIN | CLI multiprocessing/sequential discovery -> common accelerator binding; connected DEVICE_EVENT kernel mismatch is P05. |
| `moe/main.py` | `@@ -856,7 +860,7 @@ def profile_model(` | RETAIN | CLI multiprocessing/sequential discovery -> common accelerator binding; connected DEVICE_EVENT kernel mismatch is P05. |
| `moe/main.py` | `@@ -916,59 +920,8 @@ def profile_model(` | RETAIN | CLI multiprocessing/sequential discovery -> common accelerator binding; connected DEVICE_EVENT kernel mismatch is P05. |
| `moe/moe_impl.py` | `@@ -24,15 +24,38 @@ from frontier.profiling.common.parallel_utils.tensor_parallel_layers import (` | BUGFIX | MoEGatingNetwork -> current/legacy vLLM adapters; narrow broad exception-to-nn.Linear fallback (P08). |
| `moe/moe_impl.py` | `@@ -130,12 +153,19 @@ class MoEGatingNetwork(nn.Module):` | BUGFIX | MoEGatingNetwork -> current/legacy vLLM adapters; narrow broad exception-to-nn.Linear fallback (P08). |
| `moe/moe_vllm_kernel.py` | `@@ -10,8 +10,9 @@ Design rationale:` | BUGFIX | MoEWrapper -> functional/legacy kernel -> event timing; P05 DEVICE_EVENT and P08 actual MXFP4 identity/layout validation. |
| `moe/moe_vllm_kernel.py` | `@@ -19,27 +20,113 @@ Note: vLLM 0.3.x support has been removed. Please use vLLM >= 0.10.0.` | BUGFIX | MoEWrapper -> functional/legacy kernel -> event timing; P05 DEVICE_EVENT and P08 actual MXFP4 identity/layout validation. |
| `moe/moe_vllm_kernel.py` | `@@ -335,6 +422,138 @@ def _run_fused_moe_iteration(` | BUGFIX | MoEWrapper -> functional/legacy kernel -> event timing; P05 DEVICE_EVENT and P08 actual MXFP4 identity/layout validation. |
| `moe/moe_vllm_kernel.py` | `@@ -396,6 +615,7 @@ def profile_fused_moe_kernel(` | BUGFIX | MoEWrapper -> functional/legacy kernel -> event timing; P05 DEVICE_EVENT and P08 actual MXFP4 identity/layout validation. |
| `moe/moe_vllm_kernel.py` | `@@ -423,6 +643,7 @@ def profile_fused_moe_kernel(` | BUGFIX | MoEWrapper -> functional/legacy kernel -> event timing; P05 DEVICE_EVENT and P08 actual MXFP4 identity/layout validation. |
| `moe/moe_vllm_kernel.py` | `@@ -436,12 +657,21 @@ def profile_fused_moe_kernel(` | BUGFIX | MoEWrapper -> functional/legacy kernel -> event timing; P05 DEVICE_EVENT and P08 actual MXFP4 identity/layout validation. |
| `moe/moe_vllm_kernel.py` | `@@ -470,29 +700,115 @@ def profile_fused_moe_kernel(` | BUGFIX | MoEWrapper -> functional/legacy kernel -> event timing; P05 DEVICE_EVENT and P08 actual MXFP4 identity/layout validation. |
| `moe/moe_wrapper.py` | `@@ -31,6 +31,7 @@ from frontier.profiling.moe.moe_impl import (` | BUGFIX | CLI config -> quantized grouped GEMM -> metadata; P05 method flow and P08 backend-label truth require boundary fixes. |
| `moe/moe_wrapper.py` | `@@ -45,6 +46,18 @@ WARMUP_STEPS = 2` | BUGFIX | CLI config -> quantized grouped GEMM -> metadata; P05 method flow and P08 backend-label truth require boundary fixes. |
| `moe/moe_wrapper.py` | `@@ -114,9 +127,17 @@ class MoEWrapper:` | BUGFIX | CLI config -> quantized grouped GEMM -> metadata; P05 method flow and P08 backend-label truth require boundary fixes. |
| `moe/moe_wrapper.py` | `@@ -128,6 +149,8 @@ class MoEWrapper:` | BUGFIX | CLI config -> quantized grouped GEMM -> metadata; P05 method flow and P08 backend-label truth require boundary fixes. |
| `moe/moe_wrapper.py` | `@@ -588,12 +611,18 @@ class MoEWrapper:` | BUGFIX | CLI config -> quantized grouped GEMM -> metadata; P05 method flow and P08 backend-label truth require boundary fixes. |
| `moe/moe_wrapper.py` | `@@ -613,6 +642,7 @@ class MoEWrapper:` | BUGFIX | CLI config -> quantized grouped GEMM -> metadata; P05 method flow and P08 backend-label truth require boundary fixes. |
| `moe/moe_wrapper.py` | `@@ -641,6 +671,7 @@ class MoEWrapper:` | BUGFIX | CLI config -> quantized grouped GEMM -> metadata; P05 method flow and P08 backend-label truth require boundary fixes. |
| `utils/__init__.py` | `@@ -28,6 +28,7 @@ class ProfileMethod(enum.Enum):` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/__init__.py` | `@@ -36,6 +37,7 @@ class ProfileMethod(enum.Enum):` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/__init__.py` | `@@ -54,6 +56,8 @@ def normalize_profile_method(profile_method: str) -> str:` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/__init__.py` | `@@ -63,13 +67,31 @@ def profile_method_to_measurement_type(profile_method: str) -> MeasurementType:` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/__init__.py` | `@@ -120,6 +142,8 @@ def build_profile_method_output_path(` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/__init__.py` | `@@ -490,6 +514,7 @@ def get_collectives_inputs(` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/__init__.py` | `@@ -497,7 +522,7 @@ def get_collectives_inputs(` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/__init__.py` | `@@ -506,7 +531,11 @@ def get_collectives_inputs(` | RETAIN | Public methods/measurement taxonomy/output paths and collective precision grid; preserve CUDA aliases and distinct DEVICE_EVENT artifacts. |
| `utils/confirmation.py` | `@@ -173,9 +173,9 @@ def build_linear_op_config_sections(` | RETAIN | Configuration display -> renamed attention_linear_ops registry field; terminology-only change. |

## Reproducible inspection and remaining work

Read-only inspection used `git diff --stat`, `git diff --unified=3`, targeted `sed`/`cat`, and `rg` caller/test searches. The exact baseline command is:

```bash
git diff --unified=3 0515589ac7f49ac5288a5f55b0ce38b0ede29bb2 -- frontier/profiling
```

A search naming nonexistent `frontier/attention/families/gated_delta_net.py` returned a nonzero status; inspection immediately continued with the real declarative family in `frontier/attention/families.py`. This was an inspection path correction, not a product/test failure.

Pending tasks: P01-P08 disposition reconciliation and focused implementation/verification in W05/W07; native T12/T14 lanes; W01 owner coordination for common model config. Newly discovered issues beyond N05/N06/N09: P04-P08. No production/test changes or commits belong to this review subtask. Next steps are (1) reconcile with runtime-owner changes, (2) add focused failing regressions for confirmed boundaries, (3) repair and run CPU/native-capability lanes with separate PASS/SKIP/NOT RUN evidence.


## W05 independent preflight regressions — 2026-09-16

This follow-up is separate from the source-only snapshot above. The parent assigned only a new preflight test file and this review appendix; production files remain outside this reviewer's ownership. No commit was created.

### Delivered test boundary

`tests/unit/test_gdn_campaign_preflight.py` contains 27 parametrized cases. It uses the actual CLI parser and a real `ModelConfig.from_model_name("Qwen3.8-2.4T-A95B-Quark-MXFP4")` fixture with validated GDN dimensions. The native wrapper constructor is replaced with a recorder that raises `NativeConstructionReached` immediately, before importing/allocating native runtime state. Existing canonical `gdn.csv` bytes and the entire output-directory entry set are checked in a finally block after every invocation.

The 25 negative cases require ValueError or parser rejection before this constructor. Cases cover a mixed descriptor appended after valid work, later invalid query/batch entries, logical/physical batch capacity, sequence capacity, context + current query limits, invalid TP capacity, unsupported writer/profile methods and warmup/profile iteration counts. The physical-batch test supplies valid typed descriptors through the existing campaign builder seam because physical padding has no CLI selector. Two positive controls prevent a blanket reject-all implementation from passing: cold prefill + decode, and one-token continuation/decode exactly at sequence and batch limits.

### Execution and environment

- Interpreter: `/usr/bin/python`, Python 3.12.3; no active conda environment.
- NumPy 2.4.6: `/usr/local/lib/python3.12/dist-packages/numpy/__init__.py`.
- pandas 3.0.3: `/usr/local/lib/python3.12/dist-packages/pandas/__init__.py`.
- pytest 9.1.1: `/home/i-fengyicheng/.local/lib/python3.12/site-packages/pytest/__init__.py`.
- Persistent source: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_campaign_preflight.py`.
- Raw log: `/data/ycfeng/tmp/pr33-w05-preflight-regression/pytest.log`.
- No `task_memory/env_handbook.md` exists in this worktree. Interpreter selection follows the current task's recorded coherent Python 3.12 environment.

Reproducible command, from the active worktree root:

```bash
mkdir -p /data/ycfeng/tmp/pr33-w05-preflight-regression
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR /usr/bin/python -m pytest tests/unit/test_gdn_campaign_preflight.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w05-preflight-regression/pytest > /data/ycfeng/tmp/pr33-w05-preflight-regression/pytest.log 2>&1
```

### Observed RED evidence

**25 failed, 2 passed in 3.67s**, process exit 1. Every negative case failed because the forbidden native-construction boundary was reached, with the same assertion:

```text
E test_gdn_campaign_preflight.NativeConstructionReached: Native GDN producer construction reached
```

All output sentinel and output-directory preservation assertions passed. The two valid controls passed by reaching the recorded native boundary. This directly reproduces the CLI ordering defect without executing native code. It does not yet prove that a repaired campaign runs native GDN or produces valid samples; those remain W05/T12 responsibilities.

Failed node IDs:

- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[mixed-after-valid-workloads]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[zero-prefill-batch]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[later-prefill-batch-overflow]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[later-decode-batch-overflow]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[later-zero-prefill-query]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[later-prefill-query-overflow]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[zero-batch-capacity]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[zero-sequence-capacity]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[zero-tensor-parallel-size]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[zero-decode-context]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[negative-decode-context]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[decode-context-plus-query-overflow]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[negative-continuation-context]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[continuation-context-plus-query-overflow]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[negative-warmup-count]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[zero-profile-count]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[negative-profile-count]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[unsupported-method-cuda_event]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[unsupported-method-cuda]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[unsupported-method-record_function]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[unsupported-method-kernel_only]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[unsupported-method-perf_counter]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[unsupported-method-kineto]`
- `tests/unit/test_gdn_campaign_preflight.py::test_invalid_campaign_rejects_before_native_construction[unsupported-method-unknown]`
- `tests/unit/test_gdn_campaign_preflight.py::test_physical_batch_overflow_rejects_before_native_construction`

### Exact remaining timer and active-stat coverage gaps

Existing `test_device_timer_contract.py` has only two direct timer-lifecycle tests: `test_device_timer_records_device_events_without_importing_gpu_runtime` (fake event, default standalone named timer), and `test_cuda_timer_reuses_existing_kineto_store` (pre-created KINETO store with fake profiler). Its remaining tests concern taxonomy, paths, manager storage and replica measurement selection. These ownership cases are absent:

1. First unnamed timer with no existing store, followed by a named timer; the named timer must remain enabled.
2. Explicitly disabled existing store, explicitly enabled existing store, and unnamed timers reusing either owner without mutating ownership.
3. Explicit timer method matching or conflicting with the existing store, including alias normalization and documented precedence/conflict behavior.
4. CUDA_EVENT, PERF_COUNTER, RECORD_FUNCTION direct timer entry/exit; event/KINETO exception exit; no GPU import for disabled timers.
5. Repeated timer use and cleanup when the operation body raises; scalar/event sample units remain milliseconds.

`test_profiling_timing_stats_contract.py` verifies that manually inserted two-sample CUDA-event timing and RecordFunctionTracer output include `count=2`; it does not require producer-specific active samples. `test_gdn_profiler_cpu_increment7.py` tests descriptor semantics, one per-call mixed rejection, planning and import/path labels. No inspected test covers these GDN measurement-admission cases:

1. Missing active input projection, active phase core, or output projection; missing diagnostic `gdn_layer_e2e`; completely empty timing store.
2. Incorrect sample count for one active operator/E2E versus requested `profile_iterations`; empty lists; mismatched TP-rank sample schemas.
3. Negative, NaN or infinite samples; measured zero remains valid rather than a reason to substitute or reject by positivity alone.
4. Zero-fill only the inactive core phase, while retaining observed active/E2E samples and refusing output publication on invalid evidence.

No timer/stat tests were added here, as the bounded follow-up owns only the preflight test file. The parent should move this execution evidence into its required W05 test report when implementing the repair; a separate report file was outside this reviewer's explicitly assigned ownership.
