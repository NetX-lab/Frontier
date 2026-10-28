# Stage 1a: the graph-replay ownership route (implementation design)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: implementation design of stage 1a ([design.md](design.md) §5, bootstrap parts of entries 13 and 17), the evidence it rests on, and the split into Codex assignments A and B. |
| 2026-10-27 | A2 revised to declared container scopes (§3.3); graph-replay scope tree added (§3.5); assignment B split into B1 (bridge-check minimum) and B2 (§3.4, §4); assignment A accepted. |
| 2026-10-27 | B1 accepted and committed (`03d246f9`): implementation decisions, the P1h package layout and the driver CLI recorded in §3.4; the `dsv4_o_proj` row and the eager break under the V2 runner in §3.5. |
| 2026-10-27 | §3.4: the re-capture takes a fresh graph pool (`548ec6c0`, after P1h attempt 1); P1h bridge check PASS on H200 (§4). |
| 2026-10-28 | B2 accepted and committed (`32afae01`): implementation decisions in §3.4; A4 result on the bridge data and the remedy recommendation in §4. |
| 2026-10-28 | §2: second checkout of the native commit noted; B3 assignment submitted (§4). |
| 2026-10-28 | B3 accepted: implementation in §3.4; result and the tile-schedule planner finding in §4. |
| 2026-10-28 | B4 accepted: KV-cache group rule and the 11-layer reduced model in §3.4; result in §4. |
| 2026-10-28 | Package 03 result and the kernel identity coverage finding in §4; B5 submitted. |
| 2026-10-28 | B5 accepted: result in §4; B6 next. |
| 2026-10-28 | B6 design (§3.4) and submission (§4). |
| 2026-10-28 | B6 accepted: result in §4; B7 next. |

## 1. Purpose

Design §5 stage 1a takes kernel ownership from the CUDA graph capture itself, because an eager pass captures different kernels (F016). This note fixes how that is implemented, so the Codex assignments carry settled decisions. It covers the bootstrap parts of entry 13 (capture-time ownership, eager-break ownership, the node bridge with the name check, the joined boundary, kernel intervals in raw records) and entry 17 (graph gaps from captured dependencies). The stop rule of §5 is unchanged: if the node bridge cannot be read on the reduced engine, stop and report to YC.

## 2. Evidence

vLLM, native commit `2cf0a69` (`/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm`; the same commit is also checked out at `/data/ycfeng/tmp/opencode/vllm-v0.28.0`):

- Breakable capture is auto-enabled for `DeepseekV4ForCausalLM` and then forces `compilation_config.mode = NONE` (`config/vllm.py:1310-1345`). The V4 forward therefore runs as ordinary Python during capture, so Python scope hooks run at capture time.
- `BreakableCUDAGraphCapture` (`compilation/breakable_cudagraph.py:124-214`): each graph segment is a separate `torch.cuda.CUDAGraph` with its own `capture_begin(pool)` / `capture_end()`; `add_eager(fn)` ends the segment, runs `fn()` once with no capture open, appends `fn` to `segments`, and begins a new segment; `replay()` runs `segments` in order.
- `eager_break_during_capture` (`breakable_cudagraph.py:90-116`) returns the undecorated function when Breakable capture is disabled at decoration time (so the integration must check that PIECEWISE captures record eager breaks), and otherwise passes through when no Breakable capture is active, when it is not capturing (so a nested break inside `add_eager` runs inline), or when the forward-context mode is FULL.
- The V4 eager breaks are `_sparse_indexer_and_attn` (`models/deepseek_v4/attention.py:577-601`; it calls `self.indexer.indexer_op` and then `self.forward_mqa` as siblings) and the wide `_prepare_and_attn_eager` (`attention.py:405-431`), chosen by `_prepare_and_attn_fn` (`attention.py:302-307`).
- `torch.cuda.CUDAGraph.capture_end` instantiates the graph and, without `keep_graph`, drops the captured graph. The capture graph can therefore only be read before the original `capture_end` runs, through `cudaStreamGetCaptureInfo` on the capture's origin stream.

CUPTI 13.0.85 (locked wheel headers; RCA "Verdict" in `../h200_graph_probe_attempt3_20261021/rca/rca_pending_2026-10-21.md`):

- Resource-domain `cbdata` is a `CUpti_ResourceData`; the `CUpti_GraphData` is its `resourceDescriptor` field. `CUPTI_CB_DOMAIN_RESOURCE = 3`, `CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED = 20`, `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED = 39`. `cuptiGetGraphNodeId` is legal inside the callback.

Observed identities:

- Attempt 1 on H200 (`/data/ycfeng/tmp/glm53_dsv4/h200_graph_probe_20261021/w128_graph_probe_cuda13_20261021/probe_report.json`, a five-kernel test graph): the captured graph had CUPTI graph id 1 and node ids `(1 << 32) + k`; the executable graph had id 2; the replayed kernels carried `graph id` 2 and `graph node id` `(2 << 32) + k`, with the same local indices `k`. Replay kernels therefore carry the ids of the instantiated clones, not of the captured nodes, and the bridge needs the clone map. The equal local index is an observation on one test graph, not a documented contract, so the route uses the clone callbacks, not the index.
- H200 native decode trace (P2-32 decode dp0 rank 0): kernel events carry `graph id`, `graph node id`, `correlation`, `grid`, `block`, `stream`; `gpu_memcpy` events carry `graph id` and `graph node id` too (0 for eager copies). Graph kernels share the correlation of their `cudaGraphLaunch` (`cuda_runtime`) event.
- Kineto names a kernel with `__cxa_demangle` of the raw name and keeps the raw name when demangling fails or the name exceeds its length limit. The name check therefore accepts either form.

## 3. Design

### 3.1 Capture-time ownership (`frontier/profiling/attention/graph_capture.py`)

- The recorder patches `capture_begin` and `capture_end` of the torch graph class (class-level, restored on exit). `capture_begin` opens a capture record: ordinal, kind (`breakable_segment` when a Breakable capture is active, otherwise `graph`), the Breakable capture's ordinal, and the origin stream (the current stream at `capture_begin`). `capture_end` sweeps once and then calls the original, during which instantiation fires the clone callbacks.
- Scope stack: the integrator wraps every hooked scope in `recorder.scope(owner)`, with `owner = (global_layer_id, scope)`. At each boundary (enter and exit) the recorder sweeps when a capture record is open, then pushes or pops. A sweep reads the nodes of the graph under construction, takes the ones not yet seen in this capture, and gives them the current stack top as owner; an empty stack means the outside owner (`None`). The host issues launches in program order, so the nodes a scope adds outside its child scopes are exactly the nodes swept at its boundaries while it is on top.
- Per node: capture ordinal, CUPTI node id, node type, and for kernels the raw name, the demangled name (or `None` when demangling fails), grid and block; for every node the CUPTI ids of its dependencies. Node types: kernel, memcpy, memset (activities); event record, event wait, empty (pass-through for dependencies); any other type fails the capture with its type.
- Per scope entry while a capture is open: capture ordinal, owner and parent (the stack top before the push). The parents record the captured scope tree; the bridge check compares it with the declared tree of §3.5.
- Eager breaks: the recorder patches `add_eager`. It reads the owner (stack top) at registration, calls the original, checks that the last segment is the registered function, and replaces it with a wrapper that runs the function through the integrator's `eager_scope_runner(owner, fn)` at replay. The capture-time call is unchanged. Each registration records the Breakable capture's ordinal, the segment index and the owner.

### 3.2 Clone map (same module)

A CUPTI subscriber records, for every `GRAPHNODE_CLONED` callback, the CUPTI ids of the clone (`node`) and of its original (`originalNode`), read through `CUpti_ResourceData.resourceDescriptor`. A null descriptor or a failed id query is stored and raised when the recorder stops (a ctypes callback cannot raise). The library is the `libcupti.so*` already mapped in the process (`/proc/self/maps`, exactly one path), the one Kineto uses. A multiple-subscriber error fails with a message naming the conflict; the recorder never unsubscribes a slot it did not create, and it stops before any torch.profiler session.

### 3.3 Replay attribution (`frontier/profiling/attention/graph_replay_trace.py`)

- Steps: the integrator wraps each measured `execute_model` in a step range `frontier_dsa_step|<step_index>` (new helper beside the existing `annotation_name`).
- Graph activities (kernels, memcpy and memset with a positive `graph node id`): correlation → `cudaGraphLaunch` → the step range on its thread. The node id goes through the clone map to the captured node and its owner. An id that is not a clone id fails; the message says whether it is a captured node id itself (the other case of §5). All activities of one launch belong to one capture; two activities of one launch on one captured node fail. Kernel names must equal the captured raw or demangled name, and grid and block must be equal.
- Eager kernels: correlation → runtime or driver launch → the innermost scope range (`frontier_dsa|step|layer|scope`) containing the launch on its thread. Ranges nest properly; a partial overlap fails. A scope key may have several ranges in one step (one per eager break); their kernels add. Eager kernels inside a step range but outside every scope range are owned by the outside owner of that step. Launches outside every step range are counted and not attributed.
- A2: a kernel owned directly by a container scope fails. The containers are declared by the integrator's scope tree (§3.5) and passed to the attribution, not inferred from nesting: a scope may have child scopes and still own the kernels it launches outside them (design §5). `dsv4_index_topk` is such a scope: it owns the top-k buffer fill, the top-k kernels and the packing kernels of the native indexer op, which cannot be wrapped, and it has the child `dsv4_index_scores`. Unmapped graph kernels fail as above. A scope that was entered (capture entries of the launched captures, or ranges) and owns no kernel in the step is valid with time 0 and count 0.
- Costs: a scope's kernel-only time and count are the sum and number of its kernels, as in `dsa_kernel_trace.aggregate_kernel_trace`; memcpy and memset are excluded from costs, kept in the raw intervals, and used as dependency ends.
- Layer window and joined boundary: per (step, layer), the kernels of all scopes of that layer; the joined boundary is the first kernel start to the last kernel end over all streams.
- Graph gaps (entry 17 input): for a graph kernel, the latest end among its captured dependencies in the same launch, after passing through event and empty nodes; the gap is its start minus that end. A kernel, memcpy or memset dependency without an activity in the launch fails. A kernel with no activity dependency (a segment's root) has no graph gap; the seam reducer of entry 17 treats it.
- Raw intervals per kernel: step, owner, kind (graph or eager), start, end, stream, graph id, graph node id, original node id, correlation, graph gap.

### 3.4 Integration (assignments B1 and B2)

B1 is the minimum for the P1h bridge check; B2 completes the raw records and the reducers after the bridge holds.

B1:

- Engine policy `graph_replay` in `dsa_plan.py` and `build_engine`: no `enforce_eager`, `async_scheduling=False`, the plan's capture sizes, `compilation_config.cudagraph_mode` of the engine's mode (one engine per FULL_AND_PIECEWISE, PIECEWISE, NONE), the native multi-stream threshold; `require_scope_timing` accepts it with KERNEL_ONLY only (CUDA_EVENT records would add event nodes to the captured graph).
- The scope hooks of the graph-replay tree (§3.5) run inside `recorder.scope`, every scope annotated; the step range around `execute_model`; the eager-scope runner annotates with the step read at call time.
- Re-capture after the hooks are installed: clear the V2 graph manager's graphs and the Breakable runner's graphs, mark it uncaptured, and call the runner's `capture_model()` with both recorders active; stop the CUPTI subscriber before the first profiler session.
- The runtime mode per step from the V2 dispatch result (FULL replay runs no Python hook, so neither the mode nor the metadata can come from a model hook); attention metadata from the runner's execute-model state.
- PIECEWISE captures must record eager breaks; zero breaks fail, because `eager_break_during_capture` returns the undecorated function when Breakable capture is disabled at decoration time.
- The P1h bridge-check driver under `tests/` and its launcher, modelled on the W6 `producer_jobs.py` and the attempt-3 H200 config. The driver keeps the capture record JSON and the replay traces, so the attribution can be rerun on CPU after a library correction without a new GPU run.

B1 as implemented (`03d246f9`, [test_report_2026-10-27_b1_graph_replay_integration.md](../test_report_2026-10-27_b1_graph_replay_integration.md)):

- The re-capture takes a fresh private graph pool (`current_platform.graph_pool_handle()`) for the V2 graph manager and the Breakable runner (`548ec6c0`). Once all graphs of a pool are released, PyTorch refuses to capture into that pool again (P1h attempt 1 failed on the `use_count > 0` assert of the host caching allocator); vLLM also takes a fresh pool when it discards captures.
- Breakable capture and the V2 runner are selected by `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and `VLLM_USE_V2_MODEL_RUNNER=1`, which must be set before vLLM is imported. The bridge driver sets both for each mode's process. The runner fails fast when a PIECEWISE mode runs without the Breakable variable, and the hook installation fails when `use_v2_model_runner` is false.
- NONE has no capture sizes (`cudagraph_capture_sizes=[]`); the plan rejects sizes for NONE, sizes that do not increase strictly, and a largest size above the smallest chunk budget. vLLM may resolve another mode or size list (`resolve_cudagraph_mode_and_sizes`); `recapture_graphs` fails fast on any difference.
- The runtime mode of each step comes from wrapping `dispatch_cg_and_sync_dp`, imported by name into `vllm/v1/worker/gpu/model_runner.py`; its `BatchExecutionDescriptor` (mode, token count, request count) is written into each raw step record under the key `cudagraph_dispatch`. Attention metadata comes from `execute_model_state.attn_metadata`.
- Graph-replay records are KERNEL_ONLY (`timing_measurements`); V4 row aggregation refuses graph-replay engine records (`index_records` calls the unchanged `require_scope_timing`) until B2 adds the interval rows.
- The capture record and its summary are saved as `dsv4_capture_e<engine>.json` beside the raw records, so the bridge report reruns the attribution on CPU.
- P1h package (option Y): one bootstrap invocation whose script is the bridge driver. The committed `frontier` and `data/config` trees sit at the package root, because the accepted bootstrap puts only the package directory on `PYTHONPATH`. The driver runs each mode as `[sys.executable, profile_dsv4_prefill.py, --config, <mode>/bootstrap_config.json]` with the bootstrap identity fields unchanged, and continues after a failed mode. Its CLI is exactly `--config PATH` (run) and `--config PATH --report` (CPU report). A second worker step runs the report; the worker stops at the first failed step, so after a failed run the report is run locally from the cloud copy.

B2 as implemented (`32afae01`, [test_report_2026-10-28_b2_replay_gaps_composition.md](../test_report_2026-10-28_b2_replay_gaps_composition.md)):

- Graph-replay raw records keep the call rows unchanged and add one interval row per activity: kernels and eager copies (memcpy, memset), with capture kind, owning scope, capture ordinal, launch-call end and the signed graph gap. Copies are dependency endpoints only; they never enter scope costs, layer windows, outside counts or A2. The engine record adds the driver version and the target TP of the plan.
- Gaps stay signed, because programmatic dependent launch lets a kernel start before its captured dependency ends. For each activity, `gap = start - dependency_end` and `host_wait = clamp(launch_end - dependency_end, 0, max(gap, 0))`; only host-ahead samples (`host_wait = 0`) price a class. The reducer writes `kernel_gap.csv` only when all four V4 classes (`cuda_graph`, `cuda_graph_piecewise`, `eager`, `eager_preparation`) have a value, and always writes its diagnostics.
- Dependencies: a graph kernel depends on its captured predecessors (a graph root on the latest earlier activity end). A scope-owned eager activity depends on the previous activity of its stream; the first activity of a scope on a stream also depends on the ends of its scope predecessors on other streams (`attention_scope_predecessors`, the same frontier walk that `compose_attention_phase` schedules), or, for a scope without predecessors, on the latest end of the other layers' activities before the call. An unowned eager activity depends on the latest earlier activity end. In a graph-mode step a scope-owned eager activity also depends on the end of the whole preceding graph launch: the replay nodes of a graph are reported on other Kineto streams (368-371), while the eager break runs on the launch stream. This rule is a lead correction; without it the `eager` mean was 3.508 instead of 1.935 µs.
- Ratio convention: the engine module records `compress_ratio = max(1, native ratio)`, while the plan keeps the native `compress_ratios` entry (0 for uncompressed layers). A4 compares the plan ratio as `max(1, ratio)`.
- A4 (`tests/comparison/calibration/dsv4_composition_check.py`) groups calls by engine mode, step mode, target TP, variant, planner role, case, threshold side and short-context state, never pooling them. A call counts only when every owned activity has a dependency end and is host-ahead. Its target is the joined kernel window plus the first kernel's signed gap. A diagnostic composes each call with its own measured signed gaps; it is reported but does not gate.

### 3.5 Graph-replay scope tree

vLLM source: `models/deepseek_v4/attention.py`, `compressor.py`, `model_executor/layers/sparse_attn_indexer.py` and `nvidia/ops/o_proj.py` at native commit `2cf0a69`, single GPU (TP1, so no sequence-parallel collectives). The W6 eager-chunked hooks stay unchanged; this tree is used by the `graph_replay` policy only.

Containers (they must own no kernel): `dsv4_attention_layer`, `dsv4_parallel_input_projections`, `dsv4_index_prepare`, `dsv4_compressor`, `dsv4_index_compressor`, `dsv4_o_proj`.

| Scope | Parent | Hook |
| --- | --- | --- |
| `dsv4_attention_layer` | outside | `DeepseekV4Attention.forward` |
| `dsv4_parallel_input_projections` | layer | `_run_parallel_input_projections` |
| `dsv4_fused_qkv_projection` | projections | `_fused_wqa_wkv_gemm` (default stream) |
| `dsv4_compressor_projection`, `dsv4_index_weights_projection`, `dsv4_index_compressor_projection` | projections | aux functions 0, 1, 2 of the `execute_in_parallel` call made while the projections scope of the same layer is on top (attention-module name `execute_in_parallel`) |
| `dsv4_fused_qkv_rmsnorm` | layer | attention-module name `fused_q_kv_rmsnorm` |
| `dsv4_q_projection` | layer | `wq_b.forward` |
| `dsv4_fused_q_kv_prepare` | layer | `_fused_qnorm_rope_kv_insert` |
| `dsv4_compressor` / `dsv4_index_compressor` | layer / index prepare | the layer's and the indexer's `Compressor.forward` |
| `<compressor>_save_partial_states` | compressor | compressor-module name `save_partial_states` |
| `<compressor>_fused_compress_store` | compressor | compressor-module names `compress_norm_rope_store_triton` and `compress_norm_rope_store_two_stage_triton`, and `compress_norm_rope_store_cutedsl` of `nvidia/ops/sparse_attn_compress_cutedsl` (imported inside `forward`, so the module attribute is wrapped); the c128 early return skips it outside FULL |
| `dsv4_index_prepare` | layer | `DeepseekV4Indexer.forward` |
| `dsv4_index_query_projection` | index prepare | `indexer.wq_b.forward` |
| `dsv4_index_fused_query_prepare` | index prepare | attention-module name `fused_indexer_q_rope_quant` |
| `dsv4_index_short_context_fill` | index prepare | the launch of `_fill_short_context_topk_indices` (eager only: the short-context branch is skipped while capturing) |
| `dsv4_index_topk` | layer | `indexer.indexer_op.forward`; owns the kernels outside its child |
| `dsv4_index_scores` | index top-k | sparse-attention-indexer module names `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits` and `ops.cp_gather_indexer_k_quant_cache` |
| `dsv4_sparse_attention` | layer | `forward_mqa` |
| `dsv4_o_proj` | layer | `_o_proj` of the attention module (the FlashMLA override calls `deep_gemm_fp8_o_proj` of `nvidia/ops/o_proj.py`) |
| `dsv4_output_inv_rope_quant`, `dsv4_grouped_output_projection` | o_proj | o_proj-module names `fused_inv_rope_fp8_quant`, `fp8_einsum` |
| `dsv4_output_projection` | o_proj | `wo_b.forward` |

`<compressor>` is `dsv4_compressor` or `dsv4_index_compressor`, the compressor scope on top of the stack. The eager break `_sparse_indexer_and_attn` is registered while the layer scope is on top; at replay the eager-scope runner encloses it in a layer range, and the `dsv4_index_topk` and `dsv4_sparse_attention` ranges inside own its kernels. With the V2 runner the break is this narrow `_sparse_indexer_and_attn`; the wider `_prepare_and_attn_eager` is used only when the V2 runner is off, which the graph-replay policy does not run.

B3 as implemented ([test_report_2026-10-28_b3_hold_pilot_identity.md](../test_report_2026-10-28_b3_hold_pilot_identity.md)):

- Pilot plan: `--pilot_cases PATH` (strict JSON: engines with their `prefill_query_len`, cases with phase, batch, prompt and an optional `dp_peer`) and `--step_hold_us`. Phases `first_tail` and `dummy` join `prefill` and `decode`. Plans without these options serialize byte-identically.
- Step hold: a device spin of `step_hold_us` at the start of each recorded step, calibrated in cycles per µs at hook installation, so that the host runs ahead of the device inside the step. The step range now also covers `sample_tokens`.
- DP-peer emulation: for a case with a `dp_peer`, the wrapped `dispatch_cg_and_sync_dp` calls vLLM's own synchronization with `dp_size` 2 and temporarily replaces `dp_utils.dist`, whose `all_reduce` writes the planned peer column into the CPU tensor; padding and mode therefore come from the native rule.
- A4 is the own-gap gate: each call composes with its own measured signed gaps; the one-value-per-class composition stays as a diagnostic.
- `tests/comparison/calibration/dsv4_kernel_identity_check.py` expands the pilot's per-layer attention kernels to the full model and requires count-wise containment in a native TP0 dispatch multiset of the same step class and mode (P2-32 and P2-64).
- The bridge config lists runs as (mode, engine index, case ids); package 03 runs FULL_AND_PIECEWISE for engines e0 (21 cases) and e1 (4 first-tail cases).
- Lead correction: `build_dsv4_rows` refuses graph-replay plans before reading records.

B4 as implemented ([test_report_2026-10-28_b4_tile_schedule_groups.md](../test_report_2026-10-28_b4_tile_schedule_groups.md)):

- `frontier/attention/deepseek_v4.py` reproduces vLLM 0.28's decoder KV-cache groups (`_approximate_gcd` over [max(n_csa, n_hca), N, n_csa, n_hca] with lower bound max(n_csa, n_hca), strided split of the SWA, CSA-state and HCA-state groups). The first layer of each variant in each SWA group plans the decode tile schedule: layers 0-3 in the full model.
- The predictor and the profiling plan share this rule. The default reduced model is the fewest leading layers with the full model's group counts, ordered SWA variant sets and (variant, role) coverage: 11 layers, planners 0-3, CSA reuse {4, 6, 8, 10}, HCA reuse {5, 7, 9}. `--num_layers 6` is rejected.
- Both runners record the realized `kv_cache_groups`; `build_dsv4_rows` and the A4 planner-role check require them to equal the plan's.
- The identity check expands per (variant, planner role) with the full-model counts {uncompressed planner 2, CSA planner 1, CSA reuse 20, HCA planner 1, HCA reuse 19}; the reduced layers of one role must carry equal kernel multisets.

B6 design ([codex_assignment_B6_replay_rows.md](codex_assignment_B6_replay_rows.md)):

- New `frontier/profiling/attention/dsv4_replay_rows.py`: rows from graph-replay records (schema `dsv4_attention_replay.v1`, KERNEL_ONLY only), written by `dsv4_attention_main aggregate` as `attention_graph_replay_kernel_only.csv`; the eager rows and their table are unchanged.
- Row key: engine, case, variant, planner role, step dispatch (`cg_mode`, `num_tokens`, `num_reqs`), request signature, variant metadata and the observed scope set. Per scope: `time_stats` (with `kernel_count`) and `gap_stats` (the per-call sum of the scope's kernels' signed gaps); per row: `layer_window_stats` (the A4 target).
- Scope sets are observed, not derived from a rule: on CSA layers vLLM skips the scores and the prefill top-k when a prefill chunk has no compressed key, and runs the short-context fill only eagerly. Entry 4 must predict the observed sets.
- A non-host-ahead call raises: under the step hold every owned activity must be host-ahead. A4 and the rows share one per-call timing function and one plan-engine check.
- Dummy rows carry the dispatch, phase and metadata columns but no request or key-length columns; first-tail mixed batches get the load columns through `request_load_columns`, split from `batch_load_columns`.

## 4. Order and acceptance

`A (graph_capture.py, graph_replay_trace.py, CPU tests with fakes) -> review and commit -> B1 (policy, scope tree, re-capture, step range, bridge-check driver and launcher) -> review and commit -> P1h bridge check (1 H200, via /ds4-exec) -> B2 (intervals, gap reducer, A4) -> rest of P1h`.

- A: the new unit tests pass on CPU; the existing DSA kernel-trace and V4 composition tests still pass.
- Bridge check: on the reduced engine, every replayed kernel of FULL and PIECEWISE steps resolves through the clone map to exactly one captured node with a matching name, grid and block; otherwise the §5 stop rule applies.
- Result: PASS on 1 H200 (`exp-1028-031147-398843`, source `548ec6c0`): 24 FULL and 72 PIECEWISE steps, 12 traces without an attribution error, no A2 error; repeated on CPU by the lead ([test report](../test_report_2026-10-27_b1_graph_replay_integration.md), section "P1h submission").
- B2: the CPU suites pass with only the known kernel-gap failure; the rebuilt call rows equal the worker's; the bridge report still passes. The reducer gives `cuda_graph` 0.4029, `cuda_graph_piecewise` 0.9618 and `eager` 1.9345 µs, and no `eager_preparation` value (no host-ahead sample on the host-bound bridge engine), so no CSV is written.
- A4 on the bridge data: FAIL. Of 74 groups, 38 have no host-ahead call, 28 fail and 8 pass. With one gap value per class, FULL CSA passes (1.6-2.2%), FULL uncompressed and HCA are 4-7% high, PIECEWISE decode 8-9% high, PIECEWISE prefill 6-9% high except CSA 6-7% low. With each call's own gaps the composition equals the measured window (errors about 1e-16 in all 36 evaluated groups), so the stream program is right and the class approximation is the cause. This is the design §3.3 stop rule; the decision was reported to YC.
- Recommendation (b), adopted by default under D-NO-INTERRUPT: each attention scope's measured signed gaps (entry and intra-scope, host-ahead samples) are recorded with its cost row at its shape as a separate gap column, and the predictor learns them like durations; class values stay for non-attention operators and pre- and post-forward work. Diagnostic support: per-scope gaps taken from the nearest other decode shape compose all 13 decode groups within median 0.7% and p90 1.2%. A4 then checks the per-call composition, and shape generalization moves to A6. The change lands in entries 1, 3 and 10.
- B3 is next: the kernel identity checker, the host-ahead hold, a case above the 1024-row threshold, the step range over `sample_tokens`, and the rest-of-P1h package.
- B3 (CPU, lead rerun): pytest groups pass with only the two known failures; legacy plans and the census are byte-identical; on the B2 data the A4 gate gives {no_host_ahead_call 38, pass 36} (no hold in B2), the diagnostic {38, fail 28, pass 8}; the package 03 plan has 25 cases and 335 steps.
- Finding (blocks package 03): the 6-layer reduced engine runs the FlashMLA tile-schedule planner (`get_mla_metadata_kernel`, about 26 µs) in every layer; native runs it in layers 0-3 only. vLLM keeps one plan per (SWA KV-cache group, layer type); `_approximate_gcd` splits the 43-layer model's SWA layers into 2 strided groups (even, odd; the native `cache.kv_cache_groups` inventory shows them) and the 6-layer model's into 3, so every reduced layer plans and no layer measures reuse. Frontier's "first layer of each variant" rule also misses that layer 1 plans natively. The expansion overcounts 39 planners, about 1 ms per decode step. Fix: assignment B4 (a vLLM-grouping planner rule shared by the plan and the predictor, a reduced layout with the native group structure, the realized groups recorded by the engine, and an identity expansion per (variant, role)).
- B4 (CPU, lead rerun): pytest groups pass with only the two known failures; the predicted 43-layer groups equal the native P2 serving inventory; DSA plans byte-identical; V4 plans change only in the reduced layout; the package 03 plan has 11 layers, 25 cases and steps [280, 55]. Package 03 is next: first confirm that the recorded groups equal the 11-layer prediction and that planner kernels run in layers 0-3 only.
- Package 03 (1 H200, `exp-1028-132243-707076`, about 0.244 GPU-hours; [test report](../test_report_2026-10-28_p1h_package03.md)): the realized groups equal the 11-layer prediction and the planner runs in layers 0-3 only; the bridge check, the A4 own-gap gate (125 groups), the host-ahead hold and the reducer pass. The kernel identity tool fails because it compares a pilot step with a native forward of another shape when the native traces lack its shape; at every shape the native traces contain, the kernels agree once the eager-break rows are part of the key. B5 keys the identity check by (class, mode, padded rows, eager rows) and reports `NO_REFERENCE` for uncovered shapes. After B5: entries 1, 10 and 3 with the per-scope gap column of remedy (b).
- B5 (CPU, lead rerun; [test report](../test_report_2026-10-28_b5_identity_shape_keys.md)): P2-32 {46 PASS, 129 NO_REFERENCE, 0 FAIL} and P2-64 {54, 121, 0}, both verdict PASS; 7 referenced and 16 unreferenced pilot shapes per trace set. The unreferenced shapes stay unqualified by this check; the stage 4 end-to-end replays qualify them. Next: B6, the graph-replay V4 rows with a per-scope gap column (rows part of entry 13), then entries 1, 10 and 3.
- B6 submitted (`4663821b`): acceptance is 505 rows from the 1925 package 03 layer calls, an A4 report byte-identical to the `d2a226fe` baseline, and row compositions of the medians within the A4 limits.
- B6 (CPU, lead rerun; [test report](../test_report_2026-10-28_b6_replay_rows.md)): 505 rows from 1925 calls with the expected phase and mode counts; A4 byte-identical; row compositions of the medians: median 2.1e-16, p90 0.77%, max 3.19% (single-call rows exact). Next: B7, the V4 attention loader, trainer and predictor on these rows (entry 1), each scope weighed by its own gap in `compose_attention_phase` (entry 2 revision) and the V4 gap selection (entry 10).
