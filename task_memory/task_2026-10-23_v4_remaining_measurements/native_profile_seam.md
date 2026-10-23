# Native V4 profiling seam (TP8/DP2/EP16, no native edit)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Traced the existing native worker-extension seam, profiler lifecycle, phase/graph semantics and hook set; established that no shared profiler TP parameter is needed. Source-only, no implementation. |
| 2026-10-23 | Root recheck corrections: actual serving descriptors (P NONE/6144 single request; D ≤11 requests at the padded 16-token key, FULL+PIECEWISE), breakable CG actually enabled in native06, profiler restart proven by a CPU control, RPC fanout traced, aggregate kernel table demoted to diagnostic, V2 capture-event compatibility kept as a runtime gate. |

Task `W128-V4-NATIVE-PROFILE-SEAM`. Pinned native source
`/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`
(commit `2cf0a69`), native06 evidence
`/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023`. No GPU, no native/C26 edit.

## 1. Verdict

The existing native P/D path can host a non-native profiling extension that records Python
scope boundaries, the model's own stream/event fan-out structure, and per-scope CUDA-event
timings — with no native/C26 edit and **no shared profiler TP parameter**.

The J5 "shared TP parameter" claim in `proposal.md` is **not necessary**. The extension is
loaded once per worker process (`vllm/config/parallel.py:265` → `arg_utils.py:1191` →
`worker_base.py:265-288`) inside whatever topology vLLM already built; TP is topology, not a
profiler argument. The claim came from the isolated TP1 DSA producer, a different construction.

Two mechanisms are **not** proven and stay gates (§8): per-scope CUDA-event capture surviving
torch.compile/breakable-capture unchanged, and CUPTI behaviour across profiler restarts.

## 2. Existing seam (exact)

- Config/boot: `vllm/config/parallel.py:265` `worker_extension_cls`; `arg_utils.py:1191`
  (`--worker-extension-cls`), `:2274`; `worker_base.py:265-288` qualname binding.
- Harness: `native_pd_worker.py:269` gates on
  `worker_extension_cls == 'native_full_structure.FullModelInventory'`; `:270-292` injects
  `FRONTIER_NATIVE_SERVING_EVIDENCE_DIR` / `_RUN_GENERATION` / `_ROLE` and records
  `module_sha256`; `:294-299` `Popen(env=env)`.
- Extension: `native_full_structure.py:642-856` `_install_native_serving_inventory()`,
  `:863` at import, `:620-639` `emit()` → `serving/rank<worker.rank>.json`.
- Already wrapped (`:847-853`): `Worker.execute_model`, `GPUModelRunner._model_forward`,
  `_dummy_run`, `torch.cuda.CUDAGraph.replay`, `shutdown`, `compile_or_warm_up_model`,
  `load_model`; V2 (`:831-837`): `execute_model`, `ModelCudaGraphManager.capture`,
  `run_fullgraph`, `run_pw_graph`, `install_v2_model_call`, `install_v2_dispatch`.

Topology, from the executed config (`inputs/engine.json` via node `result.json` `plan.engine`):
`tensor_parallel_size: 8`, `data_parallel_size: 2`, `data_parallel_size_local: 1`,
`replica: 4` → internal DP across two nodes per role, TP8 × DP2 = EP16. No TP/DP argument
appears in the extension.

## 3. Actual phase and batch facts (corrected)

Read from all four `nodes/*/serving/rank0.json` (this is the corrected basis; capture capacity
is not serving evidence):

| Node | Role | serving rows | graph modes | max sched. reqs | max sched. tokens |
| --- | --- | --- | --- | --- | --- |
| 0214 | prefill | 7 | `NONE` only | 1 | 6144 |
| 0310 | prefill | 7 | `NONE` only | 1 | 6144 |
| 0346 | decode | 253 | `FULL` + `PIECEWISE` | 11 | 11 |
| 0399 | decode | 255 | `FULL` + `PIECEWISE` | 11 | 11 |

- Prefill is eager `NONE` at 6144 tokens (one 6144-token request per step).
- Decode runs at most **11 concurrent requests / 11 scheduled tokens**, and every serving row
  carries padded `input_shapes {'input_ids': [16], 'positions': [16]}` with batch descriptor
  `num_tokens: 16` (`num_reqs: 16` for FULL, `None` for PIECEWISE, `uniform: False`).
- `_is_compatible` (`cudagraph_utils.py:75-99`) requires `desc.num_tokens >= num_tokens` and
  `desc.num_reqs >= num_reqs`, so the **16-token descriptor serves the ≤11-request batch**.
- The 19 captured descriptors are all `FULL`, `num_tokens == num_reqs ∈ {1,2,4,8,16,…,128}`,
  `uniform_token_count: 1`. The 128-token descriptor exists as **capture capacity** and is
  never used by native06 serving; it must not be quoted as real decode coverage.
- Both `FULL` and `PIECEWISE` occur in the same real serving batch shapes (e.g. batch 11:
  FULL forwards/replays 408/408, PIECEWISE 10/440), so both modes need coverage.

Context length is a per-request KV property (`6143…7166`), not a graph key; the graph key is
batch shape only. The first decode step of a request can carry `has_prefill == True`
(`model_runner.py:1092-1099`), so step classification must use that flag per step — the
native05 classifier mistake must not be repeated here.

## 4. Breakable CUDA graph is ACTIVE in native06 (corrected)

`vllm/config/vllm.py:1313-1336` auto-sets `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for models without
`@support_torch_compile`, including `DeepseekV4ForCausalLM`, unless the env var is set. The
executed bootstrap proves the value, not the default: `server.log` line 16 on all four nodes
reads `Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1`. `native06` `server_env_set` contains no
such key, consistent with auto-enable.

Consequences that replace the earlier draft:

- `use_breakable_cg` is True (`cudagraph_utils.py:139-142`, `FULL_AND_PIECEWISE`).
- `@eager_break_during_capture` (`attention.py:405`) is therefore **live**, not inert: the
  attention core (`_prepare_and_attn_eager`, `_sparse_indexer_and_attn`) becomes an eager
  segment, not captured-graph content.
- `BreakableCUDAGraphCapture.replay()` re-executes the recorded eager callables
  (`breakable_cudagraph.py:195-217`: `segments.append(fn)` then `for r in self.segments: r()`),
  so **Python does run during decode replay** in those segments. The earlier statement
  "replay executes no Python" holds for captured segments only, not for this run's eager
  breaks. This is what makes Python-scope KERNEL_ONLY annotation possible at decode.
- A `torch.cuda.Event.record()` inside an eager segment is re-recorded on every replay
  (it is ordinary eager code), so per-replay elapsed times are individually valid there.
  Events recorded around *captured* segments become graph nodes and only reflect the last
  replay.
- `maybe_execute_in_parallel` disables the aux stream only while
  `BreakableCUDAGraphCapture.is_active()` (`multi_stream_utils.py:44-47`), i.e. during
  capture, not during replay. Inference to confirm at runtime: the eager segment replays with
  the multi-stream path enabled, so cross-stream overlap is real at decode and additivity
  needs the stream/ticket account, not just a kernel sum.

## 5. Three evidence classes (do not conflate)

| Class | Where | Python at measurement | Per-scope cost status |
| --- | --- | --- | --- |
| KERNEL_ONLY eager references | prefill `NONE` 6144 | runs | attributable per scope by kernel→launch correlation |
| Python annotations at decode | breakable eager segments, `FULL` and `PIECEWISE` | runs per replay | attributable for the segment's own kernels; graph-segment kernels have no Python scope |
| Captured-graph kernels | captured `FULL`/`PIECEWISE` segments | absent | not attributable to scopes; aggregate only |

## 6. Existing torch profiler controls, lifecycle and RPC fanout

- `Worker.profile` `vllm/v1/worker/gpu_worker.py:1146-1208`; wrapper built on first start
  (`:1170`), `activities=["CPU","CUDA"]` hardcoded (`:1176`), only `proton` resets the wrapper
  (`:1205-1208`).
- `TorchProfilerWrapper` `vllm/profiler/wrapper.py:174-352`: `_start` `:302-307`,
  `_stop` `:309-327`, `_profiler_step` `:330-343`, `annotate_context_manager` `:351-352`.
- Config-only enablement: `vllm/config/profiler.py:49-123` (`profiler=torch`,
  `torch_profiler_dir`, `torch_profiler_dump_cuda_time_total`, `delay/max_iterations`,
  `wait/warmup/active_iterations`).
- RPC fanout, traced: HTTP `POST /start_profile` / `/stop_profile`
  (`vllm/entrypoints/serve/profile/api_router.py:21-32`, mounted only when
  `profiler_config.profiler is not None`, `:40-46`) → `AsyncLLM.start_profile/stop_profile`
  (`vllm/v1/engine/async_llm.py:945-955`) → `engine_core.profile_async` →
  `DPLBAsyncMPClient.call_utility_async` (`core_client.py:1524-1533`) **gathers over all
  `core_engines`, i.e. every DP rank including headless peers** → `EngineCore.profile`
  (`core.py:768-769`) → `model_executor.profile` → `collective_rpc("profile", ...)`
  (`vllm/v1/executor/abstract.py:258-259`) → every TP worker. One request to the head URL
  covers both DP lanes and all 8 TP workers of that role. Internal DP is confirmed by
  `native_pd_worker.py:132-166` (`data_parallel_size_local`, `data_parallel_address`,
  `headless = lane != 0`, `data_parallel_start_rank`).

Lifecycle, corrected and measured: the earlier "one window per process" claim is **not
proved** and is retracted. `torch.profiler.profile.start()` → `prepare_trace()` constructs a
fresh kineto profiler object on every start (host torch 2.1.0 source), and a CPU-only control
on the installed interpreter ran three start/stop cycles on one object, producing three
distinct traces (`native_profile_checks/`, torch `2.1.0+cu121`, no CUDA init). What remains
unproved is GPU-side re-init: torch itself documents a CUPTI lazy-reinit/cuda-graph hazard
(`torch/profiler/profiler.py`, `start_trace`, `DISABLE_CUPTI_LAZY_REINIT` FIXME), and the
runtime torch is `2.13.0+cu130`, which is not importable on this host. So:
- Measured limitation: none at API level; restart works.
- Real constraints: CUPTI lazy re-init under CUDA graphs is an open risk, and
  `_write_profiler_table` writes a fixed `profiler_out_{rank}.txt` (`wrapper.py:267-275`),
  so that file is overwritten on restart; trace files are timestamped and do not collide.
- Recommendation, not a limitation: one window per measured phase, with a fresh launch when
  the risk matters.

## 7. Minimal hook set

Class methods (`setattr` on class, native files untouched): `DeepseekV4Model.forward`
(`nvidia/model.py:1130`), decoder-layer forward, `DeepseekV4Attention.forward`
(`attention.py:361`), `_run_parallel_input_projections` (`:515`), `_o_proj` (`:174`),
`DeepseekV4Indexer.forward` (`:895`), `DeepseekCompressor.forward` (`compressor.py:329`).

Free functions must be patched on the **importing module**: `mhc_post_tilelang`
(`nvidia/model.py:1176,1189`), `hc_head_fused_kernel_tilelang` (`:1203`),
`maybe_execute_in_parallel` and `execute_in_parallel` (`attention.py:53-56`, calls `:952`,
`:515-531`).

Stream/dependency metadata is readable from source: `model.py:1023-1028` (three aux streams),
`attention.py:889-893` (`aux_stream`, `ln_events`), `multi_stream_utils.py:44-59` and
`:65-105` (fan-out/fan-in edges, `enable` opt-in for the N-ary variant). Wrapping the two
helpers yields per-call `(stream, event, edge)` records. CUDA exposes no API to recover which
stream recorded an event or what it waits on, so these edges must be observed at the Python
call.

## 8. Gate: V2 capture-time external events under compile/breakable capture

Source-level feasibility of the hook set does **not** prove that `torch.cuda.Event` pairs
recorded inside `ModelCudaGraphManager.capture` (`cudagraph_utils.py:307-374`) survive
torch.compile and breakable capture unchanged. Under breakable capture the wrapping is
`BreakableCUDAGraphWrapper` segments plus eager breaks, so events placed at scope boundaries
may land in graph segments, eager segments, or both, depending on where the break falls — a
run-time property, not a static one. Retained as a runtime gate; the accepted DSA evidence
covers only the non-breakable whole-model `CUDAGraphWrapper` path
(`frontier/profiling/attention/dsa_full_dummy.py:90-99`).

## 9. Uninstrumented reference, demoted

`torch_profiler_dump_cuda_time_total` emits `key_averages(sort_by="self_cuda_time_total")` to
`profiler_out_{rank}.txt` (`wrapper.py:315-320`). It is **diagnostic only**: per-kernel-name,
aggregate over the window, not per-owner, not per-step, and without coverage proof that every
attention/op kernel is represented. It must not be quoted as a complete uninstrumented GPU
reference until attribution and coverage are established. The operation-level gap (mHC, head,
embedding) may use it as a hint requiring independent confirmation.

## 10. Unknowns

1. Kineto attribution of kernels launched by graph replay (expected readable; unverified here).
2. Whether event pairs survive torch.compile/breakable capture as intended (§8 gate).
3. CUPTI lazy re-init across profiler restarts under CUDA graphs (torch FIXME).
4. Aux streams actually enabled during eager-segment replay (§4 inference).
5. Prefill and decode are separate processes; the NIXL P/D transfer is outside every scope.

## 11. Proposed code ownership (proposal only, not approval)

One new file `tests/comparison/calibration/native_serve_profile.py` (worker extension: hooks,
per-rank scope/edge registry, `serving_profile/rank<rank>.json` beside `serving/`), 350–450
lines, single owner; enablement through existing `worker_extension_cls` and
`--profiler-config.*`; any driver bracket in a new driver file, not `pd_replay.py`. No job
budget proposed; no TP1/EP16-only/graph-from-eager equivalence claimed.
