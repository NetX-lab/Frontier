# Native capture scope: are Python per-op scopes current during native graph capture?

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Closed independent prerequisite review, copied the root census readably without changing its bytes/ACL, and verified retention limits from the same input. |
| 2026-10-24 | Root correction: independently counted the retained 105,492,191-byte artifact, separated overlapping capture flags, verified the available official CUPTI header, and removed the cross-run join proposal. v2 is preserved at `/data/ycfeng/tmp/glm53_dsv4/native_capture_scope_v2_preserved_4rxhxz0u/native_capture_scope.md`. |
| 2026-10-24 | v2 correction: replaced the replay-launch framing with the capture-time resource-callback mechanism (a-c) plus replay labels (d); added observed capture/replay rows from retained profile02 artifacts; retracted "absence settles" and the executed-04 reading; bound provenance to the actual-image receipt. v1 (`24ee997c…`, 76 lines) kept at `/data/ycfeng/tmp/glm53_dsv4/w128_native_capture_scope_v1_24ee997c.md`. |
| 2026-10-24 | v1 read-only source assessment. |

`V`=`/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` (commit `2cf0a69`), `W`=this worktree, `C`=`W/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_04`, `R`=`/data/ycfeng/tmp/glm53_dsv4/profile02_results_readonly_7ba56n6z/artifacts/nodes`. Source/retained-artifact reading and CPU static-source checks only; no GPU allocation, native graph execution or callback subscription.

## Verdict

The seam under assessment is **CAPTURE-time**: CUPTI resource callbacks `RESOURCE_GRAPHNODE_CREATED`=13, `GRAPHNODE_CLONED`=20 and `GRAPHEXEC_CREATED`=18 around `ModelCudaGraphManager.capture` (`V/vllm/v1/worker/gpu/cudagraph_utils.py:307-374`, `:466-588`). Their ordinals are verified in the available official 13.0.85 header `/data/ycfeng/tmp/cupti13_sample_src/cuda_cupti-linux-x86_64-13.0.85-archive/include/cupti_callbacks.h:209,229,237` (SHA256 `b2f82221…1105`). Replay-time launch labels are a different, weaker mechanism and cannot by themselves name captured-node owners.

Capture drives the ordinary eager forward, and retained profile02 decode-node `rank0.before_profile.json` (`R/gpu-h800-0216…/serving_profile/`) **observes per-op Python wrapper entries while the current stream reports capture**. All 159,748 capture-labelled rows span 31 scopes; the **72,798 stream-capturing rows span 29 scopes**, with layers through 42 and counts `decoder_layer` 2,322, `attention_forward` 2,322, `mhc_fused_post_pre_tilelang` 4,590, `compressor`/`save_partial_states` 3,348 and `execute_in_parallel` 3,456. Of 49,035 breakable-active rows, 43,120 are also stream-capturing; 29,678 stream-capturing rows are not breakable-active. Every stream-capturing row has `graph_mode=null`, so 72,798 must not be relabelled as FULL-only. A callback can read the per-op scope **conditional** on delivery during that wrapper body, same-thread delivery, fresh TLS and the recording gate. Scope-entry flags do not witness a specific kernel or node; node→owner identity, native clone correspondence and completeness remain UNKNOWN.

Root census: [readable unchanged copy](native_capture_scope_census.json) of `/data/ycfeng/tmp/glm53_dsv4/native_capture_scope_root_3ov6bjro/receipt.json`, input SHA256 `147c37ec…3fa41`, Python3.10.6 under60s/30G; [limits check](native_capture_scope_limits.json)/[root report](test_report_2026-10-24_native_capture_prerequisite.md). The first31-vs29-scope assertion and root/reviewer EACCES are preserved; original bytes/ACL stay unchanged. Exact census script: `/data/ycfeng/tmp/glm53_dsv4/native_capture_scope_root_20261024.py` (`10a383a6…e71c`); first failure in `native_capture_scope_first_failure_z653su8w`.

Graph-replayed kernels, by contrast, expose only outer scopes (2,070 `model_dispatch_run_fullgraph`,10 `model_dispatch_run_pw_graph` serving rows and11 `replay` rows). PIECEWISE's eager attention segment has its own scopes, separated below; they do not identify graph-replayed per-op owners. A new replay-time launch probe is therefore **not** the minimum proof for captured per-op owners.

## Source provenance

- **Actual-image binding.** `/data/ycfeng/tmp/glm53_dsv4/profile04_root_image_58u02yq7/receipt.json`, case `image_entry` PASS: `docker run … --network=none --entrypoint python3 hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e853… -I -B -` with the profile04 stage bound read-only; witness `{"status":"PASS","python":"3.12.3","source_pins":87,"tool_pins":23,"binary_pins":20,"torch_vllm_imported":false,"cuda_initialized":false,…}`. Its stdin (`…/image_stdin.py:19-21`) hashes `bootstrap.PACKAGE.parent/<name>` where `PACKAGE=/usr/local/lib/python3.12/dist-packages/vllm` (`C/native_dummy_bootstrap.py:9`) — i.e. the **image's installed vllm tree**, all 87 pins. Caveat: the receipt's top-level `status` is FAIL because the later `position*` cases exhausted the 128-PID limit (`profile01/profile04/test_report_2026-10-24_profile04_root.md:37`); only case 0 passed, and the witness reports counts rather than a per-file table.
- **Configured-source binding (separate).** Re-hashing those 87 pins against `V` read-only: 86 match, 0 missing; `vllm/models/deepseek_v4/nvidia/model.py` matches the declared C26 overlay leaf `d3556a47056b…` (14-line load-time-only diff vs `V`). That local comparison is configured-source binding, not an image or worker rehash. The in-image case above is the binding witness.
- `breakable_cudagraph.py` `722683b44b44…`, `cudagraph_utils.py` `ee1f6eb37bc2…`, `model_runner.py` `fb08762e3a02…`, `config/vllm.py` `fb07f95658ae…`, `attention.py` `51add6614dc2…`, `gpu_worker.py`, `forward_context.py`, `profiler/wrapper.py`: all 87-pin verified in-image (count-level) and pin-equal to `V`.
- Collector: `W/tests/comparison/calibration/native_serve_profile.py` `c8f712177fcd…` ≡ `C/` and `profile04_package_W128/new_launcher_green/`; profile02's generation `ea810b91665c…` differs only by DP-sequence fields and the CUPTI SONAME fix, so its `recording()`/`range()`/`wrap()` semantics are the ones cited below.
- **Exact-image Torch API source.** `/data/ycfeng/tmp/glm53_dsv4/native_capture_graph_image_7xt30mlq/receipt.json` PASS: installed `torch/cuda/graphs.py` SHA256 `15b53590…ed81`, Python3.12.3, constructor `:107` defaults `keep_graph=False`; `raw_cuda_graph:217` requires True, while `raw_cuda_graph_exec:224` permits False after `capture_end`. This was AST/docstring inspection without torch/vllm import or CUDA execution; handle lifetime remains a runtime question.

## Configuration: what is observed, what is only selected

- **native06 (observed, retained log).** `…/nodes/gpu-h800-0346…/server.log:16` auto-enables breakable (`vllm.py:1313-1336`); `:41` prints `mode: CompilationMode.NONE`, `cudagraph_mode: FULL_AND_PIECEWISE`, `splitting_ops: []`; `:285-288` `Breakable CUDA graph enabled` in worker pids.
- **profile04 (selected; execution not yet verified).** Submitted `exp-1024-172225-512616`; the retained 09:49:36Z snapshot is `Starting`, four replicas, zero assigned nodes (`profile01/profile04/test_report_2026-10-24_profile04_root.md:25-26`). That is a dated observation, not a fresh platform read. Its executed compilation/capture state remains **UNKNOWN**.
- Source selection: `config/vllm.py:1419-1426` keeps `FULL_AND_PIECEWISE` legal without VLLM_COMPILE only because breakable is enabled; `breakable_cudagraph.py:90` reads the flag at import, and `attention.py` is imported lazily at model load.

## Reachable paths

1. **Eager NONE**: `model_runner.py:1655-1668` else-branch → `self.model(**model_inputs)`.
2. **Monolithic FULL capture**: `model_runner.py:848-897` → `cudagraph_utils.py:307-374`; one warmup `forward_fn(NONE)` `:337`, then `forward_fn(NONE)` inside `torch.cuda.graph(graph, self.pool)` `:365-366` → `ModelCudaGraphManager.forward_fn` calls `model(**model_inputs)` `:552`. Replay `:412-425` (`self.graphs[desc].replay()`), dispatched at `model_runner.py:1633-1635`.
3. **Breakable PIECEWISE**: `use_breakable_cg` `cudagraph_utils.py:139-142`, wrapper `:427-431`, branch `:550` → `:433-438`; capture `breakable_cudagraph.py:353-383` (`with capture: self.runnable(...)`); `_end_segment` stores `graph.replay` `:185-190`; `add_eager` ends the segment, runs `fn()` eagerly at capture and stores it `:195-205`; replay loops segments `:212-214`.
4. **Compiled PIECEWISE** (`run_pw_graph` non-breakable `:434-436`): not selected under `CompilationMode.NONE`; no Dynamo/Inductor partition, no attention `splitting_ops`.
5. Break point: only `_sparse_indexer_and_attn` (`attention.py:577`, called `:497`) under V2 (`_prepare_and_attn_fn = _prepare_and_attn` `:302`; `_prepare_and_attn_eager` `:405` is the MRV1 wide variant). The decorator is a no-op when no capture is current or mode is FULL (`breakable_cudagraph.py:95-102`).

## Where Python is current (observed rows in *italics*)

| Path | Python scopes current | Observed rows | A scope-reading callback would name |
| --- | --- | --- | --- |
| Eager NONE | per-op wrappers | 10,400 `eager_serving` rows (decode 0216) | innermost per-op scope |
| Capture warmup pass | whole forward | capture-labelled rows with `stream_capturing=false` | scope entry alone does not prove node creation or its absence |
| Non-breakable stream capture (FULL source candidate) | whole forward | *29,678 stream=true/breakable=false rows; 19 model forwards and 817 decoder/attention entries; no raw mode label* | per-op scope, conditional on the callback witness |
| Breakable capture segments | whole forward | *43,120 rows with both flags true; 49,035 total breakable-active rows also include non-capturing work* | per-op scope, conditional on the callback witness |
| Breakable break (`add_eager`) | `_sparse_indexer_and_attn` runs outside its segment | capture-labelled eager-segment rows need flags/path qualification; the same wrapper can execute inside FULL capture | eager-segment scope, not a node-ownership witness |
| FULL replay | `run_fullgraph` only | 2,070 rows, no per-op scope | outer dispatch only |
| PW replay, graph segments | `BreakableCUDAGraphCapture.replay` only | 11 `replay` rows | outer replay only |
| PW replay, eager segment | `_sparse_indexer_and_attn` | 473 `sparse_eager_segment` + 473 `sparse_attention` (all PIECEWISE) | attention eager segment + layer |

Capture rows carry `startup_warmup=true` and `graph_mode=null` (`context()` `native_serve_profile.py:157-172`), land in the `before_profile` archive emitted at `begin_window` (`:113-119`) before the lists are reset, and were retained with `dropped=0` against limits `{records:250000, bytes:268435456}`. The gate is `recording()` (`:105-111`): true while `window is None`, until `begin_window` opens the measured window — so capture work is archived outside the measured window, and a capture-time callback would write there, not into per-step rows.

## Four evidence classes (do not merge)

- **(a) Synchronous resource-created during capture** (`RESOURCE_GRAPHNODE_CREATED`): can name the innermost per-op/layer scope, because Python is current at the launch (rows above). Requires: delivery on the capturing thread, fresh TLS stack, recording gate open, and a launch→node correspondence that is not assumed from order.
- **(b) Clone/exec-created forwarding** (`GRAPHNODE_CLONED`, `GRAPHEXEC_CREATED`): the identified direct-witness candidate for forwarding captured node→owner metadata to an executable. Requires measured clone→source mapping; creation order is not identity. This is not a claim that no other mechanism could exist.
- **(c) Replay activity / full-ID joins**: kernel launches at replay mapped to graph-node/exec full IDs and joined to (a)/(b) records. Requires retained graph/exec IDs and a join key; no numeric remap.
- **(d) Replay-time launch labels**: name only the outer scope (`model_dispatch_run_fullgraph`, `replay`, or `sparse_eager_segment`), as observed.

## Existing mechanisms

`ServingProfile` installs 19 method scopes, 10 free functions, 2 launchers (`native_serve_profile.py:20-52`) at `load_model` (`:383-399`), re-decorates `_sparse_indexer_and_attn` with the real `eager_break_during_capture` (`:213-224`), re-points `_prepare_and_attn_fn` (`:377-378`), wraps `BreakableCUDAGraphCapture.replay` (`:357-359`, `:401-415`), and records `stream_capturing`/`breakable_capture_active` per row (`:157-169`). The frozen inventory additionally wraps `ModelCudaGraphManager.capture`, `run_fullgraph`, `run_pw_graph` and `torch.cuda.CUDAGraph.replay` (count only) at `native_full_structure.py:782-842`, `:858`. Capture-phase annotations already exist in retained output; no new subscription exists today.

## Unresolved (UNKNOWN)

1. Whether CUPTI delivers resource callbacks synchronously on the capturing thread with current Python state (mechanism declared, unmeasured).
2. Node-created records vs the scope stack at that instant (freshness) and whether every captured kernel yields an attributable node.
3. Clone relation and exec-created full IDs: the accepted tiny probe covers 9 nodes / 5 kernels in a synthetic graph only (`profile01/graph_probe/results/primary_acceptance.json`).
4. Completeness: unwrapped ops (e.g. `indexer_op`), ops between wrapper boundaries, library/custom-op internal launches, aux-stream kernels under `multi_stream_utils`.
5. profile04's executed capture set, decorator liveness and breakable graph segments. Profile02 observes eager-break wrapper entries, which alone do not prove a CUDA graph segment contains kernels.
6. A retained run/graph/exec key linking startup owner witnesses to formal replay after archive/reset. Capture-time identity metadata need not be collected as formal-window cost, but its correspondence must be directly witnessed.

## Smallest next proof (constrained to capture-time ownership + direct correspondence)

1. **CPU source-control (semantics only).** In the existing `W/tests/unit/test_native_serve_profile.py` stand-in seam, assert that `range()` sets and restores `self.local.scope` around the native call (`:179-206`), that `recording()` is true with `window=None` and false when closed (`:105-111`), and that a capture-phase row carries `kind='capture'`/`stream_capturing` from the real flags. This settles wrapper/gate plumbing only — **not** native worker import state, CUPTI delivery, or observed native execution.
2. **Retained fields, no cross-run join.** Inspect native capture/replay metadata for a shared run-qualified node/exec/owner key and record the missing direct witness. Profile02 annotation IDs and the accepted tiny probe's node IDs belong to different runs/namespaces and cannot be joined. The tiny probe can supply an example record structure, not native correspondence.
3. **Only if per-node ownership is required:** one descriptor's capture-time subscription to all three callbacks 13/20/18, with current scope/layer, directly witnessed source/clone/exec IDs and exact replay-kernel coverage. Handle the native default `keep_graph=False` lifetime without relying on post-capture raw-source enumeration; unsubscribe before Kineto. An implementation/resource allocation must be separately recorded under existing authority; this report selects neither code nor an automatic GPU run.

Absence of `sparse_eager_segment` rows is **inconclusive** (no PIECEWISE dispatch in the retained window, gating/retention/installation error, inert decorator, or label gap). Only positive rows are evidence, and only for the path they show.

## Retractions and limits

Retracted from v1: (i) "every captured node UNKNOWN" inferred from Python-free replay — capture-time Python is observed, so the capture witness is reachable though unmeasured; (ii) "absence settles UNKNOWN 1/2"; (iii) the unqualified "configuration actually executed" (now native06-observed vs profile04-selected-unexecuted); (iv) "image-bound" for a local hash comparison (now bound to the actual-image receipt, with its FAIL-status caveat). Prior retractions respected: captured kernels exist in profile02 (6,802,600 rows), `graph_mode None` is a labelling gap, creation order is not identity, reduced layers are not ruled out. No claim here about callback behaviour, kernel completeness, stream overlap, cost additivity, or operator/serving admission.
