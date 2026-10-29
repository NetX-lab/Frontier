# Test report: H3o-(b) stage 1a library (graph-capture ownership and replay attribution)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: review and checks of Codex assignment A, with the lead's A2 correction. |

## 1. Change under test

Codex assignment A ([h3o_operator_costs/codex_assignment_A_stage1a_library.md](h3o_operator_costs/codex_assignment_A_stage1a_library.md), request `c7c481b4-39d6-4412-bb79-24d4c8e6d14a`, report [h3o_operator_costs/codex-implement-c7c481b4-39d6-4412-bb79-24d4c8e6d14a/implementation.md](h3o_operator_costs/codex-implement-c7c481b4-39d6-4412-bb79-24d4c8e6d14a/implementation.md), outcome `implemented`). Design: [h3o_operator_costs/stage1a_replay_route.md](h3o_operator_costs/stage1a_replay_route.md) §3.1-§3.3.

Four new files, no existing file changed:

- `frontier/profiling/attention/graph_capture.py` (513 lines): the capture record, the CUDA graph queries, the CUPTI clone recorder and the capture-time scope recorder.
- `frontier/profiling/attention/graph_replay_trace.py` (363 lines after the correction): step annotations and `attribute_replay_trace`.
- `tests/unit/test_graph_capture.py` (438 lines) and `tests/unit/test_graph_replay_trace.py` (359 lines after the correction).

## 2. Lead review

Checked against the assignment and the vLLM source at native commit `2cf0a69`:

- Sweep order: `capture_end` sweeps, clears the open capture, then calls the original, so the capture graph is read before instantiation drops it, and clone callbacks fire after the final sweep. Scope boundaries sweep before the push and before the pop.
- Dependency ids: all new node ids are assigned before any dependency is resolved, so reverse node enumeration resolves (test `reverse node enumeration`).
- Eager wrapper: vLLM `add_eager` (`compilation/breakable_cudagraph.py:198-212`) ends the segment, which appends the graph replay, runs `fn`, appends `fn`, then begins a new segment without appending. The wrapper's check that `segments[-1] is fn` and its replacement at that index therefore match the native order. `current()` exists (`breakable_cudagraph.py:140-142`) and is set before the first `_begin_segment`, so every Breakable segment has its capture ordinal.
- CUPTI: `cbdata` is read as `CUpti_ResourceData`, the graph data through `resourceDescriptor`, as in NVIDIA's sample. Callback errors are deferred to exit. A failed unsubscribe keeps the recorder in `_ACTIVE_SUBSCRIPTIONS`, so the native callback pointer stays alive.
- Replay: complete-launch checks, raw-or-demangled name check, grid and block check, nested ranges with partial-overlap and identical-range failures, and the iterative dependency traversal are as designed.

Defects corrected by the lead (small, in scope):

1. A2 container rule. Codex implemented the rule of the first design note: containers are the parents of capture-time scope entries and the enclosing ranges. That rule is wrong for `dsv4_index_topk`, which has the child `dsv4_index_scores` and still owns the top-k fill, top-k and packing kernels of the native indexer op, which cannot be wrapped (design §5: the nodes a scope adds outside its child scopes belong to it). Correction: `attribute_replay_trace(events, record, container_scopes)` takes the declared container scope names of the integrator's tree; `_RangeIndex` no longer collects containers. Design note §3.3 and the new §3.5 record the rule and the tree.
2. Checks for states the code cannot reach, removed (project rule): the dependency traversal's "absent dependency", "unsupported pass-through type" and cycle checks (every activity node of a launch is already required to be present, dependencies come from the same captured graph, and CUDA graphs are acyclic), the clone-map cycle check (a clone id is always new), and the then-unused `_PASS_NODES`.

Accepted risks, to be settled by the bridge check: `CudaGraphQueries` uses the cuda-bindings 13 return shapes of `cudaStreamGetCaptureInfo` and `cudaGraphNodeGetDependencies` (inspected in 13.4.1; the target image has 13.3.1); `__cxa_demangle` is loaded through `ctypes.util.find_library("stdc++")`; a clone whose original was not recorded fails attribution even if it is never replayed. The B1 driver keeps the capture record and the traces, so a library correction can be re-checked on CPU without a new GPU run.

## 3. Environment

- Python 3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python` (no torch, no cuda-bindings).
- `PYTHONPATH=$PWD:$PWD/tests/unit`, `PYTHONDONTWRITEBYTECODE=1`, `ulimit -v 31457280`.

## 4. Checks (rerun by the lead after the correction)

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `python -m pytest tests/unit/test_graph_capture.py tests/unit/test_graph_replay_trace.py -q -p no:cacheprovider` | all pass | 59 passed in 0.34 s (Codex: 57; the correction adds the two-case test that a scope with a child owns its leftover kernels unless declared a container, and drops the clone-cycle case) | PASS |
| 2 | `python -m pytest tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsv4_composition.py -q -p no:cacheprovider` | 34 passed (baseline) | 34 passed in 2.49 s | PASS |
| 3 | import both modules and assert that no `torch` or `cuda` module is loaded | no error | `PASS: no torch or cuda module imported` | PASS |
| 4 | Mutation: replace the A2 condition by `False`, run the replay tests, restore from `/data/ycfeng/tmp/graph_replay_trace.py.bak` | the A2 tests fail | 4 failed (both A2 tests, graph and eager), 33 passed; after the restore the condition count is 1 and 37 passed | PASS |
| 5 | `wc -l` of the four files | each at most 2,000 lines | 513, 363, 438, 359 | PASS |

## 5. Observations and inferences

Observations: the checks above. Codex's first test run had one failure (a fixture's clone ids collided with captured ids); its final run and the lead's rerun pass.

Inferences: the CPU tests establish the ownership algorithm, serialization, the native payload layout against in-memory structs, and the attribution rules on synthetic traces. They do not establish GPU correctness.

## 6. Limits

- NOT RUN: `CudaGraphQueries` against CUDA, CUPTI clone callbacks during native graph instantiation, and the reduced-engine FULL and PIECEWISE bridge check (P1h, after assignment B1).
- 0 GPUs used.
