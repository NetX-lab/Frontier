Assignment and outcome

Implemented assignment A of stage 1a in `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. The objective was a CPU-tested library for capture-time operator ownership, the CUPTI clone bridge, and replay-trace attribution. The assigned edits and checks are complete; review, acceptance, integration, commits, and task records remain with Claude.

The writable scope was `frontier/profiling/attention/` and `tests/unit/`, restricted to creating exactly the four named files. All four were absent before implementation. No existing source file was modified. The assignment-reported untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/` were not touched; Git operations were not performed.

Read the applicable repository AGENTS.md instructions, `stage1a_replay_route.md` sections 2 and 3.1–3.3, the parent `design.md` sections 3.3 and 5, the probe implementation and RCA Verdict, the specified vLLM Breakable capture implementation, and the local CUPTI headers. No applicable CLAUDE.md was found. The explicit assignment governed the bounded implementation and checks.

Implemented the settled ownership route: sweep before scope-stack transitions and before original capture_end, retain eager-break registration owners for replay, read graph clone payloads through CUpti_ResourceData.resourceDescriptor, and require clone identity and kernel metadata matches at replay. Ordinary implementation choices include zero-based capture ordinals, immutable typed record rows, half-open annotation ranges, and rejection of ambiguous identical nested ranges or launch correlations.

Changes

- New `frontier/profiling/attention/graph_capture.py` — 513 lines. Defines `ScopeOwner`, frozen `KernelLaunch`, `GraphCapture`, `GraphNode`, `ScopeEntry`, `EagerBreak`, `NodeClone`, and `CaptureRecord` (lines 39–111). `CaptureRecord.to_json/from_json` implements the specified raw-record structure, including list-form owners and dimensions, and rejects duplicate captured or clone node ids. `GraphQueries` begins at line 152. `CudaGraphQueries` at line 195 lazily imports torch and cuda-bindings, checks CUDA errors by API name and code, queries names through function/kernel handles, and demangles with buffer release. `resolve_cupti_library` at line 163 requires one distinct mapped CUPTI library path and is shared with the clone recorder. The native ctypes structs and `parse_clone_payload` begin at lines 284–298. `CuptiCloneRecorder` at line 311 owns its subscription, diagnoses error 39, defers callback errors until exit, and retains callback lifetime if unsubscribe fails. `GraphCaptureRecorder` at line 377 implements class patches and restoration, capture ordinals, scope sweeps, dependency-id assignment before dependency resolution, and eager-break replay wrappers.
- New `frontier/profiling/attention/graph_replay_trace.py` — 369 lines. Defines the step annotation API at lines 25 and 61–65, and frozen `ReplayInterval`/`ReplayAttribution` at lines 36 and 53. `attribute_replay_trace` at line 204 scans events once and uses sorted per-thread range indices. It validates clone mapping, one capture and exactly one activity per required node in each graph launch, raw/demangled names, grid/block values, nested eager scopes, and container ownership (A2). It reports zero-kernel scopes, kernel-only milliseconds and counts, layer windows over all streams, outside kernel counts, and unique unattributed launches. Dependency gaps use iterative memoized traversal through event/empty nodes, including memcpy/memset ends. It imports the existing ScopeKey, ScopeKernelTime, annotation parser, and trace loader; `dsa_kernel_trace.py` remains unchanged.
- New `tests/unit/test_graph_capture.py` — 438 lines. Stream-aware graph fakes mirror capture begin/end and vLLM's segment/eager-break ordering. Coverage includes FULL nesting and outside ownership, final sweep ordering, warm-up behavior, reverse node enumeration and dependency ids, PIECEWISE ownership, replay-only eager wrappers, patch restoration, JSON round trips, duplicate ids, native ctypes payload layout and descriptor indirection, deferred callback errors, subscriber conflicts, cleanup failures, and mapped-library resolution.
- New `tests/unit/test_graph_replay_trace.py` — 343 lines. Synthetic Kineto traces cover FULL and PIECEWISE joins, raw/demangled names, identity mismatches, captured-id hints, incomplete and duplicate activities, multiple-capture rejection, nested/repeated eager ranges, A2, zero-kernel scopes, outside and unmeasured launches, stream-spanning windows, memcpy/memset dependency gaps, root gaps, invalid ranges, thread and step seams, correlation failures, clone chains, and a dependency chain deeper than Python's usual recursion limit.

No file-scope deviation or unresolved design blocker was encountered. The capture module is somewhat above the approximate 300–450-line estimate, but all four files are well below the 2,000-line limit.

Verification

All checks ran from the working root using `/data/ycfeng/envs/frontier-py310/bin/python` (the assigned Python 3.10 environment). `TMPDIR` was exported as `/data/ycfeng/tmp/codex-implement/c7c481b4-39d6-4412-bb79-24d4c8e6d14a`; bytecode writes and pytest's cache provider were disabled.

1. PASS — new tests:

```bash
PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_graph_capture.py tests/unit/test_graph_replay_trace.py -q -p no:cacheprovider
```

Expected: all new tests pass. Final observed output tail:

```text
.........................................................                [100%]
57 passed in 0.26s
```

The first run reported `1 failed, 56 passed in 0.29s`: the deep-dependency test's synthetic clone offset collided with captured ids. The fixture was corrected to use disjoint ids; the final run above passed.

2. PASS — existing baseline:

```bash
PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsa_attention_profiling.py tests/unit/test_dsv4_composition.py -q -p no:cacheprovider
```

Expected: 34 passed. Observed output tail:

```text
..................................                                       [100%]
34 passed in 1.83s
```

3. PASS — module import:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -c "import frontier.profiling.attention.graph_capture, frontier.profiling.attention.graph_replay_trace"
```

Expected: no error without torch or cuda-bindings. Observed: exit 0, no output. An additional executed assertion confirmed import isolation:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python -c 'import sys; import frontier.profiling.attention.graph_capture, frontier.profiling.attention.graph_replay_trace; assert not any(name == "torch" or name == "cuda" or name.startswith("torch.") or name.startswith("cuda.") for name in sys.modules); print("PASS: no torch or cuda module imported")'
```

Observed: exit 0 and `PASS: no torch or cuda module imported`.

4. PASS — file sizes:

```bash
wc -l frontier/profiling/attention/graph_capture.py frontier/profiling/attention/graph_replay_trace.py tests/unit/test_graph_capture.py tests/unit/test_graph_replay_trace.py
```

Observed:

```text
  513 frontier/profiling/attention/graph_capture.py
  369 frontier/profiling/attention/graph_replay_trace.py
  438 tests/unit/test_graph_capture.py
  343 tests/unit/test_graph_replay_trace.py
 1663 total
```

Read-only API inspection used `/data/ycfeng/envs/oneshot-opd-py312/bin/python` to inspect cuda-bindings signatures/docstrings for capture info, graph nodes, node types, dependencies, kernel parameters, and function/kernel names. No CUDA API calls or GPU operations were executed.

Remaining work and decisions for Claude

No assigned work remains unfinished and no blocker requires a scope decision. Observed CPU evidence supports the ownership algorithm, serialization, payload decoding against in-memory native-layout structs, and replay attribution. It does not establish real GPU correctness.

NOT_RUN: `CudaGraphQueries` against CUDA, corrected CUPTI callbacks during native graph instantiation, and the reduced-engine FULL/PIECEWISE bridge check. Those remain for assignment B and the separately authorized native validation. The inspected local cuda-bindings version is 13.4.1; compatibility with the target image's 13.3.1 has not been exercised.

Review handoff

Claude should inspect the four new files, especially sweep ordering before original capture_end, dependency-id assignment, eager-break wrapper placement, CUPTI descriptor indirection and subscription cleanup, nested range indexing, launch completeness checks, and graph-gap traversal. Rerun the two pytest commands and the import check during acceptance. Later native validation must verify clone ids, kernel names/grid/block values, real capture-stream queries, and PIECEWISE eager-break registration before treating replay ownership as GPU-proven.
