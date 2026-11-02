# W128-GRAPH-ATTEMPT3-RCA: pending source/ABI findings

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Saved unfinished source-only RCA findings when the W128-MAINLINE-CHECK checkpoint took priority. RCA paused, not complete. |
| 2026-10-27 | H1 confirmed from NVIDIA's CUPTI 13.0.85 `cuda_graphs_trace` sample (section "Verdict"); open item 1 closed. |

## Status

Source RCA done on 2026-10-27: H1 is confirmed by NVIDIA's own sample (see "Verdict"). Native confirmation is still open (item 2). The text below the verdict is the 2026-10-21 record, kept as written.

2026-10-21 status: PAUSED, not finished. No source changes, GPU jobs, platform, SSH, cloud or network operations. Frozen package inputs were not modified.

## Verdict (2026-10-27)

Evidence, retrieved by Pi run `cupti-graph-sample-1027a` (public NVIDIA downloads, no GPU) and checked by the lead:

- Manifest `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json` names `cuda_cupti` 13.0.85, archive `cuda_cupti/linux-x86_64/cuda_cupti-linux-x86_64-13.0.85-archive.tar.xz`, sha256 `92fb7e0430521517174c4c171173b888aaa8d78d8762e59752dea2ed14a0ad7d`. The downloaded archive has this sha256 (lead re-hash).
- The archive's `include/cupti_callbacks.h` (b2f82221...1105) and `include/cupti_activity.h` (914cd60f...3fff) are byte-identical to the locked wheel headers used above.
- The archive's sample `samples/cuda_graphs_trace/cuda_graphs_trace.cu` (sha256 `0458254b6d8ada6c6db82492f14f6bf029f70cafaf97a07dfc6168c95dc312f8`, 339 lines) reads the payload this way:

```c
227:            CUpti_ResourceData *pResourceData = (CUpti_ResourceData *)pCallbackInfo;
...
233:                    if (!strncmp(s_pFunctionName, "cudaGraphInstantiate", strlen("cudaGraphInstantiate")))
...
237:                    CUpti_GraphData *callbackData = (CUpti_GraphData *) pResourceData->resourceDescriptor;
...
241:                    CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->node, &nodeId));
...
250:                    CUpti_GraphData *callbackData = (CUpti_GraphData *) pResourceData->resourceDescriptor;
...
254:                    CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->originalNode, &originalNodeId));
...
258:                        CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->node, &nodeId));
```

  Kernel and memcpy activity records are joined to the stored nodes by `graphNodeId` alone (lines 181 and 194, `CUpti_ActivityKernel10`).

Conclusions:

- H1 holds. For the RESOURCE domain, `cbdata` is a `CUpti_ResourceData *` (header lines 407-441 and 645-648), and the `CUpti_GraphData *` is its `resourceDescriptor` field, although the header still calls that field "Reserved for future use". The probe cast `cbdata` itself (O1). It therefore read `node` from offset 16, which is the `resourceDescriptor` pointer, and `originalNode` from offset 24, past the end of the 24-byte struct (O4). The fault at line 197 is the query on that out-of-bounds word (O5). The probe's GRAPHEXEC_CREATED branch has the same misread (`graph` at offset 0 is `context`, `graphExec` at offset 48 is out of bounds).
- H2 is not needed to explain the fault. NVIDIA's sample calls `cuptiGetGraphNodeId` inside GRAPHNODE_CLONED, so the query is legal in that callback.
- Correction for any reuse of the method (stage 1a): read `CUpti_ResourceData`, take `CUpti_GraphData` from `resourceDescriptor`, and fail with a clear error when it is null in a graph callback. The ctypes `CuptiGraphData` layout itself was correct (O3).
- Limit: this is source evidence. The corrected reader has not run on a GPU. Attempt2's exit 139 is still not assigned (item 3).

Full Pi notes: `/data/ycfeng/tmp/ds4_runs/cupti-graph-sample-1027a/scratch/FINDINGS.md`. It says the node query received the `CUstream` value; that is wrong by one field (offset 16 is `resourceDescriptor`, the stream is at offset 8). The verdict above uses the lead's reading.

## Evidence used

- Retained fault: `observed/fault.log` (`_read` line 119 <- `node_id` line 125 <- `_callback` line 197 <- `torch/cuda/graphs.py:180 instantiate`). The last stage in `observed/stages.jsonl` is sequence 30, `instantiate.enter`.
- Probe source `probe_diagnostic.py`, SHA256 f7314cfb... (re-hashed locally, matches the manifest).
- Locked CUPTI headers: local wheel `nvidia-cuda-cupti==13.0.85` at `/data/ycfeng/envs/oneshot-opd-py312/lib/python3.12/site-packages/nvidia/cu13/include/`. This is the same version as `runtime_lock/runtime_lock.json` and the attempt1 worker report (`cupti_version 13.0.85`). The bytes match the wheel RECORD: `cupti_callbacks.h` sha256 b2f82221...1105 (34092 B), `cupti_activity.h` 914cd60f...3fff, `libcupti.so.13` e2f9ed86...be3e. This shows wheel identity, not a byte read from the image.

## Observations (O)

- O1. In the probe, the callback casts `cbdata` directly to `CUpti_GraphData` (`probe_diagnostic.py:189`). It reads `node` and `originalNode` (line 192) and checks only that they are non-null (line 194). It then calls `cuptiGetGraphNodeId` on `node` (line 196) and on `originalNode` (line 197).
- O2. The header defines the callback payload type by domain. At `cupti_callbacks.h:645-648`: "If domain is CUPTI_CB_DOMAIN_RESOURCE the type of cbdata will be CUpti_ResourceData." `CUpti_ResourceData` (lines 418-441) has three pointer-size fields: `context`, `resourceHandle` (a union holding only `CUstream`) and `resourceDescriptor`, documented as "Reserved for future use".
- O3. The documentation of `CUpti_GraphData` (lines 471-512) says it is passed "as the cbdata argument" for the RESOURCE domain. The same header therefore gives two conflicting payload types. The struct layout itself is consistent: offsets 0/8/16/24/32(int)/40/48, size 56. This matches the probe's ctypes struct (lines 72-77) on x86_64.
- O4. If `cbdata` is really `CUpti_ResourceData` (24 bytes), the probe reads `node` at offset 16, which is `resourceDescriptor`. It reads `originalNode` at offset 24, which is past the end of the struct. Both values can be non-null, so the line-194 guard does not reject them.
- O5. Line attribution: compiling the same-SHA source with CPython 3.12.13 puts the `node_id(row["node"])` CALL at offset 516 on line 196 and the `node_id(row["original_node"])` CALL at offset 574 on line 197. Under the frame-line semantics of CPython 3.12, the faulthandler line 197 therefore identifies the **original-node query** as the active call. In the same callback invocation the node query had already returned success; a nonzero result would have raised at line 121 instead. Limits: the worker ran Python 3.12.3 (same 3.12 bytecode format, not verified on the worker), and earlier callback invocations are unrecorded because no report exists.
- O6. `cupti_activity.h:7368-7380` documents only a NULL check for `cuptiGetGraphNodeId`. A non-null invalid handle is not validated, so a crash on such a handle is not excluded. No grep match in the retained headers restricts CUPTI ID queries inside resource callbacks.
- O7. In attempt1 (no callback subscription), `cuptiGetGraphNodeId` on genuine `cudaGraphGetNodes` handles completed (U2 PASS) in `/data/ycfeng/tmp/glm53_dsv4/h200_graph_probe_20261021/w128_graph_probe_cuda13_20261021/probe_report.json`. The ID API itself works on this runtime with valid handles.

## Hypotheses (H), ranked

- H1 (leading, medium confidence): payload type is misread. Resource-domain `cbdata` is `CUpti_ResourceData`, and the graph data sits behind `resourceDescriptor`. This is NVIDIA's public CUPTI sample usage (`cuda_graphs_trace`), recalled from memory; it is **not retained locally** and was not found by local search. Under H1, line 197 passes an out-of-bounds stack word as a `CUgraphNode`, which fits O4 and O5.
- H2 (lower): the payload is `CUpti_GraphData` as the struct doc says, and the crash comes from a CUPTI-internal condition when the ID is queried during GRAPHNODE_CLONED inside instantiate (reentry or state). No retained evidence supports or excludes this.
- Handle lifetime is not a likely cause. The queries run inside the callback invocation, which is within the documented validity window (`cupti_callbacks.h:408-416`).

## Minimal correction proposal (not applied, needs review)

Read `cbdata` as `CUpti_ResourceData` and take `CUpti_GraphData *` from `resourceDescriptor` before reading node, originalNode, graph or graphExec. Add no other change. Before any GPU use, retain an authoritative NVIDIA source (the CUPTI 13.0 `cuda_graphs_trace` sample or official docs) as a byte-pinned file. CPU tests cannot validate the native payload.

## Remaining open items

1. Closed 2026-10-27: the NVIDIA sample is pinned by sha256 in "Verdict"; the archive is kept at `/data/ycfeng/tmp/ds4_runs/cupti-graph-sample-1027a/scratch/`.
2. Any native confirmation needs a new, explicitly approved GPU attempt. The 3/3 budget is exhausted.
3. Attempt2's exit139 is still not assigned to this cause.
