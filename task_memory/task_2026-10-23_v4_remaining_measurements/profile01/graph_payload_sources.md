# CUPTI 13.0 RESOURCE graph callback payload: authoritative sample and header

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Root corrected inference/signature wording and confirmed that the old probe already labels source scopes through event handles and DAG edges; CREATED callbacks are not a required repair. |
| 2026-10-24 | Corrections: official `cuda_graphs_trace` source obtained; retracted "no graph payload", "type mismatch documented", "API callbacks not resource payload", and the multi-GB-only download claim. |
| 2026-10-24 | First version: docs/header text only, sample bytes unavailable. |

Task ID: `W128-GRAPH-PAYLOAD-SOURCE`. Source-only. No install, no execution, no CUDA init, no callback subscription, no GPU/platform/cloud/SSH, no code or Git writes.

## Retraction of my earlier claims

The earlier report said CUPTI "assigns no graph payload" to RESOURCE callbacks, that the cast was a "documented type mismatch", and that the "documented mechanism is API callbacks, not resource payload". Those are **withdrawn**: the official 13.0.85 sample reads the payload through `CUpti_ResourceData::resourceDescriptor`. The payload comments and concrete sample usage disagree; actual runtime payload behavior still needs validation.

## Authoritative acquisition (verified)

- Manifest: `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.1.json` → HTTP 301 → `developer.download.nvidia.cn` → 200, 47792 B, sha256 `9c494bc13b34e8fbcad083a6486d185b0906068b821722502edf9d0e3bd14096`; `cuda_cupti.version` = **13.0.85**; linux-x86_64 `relative_path` = `cuda_cupti/linux-x86_64/cuda_cupti-linux-x86_64-13.0.85-archive.tar.xz`, `size` `16253056`, `sha256` `92fb7e0430521517174c4c171173b888aaa8d78d8762e59752dea2ed14a0ad7d`. All three values match root's finding exactly.
- Archive downloaded from the redirect host: **16253056 B**, sha256 `92fb7e0430521517174c4c171173b888aaa8d78d8762e59752dea2ed14a0ad7d` — **published size and digest both match**. 491 entries.
- **Correction:** the earlier "multi-GB toolkit only" statement was a search limit, disproved by the primary manifest. A standalone15.5MiB CUPTI archive exists and matches the publisher's SHA256; no cryptographic signature was verified.
- Extracted (source only): `samples/cuda_graphs_trace/cuda_graphs_trace.cu` 339 lines sha256 `0458254b6d8ada6c6db82492f14f6bf029f70cafaf97a07dfc6168c95dc312f8`; its `Makefile` sha256 `2fb6fc8e6556b9f06a11ad35fdf3db1dc52d68969921d6c412a04637da51782e`; `include/cupti_callbacks.h` sha256 `b2f82221fb03ecb2b2a27afa2d64284babdf49fee1a3a2811258d473285e1105`; `LICENSE` 67875 B sha256 `f176e5fd245a4ff1d7aee743e345e14b29c4b724f150e4858c4c6a313966d1fb`.

## What the official sample does (quotes)

1. Payload indirection, `cuda_graphs_trace.cu:227` then `:237`:
   `CUpti_ResourceData *pResourceData = (CUpti_ResourceData *)pCallbackInfo;`
   `CUpti_GraphData *callbackData = (CUpti_GraphData *) pResourceData->resourceDescriptor;`
   Same pair repeated for the clone case at `:250`.
2. ID queries are **inside** the callback, `:241`, `:254`, `:258`:
   `CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->node, &nodeId));`
   `CUPTI_API_CALL(cuptiGetGraphNodeId(callbackData->originalNode, &originalNodeId));`
3. Ownership map is built on node **creation**, `:242-245`, keyed by node ID and valued with the enclosing runtime API name/correlation captured on `CUPTI_API_ENTER` at `:272-276` (`s_correlationId = pCallbackInfo->correlationId; s_pFunctionName = pCallbackInfo->functionName;`).
4. The clone callback propagates the original node's attribution to the cloned node's ID, `:254-261` (`erase(it); nodeIdCorrelationMap[nodeId] = apiData;`). This is a direct handle/ID relationship rather than arithmetic recombination.
5. Final attribution joins activity `graphNodeId` to that map, `:175-177` and `:193-195` (`nodeIdCorrelationMap.find(pKernelRecord->graphNodeId)`).
6. The creation branch handles an anticipated instantiate-time case by skipping it, `:232-236`: `if (!strncmp(s_pFunctionName, "cudaGraphInstantiate", ...)) break;`. This source branch is not a runtime observation.
7. Enabled callbacks only, `:304-306`: `CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED`, `CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED`, plus the runtime-API domain.

Header facts (same official bytes): `cupti_callbacks.h:426,435,438-441` — `CUcontext context; … } resourceHandle;` then `void *resourceDescriptor;` documented "Reserved for future use"; `:487-512` `CUpti_GraphData` holds `graph, originalGraph, node, originalNode, nodeType, dependency, graphExec`. `:412-415` callback data is valid only within the invocation. The sample provides concrete usage despite the conflicting field comment; it does not establish post-callback handle lifetime.

## Bearing on attempt3 (hypothesis, strongly source-supported)

Attempt3 cast `cbdata` directly to `CUpti_GraphData` (`probe_diagnostic.py:189`) without `resourceDescriptor` indirection. If the runtime supplies the sample's envelope, the direct cast reads `node` from offset16 (the descriptor pointer) and `originalNode` from offset24 (past `CUpti_ResourceData`). Passing these values to `cuptiGetGraphNodeId` is a source-supported hypothesis for exit139. The actual invalid values and a corrected native execution have not been observed; the crash cause remains unconfirmed.

The earlier "wrong callback set" claim is withdrawn. Root inspected old `probe_diagnostic.py:359–420,467–480`: it already assigns source scopes using four retained event handles and explicit DAG reachability. CREATED/runtime callbacks are an alternative attribution method used by the NVIDIA sample, not a necessary repair to this existing probe. The minimal proposed correction preserves its scope method, callback set and query order and changes only payload decoding.

## Remaining limits

Sample never mentions stream capture (`grep -i capture` → no match); its graph is built with `cudaGraphCreate`/`cudaGraphAddKernelNode` (`:97-136`). So it does **not** prove that a captured `cudaLaunchKernel` emits `GRAPHNODE_CREATED` with a usable enclosing correlation, nor that instantiate of a captured graph emits the clone linkage. Those remain the unknowns a probe would test. Sample usage also does not prove the driver passes this layout at 580.95.05, and nothing here settles native serving ownership.

## Decision

YC selected Option A under D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY: correct payload indirection in a fresh copy of the existing event/DAG probe, preserve its callback set and in-callback queries, then at most1×H800/600s after CPU checks. Existing3/3 failures remain historical. Similar bounded diagnostic work needs no repeated approval. Native serving ownership remains unverified.
