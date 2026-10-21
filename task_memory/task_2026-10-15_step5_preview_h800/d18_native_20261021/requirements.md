# D18-NATIVE requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded the parent assignment, primary D18 decision and shared-client extension approval. |

## [Original Request]

Task D18-NATIVE works exclusively in `/data/ycfeng/Frontier/.worktrees/step5-native-pd`, branch `feat/step5-native-pd`. Read the current primary task requirements D18, spec and plan in the `step5-preview` worktree, not stale local root records. The user adopted GLM/V4 `68f891b`, locally `958d81a`.

Implement native consumption of the common 128 requests of 6144/1024 tokens at QPS 1, with 16 separate same-shape warmups per each of three repetitions. Stable template IDs seed prompts; transport IDs are unique across phases/repetitions. Stream real generated token-ID deltas if supported by the locked Step API. Report mean client TTFT, TPOT and per-request E2E, and prompt-plus-output throughput from first actual formal submission to last formal output. No R2 hooks in clean serving. Reuse the existing Step native PD machinery and tested GLM contracts; do not create a second HTTP replay client or bulk-copy the foreign client.

Source-derive H200 limits and graph settings: propose and verify 7168 context, 8192 batched-token budget, 128 sequences; chunked prefill, prefix caching, Thinking Mode and MTP off; actual decode graph replay required. Stage new offline-verified qualification/canonical packages with resource estimates and explicit H200 runtime dependencies. Do not reuse R1's 512-token/one-sequence serving limits.

Own only Step native consumer/run configuration, affected tests, and this local directory. Preserve frozen common, historical Step and SharedGPT packages; simulator core, registries, GLM worktree, sealed stages, root task documents and the parent-owned R2 verdict/observations are out of scope. No GPU, authentication, platform, cloud, SSH or Docker operations. Parent obtains missing private logs. Validators on 32+ GPU workloads are nonblocking diagnostics and never stop healthy native work.

Run meaningful HTTP/consumer tests for counts, stable prompts, actual sends and timestamps, plus canonical pin regressions. No full simulation sweep for harness-only changes. Commit scoped verified changes; no push or merge to main. Return task ID, commits/files, evidence/commands, stage path, exact parent submission command, GPU cost and remaining blockers.

## [Original Request / Follow-up Q&A]

Question: the existing `tests/comparison/dp_placement_pp/vllm_replay.py` lacks opt-in `return_token_ids` and token-delta timestamps/counts. Is a minimal shared-client extension, preserving other adapters' defaults and reusing existing replay, authorized despite the initial ownership boundary?

Answer: **Authorize minimal extension** — extend the existing shared client for opt-in canonical token-ID streaming and per-request failure retention, with regression tests. The same existing native `server_command` also passes the source-supported built-in graph-statistics flag; no new native observation hook is introduced.

## Acceptance and boundaries

- Shared manifest SHA-256: `5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3`; SHA256SUMS SHA-256: `594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9`.
- Every canonical repetition sends 16 warmup + 128 formal requests; 432 total, 384 formal, 48 warmup. Every formal population has 917504 total prompt-plus-output tokens.
- Native usage and returned prompt/token deltas must corroborate exact lengths and prompt identity. A missing/broken request invalidates the full-population metric, not the remaining requests.
- Both roles expose fresh running/waiting/KV and completion metrics for all four DP lanes at phase boundaries. Observation failure marks the reference unverified, continues subsequent work, and never silently accepts a clean boundary.
- Offline/source checks do not qualify H200 capacity, NIXL content/lifetime, lane coverage, measured latency or actual CUDA graph replay. Those remain runtime dependencies owned by the parent.
