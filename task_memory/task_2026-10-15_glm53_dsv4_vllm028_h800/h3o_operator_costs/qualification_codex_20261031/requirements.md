# P2 H200 qualification investigation requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Created the authorized read-only P2 qualification investigation record. |
| 2026-10-31 | Recorded continuation limits and the lead-owned UUID/F2/links/A6/A10 work. |

## [Original Request] 2026-10-31

B30b integrated as main 5d1754d0 and merged 163 tests PASS. Continue as a bounded parallel sidecar on Tickets05/09/11/18/19: audit only the remaining H200 native NIXL packed payload, non-KV memory/capacity, and promotion/admission qualification for P2-32/P2-64. Read current main requirements/spec/plan and exact existing artifacts from /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 (current route section controls). Known findings: native one packed region 1,002,240 B/block, T6144 46 blocks=46,103,040 B vs simulator 39,521,664 B/lane; F4 planner116034/120623 vs native109934/112914; p2_32r/p2_64r original R4 PASS; no retroactive prelaunch promotion. Need a concrete criterion/evidence table stating which hardware/payload/memory/promotion prerequisites are actually satisfied and which still missing, with exact sources and minimal next actions within standing scope. Do not duplicate A6 evaluation, F1a/A10 comparisons, GPU probes, broad trace census, or B30b implementation. No GPU/platform/cloud operations; no simulator/source repair; no change to main docs or old receipts. Write only task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/qualification_codex_20261031/ in your own worktree. Reports in English with modification history. Commit scoped report/evidence in your branch when verified and list changed paths/hash. This is an investigation deliverable, not permission to claim final qualification or expand core.

## Acceptance and limits

- Read the current main route and exact retained evidence; classify hardware, packed payload, non-KV capacity and promotion separately.
- Produce a criterion/evidence table with source locations, measured values, remaining gaps and minimal next actions.
- Preserve historical R4 admission and launch receipts; do not infer final qualification from numerical comparisons.
- Verify retained-source hashes, arithmetic, evidence references and the scoped report commit.
- Only this local directory is writable. All implementation, GPU/platform/cloud work and other active comparisons are excluded.

## [Original Request / Follow-up] 2026-10-31

We resumed after compaction on the same task. Please continue your existing packed-KV/non-KV memory/capacity and promotion/admission audit in h3o_operator_costs/qualification_codex_20261031/ only. Do not duplicate UUID fix/GPU/F2/A6/A10 work. Links/KV01 Failed due to bare torch UUID versus GPU-prefixed inventory; F2 Succeeded and main rollout will analyze it, then repair selector and submit fresh links/KV02. Return current status and a scoped commit when ready; preserve diagnostic vs formal qualification and unchanged open tickets 05/09/11/19.

The reported GPU attempt states are user-provided routing context. This sidecar does not monitor or diagnose those jobs. Its acceptance is an evidence/report commit, not final qualification, permission to promote an input, or a new GPU authorization request.
