# D24-G3-STATE-VALUES plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Defined the minimal retention edit, real-seam regressions and parent-only staging handoff. |

## Scope and dependency graph

`1 (base/source inspection) -> 2 (bounded retention + regression cases) -> 3 (verification/checkpoint) -> 4 (commit/handoff)`.
No delegation. Tests and task records may be prepared independently; native execution and numerical acceptance are outside this slice.

1. Verify requested base and clean worktree. Read existing reference, worker caller, caps, stage pins and focused tests.
2. Keep `raw_hex` only at `first_local_tail_complete` on computed-token logical page and predecessor; retain existing full mean wire rows at that boundary. Reuse current diagnostic and source-pin paths; avoid flags if existing boundaries already scope the capture.
3. Real CPU collector tests: BF16 raw + uint8 mean; remapped physical pages; page/region boundary and partial pages; all ranks; actual local-tail and independent full-local caller; byte/hash/geometry/census identity; no extra copies/reservations/synchronization; inactive predecessor skipped by the existing consumer; unchanged UNVERIFIED numerical status; isolated request/file/time cap failures. Reuse stage regression without launching or restaging a GPU package. Run self-check and fix in-scope failures.
4. Record exact commands, environment, logs, budget, source-pin handoff and limitations; reread requirements; commit only owned files and task records. Parent integrates HTTP/census/input corrections and builds one new generation.

## Acceptance

- Retained values exist only at first-tail for at most two valid logical pages, both component kinds, each side, TP0–TP7.
- Complete hash/geometry census and exact-byte comparison policy are unchanged. No numerical interpretation of uint8 FP8 bytes.
- Same GPU-copy/reserve/sync counts as the original collector; no extra observer fences.
- Existing wrapper isolates request/file/time cap errors and preserves the native result.
- No changes to connection, census, graph, workload, pressure, native source or old artifacts.
- Focused tests pass; task records and verified commit are delivered. Native capture and rule proposal remain parent-owned follow-up.
