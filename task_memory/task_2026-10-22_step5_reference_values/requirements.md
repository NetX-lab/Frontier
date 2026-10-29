# D24-G3-STATE-VALUES requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded the explicit D24 decision, delegated CPU implementation scope and acceptance criteria. |

## [Original Request]

New explicit YC decision D24-G3-STATE-VALUES: user selected 补采后定数值判据 (Recommended). Authorizes adding bounded raw/mean value capture to the NEXT H80032 diagnostic (about183MB extra; parent-only run, existing90min/48GPUh ceiling), then proposing a numerical acceptance rule for separate approval. Until then independent-state verdict stays UNVERIFIED. No tolerance or acceptance change is authorized. Your read-only scope is complete; implement this fresh CPU slice now.

Create NEW /data/ycfeng/Frontier/.worktrees/step5-reference-values branch feat/step5-reference-values FROM0c9d1ef299e27d0e9bddbe4eb888527f966ba718 (contains parent startup census e8d2f1a and merged inputs). Keep main tree and finished report unchanged. No GPU/platform/auth/cloud/SSH/push or further delegation. Own tests/integration/step5_pd/r2_reference.py; r2_observation.py ONLY bounded mean retention at currenttail; appropriate existing stage/config collector seam if needed; focused tests. Parent owns core/input integration and native submission; HTTP diagnostics is separatecompletedbranch42fb46ce. Do not alter connection/census/graph policy, frozen workload, pressure parameters or old artifacts.

Use your inspected seam: checkpoint already copies/reserves/hashes raw bytes then popsraw_hex. Retain raw_hex only for computed-token logicalpage + preceding logicalpage at first_local_tail_complete, on both actual/reference sides and allTP ranks. This preserves known-different pages and unchanged controls, no second GPUcopy/fence. Inactive predecessor tail accumulators must not be compared positionally. Original complete hash/geometry census remains. Scope values via existing diagnostic config/definition only if a supported opt-in is needed; avoid unnecessary flags/frameworks. Retain bounded mean wire bytes at the same selected boundary using existing copy. Enforce current per-request/record/time caps through existing soft diagnostic behavior. Do not retain whole checkpointbytes (~561MB/pair), or expand allcheckpoints.

Acceptance: real collector CPU seam (not a duplicate capture implementation) proves selected2logicalpages and firsttail only, bothcomponent kinds, actual/reference caller paths, fullhashcensus unchanged, rawhex sha/byte count correct, noextraGPUcopy/sync/reserve, numericalstatus remainsUNVERIFIED for non-identical state, cap exceptions isolated. Reuse existing reference/observer/stage tests; preserve reports of failures then repair in-scope. No BF16/FP8 tolerance, bitmean reinterpretation or heuristic waveform corrections. A small descriptive receipt may show payload presence/counts, but FP8 uint8 storage must not be treated as numerical values without source-backed ABI decoding. Returning the capture capability is enough for this slice.

Records in task_memory/task_2026-10-22_step5_reference_values/{requirements,plan,progress,summary,test_report...}.md with ModificationHistory, raw user decision and exactbytesestimates from priorreport. Commit verified changes and return exacttests/commands/logs, files, commits, addedpayloadbudget and any sourcepin/staging handoff. Do not launch/restage GPU packages; parent combines corrections in one generation.

## Explicit decisions and constraints

- CPU capture capability only. No numerical rule, tolerance, FP8 decoding or acceptance changes; independent-state mismatch remains UNVERIFIED.
- New worktree starts at the exact requested base. No edits to `step5-preview`, the finished D23 report or old packages. Parent integrates and stages one fresh generation.
- Preserve the existing diagnostic selection, workload, capacity, engine, source bindings and soft caps. No additional GPU reads, reservations or fences.
- Task/session: `D24-G3-STATE-VALUES` / `ses_ea074b7bfffe5UImN2k3jg0Uax`.

## Evidence-derived payload budget

Prior report (read-only): `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-21_step5_priority_runs/canonical_g3/reference_followup/report.md`, section 3.4. Actual first-tail checkpoint shapes: 92 raw layers, `[16,1,384]` BF16 full page (12,288 B); 23 mean layers, `[2,256]` uint8 full page (512 B). Partial raw pages have 15 or 1 valid tokens; partial mean retains `ceil(tokens/8)` region rows. FP8 storage bytes are not numerical values.

| Prompt | Two selected logical pages | Payload per side, binary | Both sides, hex |
| --- | --- | ---: | ---: |
| 6144 | 382, 383 | 2,284,544 B | 9,138,176 B |
| 6143 | 382, 383 | 2,213,888 B | 8,855,552 B |
| 6145 | 383, 384 | 1,218,816 B | 4,875,264 B |

- Three normal actual/reference pairs: 22,868,992 B hex/TP rank, **182,951,936 B** across TP8 (prior approximate 183 MB).
- Existing `tail_state` mean wire rows: 23 layers × 2 full rows × 512 B × 6 requests × 2 hex × TP8 = **2,260,992 B** additional hex. Normal-pair total: **185,212,928 B** (176.63 MiB), excluding unchanged JSON metadata.
- Existing content selection also includes cancel6144 and reuse6144. If both reach first-tail, each adds 4,616,192 B hex/TP rank (checkpoint + mean wire rows). Whole diagnostic upper bound: **259,072,000 B** (247.07 MiB) across TP8. This is not an expanded workload; the prior 183 MB estimate counted only three normal pairs. Parent receives this corrected budget before staging.
- Whole canonical6144 checkpoint: 140,181,504 B/side, 560,726,016 B hex/pair. This implementation must not retain that scope.
- Existing canonical config caps: 1 GiB per file and per request; 600 s post-return diagnostic time; 500,000 events. All remain unchanged.
