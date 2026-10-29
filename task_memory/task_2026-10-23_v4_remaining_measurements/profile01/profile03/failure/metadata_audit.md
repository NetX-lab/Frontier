# W128-PROFILE03-METADATA
## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Corrected only trace/table census using a fresh pinned adapter; preserved original receipts and D0 host label. |
| 2026-10-24 | Closed the review's P2 host-label finding: D0 is gpu-h800-0422, matching the unchanged inventory and JSON. |
| 2026-10-24 | Full-inventory metadata audit; no data admission or runtime changes. |

## Scope and artifact status
- Audit completed with missing profile-stop artifacts; serving run remains **FAILED**.
- Input: `/data/ycfeng/tmp/glm53_dsv4/profile03_results_7dsz5a22/`; all 190 inventory files, totaling 8,208,668,144 bytes, match their sizes and SHA-256; receipt binds the inventory hash.
- Retained: 32 `nodes/*/serving_profile/rank*.json`, 32 `rank*.before_profile.json`, 32 linked `nodes/*/serving/rank*.json`, 32 `.pt.trace.json.gz` raw traces and 32 optional `profiler_out_*.txt` tables. Raw bytes were hashed, not decompressed; trace contents remain unverified here.
- **Missing:** all 32 requested `rank*.profile.json`. P1/D0 have 16 published `pre_shutdown` collector snapshots with a window; P0/D1's 16 published snapshots are old `post_warmup`, not formal profiles.
- D1 rank6 `.json.1735.tmp` is empty. Rank7 `.json.1736.tmp` is parseable, 450,676 bytes, `pre_shutdown`, but unpublished; it remains separate diagnostic evidence, not a replacement or admission.

## Every lane and all eight local ranks
Values are identical across local ranks 0–7 of each lane. Useful candidates require canonical repeat01 formal IDs, positive local tokens and linked annotations; kernel-bearing useful steps require root's raw-trace correlation.
| Lane / host suffix | Published stage | Retained steps: count; min–max; last | Useful candidates: count; min–max; last | Annotations / step=null / sync rows |
| --- | --- | --- | --- | --- |
| P0 / gpu-h800-0108 | post_warmup | 8; 0–7; 7 | formal metadata missing; 0 observed in stale snapshot | 165103 / 162067 / 58020 |
| P1 / gpu-h800-0329 | pre_shutdown | 512; 23–534; 534 | 60; 23–533; 533 | 104942 / 28322 / 28002 |
| D0 / gpu-h800-0422 | pre_shutdown | 64; 2181–2244; 2244 | 61; 2184–2244; 2244 | 485 / 6 / 0 |
| D1 / gpu-h800-0618 | post_warmup | 8; 0–7; 7 | formal metadata missing; 0 observed in stale snapshot | 165103 / 162067 / 58020 |
- P1 last raw step 534 has zero local tokens and no DP sequence; last useful step 533 has sequence 1130. Last annotation is `step=null`, sequence 1131, dummy; it is not a serving-step completion timestamp.
- D0 last raw/useful step 2244 and last annotation both have sequence 2313.
- Unpublished D1 rank7: 64 steps 1120–1183, 61 candidates 1123–1183; 765 annotations, 199 `step=null`, no sync rows; last raw/useful step sequence 2366. Other D1 ranks' formal counts remain unavailable.

## Raw DP distributions and pairability
Exact histograms, missing-sequence counts, last rows and per-rank paths are in `metadata_audit.json`; ranges below do not replace those histograms.
| Published snapshot | Step sequences: distinct; range; missing | Annotation sequences: distinct; range; missing | step=null annotations: distinct; range; missing |
| --- | --- | --- | --- |
| P0/D1 stale | 6; 3–9; 2 | 8; 2–9; 159749 | 2; 2–5; 159749 |
| P1 formal | 60; 42–1130; 452 | 1090; 42–1131; 0 | 1030; 43–1131; 0 |
| D0 formal | 61; 2253–2313; 3 | 64; 2250–2313; 0 | 3; 2250–2252; 0 |
- P1 null annotations are 28,322 dummy rows; D0 has six dummy rows. P0/D1 stale null rows are 159,748 capture, 2,318 profile, and one unattributed; these are not admitted as formal work.
- Common key is `[run_generation, role, ordered DP group members]`; generation is `v4-native-v028-h800-p0-profile-20261024-03`, with `[r,r+8]` for each role and r=0–7. No `dp_owner`, lane, offset or host-time adjustment enters the key.
- Published formal snapshots give **zero two-sided step pairs** in all 16 groups because the counterpart's formal publication is missing. P1 retains 60 unmatched step sequences per group; D0 retains 61.
- Only diagnostic D1 rank7 tmp supplies a formal counterpart for decode `[7,15]`: eight raw step pairs 2306–2313; 53 D0-only values 2253–2305 and 53 D1-only values 2314–2366 remain unmatched.
- The same tmp-inclusive group has 64 shared all-observation sequences 2250–2313 and 53 D1-only sequences 2314–2366; null-annotation partitions share three values, with 53 D1-only values. Annotation equality is not step equality.
- `before_profile` has 40 shared sequences per prefill group and 2,248 per decode group; it is reported separately and never fills formal windows.
- Equality means diagnostic pairability only, not GPU synchronization, dependency, completeness or successful serving.

## Baselines, counters and streams
- `before_profile` step counts/ranges: P0 22/0–21, P1 23/0–22, D0 2181/0–2180, D1 1120/0–1119. These 32 startup/warmup snapshots are distinct from formal collection.
- Linked inventory forward/replay counters per rank: P0 116/47, P1 1240/1083, D0 6516/10876, D1 116/47. They are snapshot/lifetime counters, **not** formal successes; JSON preserves their kind/phase partitions.
- All 64 published/baseline snapshots have `dropped=0`, `suppressed_errors=0`, and empty collector/inventory errors; all 32 linked inventories have zero dropped rows and no observation errors. These counters do not prove complete publication or capture.
- Every rank has six non-null collector stream entries: 192 total. Trace IDs are 23,35,39,43,55,59; handles, device indices and Torch IDs are preserved per host/rank in JSON. Repeated trace IDs are not global identities.
- All 32 published stream maps equal their `before_profile` maps; these are lifetime collector maps, not proof that streams appear in active raw traces.

## Source, clocks and limits
- All 32 identities join to role/lane/rank, generation and DP groups. Native reference: `2cf0a6915ce544dc493a0990f2ea38d81601128a`; source-hash mapping digest: `1737349f185dea21aea168d128bb2aa51b5a914cc7dda0d845a284dfcf11659e` everywhere.
- Staged `native_serve_profile.py`: `c8f712177fcd3e8ef6fbbfe2ceb62a30e06d298d432da48bd7790ed214669f58`; `native_full_structure.py`: `a37ec448730dd1f3be26e466c44f14bd9cec2518c5cd42624fa4e6fb6a39455c`; both match manifest bindings. Exact package paths are in JSON `bindings.sources`.
- Source refs: `native_serve_profile.py:113–136` archives/resets and emits `.profile` on explicit stop; `:417–439` uses tmp/replace; `:472–499` retains finished-step rows without timestamps. Existing profile02 metadata candidacy was reused without importing its raw-trace parser.
- Annotation/step/sync metadata has no clock timestamps. Existing P1 autostop log evidence is linked, but `finish_step after P1 autostop` is **UNKNOWN**. Filename/export times and step counters cannot supply that ordering; no cross-host clock alignment was performed.
- No trace decompression, hang reproduction, GPU/platform/network/Docker/Git operation, parent-document edit, runtime fix or new data admission. Prior timeline files remain byte-identical; commands/results are in `metadata_command.md`.
