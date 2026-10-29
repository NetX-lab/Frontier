# D22-TIMING requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record delegated D22 implementation and parent-only GPU handoff. |

## [Original Request]

Task D22-TIMING: implement the approved independent outer-window reference protocol in the clean `step5-profiling` worktree at `79c8b45`. Parent decision D22 explicitly records **“改为独立窗口参照（推荐）”** in `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/requirements.md`.

- Collect independent candidate and reference clean-replay windows through existing `collect_replay_timings` / `measure_window`, primed outer events only, matching replay counts and minimum-window treatment. Never reuse samples across cohorts.
- Retain boundary single-replay observations as labelled diagnostics only. Preserve 5% mean-versus-reference and CV<=5%, native capture/output/cache/scoped operator and linear protocols. No subtraction, threshold changes, added wrapper or quality-dependent retry.
- Obtain meaningful RED->GREEN at the actual timing seam with injected event overhead; preserve all 851 baseline test IDs. Do not merge compute `83af0a7`; parent resolves its three-line `rows.py` overlap.
- Stage the full original D18 H200 probe: 18 event + 4 GPU-active attention rows, unchanged linear methods/shapes, one H200, distinct output and sealed generation. Preserve every historical FAIL and old stage02 byte.
- D21 requires all personal H200 non-preemptible reservations <=32. Choose this new auxiliary probe preemptible; parent still checks actual quota and priorities.
- No GPU, authentication, SSH, network, cloud, Docker or nested delegation. Commit verified source/records and stop at parent-only submission handoff.

## Ownership

Attention timing/rows/wrapper, minimal existing CLI/judge/tests/stage-resource wiring, and this task directory. Simulator and linear timing behavior are excluded. Native qualification remains pending the parent-run fresh receipt.
