# W128-V4-NATIVE-SERVE-PROFILE scope-blocked archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived verified launch-seam blockers and separate baseline-suite results; collector implementation is pending root file allocation. |

## Outcome

**BLOCKED, not implemented.** Work stopped under the explicit instruction to report the exact increment when the smallest complete component exceeds the assigned scope. No source files changed, no further delegation, and no GPU/platform/SSH/cloud/native/C26 operation occurred.

The existing launch path refuses a new extension when `balanced_topk` is selected, omits its worker `PYTHONPATH`, and does not forward `profiler_config`. These are observed by executing the real modules from `/data/ycfeng/Frontier/.worktrees/v4-native-serve-profile`, not inferred from a different worktree.

## Deliverables

All paths are under `task_memory/task_2026-10-23_v4_remaining_measurements/collector/`:

- `progress.md`: blocker, exact proposed allocation increment, pending work, and phase checkpoint.
- `test_report_2026-10-23_collector.md`: separate suite results and exact environment/commands.
- `launch_seam_receipt.json`: source hashes, executable probe command, observed refusals/omissions and local image identity.
- `scope_increment.diff`: **unapplied** minimal launch integration proposal, 12 changed lines across two additional files. It is not a shipped source change.
- `summary.md`: this archive.

## Validation

CPython 3.12.13, `/data/ycfeng/envs/asyncopd-py312/bin/python`; no installs:

- `tests/unit/test_native_serving_inventory.py`: **28 PASS**.
- `tests/unit/test_native_v2_serving_inventory.py`: **45 PASS**.
- `tests/unit/test_native_reader_binding.py`, explicitly bound to the worktree's R4 checker: **135 PASS**.
- Collector suite: **NOT RUN**, because no collector implementation was made.
- Installed-image hook resolution: **NOT RUN**. Local `docker image inspect` confirms the requested locked digest exists; that is not hook-resolution evidence.

## Driver seam and remaining work

Root should allocate `tests/comparison/dp_placement_pp/vllm_replay.py` (8 changed lines) and `tests/comparison/calibration/pd_replay.py` (4 changed lines), or own that increment separately. The authorized `native_pd_worker.py` allowance can reuse the same extension table and preserve the old branch exactly.

The out-of-scope driver bracket is `pd_replay.py:464-479`: before the measured call at lines 467-468, POST `/start_profile` to each role's head URL; after its `finally`/drained boundary, POST `/stop_profile`. Native API: `vllm/entrypoints/serve/profile/api_router.py:21-32`. It fans out through all DP engines and TP workers (the parent seam records the callgraph). Profile configuration forwarding belongs to `dp_placement_pp/vllm_replay.py:341-347`, not a new TP flag. Bracketing must remain auxiliary and non-blocking and must not stop healthy native calls.

New stream handles/event generations, per-kernel PDL visibility, capture/replay event survival, and CUPTI restart remain **UNKNOWN**. Existing source semantics do not establish observed edges or timings. After the allocation is settled, build and CPU-verify the collector, resolve every installed hook, add the root-owned driver, review completeness/dependency/runtime gates, and obtain explicit GPU approval. Parent requirements were reread before closure.
