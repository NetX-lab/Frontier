# W128-V4-NATIVE-SERVE-PROFILE CPU collector archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived verified launch-seam blockers and separate baseline-suite results; collector implementation is pending root file allocation. |
| 2026-10-23 | Completed approved collector/launch integration; archived CPU/image results and runtime gates. |

## Outcome

**CPU collector COMPLETE; GPU qualification UNVERIFIED.** Root approved the two-file increment. No delegation, GPU/platform/SSH/cloud/native/C26 operation, root merge, driver, scaling or CPU-cost inference occurred.

Original launch RED probes are GREEN through real local functions. One NATIVE_WORKER_EXTENSIONS mapping is reused by all three launch callers.

## Source deliverables

- `tests/comparison/calibration/native_serve_profile.py`: composed observer, unchanged inherited inventory RPC, outer/component/ownership ranges, phase/step dispatch join, actual event calls/generations/handles, CUPTI ID query, capture/Python-replay labels, PDL/config facts and traceback isolation.
- `tests/comparison/calibration/native_pd_worker.py`: shared selector/module hash; explicit collector capacity input.
- `tests/comparison/calibration/pd_replay.py`: same-table worker PYTHONPATH.
- `tests/comparison/dp_placement_pp/vllm_replay.py`: same-table routing registration; native profiler JSON forwarding.
- `tests/unit/test_native_serve_profile.py`: one new test file executes actual local collector and pinned native stream bodies.

Output: `<FRONTIER_NATIVE_SERVING_EVIDENCE_DIR>/serving_profile/rank<rank>.json`; legacy `serving/rank<rank>.json` still emitted by unchanged observer. Profile records always UNVERIFIED, not device qualification.

Enable `worker_extension_cls="native_serve_profile.NativeServeProfile"`, native `profiler_config`, and explicit native-config `serving_profile_limits={"records":N,"bytes":B}`. Missing/invalid limits produce isolated errors/dropped evidence, never native failure. Root selects capacity.

## Deliverables

All paths are under `task_memory/task_2026-10-23_v4_remaining_measurements/collector/`:

- `progress.md`: implementation reasons/results, historical allocation and phase checkpoint.
- `test_report_2026-10-23_collector.md`: separate suite results and exact environment/commands.
- `launch_seam_receipt.json`: source hashes, executable probe command, observed refusals/omissions and local image identity.
- `scope_increment.diff`: historical12-line proposal, approved and applied as source now.
- `completion_receipt.json`: GREEN probes, direct CPU and image controls, suite results.
- `collector_source.diff`: final five-file source diff against initial baseline.
- `summary.md`: this archive.

## Validation

CPython 3.12.13, `/data/ycfeng/envs/asyncopd-py312/bin/python`; no installs:

- `tests/unit/test_native_serving_inventory.py`: **28 PASS**.
- `tests/unit/test_native_v2_serving_inventory.py`: **45 PASS**.
- `tests/unit/test_native_reader_binding.py`, explicitly bound to the worktree's R4 checker: **135 PASS**.
- Collector `test_native_serve_profile.py`: **22 PASS**, separate from baseline counts.
- Native worker **16 PASS**; DP command harness **75 PASS**; P/D harness **68 PASS**; manifest **14 PASS**, separately.
- Image: **24/24 source symbols PASS**, real torch **2.13.0+cu130 CPU nested ranges PASS**, CUDA uninitialized. Native runtime imports **INCOMPLETE** after120-second import/JIT timeout; stopped only that CPU container.
- Direct controls: local batches1..11/padded16 retained, and decorated sparse callback emitted two ranges with identical native marker return.

## Driver seam and remaining work

Approved source increment complete. Root owns review and driver; no separate TP profiling argument needed.

The out-of-scope driver bracket is `pd_replay.py:464-479`: before the measured call at lines 467-468, POST `/start_profile` to each role's head URL; after its `finally`/drained boundary, POST `/stop_profile`. Native API: `vllm/entrypoints/serve/profile/api_router.py:21-32`. It fans out through all DP engines and TP workers (the parent seam records the callgraph). Profile configuration forwarding belongs to `dp_placement_pp/vllm_replay.py:341-347`, not a new TP flag. Bracketing must remain auxiliary and non-blocking and must not stop healthy native calls.

CPU stand-ins establish event handle/generation/stream rows and absence of invented sequential edges. GPU raw-to-trace mapping, per-kernel PDL, capture/replay annotation/event survival, captured-segment ownership and GPU CUPTI restart remain UNKNOWN. No added CUDA timing events or captured-event per-replay timing claims. Root must review/add driver/validate native installed execution and prepare explicit coverage/topology/runtime gates before GPU approval. Parent requirements reread before closure.
