# Native profiling replay lifecycle

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | W128-H800-AUTHORITY-DOCS: recorded YC's approved first32-H800 run and standing task-required H800 resource authority; package/source/image and runtime checks remain prerequisites, CPU implementation unchanged. |
| 2026-10-23 | Root traced the real P0 replay path and specified the remaining driver boundary before a GPU proposal. |
| 2026-10-23 | H source/CPU accepted at64f3c950; recorded startup-to-window retention coordination for G2. |
| 2026-10-23 | Specified the first-formal window, native iteration controls, separate collector retention and direct root ownership after both delegation attempts failed. |
| 2026-10-23 | Added the bounded cold-start test correction and clarified three independently reviewed failure-isolation repairs. |
| 2026-10-23 | Recorded accepted CPU implementation 05952396 and replaced stale line/status pointers with the implemented callsites. |

## Existing execution path

`native_pd_worker.run()` starts the replay subprocess only on node position 0 after both role heads report ready. Its actual entry is `pd_replay.py`, with the existing engine, trace, observer, model, P/D head URLs, startup/drain and idle settings.

`pd_replay.replay_repetitions()` owns the seven idle boundaries and the warmup/formal phases of the three repetitions. The profiling context begins before the selected formal phase dispatches its first request and stops after its drain is observed. Warmups, repetitions and request membership remain unchanged.

The profile API fanout to headless DP peers and all TP workers is already traced in `native_profile_seam.md` §6. Requests target the two role-head URLs, not 32 individual ranks.

## Required driver behavior

- Use the existing worker-extension selection and profiler configuration. Root implemented G2 after both initial design delegations failed on API quota. Both independent reviews subsequently completed and closed all findings; source `05952396` is CPU-accepted.
- Preserve the current clean/default replay byte-for-byte where profiling is unselected. A profiled replay is diagnostic and cannot replace native06's clean reference.
- Record the selected repeat/phase, head URLs, start/stop attempts and outcomes, actual profile output locations, and join to the existing request IDs and worker step evidence.
- Keep auxiliary RPC, trace-export and receipt errors separate from native replay exceptions. A profile failure marks its evidence UNVERIFIED while the native workload continues. Original business exceptions and terminal behavior remain intact.
- Use the existing profiler iteration/window controls when bounding collection. The first formal batch and at least three distinct aligned logical batches in every claimed phase must be present. A truncated window supplies only the tuples actually observed; it cannot qualify the entire context6143..7166 range or all descriptor/phase combinations.
- Align collector retention with the selected profiler window. At f8621a8, the collector begins observing during worker/model startup, retains capture/warmup records, and emits without resetting its buffers; it does not hook Worker.profile. Consequently /start_profile alone does not start a fresh collector budget. G2 must explicitly account for earlier records and prove that the chosen window's step/ownership/synchronization records survive the records/bytes limits.
- Avoid silently replacing fixed per-rank profiler summary files across windows. Timestamped traces and the window mapping must remain identifiable. CPU restart evidence does not qualify GPU CUPTI restart under graphs.

## Current authority and package prerequisites

H's source/CPU review is accepted at64f3c950 (`collector_primary/primary_acceptance.json`); real installed execution and GPU mapping remain runtime gates. G2 source `05952396` passed the required actual-callsite CPU tests, independent re-reviews and explicitly bound existing77-case matrix (`driver/primary_acceptance.json`); no new matrix capture. Accepted CPU source05952396 and recordsb8fe6b2d remain unchanged.

YC's D-V4-H800-RJOB-AUTHORITY in [requirements.md](requirements.md) approves the current32-H800/codesign/non-preemptible, maximum60-minute/32GPUh, first-formal64worker-step run in [proposal.md](proposal.md). Task-required H800 RJobs of any necessary scale need no repeated resource approval; subsequent runs retain a concrete purpose/resource/evidence plan and respect actual quota/capacity. The design itself grants no authority, and resource approval does not change source/native/C26 approval, calibration predicates, publication, broad CUPTI investigation, CPU-overhead collection or paused GLM scope.

Next: fresh diagnostic package/source/image checks and quota verification, then seal/submit/monitor and retained-evidence checks. No new package is sealed or run submitted; package/runtime/GPU qualification remains pending. The first window preserves P16+D16/TP8/DP2/EP16/PP1, native V2,MXFP4/MARLIN,balanced_topk and the frozen workload. Actual coverage, event/stream/PDL/graph evidence and observation failures remain runtime gates; missing evidence stays UNVERIFIED and auxiliary gates cannot interrupt healthy workload. Native06 remains ACCEPTED_NATIVE_REFERENCE_R4, not replaced by this diagnostic; old R3 FAIL/native05 UNVERIFIED, GLM pause and official mean/30%/10% gates remain unchanged. The handoff is informational while work continues in this session.

## G2 implementation decision

The selected extension activates one diagnostic window around the first formal repetition. Both role heads receive concurrent start requests before dispatch and stop requests after the existing idle boundary. On a native replay exception, attempt an independent idle observation and stop cleanup, then propagate the original exception. Profiling outcomes live in a separate `profiling` receipt and never enter the client-workload error list. Invalid profiling configuration disables this auxiliary window with a recorded error while replay continues.

Use pinned native `ProfilerConfig` fields: `profiler=torch`, positive `max_iterations`, zero delay/warmup/wait, and `ignore_frontend=true`. This records the first formal worker step, avoids an unbounded frontend trace and uses native automatic stop. `Worker.annotate_profile` calls `step()` before model execution (`gpu_worker.py:931–940,1112`); `WorkerProfiler.step` stops when recorded iterations exceed the limit (`wrapper.py:97–128`). No collector timer or iteration scheduler duplicates that mechanism. Reaching the limit may retain fewer than three useful logical batches on a lane; actual coverage remains a later runtime gate.

The collector wraps the existing `Worker.profile` RPC. Before start it archives the startup snapshot, then gives the selected window a fresh row/byte allowance. It preserves global annotation/step identities, stream/event generations, installation facts and errors. During the window it retains records only while native `profiler.is_running` is true; stop freezes a window snapshot. Startup drops stay in the archive, selected-window drops stay explicit, and neither is a completeness verdict. Auxiliary archive/receipt failures cannot replace native return values or exceptions.

Each native node derives its trace directory from the configured root plus role and local DP start rank, preventing the two roles/nodes from overwriting fixed-rank summaries. Rank receipts retain the actual effective profiler configuration. Exactly one window avoids claiming GPU CUPTI restart support.

Allocation from baseline313c8167: seven existing non-core files, at most750 changed lines (production350, tests400): `pd_replay.py`, `native_serve_profile.py`, `native_pd_worker.py`, `test_native_streaming_replay.py`, `test_native_serve_profile.py`, `test_native_pd_worker.py`, and the16-line cold-start fixture correction in `test_native_cold_start.py` (`driver/cold_start_scope.md`, at most25lines). No Frontier/native/C26 edits. Acceptance requires actual replay/HTTP ordering and failure controls, native-shaped worker lifecycle/capacity controls, separate affected-suite results, independent Spec/Standards review and an explicitly bound default matrix.

Review repair details: startup archive emission has its own soft observation boundary, so its failure cannot prevent resetting the selected window. Only torch configurations with absolute local trace paths are transformed; native CUDA/URI configurations remain unchanged and do not activate this local-torch diagnostic window. Once a profiling receipt is attached, checkpoint failures are recorded in that receipt, including active-window and later persistent write errors; unselected replay retains its original checkpoint-error behavior. Native request, idle-boundary and metrics failures retain their existing validation paths.
