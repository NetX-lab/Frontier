# Native profiling replay lifecycle

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Root traced the real P0 replay path and specified the remaining driver boundary before a GPU proposal. |
| 2026-10-23 | H source/CPU accepted at64f3c950; recorded startup-to-window retention coordination for G2. |

## Existing execution path

`native_pd_worker.py:329–348` starts the replay subprocess only on node position0 after both role heads report ready. Its actual entry is `pd_replay.py`, with the existing engine, trace, observer, model, P/D head URLs, startup/drain and idle settings. An unused standalone driver would not activate profiling in this path.

`pd_replay.py:458–489` owns the seven idle boundaries and the warmup/formal phases of the three repetitions. Per-phase streaming replay is at lines467–468, followed by request completion accounting and an idle boundary at line479. Profiling must begin before the selected formal phase dispatches its first request; stop after its drain is observed. Warmups, repetitions and request membership remain unchanged.

The profile API fanout to headless DP peers and all TP workers is already traced in `native_profile_seam.md` §6. Requests target the two role-head URLs, not 32 individual ranks.

## Required driver behavior

- Use the existing worker-extension selection and profiler configuration. The H owner owns the launch registration and environment paths; any final driver callsite change returns to that owner to avoid competing file ownership.
- Preserve the current clean/default replay byte-for-byte where profiling is unselected. A profiled replay is diagnostic and cannot replace native06's clean reference.
- Record the selected repeat/phase, head URLs, start/stop attempts and outcomes, actual profile output locations, and join to the existing request IDs and worker step evidence.
- Keep auxiliary RPC, trace-export and receipt errors separate from native replay exceptions. A profile failure marks its evidence UNVERIFIED while the native workload continues. Original business exceptions and terminal behavior remain intact.
- Use the existing profiler iteration/window controls when bounding collection. The first formal batch and at least three distinct aligned logical batches in every claimed phase must be present. A truncated window supplies only the tuples actually observed; it cannot qualify the entire context6143..7166 range or all descriptor/phase combinations.
- Align collector retention with the selected profiler window. At f8621a8, the collector begins observing during worker/model startup, retains capture/warmup records, and emits without resetting its buffers; it does not hook Worker.profile. Consequently /start_profile alone does not start a fresh collector budget. G2 must explicitly account for earlier records and prove that the chosen window's step/ownership/synchronization records survive the records/bytes limits.
- Avoid silently replacing fixed per-rank profiler summary files across windows. Timestamped traces and the window mapping must remain identifiable. CPU restart evidence does not qualify GPU CUPTI restart under graphs.

## Exit conditions for the GPU proposal

H's source/CPU review is accepted at64f3c950 (`collector_primary/primary_acceptance.json`); real installed execution and GPU mapping remain runtime gates. The driver must next pass stand-in API tests through the actual replay callsite: both heads activated before dispatch, drain before stop, unselected compatibility, instrumentation-failure isolation and native-error preservation. The final proposal pins source, input trace, profiling window limits, expected artifact volume, runtime unknowns, GPU count and duration. Source feasibility alone is not launch readiness.

No driver code or GPU run is authorized by this design beyond the existing CPU implementation scope. The exact implementation allocation and new run resources remain recorded at their respective steps.
