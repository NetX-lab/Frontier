## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Added independent W02 capacity/lifecycle acceptance tests and request-termination inventory; reproduced and independently verified the repair of waiting-queue atomicity. |

# W02 independent acceptance construction

## Scope and status

- Requirements: R02, R03, R09, R13; T02, T03, T04; D03.
- Ownership: only `tests/unit/test_gdn_scheduler_slots.py`, `tests/unit/test_gdn_hybrid_e2e_increment14ab.py`, and this document. No production changes or commit by this worker.
- Implementation: tests extended using existing fixtures and normal constructors.
- Verification: all 19 owned-file tests pass after the integration owner's W02 repair, including the previously failing admission-failure atomicity regression.
- Readability/ownership: replaced the automatic capacity test's self-referential planner oracle with independent state/KV arithmetic; replaced a phase-cap `SimpleNamespace` with validated `VllmV1SchedulerConfig`.

## T02 independent capacity oracle

The tiny four-layer fixture has two resident GDN layers and two KV-carrying dense layers. At TP=1:

- Conv state per GDN layer: `2 * 3 * (2 * 2 * 32 + 4 * 32) = 1536` bytes.
- Recurrent state per GDN layer: `4 * 4 * 32 * 32 = 16384` bytes.
- Per-request resident state: `2 * (1536 + 16384) = 35840` bytes.
- At block size 1024, the selected stage's KV block denominator is `2 * 1024 * (2 * 2 * 64) * 2 = 1048576` bytes.
- Expected blocks: `(requested_budget - accepted_ParamCounter_weights - accepted_overhead - effective_capacity * 35840) // 1048576`.

The real scheduler is constructed with automatic `num_blocks=0`, capacities 1/2/3/64, both `memory_planner` and `memory_planner_profiled`, and optional hidden-phase cap 5. The test checks the effective cap reaches both state ownership and memory reservation. Only profiled mode consumes the configured three-MiB overhead. Adjacent low capacities yield equal block counts, while capacity 64 lowers capacity; nonincreasing flooring is explicitly accepted.

Structured OOM checks leave zero and one byte after reservation and verify `FrontierMemoryOOMError.reason`, exact remaining budget, GDN reservation, and cluster identity. A real automatically planned homogeneous Llama scheduler has no GDN slot owner and matches an independent dense KV block oracle. D57 weight policy remains unchanged, with the accepted ParamCounter weight result reused.

## T03 real Simulator lifecycle

The existing production-constructor test now runs one offline request and three closely spaced online arrivals, capacity one, automatic blocks, and four-token chunked prefill. The existing synthetic GDN trainer/artifact loader and standard profile-backed manager/predictor constructors remain active. Thin observers call the original allocation and completion methods and inspect their results; they do not replace predictor output or scheduler behavior.

Observed assertions:

- All requests execute 16 prefill plus two decode tokens and complete with positive E2E time.
- The online schedule reaches an exhausted one-slot pool while later requests remain waiting without KV or GDN ownership.
- Admitted requests retain the same state slot across all continuation rounds.
- Slot zero is released on completion and reused by subsequent requests.
- At quiescence, allocations, slots, waiting/running queues, and running-batch counters are empty/zero.
- Existing GDN and dense per-layer identity trace assertions continue to pass, and metrics completion is 1/1 and 3/3 respectively.

This is synthetic profile-backed CPU integration evidence, not native AMD/GPU correctness or production timing parity.

## T04 reproduced atomicity defect

`test_gdn_slot_allocation_rolls_back_kv_blocks_on_slot_failure` now injects failure through the real waiting admission path rather than directly calling `_allocate_request`. It snapshots KV blocks, slots, queues, scheduled-token frontiers, running-batch counters, request epochs/preemption state, and request waiting counters. It also requires successful readmission after the fault is removed.

Observed failure before production repair:

```text
FAILED tests/unit/test_gdn_scheduler_slots.py::test_gdn_slot_allocation_rolls_back_kv_blocks_on_slot_failure
At index 3 diff: () != (<frontier.entities.request.Request object ...>,)
```

Cause visible in `_schedule_waiting_requests`: it removes the request from waiting queues and calls `request.on_leave_waiting_queue()` before `_allocate_request()`. The latter rolls back KV blocks on slot allocation failure, but the former does not restore queue ownership or waiting counters. The integration owner moved KV/state allocation ahead of queue removal and waiting-counter updates. The final combined run passes the unchanged regression and its subsequent successful readmission check. The integration owner also resolves `_admitted_request_capacity` once for planner and slot owner and uses the public `ParamCounter.get_resident_attention_stage_id()` for planner ownership. The unsupported-preemption check verifies the same before/after state snapshot remains unchanged and the caller's preempted list remains empty.

## Execution evidence

Environment: `/usr/bin/python` 3.12.3; no active conda environment. NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0 use `/usr/local/lib/python3.12/dist-packages`; pytest 9.1.1 uses `/home/i-fengyicheng/.local/lib/python3.12/site-packages`. No Python-version mixing.

Executed from the worktree root:

```bash
python -m pytest tests/unit/test_gdn_scheduler_slots.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-slots-first
python -m pytest tests/unit/test_gdn_hybrid_e2e_increment14ab.py -k production_constructor -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-e2e-first > /data/ycfeng/tmp/pr33-w02-e2e-first.log 2>&1
python -m pytest tests/unit/test_gdn_scheduler_slots.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-acceptance-before > /data/ycfeng/tmp/pr33-w02-acceptance-before.log 2>&1
python -m pytest tests/unit/test_gdn_scheduler_slots.py::test_effective_memory_planner_capacity_includes_phase_specific_caps -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-phase-valid > /data/ycfeng/tmp/pr33-w02-phase-valid.log 2>&1
python -m pytest tests/unit/test_gdn_scheduler_slots.py -k 'not rolls_back' -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-slots-positive > /data/ycfeng/tmp/pr33-w02-slots-positive.log 2>&1
python -m pytest tests/unit/test_gdn_scheduler_slots.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-acceptance-after > /data/ycfeng/tmp/pr33-w02-acceptance-after.log 2>&1
git diff --check -- tests/unit/test_gdn_scheduler_slots.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py
```

| Check | Observed result | Limits |
| --- | --- | --- |
| Initial scheduler check | 10 passed, 1 failed, 2.81s | Atomicity regression intentionally remains red until production repair. |
| Production constructor E2E | 2 passed, 4 deselected, 6.95s | Synthetic profile-backed CPU evidence. |
| Combined check | 15 passed, 2 failed, 8.18s | One failure was the atomicity defect; one was a newly converted test fixture omitting `enable_phase_aware_thinking_profile=True`. |
| Corrected valid phase fixture | 1 passed, 2.68s | The test-only construction error was fixed at its source. |
| Expanded capacity/positive scheduler checks | 12 passed, 1 deselected, 2.70s | Explicitly excludes the already reproduced atomicity defect. Includes both auto modes and phase-cap matrix. |
| Final combined acceptance after production repair | 19 passed, 8.84s | Includes atomic rollback/readmission, full phase-cap matrix, and real single/multi-request Simulator execution. |
| Targeted whitespace check | PASS | Only the two owned test files. |

## D03 request-termination inventory

Read-only inventory commands:

```bash
rg -n 'def .*cancel|def .*abort|def .*shutdown|def .*terminate' frontier --glob '*.py'
rg -n '\.cancel\(' frontier
rg -n '_free_request_resources|def _set_time|def _maybe_export_sequential_checkpoint|_terminate = True' frontier/simulator.py frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py
rg -n 'def stop|def close|def cleanup|signal|KeyboardInterrupt|finally:' frontier/simulator.py frontier/cluster_simulator.py frontier/main.py
```

| Route | Existing owner/action | Reachability and GDN implication |
| --- | --- | --- |
| Normal request completion | `Batch.on_batch_end` -> replica scheduler `on_batch_end` -> `_free_request_resources` | Supported co-location GDN path; real lifecycle test verifies release. |
| Cleanup by request ID after queue removal | `_free_request_resources_by_id` | Idempotent resource boundary; existing direct scheduler regression verifies orphan slot cleanup. It is not a public cancellation API. |
| KV handoff completion and non-monolithic prefill cleanup | `complete_kv_transfer_for_requests`, prefill branch of `on_batch_end` | Disaggregated routes; GDN PD/PD-AF remains unsupported. |
| Deferred monolithic PP terminal release | Two iteration-boundary methods call `_free_request_resources_by_id` | PP route; GDN PP remains unsupported. |
| State-dropping preemption | `_preempt_request` | GDN guard rejects before mutation; strengthened snapshot test passes. |
| Admission allocation failure | `_allocate_request` exception rollback | KV rollback exists; the W02 repair delays waiting-state mutation until successful allocation. The unchanged regression passes. |
| Stage ticket cancellation | `StageExecutionContext.cancel(ticket)` | Cancels queued operation ownership, not request lifecycle. Callers are stage scheduler preflight/stale-event handling, `layer_admission.discard_admission_ticket`, and dense/MoE FFN M2N preflight rollback in round-robin cluster scheduler. |
| Sequential event-handler failure | `_run_sequential` logs and rethrows | Whole-run failure; no request-level cancellation transition or resume guarantee. |
| Sequential checkpoint stop | `_maybe_export_sequential_checkpoint` writes survivor state and sets `_terminate` | Deliberately preserves live requests in a checkpoint; not cancellation. |
| Sequential exhaustion with live schedulers | `_run_sequential` raises a diagnostic `RuntimeError` | Failed run, not successful termination or cleanup evidence. |
| Parallel fatal error/time limit/deadlock | `_monitor_parallel_simulation` and `_run_parallel` finally block | Whole-run stop; cluster `stop()` stops/joins worker threads and writes statistics. GDN supported co-location execution uses the sequential loop. |
| CLI memory/config failure | `frontier.main.main` emits structured memory exit 2 or release-guard exit 1 | No per-request cancellation API. |

No public request cancellation or abort API was found. Consequently, these tests make no cancellation claim. D03 choice for the integration owner: retain the current supported completion/continuation/preemption-rejection scope, or explicitly seek authorization for a minimal new request cancellation surface and its event/queue/resource semantics. Recommended current scope is retention, since adding a new API expands supported behavior.

Additional read-only observation: a sequential time-limit check follows `return True` in `_maybe_export_sequential_checkpoint`, while `_set_time` currently only assigns time. This source observation is not a reproduced W02 failure and is outside the owned tests; the integration owner should classify it separately if relevant.

## Handoff

- Test implementation and focused CPU verification: complete.
- Open correctness defects in this owned lane: none after the observed production repair.
- D03 remains an explicit supported-scope decision; no request cancellation API was added or claimed verified.
- Commit SHA and final required `test_report_2026-09-16_*.md` integration belong to the root integration owner.
