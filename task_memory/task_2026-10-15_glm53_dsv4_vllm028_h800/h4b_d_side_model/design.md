# H4b-(b) D-side remote-KV lifecycle in Frontier: design proposal

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: measured native mechanism, Frontier's current path, item 1 (READ completion at engine steps), item 2 (DECODE batch queue depth 2), gating options, expected effects, deferred parts, verification plan and approval items. |

**Status.** Proposal. No `frontier/` file is changed. Item 1 exceeds the D-H4A-CHECKPOINT bounds (more than 5 `frontier/` files and a new shared config field), and item 2 is a separate, larger change. Both return to YC before core edits.

**Authority.** D-H4A-CHECKPOINT ("Frontier models the D-side asynchronous remote-KV lifecycle from those measured inputs, with no constant fitted to native06"), D-MAINLINE-LOOP-CAP-DSV4, D-NO-INTERRUPT and the DIAGNOSTIC label in [requirements.md](../requirements.md). GLM, co-location, existing PDD and PD-AF results must stay identical.

**Inputs.** The H200 P2-32 lifecycle run (H4b-(a) on H200, [test report](../test_report_2026-10-25_h200_h4b_lifecycle.md), persisted under `../h200_mainline/diagnostic/inputs/p2_32l/`). This is H200 evidence. The H800 lifecycle run (`exp-1025-121318-827142`, Succeeded) is not yet copied; it is needed before any H800 cell is judged.

## 1. What is missing

On every DSV4 case, the largest TTFT error is the D-side handoff: the time from the proxy's P response to the proxy's first D byte.

| Case | Native handoff (ms) | Frontier handoff (ms) | Error (ms) | Source |
| --- | --- | --- | --- | --- |
| H800 P0 (native06) | 123.49 | 2.57 + 39.71 = 42.28 | −81.2 | `../h4a_p0_diagnostic/results/h4a5_attribution.json` |
| H200 P2-32 (P2-32r) | 127.92 | 2.57 + 43.24 = 45.81 | −82.1 | `../test_report_2026-10-25_h200_p2_diagnostic.md` section 6 |
| H200 P2-64 (P2-64r) | 187.76 | 2.57 + 38.98 = 41.55 | −146.2 | same |

The H200 lifecycle run splits the native handoff into parts a-g (test report section 5, decoding lanes, means): a 0.96, b 13.61, c 0.056, d 77.94, e 0.19, f 42.05, g 3.67 ms; total 138.47 ms. Part d (READ requested → reported in `finished_recving`) and part f (promotion → first token) together are 120 ms. Frontier has only the 2.57 ms wire time and one first D step.

## 2. Native mechanism (measured)

Source references are vLLM 0.28.0, the version of the native image (local copy: `/data/ycfeng/envs/oneshot-opd-py312/lib/python3.12/site-packages/vllm`).

1. **Batch queue of depth 2.** native06 and the lifecycle runs use `AsyncScheduler` (`v1/core/sched/async_scheduler.py:12`) with the engine batch queue (`v1/engine/core.py:209`, `step_with_batch_queue` at `core.py:624`). Each engine iteration schedules a new step while the previous step still runs, then blocks on the oldest step's result and applies it (`update_from_output`). The measured loop is update(N) → schedule(N+2); the decode step S is 20.35 ms mean, 20.52 ms p50 (`evidence/iter_split.txt`, `evidence/step_gaps.txt`).
2. **Decoding requests are rescheduled every step.** `AsyncScheduler._update_after_schedule` adds output placeholders (`async_scheduler.py:19-49`), so a decoding request gets one new token in each step even while its previous step is in flight.
3. **The READ is requested at schedule time and posted at execution start.** The scheduler allocates the D blocks and sets `WAITING_FOR_REMOTE_KVS` when it schedules the request (`v1/core/sched/scheduler.py:1026-1086`). The worker posts the NIXL READ in `start_load_kv` (`v1/worker/kv_connector_model_runner_mixin.py:95`; NIXL pull worker `distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py:42`, `_read_blocks` at `:225`) when that step starts executing, which with depth 2 is about one step after the schedule.
4. **Completion is checked once per step.** The worker calls `get_finished` (`kv_connector_model_runner_mixin.py:103`; NIXL `nixl/base_worker.py:2058`, `check_xfer_state` at `:2237`) once per executed step. A request is finished only when all 8 TP handles are done, so the slowest handle decides. The result reaches the scheduler in that step's `update_from_output` as `finished_recving` (`scheduler.py:2802`), and the next schedule promotes the request (`_update_waiting_for_remote_kv`, `scheduler.py:2723`).
5. **The engine keeps stepping while a READ is pending.** An engine with no scheduled tokens still executes a connector-only step (`kv_connector_no_forward`, `kv_connector_model_runner_mixin.py:36`). When no model ran but requests remain, `_process_engine_step` sleeps 1 ms and steps again (`core.py:1432-1447`). With DP, idle lanes run dummy batches (`execute_dummy_batch`, `core.py:927`).

Measured consequences (H200 P2-32 lifecycle, 378 decoding-lane requests):

- d = about 39.6 ms (2 steps) + the READ transfer maximum, and the transfer maximum is quantized at the step size (test report section 5 table). Requests whose 8 handles all finish within the first step (45 of 378) have d = 41.8 ms and a transfer maximum of 1.9 ms.
- e is 0.19 ms, and f is always one engine iteration and 2 steps (42.05 ms) (`evidence/iter_split.txt`).
- b is almost entirely the wait for the engine loop: 13.41 ms from the D API arrival to the update before the `add`, then 0.20 ms to the `add` (`evidence/b_split.out`).
- The P side is idle when the READ starts for 422 of 432 READs, and READ windows do not overlap across requests (`evidence/read_vs_prefill.out`, `evidence/split_d.json`). P-side load does not explain the late handles.

**Offline check of the READ rule** (`evidence/model_d_on_native_steps.out`). Price d on the native step timeline with the rule "post at the start of the step that carries the READ; the first step that starts at or after post + T detects it; that step's update reports it". With T = 2.57 ms (the K5 H800 fit used in Frontier): model d = 61.36 ms against native 77.94 ms, −16.58 ms. The rule gives 3 steps for every request; native gives 2.81 iterations on average, with 2 steps for the 45 fast requests and up to 6 for requests with late handles. The −16.58 ms is the late-handle part, section 6.2.

## 3. Frontier today

- **P end.** The NIXL branch of `KVCacheTransferStartEvent` (`frontier/events/kv_cache_transfer_start_event.py:150-264`) reserves the D blocks and schedules `KVCacheTransferEndEvent` at P end + T, T = 2.57 ms (126.2 Gbps, 0.0647 ms).
- **Transfer end.** `KVCacheTransferEndEvent` (`frontier/events/kv_cache_transfer_end_event.py:108`) calls `handle_reserved_decode_arrival` (`frontier/scheduler/utils/kv_arrival.py:6`). The request becomes ready at once and a `ReplicaScheduleEvent` follows. No engine step separates the transfer end from scheduling.
- **Batch queue depth 1 on DECODE.** DSV4 D configs leave `max_concurrent_batches` unset (checked in `../h4a_p0_diagnostic/results/run5/config.json` and the P2-32d/P2-64d configs), so capacity is `num_stages` = 1 (`frontier/scheduler/replica_scheduler/base_replica_scheduler.py:52`). An explicit value is accepted only for ordinary-cache MONOLITHIC PP1/DP1 (`vllm_v1_engine_replica_scheduler.py:369-386`). With capacity above 1, Frontier skips decoding requests that are already in an in-flight batch (`vllm_v1_engine_replica_scheduler.py:1018-1055`); that matches vLLM's non-async scheduler with a PP batch queue, not `AsyncScheduler` placeholders.
- **First D step.** Frontier's first D step is the wait for the running lane step plus one k=1 step: on H800, 39.71 ms mean (`h4a5_attribution.json`, `frontier.d_first_step`).

So Frontier misses two things: the step-quantized READ lifecycle (item 1), and the one-step lag that depth 2 adds both to the READ post and to the first token (item 2).

## 4. Item 1: READ completion at engine steps

### 4.1 Event flow

Applies only to the NIXL connector on a `vllm_v1` DECODE lane, and only in the new mode of section 5.

1. **P end** (`KVCacheTransferStartEvent`, NIXL branch). Reserve the D blocks as today. Instead of scheduling `KVCacheTransferEndEvent` at P end + T, register a pending READ on the target lane with its wire time T, and return a `ReplicaScheduleEvent` for that replica.
2. **Post.** The first engine iteration on that lane after registration carries the READ. If the iteration forms a batch, the READ is posted at that batch's execution start (`ReplicaStageScheduleEvent._schedule_popped_batch`, `frontier/events/replica_stage_schedule_event.py:124`). If it forms a dummy pass, the READ is posted at the dummy forward's start (`start_dummy_forward`, `replica_stage_schedule_event.py:167`). Wire end = post + T.
3. **Check.** Every later execution start on that lane marks posted READs whose wire end is at or before that time as detected.
4. **Report.** Detected READs are reported when that batch's output is popped: inline in `GlobalBatchEndEvent._apply_output` right after `replica_scheduler.on_batch_end(self._batch)` (`frontier/events/global_batch_end_event.py:247`), and at the dummy forward's end (`BatchStageEndEvent`, `frontier/events/batch_stage_end_event.py:169`). Reporting runs the existing transfer-end work (validation, metrics, `handle_reserved_decode_arrival`, source release), extracted from `KVCacheTransferEndEvent` into one function that both paths call. Reporting inline matters: at equal times `REPLICA_SCHEDULE` (3) runs before `KV_CACHE_TRANSFER_END` (15) (`frontier/types/event_type.py:7,19`), so a separate end event would miss the iteration that follows the pop.
5. **No batch.** An iteration that forms nothing checks and reports at once. An idle engine without DP schedules a wake-up at post + T. With DP, the existing dummy passes keep the lane stepping.
6. **Lane stays active.** `has_unfinished_requests` (`vllm_v1_engine_replica_scheduler.py:636`) also counts registered or posted READs, as vLLM counts `WAITING_FOR_REMOTE_KVS` requests, so the DP wave and dummy passes continue.

On a busy depth-1 lane this gives: post at the next step start, detection at the step start after that (T < S), report at that step's output pop, then the k=1 step. It reproduces native's "2 steps + transfer" structure up to the depth-2 lag of item 2.

### 4.2 Simplifications (stated, not fitted)

- **Check delay δ omitted.** Native checks about 1.8 ms after a step's CPU launch. Frontier checks at execution start. Because T (2.57 ms) > δ, a READ posted at a step start is never detected in that step under either rule.
- **Blocks reserved at P end.** vLLM allocates the D blocks when it schedules the READ. Frontier keeps today's reservation at P end. The difference is at most one step of block occupancy.
- **Source released at report.** P-side blocks are freed when the READ is reported, as today's end event does at the transfer end.
- **No late handles.** One wire time per READ. The step spread from late handles is open (section 6.2).

### 4.3 Files and size

| File | Change | Lines (estimate) |
| --- | --- | --- |
| `frontier/config/kv_cache_transfer_config.py` | new mode field (section 5) | 10 |
| `frontier/events/kv_cache_transfer_start_event.py` | NIXL branch registers the READ in the new mode | 20 |
| `frontier/events/kv_cache_transfer_end_event.py` | extract the completion work into one function | 30 (mostly moved) |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` (1802 lines) | READ register / post / check / report; `has_unfinished_requests`; idle wake-up | 60 |
| new `frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py` | per-lane READ table, if the engine scheduler would exceed 2,000 lines otherwise | 0-50 |
| `frontier/events/replica_stage_schedule_event.py` | post and check at batch and dummy-forward start | 15 |
| `frontier/events/global_batch_end_event.py` | inline report after `on_batch_end` | 15 |
| `frontier/events/batch_stage_end_event.py` | inline report at dummy-forward end | 10 |
| new `tests/unit/test_nixl_read_completion.py` | busy lane, idle lane, DP dummy pass, equal-time ordering, default mode unchanged | 150 |

Total: 7-8 `frontier/` files, about 160-210 changed lines plus the test. This exceeds the 5-file bound and adds a shared config field, so it returns to YC.

## 5. Gating: keep GLM identical or not

| Option | What | GLM / other results | Cost |
| --- | --- | --- | --- |
| **B (recommended)** | New field on the KV-transfer config, e.g. `nixl_read_completion`, values `transfer_end` (default, today's path) and `engine_step` (item 1). DSV4 diagnostics opt in. | Identical by construction; the default path is unchanged. | One new shared config field. A later GLM diagnostic decides whether `engine_step` becomes the only path; if it does, the old path is deleted in that change. |
| A | Item 1 replaces the NIXL path everywhere. | GLM NIXL PDD results change. YC must lift the "GLM identical" condition for this fidelity fix. | No new field. |

Recommendation: B. It satisfies the recorded GLM-identical condition now and leaves the single-path decision to evidence. The new field is the approval item.

## 6. Not in item 1

### 6.1 Item 2: DECODE batch queue depth 2 with output placeholders

Native schedules each step one step ahead of its output. This delays the READ post by one step and makes f two steps instead of one. To model it, the DECODE role needs batch capacity 2 on DP>1 with grouped KV allocation, and decoding requests that are in flight must be scheduled again with one output placeholder each, as `AsyncScheduler` does, instead of being skipped (`vllm_v1_engine_replica_scheduler.py:1018-1055`). This changes an accepted value range of `max_concurrent_batches` (today: MONOLITHIC PP1/DP1 only), the scheduler's token accounting for in-flight requests, and their interaction with preemption and KV growth. Rough size: 3-5 `frontier/` files, 150-250 lines, plus tests. It is a separate change with its own design section and approval. Expected effect in section 7.

### 6.2 Late READ handles (open)

Native needs 0.81 more steps on average than the item 1 rule (−16.58 ms on H200). The spread across the 8 handles grows with the step count (test report section 5 table). Hypotheses, none verified:

1. NIXL/UCX progress for a READ advances only inside the per-step `check_xfer_state` calls, so a large READ needs several checks.
2. The D lane's EP all-to-all traffic shares the NICs with the READ and slows some handles.
3. Some of the 8 TP handles start late because the 8 workers post at slightly different times.

Proposed measurement (needs GPUs and a later approval): a K5-style READ benchmark of 368,824,320 B per request, 8 handles, with completion checked every 20 ms instead of continuously, and the same READ under decode load. Until then, item 1 uses one wire time and the late-handle part stays a known gap.

### 6.3 Ingress (deferred)

The non-engine parts are a 0.96, about 4 ms of b (b minus the engine-loop wait that Frontier's wait for the running step already covers), and g 3.67 ms: about 8.6 ms on H200. Modeling them needs a new delay field with measured values per platform. Deferred until the H800 parts are known, so the value is not set from H200 alone.

## 7. Expected effects (inference)

With a busy D lane, the arrival instant is random with respect to the lane's step phase, so the mean wait for the running step is the same whether it is measured from P end or from P end + T. Then:

- item 1 adds 2 ordinary steps and removes T: Δ1 ≈ 2·S_F − T;
- items 1 and 2 add 4 steps and remove T: Δ12 ≈ 4·S_F − T;

where S_F is Frontier's ordinary decode step, taken as its TPOT mean, and T = 2.57 ms. TTFT values are the client-gate means.

| Case | S_F (ms), Frontier TPOT mean | Native TTFT | Frontier TTFT | Item 1: TTFT, error | Items 1+2: TTFT, error |
| --- | --- | --- | --- | --- | --- |
| H800 P0 | 19.48 (`../h4a_p0_diagnostic/results/run5/system_metrics.json`) | 398.82 | 329.63 | 366.03, −8.2% | 405.00, +1.6% |
| H200 P2-32 | 18.64 (`../h200_mainline/diagnostic/results/p2_32d/system_metrics.json`) | 364.03 | 327.23 | 361.94, −0.6% | 399.23, +9.7% |
| H200 P2-64 | 24.77 (`../h200_mainline/diagnostic/results/p2_64d/system_metrics.json`) | 741.22 | 601.00 | 647.97, −12.6% | 697.50, −5.9% |

Notes:
- H800 handoff with items 1+2: 42.28 + 75.37 = 117.65 ms against native 123.49 ms.
- P2-32 overshoots with items 1+2 because Frontier's P side is already 45.9 ms longer than native (P2 diagnostic report section 6); the handoff error itself shrinks.
- These are arithmetic estimates from the formulas, not simulator runs. The verification runs in section 8 replace them.

## 8. Verification plan

1. Default mode (`transfer_end`): the existing unit suite (`tests/unit`), `bash examples/architecture/run_dummy_smoke_matrix.sh`, `pytest tests/unit/test_e2e_gate.py -q`, and the GLM, co-location, PDD and PD-AF comparison cases already recorded for this task. Expected: byte-identical metrics files against the pre-change baseline.
2. New unit test `tests/unit/test_nixl_read_completion.py`: post time, detection step, report at output pop, the following iteration schedules the request, idle-lane wake-up, DP dummy-pass path, and the equal-time event order.
3. New mode on DSV4: rerun P2-32d, P2-64d and the native06 (H4a) diagnostic with `engine_step`. Compare parts c-f per request with P2-32l, and with the H800 lifecycle copy once it exists. Expected: d about 3 Frontier steps, matching the offline rule within its −16.6 ms late-handle gap; TTFT moves as in section 7.
4. All results stay DIAGNOSTIC. H200 evidence never qualifies an H800 cell.

## 9. Approval items for YC

1. Item 1 with option B: the new KV-transfer config field `nixl_read_completion` (default `transfer_end`), 7-8 `frontier/` files, about 160-210 lines. Or option A, which changes GLM NIXL results.
2. Item 2 (DECODE depth 2 with placeholders): approve a separate design section first.
3. Ingress field: deferred until the H800 parts are known.
