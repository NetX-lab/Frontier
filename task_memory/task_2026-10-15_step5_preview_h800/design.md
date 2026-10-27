# Design: calibration item 3, host time of the KV block reset

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Frontier side implemented (section 8); added the NIXL READ profile regeneration (3.4). |
| 2026-10-27 | Revised after the codex design review (`codex-review-c85d179e-.../review.md`): 23 hooks / 46 writes; the host time joins the DP forward-start agreement before first attention; direct host-interval probe at real seams; DP/TP/PP/graph validation; review response in section 7. |
| 2026-10-27 | First version: native mechanism, Frontier model, measurement probe, verification. |

## 1. Problem

Native Step-5 decode TPOT is 13–15% above Frontier at 20–40 requests per engine
(`test_report_2026-10-26_decode_step_slope.md`, sections 6–9). The device timeline shows the
cause: some steps leave the GPU idle for about 9 ms between forwards. Frontier models no
host time for Step-5 (`skip_cpu_overhead_modeling=True`), so its decode step equals the
device step of the native steps that do not stall.

## 2. Native mechanism (source `/data/ycfeng/tmp/d29_published_sources01/source/vllm`)

Scope: the canonical Step-5 PDD workload: MTP off, PP1, async scheduling, DP coordination on,
V2 GPU model runner.

1. Step-5's sparse summary caches mark their KV group `requires_zeroing`
   (`models/step5/nvidia/model.py:1243-1247`), so `KVCacheConfig.needs_kv_cache_zeroing` is
   true (`v1/kv_cache_interface.py:1792-1806`). Base KV pages are not zeroed (no Mamba layers,
   no mixed-precision KV); only the side storage is reset.
2. Each step the scheduler drains the block IDs allocated since the last step
   (`v1/core/sched/scheduler.py:1622-1650`). For an asynchronous NIXL load, the blocks of the
   external (loaded) tokens in the loaded groups are allocated with `record_for_zeroing=False`
   (`scheduler.py:1297-1300`, `v1/core/kv_cache_coordinator.py:336-342`). Later local
   allocations of the same request (its first decode block, a D-side prompt tail) are
   recorded.
3. In the worker, `execute_model` first calls `update_requests`
   (`v1/worker/gpu/model_runner.py:1757-1764`), which calls `KVBlockZeroer.zero_block_ids`
   (`:1273-1280`, `v1/worker/utils.py:580-598`): one asynchronous H2D copy of the flattened
   block IDs, then one side-storage reset hook per sparse layer.
4. The model has 92 layers: 69 sliding-window and 23 sparse
   (`data/config/models/stepfun-ai__Step-5-Preview.json`). So a resetting step runs **23
   hooks**. Each hook converts the GPU block-ID slice with `.tolist()`
   (`models/step5/nvidia/sparse_summary_cache.py:486-488, 507-509`) and makes two indexed
   writes, `mean[list] = 0` and `last_slot[list] = -1` (`:353-355`): **46 indexed writes**
   per resetting step, whatever the number of blocks. The retained profiler trace shows one
   `cudaStreamSynchronize` inside each indexed write.
5. The first `.tolist()` waits until all earlier work on the main stream has finished:
   the previous step's forward, sampling and request-state updates. Output copies run on a
   separate copy stream and are not part of this wait (`v1/worker/gpu/async_utils.py:137-171`).
   Under async scheduling the host normally prepares step N+1 while the GPU runs step N. In a
   step that resets, the host instead waits for step N's device work, and the rest of the
   resets, the DP collective, the input preparation and the forward launch then run while
   the GPU has no forward to execute.
6. DP coordination: `dispatch_cg_and_sync_dp` calls `sync_cudagraph_and_dp_padding`
   (`v1/worker/gpu/dp_utils.py:44-77`) unless DP size is 1 or a DP sync state is reused
   (`:290-310`). It runs a blocking gloo `all_reduce` on the DP CPU group unless
   `should_skip_dp_coordination()` (`:74-75`). The collective precedes input preparation
   (`model_runner.py:1810-1842`) and the forward (`:2022-2033`). A step that resets on one
   engine therefore delays the forward of every engine in that DP group.
7. NIXL: the asynchronous READs start in `KVConnector.post_forward`
   (`v1/worker/gpu/kv_connector.py:127-131`), after the forward launch, so a reset delay
   also delays the READs that the step posts.

A decode step resets when any of its requests opens a new 16-token block. The profiled share
matches this: 31% of steps at 5 requests, 59% at 20 requests, all steps at 40–45 requests
(block-boundary model: 28%, 72%, 92–95%). Every prefill step allocates blocks, so every
prefill step resets.

## 3. Frontier model

### 3.1 When a step pays

- Frontier already records the native drain. With `AttentionKVCacheCapacity.zero_new_blocks`,
  the allocator returns the new block IDs (`frontier/kv_cache/base_kv_cache_manager.py:374-381`)
  and `_create_batch` attaches them to the batch as `batch.cache_block_zeroing`
  (`vllm_v1_engine_replica_scheduler.py:452-473`). DeepSeek-V4 uses this today.
- Change: Step-5's capacity (`frontier/attention/step5_memory.py:283`) sets
  `zero_new_blocks=True`, because native Step-5 has `needs_kv_cache_zeroing=True`. The only
  existing timing consumer of `cache_block_zeroing` is DeepSeek-V4
  (`execution_time_predictor/deepseek_v4.py:267-279`), so this alone changes no timing.
- Change: a NIXL read reservation (`_allocate_request` for a receiving request,
  `vllm_v1_kv_allocation.py:694-739`) records no blocks for zeroing, as native skips the
  external-token allocation in the loaded groups. Later local allocations of that request are
  recorded.
- A lane's step pays when its batch carries `cache_block_zeroing`. A dummy forward never pays
  locally.

### 3.2 What it pays and where

The term is placed at the existing Step-5 DP forward-start agreement, before any lane
schedules its first attention. It is not part of `OverheadTime`.

- `hold_forward_start` (`frontier/scheduler/utils/forward_start_agreement.py:83`) already
  holds every lane's admitted batch or dummy forward until all lanes of the stage have
  arrived, then resumes them through `ReplicaStageScheduleEvent`s. Change:
  - When a lane is held, its batch records its arrival time `a_i`.
  - When the last lane arrives (time `t`), each lane gets its local host time `h_i` from the
    predictor (0 without the profile, without `cache_block_zeroing`, or for a dummy).
  - The agreed start is `s = max_i (a_i + h_i)`, which is at least `t`. Every batch of the
    group, dummies included, records the host delay `s - t`.
- The resumed stage event then uses the delay for every lane:
  - Prefill and decode (`replica_stage_schedule_event.py:270` and `:564`):
    `_forward_launch_start_time = time + host_delay + preparation`. The first-attention sync
    event is scheduled from this launch start, as today.
  - A dummy forward (`dp_dummy_forward.py:67-84`) starts its first layer at
    `time + host_delay`.
  - The DECODE engine-step start (`on_engine_step_start`, which posts and finds NIXL READs)
    runs at `time + host_delay`, because native posts the step's READs after the launch.
  - The stage start time stays at `time`, so the delay is part of the step's stage time, and
    `prepare_*_final_timing` sees an unchanged preparation term (no double counting).
- So a resetting lane delays every lane's first attention, as the native DP collective does.
  Example: host times 9/0 ms and first attention 1/5 ms give a first pre-MoE readiness of
  9 + 5 = 14 ms on both lanes.
- The role comes from the cluster type: a PREFILL cluster uses the prefill entry, a DECODE
  cluster the decode entry, including D-side prompt-tail batches.
- Without the profile every delay is 0 and nothing changes.

### 3.3 Where the value lives

- New optional input `ExecutionTimePredictorConfig.kv_block_zeroing_host_profile`: a JSON
  file, like `nixl_read_profile`, and bound into the predictor bundle identity the same way
  (`shared_prediction_model_manager.py:147`).
- Contents (schema `kv_block_zeroing_host_time.v1`): model architecture profile, device,
  topology (attention TP, attention DP, MoE EP, pipeline stages), decode graph mode, runtime
  scope (async scheduling, MTP off), provenance (job, Frontier commit, probe source hash, image
  digest), and per role (`prefill`, `decode`) the local host time in ms with its sample count
  and quartiles. If the probe shows that the decode value depends on the request count by more
  than 1 ms over the observed range, the decode entry is a table keyed by request count
  (linear interpolation; a request count outside the table is an error, no extrapolation).
- The loader fails with a clear message when:
  - the schema, model profile, device, attention TP, attention DP, MoE EP, pipeline stages or
    decode graph mode differ from the run;
  - the run's cluster type has no entry, or is not PREFILL or DECODE;
  - the run uses dummy predictor mode (there is no forward-start agreement there);
  - CPU-overhead modeling is on (the measured interval already contains the input
    preparation of a resetting step).
- The value is a component host-time term measured by a probe, which D31.10 allows. The
  device time of every step still comes from the operator profiles.

### 3.4 Effect on the NIXL READ profiles

The NIXL READ profile (`nixl_read_profile`) records the run's full KV cache capacity, and its
loader compares that record with the run, including `zero_new_blocks`. With 3.1, Step-5
capacity sets `zero_new_blocks=True`, so the existing H200 profiles no longer load. Both were
regenerated from the same native server logs with the same command and the new capacity:
`evidence/d31_h200_32_nixl_read_cost_20261027.json` and
`evidence/d31_h200_64_nixl_read_cost_20261027.json`. The READ values are unchanged (51.37 and
56.55 ms). The old files stay as the inputs of the earlier E and G runs; runs from the item 3
commit on use the new files.

## 4. Measurement: host-step probe run (new diagnostic native run kind)

The torch-profiler numbers (8.7–9.7 ms) are upper bounds because the profiler slows host
code. The probe measures the host interval directly with `perf_counter` stamps; CUDA events
are only a cross-check.

### 4.1 Run kind and installation

- Run kind `host_probe` in `canonical_stage.py`: the same workload, length sweep and engine
  as `workload`, with probe hooks on the prefill and decode servers. `verify` accepts the
  probe only for this kind and still rejects every other diagnostic hook. Staging, replay and
  formal-release admission never accept a probe run as a reference.
- The worker adds `host_probe_bootstrap/` (a `sitecustomize.py`) to the servers'
  `PYTHONPATH` with `STEP5_HOST_STEP_PROBE` set, like `pd_state_observation`. The bootstrap
  patches modules when they are imported. Every probe error is caught and logged as
  `STEP5_HOST_STEP_PROBE UNVERIFIED`; the probe never changes arguments, returns or
  exceptions of native code.
- Only TP rank 0 of each engine records. Records go to worker-local scratch as JSONL, flushed
  every 256 steps and at exit; the node copies them into its record, as for `pd_state`.

### 4.2 Hooks (all at seams that run for real workload forwards)

| Seam | Records |
| --- | --- |
| `GPUModelRunner.execute_model` enter/exit | step ordinal, `dummy_run` flag, request and token counts, whether it returned before dispatch (no-forward) |
| `KVBlockZeroer.zero_block_ids` enter/exit | block count; hook count |
| First side-storage hook (bound methods wrapped in place on first call) | enter, exit, and whether the previous step's `sample_tokens` exit event was still pending at enter (`query()`), which means the hook waited |
| `dp_utils.sync_cudagraph_and_dp_padding` enter/exit (module global, which `dispatch_cg_and_sync_dp` resolves) | collective ordinal, skip flag from `should_skip_dp_coordination()` |
| `StepTimingCollector.forward_start` | stamp and CUDA event |
| Launch: `cudagraph_manager.run_fullgraph` / `run_pw_graph` enter, or a forward pre-hook on the model for eager | stamp and CUDA event; graph mode |
| `GPUModelRunner.sample_tokens` exit | stamp and CUDA event (end of the step's main-stream work) |

Startup and capture records are marked by phase (before the first workload request) and kept
apart. Each record carries the process generation (pid and start time).

### 4.3 Estimator (`tests/integration/step5_pd/kv_zeroing_host_time.py`)

- Join the engines of one DP group by collective ordinal within (role, instance, process
  generation). Only calls that ran the all-reduce count; skipped and early-returned calls do
  not. Dummy forwards take part. A group whose ordinal counts disagree is reported and
  excluded.
- For an engine step `k` that reset and whose first hook waited:
  - `exposed_k` = launch stamp − first hook exit;
  - split: remaining resets, before the collective, collective, after the collective;
  - the collective time includes waiting for slower engines. The engine that arrived last
    waited least, so `intrinsic_k = min over the group of the collective duration` (no
    cross-node clock is needed);
  - `local_k = exposed_k − (collective_k − intrinsic_k)`.
- Per role (and per request count if the trend exceeds 1 ms): median, quartiles, sample count
  of `local_k`. A role with fewer admitted samples than the `--min-samples` argument gets no
  entry, and the script refuses to write a profile without both roles.
- Reported but not used for the value: the share of resetting steps whose first hook did not
  wait, and their intervals; the resetting-step share per request count against the
  block-boundary model; hook counts per resetting step (expected 23).
- Cross-check: device gap = launch event[k] − `sample_tokens` exit event[k−1]
  (`elapsed_time`, read after `query()`), compared with `exposed_k`.
- Perturbation check: the probe run's native TPOT/TTFT per request-count bucket against the
  clean sweep of the same scale. A difference larger than the clean sweep's repeat spread
  marks the profile UNVERIFIED.

### 4.4 Runs

One probe run per DP size, because the collective and its intrinsic time depend on it:
h200_32 (DP2, for step E) and h200_64 (DP4, for step G). h200_32 can run together with the
h200_32 sweep03 rerun (64 GPUs in total, within D31.2); h200_64 runs after both end.

## 5. Verification

1. Unit tests:
   - loader: a valid load; a mismatch of schema, model, device, DP size (a DP2 profile in a
     DP4 run), TP, PP or graph mode; a missing role; dummy mode; CPU-overhead modeling on; a
     request count outside the table;
   - trigger: a Step-5 batch carries `cache_block_zeroing` exactly on block-opening steps; a
     NIXL read reservation records none; a later local allocation of the same request is
     recorded;
   - composition: the 9/0 + 1/5 ms example gives 14 ms first pre-MoE readiness on both lanes,
     checked on the actual first-attention event times; useful/useful and useful/dummy lanes;
     PREFILL and DECODE; DECODE NIXL posting moves by the delay; event times unchanged without
     the profile.
2. Dummy smoke matrix and the 77-case fidelity comparison: unchanged results without the
   profile.
3. Estimator synthetic tests: a known local term with unequal collective waits, skipped and
   early-returned collectives, dummy participants, missing records and a process restart; a
   role without enough samples emits nothing.
4. Probe runs: one launch/`sample_tokens` pair per real forward; collective ordinals agree
   across engines; 23 hooks per resetting step; perturbation within the clean repeat spread.
5. Reruns of step E (h200_32, DP2 profile) and step G (h200_64, DP4 profile) against the clean
   sweep references. Expectation: TPOT error at 20–40 requests drops from 13–15% to the
   remaining operator gaps (item 2).

## 6. Not in scope

- Host time of steps that do not reset (async overlap holds; the profiled residual is
  0–1.4 ms).
- A general async-scheduling host model for every CPU-overhead term.
- DeepSeek-V4: its zeroing stays a device operator; the NIXL read-reservation rule applies to
  it too, matching native.
- Co-location Step-5: the loader rejects the profile for a MONOLITHIC cluster.

## 7. Review response (codex review 2026-10-27, verdict changes_required)

| Finding | Fix in this design |
| --- | --- |
| F001 (P1): the DP maximum was taken after first attention, so a lane with short attention hid part of a peer's host stall (9/0 + 1/5 ms gave 10 ms instead of 14 ms). | The term joins the forward-start agreement before any first attention (3.2); dummies take the same delay; unit test on actual event times (5.1). |
| F002 (P1): `StepTimingCollector.forward_end` is called only in `_dummy_run`, so real forwards had no end hook. | Hooks at `execute_model`, the launch call and `sample_tokens` exit, which run for every real forward (4.2); records keep real, dummy, no-forward and startup apart. |
| F003 (P1): the reset-minus-non-reset gap difference is not identified and has no control group for prefill or 40+ requests. | Direct host interval per resetting step from the first hook's exit to the launch, minus the wait for slower DP engines (4.3); no control group needed; a role without enough samples emits nothing; device gap only as a cross-check; perturbation check. |
| F004 (P2): the loader did not validate DP size although the interval contains the DP collective. | The loader checks DP, TP, EP, PP and graph mode, and rejects dummy mode and CPU-overhead modeling (3.3); one probe run per DP size (4.4). |
| F005 (P3): 46 was the count of indexed writes, not hooks. | 23 hooks, 46 indexed writes (2.4); the probe checks the hook count (4.3). |

Open points from the review, handled as follows:

- Native NIXL scope: only the external-token allocation of a read reservation skips
  recording; later local pages of the request are recorded (3.1), with a unit test.
- Collective joining across startup, skipped coordination and restarts: phase and process
  generation in every record, ordinals only for calls that ran the all-reduce (4.2, 4.3).
- Table range: no extrapolation; an out-of-range request count is an error (3.3).

## 8. Implementation (Frontier side)

Implemented as designed in 3.1–3.3; checks in `test_report_2026-10-27_kv_block_zeroing_host_time.md`.

| Part | Code |
| --- | --- |
| Profile loader and lookup | `frontier/execution_time_predictor/kv_block_zeroing_host_time.py` (`KVBlockZeroingHostTime`) |
| Config field | `ExecutionTimePredictorConfig.kv_block_zeroing_host_profile`, bound into the predictor bundle identity in `shared_prediction_model_manager.py` |
| Per-lane host time | `BaseExecutionTimePredictor.get_kv_block_zeroing_host_time_ms(batch)`; the predictor rejects the profile without the DP forward-start agreement (dummy mode or a non-Step model) |
| Agreement | `forward_start_agreement.py`: arrival time per held batch; `s = max(a_i + h_i)`; every batch records `_forward_start_host_delay = s - t` |
| Consumers | `replica_stage_schedule_event.py` (prefill and decode launch start, DECODE engine-step start), `dp_dummy_forward.py` (dummy first layer) |
| Trigger | `step5_memory.py` (`zero_new_blocks=True`), `vllm_v1_kv_allocation.py` (a NIXL read reservation records no zero blocks) |

One difference from 3.3: the dummy-mode rejection sits in the predictor, not in the loader,
because the loader has no access to the predictor mode.
