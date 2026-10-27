# Design: calibration item 3, host time of the KV block reset

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Corrected after the codex review of the probe run (`codex-review-223fee59-.../review.md`, F001–F004): completeness is checked on every TP rank against the client-completed work, with no preemption in a window and partial all-reduces only after the last step with requests; the formal phases' native work counters must match the client-completed requests; the lost probe steps lie inside the last window; the perturbation check failed and is not settled (4.1, 4.3, verification 4, Q6). |
| 2026-10-27 | Corrected after the h200_32 probe run: the launcher force-kills the workers, so probe files have no `end` line (completeness from gap-free steps and equal window tokens); idle boundaries expect completions only for client-completed requests (4.1, 4.3, F002). |
| 2026-10-27 | Revised after the codex implementation review (`codex-review-668b0946-.../review.md`, section 9): reference admission rejects probe runs; the worker owns the probe variable; shutdown drains the probe and writes an `end` line; dummy steps end at `_dummy_run` exit; the DP wait is the gloo `all_reduce` alone; the engine interval runs from its earliest first sync to its last launch on one node clock; formal windows come from the verified idle boundaries; the request-count trend rule writes a table. |
| 2026-10-27 | Probe and estimator implemented; 4.1–4.3 revised to the implementation: every TP rank records, line-buffered per-process files, the interval starts at the first `.tolist()` return, event-pair device gaps, formal windows by wall time, the mean as the profile value, 4 probe and 6 estimator unit tests. |
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
  and quartiles. If the probe shows that a role's value depends on the request count by more
  than 1 ms over the observed range, that role's entry is a table keyed by request count
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
  probe only for this kind (config flag, staged bootstrap and server environment must agree)
  and still rejects every other diagnostic hook. `formal_release.py` lists the kind in
  `DIAGNOSTIC_KINDS`, and reference admission (`e2e_gate/admission.py` `client_reference`)
  rejects a run whose config or client summary declares a diagnostic kind, whose config has
  `host_step_probe`, or whose node records show the probe variable in the server environment,
  even with a passing clean-run receipt.
- The worker adds `host_probe_bootstrap/` (a `sitecustomize.py`) to the servers'
  `PYTHONPATH` with `STEP5_HOST_STEP_PROBE` set, like `pd_state_observation`. The worker owns
  that variable: an inherited value is removed for every server, so only a run that declares
  `host_step_probe` is instrumented. The bootstrap
  wraps the methods of 4.2 when their modules are imported. The first probe error disables
  the probe in that process, writes an `error` line and prints
  `STEP5_HOST_STEP_PROBE UNVERIFIED`; the probe never changes arguments, returns or
  exceptions of native code and calls every original exactly once.
- Every TP rank records, because the DP all-reduce runs once per TP rank and the forward
  starts when the slowest TP rank launches. Each worker process writes its own file
  `<role>_<host>_<pid>.jsonl` to worker-local scratch with a header line (host, role, TP and
  DP rank and size), line-buffered so a killed process loses no finished line. Native
  `GPUModelRunner.shutdown`, which synchronizes the device first, would write the steps still
  waiting for their events and a final `end` line with the step count. The native P/D launcher
  stops its engines with `mode=abort` and force-kills the worker processes (observed in
  hostprobe03 and sweep03), so that path does not run: a file ends with its last finished
  line, and the last step and the 1–2 steps whose events were still pending are lost. In
  hostprobe03 they lie inside the last formal window (it ends at the idle boundary seconds
  later) but after the role's last step with requests; the estimator rejects a loss that
  touches a step with requests (4.3). Forked workers leave through `os._exit`, so `atexit` cannot
  help. After the server stops, the node gzips the files into `host_probe/` of its node record.

### 4.2 Hooks (all at seams that run for real workload forwards)

| Seam | Records |
| --- | --- |
| `GPUModelRunner.execute_model` enter/exit | per-process step sequence, wall time, `dummy_run` and `is_profile` flags, request and token counts |
| `KVBlockZeroer.zero_block_ids` enter/exit (only inside a step; the startup warmup reset is ignored) | block count, block count per group, hook count |
| First side-storage hook `Step5SparseSummaryCacheLayer.zero_kv_cache_side_storage` (wrapped on the class at import; the zeroer binds the hooks later at KV setup) | enter, exit, the return of its `_ids` `.tolist()` (`first_sync`), and whether the previous step's end event was still pending at enter (`prior_pending`) |
| `dp_utils.sync_cudagraph_and_dp_padding` enter/exit (module global, which `dispatch_cg_and_sync_dp` resolves) | collective call ordinal per process; a step with 0 tokens returns before the call and has none |
| `dist.all_reduce` inside `dp_utils` only (a proxy for the module's `dist`; other all-reduces are untouched) | `reduce_enter`/`reduce_exit` around the gloo all-reduce, the only call in the sync that waits for the other engines |
| `StepTimingCollector.forward_start` | launch stamp and CUDA event |
| `GPUModelRunner.sample_tokens` exit (real step) or `_dummy_run` exit (dummy step; it queues more main-stream work after its `execute_model` returns) | stamp and CUDA event at the end of the step's main-stream work |
| `GPUModelRunner.shutdown` exit | drains the remaining steps and writes the `end` line (not reached when the launcher force-kills the workers, 4.1) |

When a step's events complete, its line gets `gap_ms` (main-stream interval from the
previous step's end event to this launch event; reset and input-preparation GPU work can
fall inside it, so it is not a pure device-idle time) and `span_ms` (launch to end). Event
pairs avoid the float precision loss of times measured against one early reference event. `should_skip_dp_coordination()` is set
only by elastic EP, which this workload does not use, so every call runs the all-reduce.

### 4.3 Estimator (`tests/integration/step5_pd/kv_zeroing_host_time.py`)

- Inputs: the copied stage (config, probe files, client summary), the clean run's client
  summary, the job ID and `--min-samples`.
- A role is usable only with exactly one error-free process per (DP rank, TP rank). A restart
  (two processes for one rank), a missing rank, a file whose step numbers do not run from 1
  without gaps, or TP ranks of one engine on different hosts stop the estimator. Completeness
  inside the windows comes from the scheduled tokens, on every process: in each formal window
  the TP ranks of an engine must hold equal non-dummy tokens, and the engines together exactly
  the work of the window's client-completed formal requests (P: the transferred whole blocks,
  `prompt − 16` = 6,128 per request; D: the last block plus one step per output after the
  first, 16 + 1,023). Work of another request inside a window, or a lost step with requests,
  breaks one of the two. A preempted request recomputes its tokens and could hide a lost step,
  so `vllm:num_preemptions_total` must not change across a formal window on either role.
- Only steps whose wall time lies in a formal window count. A window runs from the idle
  boundary after a repeat's warmup to the idle boundary after its formal phase (client
  monotonic times of each boundary's last observation, converted with the phase's wall-clock
  offset). The estimator re-checks each boundary from that observation with the replay's rule:
  every lane idle, and each role's completion count at least the count at the verified
  length-sweep boundary plus the requests the client saw complete in the phases since. A
  request that failed at the client may or may not have run on a role, so it adds no expected
  completion. The idle gauges are snapshots from the last engine output, so a failed request
  may also still run after a boundary; the exact token check above rejects a window that holds
  any of its work. Every engine was idle at both boundaries, so the edges lie in idle gaps. Server and client wall
  clocks may differ; the report gives the smallest distance from an edge to any step
  (`window_edge_clearance_s`), which must exceed that difference. A boundary that fails the
  check, or an unverified length-sweep boundary, stops the estimator.
- Join the engines by collective ordinal per TP rank. An ordinal may lack a participant only
  after the role's last ordinal with a non-dummy step (the force-killed tail); an earlier
  partial ordinal, such as a useful step lost on a nonzero TP rank or a lost dummy peer of a
  resetting step, would drop or bias a sample and stops the estimator. All TP ranks of one engine must agree on the batch (dummy flag,
  request and token counts) at an ordinal; a disagreement means the join is broken and stops
  the estimator. Dummy steps take part in the collective but yield no sample.
- For all-reduce `k` at TP rank `r`, the engine that arrived last waited least, so its
  all-reduce duration (`reduce_exit − reduce_enter`) is the collective's own cost and
  `excess` = duration − minimum over the engines (no cross-node clock is needed). The local
  tensor preparation before the all-reduce and the graph dispatch after it stay in the host
  time. For a resetting step of engine `i` (every TP rank ran at least one hook and launched),
  on the node's shared monotonic clock:

  `h_i = max over TP ranks r of (launch_r − excess_r) − min over r of first_sync_r`.

  The forward starts when the slowest TP rank launches. The earliest first `.tolist()` return
  marks the end of the previous step's device work (the device finished, so a rank that
  waited returns at once); from then on the device idles until the launch. The interval
  contains the rest of the first hook, the other 22 hooks, the input preparation, the
  collective's own cost and the dispatch.
- Profile value per role: the mean of `h` over the resetting steps, because the mean sets the
  aggregate TPOT effect. When the least-squares line of `h` over the observed request counts
  changes by more than 1 ms (`TREND_LIMIT_MS`) across that range, the role's entry is instead
  the mean per request count (`by_num_requests`, which the loader interpolates and does not
  extrapolate). The rule applies to both roles. A role with fewer resetting steps than
  `--min-samples` stops the estimator after the report is written; no profile is written then.
- Report (not used for the value): distribution of `h`; the per-rank variant
  `max_r(launch_r − excess_r − first_sync_r)`, the rank-0-only value and the TP skew of
  `first_sync`; the share of steps on which a rank's previous step was still running at the
  first hook and `h` for both subgroups; `h` per request count and the trend; hooks per step
  (expected 23); the wait for other engines; the joined and trailing partial collective
  counts; the tokens per engine and window; the window edge clearance.
- Cross-check: the device gap `max over TP ranks of (gap_ms − excess)` for resetting steps
  whose previous step has `span_ms`, against `h`; and the same gap for steps that do not reset.
- Perturbation check: the probe run's client TTFT and TPOT means against the clean run of the
  same case. A difference larger than the clean run's repeat range marks the profile
  `PERTURBED` in its provenance.

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
3. Probe and estimator synthetic tests (`tests/unit/test_step5_host_step_probe.py`,
   `tests/unit/test_kv_zeroing_host_time_estimator.py`): native results unchanged, a probe or
   bootstrap error never blocks native code, a forked worker starts its own file, the
   shutdown drain writes every step and the `end` line without `atexit`, a dummy step ends
   after `_dummy_run`'s tail, only the DP sync's all-reduce is stamped; a known local term
   with unequal collective waits, local work around the all-reduce kept in the host time,
   TP ranks with skewed first syncs, dummy participants, unmatched collectives, the idle
   boundary windows and their clearance, a TP-rank batch disagreement, a missing rank, a
   restart, a probe error, a truncated file, TP ranks on two hosts, the perturbation status,
   the request-count table above the trend limit, and a role without enough samples writes no
   profile; the written profile loads for the run it was measured on. Worker and admission:
   an inherited probe variable never reaches a clean server; a probe run is rejected as a
   reference even with a passing receipt, while the clean control passes.
4. Probe runs: one launch/`sample_tokens` pair per real forward; collective ordinals agree
   across engines; 23 hooks per resetting step; perturbation within the clean repeat spread.
   The h200_32 run failed the last check (TPOT +0.357 ms against a clean repeat range of
   0.207 ms), so its profile is `PERTURBED`: E reruns with it are exploratory sensitivity runs.
   Accepting it needs a measured probe effect, for example probe on and off within one job.
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

## 9. Review response (codex implementation review 668b0946, 2026-10-27)

| Finding | Fix |
| --- | --- |
| F001 (P1): a declared host-probe run could pass reference admission with a passing receipt. | `client_reference` rejects a diagnostic `canonical_replay.kind` or summary `replay_kind` (`formal_release.DIAGNOSTIC_KINDS`), a config with `host_step_probe`, and a node whose `server_env_set` holds `STEP5_HOST_STEP_PROBE` (4.1). Three new faults in `test_step5_client_metrics.py`; the clean fixture still passes. |
| F002 (P2): forked workers leave through `os._exit`, so the `atexit` flush never ran and steps were lost. | The drain runs at native `GPUModelRunner.shutdown` exit, after its device synchronize, and writes an `end` line; the estimator rejects a file without it or with missing steps (4.1, 4.3). The sweep03 servers stopped through SIGINT, which takes this path. Corrected after hostprobe03: the launcher force-kills the workers, so no `end` line is written; completeness comes from gap-free step numbers and equal window tokens (4.1, 4.3). |
| F003 (P2): the whole sync function's duration was taken as the wait, so local preparation and graph dispatch were subtracted. | The probe stamps only the gloo `all_reduce` through a proxy for `dp_utils.dist` (other all-reduces untouched); `excess` uses that duration (4.2, 4.3). |
| F004 (P2): the estimator always wrote a scalar. | Least-squares trend over the request-count range; above 1 ms the role's entry is the per-count mean table, for both roles (4.3). |
| F005 (P2): an inherited `STEP5_HOST_STEP_PROBE` survived clean server preparation. | `host_probe_environment` removes the inherited value for every server and sets it only for a declared probe run (4.1). The variable is reachable only through the image or launch environment, which do not set it today; the removal makes the worker its only owner. |
| F006 (P3): the dummy end event was recorded before `_dummy_run`'s remaining main-stream work. | The dummy end event is recorded at `_dummy_run` exit (4.2). |

Open questions:

- Q1 (engine interval): the interval now runs from the engine's earliest first sync to its
  last launch on the node's shared monotonic clock (TP ranks on one host are checked). The
  earliest first sync is the device-done point because a rank that waited for the device
  returns as soon as the device finishes; a later first sync on another rank is host work of
  that rank and is part of the stall. The per-rank variant and the first-sync skew are
  reported for comparison.
- Q2 (first sync as the idle origin): kept, with its limit. On a step whose previous step had
  already finished at the first hook, the device idled before the first sync, so `h` is a
  lower bound there. The report splits `h` into waited and idle steps; a material difference
  between them is checked before the profile is used.
- Q3 (server and client clocks): windows now end at verified idle boundaries, and the report
  gives the edge clearance; a clearance of seconds bounds any plausible clock offset.
- Q4 (mean versus `E[max]`): the mean stays the design statistic; the reruns of E and G
  against the clean references test whether it reproduces the aggregate effect.
- Q5 (external clean-run-check producer): not audited here; the consumer now rejects probe
  runs on its own.
- Q6 (live coverage and perturbation): coverage is settled by the probe runs (4.3). The
  perturbation is not: the h200_32 run failed the TPOT check, and the size of the probe's
  effect on `h` is not measured (verification 4).
