# D29-FULL-WORKLOAD-MULTIPLICITY report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Reconfirm90,200 waves; correct uniform208 event extrapolation for BF16 layers, separate expected counts from observations, record capacity units and the investigator's out-of-bound CSV read. |
| 2026-10-23 | Parent aligns process and scheduler clocks, withdraws the unsupported iteration-4 slowdown, and labels byte-derived row equivalents correctly. |
| 2026-10-23 | Bounded determination of the decode EP-wave multiplicity, trace volume and projected wall cost for the unchanged main08 workload. |
| 2026-10-23 | Correction pass for root acceptance: exact phase/operator split of the 208 events; layer-60/61 proof-identity wording; token/prefill-chunk bookkeeping from the production replay target; projection relabelled as a current-implementation estimate with the cost-fields emission value. |

Task ID **D29-FULL-WORKLOAD-MULTIPLICITY**. Read-only. Source fixed at
`48d34106` (code `eece87b2`), `/data/ycfeng/Frontier/.worktrees/step5-preview`.
Python 3.10.6 (`/data/ycfeng/envs/frontier-py310`) for the one source-count
probe; every other number comes from retained artifacts plus bounded (<= 8 MiB)
trace windows. No full CLI, trainer, GPU, network, dependency install, source,
test or runtime-input edit.

## Current reconciliation: 2026-10-24

Source db581ff6 and retained main08 scheduler records reconfirm one6128-token
PREFILL, one16-token DECODE replay committing one output, then1023 one-token
DECODE forwards. Thus90,200 waves is source-backed. The old uniform208-event
projection was too large: layers3–87 use block-FP8 and select `moe_input_quantize`;
layers88–90 use BF16 and do not. Applicable prices must be positive
(`moe_operator_times.py:127–141`); operator projection includes input quantization
only when selected (`execution_time.py:956–959`). The expected successful-run
EP-wave count is `(85×208 + 3×192)×1025 = 18,712,400`,49,200 below the old
18,761,600 estimate. This is not a completed-run census: P6128, D16 and BF16
emission remain unmeasured. Other, non-wave JSONL events are additional.

The all-costs-zero allowance is `2400/90200 = 26.608ms/wave`. The retained
key-only39.956ms FP8 emission diagnostic is above that average allowance;
assigning its rate to every wave projects3604s, not an observed or universal
lower bound. Even restricting that projection to the86,955 steady FP8 waves
gives3474s. The accepted snapshot's0.162715s pricing and the old0.026s flush
are separately revision-bound measurements. The actual key implementation,
other shape/group timings and post-repair startup are not established here.
Earlier timing tables below remain historical checkpoint projections.

The old262.1GB/mode figure uses two sampled FP8 DECODE wave sizes. Adjusting
only event multiplicity suggests about261.4GB, but BF16 metadata sizes are
unmeasured and non-wave rows add output. Neither number is an upper bound.
The appropriate storage statement is an approximately520GB two-mode projection.
Parent `df -B1 --output=source,target,size,used,avail /data/ycfeng/tmp` observed
265,987,776,512 available bytes (265.99GB,247.72GiB), not248GB. A single-mode
EP-only projection leaves only a few GB before other output; both modes exceed
available space by roughly250GB. `/home/brainpp` and `/mnt` resolve to the same
root filesystem, with59,530,031,104 bytes available. Capacity is a point-in-time
observation, not reserved space. No cleanup or new storage operation is approved
by this report. Any cleanup must follow D8's handoff scope, retain cited evidence
first, and preserve still-used directories and other sessions' work
(`task_memory/task_2026-10-15_step5_preview_h800/requirements.md:389`).

The private ownership diagnosis is closed without allocation; see
`../detached_emission_ownership/parent_receipt.md`. Neither the estimated<=3ms
interface change nor the2.5ms scalar memo settles this workload's gap.

### Investigation deviation

Despite a read-only assignment forbidding full profiling-table scans, the
investigator materialized all37,044 rows and42 fields of
`fixture_compute_kernel_only/moe.csv` (2,114,942,423 bytes) with
`rows=list(csv.DictReader(open(p)))`. Its shell timeout was120s; no memory cap
was applied, no wall time or output file was retained, and the command exited0.
This was an instruction deviation. No input write was performed. The decisive
operator applicability evidence was already available in source and the small
`registry_binding/evidence/bundle_receipt.json`; the full read was unnecessary.
`evidence.json` retains the original command and reported output, not a rerun.
FP8 has16,464 blank quantize cells and2,058 positive cells; BF16 has18,522 blank
cells. Blank cells are not zero prices and do not imply missing selected costs.

## Historical verdict, corrected event interpretation

1. One 208-event wave is executed **once per actual MoE layer of a layer set**,
   not once per model group: `88` times per layer set, inside `92` layer steps.
   Layers `0,1,2,91` are dense and execute no EP wave.
2. Per layer set: `88` pricing plans, `88` emission loops, `88 x 208 = 18,304`
   events in the former uniform-FP8 projection. With three BF16 layers the
   expected EP-wave count is18,256 per layer set. The old~255.7MB size uses
   sampled FP8 output; see the current reconciliation for its limits.
3. The workload executes **1,025 layer sets**: one PREFILL set on the 6,128-token
   chunk plus 1,024 DECODE sets, the first of which replays the remaining 16
   prompt tokens. Total **90,200 waves** and expected **18,712,400 EP-wave events**
   per mode. The historical262.1GB/mode figure is a sample-based projection.
4. Persisted traces are **not sampled, capped, collapsed or rotated**. Every
   layer set x every MoE layer x16 EP lanes x12/13 applicable operators is written.
5. At the original `eece87b2` checkpoint, the projected per-mode cost is
   **86-101 ks (24.0-28.1 h)** including startup, i.e. **36-42x** the 2400 s
   cap. This is a current-implementation estimate, not a bound on any future
   output-preserving serialization repair.
6. The mandatory output path is the binding term: with pricing set to zero, the
   current emission+flush cost is `7.85 ks` of waves plus `0.84-1.29 ks` of
   startup (`8.7-9.1 ks`, 3.6-3.8x). To fit 2400 s with zero pricing, the
   per-wave output cost would have to fall from `87 ms` to `<= 17.3 ms` (or
   `<= 12.3 ms` with the slower startup); the planned cost-field change takes it
   to `80 ms`, an 8 % reduction. That is the honest headroom statement; it is
   not a proof that no future serialization repair can help.

## 1. Workload and real argv facts

From `d29-parent-position-main08/offline/command.json`, `requests.csv` and
`execution.json`:

| Fact | Value | Source |
| --- | --- | --- |
| sys_arch / mode | `pd-disaggregation`, offline (online mirrored) | argv |
| model | `stepfun-ai/Step-5-Preview`, TP8 / attn_dp2 / EP16 / PP1, 352 experts, top-8 | argv |
| request | 1 request, trace replay, `num_prefill_tokens 6144`, `num_decode_tokens 1024` | `requests.csv` |
| serving limits | `max_model_len 7168` (= 6144 + 1024), `max_tokens_in_batch 8192`, `batch_size_cap 128`, `block_size 16` | argv |
| chunked prefill | disabled | `--no-vllm_v1_scheduler_config_enable_chunked_prefill` |
| op tracing | **enabled** (`--metrics_config_enable_op_level_tracing`) | argv |
| per-layer expansion | flag absent, default `False` (`metrics_config.py:72-84`) | argv + config |
| timed out | 2400.175 s offline, 2400.152 s online; 4 recorded emitted decode tokens; iteration 4 scheduled | `execution.json` |
| trace written | 1,447,809,841 B offline; 1,517,626,194 B online | `execution.json` |

## 2. Exact composition of the sampled FP8 wave's208 events

Measured directly from the retained offline trace, one complete layer wave
(layer 60), 16 lane batches:

| Phase | COMPUTE per lane | COMM per lane | Events per lane | Events total |
| --- | --- | --- | --- | --- |
| `pre_dispatch` | 7 | 1 | 8 | 128 |
| `dispatch` | 0 | 1 | 1 | 16 |
| `routed_compute` | 1 | 0 | 1 | 16 |
| `combine` | 0 | 1 | 1 | 16 |
| `post_combine` | 1 | 1 | 2 | 32 |
| **total** | **9** | **4** | **13** | **208** |

`16 lanes x 13 operators = 208`; equivalently `144 COMPUTE + 64 COMM`. The 13
operators are:

- `pre_dispatch`: `post_attention_layernorm`, `share_expert_up_proj`,
  `share_expert_act`, `share_expert_down_proj`, `moe_gating_linear`,
  `moe_gating_routing_topk`, `moe_input_quantize` (COMPUTE) +
  `share_expert_tensor_parallel_allreduce` (COMM)
- `dispatch`: `step5_moe_dispatch` (COMM)
- `routed_compute`: `moe_grouped_gemm` (COMPUTE)
- `combine`: `step5_moe_combine` (COMM)
- `post_combine`: `step5_ffn_merge` (COMPUTE) +
  `moe_tensor_parallel_allreduce` (COMM)

An earlier draft wrote "16 lanes x 5 phases x 13 operators", which is wrong both
in arithmetic and in reading: `13` is the per-lane operator total, and the five
phases contribute `8 + 1 + 1 + 1 + 2`.

The 208 figure is independently confirmed by the retained replay artifact
(`d29_main08_residual_repro/parent-red01/result.json`: `events = 208`,
`participants = [0..15]`) and by the parent-verified "144 operator_cost
objects" in the layer-60 proof payload.

## 3. Call chain (one wave) and layer-set multiplicity

```
DecodeSyncCollectiveEvent                                  events/decode_sync_collective_event.py:37
  -> on_decode_sync_collective                             cluster_scheduler/base_cluster_scheduler.py:1345-1363
       -> handle_decode_sync_collective                    scheduler/utils/decode_collective.py
            next_layer_id = layer_id + 1                   decode_collective.py:103
            if next_layer_id < stage_layer_end:            decode_collective.py:116
              predict attention (num_layers=1, include_ffn=False)   :130-134
              DecodeSyncEvent(..., "pre_moe", next_layer_id)        :142-148
  -> on_decode_sync                                        base_cluster_scheduler.py:1317-1343
       -> enter_decode_sync                                sync_entry.py:188-207
            -> enter_layer_sync                            sync_entry.py:37-163
                 expected_lanes = scheduler._replica_dp_size (=2)   :103
                 return when len(room) < expected_lanes              :108-109
                 ready = scheduler._on_decode_ep_wave_ready          :153
  -> _on_decode_ep_wave_ready                              base_cluster_scheduler.py:1144-1158
       -> schedule_layer_wave(mode="decode", layer_id=...)  ep_wave_schedule.py:52-245
            if model_config.is_moe_layer(layer_id):         ep_wave_schedule.py:100
              _prepare_moe_ep_wave_plan(...)                :101-108  [PRICING]
              metrics_store.on_ep_wave_schedule(plan, ...)  :109-113 -> metrics_store.py:525-535
                -> record_ep_wave(store, plan, ...)         ep_wave_metrics.py:10-134
                     for record in phases.lane_records        :26   (16 EP lanes)
                       for phase in the 5-phase dict           :58
                         for kind,name,duration > 0            :91-93
                           trace_store.log_event(TraceEvent)   :112-117  [EMISSION]
            else: dense branch, one prediction per lane + DenseLayerCompleteEvent   :125-184
```

`num_layers_per_pipeline_stage = num_layers // num_pipeline_stages = 92 // 1`
(`base_execution_time_predictor.py:97-99`), so `stage_layer_end = 92`
(`decode_collective.py:89-102`). The two DP lanes meet in one `pre_moe` room and
the wave is dispatched once per layer for the whole cohort; there is no
per-DP-lane or per-pipeline-stage multiplication.

Layer set (confirmed by calling the production resolver on the checked-in
config, no trainer):

```
BaseModelConfig.create_from_name('stepfun-ai/Step-5-Preview')
  num_layers 92   is_moe True   moe_layers 88
  first [3,4,5]   last [88,89,90]   dense [0,1,2,91]
```

## 4. Token and chunk bookkeeping: how many layer sets

This is the part an earlier draft got wrong by assuming "1024 decode tokens x 88
plus 88 prefill". The production path is:

1. **Prefill target is 6,128, not 6,144.**
   `_step5_kv_transfer_prefill_target(prompt_tokens, block_size)`
   (`frontier/attention/families.py:572-577`) returns
   `max(prompt - block_size, 0)` rounded down to a block:
   `6144 - 16 = 6128`. Its comment states the intent: "Native Step replays one
   block on D and resumes at a full-block boundary." It is installed as the
   Step family's `kv_transfer_prefill_target_resolver`
   (`families.py:586`) and applied only on the measured PD path
   (`vllm_v1_kv_allocation.py:32-56`, requires `pd-disaggregation` + `nixl` +
   grouped KV + non-dummy + PP1 + no prefix cache + no spec decode — all true
   here), via `request.set_prefill_compute_target(...)`.
   Retained confirmation: the PREFILL scheduler decision schedules
   `num_tokens=6128` and leaves `token_budget=2064` (= 8192 - 6128), and the
   prefill trace rows carry `tensor_shape.input = [6128, 4096]`.
2. **The remaining 16 prompt tokens are replayed on the decode cluster as its
   first batch.** Retained confirmation: DECODE `iteration 0`
   `grouped_cache_allocation num_tokens=16`, `decision ADMISSION num_tokens=16`,
   `iteration_end batch_num_tokens=16`; the first decode layer wave carries
   `tensor_shape.input = [16, 4096]` and
   `effective_total_tokens_compute = 16`, whereas later decode waves carry
   `[2, 4096]` (2 lanes x 1 token). Simulation time for that batch is 0.116 s
   versus 0.032 s for the 1-token iterations.
3. **That replay batch commits exactly one decode token.**
   `request.py:1383-1393`: while `_num_recomputed_tokens` is not `None`, the
   callback advances the recompute counter and, on the chunk that reaches the
   last kept token, sets `_num_recomputed_tokens = None` and
   `num_tokens_processed = 1`. So the 16-token batch advances decode progress by
   1, matching the parent's record of **4 emitted tokens after 4 completed
   batches** (iterations 0-3).
4. **No output token is emitted by prefill in this mode.** The handoff-token
   convention (`request.on_disaggregated_decode_handoff`,
   `request.py:782-807`, `_num_handoff_emitted_decode_tokens = 1`) is called
   only from `handle_decode_attn_arrival` (`kv_arrival.py:61`). The
   pd-disaggregation path calls `handle_decode_arrival`
   (`base_cluster_scheduler.py:1379,1394-1402` → `kv_arrival.py:98-127`), which
   only calls `on_arrival` and `add_request`. So the 1,024 output tokens are all
   post-prefill, and the decode cluster executes 1,024 batches.

Layer-set count:

| Cluster | Batches | Tokens per batch | Layer sets | MoE waves |
| --- | --- | --- | --- | --- |
| PREFILL | 1 | 6,128 (prefill chunk 1) | 1 | 88 |
| DECODE | 1,024 | 16 (chunk 2 replay) then 1 x 1,023 | 1,024 | 90,112 |
| **total** | 1,025 | 6,144 prompt + 1,024 output | **1,025** | **90,200** |

So the total is `88 x 1025 = 90,200` waves. The earlier draft's
`1024 x 88 = 90,112` decode figure plus `88` prefill reached the same 90,200
total but mis-described which batch carries the 16 residual tokens; the volume
total is unchanged, the composition is now source-backed.

## 5. Trace persistence: no sampling, no cap

- `TraceStore` appends every `log_event` to a 5,000-event buffer and writes all
  buffered rows with one `json.dumps` per row (`trace_store.py:62-65, 90-126`).
  No sampling, drop, rate limit, rotation or per-request quota.
- The only request-level quota, `num_requests_to_trace_per_layer` (default 5),
  is consulted only by `_should_expand_layers` (`metrics_store.py:973-1010`),
  whose result only sets the `layer_id` field value (`ep_wave_metrics.py:36-38,
  115`). It cannot suppress emission: the `log_event` call at
  `ep_wave_metrics.py:112-117` is guarded only by
  `enable_op_level_tracing and store.trace_store` (`ep_wave_metrics.py:22, 96`).
- Observable consequence: all 598 rows of the 8 MiB tail and all 602 of the
  head sample carry `layer_id = -1` while carrying
  `meta.global_layer_ids`. The trace is fully per-layer in content, not in that
  column, and it is persisted for every layer set and every layer.

## 6. Byte arithmetic and closure

Bounded measurement of the retained offline trace (`op_traces.jsonl`,
read-only, first 8 MiB and last 8 MiB):

| Quantity | Value |
| --- | --- |
| tail rows / bytes | 598 / 8,353,308 |
| head rows / bytes | 603 / 8,357,995 (includes the 180 B metadata header) |
| mean row size, DECODE | 13,968.7 B (median 3,944 B; max 37,438 B) |
| complete wave, layer 59 | 16 lane batches x 13 rows, **2,906,520 B** |
| complete wave, layer 60 | 16 lane batches x 13 rows, **2,905,320 B** |
| per-wave mean | **2,905,920 B** |
| historical uniform-FP8 events per layer set | 88 x 208 = 18,304; expected mixed-group count18,256 |
| bytes per layer set | 255,720,960 B (255.7 MB) |

Derived volume:

| Term | Value |
| --- | --- |
| layer sets | 1,025 |
| waves | **90,200** |
| expected EP-wave events | **18,712,400**; old uniform208 extrapolation18,761,600 |
| bytes per mode | **262,113,984,000 B (262.1 GB)** |
| both modes | ~524.2 GB |

Closure against the retained offline file (1,447,809,841 B):
Dividing by the representative DECODE row size gives approximately 103,648
row equivalents, or 5.66 layer-set equivalents. This is a rough consistency
check, not a complete event census: prefill and decode row sizes differ, and
the residual does not establish an aggregate-event count. The tail reaches
layer 60 (58 MoE layers starting at layer 3), with possible buffered progress
beyond it. Online is approximately 5.93 layer-set equivalents.

## 7. Projected wall cost - current-implementation estimate

Inputs, each labelled by origin:

| Input | Value | Origin |
| --- | --- | --- |
| pricing, warm | 1.021 s/wave | accepted parent value on `eece87b2`; not re-measured here |
| emission, current | 0.061 s/wave | accepted baseline (`resumed_performance`) |
| emission, cost-fields proposal | 0.054 s/wave | active 8-gross proposal as described by the parent |
| flush, post trace-repair | 0.026 s/wave | measured 1.174 -> 0.076 s per 598 events (`trace_wave` replay) |
| flush, pre-repair | 0.130-0.141 s/wave | retained `parent-red01/result.json` |
| same-structure reference | 2.926 s/wave (2.735 + 0.061 + 0.133) | retained `79f1a3fd` replay |
| observed decode batch duration (tokens 1-4) | 214.7, 221.4, 220.2, 224.6 s (mean 220.2) | `scheduler.jsonl` |
| model-manager window, measured pre-repair | 1,288 s | retained main07 window 15:05:16-15:26:44 |
| startup, post loader repair | ~838 s | **extrapolated** from the accepted ~450 s scaled loader saving; not measured |
| main08 launch to first scheduler event | ~1,112.345 s | Parent reconstruction from retained file times and scheduler epochs; includes more than model-manager construction. |

Arithmetic:

| Step | Value |
| --- | --- |
| per wave, current emission | 1.021 + 0.061 + 0.026 = **1.108 s** |
| per wave, cost-fields emission | 1.021 + 0.054 + 0.026 = **1.101 s** |
| 90,200 waves, current | **99,942 s = 27.8 h** |
| 90,200 waves, cost-fields | **99,310 s = 27.6 h** |
| in-situ factor (tokens 1-4) | 220.2 / (88 x 2.926) = **0.855** |
| 90,200 waves x 1.108 x 0.855 | **85,450 s = 23.7 h** |
| **per-mode total, incl. startup** | **86.3-101.2 ks = 24.0-28.1 h** |
| ratio to the 2400 s cap | **36-42x** |

How to read this: the direct product of an accepted warm wave and the wave count
is the current-implementation estimate. It is **not** a universal lower bound
and it is **not** an impossibility proof for future output-preserving
serialization repairs, because the per-event cost is an implementation
property. It also **excludes** per-layer attention prediction (2 DP lanes x 92
layers, `decode_collective.py:130-134`), the sampler/LM-head and scheduler
overhead, so the wave term alone understates decode CPU. Environment variation
(machine, page cache, I/O of a growing 262 GB file), the fact that the phase
seconds are a sum of per-operator items, and any unmodeled work are all outside
this estimate.

### Parent correction: process and scheduler origins

The previous iteration-4 slowdown claim mixed two origins and is withdrawn.
The 1,085.041 s offset begins at the first scheduler event; the timeout begins
before simulator construction. Retained epoch timestamps are:

| Boundary | Epoch seconds |
| --- | --- |
| command.bash mtime, before launch preparation | 1792751317.662415 |
| approximate process start: execution mtime minus wall_s | 1792751317.666937 |
| first scheduler event | 1792752430.011742 |
| active DECODE iteration 4 start | 1792753515.053171 |
| execution receipt mtime, just after timeout | 1792753717.842189 |

Approximately 202.789 s remained after iteration 4 began, not 1,315 s.
The 58 persisted waves imply at most about 3.50 s per written wave over that
interval, while up to 24 waves may be buffered. Neither an anomalous slowdown
nor a write stall is established. File times are approximate boundaries,
not direct process-start telemetry. The 0.855 ratio from completed batches
remains an illustrative cross-run scaling assumption.

The tables retain their historical 838-1288 s startup sensitivity. Using the
separately reconstructed main08 pre-scheduler interval instead gives roughly
86.6-101.1 ks for the same projected wave costs. Neither range is a full run.

## 8. Output path alone: the headroom requirement

| Scenario | per wave | waves | + startup 838-1,288 s | vs 2400 s |
| --- | --- | --- | --- | --- |
| current implementation (1.108 s) | 1.108 s | 99,942 s | 100.8-101.2 ks | 42x |
| in-situ scaled (0.947 s) | 0.947 s | 85,450 s | 86.3-86.7 ks | 36x |
| **zero pricing**, emit 61 + flush 26 ms | 87 ms | 7,847 s | **8.69-9.14 ks** | **3.6-3.8x** |
| zero pricing, cost-fields emit 54 + flush 26 ms | 80 ms | 7,216 s | 8.05-8.50 ks | 3.4-3.5x |
| zero pricing, emission only (61 ms) | 61 ms | 5,502 s | 6.34-6.79 ks | 2.6-2.8x |
| zero pricing, zero emission, flush only (26 ms) | 26 ms | 2,345 s | **3.18-3.63 ks** | **1.33-1.51x** |

Required headroom, stated as a target rather than a verdict: to finish inside
2,400 s with pricing at zero, the per-wave output cost must be
`(2400 - 838) / 90,200 = 17.3 ms` (or `(2400 - 1288) / 90,200 = 12.3 ms` with
the pre-repair startup). The current output path is `87 ms`; the planned
cost-field change is `80 ms` (8 %). A future output-preserving serialization
repair would therefore need a 4.6-7x reduction in per-wave emission+flush to
bring the workload within the cap on its own. That is a quantitative statement
about this argv's gap, not a claim that no such repair exists.

## 9. Mechanism worth a future RED probe (no implementation, no allocation)

The 208 events have distinct `(lane, operator)` identities; that does not rule
out equal immutable leaf values inside them. Duplication includes both the
multiplicity and how each payload is produced:

1. The old per-operator common-metadata hypothesis is closed by the2026-10-24
   ownership inspection: the Step path builds `common` once per lane, and each
   required getter is called once per lane. Legacy-path construction must not
   be attributed to this Step path. The remaining duplicate plan dictionary
   proposal is estimated<=3ms and is not allocated. See the current reconciliation.
2. Across layers, identity is **not** interchangeable. The retained
   layer-60/61 evidence is:
   - the layer-60 replay reproduces its **own** captured payload, 208 of 208
     pairs, 0 mismatches;
   - the layer-61 replay reproduces 208 pairs of which 165 were in the bounded
     captured tail;
   - comparing layers 60 and 61 directly, all durations, native contexts and
     collective plans/costs match, while **144 `operator_cost` objects differ in
     `global_layer_id`** (`main08_wave_verification.json:30`).
   So "208/208 equal" is a replay-vs-own-capture statement, not a cross-layer
   equality. Cross-layer proof objects are not interchangeable, which is why
   this report does **not** propose sharing them.
3. Neither `record_ep_wave` nor `TraceStore.flush` has a sampling, grouping or
   layer-batching seam (`trace_store.py:90-126`).

A probe worth running **without changing output** is therefore a
count-and-byte-identity probe: assert that one layer set emits exactly
`88 x 208` events and `255.7 MB`, and that any proposed serialization path
reproduces the byte stream exactly. That is the seam section 8 quantifies.

## 10. Evidence inventory (bounded and read-only)

| Evidence | Path | Use |
| --- | --- | --- |
| real argv | `d29-parent-position-main08/offline/command.json` | tracing flags, model, parallelism, limits |
| execution receipt | `.../offline/execution.json`, `resumed_performance/main08_execution.json` | 2400 s timeouts, 4 emitted tokens, trace bytes |
| request input | `.../offline/requests.csv` | 6144 prompt / 1024 output |
| scheduler timeline | `.../offline/scheduler.jsonl` (52 records) | PREFILL 6128 decision, DECODE 16-token iteration 0, 1-token iterations, batch durations |
| wave replay | `d29_main08_residual_repro/parent-red01/result.json`, `wave-final-*/result.json` | 208 events, 16 participants, per-phase seconds |
| trace replay | `d29_main08_residual_repro/trace-baseline01`, `trace-shallow01` | flush 1.174 -> 0.076 s per 598 events |
| trace volume | `.../offline/metrics/.../op_traces.jsonl`, head 8 MiB + tail 8 MiB | 13 rows/lane/layer, 208 events and 2.905 MB/wave, 16-token first decode wave |
| layer identity | `resumed_performance/main08_wave_verification.json` | own-capture replay equality; 144 proofs differ by `global_layer_id` |
| layer set | `data/config/models/stepfun-ai__Step-5-Preview.json` + real `create_from_name` | 92 layers, 88 MoE, dense 0,1,2,91 |
| prefill target | `frontier/attention/families.py:572-577,586`; `vllm_v1_kv_allocation.py:32-56` | 6128 = 6144 - 16, rounded to block |
| replay token count | `frontier/entities/request.py:157,570-573,1387-1393` | 16-token replay batch commits 1 token |
| PDD handoff scope | `frontier/scheduler/utils/kv_arrival.py:61,98-127` | prefill emits no output token in pd-disaggregation |
| startup | `early_init_repro/report.md` | 1,288 s measured; ~450 s loader repair (scaled) |

`evidence.json` carries the machine-readable form of these numbers.

## 11. Proven vs inferred

Proven from source and retained bytes:

- One wave per MoE layer; 88 per layer set; dense `0,1,2,91` excluded.
- The sampled FP8 wave has208 events =16 lanes x13 operators, split
  128/16/16/16/32 by phase and144/64 by type; this is not every layer's count.
- 1,025 layer sets: one 6,128-token PREFILL set, then 1,024 DECODE sets whose
  first carries the 16-token replay.
- No sampling/cap in persistence; every layer set and layer is written.
- 1.448 GB offline and 1.518 GB online were already written inside 2,400 s.

Inferred (labelled):

- Expected successful-run EP count18,712,400 from applicable operators and
  positive-price admission; no P6128/D16/BF16/full-run emission census.
- Future per-wave cost from the accepted `1.021 + 0.061|0.054 + 0.026 s`.
- The in-situ `0.855` scaling from the first four tokens.
- Post-repair startup (~838 s).
- The 262.1 GB volume extrapolated from representative DECODE waves; the
  90,200 scheduled waves follow the source-bound batch/layer counts.
- The approximately 1,112.345 s pre-scheduler interval from retained file times.

Not established: an iteration-4 slowdown; per-layer attention/sampler cost at
~6,148-token context; the exact post-repair model-manager phase duration;
anything about online mode beyond its recorded timeout and bytes.

## 12. Remaining measurement boundaries (no allocation)

The wave multiplicity is source-backed. A dense0/MoE3 check alone would not
settle the remaining P6128/D16/BF16 event and byte questions; their complete
output remains unmeasured. No new probe is allocated by this report. The
historical1.1s replay estimate is superseded by later component timings, which
still do not establish full CLI performance or storage requirements.

## 13. Parent arithmetic verification

The first check failed on a transcribed3604.3912s projection. It was corrected
to3604.0312s; all five assertions and whitespace verification then passed,
exit0. This checks record arithmetic only, not a new performance measurement.
Exact command, from the primary worktree:

```sh
/data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
import json,math,subprocess
from pathlib import Path
p=Path('task_memory/task_2026-10-22_step5_position_routing/workload_multiplicity/evidence.json')
d=json.loads(p.read_text());r=d['current_reconciliation']
assert r['waves']==(1+1024)*88
assert r['expected_ep_wave_events']==(85*208+3*192)*1025==d['volume']['events']
assert math.isclose(r['zero_other_cost_wave_allowance_ms'],2400*1000/r['waves'])
assert math.isclose(r['uniform_rate_all_wave_projection_s'],r['waves']*.039956)
assert math.isclose(r['steady_fp8_rate_projection_s'],85*1023*.039956)
print('arithmetic PASS',r['expected_ep_wave_events'],r['zero_other_cost_wave_allowance_ms'])
subprocess.run(['git','diff','--check'],check=True)
print(subprocess.check_output(['git','status','--short'],text=True))
PY
```
