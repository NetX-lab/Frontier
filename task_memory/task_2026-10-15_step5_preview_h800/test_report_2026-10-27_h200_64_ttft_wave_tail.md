# Test report: h200_64 TTFT gap and the native DP wave tail before a prefill

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Section 7: Frontier probe result (the PREFILL wave ends after step 1) and the native async-scheduling batch queue that keeps the native wave running to step 16. |
| 2026-10-27 | Created: TTFT split, native prefill sawtooth, vLLM wave mechanism, wave-tail measurement from the host-probe stamps, Frontier code reading. |

## 1. Scope

Calibration item 4, step 1 (`plan.md`). Step G (h200_64, TP8/DP4/EP32, Frontier commit `02bd453a`
with the DP4 host profile) is at TTFT −12.9% (p95 −20.5%) against the clean native reference
`h200_64_sweep02`, while step E (h200_32, DP2) is at −1.0%. This report locates the DP4 TTFT gap.

Everything here is diagnostic (D31.10). The native step stamps explain a mechanism; they are not
used as step times in Frontier. The planned fix prices the dummy-step device work from op
profiling and uses only a measured host period per dummy step.

## 2. Inputs and commands

All scripts are in `/data/ycfeng/tmp/d31_diag/ttft64/` and run with `/usr/bin/python3` (3.10.6),
`timeout 60`, reading only:

- Native clean references: `/data/ycfeng/tmp/step5_d31_workload_20261027/h200_64_sweep02/stage/evidence/replay/`
  and `.../h200_32_sweep03/stage/evidence/replay/` (`client_requests.jsonl`, `proxy_requests.jsonl`).
- Native host-probe runs (step stamps of every engine): `.../h200_64_hostprobe01/stage/evidence/nodes/*/host_probe/`
  and `.../h200_32_hostprobe03/stage/evidence/nodes/*/host_probe/`.
- Frontier runs: `/data/ycfeng/tmp/d31_e05_run/h200_64_m04_host/` and `.../h200_32_m04_host/`
  (`request_metrics.csv`).
- vLLM source: `/data/ycfeng/tmp/d29_published_sources01/source/vllm/`.

| Script | Arguments | Output |
| --- | --- | --- |
| `ttft_split.py` | none (paths inside) | `ttft_split.txt` |
| `prefill_overlap.py` | stage name (`h200_64_sweep02`, `h200_32_sweep03`) | `prefill_overlap.txt` |
| `probe_index.py` | `.../h200_64_hostprobe01/stage/evidence` | `probe_index.txt` |
| `prefill_steps.py` | none (uses `probe_index.txt`) | `prefill_steps_0_4.txt` |
| `wave_tail.py` | `.../h200_64_hostprobe01/stage/evidence`, then `.../h200_32_hostprobe03/stage/evidence` | `wave_tail.txt` |

## 3. TTFT split

Native: dispatch to the proxy's prefill response, and prefill response to the first client
token (128 of 128 formal requests joined per repeat). Frontier: `ttft_prefill_only` and
`ttft_kv_transfer` (`ttft_decode_first_token` is 0 for every request).

| Case | Part | Native mean (3 repeats) | Native min | Frontier mean | Frontier min / max |
| --- | --- | --- | --- | --- | --- |
| h200_64 | prefill | 816.1 / 812.4 / 813.9 | 615.5–619.6 | 689.7 | 689.7 / 689.7 |
| h200_64 | handoff | 369.4 / 363.4 / 364.5 | 214.1–225.9 | 337.7 | — |
| h200_32 | prefill | 588.3 / 591.8 / 591.8 | 578.9–580.9 | 597.8 | 597.8 / 597.8 |
| h200_32 | handoff | 319.1 / 323.6 / 322.0 | 179.0–195.8 | 305.0 | — |

Frontier's handoff is `client_ttft − ttft_prefill_only` (1027.4 − 689.7 and 902.8 − 597.8).
Frontier's `request_waiting_time_prefill` is 0 for every request in both cases.

Observation: the DP4 TTFT gap is mostly in the prefill part (native mean about 814 ms against
689.7 ms, −124 ms); the handoff contributes about −28 ms. At DP2 both parts are within 10 ms
and 17 ms.

## 4. Native prefill time follows a sawtooth at DP4

`prefill_overlap.txt`, h200_64_sweep02 repeat 1, the first 24 formal requests (arrivals 1.000 s
apart, 125 of 128 requests overlap no other prefill):

```text
671 770 831 908 983 716 661 735 797 855 903 983 774 627 703 763 834 902 956 900 633 701 768 828 (ms)
```

The prefill time rises by about 60–100 ms per request, then drops back to about 630–720 ms. At
DP2 (h200_32_sweep03) it is flat: median 586.8 ms, range 579–628 ms in repeat 1. Overlap with
another prefill does not explain the DP4 rise: the requests that overlap no other prefill have
about the same median as all requests (824.9 ms against 828.1 ms in repeat 1).

## 5. Native mechanism (vLLM source)

- `v1/engine/core.py:2323` `DPEngineCoreProc.run_busy_loop`: when an engine has no batch to run
  while the DP wave is running (`engines_running`), it runs `execute_dummy_batch()` (line 2358),
  so every engine joins the wave's MoE collectives.
- `v1/engine/core.py:2405` `_has_global_unfinished_reqs`: the engines agree on whether any engine
  still has unfinished requests only on step 1 of a wave and every `dp_sync_interval` steps
  (line 2408); `config/parallel.py:162` sets `dp_sync_interval` to 16. A wave therefore lasts at
  least until the next multiple of 16 steps after its last real step.
- A dummy step has no sampled output, so the host does not wait for its device work before the
  next step. The host issues dummy steps every few milliseconds while each needs about 30 ms on
  the device, so the device falls behind the host during the wave tail.
- The next real prefill step blocks on its first device-to-host copy (the Step-5 KV side-storage
  reset `.tolist()`, calibration item 3) until the device has finished the queued dummy steps.

## 6. Measurement of the wave tail (host-probe stamps)

`wave_tail.py` reads the step stamps of TP rank 0 of each DP engine (enter, launch, exit, CUDA
event span). For each real prefill step (at least 1000 tokens) it reports the time from step entry
to forward launch and an estimate of the device backlog at entry: the stream is assumed to run the
engine's last 40 launched steps back to back from their launch times.

| Quantity | h200_64 (DP4, hostprobe01) | h200_32 (DP2, hostprobe03) |
| --- | --- | --- |
| Real prefill steps | 457 | 441 |
| Entry to launch, p10 / p50 / p90 / mean (ms) | 47.3 / 227.0 / 376.1 / 236.5 | 22.1 / 26.2 / 37.7 / 45.3 |
| Estimated device backlog at entry, p10 / p50 / p90 / mean (ms) | 7.6 / 188.4 / 337.5 / 180.0 | 0.0 / 0.0 / 0.0 / 2.9 |
| Median of (entry to launch − backlog) (ms) | 33.8 | 26.0 |
| Dummy steps since the engine's previous real step, p50 | 47 | 31 |
| Dummy step device span, p10 / p50 (ms) | 28.9 / 32.0 | 23.7 / 24.8 |
| Dummy step launch period, p10 / p50 (ms) | 5.0 / 5.6 | 4.0 / 4.4 |
| Backlog histogram (50 ms bins, count) | 0:92, 50:44, 100:62, 150:51, 200:55, 250:65, 300:61, 350:20, 400:6, 450:1 | 0:429, 50:5, 100:5, 150:2 |

The median difference between entry-to-launch and the estimated backlog (33.8 ms and 26.0 ms)
is the ordinary prefill preparation time, so the backlog estimate accounts for the extra delay.

The dummy-step count since an engine's previous real step (47 at DP4, 31 at DP2) is about
16 dummy steps for each wave started by another engine (3 × 16 − 1 and 2 × 16 − 1). The count
stops at the engine's own empty scheduler steps after its prefill, so its own wave tail is not
included. This agrees with one wave of about 16 steps per request.

Example (`prefill_steps_0_4.txt`, h200_64_hostprobe01, formal request r000 of repeat 1, times in
ms after the proxy arrival): dp2 runs the 6128-token step (enter 19.3, launch 61.4, host exit
575.0, span 596.2). The other three engines run a concurrent dummy step whose span is about 638 ms
because its collectives wait for dp2. After the prefill, from about 684 ms, every engine launches a
dummy step about every 5 ms (DP all-reduce wait about 2 ms) with a device span of 29–38 ms. Ten
such steps add about 320 ms of device work in about 55 ms of host time. The next request (r001)
took 781.9 ms.

Inference: at DP4, one 1 s arrival period holds a prefill step of about 620 ms plus about 15
dummy steps of about 32 ms (about 480 ms), more than 1 s of device work. The backlog therefore
grows from request to request until a wave ends early enough for the device to catch up, which
produces the sawtooth of section 4. At DP2, about 590 ms plus 15 × 24.8 ms (about 370 ms) fits
within 1 s, so no backlog forms. Backlog mean (180 ms) plus the native solo prefill (about 625 ms)
is about 805 ms, close to the native mean of 814 ms.

## 7. Frontier side

Code reading (`frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`
145–190, `on_schedule`, `_end_iteration`, `_issue_dummy_pass`, `on_dummy_forward_end`,
`next_pass_start`; `frontier/scheduler/utils/dp_dummy_forward.py`;
`frontier/events/batch_stage_end_event.py` 120–185):

- The PREFILL role runs DP dummy passes: it has an engine batch queue, the model is MoE and
  `attn_dp` is 4. The wave sync follows the native rule (step 1 and every 16 steps;
  `VLLM_NATIVE_DP_FINISH_SYNC`).
- The dummy forward is priced layer by layer through `predict_stage_execution_time(...,
  include_ffn=False)` and enters the MoE synchronization room of each layer.
- The engine issues its next pass only after the dummy forward's last stage ends
  (`on_dummy_forward_end` → `ReplicaScheduleEvent(next_pass_start(...))`), so the host never runs
  ahead of the device.
- The PREFILL role has a batch queue of one step (`_max_concurrent_batches = _num_stages`; an
  explicit capacity is accepted only for DECODE and for MONOLITHIC PP1/DP1, lines 391–427). The
  prefill lane therefore takes its batch output before it ends its first iteration.

Diagnostic Frontier run (`wave_probe.py`, Frontier at `02bd453a` in
`/data/ycfeng/Frontier/.worktrees/step5-d31-native-02bd453`, 16 requests with 6144 prompt and
2 decode tokens, arrivals 1 s apart, h200_64 host options; output
`/data/ycfeng/tmp/d31_diag/wave_frontier/h64host_wave.jsonl`, summary `probe_summary.txt` in the
same directory). All 16 requests give the same pattern:

| Event (times after the arrival) | Frontier |
| --- | --- |
| Lane k runs the 6128-token prefill | starts at 0 |
| The other three lanes each start one dummy forward | at 0 |
| All dummy forwards end, with `on_dummy_forward_end` returning "last pass of the wave" | at 689.1 ms |
| Further passes on any lane before the next arrival | none |

So Frontier's PREFILL wave ends at the step-1 synchronization: when the lanes compare their
unfinished requests, the prefill lane has already applied its output and has none. No dummy
tail runs and no device backlog forms before the next prefill. The dummy forward cost alone is
not visible in this run, because each dummy forward waits in the MoE rooms for the concurrent
prefill and ends with it.

Why the native wave continues: async scheduling is on for every node of `h200_64_sweep02`
(each `server.log` prints "Disabling NCCL for DP synchronization when using async scheduling"
and "Using V2 Model Runner"; extract `native_async_flags.txt` in the same directory). With async
scheduling, `max_concurrent_batches` is `pp_size + 1 = 2` (`config/vllm.py:579-589`), which sizes
the EngineCore batch queue (`v1/engine/core.py:215-221`). `step_with_batch_queue`
(`v1/engine/core.py:693-763`) returns right after it launches the prefill while the queue has
room (lines 749-754), so the prefill engine reaches the step-1 synchronization with an
unfinished request and the wave runs to step 16.

Consequence: two Frontier changes are needed to model the native wave tail: a PREFILL batch
queue of two steps, and dummy steps that the host issues at its own pace (about 5.6 ms apart at
DP4) while the device runs them in order. Design: `design_dp_wave_tail.md`.

## 8. Limits

- The device backlog is an estimate from host stamps and CUDA event spans; the probe does not
  record the device start of each step.
- The probe run (`h200_64_hostprobe01`) is a different run from the reference (`h200_64_sweep02`).
  It matches the reference's sawtooth only in shape; its timing perturbation is not bounded
  (`test_report_2026-10-27_host_probe_runs.md` section 5).
- Two smaller DP4 parts remain unexplained: Frontier's solo prefill is about 60 ms above the
  native minimum (689.7 against about 625 ms), and the handoff is about 28 ms low.
