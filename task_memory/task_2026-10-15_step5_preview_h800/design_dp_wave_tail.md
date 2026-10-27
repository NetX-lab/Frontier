# Design: DP wave tail before a prefill (calibration item 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-27 | Created: native mechanism, Frontier probe result, the two Frontier changes (PREFILL batch queue of 2, host-paced dummy steps), data, validation and open questions for codex. |

## 1. Problem

Step G (h200_64, TP8/DP4/EP32) is at TTFT −12.9% (p95 −20.5%) against the clean native reference
`h200_64_sweep02`; step E (h200_32, DP2) is at −1.0%. The DP4 gap is in the prefill part of TTFT:
native mean about 814 ms against Frontier's 689.7 ms for every request. The native prefill time
follows a sawtooth over consecutive requests (671, 770, 831, 908, 983, 716, ... ms) because each
prefill waits for dummy steps that the previous DP wave queued on the GPU (estimated backlog at
prefill entry: mean 180 ms at DP4, about 0 at DP2). Evidence:
`test_report_2026-10-27_h200_64_ttft_wave_tail.md`.

## 2. Native mechanism (vLLM source `/data/ycfeng/tmp/d29_published_sources01/source/vllm/`)

1. Async scheduling is on for both roles. `config/vllm.py:1587-1646` turns it on by default unless
   an incompatible option is set; every node log of `h200_64_sweep02` prints "Disabling NCCL for
   DP synchronization when using async scheduling" and "Using V2 Model Runner". With it,
   `max_concurrent_batches` is `pp_size + 1 = 2` (`config/vllm.py:579-589`), which sizes the
   EngineCore batch queue (`v1/engine/core.py:215-221`).
2. `step_with_batch_queue` (`v1/engine/core.py:693-763`) returns right after it launches a batch
   while the queue has room (lines 749-754). It blocks on the oldest batch only when the queue is
   full or nothing new was scheduled.
3. `DPEngineCoreProc.run_busy_loop` (`v1/engine/core.py:2323`) runs `execute_dummy_batch()`
   (line 2358) after a step that executed nothing while the wave runs, and then calls
   `_has_global_unfinished_reqs` (line 2405), which synchronizes the engines only on step 1 of a
   wave and every `dp_sync_interval` steps (line 2408; 16 by `config/parallel.py:162`).
4. So the prefill engine reaches the step-1 synchronization before its prefill output is
   popped: it still has an unfinished request, and the wave runs to step 16. After the prefill,
   every engine runs about 15 dummy steps.
5. A dummy step samples nothing, so no host waits for its device work. Each step's
   `dispatch_cg_and_sync_dp` (`v1/worker/gpu/model_runner.py:1810`) is a CPU all-reduce across
   DP engines, so the hosts move in step. When every engine runs dummy steps, the host issues
   one about every 5.6 ms (DP4; 4.4 ms at DP2), while the device needs 32.0 ms (DP4; 24.8 ms at
   DP2) per dummy step. The device runs the queued steps in order.
6. The next prefill's first KV side-storage reset copy (`.tolist()`, item 3) waits for the
   device to finish the queued dummy steps; its remaining host work then follows (median 33.8 ms
   from entry to launch after subtracting the backlog at DP4, 26.0 ms at DP2).

When one engine runs a real batch, the others' dummy steps cannot run ahead: their hosts wait
in the next step's DP all-reduce for the engine whose host waits for its batch output.

## 3. Frontier today (probe, diagnostic only)

`/data/ycfeng/tmp/d31_diag/wave_frontier/wave_probe.py` runs Frontier (commit 02bd453a, h200_64
host options) on 16 requests with arrivals 1 s apart and logs every PREFILL pass. Every request
gives the same pattern: lane k runs the 6128-token prefill from t = k s, the other three lanes
start one dummy forward at the same time, and all of them end at k + 0.689 s; then no lane runs
anything until the next arrival.

Cause: Frontier's PREFILL role has a batch queue of 1 (`_max_concurrent_batches = _num_stages`;
an explicit capacity is accepted only for DECODE and for MONOLITHIC PP1/DP1,
`vllm_v1_engine_replica_scheduler.py:391-427`). The prefill lane therefore blocks on its batch
and applies the output before it ends its first iteration. At the step-1 synchronization
(`_end_iteration`) no lane has an unfinished request, and the wave ends after one step. A second
difference stays hidden by the first: the next pass of a dummy lane starts only when its dummy
forward ends (`batch_stage_end_event.py:165-185`, `next_pass_start`), so even a longer wave
would leave at most one dummy forward ahead of a new prefill.

## 4. Changes (off by default)

### 4.1 PREFILL batch queue of 2

Accept the explicit `max_concurrent_batches = 2` for the vLLM V1 PREFILL role under the
DECODE conditions (FCFS, PP1, no speculative decoding, no GDN) and without prefix caching,
through a new role flag
`--cluster_config_prefill_replica_scheduler_config_max_concurrent_batches` that mirrors the
existing DECODE flag (`cluster_config.py:555`, `cluster_role_config.py:261`). A PREFILL request
leaves the role when its prompt ends, so no output placeholder is needed. The existing in-flight
rule (`vllm_v1_engine_replica_scheduler.py:1136-1147`: skip a request whose prompt is fully
scheduled and in flight; schedule the next chunk while the previous chunk is in flight) applies
to PREFILL as well; today it is limited to MONOLITHIC and DECODE (line 1140). The Step model
uses grouped KV allocation (`_is_grouped_kv_allocation`), which the DECODE path already accepts
with a queue of two and the MONOLITHIC path rejects. The Step workload command (`tests/integration/step5_pd/frontier_workload.py`)
sets it, as it already does for DECODE.

Effect: the prefill lane ends its first iteration right after it launches the prefill, joins the
step-1 synchronization with an unfinished request, and the wave continues to step 16. The batch
output is held (`hold_batch_output`) and popped by the lane's next iteration, as on DECODE.

### 4.2 Host-paced dummy steps

When every lane of the DP group has no real batch in flight and no output left to pop, a lane's
next pass starts one measured dummy-step host period after it issued its dummy pass, instead of
when the dummy forward ends. The dummy forwards wait on the lane's stage in issue order (the
stage is busy until its current forward ends), which is the device order. The forward-start
agreement and the per-layer MoE rooms still pair the k-th forward of every lane. While any lane
has a real batch in flight or a held output, the current rule stays: a dummy lane's next pass
starts when its dummy forward ends, because the native hosts then move at the pace of the lane
that waits for its batch.

The step counter and the synchronizations on step 1 and every 16 steps are unchanged; they now
run at host time. A request that arrives after the wave ended starts a new wave on every lane at
its arrival. Its prefill batch waits on the stage behind the lane's queued dummy forwards and
enters the forward-start agreement when the stage starts it; the item 3 reset host time then
starts at that point, as the native `.tolist()` waits for the device.

The host period is a measured component host time, one value per role and topology, in a new
profile `dp_dummy_step_host_time.v1` with the same provenance and scope checks as
`kv_block_zeroing_host_time.v1` (`frontier/execution_time_predictor/kv_block_zeroing_host_time.py`)
and a new predictor option `dp_dummy_step_host_profile`. Without the profile Frontier behaves
as today. The value is estimated from the existing host-probe stamps (`h200_64_hostprobe01`,
`h200_32_hostprobe03`): the median interval between the launches of consecutive dummy steps of
an engine while every engine of its DP group runs dummy steps. No new GPU run is needed.

### 4.3 What is priced from where (D31.10)

- Dummy forward device time: the existing dummy forward pricing from op profiling (agreed MRv2
  descriptor, `step5_cost.py:155-185`, MoE rooms). The native dummy step spans (32.0 / 24.8 ms)
  are compared only as a diagnostic.
- Host period between dummy steps: a component host time from the probe (like item 3's reset
  host time). It is not a step time keyed by batch composition.

## 5. Validation

1. Unit tests: the PREFILL flag and its admission rules; with a queue of 2 the prefill lane's
   wave continues past step 1 and ends at step 16; host-paced dummy passes start at the measured
   period while the group has no real batch, and fall back to the forward end otherwise; a new
   prefill waits behind the queued dummy forwards and its reset host time starts at its stage
   start; no profile means no change.
2. Dummy smoke and the 77-case fidelity matrix on the default path (identical results).
3. Diagnostic probe rerun: wave length 16 steps, dummy forward spans and backlog at the next
   prefill, against the native estimates (diagnostic only).
4. Rerun G and E against the clean references. Expectation: at DP4 the prefill part of TTFT rises
   from 689.7 ms toward about 800 ms with a sawtooth, and TTFT moves from −12.9% toward the
   reference; at DP2 the backlog stays near zero and E stays within its current rows. If
   Frontier's dummy forward is more than about 27 ms at DP2, a backlog appears there that the
   native run does not show; that would point to the dummy forward pricing (item 2 operators).

## 6. Open questions for the codex review

1. Is "every lane has only dummy work" the right condition for host-paced dummy steps, or must
   the per-step DP all-reduce be modelled as a host barrier in general?
2. Does a PREFILL queue of 2 change anything else on the PREFILL role, such as when the KV
   transfer starts (batch end or output pop) or how a long prompt is chunked?
3. Should the host period live in its own profile file or as a second term of the item 3
   profile, which the same probe measures?
4. With a queue of two, a prompt longer than the token budget sends its next chunk while the
   previous chunk is in flight. Is that safe with grouped KV allocation and the KV block zeroing
   of the item 3 trigger (`_create_batch`, lines 445-470), which the MONOLITHIC path excludes?
