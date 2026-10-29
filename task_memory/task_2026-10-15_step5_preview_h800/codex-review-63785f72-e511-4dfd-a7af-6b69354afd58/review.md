# DP wave tail design review

## 1. Review context

**Request:** Independently review calibration item 4, `design_dp_wave_tail.md`, against the existing Frontier implementation, the supplied native vLLM source, and diagnostic artifacts. This is a design review; no implementation of the proposed changes exists.

**Working root:** `/data/ycfeng/Frontier/.worktrees/step5-preview`.

**Resolved revision:** Branch `feat/step5-preview`, HEAD `8127b87f5fe14b8ea31194368072dd018378fa17`, matching the requested `8127b87f`. The reviewed design, current task records, Frontier sources, and named workload command have no local differences from HEAD. No base-to-head patch was requested or inferred.

**Reviewer context:** Caller-supplied model `gpt-6.1-sol-sys-paygo`, reasoning effort `xhigh`. Review date: October 27, 2026.

Path notation below:

| Prefix | Absolute root |
| --- | --- |
| `R/` | `/data/ycfeng/Frontier/.worktrees/step5-preview/` |
| `T/` | `R/task_memory/task_2026-10-15_step5_preview_h800/` |
| `N/` | `/data/ycfeng/tmp/d29_published_sources01/source/vllm/` |
| `A/` | `/data/ycfeng/tmp/d31_diag/` |
| `S/` | `/data/ycfeng/tmp/codex-review/63785f72-e511-4dfd-a7af-6b69354afd58/` |

The primary reviewed document is [design_dp_wave_tail.md](/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design_dp_wave_tail.md:19), particularly sections 2, 4.1–4.3, and 5. Relevant evidence includes `T/test_report_2026-10-27_h200_64_ttft_wave_tail.md`, `T/test_report_2026-10-27_host_probe_runs.md`, and the supplied `A/wave_frontier/` and `A/ttft64/` artifacts. The review followed the named scheduler, stage-event, allocation, predictor, configuration, and workload callers, plus native queue, dispatch, output, and NIXL export paths.

Applicable root `AGENTS.md` instructions were inspected. No closer applicable AGENTS.md or CLAUDE.md was found in the relevant source and task paths. Material project rules are:

- `R/AGENTS.md:796–801,835–839`: integrate with existing owners, preserve unchanged behavior and fidelity, and deliver regression coverage of at least 50 scenarios.
- `T/requirements.md:117–121`: frozen 6144/1024 workload, 128 requests at QPS 1, 16 warmups, three repetitions, and the final 10% metric gate.
- `T/requirements.md:135–137`: profiling admission requires within-run coefficient of variation at most 5%.
- `T/requirements.md:161–172`: D31.10 prohibits measured whole-step lookup tables in predictions. Operator device costs and separately measured component host costs remain distinct.
- `T/requirements.md:188–200,208–237,349–367`: preserve MRv2 first-prediction descriptors, the P6128/D16 handoff, component ownership, and Step finish synchronization on step 1 and each 16 steps.
- `T/plan.md:332–397`: item 3 host-time admission and item 4 wave-tail work. Tickets 17 and 19 require causal handoff, held-output, dummy participation, and once-only cost ownership (`T/tickets/17-single-request-pd-handoff.md:58–68`; `T/tickets/19-graph-host-scheduling.md:58–67`).

**Exclusions:** Unrelated unstaged changes under `task_memory/task_2026-10-22_step5_position_routing/`, untracked `.cortexkit/.gitignore`, `.pi/subagents.json`, and `tests/integration/step5_pd/attention_census.py` were excluded. This is not a complete repository audit or a new native qualification campaign.

All inspection was read-only. Temporary checks and pytest files were confined to `S/`; source files, task records, configurations, and Git state were left untouched. No software installation, GPU submission, publication, or external messaging occurred.

## 2. Assessment

**Verdict: `changes_required`.** The native queue explanation is substantially supported, and the proposed separation of operator device pricing from component host timing respects D31.10 in principle. The design nevertheless leaves incompatible scheduler state unchanged, uses an incorrect proxy for native host blocking, and promises grouped PREFILL chunk overlap that the allocator explicitly rejects. Its proposed DP2 measurement cannot yet support an admitted cadence profile.

The most consequential risk is obtaining a plausible sawtooth by moving timers while losing the native ordering between EngineCore iterations, worker CPU dispatch, queued GPU work, and output visibility. Correct wave length alone does not establish correct TTFT.

| Requested question or requirement | Assessment | Evidence and implication |
| --- | --- | --- |
| Native async/V2 queue and step-1 unfinished state | **Satisfied for the isolated canonical prefill mechanism.** | `N/config/vllm.py:579–589` gives PP1 capacity 2. `N/v1/engine/core.py:746–754` returns before popping the result; `763–780` later waits and updates output. Finish consensus is on step 1 and multiples of 16 (`2405–2422`). With the request still running at step 1, the wave cannot stop there. It can extend beyond 16 if work remains or new work is processed. |
| Native dummy host/device overlap and real-work synchronization | **Partly supported; the proposed fallback violates the mechanism.** | `_dummy_run` executes without the sampler output wait (`N/v1/worker/gpu/model_runner.py:821–972`). Dispatch uses a CPU tensor and CPU DP group all-reduce (`N/v1/worker/gpu/dp_utils.py:65–77`). This blocks a later dispatch when a peer has not reached it; it does not require each current dummy GPU forward to finish. See F002. The 5.6/4.4 ms cadence and 32.0/24.8 ms spans are diagnostic observations. |
| PREFILL queue capacity 2 | **Capacity choice supported; full claimed fidelity violated.** | Configuration can reasonably mirror the existing DECODE role override. Omitting output placeholders is appropriate only while PREFILL eligibility stops at its P compute target and cannot schedule phantom decode work. The separate grouped allocator blocks active-request chunk overlap: F003. |
| PREFILL output and NIXL handoff | **Output-pop boundary supported; timing under the new host model unverified.** | Frontier creates transfer-start events inside `_apply_prefill_output`, after applying/popping output (`R/frontier/events/cluster_batch_end_event.py:534–619`). Native exports transfer parameters through `update_from_output` → `_free_request` → connector finish, after the queue pop. Current `hold_batch_output` only holds at finish consensus; it is not a general queue of completed futures. Preserve the pop boundary and define EngineCore visibility under the revised timing model. |
| Host-paced predicate, pass identity, finish-sync release, and room pairing | **Violated / incomplete design.** | The existing state represents one dummy pass, not multiple queued passes: F001. Stage agreement is a device-admission operation, not the earlier worker CPU dispatch barrier: F002. FIFO can preserve device order only after equal collective sequences have been established. In a mixed group, the k-th dummy of one lane does not necessarily pair with the k-th dummy of another; pairing must use the ordinal of all executed forwards, including real work. |
| Profile placement | **Separate profile recommended; admission unverified.** | A separate dummy-step profile and option fit the existing predictor-option pattern better than adding another trigger to the reset profile. The reset getter intentionally returns zero without `cache_block_zeroing`. Reuse scope-validation helpers, but keep distinct consumers and input bindings. See F004. |
| D31.10 | **Satisfied by the stated pricing architecture; measurement composition remains to be resolved.** | Section 4.3 retains operator-based dummy device pricing. A scalar host service/cadence term is not a batch-composition whole-step table. Do not include measured GPU backlog in that scalar or add a cadence containing peer waits on top of a separately modeled peer barrier. |
| DP4 TTFT recovery, DP2 preservation, and DECODE waves | **Unverified.** | No candidate exists. Section 5 proposes default-path regression, but the new path also needs mixed real/dummy cohorts, arrival during GPU drain, and DECODE/NIXL ownership checks. Existing tests exercise the current single-pass semantics. |
| Default fidelity and scenario gate | **Planned, not demonstrated for this change.** | The proposed 77-case matrix exceeds the project minimum, but it has not run for an implementation of item 4. The review ran only the focused existing suite described below. |

### Handoff and profile conclusions

Frontier's KV transfer follows **output application/pop**, which can coincide with batch end or occur later. `ClusterBatchEndEvent.handle_event` holds a PREFILL callback when `hold_batch_output` accepts it (`248–267`); `_apply_prefill_output` then completes the batch and creates `KVCacheTransferStartEvent` at its supplied pop time (`534–619`).

Native `step_with_batch_queue` pops the oldest future and calls `update_from_output` (`N/v1/engine/core.py:763–780`). A stopped request reaches `_free_request` and `_connector_finished` (`N/v1/core/sched/scheduler.py:2325–2332,2816–2825,3153–3158`). NIXL's finish hook retains blocks and returns the remote transfer parameters (`N/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:248–289`). Therefore moving the Frontier handoff to raw device completion would be the wrong correction. Preserve export/pop, target reservation, READ, readiness, and source-retention ownership.

A separate `dp_dummy_step_host_time.v1` profile fits the current plumbing. Relevant extension points are `R/frontier/config/execution_time_predictor_config.py:39–46`, the loader/getter pattern in `R/frontier/execution_time_predictor/base_execution_time_predictor.py:93–109,132–136`, and the input-binding list in `R/frontier/execution_time_predictor/shared_prediction_model_manager.py:147–152`. The new file must join that binding list. Extending the reset profile would couple two different triggers and admission histories; sharing validation utilities does not require sharing the profile schema.

## 3. Findings

### F001 — P1 high — Queued dummy passes cannot reuse the current single-pass completion and wakeup state

**Confidence:** High.

**Location:** [Design section 4.2, lines 88–101](/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design_dp_wave_tail.md:88), especially the instruction to queue multiple forwards while leaving iteration/synchronization logic unchanged.

**Violated contract:** Multiple host-issued dummy passes must retain distinct device-completion ownership, count each EngineCore iteration once, and resume each blocked operation once. Ticket 19 requires causal queues, launch/drain behavior, and held outputs (`T/tickets/19-graph-host-scheduling.md:60–66`).

**Reachable trigger:** An all-dummy DP4 tail has a proposed host interval of about 5.6 ms while each forward costs about 32 ms. Several passes must consequently exist before the first device completion.

**Expected versus actual:** The design expects pass k+1 to queue while pass k executes. With the present `on_schedule` guard retained, the early timer produces no work:

```python
if (self._blocking_batch_id is not None
    or self._dummy_forwards_in_flight
    or self._waits_at_dp_sync):
    return []
```

This is `R/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:562–567`. Bypassing that guard alone is also unsafe: `_issue_dummy_pass` assigns `_dummy_forwards_in_flight = _num_stages`, `_dummy_pass_start = time`, and `_iteration_open = True` for every issue (`723–742`). `on_dummy_forward_end` receives no batch/pass identity and decrements the shared counter (`744–751`). `next_pass_start` reads the one shared start time (`753–765`).

**Reproduction:** `S/dp_wave_method_checks.py` invoked the exact existing methods on a PP1 lane. A second direct issue represents the state reached if the early-issue guard is bypassed. Observed sequence:

| Action | Shared outstanding count | Completion result |
| --- | ---: | --- |
| Issue pass 1 | 1 | — |
| Attempt `on_schedule(0.0056)` | 1 | No batches |
| Directly issue pass 2 | 1 | Previous count overwritten |
| Complete pass 1 | 0 | `True`: engine is treated as released |
| Complete pass 2 | −1 | `False` |

This is a method-level counterexample to the proposed reuse, not an observed normal-path negative count or a test of a candidate implementation.

**Causal impact:** `R/frontier/events/batch_stage_end_event.py:163–185` currently creates the next engine schedule event from the device callback. A new host timer would introduce a second owner of that continuation. Without redesign, it either cannot queue the tail or permits old completions to release/count newer work. A negative counter is truthy in the scheduling guard and can leave later arrivals blocked. Old completion events can also restart iteration processing after a host wave has ended.

The same separation matters for finish consensus: `_end_iteration` stores one waiting state and one unfinished snapshot (`637–665`). Native computes `local_unfinished_reqs` before the synchronous dummy call and later submits that snapshot to finish consensus (`N/v1/engine/core.py:2348–2368`); an arrival during a blocked dummy call must not silently change that iteration's snapshot.

**Focused correction:** Specify separate host-iteration and device-outstanding state. Give each dummy pass an identity, retain completion ownership per pass, and make the host timer the sole continuation owner for the paced path. Device callbacks should drain their own pass without recounting the host iteration. Preserve the step-1/16 policy, but define snapshot timing, wave identity, stale timer/completion handling, and finish-sync release ownership. Host idleness must permit a new wave while previously issued GPU dummies drain.

**Verification of correction:** Exercise at least three pending PP1 dummy passes, device completion after host step 16, a new arrival during drain, arrival between unfinished snapshot and consensus, and stale timer/completion delivery. Assert nonnegative outstanding counts, exactly one host iteration per pass, equal collective sequence counts, one wave termination, and successful completion of the next real request.

### F002 — P1 high — Waiting for dummy device completion in mixed groups does not model the native CPU dispatch barrier

**Confidence:** High for the source-level mismatch; medium for its exact TTFT contribution.

**Location:** [Design sections 2 and 4.2, lines 45–46 and 88–101](/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design_dp_wave_tail.md:88), with the corresponding validation rule at `121–125`.

**Violated contract:** Preserve native asynchronous queue and host/launch boundaries. An in-flight real batch is not equivalent to a blocked host, and a worker CPU collective is not equivalent to device completion.

**Reachable trigger:** The canonical wave's first cohort contains one real PREFILL and three dummy forwards. The real EngineCore can return before its output; dummy worker hosts can return before their GPU forwards finish and then wait in the next dispatch collective. The same issue arises when queue capacity 2 admits another real chunk or request.

**Expected versus actual:** Native calls `dispatch_cg_and_sync_dp` before preparing/launching the forward (`N/v1/worker/gpu/model_runner.py:1810–1838`). Its agreement uses a CPU tensor and `get_dp_group().cpu_group` (`N/v1/worker/gpu/dp_utils.py:65–77`). This can agree a future forward while older GPU work remains queued. The real queue returns early while room remains (`N/v1/engine/core.py:746–754`). The design instead delays an idle lane's next pass until its current dummy GPU forward ends whenever any real batch or held output exists.

Existing forward-start agreement cannot supply the missing host barrier. `ReplicaStageScheduleEvent` first pops/adopts an available device-stage owner, then calls `hold_forward_start` (`R/frontier/events/replica_stage_schedule_event.py:81–125`). `StageExecutionContext.arrive_forward_start` requires an active owner and allows one ticket per lane (`R/frontier/scheduler/replica_stage_scheduler/stage_execution_context.py:86–115`); its group resets only after owners release (`430–435`). It therefore cannot agree host dispatch k+1 while device group k still owns the stage.

**Artifact evidence:** `A/ttft64/prefill_steps_0_4.txt:3–25` records the first formal request:

- dp0's first dummy launches at 33.6 ms and its host exits at 548.7 ms; its reported CUDA span is 638.6 ms.
- dp0 enters its next dummy at 551.9 ms and spends 129.1 ms in DP reduce before launching at 682.4 ms.
- dp1 and dp3 show the same pattern (`14–15,40–41`).

The host has entered the next dispatch well before the prior forward's estimated GPU finish. The artifact does not record exact GPU start times, so launch-plus-span is only an estimate; the source independently establishes the asynchronous distinction.

**Practical impact:** The fallback can delay idle-lane finish-consensus participation and the subsequent tail. Conversely, removing only the fallback would allow host timers to advance without accounting for a peer's reset or dispatch wait. Merely preserving device FIFO and MoE rooms cannot repair either host timeline. A real request's output/pop timing also needs an explicit EngineCore queue state; `hold_batch_output` currently holds only at finish consensus (`R/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:774–786`).

**Focused correction:** Model worker host readiness and per-dispatch CPU agreement separately from GPU-stage readiness. Also distinguish worker host state from EngineCore queue/pop state: a nonblocking real RPC and a synchronous dummy RPC have different effects on EngineCore progress. Establish each forward's shared ordinal/descriptor at the appropriate host agreement, retain GPU FIFO, and bind the agreed descriptor before the first predictor call. Existing stage and MoE owners can remain the device execution mechanism.

Keep reset waiting and item 3 host work under one owner. Native resets occur before dispatch (`N/v1/worker/gpu/model_runner.py:1271–1279,1757–1764`). The item 3 estimator already excludes excess peer wait (`R/tests/integration/step5_pd/kv_zeroing_host_time.py:6–24`). A new barrier must not re-charge that wait or apply the dummy cadence as an additional reset-step host term.

**Verification of correction:** Use a controlled mixed cohort where dummy host completion precedes device completion, a second real batch admitted with room in the queue, and a reset that stalls a subsequent dispatch. Assert native-order host return, next-dispatch join/release, finish-consensus snapshot, GPU execution, and output pop. Verify that all-forward ordinals pair consistently and reset/local host/peer-wait time is charged once. Replace the proposed mixed-group test that asserts unconditional fallback to device end.

### F003 — P2 medium — Grouped KV allocation rejects the overlapping PREFILL chunks promised by queue capacity 2

**Confidence:** High.

**Location:** [Design section 4.1, lines 74–79](/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design_dp_wave_tail.md:74), and open question 4 at `143–145`.

**Violated contract:** The proposed PREFILL path must schedule the next prompt chunk while its previous chunk is in flight, without artificial memory pressure, stale request invalidation, or incorrect reset/block ownership.

**Reachable trigger:** Enable the proposed PREFILL capacity 2 on a grouped Step cache and admit a prompt whose P compute target exceeds the per-batch token budget. The first chunk leaves the request active while its optimistic scheduler frontier makes the next chunk eligible.

**Expected versus actual:** Extending the engine's role membership at `R/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:1136–1150` does not remove two independent allocation guards:

```python
if not self._uses_output_placeholders and self._is_request_active_in_batch(request):
    return False
```

`R/frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:591–595` rejects the allocation. The analogous guard at `712–719` raises `RuntimeError` if allocation is attempted directly. The DECODE path avoids these guards because its capacity-2 implementation enables output placeholders; the design explicitly leaves them off for PREFILL.

The running-request caller routes rejection through `_try_allocate_with_preemption` (`R/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:1225–1229`). Its allocation failure is treated as memory pressure: with preemption enabled it selects a victim and preempts it (`R/frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:1069–1098`). PREFILL preemption advances the runtime epoch, removes active membership, and frees allocation (`935–955`), so this rejection is not merely a harmless queue wait.

**Reproduction:** The scratch method check used the existing grouped-cache unit fixture with capacity 2 and a 192-token PREFILL request. A first 64-token allocation succeeded. With that chunk marked active and scheduler frontier 64, the next 64-token allocation returned `False` despite 372 free blocks. Direct allocation raised the documented one-in-flight error. Running the exact preemption loop with a recorded victim callback selected the request itself and returned `False`. The check did not execute the full preemption mutation or a simulator run.

**Practical impact:** With preemption disabled, the advertised chunk overlap is absent. With it enabled, a supported long prompt can induce artificial preemption and invalidate its in-flight work. This also changes the newly allocated/reset block sequence and timing, undermining the intended native chunk model.

**Focused correction:** Explicitly extend the grouped allocator's active-request contract for PREFILL chunks; changing the engine role check alone is insufficient. Allocate from the optimistic scheduler frontier while retiring sliding-window pages only from safe settled progress (`vllm_v1_kv_allocation.py:603–611`). Preserve per-group zero-required IDs and their batch drain (`725–743`; engine `_create_batch`, `445–470`), active counts, and final-output-only transfer. If this is deferred, exclude grouped PREFILL chunk overlap from the opt-in and narrow the design's fidelity claim.

**Verification of correction:** Run a real scheduler/event scenario with two chunks of the same grouped PREFILL request outstanding. Assert correct context frontiers, no false preemption or allocation exception, no early page reuse, each reset sequence consumed once, and exactly one KV handoff after the final output pop. Include a sliding-window/page boundary and a genuine memory-pressure case.

### F004 — P2 medium — The proposed dummy cadence profile lacks the admission evidence claimed by the design

**Confidence:** High.

**Location:** [Design section 4.2, lines 103–109](/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design_dp_wave_tail.md:103), especially the assertion that the cited existing probes suffice and no new GPU run is needed.

**Violated contract:** A profile used for predictive calibration must have applicable measurement/admission evidence. The design also promises the same provenance and scope checks as item 3. D31.5 applies to admitted profile rows (`T/requirements.md:135–137`).

**Reachable trigger:** Extract a DP2 scalar from `h200_32_hostprobe03`, enable the proposed profile, and use E remaining close to its current rows as predictive validation.

**Expected versus actual:** The cited DP2 probe is expressly unaccepted. `T/test_report_2026-10-27_host_probe_runs.md:120–135` records TPOT shift 0.357 ms against a clean repeat range of 0.207 ms, withdraws the earlier bound on probe effects, and retains `PERTURBED`. It states that acceptance requires probe on/off evidence within one job. This limitation includes stamps on non-resetting steps and the DP synchronization; extracting a different host term from those stamps does not remove it. `T/plan.md:350–355` likewise calls DP2-profile E/G runs exploratory.

The existing wave estimator also does not implement the design's all-engine condition. `A/ttft64/wave_tail.py:8–12,25–30` uses TP rank 0 and accepts any adjacent dummy rows of the same lane. It does not join the DP dispatch ordinals to prove an uninterrupted all-dummy group, exclude idle boundaries/finish waits, or account for the slowest TP rank. Its diagnostic output demonstrates the mixture: DP4 launch intervals have p50 5.6 ms, p90 279.7 ms, and mean 97.3 ms (`A/ttft64/wave_tail.txt`, first case); DP2 p90 is 29.8 ms with mean 97.3 ms. Those numbers do not invalidate the typical cadence, but they are not an admitted component estimate.

**Practical impact:** A small cadence error shifts how many dummy forwards remain at the next arrival, affecting the DP4 sawtooth and DP2's limited backlog margin. If F002 is implemented with an explicit peer barrier, treating a measured launch interval containing peer waits as additive local service can also count waiting twice.

**Focused correction:** Keep these medians diagnostic or explicitly exploratory until a new estimator establishes complete, matched all-dummy dispatch cohorts in the applicable windows, handles TP skew and trailing loss, reports variation on the defined measurement unit, and defines whether its result is local host service or a group cadence. Resolve probe effects for DP2 before formal admission; the design should not assert that no further native measurement is needed. DP4's item 3 acceptance (`T/test_report_2026-10-27_host_probe_runs.md:209–212`) is useful evidence, but does not itself validate the new estimator or dummy interval.

Use a distinct profile schema and option. Bind it into predictor input identity and keep its admission status separate from the reset profile. This correction does not require replacing operator-priced device costs with measured spans.

**Verification of correction:** Test the estimator with a mixed real/dummy cohort, a missing participant, a late TP rank, an idle gap, a finish-consensus boundary, and a clean all-dummy cohort. Confirm the intended exclusions and once-only barrier composition. Produce explicit DP2/DP4 cadence admission reports; resolve DP2 perturbation evidence before using its result to claim predictive accuracy.

## 4. Verification

Environment: local CPU host; existing interpreter `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python`. No dependency changes were made. Bytecode, pytest cache, and automatic third-party plugin loading were disabled. Temporary pytest paths were redirected to the supplied scratch directory.

| Check | Expected | Observed | Status |
| --- | --- | --- | --- |
| Revision and scope inspection | Requested branch/HEAD; distinguish unrelated changes | `feat/step5-preview`, full HEAD matching `8127b87f`; no local diff in reviewed task/source/workload paths | PASS |
| Focused existing unit suite | Existing contracts execute; identify baseline failures | 143 passed, 1 failed in 14.19 s | FAIL |
| Existing-method dummy overlap counterexample | Confirm whether current state can hold two passes | Early scheduling blocked; direct second issue overwrote count; completions produced 0 then −1 | PASS — counterexample reproduced |
| Existing-method grouped PREFILL overlap check | Confirm active-request allocation behavior | First chunk admitted; second rejected with 372 free blocks; direct allocation error; preemption loop selected self | PASS — counterexample reproduced |
| Supplied Frontier probe recount | Match reported short-run structure | 112 records: 16 real batches, 48 dummy starts, 48 dummy ends; end offset 0.689135 s | PASS |
| Native queue, dispatch, and NIXL export inspection | Establish causal source boundaries | Early queue return, CPU DP dispatch, and pop-triggered transfer export confirmed | PASS — source inspection only |
| New paced-path integration, default 77-case matrix, dummy smoke, G/E accuracy comparison | Validate implemented item 4 | No implementation available; not executed by this review | NOT_RUN |
| Native probe on/off or new device-start measurements | Resolve cadence/backlog admission | No GPU or external job actions performed | NOT_RUN |

Exact validation commands, run from `R/`:

```bash
git rev-parse HEAD
git branch --show-current
git diff --stat HEAD -- task_memory/task_2026-10-15_step5_preview_h800 frontier tests/integration/step5_pd/frontier_workload.py
git status --porcelain=v1 --untracked-files=all
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview:/data/ycfeng/Frontier/.worktrees/step5-preview/tests/unit /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python -m pytest tests/unit/test_step5_mrv2_forward.py tests/unit/test_kv_block_zeroing_host_time.py tests/unit/test_grouped_kv_cache_scheduler.py tests/unit/test_nixl_read_completion.py -q --basetemp /data/ycfeng/tmp/codex-review/63785f72-e511-4dfd-a7af-6b69354afd58/pytest-dp-wave -p no:cacheprovider -p no:logging
```

The failed test was `tests/unit/test_nixl_read_completion.py::test_decode_step_start_posts_and_checks_the_lane_reads`. At `210–225` its fixture constructs `object.__new__(DummyForwardBatch)` without initialization. `R/frontier/events/replica_stage_schedule_event.py:183` then accesses `_forward_start_host_delay`, raising:

```text
AttributeError: 'DummyForwardBatch' object has no attribute '_forward_start_host_delay'
```

The ordinary Batch constructor initializes that field at `R/frontier/entities/batch.py:576`. This is an existing incomplete fixture in the reviewed HEAD, not evidence of a runtime failure caused by item 4. It prevents reporting the affected suite as green and should be repaired in the implementation's validation work.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-preview:/data/ycfeng/Frontier/.worktrees/step5-preview/tests/unit /data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python /data/ycfeng/tmp/codex-review/63785f72-e511-4dfd-a7af-6b69354afd58/dp_wave_method_checks.py
```

The scratch script calls existing methods using existing lightweight unit fixtures. Its direct second dummy issue and recorded preemption callback deliberately expose the incompatible contracts without modifying production code. It is not an implementation of the design or an E2E reproduction.

Coverage limits:

- The inspected integration dummy tests enforce the current device-end hold and legacy PP2/PP4 finish behavior (`R/tests/integration/test_vllm_dp_dummy_pass_runtime.py:65–78,101–179`). They do not validate PP1 host-paced overlap.
- The supplied `A/wave_frontier/wave_probe.py:6` imports an older checkout at commit `02bd453a`. Its records support the reported pattern, but are not an execution of reviewed HEAD. Its `last` field comes from `on_dummy_forward_end` (`24–28`) and means the final pipeline part of a pass, not direct observation of wave consensus.
- Native backlog is reconstructed from host timestamps and CUDA spans, not measured GPU start times (`T/test_report_2026-10-27_h200_64_ttft_wave_tail.md:184–188`).
- No actual operator-priced isolated all-dummy DP2/DP4 tail was executed in this review. The new numerical backlog remains unverified.
- Final Git status contained the same excluded changes and untracked paths; the review introduced no workspace changes.

## 5. Open questions and next actions

### Unverified questions requiring evidence

| Question | Evidence needed |
| --- | --- |
| Will operator-priced DP2 dummy work create a false backlog? | Run the implemented all-dummy path with the actual E predictor inputs and separate device work from host/room waits. Using the reported Frontier baseline 597.8 ms, a simple one-second budget leaves `(1000 − 597.8) / 15 = 26.81 ms` per tail forward. This is a diagnostic sensitivity calculation, not a calibrated threshold. The native 24.8 ms span must not replace operator pricing. |
| What is the cost of idle-lane dummies in the first mixed PREFILL cohort? | Record the agreed descriptor, local attention work, MoE room waits, and completion for one real plus idle dummy lanes. The supplied first-cohort spans around 638 ms include synchronization with real work; they are not the isolated 32 ms all-dummy cost. Current source supports metadata-bearing dummy pricing (`R/frontier/execution_time_predictor/step5_cost.py:153–185`), but numerical sufficiency was not demonstrated. |
| How does the DECODE wave affect READ visibility and first-token timing? | Add new-path DECODE cases with READ posting, finding, reporting, finish consensus, and a new request during dummy drain. `R/frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py:57–58,74–77` currently has one shared dummy-found READ list. Serial device completion may keep that safe, but host-paced output/report semantics must be specified and verified rather than assumed. |
| Does the new EngineCore model pop completed output at the native time? | Distinguish nonblocking real RPCs, synchronous dummy RPCs, oldest-future waiting, and finish-consensus waiting. Preserve FIFO output application and transfer export. A callback held only at finish consensus is not a complete specification of all future visibility. |
| Will scheduling changes alone reach the DP4 TTFT gate? | Rerun G/E with separated prefill, host, backlog, and handoff contributions. The plan already records Frontier solo prefill about 60 ms high and handoff about 28 ms low (`T/plan.md:396–397`). These residuals can alter backlog growth and must not be hidden by a matching overall mean. |
| Is one scalar cadence stable across relevant windows? | Matched all-dummy cohorts, TP/DP completeness, defined local/group timing composition, variation statistics, and perturbation disposition. The current diagnostic medians alone do not answer this. |

These are validation questions, not additional confirmed defects.

### Corrections in dependency order

1. **Define the native timing boundaries and measurement composition first — F002 and F004.** Separate EngineCore iteration/pop state, worker CPU dispatch/reset readiness, and GPU FIFO. Decide whether the profile represents local host service or one group cadence, and mark the cited DP2 measurements exploratory until admission is resolved.
2. **Specify pass and wave ownership — F001.** Introduce per-pass completion state and exactly one host continuation owner. Preserve native unfinished snapshots and the step-1/16 rule; reject stale timers/completions without blocking the next wave.
3. **Extend or narrow grouped PREFILL chunk support — F003.** Resolve allocation, settled retirement, reset IDs, active counts, and final-output-only handoff before advertising native queue-2 chunk fidelity.
4. **Validate the implemented opt-in through existing event owners.** Include queued dummy drain, mixed cohorts, chunk boundaries, reset stalls, finish-sync arrivals, output pop, and DECODE/NIXL scenarios. Repair the existing incomplete NIXL fixture, run the default fidelity matrix, then perform G/E comparisons using admitted inputs.

The source evidence is sufficient to request these design corrections. It is not sufficient to claim the corrected design will meet the numerical TTFT gate without implementation, profile admission, and the stated validation.
