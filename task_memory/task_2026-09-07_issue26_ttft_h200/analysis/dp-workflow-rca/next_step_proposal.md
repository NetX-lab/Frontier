## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Proposed the smallest route/engine-visibility seam and named missing full-step evidence; no implementation or tests started. |

# Next workflow step proposal

Status: PROPOSAL ONLY. D018 authorizes RCA and diagnostic measurements, not this shared runtime/interface change. The recommendation is to preserve separate frontend routing and engine queue/admission boundaries using one engine-input queue and the existing replica scheduling path. Do not turn the successful route-input diagnostic into a one-line production arrival replacement.

## Why a second boundary is required

Fresh same-run evidence and unchanged-predictor DES controls are recorded in fresh_boundary_controls.json and same-run-01/same_run_join.json. Enqueue-input gives6/8correct DP owners; route-input gives8/8. Yet request0prior scheduled tokens in the second mixed batch are4096(vLLM),4097(enqueue-input Frontier),4098(route-input Frontier). A single arrival timestamp can preserve either the observed routing order or the current engine-queue endpoint, but cannot express both timings and their intervening execution-dependent wait.

Actual vLLM route5 precedesroute6 by2.023212146ms; engine enqueue6 precedesenqueue5 by0.048529357ms. The selector uses counts[[0,3],[0,2]] then[[0,3],[1,2]]. Both Frontier controls reach the same count sequence. Correct owner selection therefore requires routing before the independent engine queues expose requests.

The current Frontier trace generator constructs Request from row.arrived_at. RequestArrivalEvent calls Request.on_arrival, records metric arrival, and sends the request to the global/cluster scheduler. VllmLoadBalancingClusterScheduler.schedule_at chooses DP at that event time; ClusterScheduleEvent immediately calls the selected replica scheduler's add_request and creates ReplicaScheduleEvent at the same timestamp. GlobalBatchEndEvent immediately creates the next ReplicaScheduleEvent at its own timestamp. Those two immediate transitions are the seams to change; the weighted-load formula and vllm_v1 admission algorithm do not need another rewrite.

## Concrete minimum runtime seam

The proposed causal sequence is:

`frontend route arrival -> existing DP selector -> selected engine input queue -> engine input drain at its next valid loop boundary -> existing replica waiting queue -> existing admission and forward -> full engine step completes -> next input drain/admission`

1. **Keep routing ownership fixed after selection.** Preserve the existing `(replica_id, replica_local_id, request)` mapping returned by schedule_at. Do not change the shared tuple merely to carry an optional timestamp. At the current add_request point in ClusterScheduleEvent, submit the selected request to a small engine-input owner instead of immediately making it visible to vllm_v1. The owner belongs to the existing single-Replica vllm_load_balancing path, where the lane and input state are known. Preserve other cluster policies through the existing scheduler abstraction; avoid scattered model-name or case-name conditions.
2. **Distinguish input-pending from scheduler-waiting.** A routed request remains input-pending while its selected engine is inside a step. Only an actual input drain calls the existing replica scheduler.add_request. get_request_load must continue reporting the scheduler's waiting/running populations; adding input-pending requests to scheduler.waiting early would publish the wrong vLLM coordinator counts. The frontend selector already applies its own local waiting increment at route time, which is a separate estimate and must remain separate.
3. **Reuse replica admission, not a second batch builder.** The engine-input drain transfers ready requests in that engine's input order, then invokes the existing ReplicaScheduleEvent/on_schedule machinery. Route arrival while an engine is busy must not schedule a new batch. Route arrival to an idle engine must wake its input drain through a DES event. An engine-ready callback/event must also drain input before scheduling the next batch. These are runtime events, not wall-clock polling or a fixed sleep.
4. **Give the engine a full-step readiness boundary.** The current GlobalBatchEndEvent updates per-request forward progress and emits a next-schedule event immediately. Retain the existing shared EP forward protocol and per-source token progression. Introduce the minimal handoff that releases input drain/admission only when the corresponding engine step is ready. Do not delay every metric and token callback indiscriminately: model forward completion, sampled-output availability, and next engine admission are distinct endpoints. Their actual placement requires the measurements listed below.
5. **Keep one source of engine readiness.** One lane/engine input owner should hold pending routed requests and whether the engine can drain them. Reuse existing batch/engine lifecycle callbacks to change that state. Do not add independent phase-specific ready timers or another prefill/decode scheduler. The same engine-input mechanism handles pure prefill, pure decode and local mixed batches; EP rendezvous remains the previously repaired common forward-group/layer mechanism.

This is a small conceptual seam but a cross-cutting implementation: ClusterScheduleEvent's immediate handoff, the cluster scheduler contract for input delivery/readiness, GlobalBatchEndEvent's next-step transition, and the event/trace entry point are shared surfaces. The precise event type and default policy dispatch must be reviewed together before changing them. A new dedicated engine-input/ready event is preferable to reusing RequestArrivalEvent, whose present handler reroutes and records arrival, or BatchStageArrivalEvent, which already represents a formed batch entering a model stage.

## Trace and metric interface decision

Request.ttft currently returns prefill_completed_at minus Request.arrived_at; Request.on_arrival also creates waiting-queue bookkeeping. Reinterpreting arrived_at globally as frontend route time would silently change that contract and reproduce the failed route-input progress behavior.

Recommended trace seam: introduce a small typed route-arrival record at the trace/event boundary, holding the request reference, route timestamp and deterministic trace order. Keep engine QUEUED timestamps separately as diagnostic reference observations. Do not make the normal predictor consume future ground-truth enqueue timestamps. Existing Request cluster-arrival bookkeeping can record the actual simulated engine queue entry once, when the input queue drains; the metric origin must be explicitly designated and preserved rather than inferred from whichever event fires first.

Two levels must be distinguished during review:

- A diagnostic two-timestamp replay can route at the observed route time and expose the same request at the observed enqueue time while preserving its selected owner. This is a useful acceptance fixture for event order and identity. Because enqueue time already contains vLLM execution-dependent waiting, it is not a predictive production model or a full E2E result.
- A production DES model receives external route availability and derives engine queue entry from transport/input readiness and the simulated full-step boundary. It must not look ahead to observed vLLM enqueue times. The current Request.arrived_at/metric origin, trace-row schema, and event initialization need an explicit shared-interface decision to support this without silently changing legacy schedulers or the D006 endpoint.

YC should approve whether the next implementation is scoped to the diagnostic two-boundary replay contract first, or to the predictive engine-input/full-step model after the missing timings are available. The recommended production direction is the latter; an additional replay fixture can validate its identities, but cannot replace it. Do not add a generic optional delay flag that merely hides an unmeasured interval.

## Comparison of supported choices

| Choice | Observed benefit | Observed or required cost | Suitable next action |
| --- | --- | --- | --- |
| Keep only QUEUED as arrival | Retains current internal metric origin | Reverses the first routing pair and reroutes after engine-dependent wait | Preserve as the baseline diagnostic only |
| Replace arrived_at with route timestamp | Fresh prefix DP owners improve6/8 ->8/8 | Second mixed-batch prior tokens worsen4097 ->4098 versus actual4096; immediate execution visibility and TTFT origin change | Reject as the production repair |
| Diagnostic replay with separate route and observed enqueue events | Can preserve measured route order, owner and queue visibility independently | Requires a trace/event contract; imports execution-dependent ground-truth timing and does not predict it | Optional narrow fixture after YC design decision |
| Predictive route -> engine input -> full-step drain model | Represents the inspected vLLM causal path without future enqueue knowledge | Shared lifecycle/interface changes and measured full-step boundaries required | Recommended production direction after evidence and YC review |

## Measurements required before implementation

Use one identity tuple linking engine/DP lane, frontend request, core step, worker batch and common forward participation. Existing scheduler.step and worker.batch_id are local counters; they must not be mistaken for the engine finish-sync step_counter or a shared EP round.

1. **Input visibility:** capture ADD send/receive into the EngineCore input queue, input-drain begin/end, and the existing Scheduler QUEUED event. Current route/enqueue timestamps establish the full interval but do not separate transport from time spent waiting for an engine step. This matters before assuming transport is zero or moving all waiting into the input buffer.
2. **Full step:** capture scheduler begin/end; executor dispatch/result readiness; worker preprocess; CUDA forward completion; logits/sampling/bookkeeping completion; scheduler.update_from_output completion; output/load publication; next input-drain boundary. Use existing operator/CPU instrumentation hooks where their actual endpoints fit. The current83.452619147ms scheduler interval minus80.335617065ms CUDA forward is an aggregate3.117002081ms, not a phase measurement and not a constant to implement.
3. **DP/dummy state:** capture dummy invocation begin/end and engine/forward identity, plus actual periodic finish-sync entry/exit and engines_running transitions. Current evidence shows no post-formal warmup request schedule but a stale frontend snapshot and a still-running frontend wave. It does not prove whether a dummy forward was active during the39.147422183ms first route-to-enqueue interval.
4. **Request completion endpoints:** establish when the first sampled token becomes available to EngineCore/output processing relative to forward completion and next scheduling. This determines which existing request callbacks can remain at forward end and which metric sidecar endpoint represents official server TTFT. D006 remains authoritative; do not move prefill completion merely to make TTFT match.
5. **Publication versus visibility:** retain the new snapshot_receive rows and add engine count-publication/core-step identity only where needed to connect full-step completion to the coordinator. The selector formula is already verified on400actual states; further tests of the arithmetic will not explain asynchronous timing feedback.

The initial state is a separate high-value decision: either begin formal measurement only after request drain **and** observed DP engines_running=false, or deliberately carry the warmup engine/dummy state into both sides. Waiting for client completions alone does not establish the former. A fixed39ms sleep would establish neither. The current calibration/control contract must be reviewed with YC before changing this initial condition or adding a clean-run idle gate.

## Approval boundary and proposed sequence

Already authorized: read-only source investigation and minimum same-case diagnostic measurements under D018. Not yet authorized: the shared trace/Request metric-origin contract, engine-input scheduler handoff/readiness interface, new event lifecycle, or changing the formal initial-state criterion. These are the specific YC decisions required; a task report cannot grant their approval.

`complete missing same-run phase/identity measurements -> review route/input/step and initial-state contract with YC -> implement one engine-input/readiness seam -> verify first close pair and first mixed-batch progress -> fresh full100 closed-loop comparison`

Acceptance for the implementation should include fixed owner after routing, exactly one engine queue entry, waiting/running counts matching the correct populations, no next admission before full-step readiness, and per-request progress preserved for pure/mixed/idle-DP participation. Then repeat the full current4096/1024case with fresh acceptance outputs. Do not introduce39ms,3ms, or another fitted constant; do not force observed DP owners in the production scheduler. No code or test execution is part of this proposal.
