## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Prepared D018 minimal routing/enqueue diagnostic proposal after complete D017 validation. |

# D018 proposal: distinguish DP routing from engine enqueue

Status: PROPOSED; YC decision pending. No new vLLM changes, simulator event changes, or GPU job have started. D017 implementation, regression, and full numerical replay are complete. This document proposes a new diagnostic record contract; updating the plan does not authorize it.

## Observations and limits

- Current clean replay and repaired Frontier contain100valid formal requests. Raw TTFT105.443668418ms versus131.268637180ms (19.673373105percent error); batch comparison78sameDP/5samecomplete member vectors.
- In the first5client requests, complete prefill membership matches. The first divergence in clean engine-queue order is client6, assigned FrontierDP1 versus diagnosticvLLMDP0. See cpu-shared-forward-01/batch_comparison.json. Frontier request5maps to client6; this is intentional trace ordering, not a join error.
- Same clean run: request5server arrival precedes6by2.606391907ms, but request6engine queue event precedes5by0.200871844ms. Evidence: cpu-shared-forward-01/routing_boundary_evidence.json. These are same-clock differences within each domain; wall and monotonic clocks are never subtracted.
- Pinned clean/diagnostic vLLM v1/engine/core_client.py:1085–1094 selects an engine before sending ADD. DPLBAsyncMPClient.get_core_engine_for_request():1132–1157 uses the current lb_engines snapshot and increments the selected waiting count. v1/core/sched/scheduler.py:1506–1510 records QUEUED after engine receipt. Thus QUEUED is downstream of route selection.
- Frontier trace_replay_request_generator.py:240 uses one arrived_at field, RequestArrivalEvent enqueues it for global/cluster scheduling, and vllm_load_balancing_cluster_scheduler.py:36–48 selects DP at that scheduling time. The current input file contains observed engine QUEUED offsets. Actual route timestamps/order are unavailable.
- No complete route/enqueue logger is present in inspected core_client.py. Existing server.decisions.jsonl records engine scheduler admission, with no frontend routing snapshot. Existing E2E rows record queue time but come from the separate clean run. Neither record proves the diagnostic frontend's route decision state.

Observed boundary separation is not proof that it explains all22DP differences or the25.825msTTFT gap. Different runtime speed, asynchronous snapshot delivery, and isolated diagnostic arrival timing remain confounders. A direct scheduling fix is not justified yet.

## Recommended bounded extension

**本质:** Observe the actual request route and subsequent engine enqueue in the same isolated run before changing simulation behavior.

**依据:** The two decisions occur at separate inspected vLLM hooks, while existing evidence only exposes the downstream event. Existing Frontier trace/logger support and the diagnostic worker already provide an isolated output directory and explicit enablement.

**前后对比:** Current client6 -> unknown route instant/snapshot -> observed queue timestamp -> Frontier reroutes at the queue timestamp. Proposed client6 -> recorded route instant/snapshot/chosenDP -> recorded queue event on thatDP; compare those records with Frontier selection and only then choose an event-model repair.

**好处与代价:** Establish which event order and load values actually produced a selectedDP, instead of guessing from server arrivals or fitting a time offset. Cost: a scoped vLLM diagnostic record extension and one isolated H200 step_main replay of the same4096/1024case with three drainedwarmups plus100formal requests. Per-request logging can perturb diagnostic timing; keep it separate from the existing clean numeric baseline and do not claim diagnostic latency as clean TTFT.

Use the existing v1/frontier_trace.py logging surface and VLLM_FRONTIER_SCHED_DECISION_LOG_PATH/diagnostic worker convention, with small route and enqueue hooks. Avoid importing the heavy scheduler module into the API client solely to access a logger; place the shared emission in the existing trace helper. Expected changes: trace helper, core_client route hook, scheduler enqueue hook, and the existing test worker only if needed for explicit enablement. No new simulator flags or request fields are necessary for evidence capture.

Minimum route record: event type, request_id, client_index, monotonic route timestamp, selected DP rank, immutable copy of the load counts used before the route-local increment, and whether a rank was explicitly requested. Minimum enqueue record: same request_id, DP rank, and the exact existing QUEUED event timestamp. Capture warmup rows under their existing namespaces and exclude them explicitly in the analyzer. No synchronous GPU event, CPU copy of routing tensors, or GPU op instrumentation is required for this routing question.

## Verification criteria

1. Disabled logging preserves route selection and request behavior in a source-method check; enabled logging records the exact counts read by the existing algorithm, not a later mutated list.
2. Same isolated run has exactly one route and one enqueue per formal request, unique identities, matching selectedDP/enqueueDP, and route time <= queue time in the same host monotonic clock domain. Warmups remain separate.
3. Recompute route choice from the recorded counts using the already implemented weighted-load/tie-break contract; distinguish a policy mismatch from timing/snapshot visibility mismatch.
4. Report the first divergence with actual counts and event ordering. Preserve the existing clean TTFT reference. Only a subsequent reviewed evidence-backed proposal may change Frontier's routing/enqueue event model.

## Other candidates considered

- Reuse existing logs only: insufficient; no route timestamp or pre-selection load snapshot is recorded.
- Substitute server arrival for engine queue arrival immediately: would change the existing Frontier input/TTFT boundary and still lacks actual route time; not an evidence-backed fix.
- Force observed DP owners or batch schedules: requires another replay mechanism and would bypass the policy under investigation; not part of this proposal.
- Add a constant25.825msCPU term: explicitly violates D006 and leaves28percentTPOT/E2E gaps and composition differences unexplained.
- Add operator diagnostics first: useful after local shapes/identity align, but cannot explain the missing frontend route state. Preserve as the later operator lane.

One actionable recommendation remains: the minimal same-run route/enqueue evidence extension. Holding the extension for YC to redirect is a valid alternative. Approval is requested because this adds a shared diagnostic data contract beyond D017; D017 authorization is already fully executed.

First-batch limit: fresh Frontier first-stage65.790076904ms versus isolated vLLM CUDA batch span77.125022888ms (14.696846percent diagnostic gap); official clean firstTTFT121.484279633ms. Later DP placement divergence cannot alone explain this first-request difference. The route diagnostic is proposed to qualify full-case workflow, not claimed to close all latency residual. Evidence: analysis/cpu-shared-forward-01/first_batch_timing_comparison.json.
