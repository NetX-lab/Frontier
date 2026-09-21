## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Separated round-robin correctness from vLLM internal DP policy alignment. |

# DP routing repair — approved and implemented

YC approved both proposed substeps. RR is committed ab752f97; snapshot policy is committed a4a0c496. Functional and DES validation pass; fresh numerical comparison remains pending platform/ground-truth recovery. Original proposal details below preserve the reviewed scope. Implementation details and measured limits are recorded in ../design_dp_load_balancing.md and ../test_report_2026-09-08_vllm_dp_snapshot_policy.md.

## Established findings

Scope is the active TP4/DP2/PP1, single-Replica, single-API-client vLLM V1 internal load-balancing deployment. Findings are not a claim about every vLLM release or external load balancers.

1. Frontier RR resets each Replica's DP enumeration per schedule call. Persistent request ordinal is used only for Replica selection. See frontier/scheduler/cluster_scheduler/round_robin_cluster_scheduler.py:359.
2. Active vLLM selects the engine minimizing 4*waiting+running, with strict-less-than tie-breaking from a fixed client-specific starting engine. It increments the selected local waiting estimate by client_count and later replaces its local counts with coordinator snapshots. See /data/ycfeng/tmp/vLLM-BS/vllm/v1/engine/core_client.py:1132 and :1078.
3. Frontier LOR reads num_pending_requests. In the active MONOLITHIC vllm_v1 scheduler that is request_queue+preempted_requests, excluding running_requests. See frontier/scheduler/cluster_scheduler/lor_cluster_scheduler.py:26 and frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:4952.
4. Actual-method controlled checks PASS; see ../test_report_2026-09-08_dp_routing_source_probe.md and dp_routing_source_probe.json.

Primary upstream cross-check: fetched the official [vLLM v0.10.2 core client](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/engine/core_client.py#L1132-L1157) through the company HTTP proxy. AST comparison confirms its routing method is identical to both current clean and diagnostic checkouts. This establishes the policy is upstream behavior for the pinned release, not an instrumentation-branch-only change.

## Recommended implementation

Keep round_robin as round robin. Its standalone correction uses global ordinal q: replica=q%R, dp=(q//R)%D, retaining the existing per-Replica grouped return order. It needs the current RR module and the existing tests/unit/test_cluster_scheduler_dp_lanes.py. This repairs stream-partition dependence but is not the vLLM parity implementation.

For the calibration case, add a distinct vllm_load_balancing cluster-scheduler policy through the current enum/config/registry mechanism. Keep its initial supported scope explicit: current co-location, vllm_v1, one serving Replica and one API-client equivalent. Multi-Replica external routing is a different boundary and should not be invented in this substep.

The policy needs:

- Separate waiting and running request counts from the lane scheduler. For current MONOLITHIC semantics, waiting=request_queue+preempted_requests and running=running_requests. Verify admission, preemption and completion transitions before accepting this mapping.
- A small public load-state accessor. Preserve num_pending_requests because existing callers use its schedulable-pending meaning. Avoid reaching into private queues from the cluster router and avoid reusing the expensive debug-state serialization.
- Stable DP engine order, weighted scoring, fixed tie-breaking, persistent frontend estimates, and a local waiting increment after every selected request.
- Explicit load-report visibility semantics. The actual coordinator publishes changed counts around a 100ms minimum interval, unchanged state around 5s, with step/wave handling and a 50ms collection wait in part of its loop. It is not an exact periodic 100ms sampler. Reading live scheduler state on each arrival is an approximation and must not silently become the parity contract.
- A dedicated small module for routing and snapshot state. Register it with the existing ClusterSchedulerType, BaseClusterSchedulerConfig subclasses, and ClusterSchedulerRegistry. Keep unrelated scheduling branches unchanged.

This touches shared config/interface boundaries and likely more than three source files. Therefore the implementation requires the material policy decision under YC's workspace approval gate. This document is a concrete proposal, not authorization. Inspect the project's critical-module size rules before any accessor edit in the large vllm_v1 scheduler; avoid an unrelated large refactor.

## Verification sequence

policy_decision -> {RR_regression_if_selected, load_state_and_router_checks} -> fresh_current_case_Frontier -> request_and_batch_identity_comparison -> official_TTFT_and_secondary_metrics.

- Router checks: empty tie, unequal running with equal waiting, waiting/running weighting, successive arrivals before a snapshot, replacement by a new snapshot, and state across schedule calls.
- State checks: admission/preemption/completion maintain the intended waiting/running population without double counting.
- Fresh 4096/1024 case: inspect both lane admissions, each request's assigned owner, first formal prefill request IDs/token vectors, and mixed prefill/decode composition. Balanced totals alone are insufficient.
- Report official server TTFT against Frontier with absolute and relative error. Do not attribute the previously observed 23.870ms mean residual to this routing defect before a valid re-run.

## Limits and pending evidence

Same routing algorithm does not guarantee identical individual placements or batch membership: simulated service times, feedback timing, and arrival boundary differences can change router state. Exact placement replay is a separate controlled experiment and is not implemented by this proposal.

The latest H200 replay has only232 completed warmup requests visible as of03:40UTC; last output03:31UTC. Platform queries failed with Bad Gateway and EOF. Clean400/formal100 and isolated batch artifacts are incomplete or unconfirmed. Preserve existing artifacts and recover task visibility before using or retrying that execution.
