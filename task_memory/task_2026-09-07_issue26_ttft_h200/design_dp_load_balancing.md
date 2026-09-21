## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded the approved snapshot-policy implementation boundaries before editing. |

# D016 vLLM DP load balancing

## Scope and interfaces

Implement the approved vllm_load_balancing cluster policy for current co-location/vllm_v1/PP1, one serving Replica and one frontend. DP cardinality comes from the existing topology; no case-specific DP2 constant. Keep vLLM reference behavior unchanged.

1. A small RequestLoad value stores waiting/running separately. BaseReplicaScheduler exposes get_request_load; the active V1 implementation replaces its existing decision-log-only waiting getter with this shared accessor. Existing diagnostic fields retain their meanings.
2. A dedicated pure VllmDPLoadBalancer owns engine reports, frontend estimates, last-step snapshot, publication deadline and fixed-order weighted selection.
3. BaseClusterScheduler.schedule_at(time) preserves schedule() for existing policies and exposes DES time to the new policy. A batch-completion hook defaults to no action; the new policy reports the completed lane's post-step counts. GlobalBatchEndEvent invokes the hook after the existing request-state transition.
4. The new cluster policy is selected through the existing enum/config/registry. Validate supported topology before simulation.

## Snapshot timing

Copy the pinned upstream coordinator count-publication algorithm: changed counts report after engine steps; newer step with pending changes retains the previous counts; poll timeout is max(50ms when no previous snapshot is available, publication_interval minus elapsed), with100ms changed and5000ms unchanged intervals. Each publication replaces frontend estimates, including local waiting increments.

Process timer expirations lazily before the next report or routing observation. Between those observations only timers can change load-balancer state, so this preserves visible snapshots without persistent heartbeat events extending the DES run. A report arriving exactly at a deadline wins over that timeout, matching poll returning queued input; a frontend read observes due publications. Keep this deterministic tie convention explicit.

Use the existing synchronized forward-cohort ID as a monotonic ordering key for engine reports. Its numeric value is not a vLLM step counter; only ordering/equality are used. With one Replica and PP1, no independent pipeline-wave reordering is modeled by this policy.

Initial load is empty at simulation time0; bootstrap follows the upstream initial poll collection wait. IPC/CPU delivery latency and the exact residual heartbeat phase after real warmups are not measured here and are not fitted. The policy models count publication/frontend replacement, not the vLLM transport or distributed wakeup protocol. Request placement must be checked against observed traces; source-algorithm equivalence alone does not establish identical placement.

## Large-module inspection and bounded cleanup

Before this substep, config.py has5679lines and vllm_v1_engine_replica_scheduler.py has5073lines. RR1186, base cluster1871 and base replica1167 are below the2000-line soft limit.

The inspected V1 decision logger separately obtains waiting and running counts. Replace its narrow waiting getter with the approved shared load accessor and make the logger consume that value; this removes the duplicated count-selection boundary without changing diagnostic population semantics. The remaining V1 file owns tightly coupled admission/preemption/KV/request transitions; a broad split would exceed this substep and risks the reference behavior being calibrated. A future functional split should extract decision logging first, then waiting admission, then preemption with transition tests between extractions. The current patch touches only the load accessor and logger call sites.

Move the existing cluster-scheduler config family into frontier/config/cluster_scheduler_config.py, re-exporting the same names from config.py, then add the new sibling there. This removes that self-contained family from the oversized config module instead of growing it. Other remaining config families and runtime finalization are outside this substep; future extraction should follow existing model/device/transfer config boundaries and preserve flat-dataclass type discovery. No unrelated dead code is proven by this focused inspection.

The requested two-part implementation authorizes these bounded interface and registry changes. New public flags beyond selecting the policy are unnecessary. Generic50-scenario guidance is superseded by the user's single-case scope and direct verification instructions.

## Verification

RR commit ab752f97 is independently complete with20passingchecks. Snapshot tests exercise the weighted choice, fixed tie, local increments, changed-count publication, previous-step snapshots, unchanged-count heartbeat replacement, duplicate reports and report/deadline ties. Verify accessor counts across admission/preemption/completion and real DES hook wiring; compare deterministic count-publication traces with the pinned upstream implementation. Run a small real simulator diagnostic before the fresh full4096/1024 CPU case. Numerical acceptance remains open until clean artifacts are complete and rejoined.
