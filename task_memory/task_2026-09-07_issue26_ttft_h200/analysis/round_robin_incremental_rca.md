## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded replay preparation and directly reproduced incremental DP assignment failure. |

# Incremental request assignment loses attention-DP rotation

Revision: 897d248257b712e0564a45af13db1830799dea80; active Frontier worktree only.

## Evidence and cause

- Current CPU uniform run: 1,839 stage-batch ledger rows, all execution_scope=ATTN_DP_LANE and replica_local_id=0; effective replica config attn_dp=2 and one Replica.
- First five prefill admissions progressively contain [0], [1,0], [2,0,1], [3,0,1,2], [4,0,1,2,3]. Token vectors are [4096] and [4096,1,...]; these are direct batch records, not inferred from latency.
- `RoundRobinClusterScheduler.schedule` routes MONOLITHIC to `_schedule_batch_mode`.
- `_schedule_batch_mode` uses the persistent `_request_counter` to choose Replicas but re-enumerates each invocation's requests starting at local_idx=0 to choose DP (`dp_id = local_idx % self._replica_dp_size`). Incremental singleton arrivals therefore all choose local lane0.
- Direct execution of that real method, with six distinct requests, yields streamed lanes [0,0,0,0,0,0] versus one-burst lanes [0,1,0,1,0,1]. Artifact: round_robin_incremental_reproduction.json. This falsifies a ledger-only labeling explanation for the reproduced path.

## Bounds on the conclusion

The bug establishes a Frontier routing/control-flow defect, not a numerical attribution of the 23.870 ms mean TTFT gap. The first request uses one real lane even with a correct DP policy, so this defect alone does not explain its 65.790 ms versus 119.174 ms TTFT. Current vLLM source chooses engines using waiting*4+running and periodically refreshed counts; a corrected round-robin scheduler is still not exact vLLM load balancing. Fresh batch evidence remains required for cross-side request-level placement and same-round dummy participation.

## Proposed smallest correction (D016 pending YC)

Use global request ordinal `_request_counter + request_idx` for both Replica and lane selection. Compute lane as `(ordinal // num_replicas) % attn_dp`, carry `(lane, request)` through the existing per-Replica grouping, and retain the current grouped return order. No new flag, schema, registry, simulator abstraction or vLLM source change. Existing all-at-once mapping from ordinal0 remains identical. Verify singleton, irregular chunks, empty calls, multiple Replicas and nonzero progress against the same logical round-robin sequence, using the existing cluster-scheduler DP test module. Then run current fresh case and inspect actual lanes and metrics.

Switching to LOR is a different policy and is not used as a substitute for repairing round-robin state. Exact traced placement import remains deferred. No timing scale or activation correction is included.

## Reproduction command

```bash
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'CHECK'
from types import SimpleNamespace
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import RoundRobinClusterScheduler
s = RoundRobinClusterScheduler.__new__(RoundRobinClusterScheduler)
s._num_replicas = 1
s._replica_dp_size = 2
s._cluster = SimpleNamespace(replicas={7: object()})
s._request_counter = 0
stream = []
for request in range(6):
    s._request_queue = [request]
    stream.extend(s._schedule_batch_mode())
s._request_counter = 0
s._request_queue = list(range(6))
burst = s._schedule_batch_mode()
print('stream', stream, 'burst', burst)
CHECK
```

Observed functional result: FAIL incremental/burst assignment invariance. Script exit0 records a successful reproduction of the failure. Runtime: CPU conda dev-vidur-v03-hopper-e2e, Python3.13.13. Human review pending; vLLM clean/batch replay is independent and continues.
