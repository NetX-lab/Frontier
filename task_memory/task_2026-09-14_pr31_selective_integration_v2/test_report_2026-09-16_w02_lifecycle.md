## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded automatic capacity, real Simulator lifecycle and admission rollback repair. |

# W02 capacity and lifecycle

R02/R03/N01-linked capacity: admission cap is resolved once in BaseReplicaScheduler and passed to both MemoryPlanner and state-slot owner. ParamCounter exposes the resident stage selection used by the planner, removing private cross-module access. KV/state allocation now finishes before committing waiting-queue removal and waiting counters. The failing injected state-allocation check reproduced request loss after KV rollback, and now proves queue, slots, KV, epochs, token and waiting counters unchanged.

Execution environment: `/usr/bin/python`, Python 3.12.3, no conda, same W00 dependency paths. Commands from repository root:

```bash
env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /usr/bin/python -m pytest tests/unit/test_gdn_scheduler_slots.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py tests/unit/test_gdn_increment6_memory.py tests/unit/test_gdn_state_lifecycle.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w02-fixed
env PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /usr/bin/python -m pytest tests/unit -q -p no:cacheprovider -k 'memory_planner or param_counter or preemption or waiting or admission or gdn_scheduler' --basetemp /data/ycfeng/tmp/pr33-w02-regression
```

Results: **26 passed in 7.91s** and **185 passed, 3205 deselected in 13.73s**. Logs: `/data/ycfeng/tmp/pr33-w02-fixed.log`, `/data/ycfeng/tmp/pr33-w02-regression.log`. Criteria include independent state-byte/KV floor oracle, both automatic planner modes, nonincreasing blocks at increased capacity, structured OOM, homogeneous zero reservation, forbidden preemption without mutation, and normal production-constructor synthetic Simulator with three online arrivals, capacity 1 and 4-token prefill chunks. Slot 0 is retained through continuation, released at finish and reused by all three requests. Quiescence has no remaining requests, KV allocation, slot ownership or running batches. These are synthetic CPU integration observations, not measured GPU latency.

D03: completed inventory in `w02_test_notes.md` finds no request cancellation API. Whole-simulation abort/time-limit is not request cancellation and has no supported resume/ownership reuse contract. User choice is pending; retaining supported lifecycle is recommended. No API was invented.

## Cleanup and boundaries

| File | Before LOC | After LOC |
| --- | --- | --- |
| `frontier/scheduler/replica_scheduler/base_replica_scheduler.py` | 1192 | 1195 |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 5142 | 5143 |
| `frontier/scheduler/utils/memory_planner.py` | 385 | 382 |
| `frontier/utils/param_counter.py` | 673 | 678 |

The >2000-line scheduler retains intertwined supported prefix/PP/MTP phase state machines. This sub-step removes duplicated capacity interpretation and orders the existing resource transaction correctly. A functional split should move the existing request resource lifecycle (allocation/release/KV+state transaction) behind the scheduler's admission owner, followed by queue-phase scheduling only after those contracts stabilize; duplicating a new resource manager now would increase competing ownership. No unrelated scheduler refactor is included. Remaining mock-tolerant legacy probes need hunk-by-hunk W11 adjudication; no blanket readability completion is claimed.
