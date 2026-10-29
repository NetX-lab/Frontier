# D29 fixed-model group reuse

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Accept parent real-path RED/control evidence and allocate the first performance repair inside CORE's existing quota. |

## Authority and measured gain

D29-REPAIR-PATH-AUTHORITY permits necessary repairs to the approved behavior
inside unchanged owner/global ceilings. This is a CPU-efficiency correction to
the already-approved position-routing consumer, not a new model or cost rule.

Owner diagnosis commit d8a08722 contains the exact commands and boundaries.
Parent reran the retained runtime_loop.py (SHA256
bba92e780743cea23b5c92b852e56c7cabe9867e8c4016b970fc9f599344852e) on frozen e41:

| Run | Two actual pricing calls, seconds | Repeated layer descriptions | Verdict |
| --- | --- | --- | --- |
| parent-red01 | 3.424171769 / 3.368901219 | 1056 | RED, exit1 |
| parent-reuse01 | 0.396528880 / 0.349091294 | 0 | Diagnostic GREEN, exit0 |

The only control variable reuses the actual resolved group tuple for the same
model object. Both cost/proof/full-plan-key payloads hash to
d2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952.
Output directories and logs are under /data/ycfeng/tmp/d29_cli_timeout_repro/.
Parent commands are the diagnosis report's RED/control commands with output/log
names changed to parent-red01 and parent-reuse01 respectively.

This is one real P6128 plus idle1 point at layer3/EP0, using retained paired models
and actual production pricing/Cluster/backend. Setup is excluded from timing.
It establishes the hotspot and unchanged point results; it does not establish
full6144/1024 completion within2400s. The sampled profile puts14.438 of15.239s
inside12 complete group resolutions; profiler overhead is not unprofiled timing.

## Scoped design

CORE owns exactly the three paths in model_group_reuse_scope_amendment.json:
two existing consumers and their existing dual-binding regression. Reuse the
existing resolved RoutedLayerGroup facts once per fixed-model predictor and
route both consumers through that one local selection path. The existing
step5_layer_metadata cache demonstrates the predictor's fixed model/parallel
lifetime, but it contains per-layer runtime geometry rather than the complete
grouping needed by measurement-binding selection. Document that distinction
without adding a second model registry or modifying the resolver/profiler API.

Prove the supported lifetime from construction. Keep separate predictors/models
isolated; keep layer3/88 group selection, measurement family and native source/
context selection live. Never memoize selected estimators, point admission,
queries, durations or proofs as part of this repair. Do not add invalidation
branches for unsupported model mutation. Preserve reachable error behavior.

Budget: source<=45 gross,test<=60,total<=105;CORE remains<=4404 and global<=10000.
The overlay transfers100 unused CORE path allowance with no path-set change.
Donors remain above their measured retained diffs. All earlier step caps remain.

## Acceptance and route

1. Add a failing regression at the real dual-consumer/binding seam before the fix.
   Repeated requests across layer3/88 must resolve groups only once per predictor;
   a separate predictor/model must resolve its own groups. Preserve actual model
   selection and query/plan payloads, including the existing negative coverage.
2. Apply the minimal local reuse; rerun affected consumer/binding tests and the
   retained real pricing loop against the candidate source without --reuse-groups.
   Pin the candidate source explicitly and retain any probe-only source-pin diff.
3. Parent verification and independent review precede the next source freeze.
   Profile the remaining real-point cost before deciding whether another bounded
   repair is needed. No full CLI retry is delegated in this step.
4. Final acceptance still requires unchanged6144/1024 offline/online workloads,
   the2400s per-mode cap and all business validators on a reviewed frozen source,
   plus the required fresh regression set. T2 staging and GPU resources remain
   gated on workflow readiness.

Secondary artifact-JSON and initialization observations stay diagnostic. They
require their own measured scope/ownership allocation before any implementation.
