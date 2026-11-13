# D21-NATIVE-REPLAY requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded delegated native replay diagnosis and old64 evidence reuse scope. |

## [Original Request]

> Continue native work as D21-NATIVE-REPLAY. Goal resolve why P0 complete logs lack actual EngineNNN/FULL rows despite cudagraph_metrics=true, and assess old H20064 basic reuse before another64H200 job. No GPU/auth/network/SSH or further delegation. Work code ONLY /data/ycfeng/Frontier/.worktrees/step5-priority-runs (now42dda4c; parent fix/tests already committed); records there under task_memory/task_2026-10-21_step5_priority_runs/native_replay/. Read current parent spec/req09 and frozen image source /data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm. Diagnose actual MRv2 metrics transport/logger and staged server argv/env using existing output, not assumption. If supported runtime logging toggle/caller mistake explains absent graph evidence, implement minimal harness/config correction with meaningful CPU source/caller regression and fresh offline stage only; don't change native kernels, thresholds or sealed01/02 packages. Preserve request/accounting and soft auxiliary-gate semantics. Read old H20064 /data/ycfeng/tmp/step5_d18_native_20261021/qualification02/stage/evidence similarly; report basic proof reuse and gaps. G3 remains distinct: list exact case-aligned witness/reference missing; no broad G3 implementation in this subtask. Deliver compact root cause, code/commit/tests if any, source refs, fresh-stage recommendation (no submit), old64 reuse result. Parent owns next decision and platform. Actual-main agent owns simulator files; no overlap.

## Explicit decisions and acceptance

- Authority: current primary `spec.md` §0, D21 requirements and tickets08/09/23 in the step5-preview worktree; local parent requirements/plan remain in force.
- Modify only native harness/config and focused CPU tests in this worktree; all child task records stay here. No native source/kernels, simulator, thresholds, frozen inputs or sealed historical packages may change.
- Preserve original P0/old64 outcomes and distinguish graph logging enablement from actual replay qualification. Formal readiness remains NO without genuine case-aligned prerequisites.
- Fresh stage is local offline preparation only. Parent owns authorization, launch, platform and old64 retry decision; do not invoke submit.sh or source authentication environments.
