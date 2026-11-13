# Step profiling integration requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record the parent-authorized integration slice and preserved calibration boundaries. |

## [Original Request]

> 确认，更新docs，锁定该版本的workload，开始并行推进执行

This slice integrates the profiling branch into the Step branch after source review, affected tests and legacy fidelity. Parent requirements/spec/plan remain authoritative at `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/`. D18 uses common workload68f891b/958d81a,128 requests6144/1024 at QPS1,16 warmups per three repetitions,64H200 and four independent10% metrics. Isolated operator probes are not canonical qualification. Preserve all sealed stages, existing failed timing verdicts, native precision/dimensions/guards and unrelated work. Parent owns GPU execution.
