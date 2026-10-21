# D25: complete H800 native workload measurements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recorded YC's full-workload priority and requested latency analysis. |

## [Original Request]

> 【并行】检查已有记录，当前是否以及完成了 step5-preview在workload上的vllm的测试运行（32卡，64卡）并完成p90 ttft,tpot,e2e以及吞吐的统计？如果完成，请简洁回报，并确认其数值是否正常（如果ttft的数值>5s，你需要进行分析，是否明显偏大）。如果没有完成，优先完成完整的workload运行（32卡,64卡，h800）

## Execution scope

- Prioritize native workload collection on non-preemptible H800/codesign: 32 GPUs (P16/D16, TP8/DP2/EP16/PP1) and 64 GPUs (P32/D32, TP8/DP4/EP32/PP1). Submit independent jobs concurrently when their nodes/ports are isolated and quota permits.
- Use the frozen shared workload without changing its five files: three repetitions, each with 16 warmups followed by 128 requests, 6144 input tokens, 1024 output tokens, QPS=1. Preserve actual completion, token identity, warmup drainage and per-lane idle evidence.
- Reuse the existing canonical client/worker/stager. Native performance collection is now explicitly requested before calibration readiness. Represent this purpose explicitly as `workload`; the existing `canonical` calibration release continues to require native readiness and the same-hardware 32-to-64 accuracy dependency. Collection does not establish either prerequisite or the final accuracy verdict.
- Both sizes use one API frontend per role and the same locked image, model, native graph policy, scheduler limits and connection behavior. No R2/G3 state capture, operator instrumentation, cache overrides or timing subtraction in these clean measurement jobs.
- Existing 90-minute ceilings apply: at most 48 GPU-hours for 32 GPUs and 96 GPU-hours for 64 GPUs. Use personal StepMind Python submission, mounted local source, cloud artifacts confined to `/mnt/codesign-exp/ycfeng`, and non-blocking auxiliary checks.
- Report nearest-rank p90 TTFT/TPOT/E2E, total-token throughput and output-token throughput, with units and valid populations. Retain means and per-repetition values. TTFT >5 seconds requires evidence-backed analysis, separating measured attribution from hypotheses.
- Failed requests stay visible. Incomplete populations cannot be reported as a complete successful workload. Preserve historical receipts and failed/UNVERIFIED verdicts.

## Existing evidence

The native metrics audit in `../task_2026-10-21_step5_priority_runs/metrics_check_20261022/` covers only 16 warmups plus 8 simultaneous requests. H80032 p90 TTFT was 4.791508s; H80064 was 9.093689s. Different historical API frontend counts confound that scale comparison. No complete three-repeat canonical workload has been established.
