# One-shot sparse P0 metrics-tail diagnosis

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Recorded the single60s timeout and measured construction/GC costs; allocated a separate representation correction. |

**PASS_DIAGNOSIS_WITH_CENSORED_CONSTRUCTION**,not complete-trace acceptance. The child exited124 after60.307500s enclosing time;final JSON is absent. The diagnosis allocation is spent1/1. Sort and union did not start. R5 remains24/32.

## Command and environment

Driver: `/usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_tail_20261025_01/run_diagnostic.py`.

Child: `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_metrics_timing.py torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842834745705872.pt.trace.json.gz /data/ycfeng/tmp/glm53_dsv4/p0_metrics_tail_20261025_01/prefill_dp0_rank0_timing`.

Cwd `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`;committed timer2fbd5af6,SHA256fdce5eca10e0cea28af12b87d25a94ed68f0ce0de2eda77c6b5fa2ccb5b541b5. System Python3.10.6/GCC11.3.0,`-u -B`,no conda. Driver68/110lines;[source](primary/diagnostic_driver_source.txt) and [pre-execution allocation](primary/execution_allocation.json) bind exact source/input hashes. Original accepted adapter,auditor,reader,stage and loading-GC policy remain unchanged. [Receipt](primary/prefill_dp0_rank0_timing.receipt.json) contains exact2026-10-24 UTC times;filenames retain the supplied Hong Kong task-date label.

Expected diagnosis: establish whether caller/group construction,global sort,union recurrence or GC overlaps consume the censored batch. A complete final JSON would still be diagnostic-only. Expected constraints60s/actual30GiB/max2ranks;actual cgroup limit32212254720bytes and initialfailcnt0. Only one rank was executed,no retry.

| Segment or observation | Actual | Interpretation |
| --- | --- | --- |
| Loading |0.492→30.315s,29.823s |Completed with original loading-GC policy |
| Event pass |5062740events,1215919kernels;done37.026s |Completed event census |
| Correlation |Done39.952s |Original source stage |
| Kernel pass |39.952→54.806s,14.854s |Completed stage;index exists |
| Caller groups argument |54.806→55.267s,0.461s |Separate from function timer |
| Batch |1215919events,240groups,5331778memberships |Observed at55.267s |
| Membership/count/sum construction |Start55.267s,no done |Censored;observed open through58.619355s,at least3.352355s |
| GC within construction |55.771496→56.813963 and57.480575→58.619355 |Two completed gen2scans,total2.181248s,collected0 |
| All observed gen2collections |17completed,total5.053709s |No open collection record at timeout |
| Sort/union |No start markers |Durations unknown,not assumed dominant |
| Nominal budget at batch start |60−55.267=4.733s |Projection only,not exact stage-clock kill timestamp |
| Final JSON/useful steps |Absent/unmeasured |No R5 acceptance or inherited39useful-step result |
| Sparse RSS |6367.1MiB maximum recorded |Not actual peak RSS |

[Analysis](primary/diagnostic_analysis.json) stores exact timings,paired GC rows,source inference and limits. [Closed archive binding](primary/diagnostic_archive_binding.json) verifies seven receipt/cgroup/GC/stack/stage/stdout/stderr files after the child stopped. The5,383,865-byte index remains in scratch;its existence/hash is recorded without another complete-index validation. No partial data are upgraded to full metrics.

The group-construction and GC hypotheses are supported. Per-membership union cost is untested because construction never reached the sort. Source inference: memberships.setdefault(id(event), []).append(state) adds per-event mutable lists and eagerly allocates default lists across5,331,778memberships. Avoiding those allocations targets the measured construction cost;the achievable end-to-end speedup is unverified.

The separate D-V4-P0-AUDIT-METRICS-GROUP-CORRECTION allocates a fresh adapter based on global stable order and per-group integer order positions,original-order count/sum and the exact union recurrence. Owner scope220source/200test/100Markdown;one owner and one primary unit invocation,two fresh complete controls,then eight conditional affected ranks. The remaining seven start only after a complete fresh pilot. All60s/30GiB limits and original artifacts remain unchanged.

This measures the offline auditor's computation,not serving CPU overhead. No GPU job,old driver/control/tiny rerun,source/raw/package edit,native ownership,dependency,timing/non-perturbation,cost or reference admission occurred.
