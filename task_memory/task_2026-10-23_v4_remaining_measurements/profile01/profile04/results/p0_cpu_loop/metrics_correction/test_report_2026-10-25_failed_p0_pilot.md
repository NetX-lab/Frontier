# Failed once-only P0 metrics pilot

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Archived the sole affected-rank pilot timeout, exact closed logs and complete-index verification without final-metrics acceptance. |

Verdict: **FAILED_P0_PILOT_NO_FINAL_JSON**. R5 remains24/32;the first correction's affected allocation used1/8, and the other seven ranks were not executed. The eight original60s timeouts and prior diagnostics remain immutable. Report date uses the task's Hong Kong label;execution timestamps remain2026-10-24 UTC.

## Exact command, configuration and expectations

- Driver: `/usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_affected_20261025_01/run_affected.py pilot`;driver77/110lines,SHA25637eb154a470f130bf825fc455f09b14c5710d871dce10b6816fb56058dca3813. [Allocation](primary/affected_allocation.json), [source binding](primary/affected_driver_binding.json), [retained source](primary/affected_driver_source.txt). Do not run/import this spent driver again.
- Child: `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_metrics.py torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842834745705872.pt.trace.json.gz /data/ycfeng/tmp/glm53_dsv4/p0_metrics_affected_20261025_01/prefill_dp0_rank0`. Python3.10.6;source/controlcommit0d0fe648,adapterSHA2566dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a;frozen auditor/reader/stage/loading-GC policies unchanged. [Child receipt](primary/prefill_dp0_rank0.receipt.json) records all three actual input paths/hashes and UTC start/end.
- Required PASS: complete final audit JSON and clean full index within60s, actual30GiB,max2rank concurrency, all identity/admission fields and at least three useful steps using the frozen predicate. Exit0 alone would be insufficient.
- Accepted source/controls remain source-only evidence:120/200source,175/180tests,primary8PASS and two full controls exact142567/570964kernel rows. Their invocation allocations are spent;no run_controls.py import or rerun occurred.

## Observed results

| Observation | Actual | Verdict or limit |
| --- | --- | --- |
| Child / driver |124 /1;60.329613s enclosing |FAIL full-output deadline |
| Actual memory controller |32212254720bytes/30GiB;initialfailcnt0 |Applied scope;initial usage is not peak |
| Load / kernel start / kernel end |31.143s /41.247s /56.015s |Recorded stage progression |
| Complete kernel index |1215919rows/155streams;clean gzipEOF |Index-only completion |
| Index statuses |unowned11385;unique_leaf138134;captured_segment_unknown1066400 |Attribution gaps remain explicit |
| Exact decompressed index comparison |Equals prior loading-GC index;SHA256678fb91782cc85a5068cf6bccbd51749562449abdfc90c628c0a084dddf68667 |PASS index equivalence only |
| Missing tail stages |per_stream_done,steps_done,result_built,output_written absent |Batch substage costs not yet measured |
| Final audit JSON / useful steps |Absent / not measured by this pilot |No inherited39-useful-step claim |
| Sparse observed RSS |6366.2MiB maximum among observations |Not a peak-RSS measurement |
| Remaining seven ranks |Not executed |Pilot PASS precondition not met |

[Failure/index validation](primary/pilot_failure.json) records15checks PASS in6.500774s;this was readonly index inspection,not another trace/audit invocation. Originalindex `/data/ycfeng/tmp/glm53_dsv4/p0_metrics_affected_20261025_01/prefill_dp0_rank0.kernels.jsonl.gz` remains5,383,858B,SHA2561bc09069db679759bce5f40d65c3ddc3c2e975f517ef19e2897bce2eee79f875. The retained loading-GC index and all original failures remain unchanged. Selected receipt,cgroup,GC,stack,stage andstdout/stderr bytes are archived by [closed archive binding](primary/pilot_archive_binding.json).

## Diagnosis frontier and preserved errors

The accepted adapter inserts batched_metrics before the original per-stream loop. Therefore the missing per_stream_done does not locate a failure inside that loop;the unmeasured batch includes group/membership construction,one global stable sort and union updates. New D-V4-P0-AUDIT-METRICS-TAIL-DIAGNOSIS records one sparse timing diagnostic;it does not authorize another correction,complete control or affected-rank invocation.

The first index validator used outputs[].path against the old files[path] receipt and raisedStopIteration. [Original failure](primary/pilot_index_verification_failure_01.json) remains;the corrected81-line index-only source/15checkresult are retained. A separate newline preflight assertion failure did not change the correctly written driver. No full pilot was retried after either auxiliary error.

Final metrics,hierarchy/leaf/per-stream fields,useful-step count and full32acceptance remain unavailable. Index equality,CPU controls and the independent standalone scope GPU acceptance do not provide whole-DP,transport,trace-ID,captured serving ownership,dependencies,cost or reference admission. No GPU operation or immutable source/package/raw-input edit occurred.
