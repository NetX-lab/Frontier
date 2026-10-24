# Group segment diagnostic: CPU allocator abort

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Archived the unique allocator-aborted diagnostic and preserved all unmeasured metric regions. |

The source/tiny gate is accepted at `e46cafe3`; the112-line timing source and114-line active control retain32 exact checks, actual30GiB and two independent zero-open-finding reviews. The original cgroup control's zero-check failure and its separately allocated repair stay immutable. This report records a different failure in the sole full segment diagnostic.

Environment: `/usr/bin/python3`3.10.6/GCC11.3.0, no conda; current worktree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Outer command:

```text
/usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_group_segments_20261025_01/run_segment_diagnostic.py
```

The frozen55-line driver SHA256 is `363b91db82763baf1527ad6fe97b1d5ffd0391da11e39949edefbebe37624b26`. Allocation SHA256 is `a1c85acf7bed4288777a16b1b53fb494080723a45842fbacbbff807020ca614d`. Its exact child command, input triplet and source pins are in `primary/prefill_dp0_rank0_segments.receipt.json`:

```text
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_group_metrics_timing.py torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842834745705872.pt.trace.json.gz /data/ycfeng/tmp/glm53_dsv4/p0_group_segments_20261025_01/prefill_dp0_rank0_segments
```

Expected: observe completed or timeout-censored global sort/order and group count/sum/sort/union segments, without changing expressions or numeric fields. Keep60seconds, actual32,212,254,720bytes, concurrency1 within max2, no retry. This is a diagnostic and cannot supply R5 acceptance.

Observed: start `2026-10-24T21:26:58.199458+00:00`, end `2026-10-24T21:27:38.533337+00:00` (2026-10-24UTC /2026-10-25HongKong). Command returncode-6/SIGABRT, outer driver1, enclosing40.333548916038126seconds; actual limit32,212,254,720bytes. This is an early abort rather than a60-second timeout. The original generic receipt statusCENSORED_DIAGNOSTIC is preserved; `primary/diagnostic_failure.json` classifies FAILED_CPU_SIGABRT_BEFORE_METRICS.

Stderr: `malloc(): unsorted double linked list corrupted`. Heartbeat contains `Fatal Python error: Aborted`, reporting the frozen auditor's `trace` at line96 during the kernel/index phase. The transformed AST retains original filenames and many locations; this reported Python line does not identify the native C allocation or earlier corruption origin. No native C stack is retained.

| Stage/field | Actual observation | Verdict/limit |
| --- | --- | --- |
| Loading | start0.520s,end29.236s | Completed28.716s;not a comparative performance claim |
| Event pass | completed36.072s,5,062,740events,1,215,919kernel candidates | Pre-kernel census only;not completed attribution/index |
| Correlation/kernel start |39.016s | Correlation completed;kernel pass entered |
| Last observation |kernel progress39.935s,RSS6266.1MiB,originalevent index1,700,000 | Snapshot,not peak memory or completed kernel count |
| Kernel pass end |Absent | FAILED before closure |
| All new metrics boundaries |0 | Allthree timing hypothesesUNMEASURED |
| Final JSON/useful steps |Absent/UNKNOWN | No R5 qualification |
| Partial index |478,079compressed bytes;hashf3b9ca8f760c1426dc5585662146def8d049f913c39adc2b76e284daff76dc23 | Hash-bound only;EOF/count/attributionUNVERIFIED,no decompression performed |

Primary streamed-rehashed every closed output and unchanged source pin, then copied the six compact logs/cgroup/observations and receipt into `primary/`; the index stays at the exact scratch path. `diagnostic_failure.json` retains exact paths/bytes/hashes and full sparse observations. These reads did not parse a raw trace, invoke an old driver/control or rerun the failed diagnostic.

`abort_runtime_metadata.json` reads parent/usrbinPython3.10.6 executable SHA256dbb04ace02303fa5cb3a6d1aa0e7a180d8dd23d3257cc6a0b155d0ee219432a2, selected non-secret environment fields and local core configuration. ParentRLIMIT_CORE is0/-1, core_patterncore/core_uses_pid1;no core file was found in this exactrepo root. Childcorelimit/module list/corruption origin areUNKNOWN;parent metadata does not establish them. The inspected reader/stage/GC source uses standard-library loading/index paths;this does not exclude a native-library or runtime defect.

FAIL for planned metrics-segment evidence. Original diagnostic1/1 is spent, no retry. R5 remains24/32 and both sets of seven conditional ranks stayBLOCKED. CPU/source/fixture and scopeGPU1/1 acceptance retain their limits. Native serving ownership/dependency/timing/non-perturbation, whole-DP/transport/trace-ID and cost/reference admission remain open. Only the separately recorded read-only abort RCA is allocated;no new full control, affected rank, algorithm correction, environment change or GPU action follows from this report.
