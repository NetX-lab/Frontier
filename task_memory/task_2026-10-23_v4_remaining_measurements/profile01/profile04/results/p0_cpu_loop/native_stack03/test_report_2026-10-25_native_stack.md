# Native-stack source and diagnostic verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Archived censored once-only GDB observation and measured group segments; native detection stack remains unmeasured and R5 remains24/32. |
| 2026-10-25 | Accepted file-based native control02 after127 primary checks; preserved stale launcher hash receipt and bound current source without rerunning controls. |
| 2026-10-25 | Archived failed inline synthetic argv transport and froze a separate identical-source file-based control. |
| 2026-10-25 | Retained late worker partials, verified77/118/74-line candidate sources and froze one fresh control before execution. |

Current primary control acceptance: PASS/spent1/1 for file-based control02, not a full-trace result. The earlier source/control checkpoints below are historical.

Component: profile04 retained P0/rank0 allocator-abort diagnosis. Environment: /usr/bin/python3 3.10.6, GCC11.3.0; no conda; /usr/bin/gdb12.1 and matching installed glibc2.35 symbols. Workspace: /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028. Main records and execution are primary-owned.

Primary source/AST inspection:16 explicit checks PASS, original10 dependency and6 runtime pins unchanged. GDB source77/90lines, synthetic control118/140, launcher74/110, owner notes40/70. Original late70/118/37-line worker files and their hashes are retained in primary/late_worker_delivery.json. No original source/control or full trace was invoked during inspection.

New control frozen in primary/control_execution_allocation.json with exact source/input/runtime hashes. Exact command:

```text
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/native_stack03/owner/control.py
```

Expected: two sequential synthetic cases inside one invocation; actual parent/child32212254720-byte limits; exact Python/selected environment/GC/PID metadata; inferior exit0 maps to GDB0, and observed stopped SIGABRT maps to GDB1 with a null inferior exit code. Stack/library/map capture must succeed only for the stopped inferior; no post-exit map, inferior-function call or core dump. Timeout60s, chosen concurrency1 within max2, no retry. Control remains NOT_RUN at this source checkpoint.

Full diagnostic remains NOT_RUN and requires control PASS plus frozen execution bindings. It reuses the same original Python/allocator/GC/loader/timing source and input; no performance cure, R5/native ownership or cost admission follows from this source gate.

Original control FAIL/spent1/1:outer return1,enclosing1.7025562380440533s,actual controller32212254720bytes. Both inferior exit1 status markers are retained; stderr reports SyntaxError on first line import\ and no synthetic metadata row. This does not validate intended exit0/SIGABRT and is not an allocator-abort observation. `primary/control_failure.json` binds all retained bytes.

Fresh control02 uses exclusive synthetic .inferior.py files. Its metadata function and synthetic source expression AST exactly match original; GDB77-line source and runtime/audit bindings are unchanged. Source121/140lines; `primary/control_execution_allocation02.json` freezes one separately allocated control at60s/actual30GiB/selected1. Command:

```text
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/native_stack03/owner02/control.py
```

Control02 used1/1 and returned0 at2026-10-25T04:02:08.390207Z (October25,12:02:08 Hong Kong); enclosing0.7063260679133236s. Primary inspection records127 checks/45file bindings PASS in `primary/control_acceptance02.json`. ParentPID2755664, exit0PID2755755 and SIGABRTPID2755824 each recorded32212254720bytes, exact Python3.10.6/allocator/GC metadata. Natural exit0 maps toGDB0/no captures. SIGABRT maps toGDB1/stopped inferior/null exit and successful bounded stack/shared-library/maps; GDB quit kills that stopped inferior. It is not natural exit-6. The missing `syscall-template.S` source-text warning is retained, while capture_errors is empty.

Read-only validation failed on the earlier `launcher_static_inspection.json` hash68f0d6dd. Actual unchanged74-line launcher hash7289592b matches the later `source_inspection.json`; `primary/launcher_current_inspection.json` records10 current static checks PASS and both receipt hashes. No source/control was rerun or imported for this repair. Both old inline control1/1 and repaired file control1/1 are spent. Full diagnostic remains0/1, conditional on frozen command/accepted control/current-launcher/input evidence.

Diagnosis skill limit: existing actual failed diagnostic is retained as the red-capable signal. Repeatability/minimization loops are skipped because each experiment has a once-only allocation; the next single GDB observation tests the recorded detection-path hypotheses without changing its inputs. No non-reproduction establishes a cure or timing/R5 admission.

Full diagnostic frozen before execution in `primary/diagnostic_execution_allocation.json`: original10 dependency/6 runtime pins, unchanged input triplet, accepted control02, current launcher and bounded sidecar review are bound. Preflight source/hash reads complete with no active rank process and no pre-existing diagnostic prefix. No findings within the sidecar scope. Exact controller command:

```text
/usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_native_stack.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/native_stack03/primary/diagnostic_execution_allocation.json
```

Exact resource/GDB command (JSON argv is authoritative):

```text
['sudo', '-n', 'systemd-run', '--scope', '-p', 'MemoryMax=30G', 'timeout', '60s', '/usr/bin/gdb', '--nx', '--nh', '--batch', '-iex', 'set debuginfod enabled off', '-iex', 'set auto-load off', '--command=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_native_stack.gdb', '--args', '/usr/bin/python3', '-u', '-B', '/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_group_metrics_timing.py', 'torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842834745705872.pt.trace.json.gz', '/data/ycfeng/tmp/glm53_dsv4/p0_native_stack_20261025_01/prefill_dp0_rank0_native']
```

This freezes existing diagnostic0/1, not a new invocation allocation. Early `-iex` disables remote symbols and auto-load before file load.

The staged all-file whitespace check reported five trailing-space lines in immutable GDB `owner02/sigabrt.stdout` anonymous mappings (42/49/51/57/76). Raw bytes stay unchanged; the maintained-file check excludes that exact raw log only. This is a retained-output formatting result, not a control failure.

Full native observation executed exactly once after local commit0d72782b at2026-10-25T04:07:51.603634Z; resource command returned124 at04:08:52.067372Z (12:08:52 Hong Kong). Controller returned1; enclosing60.46330882422626s includes teardown. Actual memory.limit_in_bytes32212254720. GDB stderr retains KeyboardInterrupt at timeout, stdout contains only thread startup. There is no native status marker or SIGABRT line. Inferior natural/signal exit and loaded native library map remain UNOBSERVED/UNRECORDED. This is CENSORED, not a successful stack capture or allocator cure.

`primary/diagnostic_result.json` records58 primary command/hash/limit/output checks and `primary/diagnostic_segments.json` pairs measured boundaries. Seven selected output files are byte-exact in primary/retained_native.*; compressed5,383,865-byte kernel index remains in the original scratch withSHAaa83c66a13eeaab727f4b5b5d48888b19bc9f1632ddffa6229612b7e42ba761c. It was not decompressed or counted; no EOF/full-index acceptance is claimed. No active inferior command remained on inspection.

| Measured region | Observed result | Limit |
| --- | --- | --- |
| Loading |31.373s |GDB timing only |
| Kernel pass |41.742→52.302s;155streams/status totals1215919 |Closed compressed hash only |
| Metrics batch |52.729s;240groups/5331778memberships |No end marker/finalJSON |
| Global sort/order |0.941s/0.459s |Descriptive, not non-perturbation proof |
| Group count/sum |160completed;1.069s |Partial240-group batch |
| Group sort/union |159completed;1.475s/2.945s |Firsthierarchy sortopen59.649s |
| Metrics GC |0callbacks |Priorpre-metrics scans remain retained |

The first two native detection-path hypotheses remain unmeasured. The non-reproduction branch is censored: no SIGABRT was observed before timeout, not proof it could not occur later. The global sort/construction and per-group sort/union hypotheses now have partial measured evidence in this distinct run; the aborted original run stays unmeasured. FinalJSON, per-stream tail, sync/steps and useful-step counts are absent. R5 remains24/32; no automatic retry, runtime/algorithm change or affected admission occurred.
