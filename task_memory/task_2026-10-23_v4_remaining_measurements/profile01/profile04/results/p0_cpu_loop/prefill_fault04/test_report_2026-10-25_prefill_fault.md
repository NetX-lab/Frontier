# Prefill native fault observation — 2026-10-25

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Allocated a distinct DP1rank0 SIGSEGV observation and verified new source/command pins; source review/preflightPASS; unique observation completed without reproducing the native fault. |

## Trigger and acceptance

Interval correction's new prefillDP1rank0 complete control failed with childSIGSEGV/-11,parent1/enclosing20.366353s/actual30GiB. Last29th observation19.943kernel_pass_progressidx1175000; no kernel_done, metrics_start or finalJSON. Earlier heartbeat sampled JSON encoding, while the fatal record has noPythonframe. Native fault PC/corruptionwriter remain UNKNOWN. The failed control is spent; no source/control acceptance or affected run follows.

`D-V4-PREFILL-CONTROL-SIGSEGV-NATIVE-OBSERVATION` allocates one native observation of this DP1input and candidate, keeping the original Python3.10.6/allocator/GC/loader/auditor/reader/input. The earlier P0 group-timing GDB diagnosis remains spent1/1. New complete-control/affected/GPU budgets0. Hypotheses and falsifiers are frozen in `primary/allocation.json`; this observation obtains a native signal/PC/stack or records clean exit/censoring without a cure claim.

Source caps: GDB78/90 and controller70/100. Static inspection confirms the accepted GDB callbacks/status/proc/cgroup behavior is exact; permitted changes are SIGSEGVstop/print/nopass, stopped PC instruction/symbol queries and the distinct marker. No function calls into the inferior, injected failure, source/GC/runtime change, core-limit change or automatic retry.

## Exact frozen command and environment

Controller invocation and every nested GDB argument are in `primary/execution_allocation.json`. Controller:

```text
/usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_prefill_fault.py /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/prefill_fault04/primary/execution_allocation.json
```

Resource command begins `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/gdb --nx --nh --batch`; auto-load and debuginfod are disabled before file load. It runs `/usr/bin/python3 -u -B` and the unchanged candidate with the full retained DP1 trace. Original environment is bound in the allocation; no Conda. Fresh prefix: `/data/ycfeng/tmp/glm53_dsv4/prefill_fault_20261025_01/prefill_dp1_rank0_native`. Selected concurrency1, maximum2. Child cgroup must observe32212254720bytes.

## Status and limits

Primary static source/pin checksPASS; independent48-line reviewNo blocking finding/14staticchecks/27pins. Primary preflightbindsreviewSHA5bbfc6b48dc766591ef9bb65f57807bb695cc31750759b0196dd43e22e9f6ec9; allpins/environmentmatch, noactiverankprocess/freshprefixabsent. Observation1/1spent; actual resultbelow. No new synthetic control is allocated: the accepted native GDB signal/capture seam is reused and its exact AST is statically bound.

At a stopped signal, preserve native instruction/symbol/stack/library/maps and original status; GDB quit kills the stopped inferior, which is not a natural signal exit. At timeout, preserveCENSORED and incomplete boundaries. Completed JSON/index outputs, if any, are diagnostic artifacts; they do not replace the spent failed control. Compressed index is hash-only in this observation, with no EOF/full-row acceptance. R5still24/32, fullR6/R7BLOCKED, corruptionwriterUNKNOWN, no native ownership/dependency/timing/cost/reference admission.

## Actual observation and primary archive

`primary/controller_execution.json` records controllerexit0/enclosing27.721293s. Actual resource/GDB commandexit0/enclosing27.419248s; native markerobserved naturalinferiorexit0, no stop signal/run error/capture error, batch_quit_actionnone. Actual child32212254720bytes. Native PC/stack queries were not executed because no signal stop occurred. This is **NO_FAULT_REPRODUCED**, not a cure or a replacement completecontrol.

`primary/diagnostic_result.json` records62/62 primary checksPASS, all27frozenpins unchanged, closed-output hashes and selectedbyte-exact archives. Boundaries: load0.196→11.642s; kernel14.858→19.812s; candidate metrics20.004→23.730s; output_written25.980/process_end27.017s. These are debugger/sparse-logging descriptive observations, not performance comparisons. Diagnostic finalJSON reports570964kernels/4usefulsteps. No whole JSON/floathex/index comparison was run; compressed index is only hash-bound, with EOF/full rows not independently verified.

Original prefillcontrol remainsFAILED_SIGSEGV/-11/no finalJSON/spent1/1, bothcontrols2/2spent. New nativeobservation1/1spent; no additional control/affected/GPU invocation. FaultPC/detectionstackUNOBSERVED, corruptionwriterUNKNOWN. R5still24/32 and finalsource/controlacceptanceBLOCKED. Current scope cannot declare failurecause or a safe behavioral correction from this clean debugger execution. Public primary-source precheck also found no verified match to the pinned local3.10.6scenario; no upstream fix/runtimechange was selected.
