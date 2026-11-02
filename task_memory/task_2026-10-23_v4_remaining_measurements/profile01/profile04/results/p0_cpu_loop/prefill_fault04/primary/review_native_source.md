# Independent native source review: prefill DP1 SIGSEGV observation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Read both complete new sources, compare the accepted GDB source, and check the frozen one-shot command/pins without executing any harness. |

Component: `prefill_fault04`, D-V4-PREFILL-CONTROL-SIGSEGV-NATIVE-OBSERVATION.
Reviewer: Codex, independent source reviewer; no implementation, invocation or primary acceptance role.
Checkout: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; inspected HEAD `781e902c9df867ae4170209421d29b6a007eba3c`.
Verdict: **NO FINDINGS / NO SOURCE BLOCKER for the sole allocated DP1rank0 native observation**, conditional on the exact reviewed source/command bindings below. This does not accept the failed complete-control gate or any runtime result.
Authority: main task `requirements.md:353–356` and the new allocation. This is a fresh DP1 SIGSEGV question, not a rerun of the spent P0 GDB diagnostic; native observation ceiling 1, all new synthetic/complete controls, affected ranks and GPU ceilings 0.

`T` denotes `task_memory/task_2026-10-23_v4_remaining_measurements`; `P` denotes `T/profile01/profile04/results/p0_cpu_loop/prefill_fault04/primary`. Relative paths resolve against the checkout.

| Read path | Full reading range | SHA-256 |
| --- | --- | --- |
| `tests/analysis/v4_trace_audit_prefill_fault.gdb` | **1–78, complete** | `08f20aedbf55c9b79a2a0350b21a854ba873d6ecd871143fa8a5acc99051a2ea` |
| `tests/analysis/v4_trace_audit_prefill_fault.py` | **1–70, complete** | `3e1e725678e2ae61381a11a62f03eb9966f3288f9e04694147dce2b96fcf6390` |
| `tests/analysis/v4_trace_audit_native_stack.gdb` | 1–77, complete, read-only reference | `4d0b05a1c1aa4125d5e8faf93da0dbd0f8955e83a75446f88aff6eb0707217b3` |
| `tests/analysis/v4_trace_audit_native_stack.py` | 1–74, complete, read-only controller reference | `7289592b767038234b989f4b854a4e771b54ae980649235db9769439e0603cd2` |
| `P/allocation.json` | 1–184, complete JSON | `7b3452c6ac16a85eeb1476dd5851cda67d805d5d0ba8377b3c5cb2b3ad731452` |
| `P/execution_allocation.json` | 1–226, complete JSON | `a789afe51b4f60782c8ac1eba901c4feb1ac28f4642af4a2048211dcc34115f2` |
| `P/source_static_inspection.json` | 1–21, complete JSON | `f3f755371d7fbe4906177f9d55a66a49bb8d890a4c8014535831242d8513881e` |
| `T/profile01/profile04/results/p0_cpu_loop/native_stack03/primary/control_acceptance02.json` | 1–375, complete parsed JSON | `57d99ea115c7632a79da6260c863f0a94a915742eb24cb18a7109278ae69af10` |
| `T/profile01/profile04/results/p0_cpu_loop/metrics_correction03/primary/review_spec.md` | reused complete prior review context, 1–74; bytes hash-checked unchanged | `633b3931bdc60faf558c49281c901d8bc491dcf2fab6353c942bc65f9814e556` |

## Findings and source coverage

No actionable findings for the current frozen invocation. No additional control or source repair is required by this review. Primary must bind this report to its execution gate before spending the existing one-shot observation; no invocation was performed here.

- **Exact permitted GDB delta:** new line 13 adds `handle SIGSEGV stop print nopass`; line 57 prepends `x/i $pc` and `info symbol $pc` to the existing capture command tuple; line 76 changes only the marker to `V4_PREFILL_FAULT_STATUS`. Reversing those three edits restores the old GDB text byte-for-byte and its embedded Python AST exactly. No other signal/event/capture/exit/cgroup logic changes.
- **Stopped instruction capture:** GDB 56–62 runs read-only instruction/symbol, bounded all-thread backtrace, shared-library and mapping commands only after an observed stop while the inferior is alive; each command retains its own `gdb.error`. Lines 64–67 retain the accepted live-PID `/proc` limits/cgroup reads. `set may-call-functions off` at 11 remains unchanged; neither new PC query invokes an inferior function.
- **Signal versus exit:** GDB 31–50 preserves event registration/PID/exit/SignalEvent fields. SIGABRT and SIGSEGV both stop without signal delivery; 69–77 distinguishes stopped/unobserved/exited states, records the planned batch kill, emits the new marker, and exits GDB successfully only for natural inferior exit 0 without a stop or run error. A captured stop is not relabeled as a natural exit or successful control.
- **Frozen runtime and actual command:** `execution_allocation.json` selects `/usr/bin/gdb --nx --nh --batch`, disables debuginfod/auto-load before file load, uses the new GDB file and original `/usr/bin/python3 -u -B` with the unchanged interval candidate and exact failed DP1 trace. It wraps GDB in `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s`; ASLR, direct startup, allocator/environment, GC/loading and raw-input source remain unchanged. The controller argv names only the new controller and this exact execution allocation.
- **Exclusive one-shot outputs:** controller 28–39 checks the selected environment and every frozen pin, requires an absent dedicated directory, creates it once, and writes an exclusive `.started.json` before its sole `subprocess.run` at 41–42. Output/receipt writes use `open('x')` at 21–24/41; there is no retry, loop over invocations, old driver import or input rewrite.
- **Evidence and result states:** controller 43–58 retains wrapper returncode, elapsed time, distinct native marker and optional child cgroup; rejects duplicate markers and verifies 30 GiB when child evidence exists. It records every post-run pin-match boolean at 54, then classifies timeout 124 as CENSORED, observed signal as NATIVE_STOP, natural recorded exit as INFERIOR_EXIT, otherwise FAILED_UNOBSERVED. None is source/control/R5 acceptance; GDB/inferior details remain in `native_status`, and controller 66 propagates nonzero wrapper return as exit 1.
- **Retention boundary:** controller 59–64 hashes the closed output files, including any partial compressed index, without claiming decompressed EOF or row count. The fresh prefix is `/data/ycfeng/tmp/glm53_dsv4/prefill_fault_20261025_01/prefill_dp1_rank0_native`; its parent was absent during inspection. Runtime/pin/cgroup outcomes still require primary inspection; recorded post-pin booleans do not themselves establish acceptance.

## Static evidence and limits

Independent read-only calculation: `/usr/bin/python3 -I -B -` with inline standard-library `ast`, `hashlib`, `json`, `pathlib` code. No local source import, compile/exec, GDB launch, control, trace decompression, diagnostic or subprocess invocation occurred. Static parsing/diff/hash checks passed 14/14; all 27 execution pins match, including both new sources, original runtime/accepted source, failed-control archive and the prior reviews. Trace/profile/inventory bytes and hashes match the exact DP1 bindings. Gross caps pass: GDB 78/90 and controller 70/100.
The previous accepted synthetic GDB seam records 127/127 retained checks for natural exit versus stopped SIGABRT, PID, cgroup, capture and quit behavior. Those receipts support reuse of the unchanged mechanism; they are not a new SIGSEGV control or evidence that this observation reproduces the fault.
The prior prefill complete control remains FAILED_SIGSEGV: child -11, parent 1, enclosing 20.366353s, final observation 19.943s at original traceEvents index 1175000, no kernel-pass completion, metrics entry or final JSON. Its last periodic encoder sample was original auditor 119; Fatal reported no Python frame. This native observation seeks the missing detection instruction/stack and cannot retroactively change that control verdict or identify the corruption writer from the periodic sample.
A stop may identify only a detection instruction/library; no fault under GDB or timeout is also an allowed observation outcome. Debugger and capture alter execution timing. Actual SIGSEGV/native stack, child resource evidence, terminal status, post-run pin equality and output completeness are unobserved at this source-review stage. No arbitrary allocation/command, extra retry or new control is covered by this verdict.
The frozen execution status is `FROZEN_NATIVE_SOURCE_STATIC_PASS_REVIEW_PENDING`; primary source/report binding is the remaining operational gate. Source review identifies no actual blocker in the sole selected invocation. The review-scope checkpoint remains on that prerequisite and adds no executable work.
Old P0 GDB allowance stays 1/1 spent; complete-control allowance stays 2/2 spent. R5 remains 24/32; corruption writer UNKNOWN. No allocator cure, source/control acceptance, native ownership/dependency, cost/reference or GPU admission follows. Only this report is written; prior review and all sources/records remain unchanged, with no commit or spawn.
