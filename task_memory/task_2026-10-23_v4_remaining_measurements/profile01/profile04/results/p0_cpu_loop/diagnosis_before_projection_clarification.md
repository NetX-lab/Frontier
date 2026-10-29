# W128-P0-AUDIT-CPU-LOOP: P0 rank0 CPU audit timeout

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | One sparse-boundary 60s/30G diagnosis of retained profile04 P0 rank0; no algorithm change. |

## Symptom, loop, control

All 8 P0 ranks exited 124 at the 60 s limit while 24 other ranks completed (`profile04_trace_of9lmcmi/prefill_dp0_rank0.status.json`). This is CPU budget, not native failure: profile04 is 4 Succeeded and trace/profile bytes are intact. Input (gpu-h800-0030, prefill dp0 rank0): 142,288,791 B gzip = 2,221,133,654 B plain, 5,062,740 events, 1,215,919 kernels, 512,220 runtime/driver calls, 63,944 annotation ranges, 512 steps.

`p0_audit_cpu_20261024/stage_audit.py` (128 lines) re-runs the pinned auditor (`7ed0ea45`) and reader (`fe4c3f82`) after inserting 19 statement-boundary observations and two `idx % N` loop samplers (21 points). No statement, work or predicate is edited, both sources are hash-asserted unchanged, outputs are fresh. GREEN control on completed decode dp1 rank0: exit0 in 6.69 s (retained 6.7 s), `payload_equal=true`, byte-equal kernel rows, 61/61 useful steps.

## Measured boundaries (single 60 s / 30 G run, exit 124)

| Stage | Window (s) | Cost (s) | Note |
| --- | --- | --- | --- |
| metadata + header | 0.02–0.45 | 0.43 | profile 39.2 MB + inventory |
| trace load | 0.45–45.46 | **45.01** | 49.3 MB/s, 75% of budget |
| annotation name count | 45.46–46.06 | 0.60 | 63,944 names |
| event categorization | 46.06–53.92 | **7.86** | 1.55 us/event |
| correlation | 53.92–55.85 | **1.93** | 3.8 us/call |
| kernel/JSONL loop | 55.85–kill | ≥4.61 | incomplete |

Last observed RSS 6.26 GB against the 30 G cap: time, not memory, fires. Last completed stage is the correlation pass; the kill is inside `for idx, kernel in kernels:`, which reached index 1,550,000 at t=56.335 s (338k idx/s) and then made no sampled progress for 4.12 s. The partial index (56,857 rows, first index 1,386,499, last 1,541,086) is truncated and is not a kernel count; the baseline started at the same index and flushed 71,638 rows.

Anomaly: the event pass costs 0.19–0.71 s per 500k indices except one 3.586 s interval (3.5M→4.0M), and the kernel loop matches the control's rate (290k vs 338k idx/s) until it stops dead — two discrete multi-second stalls, not a smooth slowdown. Pre-kernel work alone is 55.85 s; at the control rate the remaining 3.51M indices need 10.4 s plus ~4 s rollup, so the projection is ~70 s. Gzip-only control reads (392–425 MB/s) show the 45 s load is JSON-dominated (~40 s, extrapolated).

## Hypotheses, ranked (falsifiable)

1. **Cyclic-GC pauses.** Gen-2 collections over the 5.06M-event / 6 GB graph cause both stalls; the 0.73 GB control has none. Falsifier: `gc.callbacks` shows no gen-2 event at t≈51.7–53.9 or t≈56.3–60.4, or stalls persist with GC frozen.
2. **One quadratic owner check.** A kernel near index 1.55M gets a large `owners` set and the nested `any(...)` innermost test (`audit.py:99-108`) becomes O(n²). Falsifier: candidate counts stay 1 (fragment shows 1 up to 1.541M) and that index passes quickly with GC frozen.
3. **The 60 s ceiling is unreachable without load work.** Pre-kernel stages already consume 55.85 s and the load is 45.0 s at 49.3 MB/s. Falsifier: a load-equivalent measurement well below ~40 s.
4. **Lane density, not just size.** Versus completed prefill dp1: 2.75x events, 3.88x calls, 5.24x annotation ranges for 2.13x kernels, so a linear audit needs ~80–95 s there. Falsifier: per-unit costs equal the controls — already true for the event pass and early kernel pass, so density alone does not explain the timeout.
5. (Control) The 21 observations did not create the failure: identical first kernel index, same row/index ratio, ≤~2% inferred wall overhead; the baseline's pre-kernel time is unrecoverable.

## Smallest next experiment (separate allocation)

Re-run this exact instrumented copy once on the same input, 60 s / 30 G, adding only `gc.callbacks` start/stop + generation records and `gc.freeze()` after load, plus the decode-dp1 equivalence control. Uniform progress and a shorter load confirm H1 and refute H2; an identical stall at that index refutes H1 and points at H2; an unchanged H3 means load cost must be addressed regardless.

## Limits

One P0 rank only; the other seven are inferred from near-identical trace sizes. No full kernel/loss count or P0 acceptance follows. The gzip/JSON split is extrapolated from decode controls. GC and page-cache state were not instrumented, the diagnostic ran alone while the baseline ran two ranks at a time, and no correction, timeout change or GPU retry is part of this record.
