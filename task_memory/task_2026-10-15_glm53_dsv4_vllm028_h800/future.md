## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Added the deferred profile04 P0 trace-audit ranks (D-TRACE-AUDIT-STOP). |
| 2026-10-21 | D-MEASUREMENT-FIRST stops expanded CUPTI investigation; retained current evidence and bounded future choices without waiving DSA coverage. |
| 2026-10-20 | Created: deferred work that is out of scope for the current tickets but must not be lost. First entry: the non-NIXL PDD DECODE-side KV reservation. |

# Future work

Deferred items only. Each entry states the cause, the affected path, why it is
deferred, and what would settle it. Work that is merely pending inside a ticket
belongs in that ticket, not here.

## Ordinary (non-NIXL) PDD DECODE reservation reserves one block too many

Status: open, deferred. Owner: unassigned. Recorded 2026-10-20.

**Cause.** Two reservation rules exist for the DECODE side of a PDD handoff.

- A request that arrives through NIXL (`reserve_kv_transfer_target`,
  `frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:162`) seeds the
  scheduler frontier to `num_prefill_tokens - 1` and reserves
  `ceil(prompt / block_size)` blocks. This matches vLLM's async remote load,
  which schedules `num_new_tokens = 0` and no lookahead, and sets
  `num_computed_tokens = prompt - 1` once the load lands (vLLM
  `v1/core/sched/scheduler.py:1082-1095`, `_update_waiting_for_remote_kv`).
  Fixed in M1 (`5ad40e0`).
- A request that instead reaches ordinary DECODE admission has no such seed, so
  the first allocation charges the post-prefill lookahead token and reserves
  `ceil((prompt + 1) / block_size)`.

**Why it matters.** When `prompt % block_size` is 0, the extra token crosses a
block boundary. A D lane holding exactly `ceil(prompt / block_size)` free blocks
would then reject the request that vLLM accepts, so under KV pressure Frontier
can be one block stricter than the engine.

**Why deferred.** The GLM target path (structural PD, `async_scheduling=true`,
DP2) uses NIXL, so the target configuration is already correct. The ordinary
path is the non-async connector case, which no current ticket covers, and
changing it moves admission decisions on a path with no PD fidelity case in
`tests/e2e/refactor_fidelity/cases.py`.

**What would settle it.** Decide which connector the ordinary path represents.
If it models a synchronous load, the current reservation is right and this entry
should be closed with that reason. If it models the async remote load, seed the
scheduler frontier the same way `reserve_kv_transfer_target` does and gate the
change on a DECODE block-accounting unit test at `prompt % block_size == 0` and
`== block_size - 1`.

Related: [w3_review_fixes_20261020/progress.md](w3_review_fixes_20261020/progress.md)
open item 2; [m2_m3_design.md](w3_review_fixes_20261020/m2_m3_design.md) for the
NIXL path that was fixed.

## Expanded CUPTI graph diagnostic is stopped; DSA coverage remains required

Status: stopped/deferred by YC's D-MEASUREMENT-FIRST. No active investigation owner or fourth GPU authorization. The DSA compiled/FULL coverage requirement itself remains in H3/H4 and is not deferred out of formal acceptance.

Retained evidence: `h200_graph_probe_attempt3_20261021/test_report_2026-10-21_attempt3_execution.md` and `rca/rca_pending_2026-10-21.md` inside that directory. Three GPU attempts did not qualify source-to-exec node attribution. Attempt3 localized the crash to an ID query during instantiate; a resource-domain callback payload interpretation mismatch is only a candidate cause, with conflicting retained header descriptions and no retained authoritative sample resolving them.

Concrete choices for the next DSA decision:

1. **Current selected route:** preserve the blocker and stop this investigation while independent H3 measurements, GLM H5 and V4 advance. Additional CUPTI GPU cost: zero. Affected DSA non-dummy/accuracy acceptance remains unavailable; no eager/event-only replacement is admitted.
2. **Conditional future repair:** if this exact blocker is the remaining critical path, request a bounded review of the recorded payload candidate (`CUpti_ResourceData.resourceDescriptor` versus a direct graph-data cast), supported by an authoritative CUDA13.0 sample and focused CPU controls before any native check. A proposed same-image one-H200/600-second payload would cost at most1/6 GPU-hour for the payload plus startup. It requires new explicit approval; it is not submitted or implemented. Full-ID correspondence, scope attribution, compiled/first-tail/TP coverage still need their original evidence even if the crash is repaired.

Do not expand this into a tracing framework or a runtime-stack migration under the current work order.

## profile04 trace audit: P0 prefill ranks 0-7 have no complete output

Status: deferred by D-TRACE-AUDIT-STOP. Owner: unassigned. Recorded 2026-10-25.

**Cause.** The R5 per-rank trace audit of profile04 finished 24 of 32 ranks (P1, D0 and D1). For each of the eight P0 ranks, the audit process hit the 60 s / 30 GiB inspection limit and wrote no final JSON. The later correction attempts in `../task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/` (metrics corrections, native-stack and prefill-fault observations) did not produce a complete P0 output either.

**Affected path.** Step costs for the P0 prefill lane (DP0) and the R6/R7 audit steps that need all 32 ranks. The 24 complete outputs are bound in `../task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/review_preparation/retained24_binding.json` (useful steps P1/D0/D1 = 4/61/61 per rank).

**Why deferred.** The P0 rank outputs do not block the first diagnostic P0 prediction (H4a). P1 is the other prefill lane of the same topology and serves the same workload.

**Restart condition.** The H4a error attribution shows that the prefill lane cost is a leading error source and that P1 cannot stand in for P0. YC then approves one proposal that states its own resource limit; the 60 s / 30 GiB limit applies to inspections, not to this audit.
