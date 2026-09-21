## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Established the first DP inversion with fresh same-run routing evidence and two unchanged-predictor DES controls. |

# DP assignment and admission RCA

The first reproduced DP-owner divergence is caused by replaying an engine-enqueue ordering as frontend routing ordering. The exact weighted-load selection formula is aligned. This finding is established for the first close pair in the fresh run; it does not assign every old full-case mismatch to that mechanism. Admission/KV progress additionally diverges because the current one-arrival simulation does not retain the full route-to-engine visibility and engine-step boundaries.

## Fresh evidence envelope

- Case remains pf4096_dc1024, 100 formal plus 300 drained warmup requests, H200/step_main, TP4/DP2/EP8, uniform MoE routing, prefix OFF, eager, collective_sim/nvlink_analytic, current Frontier production code. GPU provenance is owned by analysis/h200-rca-01 manifests.
- Same-run source: analysis/h200-rca-01/runtime/batch. Batch and route diagnostics are isolated from clean E2E acceptance. Canonical validated evidence is same-run-01/same_run_join.json, fresh_same_run_evidence.json, and fresh_boundary_controls.json. Provisional paths in historical command receipts identify the actual executions; their trace and mapping bytes equal the finalized same-run-01 files and do not define a separate evidence generation.
- Strict analyzer PASS:400unique route/enqueue/client joins;100formal requests each appear in one prefill on each TP rank of exactly one selectedDP; scheduler token maps match actual batch maps. Current Frontier selector reproduces all400observed vLLM choices when given their recorded counts.399successive route states additionally match snapshot_receive plus local increments; the first route establishes initial observed state.
- Two real DES diagnostic prefixes consume independently normalized enqueue or route timestamps from this same run. Operator predictions, config and scheduler code are identical. They reuse only the current-task trained caches and stop after8routes. No E2E metric is produced.

## First DP divergence: observed cause

vLLM route5 occurs at3906952.754217817s with counts[[0,3],[0,2]]. Scores are3and2, selectingDP1. Route6 occurs2.023212146mslater with counts[[0,3],[1,2]] after the local waiting increment; scores are3and6, selectingDP0.

At the engine boundary, enqueue6 occurs at3906952.772985758s and enqueue5 at3906952.773034287s. The order is reversed by0.048529357ms. These are timestamps in the same host monotonic domain, joined by the exact server request ID. Server arrival ordering is not used.

| Control | First8 DP owners matching vLLM | First7 prefill member/token multisets matching vLLM | First close pair |
| --- | ---: | ---: | --- |
| Same-run enqueue-input DES | 6/8 | 5/7 | client6 -> DP1; client5 -> DP0 |
| Same-run route-input DES | 8/8 | 7/7 | client5 -> DP1; client6 -> DP0 |

Both DES runs reach the close pair with the same selection-state sequence:[[0,3],[0,2]] then[[0,3],[1,2]]. Changing the arrival observation boundary alone fixes the first owner swap without changing operator duration. Therefore accumulated operator error is not a necessary cause of this specific first divergence. The old case exhibited the same Frontier counts and pair-owner pattern, but its vLLM routing state was not logged; the fresh run supplies the direct causal evidence rather than retroactively inventing it.

The current workflow performs frontend selection at replayed QUEUED time. Pinned vLLM selects before sending ADD; Scheduler.add_request emits QUEUED only after EngineCore drains its input queue. EngineCore drains input between completed steps. QUEUED therefore contains execution-dependent delay and is not an external arrival stream suitable for rerouting requests independently.

## Admission membership versus execution progress

The low old full-case membership agreement(5/100) is not95independent admission defects. In the old artifacts, swapping only the already-routed client5/client6 labels restores all9later member vectors forclient7through15. Persisting decode owners propagate an early routing difference through many later batches.

Even when the fresh route-input control aligns the first7member multisets, it does not align decode/KV history. In the batch containing request1prefill:

| Side | Prior scheduled request0 tokens | Pure decode steps after its prefill and before this batch |
| --- | ---: | ---: |
| Fresh vLLM | 4096 | 0 |
| Enqueue-input Frontier | 4097 | 1 |
| Route-input Frontier | 4098 | 2 |

The Frontier enqueue-input run admits request0decode at65.790076904ms, sees request1enqueue at84.983801935ms, and admits the mixed batch at117.782847453ms. In the route-input run, request1appears at118.210053071ms, just after another pure decode starts at117.782847453ms; its mixed admission moves to169.775618002ms. Its absolute earlier engine visibility is missing because routing and engine visibility are still collapsed into one simulator arrival event. Route-input is thus a useful causal control, not a sufficient production repair.

Cumulative scheduled tokens are an observed batch-ledger quantity, not a direct KV-occupancy sensor. No preemption was inferred merely to reconcile a mismatch. Member comparisons above are multisets; they do not claim that worker input-buffer order equals scheduler diagnostic order.

## Complete step boundary matters before the next admission

Frontier GlobalBatchEndEvent immediately returns ReplicaScheduleEvent at the same simulation timestamp. Pinned vLLM batch CUDA timing ends before compute_logits, sampling and synchronous bookkeeping. The worker returns through the executor, scheduler.update_from_output runs, output/load state is published, conditional DP finish synchronization runs every32steps, and the next loop drains input before scheduling again. These omitted or separately represented intervals include GPU and host/transport work; they are not all CPU overhead.

The old measured-duration substitution is decisive as a limited inequality:77.125022888ms(first diagnostic CUDA forward) is still earlier than81.208001822ms(second clean engine queue arrival) by4.082978934ms. Replacing only that forward duration would therefore still allow an immediate Frontier decode. It cannot alone remove the old first admission/KV difference. The two old measurements are from different executions, so that inequality is not a same-run CPU estimate.

Fresh same-run timing corroborates the boundary distinction: first-to-second scheduler starts span83.452619147ms, while the first CUDA forward is80.335617065ms. Their3.117002081msdifference is aggregate work outside that CUDA scope over this engine-step interval, not a CPU-only measurement. First route-to-enqueue adds39.147422183ms before the first scheduler starts; the first enqueue-to-schedule interval is1.596956979ms. No wall-clock/monotonic timestamps are subtracted.

Across100formal requests, route-to-enqueue is mean37.555401111ms(range0.771284103–84.962751251ms); enqueue-to-first-prefill schedule is mean0.145530980ms(max1.696905121ms). This places most observed waiting before QUEUED, in contrast to treating QUEUED as an external arrival and adding a new simulated admission wait. These are diagnostic path intervals, never a constant correction to TTFT.

The first formal route sees stale frontend counts[[0,0],[0,1]]. No warmup request is scheduled after that route. Request-drained warmups do not prove that DP dummy-loop work and finish synchronization have quiesced; actual dummy invocation identity/timing is not recorded here. See analysis/first-forward-ep-contract.md for the independent source/boundary audit if present. Do not assign the39.147ms interval solely to CPU, stale counts, or dummy work.

## Scope and next evidence

- Proven: same-state selection parity, first close-pair order inversion, unchanged-predictor boundary control, and independent admission/KV-history divergence.
- Not proven: exclusive attribution of all22old DP differences, exact per-op contribution to later load-snapshot changes, or a complete host/worker/dummy breakdown. The fresh100request trace contains3route/enqueue inversion pairs(5/6,19/20,41/42); this alone cannot explain22old mismatches from another execution.
- A production correction must keep frontend routing visibility distinct from engine queue/admission visibility and preserve the agreed TTFT endpoint. Replacing the one arrived_at value with route time improves the owner prefix but worsens the second-batch progress mismatch in the measured control. No production patch, arbitrary timing scale, or forced observed-owner policy is applied.
- Full-case numerical calibration remains open. The operator lane owns first-forward operator/critical-path attribution and the new clean acceptance run must follow any approved correction.

## Deliverables and reproducibility

- same-run-01/same_run_join.json: full validated100formal route/enqueue/scheduler/batch join.
- fresh_same_run_evidence.json: compact first8records and actual first inversion.
- fresh_boundary_controls.json: paired real DES prefix comparison and full-trace delay/order statistics.
- fresh_enqueue_observation.json / fresh_route_observation.json: exact commands and raw prefix event observations.
- first_forward_admission_counterfactual.json: duration-only inequality and inspected source anchors.
- workflow_gap_table.csv and status.json: scoped workflow evidence states.
- ../../test_report_2026-09-08_dp_workflow_controls.md: exact commands, environment, checks and limits.
