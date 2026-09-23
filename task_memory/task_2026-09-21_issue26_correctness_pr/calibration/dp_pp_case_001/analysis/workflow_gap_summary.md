# Workflow gap analysis: dp_pp_case_001 (G4 native PP2 against G5 Frontier)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Post-review errata from workflow `wf_7606e14e-f10` (section "Post-review errata"): steady-segment natural history is not classified by any row; WG04 and WG06 hold under dummy timing only. No status changes. |
| 2026-09-24 | Human review G5-review recorded (C4 option a; S43 and S42 transferred to their own tasks). Frontier admission anchor corrected from `base_replica_scheduler.py:906` (the unified DECODE loop) to `:1052-1063`, the MONOLITHIC and PREFILL loop this case runs, here and in `semantic_alignment_table.csv` S23/S43 and `workflow_gap_table.csv` WG05. |

Entry `workflow-gap-analysis`, read-only. Table: `analysis/workflow_gap_table.csv`
(WG01-WG11). Status: `analysis/workflow_gap_status.json`.

## Inputs

- Native: `runs/groundtruth_clean/dpp-g4-20260923a/` (rjob `exp-0923-230103-591735`,
  4xH800, `vllm serve` DP2 PP2 TP1 EP2, `max_num_batched_tokens=20448`), extraction
  `PASS`: 1620 engine iterations, 92 reports and 92 receipts, 83 publications, 81
  frontend snapshots, 51 routes and 51 placements, 0 out-of-order receipts.
- Frontier: `runs/frontier_g4/` (commit 47d9190, inputs `frontier_g4.json` with
  `num_blocks=284775`, `dummy_execution_time_ms=0.8943`): `vllm_load_balancing`,
  `completion_reporting_control` and `round_robin`, 51/51 requests each, tokens
  conserved.
- Comparison: `analysis/g5_comparison/` (`compare_placement.py`, receipt there).
- Pre-change baseline: `runs/frontier_pre_change_d1a2a06/` records the constructor
  rejection of PP2 at `d1a2a06`.

## C3: iteration history, reports, snapshots and frontend counts

T1 replays the native receipts and routes through `VllmDPLoadBalancer`. All 51
routes (48/48 formal) get the native engine from the native counts (WG01). G3 gave
38/38 at PP1. With the inputs matched, count calculation, key grouping, snapshot
publication and frontend selection agree with the reference.

On the natural history each difference is labeled by its first cause:

| Row | Result | First cause |
| --- | --- | --- |
| WG02, WG10 | a-c: the long body routed after b3-b5, so the first forward had only 32-token requests | arrival/delivery order (client and frontend processing of the 40896-token body) |
| WG03, WG07 | d: engine 1 has no step-0 record, and the probe snapshot carries engine 1's previous-wave count | batch composition, key grouping (wave-start dummy forward, S42) |
| WG04 | 7/8 engine-bursts publish at the first admission, as the fixed policy does | MATCH |
| WG05 | the second admission waits until the engine blocks on its oldest batch; Frontier admits on arrival into the free slot | batch composition (S43) |
| WG06 | first snapshot counts `[[0,1],[0,1]]` in a-c | MATCH (published by the latch at 21-35 ms natively, by the collection wait at 50-56 ms in Frontier) |
| WG09 | the control keeps its reservations `[[3,0],[2,0]]` | count calculation (the control's expected failure) |
| WG11 | first applied output at 19.7-33.8 ms natively against 111-117 ms in Frontier | output readiness (declared dummy timing, S31) |

Operator comparison is not meaningful here. Timing is dummy (D-e), and the first
formal batches differ in composition (WG02, WG03). This case has no E2E gate, so
`op-supplement` is not required.

## C4: trace-qualified discriminating slice

| Burst | Native probe | Fixed | Control | Round-robin | Failed checks |
| --- | --- | --- | --- | --- | --- |
| a | e0 from `[[0,1],[0,1]]` | 0 | 1 | 0 | trace order; output before probe (33.8 ms, probe at 73.3 ms) |
| b | e0 from `[[0,1],[0,1]]` | 0 | 1 | 0 | trace order; output before probe (25.1 ms, probe at 97.9 ms) |
| c | e0 from `[[0,1],[0,1]]` | 0 | 1 | 0 | trace order; output before probe (21.3 ms, probe at 118.8 ms) |
| d | e0 from `[[0,1],[1,0]]` | 0 | 1 | 0 | one snapshot; schedule-time provenance on every engine (engine 1 from step 128 of the previous wave); output before probe (19.7 ms, probe at 104.8 ms) |

In every burst the fixed policy places the probe where vLLM did. The
completion-reporting control places it on the other engine because it holds the
reservations. No burst meets the pre-registered premise, so every T2 row is
`SCENARIO_NOT_REACHED` (WG08). The lane agreement is recorded but is not C4
evidence.

Two recorded facts make the premise unreachable together on this deployment:

- The frontend spends about 7 ms processing a 40896-token body. Shorter requests
  dispatched in that window route first (a-c).
- R1 needs the first two routes within about 0.5 ms. Otherwise the idle engine
  runs a wave-start dummy forward first (d, S42). With the long body among the
  first two routes, those two conditions conflict.

In a-c the short first batch then applies an output at 21-34 ms, which is before
any probe offset in the R3 window. A retune of spacing or offsets cannot put the
long body among the first two routes within 0.5 ms. The last budgeted GPU job was
therefore not used (plan section 18.20, retune rule).

## Candidate fidelity findings (outside Step 9 scope; not a repair authorization)

Reviewed 2026-09-24 (decision `G5-review`): each finding continues as its own
calibration-and-repair task, `task_memory/task_2026-09-24_s43_pp_empty_schedule_admission/`
and `task_memory/task_2026-09-24_s42_dp_wave_idle_forward/`.

- S43 / WG05: with PP>1, vLLM 0.10.2 appends an empty schedule and blocks on the
  oldest batch (`vllm/v1/engine/core.py:385-424`). Frontier admits whenever a
  stage slot is free (`frontier/scheduler/replica_scheduler/base_replica_scheduler.py:1052-1063`, report hook `:1061`).
  Confirmed by the G4 admission records. The later burst requests are admitted
  21.3-47.4 ms after the burst's first route in a-c and 129.5-311.8 ms in d.
- S42 / WG03: DP wave-start and idle dummy forwards (`core.py:1170-1216`) are not
  modeled. The d record gap agrees with that source reading, which is an inference
  because the dummy pass writes no record.

Either change needs the user's review before `workflow-repair` (contract
"Analysis Before Code Change").

## Post-review errata (2026-09-24)

Found by workflow `wf_7606e14e-f10` after the human review. Rechecked here. No
row status changes, and the review decision `G5-review` is unaffected: C4 stays
`SCENARIO_NOT_REACHED`, and S43 and S42 have their own tasks.

- The steady segment's natural history is not classified by any row. The C3
  table above covers the bursts only. Per-request lane agreement with native
  is 13 of 24 steady requests, and snapshot-count agreement is 6 of 24. Under
  the non-dummy fallback timing the figures are 19 of 24 and 7 of 24.
  Recomputed with
  `/data/ycfeng/tmp/issue26-correctness-pr/review_20260924/calib_nondummy/analysis/placement_diag.py`
  on `runs/frontier_g4/vllm_load_balancing` and on the fallback run. The first
  cause is not established. Timing is the likely cause (S31), because the
  agreement moves with the time source. That is an inference. The S43 and S42
  analyses have to classify this segment before either repair can close.
- WG04 and WG06 are MATCH under dummy timing only. In burst a, dummy timing
  puts the first MoE sync at 6.51 ms, so lane 1's admission at 0.25 ms reports
  under the same key (12) as lane 0's. Under the non-dummy fallback, lane 0's
  layer-0 EP wave starts alone at 0.242 ms, and lane 1 reports under key 13.
  Which key-grouping branch runs therefore depends on the time source (origin
  `test_report_2026-09-24_fix_review.md` section 5).
- Ground truth G4 ran in eager mode, with no CUDA graphs and no DP padding. The
  case says nothing about CUDA-graph mode.
