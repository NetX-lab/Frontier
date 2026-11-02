# H4c design: speed of the step-cost path

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: profile evidence, the two options, the chosen change with its file list and line estimate, and the verification. |
| 2026-10-25 | Recorded the `batch_id` shift found by the P0 replay and linked the verification results. |

**Goal** (plan.md H4c, D-H4A-CHECKPOINT). Make the DSV4 step-cost path fast enough to rerun P0 replays during H4b-(b). Results must stay identical. H4a run5 took 8,094.8 s wall time for 128 formal requests.

**Criteria.**
- H4a run5's four metrics and per-request outputs are unchanged.
- The fidelity regression stays at 77/77. The unit suite, `tests/unit/test_e2e_gate.py` and the dummy smoke matrix are unchanged.
- The wall-time change is measured.

**Bounds.**
- At most 5 `frontier/` files and about 200 changed lines.
- No new shared config or schema field and no registry change.
- The DIAGNOSTIC label applies.

## 1. Where the time goes

Profile: the first 180 s of run5's input (`/data/ycfeng/tmp/glm53_dsv4/h4a4/profile_speed/profile.txt`, driver `profile_run.py` beside it). It covered 2.0 s of simulated time and 6,594 EP waves.

| Function | Calls | Cumulative s |
| --- | --- | --- |
| `sync_entry.enter_layer_sync` | 13,189 | 168.8 |
| `ep_wave_schedule.schedule_layer_wave` | 6,594 | 167.9 |
| `_prepare_moe_ep_wave_plan` → `ep_wave.prepare_moe_wave_from_inputs` | 6,594 | 166.4 |
| `expert_parallel.predict_ep_wave_phase_times` (16 EP lanes per wave) | 6,594 | 126.1 |
| `materialize_layer_ep_workload` (routed workload of the wave) | 6,594 | 37.8 |
| everything outside the EP wave plan | — | about 13.6 |

What the EP wave plan computes on this path:
- The step-cost predictor spreads one measured lane step over the layers as attention (`deepseek_v4.predict_step_cost_execution`).
- Each EP lane prediction asks for FFN work only (`include_attention=False`). It therefore gets an `ExecutionTime` with no operator times and all components 0.0.
- So every lane's five MoE phases are 0.0, and `calculate_ep_wave_timing` returns `wave_end = start + 0.0 = start`.
- The plan builds 16 lane batches, materializes the routed workload and runs 16 predictions to produce exactly these zeros.

## 2. Options

**A. One computation per step.** For a table-priced step, replace the per-layer chain with one computation of the step's end time. The chain is: the pre_moe room, the EP wave, the post_moe collective, then the next layer. The arithmetic itself is a short loop, `S_k = max(S_{k-1} + max_i a_i, LS + (k+1)/L · FL_max)`. But the per-layer chain also carries state that other events can observe between layers:
- the forward-sync step binding and closure per layer;
- stage-admission tickets and the promotion and restoration of owners;
- request layer counters;
- dummy-forward advancement and idle-engine wake-ups.

A must reproduce all of these at the same simulated times, or prove that nothing reads them. It also creates the `BatchStageEndEvent` earlier, which changes the tie order against other events at the same time and type. Estimate: more than 5 `frontier/` files and more than 200 lines, with identity not guaranteed by construction. It is outside the bounds.

**B (chosen). Price a table-priced wave without lane predictions.** The per-layer events and their state transitions stay. When the predictor prices steps from the table and no lane timings are requested, `prepare_moe_wave_from_inputs` builds the wave from zero phases for each EP participant. It does not materialize the routed workload or predict the 16 lanes.
- The barrier and wave-end timing still go through `calculate_ep_wave_timing`, with the same traces.
- The participants are the canonical `0..EP-1` that `materialize_layer_ep_workload` always returns.
- Events, event times, lane arrays and state transitions are the same as before by construction.
- Only two INFO traces are no longer emitted for these waves: the conservation trace and the workload trace. They describe a routed workload that this path does not price, and they are not written in run5's logs or metrics.
- The 16 lane batches are no longer built, so they no longer take numbers from the shared `Batch` id counter. Later batches therefore get smaller `batch_id` values, in the same order. The only output that shows these values is `frontier_transfer_ledger.jsonl` (found by the P0 replay, section 4).

When lane timings are requested (`metrics_store.ep_wave_reporting_enabled`: operation metrics, the stage-batch ledger or op-level tracing), the full path still runs, so EP wave reports stay as before. run5 has all three off.

Expected gain: about 166 of the 180 profiled seconds, so roughly 12×. After B, the per-layer events remain the main cost. Whether A is worth its cost is decided from the post-B profile (section 4).

## 3. Change

| File | Change | Lines |
| --- | --- | --- |
| `frontier/scheduler/utils/ep_wave.py` | `EPWavePlan` gains `participant_ep_ids`. A table-priced wave (V4 profile, `predictor._dsv4_step_costs` set, no lane timings requested) returns zero phases for `range(moe_expert_parallel_size)` and its barrier timing. The other waves are unchanged. | about +25 |
| `frontier/scheduler/utils/ep_wave_schedule.py` | Reads `plan.participant_ep_ids` instead of `plan.layer_workload.participant_ep_ids`. | about 3 |

Total: 2 files, about 30 lines. No config, schema or registry change.

## 4. Verification

1. **Per-wave equivalence on the real input (scratch).** Run the run5 input for a short simulated window. Wrap `prepare_moe_wave_from_inputs` so that every table-priced wave is computed both ways: once with lane timings requested, which forces the full path, and once without. Compare participants, the six phase tuples and the barrier timing exactly. Expected: every wave identical.
2. **P0 replay.** Run `h4a_p0_diagnostic/h4a4_driver.py` at the new commit. Compare `request_metrics.csv`, `metrics_ground_truth.jsonl` and `system_metrics.json` with run5, byte for byte except wall-clock fields. Recompute the four metrics with `h4a4_gate.py`. Expected: identical. Record the wall time against run5's 8,094.8 s.
3. **Regressions**, as in the H4a-2 record:
   - the full unit suite (baseline 134 failed / 6114 passed / 208 skipped / 108 errors, no new failure);
   - `tests/unit/test_e2e_gate.py` (40 passed);
   - `run_dummy_smoke_matrix.sh` (12/12);
   - the 77-case refactor-fidelity matrix against the t00 baseline (77/77 identical).
4. **Profile after B.** Rerun the 180 s profile. Record what is left and give YC the cost of option A against its remaining gain.

**Results** (commit `cefbbc9d`): all four checks passed; see [test_report_2026-10-25_h4c_step_cost_speed.md](../test_report_2026-10-25_h4c_step_cost_speed.md). P0 replay wall time 8,094.8 → 758.2 s.
