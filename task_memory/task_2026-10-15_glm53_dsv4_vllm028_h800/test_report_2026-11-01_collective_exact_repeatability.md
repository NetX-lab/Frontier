# Collective input qualification for exact-point consumers (D-COLLECTIVE-EXACT-REPEATABILITY)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Created: applied YC's exact-point repeatability rule to the F2 P2-32 and P2-64 replays; both PASS. |

## Rule

Decision D-COLLECTIVE-EXACT-REPEATABILITY (requirements.md, commit `5fbee636`). For a consumer that prices every collective call at an exact measured point, the collective input qualifies when:

1. A8 exact admission holds: every call is priced at an exact measured point, with zero interpolation and zero violations.
2. The time-weighted maximum deviation of any pass median from the pooled median, over the points the replay uses, is at most 10%. A point measured in a single pass fails this condition.

This replaces the leave-one-out curve test for such consumers only. Consumers that interpolate between points still need the leave-one-out test.

## Environment and command

- Workspace `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, source base `5fbee636`. Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6. CPU only; no platform operation.
- Script `tests/analysis/v4_collective_exact_repeatability.py` (SHA `1e6281fe…`), committed in `384d1552`. The run used the same file content before that commit.
- It rebuilds the uniform and exact unequal-SP points from the measured summary with the measured backend's own row classification (`MeasuredCCBackend._retained_sp_rows`, `_ROW_CURVES`, `_EXACT_SP_CALLS`). Each rebuilt pooled median must equal the price the A8 census recorded, so the check covers exactly the points each replay used.
- Inputs: `data/profiling/network/h200_dgx/summary.csv` (`1a7c701f…`); the census and A8 receipts under `h3o_operator_costs/f_comparisons_20261101/` (`f2_p2_32_collectives.json` `13c7fde5…`, `f2_p2_32_a8.json` `a2c83c6a…`, `f2_p2_64_collectives.json` `5ff7bed6…`, `f2_p2_64_a8.json` `86311e55…`).
- Exact command, input and output hashes: `h3o_operator_costs/collective_repeatability_20261101/command.json`. Result `result.json` (`2becc838…`), console output `result.log` (`78b48915…`). Execution start and end times were not captured.

## Results

| Replay | A8 calls / exact calls / violations | Census rows exact | Points used | Single-pass points | Predicted collective time | Time-weighted max pass deviation | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| P2-32 | 6,310,733 / 6,310,733 / 0 | yes | 44 | 0 | 351,009 ms | 5.46% | PASS |
| P2-64 | 9,003,672 / 9,003,672 / 0 | yes | 38 | 0 | 764,471 ms | 0.78% | PASS |

Every rebuilt pooled median equals its census price (otherwise the script stops with an error). Exit 0.

Points with a pass more than 10% from the pooled median (all are kept; none is removed or reweighted):

| Replay | Point | Predicted time | Max pass deviation |
| --- | --- | ---: | ---: |
| P2-32 | decode reduce_scatterv/ep/16, 262,144 B | 47,180 ms | 17.25% |
| P2-32 | sp_reduce_scatter/tp/8, 32,768 B | 618 ms | 21.28% |
| P2-32 | sp_reduce_scatter/tp/8, 16,384 B | 369 ms | 15.39% |
| P2-32 | all_reduce/tp/8, 131,072 B | 230 ms | 19.57% |
| P2-32 | sp_all_gather/tp/8, 65,536 B | 168 ms | 17.05% |
| P2-32 | all_reduce/tp/8, 32,768 B | 15 ms | 12.67% |
| P2-64 | sp_reduce_scatter/tp/8, 32,768 B | 1,876 ms | 21.28% |
| P2-64 | sp_reduce_scatter/tp/8, 16,384 B | 1,195 ms | 15.39% |
| P2-64 | all_reduce/tp/8, 32,768 B | 46 ms | 12.67% |

The largest points: P2-32 decode all_gatherv/ep/16 at 16,528 B (105,604 ms, 7.38%); P2-64 decode all_gatherv/ep/32 at 8,264 B (267,865 ms, 0.58%).

## Conclusions

- Observation: under YC's rule, the collective input qualifies for both F2 replays. P2-32's heaviest unstable point (decode reduce_scatterv/ep/16, 17.25%) holds 13.4% of its predicted collective time and is inside the time-weighted 10% limit.
- The leave-one-out A6 curve failures stay recorded and still apply to any consumer that interpolates.
- This settles only the collective input. NIXL and memory qualification and the P2 recheck remain open.

## Limits

- The check covers the points these two replays used. Another workload that touches other points needs its own census and check.
- Pass medians pool all scenarios of a pass for a point; the rule does not test scenario-level differences.
- The rule bounds pass-to-pass repeatability, not the difference between the measurement setup and a production serving run.
