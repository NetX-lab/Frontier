# Test report: H4c, speed of the DSV4 step-cost path (2026-10-25)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: per-wave equivalence, P0 replay identity and wall time, regressions, and the profile after the change. |

**Scope.** Commit `cefbbc9d` ("Skip EP lane predictions for DSV4 table-priced waves"), parent `3373066b`. Two files: `frontier/scheduler/utils/ep_wave.py` and `frontier/scheduler/utils/ep_wave_schedule.py`, +46/−34 lines. Design: [h4c_step_cost_speed/design.md](h4c_step_cost_speed/design.md), option B. The label stays DIAGNOSTIC.

**Environment.** Local host, 64 CPUs. Python 3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`, `PYTHONPATH` set to the checkout, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`. Sections 1 and 2 ran on the uncommitted change. Its diff (`results/replay_b1/source.diff`, SHA-256 `c75676be…d9558d`) is byte-identical to the committed diff `3373066b..cefbbc9d`.

## 1. Per-wave equivalence on the real input

**Command.**

```bash
(ulimit -v 31457280; PYTHONPATH=$PWD ... timeout 1500 /data/ycfeng/envs/frontier-py310/bin/python \
  /data/ycfeng/tmp/glm53_dsv4/h4c/wave_equivalence.py \
  /data/ycfeng/tmp/glm53_dsv4/h4c/wave_equivalence 1200)
```

The script is kept as [h4c_step_cost_speed/wave_equivalence.py](h4c_step_cost_speed/wave_equivalence.py).

The script uses the H4a run5 configuration (`h4a4_driver.build_config`). It wraps `prepare_moe_wave_from_inputs`. For every wave that took the new zero-phase branch, it recomputes the same wave with `capture_lane_timings=True`, which forces the full path. It then compares the EP participants, the six phase tuples and the barrier timing with `float.hex`. The run stops after 1,200 s of wall time.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Waves on the new branch | > 0 | 63,376 (PREFILL 24,768, DECODE 38,608), 17.80 s simulated | PASS |
| Waves on the full path in the normal call | 0 (no lane timings requested in run5's configuration) | 0 | PASS |
| Mismatches in participants, phases or barrier timing | 0 | 0 | PASS |

Result: [h4c_step_cost_speed/results/wave_equivalence.json](h4c_step_cost_speed/results/wave_equivalence.json).

## 2. P0 replay against H4a run5

**Command.**

```bash
(ulimit -v 62914560; PYTHONPATH=$PWD ... timeout 10800 /data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic/h4a4_driver.py \
  /data/ycfeng/tmp/glm53_dsv4/h4c/replay_b1)
(ulimit -v 62914560; PYTHONPATH=$PWD ... /data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic/h4a4_gate.py \
  /data/ycfeng/tmp/glm53_dsv4/h4c/replay_b1 /data/ycfeng/tmp/glm53_dsv4/h4c/gate)
```

Same input (`frontier.json` `747a08e5…`), step-cost table (`ea817ce5…`) and overrides as run5. run5 ran at `d6ea9661`; `git diff d6ea9661 3373066b -- frontier data tests/unit` is empty.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Status and requests | FINISHED, 128 formal requests | FINISHED, 128 | PASS |
| `request_metrics.csv` | byte-identical to run5 | identical, SHA-256 `977c4d5a…` | PASS |
| `metrics_ground_truth.jsonl` | byte-identical | identical, `4fee9488…` | PASS |
| `system_metrics.json` | byte-identical except wall-clock fields | fully identical, `1845fe17…` (the file has no wall-clock field) | PASS |
| `op_precision_metadata.csv` | byte-identical | identical | PASS |
| `config.json` | identical except the output root | identical except the output root | PASS |
| `run.log` | identical except timestamps and paths | identical except the launcher's last line (`exit 0` / `rc=0`) | PASS |
| `frontier_transfer_ledger.jsonl` | not checked in the design | 128 records; every field equal except `batch_id`. Old range 691-7,296,862, new range 3-21,278; the order of the values is the same in all record pairs | see below |
| Four client metrics (`h4a4_gate.py`) | equal to the H4a gate | `reading.json` byte-identical to the H4a gate's (`625272e0…`); `rows/` identical; `matrix.json` differs only in its output path | PASS |
| Four metrics | TTFT 17.35%, TPOT 2.50%, E2E 2.10%, THPT 0.54% | TTFT 398.8224 → 329.6299 ms (17.35%), TPOT 19.0094 → 19.4849 ms (2.50%), E2E 19845.49 → 20262.73 ms (2.10%), THPT 6272.08 → 6238.11 tok/s (0.54%); verdict FAIL, as in H4a | PASS (unchanged) |
| Wall time (`run_wall_s`) | measured | 758.2 s, against 8,094.8 s for run5: 10.7× faster | measured |

**`batch_id` in the transfer ledger.** `Batch` ids come from one class-wide counter. The full path built 16 EP lane batches per wave, and each one took an id. The new branch builds none, so later real batches get smaller ids. The design did not anticipate this; it is now recorded in design section 2. The values are internal labels: the order is unchanged, and no metric, event time or other ledger field changed.

**Load during the replay.** The section 1 check ran for the whole replay, and the e2e-gate test and the smoke matrix ran for about 2 minutes of it. Each is a single-threaded process on a 64-CPU host with a load average near 4.

Evidence: [h4c_step_cost_speed/results/replay_b1/](h4c_step_cost_speed/results/replay_b1/) (receipt, diff, log) and [h4c_step_cost_speed/results/gate/](h4c_step_cost_speed/results/gate/). Raw output: `/data/ycfeng/tmp/glm53_dsv4/h4c/replay_b1/`.

## 3. Regressions

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Full unit suite | `pytest tests/unit -p no:cacheprovider -p no:logging --continue-on-collection-errors --junitxml=...` | the H4a-2 result at `d6ea9661` (134 failed / 6,114 passed / 208 skipped / 108 errors), compared test by test | 134 / 6,114 / 208 / 108; 0 tests that passed before now fail, 0 fixed, same test IDs | PASS |
| `tests/unit/test_e2e_gate.py` | `pytest tests/unit/test_e2e_gate.py -q -p no:cacheprovider` | 40 passed | 40 passed | PASS |
| Dummy smoke matrix | `PYTHON_BIN=/data/ycfeng/envs/frontier-py310/bin/python bash examples/architecture/run_dummy_smoke_matrix.sh` | 12/12, exit 0 | 12/12, exit 0 | PASS |
| 77-case refactor fidelity | from clean worktree `.worktrees/fidelity-candidate-cefbbc9d`: `run_matrix.py run --label candidate_cefbbc9d --output-root /data/ycfeng/tmp/glm53_dsv4/h4c/fidelity --jobs 8 --clean-cache --continue-on-failure`, then `compare --baseline-label /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline --candidate-label candidate_cefbbc9d`, both with `PYTHONPATH` set to the worktree | 77/77 identical | 77 compared, 77 identical, 0 mismatched, 0 failures on either side, 0 missing; source `cefbbc9d`, clean | PASS |

The first unit run omitted `--continue-on-collection-errors` and stopped after 15 collection errors (missing `torch`, `aiohttp`, `matplotlib`). The rerun used the flag, as the earlier gates did; the H4a-2 record in progress.md lists its command without the flag. Logs: `/data/ycfeng/tmp/glm53_dsv4/h4c/unit_full/`, `/data/ycfeng/tmp/glm53_dsv4/h4c/gates/`, `/data/ycfeng/tmp/glm53_dsv4/h4c/fidelity/`. Fidelity report: [h4c_step_cost_speed/results/fidelity_comparison.json](h4c_step_cost_speed/results/fidelity_comparison.json). GLM, co-location, PDD and PD-AF results are unchanged.

## 4. Profile after the change

Same driver as the H4a profile (`results/profile_b1/profile_run.py`), first 180 s of wall time under cProfile.

| Quantity | Before (H4a profile) | After | Change |
| --- | --- | --- | --- |
| Simulated time covered in 180 s | 2.017 s | 19.063 s | 9.45× |
| EP waves in 180 s | 6,594 | 67,562 | 10.2× |
| `prepare_moe_wave_from_inputs`, cumulative | 166.4 s | 17.3 s | 25.2 → 0.26 ms per wave |

Where the 180 s go after the change (cumulative, overlapping):
- Per-layer attention prediction of each lane: `predict_step_cost_execution` 215,645 calls, 53.5 s, of which `ExecutionTime.__init__` takes 48.5 s. Most calls come from `handle_decode_sync_collective`, which predicts the next layer (88.3 s cumulative).
- `enter_layer_sync` 37.5 s, of which `schedule_layer_wave` takes 30.4 s.
- Idle-lane forwards (`advance_dummy_forward`) 22.1 s; replica scheduling 18.4 s.

Profile: [h4c_step_cost_speed/results/profile_b1/profile.txt](h4c_step_cost_speed/results/profile_b1/profile.txt).

**Remaining options (for YC, not done).**
- Option A (one computation per step, design section 2) would remove the per-layer events and most of the per-layer predictions above. Its cost and risk are unchanged: more than 5 `frontier/` files and 200 lines, and identity is not guaranteed by construction.
- A smaller step would build each lane's per-layer `ExecutionTime` once per step instead of once per layer. It would remove at most the 53.5 s above, so about 1.4× more. It was not designed or measured.
- At 758 s per P0 replay, the H4b-(b) reruns are workable. Recommendation: stop H4c here.

## Limits

- Identity is shown on one input, the P0 case. Other V4 table-priced topologies are covered by the same code path but were not replayed.
- The per-wave check covers the first 17.8 s of simulated time. The full replay covers all waves through its byte-identical outputs, but not wave by wave.
- Wall times are from one run each on a shared host.
