# Test report: B26, vLLM DP host run-ahead of forwards (stage 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Open items: the load-report key and I-006 closed by B26b (`414fdd03`, merge `49e7e5fa`); the stale-report statement corrected. |
| 2026-10-30 | Created: lead review of B26 (Codex request `7218b11a`), the 77-case fidelity matrix of main `50ed2061`, and the start of the A10 K=19 replays. |

## Scope

B26 implements decision 43 (D-DP-DUMMY-RUN-AHEAD, [assignment](h3o_operator_costs/codex_assignment_B26_dp_dummy_run_ahead.md)): with `VllmV1SchedulerConfig.max_forwards_in_flight = K`, a DP host forms its forwards ahead of the GPU, up to K unfinished forwards, and each host iteration waits for every DP lane to form the same forward. K unset keeps the existing wait for each dummy forward. The A10 driver sets K = 19, the launch depth measured on the native P2-64 wave run `exp-1030-144848-164721` (A10 report section 3.5, observation 4).

- Branch `b26-dp-dummy-run-ahead`, commit `9820ef79`; merged into main as `50ed2061`.
- Codex report: `h3o_operator_costs/b26/report.md`; scratch `/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/`.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, BLAS/OMP/MKL threads 1, `ulimit -v 31457280` for tests and 62914560 for simulations.

## 1. Lead checks

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| B26 test suite (14 files, including the 22 new run-ahead cases) | `timeout 900 zsh /data/ycfeng/tmp/b26_lead_verify.sh` (the Codex `verify.sh` with lead output paths; log `/data/ycfeng/tmp/b26_lead_verify.log`) | all pass except a failure that is also on main | 320 passed, 1 failed (`test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`) | PASS |
| That failure on main `88fbef52` | `pytest tests/integration/test_vllm_dp_placement_runtime.py` in the main tree (log `/data/ycfeng/tmp/main_placement.log`) | — | 18 passed, 1 failed, the same assertion (`{2: (0, False)} != {2: (0, True)}`, line 481) | pre-existing; issue I-006 |
| Fidelity matrix, K unset | `bash /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_fidelity_b26.sh` (77 cases, worktree `.worktrees/fidelity-candidate-50ed2061` at main `50ed2061`, clean; output `stage4/fidelity_b26/`) against `stage4/fidelity/candidate_b6c28858` | 77 of 77 identical | 77 identical, 0 mismatched, 0 failures, 0 cache-name differences | PASS |
| Module size | `wc -l` | each at or below 2,000 | `vllm_v1_engine_replica_scheduler.py` 1,692 (from 1,918); `vllm_dp_busy_loop.py` 381 | PASS |

The fidelity baseline `candidate_b6c28858` is the stage 4 matrix of main `b6c28858`. Every commit between it and `50ed2061` (B19-B23, B25, H4b-(b) item 1, B26) therefore leaves the 77 cases unchanged.

Code review notes:

- The structural validator change in `tests/integration/run_dsv4_structural.py` follows the cache-zero cost metadata that B17 (`c8c5e593`) introduced (`frontier/execution_time_predictor/deepseek_v4.py:602`); it is not caused by B26.
- The scheduler change removes a stale "TEMPORARY" comment on the scheduling policy; it has no code effect.

## 2. Codex short check (from its report, not rerun by the lead)

Trace: the first 32 requests of `canonical_fixed_6144_1024_req128_qps1` with `num_decode_tokens=2` (`$B26/trace32.csv`); cache: byte copy of `stage4/a10/p2_64_warm_g1hs_r2/cache`; driver `a10_driver.py p2_64`.

| Run | Prefill side (arrival to prefill end), mean and range | Prefills queued behind earlier forwards | Largest gap to the previous forward end |
| --- | --- | --- | --- |
| K unset | 278.490 ms (273.940-282.124) | 0 | 0 µs |
| K = 19 | 508.665 ms (273.940-745.650) | 30 | 0 µs |

K unset gives `request_metrics.csv` and `system_metrics.json` byte-identical to the base checkout `.worktrees/b26-base` on the same trace. The K = 19 prefill side rises in a sawtooth and resets after a wave ends.

The CSV `ttft` column of a PDD run is the prefill side only (`Request.ttft`); the A10 gate uses client TTFT from `metrics_ground_truth.jsonl`, so the short check does not give the client TTFT.

## 3. Open items

- **Load-report key under K: closed by B26b** ([report](h3o_operator_costs/b26b/report.md), commit `414fdd03`, merged into main as `49e7e5fa`). Under K, a report is now keyed by the host formation index of the forward that the lane formed, as vLLM keys it by the host step counter (`vllm/v1/engine/core.py:2120-2135`). Codex's probe: the 8 K-unset rows equal the B26 rows exactly; the 7 K = 19 rows carry their host formation indices (lane 1 at 0.124 s: key 4 → 23). The lead also made the publish rule the same with and without K: a lane reports at admission while its batch queue (`_max_concurrent_batches`, vLLM's `batch_queue_size`, `core.py:683-689`) has room. This changes K-unset behavior only for `vllm_load_balancing` lanes with an explicit queue capacity above the pipeline depth, such as a unified DECODE lane at PP1 with capacity 2; it is a native-alignment fix. A10 uses `round_robin` placement and is not affected. The lead's run of the 13 affected test files passed (269 passed, `/data/ycfeng/tmp/b26b_lead/verify.log`).
- **Stale reports (correction).** The B26 assignment asked for a stale report to be dropped. vLLM's coordinator keeps the counts of an older report and only warns (`coordinator.py:390-417`); Frontier's `VllmDPLoadBalancer.report` already does the same, and B26b keeps it.
- **I-006: closed by B26b.** A test-fixture timing problem caused by `b0bf0c78`; the fixture's third arrival moved from 8 ms to 10 ms; no simulator change (see `issues.md`).
- **A10 K=19 replays.** P2-32 and P2-64, cold and warm, from `.worktrees/a10-k19-50ed2061` (`stage4/run_a10_k19.sh`, roots `stage4/a10/p2_{32,64}_{cold,warm}_k19`), started 2026-10-30 10:42 (warm runs simulating from 10:43, cold runs training until about 10:54). Expected from the FIFO model: P2-64 prefill side about 509-515 ms and TTFT about 648-654 ms; P2-32 nearly unchanged. Results go to the A10 report.
