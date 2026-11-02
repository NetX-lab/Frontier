# Test report 2026-10-26: H4b-(b) item 1, NIXL READ completion at DECODE engine steps (`nixl_read_completion='engine_step'`)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: the change and its size, default-mode identity, the livelock of the first engine_step runs and its fix, the three engine_step diagnostics against the design section 7 expectations, the part comparison with the three lifecycle runs, the remaining gap, persistence and limitations. |

**Authority.** D-H4B-ITEM1 (requirements item 20, option B), D-H200-MAINLINE-CONTINUE (item 21), D-H4A-CHECKPOINT (item 14) and D-NO-INTERRUPT (item 19). Label DIAGNOSTIC: no result here qualifies a cell, no constant is fitted to native06, and H200 and H800 evidence never qualify each other's cells. Design: [h4b_d_side_model/design.md](h4b_d_side_model/design.md) sections 4, 5 (option B), 7 and 8. No GPU job was used; this session holds 0 H200.

## 1. Change

| Commit | Content | `frontier/` files | Lines |
| --- | --- | --- | --- |
| `5f907f66` | New field `AnalyticalKVCacheTransferConfig.nixl_read_completion`: `transfer_end` (default, the existing `KVCacheTransferEndEvent` at the wire end) or `engine_step`. In `engine_step`, the READ is registered on the DECODE lane at P end; the lane's next engine step start posts it; a later step start at or after its wire end finds it; that step's output reports it and runs the transfer-end work inline, so the next pass schedules the request. Idle lanes without DP wake up at the wire end. DP lanes keep their wave and dummy passes running while READs are pending. The H4a-4 driver takes `field=value` overrides. | 8 | +184 / −4 |
| `96d9c1ef` | Livelock fix (section 3): `next_pass_start` starts an all-idle DP group's next pass at its earliest posted wire end. | 2 (already counted) | +18 / −5 |
| Total | | 8 | +202 / −9 |

The approved bound was 7-8 `frontier/` files and about 160-210 lines. `vllm_v1_engine_replica_scheduler.py` has 1,894 lines, below the 2,000-line limit. The unit test file `tests/unit/test_nixl_read_completion.py` has 8 tests: config default and validation, post and find at step starts, the depth-1 restriction, the idle-lane wake-up, a lane blocked on a batch, post and check at a dummy forward start, and the two DP-group cases of the fix.

## 2. Default mode is unchanged (design section 8, item 1)

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Full unit suite | `/data/ycfeng/tmp/glm53_dsv4/h4b_item1/run_unit_fix.sh` (`pytest tests/unit -q -p no:cacheprovider -p no:logging --continue-on-collection-errors -rfE`, frontier-py310) | failure set of the H4c baseline at `cefbbc9d` (`/data/ycfeng/tmp/glm53_dsv4/h4c/unit_full/unit.log`: 134 failed, 108 errors) | 134 failed, 6,122 passed, 208 skipped, 108 errors; the 241 distinct failing or erroring test IDs equal the baseline's, 0 added, 0 removed; the 8 new tests pass (6,114 + 8) | PASS |
| Focused suites | `run_focus_fix.sh` (`test_nixl_read_completion.py`, `test_e2e_gate.py`, `tests/integration/test_vllm_dp_placement_runtime.py`) | all pass except the inherited failure | 66 passed, 1 failed: `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`, which also fails on the pre-change tree (`dp_fail_base.log`) | PASS |
| Dummy smoke matrix | `run_smoke_fix.sh` (`PYTHON_BIN=…/frontier-py310/bin/python bash examples/architecture/run_dummy_smoke_matrix.sh`) | 12 cases, exit 0 | 12 cases completed, rc=0 | PASS |
| 77-case refactor fidelity at `5f907f66` | `run_fidelity.sh` (`tests/e2e/refactor_fidelity/run_matrix.py run --jobs 8 --clean-cache`, then `compare` against `/data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline`, baseline source `bfa82de8`), from the clean worktree `.worktrees/fidelity-candidate-5f907f66` | every output identical except the config echo of the new field | 77 compared, 0 failures on either side; each of the 77 cases differs only in `config.json` `/kv_cache_transfer_config/nixl_read_completion: missing in baseline` | PASS |
| 77-case refactor fidelity at `96d9c1ef` | `run_fidelity_96d9c1ef.sh`, same commands from `.worktrees/fidelity-candidate-96d9c1ef` | same | same result: 77 compared, one difference per case, the config echo only; predictor caches compared and clean | PASS |

The fidelity matrix covers GLM, co-location, PDD and PD-AF cases. `compare` exits 1 because it counts the config echo as a mismatch; the verification output lists the difference kinds and finds exactly one per case. Reports: [h4b_d_side_model/results/fidelity_comparison.json](h4b_d_side_model/results/fidelity_comparison.json) (`5f907f66`), [fidelity_comparison_96d9c1ef.json](h4b_d_side_model/results/fidelity_comparison_96d9c1ef.json) and [fidelity_verify_96d9c1ef.out](h4b_d_side_model/results/fidelity_verify_96d9c1ef.out).

## 3. Livelock of the first engine_step runs and its fix

**Observed.** The three engine_step diagnostics at `5f907f66` (scratch `diag/native06e`, `p2_32e`, `p2_64e`) stopped advancing in simulated time and were stopped by the lead (exit 143). The progress probe (`/data/ycfeng/tmp/glm53_dsv4/h4b_item1/probe/run_probe.sh`, which prints simulated time and event counts every 20 s of wall time) showed on native06:

| Probe | Code, mode | Simulated time | Events | Exit |
| --- | --- | --- | --- | --- |
| `n06e` | `5f907f66`, `engine_step` | 0.287354 s from 04:59:52 to 05:01:32 UTC, no progress | 77,439 → 469,672, mostly `DecodeSyncEvent` (45,841 → 292,006) | stopped, 143 |
| `n06d` | `5f907f66`, `transfer_end` | 105.45 s at the 600 s timeout | 1,225,851 | 124 (timeout) |
| `n06f` | fix, `engine_step` | 52.36 s at the 600 s timeout, about 3.5 s per 20 s | 609,452 | 124 (timeout) |

0.287354 s is the end of the first prefill (P compute 287.354 ms).

**Cause.** A posted READ keeps the MoE DP group's wave running, so every lane issues dummy forwards. The DSV4 step-cost path (`predict_step_cost_execution`) prices a dummy forward at zero. A group whose forwards are all dummy forwards therefore ran its passes at one simulated time and never reached the READ's wire end.

**Decision (D-NO-INTERRUPT, the lead's recommendation adopted).** Options:
- (a) When a dummy forward ends at its issue time and no lane of the group has a request to schedule (running, waiting, requested or found READs), the next pass starts at the earliest posted wire end. This is where the pass that finds the READ would run, as for an idle engine without DP.
- (b) Give a dummy forward a nonzero cost.
- (c) Stop counting posted READs as unfinished requests.

Adopted: (a).
- (b) was rejected: the DSV4 step-cost table has no measured dummy-forward cost, so the cost would be a constant without a measurement. It would also change the default-mode DSV4 diagnostics.
- (c) was rejected: vLLM counts WAITING_FOR_REMOTE_KVS requests as unfinished and keeps the DP wave running.

Under (a), a lane that receives work earlier still wakes the others at its first sync, so DP lockstep and the sync interval are unchanged. The default mode posts no READs, so it is unaffected (section 2).

**Simplification.** In native, an idle DP group keeps running dummy steps of nonzero length and finds the READ at the first dummy step start after the wire end. Under (a), Frontier finds it at the wire end itself. This applies only while every lane of the group is idle, which in these workloads happens only at the start of the run. The minimum P-end-to-report time of 2.57 ms in section 5 comes from those first requests.

## 4. engine_step diagnostics (design section 8, item 3)

Commands (`/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6), from the repository root:

```bash
# Simulation: H4a-4 driver with the override; ulimit -v 62914560, timeout 21600
bash /data/ycfeng/tmp/glm53_dsv4/h4b_item1/run_diag.sh <case> <name>
#   = h4a4_driver.py <case> /data/ycfeng/tmp/glm53_dsv4/h4b_item1/diag/<name> analytical_kv_cache_transfer_config_nixl_read_completion=engine_step
# Client gate and H4a-5 attribution; ulimit -v 31457280, timeout 1800 each
bash /data/ycfeng/tmp/glm53_dsv4/h4b_item1/run_gate_attr.sh <case> <name>
#   = h4a4_gate.py <case> diag/<name> gate/<name>; h4a5_attribution.py <case> diag/<name> attr/<name>.json
```

Cases and names: `native06` / `native06f`, `p2_32` / `p2_32f`, `p2_64` / `p2_64f`. The inputs, step-cost tables and other overrides are the same as in the default-mode diagnostics; the receipt's `override_diff` lists `analytical_kv_cache_transfer_config_nixl_read_completion` `transfer_end` → `engine_step`, and each `config.json` records `nixl_read_completion: engine_step` with `kv_connector: nixl`. The receipts record source `4e5d7f99` with two modified files, `batch_stage_end_event.py` and `vllm_v1_engine_replica_scheduler.py`. Those files were committed unchanged as `96d9c1ef`, whose only other file is the unit test. Simulation wall time: 776.3 s (native06), 800.8 s (P2-32), 1,109.5 s (P2-64).

Client gate (relative error limit 10%; the native value is the mean of the three reference repetitions):

| Case | Metric | Native | Frontier default | Frontier engine_step | Design section 7 expectation | Gate, engine_step |
| --- | --- | --- | --- | --- | --- | --- |
| native06 (H800-32) | TTFT (ms) | 398.82 | 329.63 (−17.35%) | 365.06 (−8.46%) | 366.03 (−8.2%) | PASS |
| | TPOT (ms) | 19.009 | 19.485 (+2.50%) | 19.485 (+2.50%) | — | PASS |
| | E2E (ms) | 19,845.5 | 20,262.7 (+2.10%) | 20,298.2 (+2.28%) | — | PASS |
| | THPT (tok/s) | 6,272.1 | 6,238.1 (−0.54%) | 6,237.3 (−0.55%) | — | PASS |
| P2-32 (H200-32) | TTFT (ms) | 364.03 | 327.23 (−10.11%) | 361.10 (−0.81%) | 361.94 (−0.6%) | PASS |
| | TPOT (ms) | 19.919 | 18.642 (−6.41%) | 18.642 (−6.41%) | — | PASS |
| | E2E (ms) | 20,741.1 | 19,397.5 (−6.48%) | 19,431.4 (−6.31%) | — | PASS |
| | THPT (tok/s) | 6,231.4 | 6,279.3 (+0.77%) | 6,277.7 (+0.74%) | — | PASS |
| P2-64 (H200-64) | TTFT (ms) | 741.22 | 601.00 (−18.92%) | 647.55 (−12.64%) | 647.97 (−12.6%) | **FAIL** |
| | TPOT (ms) | 25.871 | 24.768 (−4.26%) | 24.768 (−4.26%) | — | PASS |
| | E2E (ms) | 27,206.8 | 25,938.4 (−4.66%) | 25,985.0 (−4.49%) | — | PASS |
| | THPT (tok/s) | 5,963.2 | 6,000.9 (+0.63%) | 5,999.0 (+0.60%) | — | PASS |

Observed:
- TTFT matches the design section 7 arithmetic expectation within 1 ms on all three cases (−0.97, −0.84 and −0.42 ms).
- TPOT is unchanged to the printed precision on all three cases. E2E rises by the TTFT change (+35.4, +33.9 and +46.5 ms), and THPT falls by at most 0.03 points.
- The gate verdict changes from FAIL to PASS on native06 and P2-32 and stays FAIL on P2-64.
- These are DIAGNOSTIC results. The P2-32 PASS rests on two errors of opposite sign. The P side is 45.92 ms too long and the D-side handoff is 48.24 ms too short (section 5). So it is not evidence that the D side is correct.

H4a-5 attribution (ms; gap = Frontier − native):

| Case | Mode | TTFT gap | P-side gap | Handoff gap | Client-edge gap |
| --- | --- | --- | --- | --- | --- |
| native06 | default | −69.19 | +12.69 | −81.22 | −0.66 |
| | engine_step | −33.76 | +12.69 | −45.79 | −0.66 |
| P2-32 | default | −36.80 | +45.92 | −82.11 | −0.61 |
| | engine_step | −2.93 | +45.92 | −48.24 | −0.61 |
| P2-64 | default | −140.21 | +6.82 | −146.22 | −0.82 |
| | engine_step | −93.66 | +6.82 | −99.67 | −0.82 |

Only the handoff gap moves. It closes by 35.43, 33.87 and 46.55 ms, about 2 Frontier steps less T (design section 7: 2 × S_F − 2.57 = 36.40, 34.71 and 46.97 ms).

## 5. Parts against the three lifecycle runs

Tool: [h4b_d_side_model/results/item1_part_comparison.py](h4b_d_side_model/results/item1_part_comparison.py) (`/usr/bin/python3`; output [item1_part_comparison.txt](h4b_d_side_model/results/item1_part_comparison.txt)).
- Native: decoding lanes of the lifecycle runs. These are all repetitions for H800l and P2-32l. For P2-64l they are repetitions 2-3, the steady state without the first-contact READs (P2-64 lifecycle report section 5).
- Frontier: the attribution's `kv_transfer`, which in `engine_step` runs from P end to the READ report. It holds the wait for the next step start, the step that posts the READ, and the step that finds it. The attribution's `d_first_step` runs from the report to the first D token.

| Part (ms) | native06 / H800l | P2-32 / P2-32l | P2-64 / P2-64l r2-r3 |
| --- | --- | --- | --- |
| Native a + g | 4.90 | 4.63 | 3.95 |
| Native b (D API server → engine `add`) | 13.31 | 13.61 | 18.04 |
| Native c + d + e (READ schedule → promotion schedule) | 65.89 | 78.19 | 86.13 |
| Native f (promotion → first-token update) | 41.80 | 42.05 | 51.53 |
| Native handoff | 125.91 | 138.47 | 159.66 |
| Native READ transfer max over the 8 handles | 27.25 | 38.32 | 37.49 |
| Frontier default: kv_transfer + d_first_step | 2.57 + 39.71 = 42.28 | 2.57 + 43.24 = 45.81 | 2.57 + 38.98 = 41.55 |
| Frontier engine_step: kv_transfer + d_first_step | 47.70 + 30.01 = 77.71 | 45.57 + 34.11 = 79.68 | 61.34 + 26.76 = 88.10 |
| engine_step kv_transfer in Frontier steps (S_F = TPOT mean) | 2.45 | 2.44 | 2.48 |
| engine_step d_first_step in Frontier steps | 1.54 | 1.83 | 1.08 |

Observed:
- The P-end-to-report time is about 2.45 Frontier steps on all three cases. That is the wait for the running step, half a step on average, plus the two steps of the item 1 rule. The design section 8 wording "d about 3 Frontier steps" rounded that wait up to a whole step.
- The first D step became shorter (39.71 → 30.01, 43.24 → 34.11, 38.98 → 26.76 ms). The report now comes at an output pop, so the request is scheduled in the next pass without waiting for a running step. That wait moved into `kv_transfer`. The first D step itself is 1.08-1.83 Frontier steps; the reason for the case-to-case spread is not investigated here.

## 6. Remaining handoff gap

| Case | Handoff gap to the target | Frontier − lifecycle run | of which f − d_first_step | of which (b + c + d + e) − kv_transfer | of which a + g | Lifecycle run − target |
| --- | --- | --- | --- | --- | --- | --- |
| native06 | −45.79 | −48.20 | −11.80 | −31.50 | −4.90 | +2.41 |
| P2-32 | −48.24 | −58.79 | −7.94 | −46.22 | −4.63 | +10.55 |
| P2-64 | −99.67 | −71.56 | −24.77 | −42.84 | −3.95 | −28.10 |

The target is the reference mean of repeats (native06 123.49, P2-32r 127.92, P2-64r 187.76 ms). The lifecycle runs are separate runs on other hosts, so the last column is a run-to-run difference, not a model error.

Inference, from the design (sections 6.1-6.3) and the lifecycle reports; none of these is measured separately:
- (i) **Depth-2 output lag (design item 2).** On all three runs native f is about 2 decode steps, and d holds one step more than the item 1 rule. Design section 7 puts item 2 at 2 × S_F more TTFT: +38.97 (native06), +37.28 (P2-32) and +49.54 ms (P2-64).
- (ii) **Late READ handles.** The native READ transfer max (27.25 / 38.32 / 37.49 ms) exceeds one step for many requests, while Frontier's 2.57 ms wire time is always found at the next step start. Design section 6.2 estimates 0.81 steps (16.6 ms) on H200-32.
- (iii) **Ingress outside the engine.** The ingress is a + g plus the part of b that is not the engine-loop wait. Design section 6.3 estimates it at about 8.6 ms on H200-32. On P2-64, b is 4.4-4.7 ms above the 32-GPU runs, so its part is probably larger. The remainder a + b + g is 18.2 / 18.2 / 22.0 ms, but Frontier already covers part of b through its wait for the running step.
- (iv) **P2-64 target.** In the P2-64r repetition 1, the first-contact READs raise the target by about 16.5 ms (187.76 against the r2-r3 value of 171.27). The other 11.6 ms of the −28.10 is the difference between P2-64r and P2-64l in repetitions 2-3 (171.27 against 159.66).

On P2-32, (i)-(iii) sum to about 62 ms (37.3 + 16.6 + 8.6) against the measured Frontier-minus-lifecycle gap of −58.79 ms. On native06, (i) plus an ingress of the H200-32 size sum to about 48 ms against −48.20 ms; (ii) is not estimated for H800. On P2-64, (i) alone is 49.5 ms of the −71.56 ms gap against the lifecycle run, (ii) and (iii) are not estimated for H200-64, and (iv) is the −28.10 ms between the lifecycle run and the target.

**Arithmetic estimate with item 2 added** (TTFT + 2 × S_F, not a simulator run):

| Case | TTFT (ms) | Error |
| --- | --- | --- |
| native06 | 404.03 | +1.3% |
| P2-32 | 398.38 | +9.4% (the P-side +45.92 ms then dominates) |
| P2-64 | 697.09 | −6.0% |

## 7. Persistence (D-INPUT-PERSISTENCE)

Copied from scratch `/data/ycfeng/tmp/glm53_dsv4/h4b_item1/diag/<name>` and its `gate/` and `attr/` outputs and checked by hash. Each directory holds 16 files: `config.json`, `metrics_ground_truth.jsonl`, `request_metrics.csv`, `system_metrics.json`, `receipt.json`, `run.log`, `h4a5_attribution.json`, `h4a5_attribution.out`, `gate/gate.log`, `gate/matrix.json`, `gate/reading.json` and `gate/rows/` (sample, vLLM and Frontier rows).

| Directory | Scratch name | Files | Bytes |
| --- | --- | --- | --- |
| `h4a_p0_diagnostic/results/run_engine_step/` | `native06f` | 16 | 424,856 |
| `h200_mainline/diagnostic/results/p2_32e/` | `p2_32f` | 16 | 424,682 |
| `h200_mainline/diagnostic/results/p2_64e/` | `p2_64f` | 16 | 425,664 |

Digests:

| File | SHA-256 |
| --- | --- |
| `run_engine_step/request_metrics.csv` | `1e2445c83fbc3869ce84ecef76c6dcf9076023259b207ffbe3fd49c05466be7f` |
| `run_engine_step/system_metrics.json` | `eb432b2f50bd2b19e7bde0b81ae019393b259ab22304022e5a288919ed8fee6e` |
| `run_engine_step/h4a5_attribution.json` | `85353011abd996b3532af0d06611daa2bc9631a6a3b56f63b0b9b7d6e1e3c1be` |
| `run_engine_step/gate/reading.json` | `8f8cd12a86f38de07def1d67445eb78556f4401d937209f04eeb6a88afda7a13` |
| `run_engine_step/receipt.json` | `907e2985c127a0132e76e2e8bf845bf2fc982b542f154cc020371b1bc81523ff` |
| `p2_32e/request_metrics.csv` | `38280edd035eb72492ae761618dcad943e4dbe12a011aca3dd96f1c57d23adca` |
| `p2_32e/system_metrics.json` | `8b40f570badbf3f42b83ffa648ddd712f40f5bb2e522b74023434af45301ccbd` |
| `p2_32e/h4a5_attribution.json` | `7f6523524f6bf8df91e886291738d685da990b42792d2226260766a662cb6087` |
| `p2_32e/gate/reading.json` | `cc0dc320bd96d2aaa689c823b8ca2996767e6eac0a1e073e679d7dc15416f923` |
| `p2_32e/receipt.json` | `03a49fa84887bdb7b3acd310bc135bfea134a4fd9193536fd696f80941f8b9eb` |
| `p2_64e/request_metrics.csv` | `08db04747724c2a810f8e14caf19dfb60a881b59fd6af10b725bcec7ed6ebba8` |
| `p2_64e/system_metrics.json` | `3bff31706700ce822e51fe6e89bc1e7be024d3e2543f68c2b01d7417c2db3d03` |
| `p2_64e/h4a5_attribution.json` | `99964e3ccd4efecf90aec7bf66d04b38809c70ecad0dc3c488192138dfe4fc37` |
| `p2_64e/gate/reading.json` | `408f9dbbaa58ade498d95614b9f775081821fea87aa6b7ac5d7236954b485537` |
| `p2_64e/receipt.json` | `08a245290f2c44e9df82ffa6f2bd753fb2503aace5af83c0006949e77a9d7522` |
| `h4b_d_side_model/results/fidelity_comparison_96d9c1ef.json` | `c1701f733c8d858f3b189400bb41b4e0af346f157deff6530fc5a762928688cd` |
| `h4b_d_side_model/results/fidelity_verify_96d9c1ef.out` | `120bf8e1d04dacf2faca8fbedc85abb7f8186576755a0ff761dbbbf7719ed7e6` |

The scratch runs, probes, logs and the two fidelity candidate worktrees (`.worktrees/fidelity-candidate-5f907f66`, `.worktrees/fidelity-candidate-96d9c1ef`) are left in place. Removing the worktrees needs YC's approval.

## 8. Limitations

- DIAGNOSTIC only. One Frontier run per case, against one reference run each.
- The DSV4 default-mode diagnostics were not rerun after the change. Default-mode identity rests on the unit suites, the smoke matrix and the 77-case fidelity matrix (section 2), and on the unchanged P-side, client-edge and TPOT gaps in section 4.
- Section 5 compares means, not matched requests, and the lifecycle runs are separate runs on other hosts with the lifecycle observer.
- The mechanism split in section 6 is an inference. Item 2, the late handles and the ingress are not measured separately in Frontier.
- The livelock fix finds the READ at the wire end while a whole DP group is idle; native finds it at the next dummy step start (section 3).
- The P2-32 TTFT PASS comes from a P side that is 45.92 ms too long offsetting a handoff that is 48.24 ms too short.
