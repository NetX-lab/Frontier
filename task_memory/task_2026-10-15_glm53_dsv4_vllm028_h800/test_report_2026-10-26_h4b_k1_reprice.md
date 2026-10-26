# Test report 2026-10-26: H4b-(b) k=1-with-decodes DECODE entry repriced from the native lifecycle runs

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: method, harness changes and their identity checks, the three repriced tables, the three depth-2 diagnostics against item 2 and the lifecycle runs, the P2-64 reading on repetitions 2-3, persistence and limitations. |

**Authority.** D-H4B-K1-REPRICE (requirements item 24: YC adopted the lead's recommendations for next actions 1-6), D-H4A-CHECKPOINT (item 14), D-H200-MAINLINE-CONTINUE (item 21) and D-NO-INTERRUPT (item 19). Label DIAGNOSTIC: no result here qualifies a cell, and no constant is fitted to native06 or to any comparison target. The repriced entry comes from the native lifecycle runs, which are not the comparison targets. H200 and H800 evidence never qualify each other's cells. No GPU job was used; this session holds 0 GPUs. No file under `frontier/` changed, so GLM, co-location, PDD and PD-AF results are unchanged by construction.

## 1. Method

The item 2 report (section 7.1) found that Frontier's first D step is the decode-only entry plus the k=1-with-decodes entry of the step-cost table, and that this entry was priced from 1 profiled step (native06, P2-32) or substituted from the k=1-alone entry (P2-64). This change prices that one entry from the native, unprofiled lifecycle runs and reruns the item 2 depth-2 diagnostics. Nothing else changes.

- New entry: the median native D engine step whose schedule promotes one formal request (6143 computed + 1 scheduled token) while the lane has cached requests, measured as in `native_k1_step.py`: step s runs from max(schedule[s] end, update[s−1]) to update[s].
- Lifecycle runs: native06 → `h4b_lifecycle_measurement/inputs/h800l`, P2-32 → `h200_mainline/diagnostic/inputs/p2_32l`, P2-64 → `h200_mainline/diagnostic/inputs/p2_64l`. The builder checks `run_manifest.json` and every `nodes/*/scheduler_lifecycle/dp*_pid*.jsonl` against the input set's `manifest.json` before reading them.
- The builder fails if the measured decode-request counts fall outside the entry's [min, max] range.
- The repriced table keeps every other entry, records the replaced entry under `replaced_entry`, and adds `sources.k1_with_decodes_entry` (decision, source table and its SHA-256, lifecycle input, run tag, file digests).

## 2. Harness changes and identity checks

| File | Change | Check | Result |
| --- | --- | --- | --- |
| [h4b_d_side_model/k1_reprice/build_tables.py](h4b_d_side_model/k1_reprice/build_tables.py) | new: writes `k1_reprice/<case>/dsv4_step_cost.json` | rebuild from a scratch copy (`/data/ycfeng/tmp/glm53_dsv4/h4b_k1/rebuild_check/`) | all three tables byte-identical: PASS |
| [h4a_p0_diagnostic/h4a4_driver.py](h4a_p0_diagnostic/h4a4_driver.py) | the receipt names the table the loader reads: `dsv4_step_cost.json` beside the attention input when that field is set, else the case table | `step_cost_table(case, {})` against `CASES[case]["table"]` for the three cases | equal for all three: PASS |
| [h4a_p0_diagnostic/h4a5_attribution.py](h4a_p0_diagnostic/h4a5_attribution.py) | reads the run's table from its receipt and checks its SHA-256 (exits on mismatch) instead of a fixed per-case table | rerun on the three item 2 depth-2 runs | output byte-identical (`cmp`) to the persisted item 2 attributions: PASS |
| [h4b_d_side_model/results/native_k1_step.py](h4b_d_side_model/results/native_k1_step.py) | refactored: `classify_steps` returns (duration, cached count) for reuse by the builder | rerun | output byte-identical to `native_k1_step.txt`: PASS (new script digest in the item 2 report, section 7.1) |
| [h4b_d_side_model/results/repetition_reading.py](h4b_d_side_model/results/repetition_reading.py) | new: pools the native numerators and denominators over chosen repetitions (next action 4) | repetitions 1,2,3 on the P2-64 readings | reproduces the gate's pooled values (item 2: TTFT 741.216 native, 695.453 Frontier; repriced: 694.670, 6.28%): PASS |

Repriced tables (only DECODE entry 3 and `sources.k1_with_decodes_entry` differ from the source table; the PREFILL entry and the entry count are unchanged):

| Case | Old entry (ms) | Old samples | New entry (ms) | Samples | Decode requests | Table SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| native06 | 30.15 | 1 (profiled) | 21.39 | 378 | 1-10 | `2c8f83084d23d3ae77af00ab86f605b58cf3f4dca4c1fd4ed3a36c1bceac8382` |
| P2-32 | 34.25 | 1 (profiled) | 21.72 | 378 | 1-12 | `f742cf0d9e02b899374aa784831c0adacb75aacf0a8221c8a8e598fb692ef3df` |
| P2-64 | 26.76 | 0 (substituted from k=1 alone) | 25.99 | 372 | 1-8 | `f78aa39fbe8485efee59c479bda990d3934ab4e93114b05f863043653854dc07` |

## 3. Commands and environment

Python 3.10.6 (`/data/ycfeng/envs/frontier-py310/bin/python`) for the builder, the simulations, the gate and the attribution; `/usr/bin/python3` 3.10 for `persist.py` and `repetition_reading.py`. Source commit `6dff7547`; the run receipts record no dirty path under the source tree. Scratch: `/data/ycfeng/tmp/glm53_dsv4/h4b_k1/`.

```bash
# Tables; ulimit -v 31457280, timeout 900, from the task directory
python h4b_d_side_model/k1_reprice/build_tables.py .
# Simulations; ulimit -v 62914560, timeout 21600 (run_diag.sh)
#   = h4a4_driver.py <case> diag/<case>_k1d2 analytical_kv_cache_transfer_config_nixl_read_completion=engine_step \
#       cluster_config_decode_replica_scheduler_config_max_concurrent_batches=2 \
#       random_forrest_execution_time_predictor_config_atten_input_file=<task>/h4b_d_side_model/k1_reprice/<case>/attention.csv
bash /data/ycfeng/tmp/glm53_dsv4/h4b_k1/run_diag.sh <case> <case>_k1d2 <the two fields above>
# Client gate and H4a-5 attribution; ulimit -v 31457280, timeout 1800 each
bash /data/ycfeng/tmp/glm53_dsv4/h4b_k1/run_gate_attr.sh <case> <case>_k1d2
# Persistence, comparison, P2-64 reading
python3 /data/ycfeng/tmp/glm53_dsv4/h4b_k1/persist.py
python h4b_d_side_model/k1_reprice/k1_comparison.py . > h4b_d_side_model/k1_reprice/k1_comparison.txt
python3 h4b_d_side_model/results/repetition_reading.py 2,3 \
  p2_64_item2_depth2=h200_mainline/diagnostic/results/p2_64e2/gate/reading.json \
  p2_64_k1_reprice_depth2=h200_mainline/diagnostic/results/p2_64_k1_reprice_depth2/gate/reading.json \
  > h4b_d_side_model/k1_reprice/p2_64_repetitions_2_3.txt
```

The loader takes `dsv4_step_cost.json` beside the attention input file; `attention.csv` itself is not read on the step-cost path, so the field only selects the table. Every receipt records the repriced table path and digest (table above).

## 4. Run checks

| Check | Expected | native06 | P2-32 | P2-64 |
| --- | --- | --- | --- | --- |
| Driver exit; receipt status | 0; FINISHED | 0; FINISHED | 0; FINISHED | 0; FINISHED |
| Wall time (s) | about item 2 (770.1 / 793.6 / 1104.0) | 776.7 | 797.0 | 1113.5 |
| Gate and attribution exit | 0 | 0, 0 | 0, 0 | 0, 0 |
| Requests | 128 | 128 | 128 | 128 |
| Preemptions (max) | 0 | 0 | 0 | 0 |
| Log lines with "stale", "error" or "traceback" | 0 | 0 | 0 | 0 |
| Warning lines | as item 2 (8) | 8 | 8 | 8 |
| KV admission against native | as item 2 (DIFFERS, 2940 against 2944 blocks) | as item 2 | as item 2 | as item 2 |
| Median first D step = decode-only + new k=1 entry (ms) | 40.69 / 40.09 / 50.71 | 40.69 | 40.09 | 50.71 |
| Minimum first D step = k=1-alone entry (unchanged, ms) | 21.10 / 25.28 / 26.76 | 21.10 | 25.28 | 26.76 |
| P-side gap (ms), D-only change | unchanged from item 2 | +12.69 | +45.92 | +6.82 |

All PASS. The KV admission difference is the one recorded since item 2; it is not caused by this change.

## 5. Results

### 5.1 Parts against item 2 and the lifecycle runs (ms, `k1_comparison.txt`)

| Case | Part | Lifecycle (native) | Item 2 depth 2 | Repriced depth 2 | Repriced − item 2 | Repriced − lifecycle |
| --- | --- | --- | --- | --- | --- | --- |
| native06 | first D step (f / d_first_step mean) | 41.80 | 49.00 | 40.38 | −8.62 | −1.42 |
| native06 | READ side (b+c+d+e / kv_transfer) | 79.20 | 66.70 | 66.92 | +0.22 | −12.28 |
| native06 | handoff | 125.91 | 115.70 | 107.30 | −8.40 | −18.60 |
| P2-32 | first D step | 42.05 | 52.19 | 39.86 | −12.33 | −2.19 |
| P2-32 | READ side | 91.79 | 63.66 | 63.67 | +0.02 | −28.12 |
| P2-32 | handoff | 138.47 | 115.85 | 103.54 | −12.31 | −34.94 |
| P2-64 | first D step | 51.53 | 50.71 | 49.96 | −0.74 | −1.57 |
| P2-64 | READ side | 104.18 | 85.29 | 85.25 | −0.04 | −18.92 |
| P2-64 | handoff | 159.66 | 136.00 | 135.21 | −0.78 | −24.45 |

The lifecycle parts are over repetitions 1-3 (native06, P2-32: 378 requests) and 2-3 (P2-64: 248 requests). The ingress part a+g (4.90, 4.63, 3.95 ms) is not modeled (deferred, requirements item 24 sub-item 5).

Observed:
- The first D step moved by the entry change (−8.76, −12.52, −0.77 ms) less the share of requests whose first step is the k=1-alone step. It is now 1.42, 2.19 and 1.57 ms below the native f part on all three cases; before the change it was 7.20 and 10.14 ms above on native06 and P2-32.
- kv_transfer did not move (≤ 0.22 ms).
- The remaining handoff gap against the lifecycle runs is on the READ side (−12.28, −28.12, −18.92 ms) plus the unmodeled ingress. These are the late READ handles and the ingress of design sections 6.2 and 6.3.

### 5.2 Client gate over all repetitions (Frontier against the native comparison target)

| Case | Metric | Native | Item 2: Frontier, error, relative | Repriced: Frontier | Absolute error | Relative error | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native06 | TTFT (ms) | 398.822 | 403.058, +4.235, 1.06% | 394.658 | −4.164 | 1.04% | PASS |
| native06 | TPOT (ms) | 19.009 | 19.485, +0.475, 2.50% | 19.335 | +0.326 | 1.71% | PASS |
| native06 | E2E (ms) | 19845.486 | 20336.154, +490.668, 2.47% | 20174.427 | +328.941 | 1.66% | PASS |
| native06 | THPT (tok/s) | 6272.078 | 6235.657, −36.421, 0.58% | 6235.801 | −36.276 | 0.58% | PASS |
| P2-32 | TTFT (ms) | 364.030 | 397.265, +33.235, 9.13% | 384.954 | +20.924 | 5.75% | PASS |
| P2-32 | TPOT (ms) | 19.919 | 18.642, −1.277, 6.41% | 18.424 | −1.495 | 7.50% | PASS |
| P2-32 | E2E (ms) | 20741.132 | 19467.570, −1273.562, 6.14% | 19233.138 | −1507.993 | 7.27% | PASS |
| P2-32 | THPT (tok/s) | 6231.415 | 6276.160, +44.744, 0.72% | 6276.871 | +45.456 | 0.73% | PASS |
| P2-64 | TTFT (ms) | 741.216 | 695.453, −45.763, 6.17% | 694.670 | −46.546 | 6.28% | PASS (pooled); repetition 1 FAIL |
| P2-64 | TPOT (ms) | 25.871 | 24.768, −1.103, 4.26% | 24.751 | −1.120 | 4.33% | PASS |
| P2-64 | E2E (ms) | 27206.770 | 26032.868, −1173.901, 4.31% | 26014.849 | −1191.920 | 4.38% | PASS |
| P2-64 | THPT (tok/s) | 5963.206 | 5997.030, +33.824, 0.57% | 5996.892 | +33.685 | 0.56% | PASS |

Gate verdicts (limit 10% relative error per repetition and pooled): native06 PASS, P2-32 PASS, P2-64 FAIL (repetition 1 TTFT), the same verdicts as item 2.

Observed:
- TTFT moved by −8.40, −12.31 and −0.78 ms, as expected from the entry change (inference before the runs: about −8.8, −12.5 and −0.8 ms).
- On native06 the TTFT error changes sign (+4.24 → −4.16 ms). Item 2's small error was partly a cancellation between the overpriced first D step and the unmodeled READ-side parts.
- TPOT fell by 0.150, 0.217 and 0.017 ms. Inference: each request's 1023 decode steps include the k=1 steps of the requests promoted onto its lane in that time; 0.150 × 1023 / 8.76 ≈ 17.5, 17.7 and 22.6 such steps, close to the mean number of requests decoding at once (17.2, 16.5, 21.3). On native06 this reduces the TPOT error (2.50% → 1.71%); on P2-32, where Frontier's TPOT was already low, it increases it (6.41% → 7.50%).

### 5.3 P2-64 on repetitions 2-3 (supplementary DIAGNOSTIC reading, next action 4)

The gate verdict stays the one over all three repetitions (FAIL). This reading pools only the native repetitions 2-3 (`p2_64_repetitions_2_3.txt`):

| Run | Metric | Native (rep. 2-3) | Frontier | Absolute error | Relative error | Status |
| --- | --- | --- | --- | --- | --- | --- |
| item 2 depth 2 | TTFT (ms) | 721.880 | 695.453 | −26.427 | 3.66% | PASS |
| item 2 depth 2 | TPOT (ms) | 26.091 | 24.768 | −1.323 | 5.07% | PASS |
| item 2 depth 2 | E2E (ms) | 27412.832 | 26032.868 | −1379.964 | 5.03% | PASS |
| item 2 depth 2 | THPT (tok/s) | 5970.824 | 5997.030 | +26.207 | 0.44% | PASS |
| repriced depth 2 | TTFT (ms) | 721.880 | 694.670 | −27.210 | 3.77% | PASS |
| repriced depth 2 | TPOT (ms) | 26.091 | 24.751 | −1.340 | 5.14% | PASS |
| repriced depth 2 | E2E (ms) | 27412.832 | 26014.849 | −1397.983 | 5.10% | PASS |
| repriced depth 2 | THPT (tok/s) | 5970.824 | 5996.892 | +26.068 | 0.44% | PASS |

Repetition 1 raises the pooled native TTFT by 19.3 ms (741.216 − 721.880). Inference (item 1 report, section 6): about 16.5 ms of the P2-64 target excess is the first-contact READ wait of the lane pairs that meet it with a formal request (5 of 16 in the P2-64 lifecycle run); Frontier does not model it.

## 6. Persistence (D-INPUT-PERSISTENCE)

Copies with SHA-256 checks (`/data/ycfeng/tmp/glm53_dsv4/h4b_k1/persist.py`, same file set as item 2: metrics, receipt, run log, attribution, gate log, matrix, reading and rows; 16 files each):

| Run | Directory | `request_metrics.csv` | `gate/reading.json` | `h4a5_attribution.json` |
| --- | --- | --- | --- | --- |
| native06 | `h4a_p0_diagnostic/results/run_k1_reprice_depth2/` | `289d0072…7aee` | `1a983c12…1595` | `c50a5481…ca00` |
| P2-32 | `h200_mainline/diagnostic/results/p2_32_k1_reprice_depth2/` | `87fd3e77…73e7` | `ba2ffa2f…181c` | `a549ef23…7750` |
| P2-64 | `h200_mainline/diagnostic/results/p2_64_k1_reprice_depth2/` | `14421184…c845` | `b638152a…90cd` | `0e12e4ea…82a2` |

Analysis outputs: `k1_comparison.py` (`e702b01f…8bd9a`) → `k1_comparison.txt` (`5ad7a520…011e`; a second run is byte-identical); `p2_64_repetitions_2_3.txt` (`4a7269d8…58f9`); `build_tables.py` `8ee88a8a…bb71`; `repetition_reading.py` `8f6948f7…d2be`; `h4a4_driver.py` `0f9b0a9d…420e`; `h4a5_attribution.py` `7e026752…9550`.

## 7. Limitations

- **Different measurements.** The new entry is a scheduler-record step (schedule end or previous update → update) from an unprofiled run. The other DECODE entries are trace intervals from the profiled repetition 1 of the profile run. On decode-only steps the two methods give 19.30 against 19.83 ms (H800), 18.37 against 20.17 ms (H200-32) and 24.72 against 24.45 ms (H200-64). The table now mixes the two methods. Natively the k=1 step with decodes is 1.54-1.56 ms longer than the decode-only step; in the repriced tables the difference is 2.09, 3.35 and 1.27 ms.
- **Composition.** The repriced entry was measured at 1-10, 1-12 and 1-8 cached requests (median 10, 10, 6); the decode-only entry it pairs with was measured at 1-2 (P2-64: 1, 9, 13) decode requests. Both entries are used over 1-128 decode requests.
- **Unchanged entries.** The k=1-alone entries are unchanged (natively those 6, 6 and 12 steps take 61-69 ms, not analyzed). The P2-64 k=9 entries keep their D-H200-STEP-COST-SUBSTITUTION.
- **Source runs.** The lifecycle runs are native and unprofiled but are different runs from the comparison targets; the run-to-run spread between runs of one hardware (design section 11.4) also applies here. The P2-32 lifecycle run is DIAGNOSTIC evidence (R4 FAIL on the shutdown order only).
- **What is still missing.** The READ side remains 12-28 ms below the lifecycle runs (late READ handles, design section 6.2, not measured; ingress, section 6.3, deferred). The P2-32 P-side gap (+45.92 ms) is accepted as a limitation (requirements item 24 sub-item 2). The P2-64 first-contact READs are not modeled.
- **Qualification.** DIAGNOSTIC only. The repriced tables are not the profiling tables under `data/profiling/` and do not replace them; formal gate admission stays false in every table.
