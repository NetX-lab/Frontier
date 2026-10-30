# Test report: G5h-16 `_02`, G5h-32 `_01` and the isolated G5h-16 and G5h-32 runs against the native P2 kernels

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Added the isolated G5h-32 run (I-008), the native critical-path times of the eager EP calls on P2-32 and P2-64, and the revised decision: eager EP rows with a 6,144-token lane come from the isolated runs. |
| 2026-10-28 | Created: kernel comparison of G5h-16 `_02` and G5h-32 `_01`, the isolated-call diagnostic run (open item O-RS16) and the decision on which timings the collective consumer reads. |

## Scope

- Runs (all 16 or 32 H200, `step_main`, non-preemptible, submitted through /ds4-exec):
  - G5h-16 `_02` (int64 dispatch ids, B10 lead correction): RJob `exp-1028-231342-951824`, Pi `g5h-submit-1028c` (submitted both runs) / `g5h16-collect-1028d`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_02/download/`.
  - G5h-32 `_01`: RJob `exp-1028-231408-632060`, Pi `g5h-submit-1028c` / `g5h32-collect-1028e`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g5h32_01/download/`.
  - G5h-16 isolated `_01` (stage `g5h/launch/stage_g5h16_isolated_01.json`, commit `895f61af`): RJob `exp-1029-012627-132543`, Pi `g5h16-iso-1028h`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_iso_01/download/`.
  - G5h-32 isolated `_01` (stage `g5h/launch/stage_g5h32_isolated_01.json`, commit `70790174`; the G5h-32 `_01` grid with one call per repeat and 100 repeats): RJob `exp-1030-234535-245357`, Pi `g5h32-iso-1030a`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g5h32_iso_01/download/`.
- Comparator: `tests/analysis/v4_collective_kernels.py compare` (B10), native tables `h3o_operator_costs/evidence/collective_kernels/p2_{32,64}_native.json`.
- CPU environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `ulimit -v 31457280`; scratch `/data/ycfeng/tmp/g5h_compare/`.

## Runs

| Item | G5h-16 `_02` | G5h-32 `_01` | G5h-16 isolated `_01` | G5h-32 isolated `_01` |
| --- | --- | --- | --- | --- |
| Terminal phase | Succeeded | Succeeded | Succeeded | Succeeded |
| Nodes | 2 × 8 (`0700`, `1192`) | 4 × 8 (`0357`, `0374`, `0705`, `0716`) | 2 × 8 (`0375`, `0638`) | 4 × 8 (`0068`, `0192`, `0628`, `0722`) |
| Wall clock, GPU-hours | 363.25 s, 1.61 | 1,106.06 s, 9.83 | 331.01 s, 1.47 | 901.23 s, 8.01 (about 5.5 min of it queued: 16 of the 32 GPUs were free at submission) |
| Runtime checks | 12/12 per node | 12/12 per node | 12/12 per node | 12/12 per node |
| Measurements, kernel records | 246, 82 | 138, 46 | 105, 35 | 138, 46 |
| Copy | 1,330 files, 120,493,478 bytes, equal to cloud | 1,501 files, 702,293,174 bytes, equal to cloud | 578 files, 23,203,179 bytes, equal to cloud | 1,501 files, equal to the cloud listing (116,333,601 bytes remote) |
| `queue_held_fraction` | 0.977-1.0 | 1.0 | 0.993-1.0 | 1.0 |

G5h total: 4.40 (`_01`) + 1.61 + 9.83 + 1.47 + 8.01 = 25.32 of the 36 GPU-hour cap.

Observation: node `0705` of G5h-32 waited the full 600 s final barrier (`TimeoutError('3 of 4 nodes wrote result.json')`) although the three peer files were written within 0.12 s of its own. This is the second occurrence (first: G5h-16 `_01` node `0882`); `_02` and the isolated run did not show it. Inference: the shared-volume directory listing on that node did not refresh. It costs wall clock only (about 5.3 of the 9.83 GPU-hours) and no measurement. Recorded as deferred item; the launcher stays unchanged for now.

## Kernel identity

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `compare --native p2_32_native.json --benchmark <g5h16_02>` | every group equal; review lists empty | 9/9 groups `equal`, 0 case errors, review lists empty | PASS |
| 2 | `compare --native p2_64_native.json --benchmark <g5h32_01>` | EP calls matched; TP calls absent by design | 9/9 groups `unmatched`: every native NCCL EP kernel is matched and the TP rows (custom all-reduce, MNNVL) are missing, because G5h-32 measures only the EP calls (B9 grid) | expected |
| 3 | `compare --native p2_64_native.json --benchmark <g5h32_01> --benchmark <g5h16_02>` (EP from G5h-32, TP lanes from G5h-16 modulo the source DP) | every group equal; review lists empty | 9/9 groups `equal`, review lists empty | PASS |
| 4 | `compare --native p2_32_native.json --benchmark <g5h16_iso_01>` | every group equal | 9/9 groups `equal`, review lists empty | PASS |
| 6 | `compare --native p2_64_native.json --benchmark <g5h32_iso_01> --benchmark <g5h16_02>` | every group equal; review lists empty | 9/9 groups `equal`, review lists empty (`g5h32_iso01_g5h16_02_vs_p2_64.txt`) | PASS |

Check 1 closes the B10 finding: with int64 ids the four eager uneven-lane groups now launch the native 43 `AllGatherV` grid (4, 1, 1) plus 43 grid (9, 1, 1) per forward. The summaries (comparison part only, the auxiliary list left out) are kept in `h3o_operator_costs/evidence/collective_kernels/`: `g5h16_02_vs_p2_32.txt`, `g5h32_01_g5h16_02_vs_p2_64.txt`, `g5h16_iso01_vs_p2_32.txt`. Full summaries and JSON stay in the scratch directory.

## Durations (reported, not gated)

Native kernel durations include peer waits. `ep_kernel_rows.txt` (from `ep_kernel_rows.py`) lists lane 0 of every NCCL kernel. Decode rows, benchmark kernel median minus native kernel median:

| Comparison | Kernel | Native min / median (µs) | Benchmark min / median (µs) | Difference of medians (µs) |
| --- | --- | --- | --- | --- |
| P2-32, decode FULL real | EP dispatch `AllGather` grid 4 | 110.7 / 119.6 | 124.3 / 130.4 | +10.8 |
| P2-32, decode FULL dummy | EP combine `ReduceScatter` grid 2 | 33.5 / 38.8 | 49.7 / 53.8 | +15.1 |
| P2-32, decode PIECEWISE dummy | EP combine `ReduceScatter` grid 2 | 35.8 / 39.6 | 49.7 / 53.8 | +14.3 |
| P2-64, decode FULL dummy | EP combine `ReduceScatter` grid 2 | 85.5 / 98.9 | 102.1 / 107.1 | +8.3 |
| P2-64, decode FULL dummy | EP dispatch `AllGather` grid 4 | 232.9 / 260.4 | 245.6 / 248.4 | −12.0 |

In the real decode groups the native combine median includes peer waits (136.8 µs on P2-32), so only its minimum (34.7 µs) is comparable. The eager 6144-row prefill rows agree within a few µs for the dispatch `AllGatherV` grids 4 and 9; the eager combine `Reduce` grid 16 measures 773-804 µs against native 548-719 µs.

## O-RS16: are the decode EP collectives slower because the benchmark times back-to-back calls?

Open item O-RS16 (named in the isolated stage's `grid_source`): `_02` and G5h-32 time 10 back-to-back calls per repeat; the hypothesis was that this queueing inflates the latency-bound EP calls at decode widths. The isolated stage repeats 35 cases of the `_02` grid with one call per repeat and 100 repeats.

| # | Command | Expected if the hypothesis holds | Actual | Result |
| --- | --- | --- | --- | --- |
| 5 | `python iso_vs_b2b.py <iso summary.csv> <_02 summary.csv>` (median over passes of `median_us`, `min_us`, `p10_us`) | isolated decode EP medians drop to the native kernel medians | isolated medians are higher at every decode width: EP dispatch +11.5 to +57.7 µs, EP combine +10.6 to +57.8 µs, TP calls +12.5 to +36.4 µs (`iso_vs_b2b.txt`) | hypothesis rejected |

Inference: a single timed call also measures its launch and the arrival skew of the peer ranks (10-58 µs more per decode call here), which back-to-back timing amortizes. At 6144 rows the isolated EP calls are 29-314 µs lower than back-to-back; the bandwidth-bound calls therefore behave differently, and neither timing mode reproduces the native kernel medians for every call.

The kernel traces of the isolated run hold one traced call per case (minimum = median = maximum, for example 2,767 µs for the decode dispatch), so its kernel durations are not used.

## I-008: which timing reproduces the native eager EP calls (2026-10-30)

I-008 (`issues.md`): at 6,144 rows the back-to-back EP combine is 164.7 µs per layer below the native P2-64 call. The isolated G5h-32 run repeats the whole G5h-32 `_01` grid with one aligned call per repeat. The native side is the critical path of each layer: the latest end over the EP ranks less the last rank's start. Scripts and outputs are in `h3o_operator_costs/a10/k19/prefill_compute/`: `ep_critical.py`, `ep_critical_p2_32_f64.txt`, `peer_align_p2_32.txt`, `p2_32_dp0_tp0.txt`, and for P2-64 `peer_align_f64.txt`, `p2_64_dp0_tp0.txt`. The benchmark side is `iso_vs_b2b.py` (median over the 3 passes) with outputs `iso_vs_b2b.txt` (EP16) and `iso_vs_b2b_ep32.txt` (EP32).

| # | Call (lane vector) | Native, µs per layer | Isolated median | Back-to-back median | Result |
| --- | --- | --- | --- | --- | --- |
| 7 | EP32 combine `Reduce` (`eager_t6144_1_1_1`) | 2,171.2: last arrival to lane 0's end, P2-64 forward 64 (lane 0, TP rank 0, which ends last) | 2,172.0 | 2,011.1 | isolated within 1 µs |
| 8 | EP32 dispatch, 3 `AllGatherV` (`eager_t6144_1_1_1`) | about 1,048: lane 0, TP rank 0, 45.08 ms over 43 layers; the dummy lanes arrive first | 1,043.0 | 1,031.9 | both within 17 µs |
| 9 | EP16 combine `Reduce` (`eager_t6144_1`) | 717.0: critical path over 16 ranks, P2-32 forward 64 (arrival spread 25.2 µs) | 734.6 | 763.9 | isolated +17.6, back-to-back +46.9 |
| 10 | EP16 dispatch, 3 `AllGatherV` (`eager_t6144_1`) | 541.8: critical path over 16 ranks (the dummy lane arrives 4.6 ms first) | 560.2 | 801.8 | isolated +18.4, back-to-back +260.0 |

Other rows (reported):

- **Decode (CUDA graph) EP calls.** The isolated median is above back-to-back by 47-93 µs on EP32 and by 43-58 µs on EP16. The one exception is the EP32 combine at 32 tokens (−71.4 µs), where the back-to-back median of 290.7 µs lies far above its minimum of 166.6 µs. The back-to-back median is within −12 to +15 µs of the native decode kernel medians ("Durations" above). For decode, back-to-back is the closer mode.
- **TP calls.** The isolated median is above back-to-back by 4-36 µs at every width. At 6,144 rows the native P2-64 SP calls on lane 0, TP rank 0 take 165 µs (`AllGather`, 7.27 ms over 44 calls) and 155 µs (`ReduceScatter`, 6.66 ms over 43 calls). Back-to-back measures 161.6 and 156.6 µs, isolated 191.3 and 186.3 µs. For TP calls, back-to-back is the closer mode.
- **The 1-token eager EP rows.** Isolated is above back-to-back by 11-21 µs on both topologies. The consumer pools these rows with the 1-token graph rows into the same curve point.
- **The native lane-0 kernel on EP16 is not the call's time.** In P2-32 forward 64, lane 0 TP rank 0's three dispatch kernels take 806.0 µs per layer (median), which is 264 µs more than the critical path, because other TP ranks of lane 0 arrive later. Its SP `ReduceScatter` takes 806.5 µs per layer against 155 µs on P2-64. The profiled P2-32 forward 64 is also 70 ms idle on that rank (span 327.6 ms, kernel union 258.0 ms). The critical path removes these per-rank waits; the idle time is not explained here and is not used.

Reading (inference):

- In an eager prefill forward, consecutive EP calls are separated by milliseconds of per-lane compute, and the call starts when the last rank arrives. The isolated mode reproduces that start; its median is within 1-18 µs of the native critical path on both topologies.
- The back-to-back loop overlaps successive calls on the cross-node ring. On EP32 the peers start the next call early, so the root rank's time is too low (−160 µs). On EP16 the next call waits behind the previous one, so the time is too high (+47 µs for the combine, +260 µs for the dispatch).
- In a CUDA-graph decode step, EP calls follow each other within tens of µs. The calls of TP groups stay inside one node. In both cases, back-to-back matches native, and the isolated mode adds its own alignment and launch cost (about 20-90 µs).
- The isolated eager EP16 values remain about 18 µs per call above native: 1.55 ms per P2-32 prefill forward. This is a recorded limitation.

## Decision (D-NO-INTERRUPT, lead)

**Superseded on 2026-10-30 for the eager EP rows (see the revised decision below).** The collective consumer reads the back-to-back `median_us` of `summary.csv` (G5h-16 `_02` for the TP calls and for EP16, G5h-32 `_01` for EP32). The `_01` and isolated data stay as evidence only. The residual bias (about +15 µs per EP combine call and about +10 µs per EP dispatch call at decode on EP16, +8 µs combine on EP32) is a recorded limitation of the collective costs and is checked again in the end-to-end comparison, where 43 calls per forward add up to about 1.1 ms per decode step on EP16.

## Revised decision (2026-10-30, D-NO-INTERRUPT, lead)

- The H200 table takes the EP-group rows (`moe_dispatch_all_gatherv`, `moe_combine_reduce_scatterv`) of every eager scenario with a 6,144-token lane from the isolated runs: G5h-16 isolated `_01` for EP16 and G5h-32 isolated `_01` for EP32. Every other row stays back-to-back: the graph rows, the TP rows and the 1-token eager rows.
- An uneven EP dispatch (a lane vector with unequal SP rows) is priced from its own measured `AllGatherV` row, as the combine already is. Today it reads the uniform curve at the largest lane's bytes. That point comes from the uniform `eager_t6144_6144` and `eager_t6144_6144_6144_6144` rows: 801.7 µs on P2-32 and 1,076.0 µs on P2-64, against 560.2 and 1,043.0 µs for the isolated `eager_t6144_1` and `eager_t6144_1_1_1` rows.
- Assigned to Codex as B28 (consumer change, table rebuild and provenance). Then come the 77-case fidelity comparison, the P2-32 and P2-64 replays, and the A10 gate.
- Expected change per 6,144-token prefill forward (43 layers), arithmetic and not a run: `expected_prices.py` → `expected_prices.txt` in `h3o_operator_costs/evidence/collective_kernels/`. The script prices each call with the consumer on the current table and takes the planned price from the isolated row (median of the 3 passes):
  - P2-64: +5.50 ms (combine 2,011.1 → 2,172.0 µs, +6.92 ms; dispatch 1,076.0 → 1,043.0 µs, −1.42 ms).
  - P2-32: −11.64 ms (dispatch 801.7 → 560.2 µs, −10.38 ms; combine 763.9 → 734.6 µs, −1.26 ms).
- Expected effect on TTFT (inference; K19 TTFT on `901bc091`: P2-32 335.1 ms, −7.94%; P2-64 655.1 ms, −11.62%):
  - P2-64: at the FIFO slope of about 1.49 (A10 report section 5.6), TTFT rises about 8.2 ms to about 663 ms (−10.5%).
  - P2-32: Frontier's prefill side of the first request (221.6 ms) falls to about 209.9 ms, about 12 ms below the 222 ms that the canonical runs leave for the step (A10 report section 5.5). That 222 ms also holds the host time that Frontier does not model for V4. TTFT falls by at least 11.6 ms to about 323 ms (−11.1%) or lower.
  - Both cases are then expected to fail the TTFT gate until the ingress (A10 report section 5.4) and the host part of the step (section 5.6, observation 4) are modeled.

## Limits

- Native durations include peer waits; only the minima bound kernel time. Checks 7-10 use the critical path, so they do not depend on this.
- Each native critical path comes from one profiled forward per topology (forward 64). The profiled P2-32 run is slower than the canonical runs (I-007 and the idle time noted above).
- The cause of the decode EP bias is open (candidates: NCCL channel or protocol state that differs between the benchmark and the serving engine, and the peer timing inside a real forward).
- The final-barrier wait is not repaired.
