# Test report: G5h-16 `_02`, G5h-32 `_01` and the isolated G5h-16 run against the native P2 kernels

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: kernel comparison of G5h-16 `_02` and G5h-32 `_01`, the isolated-call diagnostic run (open item O-RS16) and the decision on which timings the collective consumer reads. |

## Scope

- Runs (all 16 or 32 H200, `step_main`, non-preemptible, submitted through /ds4-exec):
  - G5h-16 `_02` (int64 dispatch ids, B10 lead correction): RJob `exp-1028-231342-951824`, Pi `g5h-submit-1028c` (submitted both runs) / `g5h16-collect-1028d`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_02/download/`.
  - G5h-32 `_01`: RJob `exp-1028-231408-632060`, Pi `g5h-submit-1028c` / `g5h32-collect-1028e`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g5h32_01/download/`.
  - G5h-16 isolated `_01` (stage `g5h/launch/stage_g5h16_isolated_01.json`, commit `895f61af`): RJob `exp-1029-012627-132543`, Pi `g5h16-iso-1028h`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g5h16_iso_01/download/`.
- Comparator: `tests/analysis/v4_collective_kernels.py compare` (B10), native tables `h3o_operator_costs/evidence/collective_kernels/p2_{32,64}_native.json`.
- CPU environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `ulimit -v 31457280`; scratch `/data/ycfeng/tmp/g5h_compare/`.

## Runs

| Item | G5h-16 `_02` | G5h-32 `_01` | G5h-16 isolated `_01` |
| --- | --- | --- | --- |
| Terminal phase | Succeeded | Succeeded | Succeeded |
| Nodes | 2 × 8 (`0700`, `1192`) | 4 × 8 (`0357`, `0374`, `0705`, `0716`) | 2 × 8 (`0375`, `0638`) |
| Wall clock, GPU-hours | 363.25 s, 1.61 | 1,106.06 s, 9.83 | 331.01 s, 1.47 |
| Runtime checks | 12/12 per node | 12/12 per node | 12/12 per node |
| Measurements, kernel records | 246, 82 | 138, 46 | 105, 35 |
| Copy | 1,330 files, 120,493,478 bytes, equal to cloud | 1,501 files, 702,293,174 bytes, equal to cloud | 578 files, 23,203,179 bytes, equal to cloud |
| `queue_held_fraction` | 0.977-1.0 | 1.0 | 0.993-1.0 |

G5h total: 4.40 (`_01`) + 1.61 + 9.83 + 1.47 = 17.31 of the 36 GPU-hour cap.

Observation: node `0705` of G5h-32 waited the full 600 s final barrier (`TimeoutError('3 of 4 nodes wrote result.json')`) although the three peer files were written within 0.12 s of its own. This is the second occurrence (first: G5h-16 `_01` node `0882`); `_02` and the isolated run did not show it. Inference: the shared-volume directory listing on that node did not refresh. It costs wall clock only (about 5.3 of the 9.83 GPU-hours) and no measurement. Recorded as deferred item; the launcher stays unchanged for now.

## Kernel identity

| # | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | `compare --native p2_32_native.json --benchmark <g5h16_02>` | every group equal; review lists empty | 9/9 groups `equal`, 0 case errors, review lists empty | PASS |
| 2 | `compare --native p2_64_native.json --benchmark <g5h32_01>` | EP calls matched; TP calls absent by design | 9/9 groups `unmatched`: every native NCCL EP kernel is matched and the TP rows (custom all-reduce, MNNVL) are missing, because G5h-32 measures only the EP calls (B9 grid) | expected |
| 3 | `compare --native p2_64_native.json --benchmark <g5h32_01> --benchmark <g5h16_02>` (EP from G5h-32, TP lanes from G5h-16 modulo the source DP) | every group equal; review lists empty | 9/9 groups `equal`, review lists empty | PASS |
| 4 | `compare --native p2_32_native.json --benchmark <g5h16_iso_01>` | every group equal | 9/9 groups `equal`, review lists empty | PASS |

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

## Decision (D-NO-INTERRUPT, lead)

The collective consumer reads the back-to-back `median_us` of `summary.csv` (G5h-16 `_02` for the TP calls and for EP16, G5h-32 `_01` for EP32). The `_01` and isolated data stay as evidence only. The residual bias (about +15 µs per EP combine call and about +10 µs per EP dispatch call at decode on EP16, +8 µs combine on EP32) is a recorded limitation of the collective costs and is checked again in the end-to-end comparison, where 43 calls per forward add up to about 1.1 ms per decode step on EP16.

## Limits

- Native durations include peer waits; only the minima bound kernel time.
- The cause of the decode EP bias is open (candidates: NCCL channel or protocol state that differs between the benchmark and the serving engine, and the peer timing inside a real forward).
- The final-barrier wait is not repaired.
