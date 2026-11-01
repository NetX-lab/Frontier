# NIXL6144 Q1 run notes (descriptive only)

RJob `exp-1102-030025-971155`, 2 x 8 H200, nodes `gpu-h200-0457` and `gpu-h200-0604`.
Measured source content: lock head `a182084c`; all 23 runtime-check hashes matched on both nodes.
Cloud root: `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/p2_nixl6144_q1_20261102`
(local copy `/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download`).

Numbers below come from `case_summary.csv`, `case_pass_spread.csv`, `paired_6143_6144.csv`
and the raw `C_nixl/nixl_reads.jsonl`. No model is fitted and no trial is discarded.

## Per-reader throughput (median of the three pass medians)

| prompt | pairs | in-flight | bytes/reader | critical-path median ms | reader median ms | reader GB/s |
|---|---|---|---|---|---|---|
| 6143 | 1 | 1 | 46,103,040 | 0.964 | 0.964 | 47.8 |
| 6143 | 1 | 16 | 737,648,640 | 15.125 | 15.125 | 48.8 |
| 6143 | 8 | 1 | 46,103,040 | 0.972 | 0.966 | 47.4 |
| 6143 | 8 | 16 | 737,648,640 | 15.159 | 15.130 | 48.7 |
| 6145 | 1 | 1 | 51,114,240 | 1.066 | 1.066 | 47.9 |
| 6145 | 1 | 16 | 817,827,840 | 16.760 | 16.760 | 48.8 |
| 6145 | 8 | 1 | 51,114,240 | 1.075 | 1.068 | 47.6 |
| 6145 | 8 | 16 | 817,827,840 | 16.796 | 16.782 | 48.7 |

Per-reader bandwidth rises from 47.4-47.9 GB/s at one request in flight to 48.7-48.8 GB/s at sixteen.
Eight active pairs add 0.81-0.87% at one request in flight and 0.21-0.22% at sixteen.

## Equal-byte 6143 vs 6144 (paired_6143_6144.csv)

Both prompts use 46 blocks, 46,103,040 bytes per reader at one request in flight.
Paired median differences are -0.0014 ms to +0.0022 ms, ratios 0.9989 to 1.0012.
6145 (51 blocks) over 6143 (46 blocks) is 1.10798, against the byte ratio 1.10870.

## Pass spread

`case_pass_spread.csv`: median 0.090%, largest 0.882% (`t6143_pairs8_inflight4`,
pass medians 3.804237 / 3.837828 / 3.807770 ms).

## Tails (not trimmed)

- Median over the 90 case-passes of `complete_max / complete_median` is 1.010.
- 12 of 90 case-passes contain one trial above 3x the case median; the largest is 19.94 ms
  against a 1.918 ms median (`t6144_pairs8_inflight2`, pass 2).
- `complete_p90 / complete_median` is 1.002 at the median case and 1.781 at the worst
  (`t6145_pairs8_inflight8`, pass 1: median 8.411 ms, p90 14.979 ms, max 16.659 ms).
  The heavier p90 tails sit in 8-pair cases; all-pass medians still agree within 1% of 1-pair.
- Largest single trial overall: 27.50 ms (`t6145_pairs8_inflight16`).

## Platform timing

- created 19:00:26Z, queue approved 22:21:16Z, scheduled 22:23:32Z, running 22:26:22Z, finish 22:27:40Z.
- Benchmark phase window 22:26:32 - 22:27:40 (about 68 s of measured rounds).
- Container GPU-hours 0.346 over the running window (16 x 77.76 s); 1.10 over scheduled-to-finish.
