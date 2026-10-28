# C1: workload coordinates of the three V4 cases against existing data (H3o-(b) stage 0)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | §4.2: the mHC head and the final norm run on the descriptor rows W, not on L (pinned `model.py:1197-1211`, native grid 6,144 in the 6,144-token prefill). §4.4: G4h comes from the G1h-02 records. |
| 2026-10-28 | §4.1: amended the first-tail compression mixes and fixed the meaning of "near" in the context profiles (requirements D-G1H-GRID-AMENDMENT). |
| 2026-10-27 | Created: C1 results, the exact grids of G1h-G5h and G1-G5, and the GPU-hour estimates. |

## 1. Result

- C1 rebuilt the workload of every DP lane of the three cases from their native serving counters and lifecycle records: H200 P2-32 (DP2 / EP16), H200 P2-64 (DP4 / EP32) and H800 P0 (DP2 / EP16), each with TP8 per lane, a PREFILL role and a unified DECODE role.
- The existing data covers almost none of these coordinates in the mode they run in. Every GPU job of design §7 is needed. Section 4 gives the exact grids; they replace the planning figures of design §7.
- Estimated GPU time (section 5): H200 about 31 of the 45 GPU-hour cap, H800 about 14.5 of 21. These are estimates. Pilot P1h measures the real time per attention window and replaces them.
- Findings for the Frontier side (section 6):
  - the logits gather is eager;
  - the cost of KV-block zeroing depends only on the total number of new block ids in a forward;
  - the V4 SP payload builder admits MONOLITHIC only (file entry 8, already planned).

## 2. Inputs, commands and outputs

Script: [evidence/coverage/c1_workload.py](evidence/coverage/c1_workload.py) (476 lines). For each case it reads:

- the **reference run**: the native serving counters of every lane. Per forward they give the mode, the descriptor, the lane's own tokens and requests, and the synchronized widths of all lanes. No counter dropped a row: `dropped_rows` is 0 on every rank file.
- the **lifecycle run**: a separate native run of the same workload that records each request's cached tokens, new tokens and new blocks per scheduler step;
- the existing attention rows, collective curves and MoE scope rows of design §2.

Commands (repository root, Python 3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`, `ulimit -v 31457280`):

```bash
D=/data/ycfeng/tmp/glm53_dsv4
E=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/evidence/coverage
P6=$D/v4_attention_cwd02_results/P6144/compute/h200/deepseek-ai/DeepSeek-V4-Flash
Q1=$D/v4_h800_q1_b36c2b9_results/Q1_6143_7166/compute/h800/deepseek-ai/DeepSeek-V4-Flash
COMMON="--gemm-stream-threshold 1024 --shared-expert-stream-threshold 256"
python $E/c1_workload.py --case p2_32 --output $D/h3o_b/c1/p2_32 $COMMON \
  --reference $D/h200/downloads/dsv4-h200-p2-canonical-20261025-01/artifacts \
  --lifecycle $D/h200/downloads/dsv4-h200-p2-lifecycle-20261025-01/artifacts \
  --attention P6144_kernel_only=$P6/attention_kernel_only.csv --attention P6144_cuda_event=$P6/attention.csv
python $E/c1_workload.py --case p2_64 --output $D/h3o_b/c1/p2_64 $COMMON \
  --reference $D/h200/downloads/dsv4-h200-p2-64-canonical-20261025-01/artifacts \
  --lifecycle $D/h200/downloads/dsv4-h200-p2-64-lifecycle-20261026-01/artifacts \
  --attention P6144_kernel_only=$P6/attention_kernel_only.csv --attention P6144_cuda_event=$P6/attention.csv
python $E/c1_workload.py --case p0 --output $D/h3o_b/c1/p0 $COMMON \
  --reference $D/native06_case_20261023/runs/groundtruth_clean/dsv4-h800-p0-canonical-20261021-06 \
  --lifecycle $D/h4b/downloads/dsv4-h800-p0-lifecycle-20261025-01/artifacts \
  --attention W6_kernel_only=$D/w6_dsv4_profiles/attention_kernel_only.csv \
  --attention q1_kernel_only=$Q1/attention_kernel_only.csv --attention q1_cuda_event=$Q1/attention.csv \
  --collectives data/profiling/network/h800_dgx/glm_p0_ep16_20261021b/summary.csv \
  --moe-scope task_memory/task_2026-10-23_v4_remaining_measurements/moe/results/primary_acceptance.json
```

Outputs, copied from `/data/ycfeng/tmp/glm53_dsv4/h3o_b/c1/`: `evidence/coverage/{p2_32,p2_64,p0}.txt` (readable) and `.json` (records), plus the two reducers used for the tables below, `summarize.py` and `union.py`. Both reducers read `<case>.json` from the working directory, so run them from `evidence/coverage/`.

The reference and lifecycle runs are different executions of the same workload, so the request counts per forward differ a little between them (section 3.1). The grids therefore use the union of both runs plus a margin.

## 3. Checks and confirmations

### 3.1 Lifecycle reconstruction

C1 replays each lifecycle record and compares the cached and new token counts of every request with the scheduler output.

**Result:** 0 cached-count mismatches and 0 token-count mismatches, on every lane of all three cases.

Requests per decode lane and forward (`b`):

| Case | Reference run | Lifecycle run | Forwards with 2 or more first tails (b, tails) |
| --- | --- | --- | --- |
| P2-32 | 1-11 | 1-13 | (4, 4), (5, 2), (8, 3) |
| P2-64 | 1-9 | 1-9 | (2, 2), (3, 2), (3, 3) |
| P0 | 1-11 | 1-11 | (5, 2), (7, 2) |

Every prefill forward has one request: query 6,144 tokens, no cached tokens.

### 3.2 KV-block zeroing

- **Count rule (confirmed).** For each KV cache group, new blocks = ceil(K_new / B) − ceil(K_prev / B), with block sizes B = (256, 64, 64, 4, 8) tokens.
- **Prefill.** Every prefill forward zeroes 2,520 blocks, split by group as (24, 96, 96, 1,536, 768).
- **Decode.** The zeroed blocks per forward and the largest count of any single group:

  | Case | Blocks per forward | Largest count per group |
  | --- | --- | --- |
  | P2-32 | 0-26 | (4, 4, 4, 7, 7) |
  | P2-64 | 0-16 | (3, 3, 3, 6, 4) |
  | P0 | 0-17 | (2, 2, 2, 7, 5) |
- **Kernel.** In the pinned vLLM source, `KVBlockZeroer.zero_block_ids` zeroes the flat list `new_block_ids_to_zero` with one `_zero_kv_blocks_kernel` launch per forward. The launch grid is (number of block ids, number of segments, chunks per page), and the segments cover every KV tensor (`vllm/v1/worker/utils.py:211-232`, called at `vllm/v1/worker/gpu/model_runner.py:1038`).

So the cost depends only on the number of block ids, not on which group each block belongs to. G2 measures the zeroing by total block count. C1 still records the per-group counts as evidence.

### 3.3 Block-table writes

- **Kernel.** All the writes of a forward go through one fused `_apply_write_kernel` launch, with one program per (group, row) write, plus the host-to-device copies of the contents (`vllm/v1/worker/gpu/buffer_utils.py:211-271`). C1 records each forward's writes as (requests with writes, total entries).
- **Continuing requests.** In decode, at most 7 requests write in one forward. The largest write count by continuing requests is 26 entries (P2-32), 16 (P2-64) and 17 (P0).
- **New requests.** A new request writes its full row, 2,520 entries. The largest number of new requests in one forward is 4 in P2-32 (10,080 entries), 3 in P2-64 and 2 in P0.
- **Prefill.** Every prefill forward writes (1 request, 2,520 entries).

### 3.4 Compression-group writes

A request writes a compressed entry for ratio r when floor(new / r) > floor(old / r), for r ∈ {4, 128}. In decode, the most requests in one forward that write at each ratio, given as (ratio 4, ratio 128) per b:

| b | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2-32 | (1,1) | (2,1) | (3,1) | (4,4) | (4,2) | (5,2) | (6,2) | (7,4) | (6,2) | (7,2) | (7,2) | (7,2) | (6,1) |
| P2-64 | (1,1) | (2,2) | (3,3) | (4,3) | (4,2) | (5,3) | (6,3) | (6,2) | (3,1) | — | — | — | — |
| P0 | (1,1) | (2,1) | (3,1) | (3,1) | (4,2) | (5,2) | (5,2) | (5,2) | (6,2) | (7,2) | (7,2) | — | — |

Every prefill forward writes at both ratios: (1, 1).

### 3.5 The logits gather is eager

C1's first version counted the logits gather as part of the FULL or PIECEWISE graph. That was wrong: `compute_logits` runs inside `sample()`, after the graph replay (`vllm/v1/worker/gpu/model_runner.py:1333-1340, 1707-1764`).

R1 agrees: it places the LM head and the logits all-gather in the post-forward segment (design §3.3). Design §4 already has the logits gather after the model, so only C1 was wrong. It was corrected and all three cases were rerun.

### 3.6 Excluded forwards

- **Start-up warm-ups.** C1 excludes the forwards that run before serving starts. These are the dummies with more than one own token, for example NONE W16, PIECEWISE W256 and FULL W128: five per lane in every case (section A2 of the outputs).
- **Prefill forward counts.** On each prefill rank, the native counters show 13,460 successful forwards in both P2-32 and P0, and 12,468 in P2-64. Every rank file has 0 dropped rows, so the equal counts on two different deployments are a native observation, not a reading error.

## 4. Missing coordinates and the recommended grids

Here a coordinate is **covered** only by rows in the same mode (FULL, PIECEWISE or NONE) and at the same shape. Eager rows cannot price a FULL or PIECEWISE coordinate, because the captured kernels differ from the eager ones (design §5 stage 1a). Notation: W is the descriptor width in tokens, R the requests the attention sees, and "own" the lane's real requests. All grids are per lane.

### 4.1 Attention (G1h, G1)

Observed decode-role coordinates (W: range of own requests):

| Class | P2-32 | P2-64 | P0 |
| --- | --- | --- | --- |
| FULL serving (R = W, padding requests have sequence length 0) | W1: 1; W2: 1-2; W4: 2-4; W8: 2-8; W16: 8-11 | W1: 1; W2: 1-2; W4: 1-4; W8: 2-8; W16: 4-9 | W1: 1; W2: 1-2; W4: 1-4; W8: 2-8; W16: 7-11 |
| PIECEWISE serving (R = own; own requests with a first tail in brackets) | W1: 1 (1); W2: 1-2 (2); W4: 2-4 (3-4); W8: 4-8 (5-8); W16: 8-11 (9-11) | W1: 1 (1); W2: 1-2 (2); W4: 1-4 (2-4); W8: 3-8 (4-8); W16: 4-9 (5-9) | W1: 1 (1); W2: 1-2 (2); W4: 2-4 (2-4); W8: 3-8 (4-8); W16: 8-11 (9-11) |
| FULL dummy (one own token, R = W) | W1, 2, 4, 8 | W1, 2, 4 | W1, 2, 4 |
| PIECEWISE dummy (R = 1, query W, no cached tokens) | W1, 2, 4, 8 | W1 | W1 |

- Distinct decode serving classes (mode, W, R, own tokens, own requests, has_prefill), over all lanes of the reference run: P2-32 has 43 (17 FULL, 26 PIECEWISE), P2-64 has 54 (20 FULL, 34 PIECEWISE) and P0 has 48 (19 FULL, 29 PIECEWISE).
- **Decode rows.** Every query is 1 token. Cached tokens per request: 6,143-7,166. A first tail is decode-shaped too (query 1, 6,143 cached tokens) and carries `has_prefill`, which adds `prepare_prefill_inputs`.
- **Prefill role.**
  - serving: NONE W6144, one request, 6,144 query tokens, no cached tokens;
  - dummies: FULL W1 (12,920 per lane in P2-32 and P0, 11,931 in P2-64) and NONE W1 on the short-context branch (209 per lane in P2-32 and P0, 314 in P2-64).

Coverage today:

- **H200.** Only the prefill NONE 6,144 window is covered, by P6144 (6 KERNEL_ONLY and 6 CUDA_EVENT rows). Those rows keep no kernel intervals, so the window is measured again.
- **H800.** No coordinate has rows in its own mode. W6 (7,116 eager rows) has no row at any of these shapes, including the 6,144 prefill with no cached tokens. q1 has eager rows at b = 1 only (6,138 decode rows, plus 6 first-tail rows at 6,143 cached tokens). Those rows cannot price the FULL or PIECEWISE coordinates.

Grid, the same for G1h and G1. Each window is measured for every layer variant and planner role in the attention plan, as KERNEL_ONLY with kernel intervals, with CUDA_EVENT for validation.

1. **FULL serving.**
   - Classes: W ∈ {1, 2, 4, 8, 16} with own = 1..W (31 classes), plus W24 with own = 17-24 as a margin (8 classes).
   - About 10 windows per class: 3 context profiles (all requests at 6,144 cached tokens; all at 7,166; spread evenly over 6,144-7,166) × 3 compression mixes (no compressed writes; ratio-4 writes at the observed maximum for that b; ratio-128 writes at their maximum, from section 3.4).
   - About 390 windows.
2. **PIECEWISE serving.**
   - The same 39 classes, each without and with one first tail, at the 3 context profiles: about 234 windows.
   - Plus 7 multi-tail windows at the observed (b, tails): (2, 2), (3, 2), (3, 3), (4, 4), (5, 2), (7, 2), (8, 3).
3. **Dummies.**
   - Decode: FULL with R = W, and PIECEWISE with R = 1, for W ∈ {1, 2, 4, 8, 16, 24}.
   - Prefill: FULL W1 and NONE W1 on the short-context branch.
   - 14 windows.
4. **Prefill serving.** NONE W6144 with no cached tokens: measured again with intervals on H200, measured for the first time on H800.

Total: about 650 windows per device.

**Amendment (2026-10-28, requirements D-G1H-GRID-AMENDMENT).** The G1h grid as built by B11 applies the 3 compression mixes to every FULL and PIECEWISE class and profile, including the first-tail windows. Two rules complete it:

- A first tail (6,143 cached tokens, query 1) writes at both ratios. With t first tails, each mix target (n4, n128) becomes (max(n4, t), max(n128, t)), and duplicates are removed.
- "Near" a profile target means within 127 cached tokens (one ratio-128 group): low 6,144-6,271, high 7,039-7,166. Continuing requests stay inside 6,144-7,166.

**Why the W24 margin.** The largest observed lane holds 13 requests. Frontier places requests on lanes with its own scheduler, so one lane can hold a few more. The W16 classes cover a lane with up to 16 requests, and W24 covers one with up to 24. A lane with more requests raises the typed coverage error (design §3.7 item 4). That error is the intended signal, not a failure to hide.

This replaces the design's planning figure of "decode batch up to 40".

### 4.2 V4 operators (G2h, G2)

Row axes that the cases use. Here N is a lane's synchronized width, L = ceil(N / 8) its rows per rank under SP, and 8L the gathered rows.

| Axis | Operators | Decode | Prefill | Margin |
| --- | --- | --- | --- | --- |
| Rows per rank under SP: L | mHC scopes, shared expert | 1, 2 | 1, 768 | 3, 4 |
| Gathered rows: 8L | attention-side rows | 8, 16 | 8 (dummy), 6,144 | 24, 32 |
| Descriptor rows: W | embedding, final norm, mHC head (corrected 2026-10-28: they run after the gather) | 1, 2, 4, 8, 16 | 1, 6,144 | 24 |
| Real requests: b | LM head, sampling | 1-13 | 1 | up to 24 |
| New block ids per forward | `kv_cache_block_zero` | 0-26 | 2,520 | up to 32 |
| (requests with writes, entries) | `dsv4_block_table_write` | the observed pairs: up to (7, 26) for continuing requests, plus 2,520 entries per new request, up to 4 new requests | (1, 2,520) | requests up to 8, continuing entries up to 32 |
| Synchronized descriptor | post-DP preparation kernels | the FULL and PIECEWISE descriptors above | NONE 6,144, FULL W1 dummy | as section 4.1 |

Design §2 lists no existing rows for these operators on either device, so the whole grid is new on both.

### 4.3 Routed MoE and gating (G3h, G3)

Rank classes are given as (rows routed to the rank, local experts that get one extra row, base rows per expert). With balanced routing, expert e gets floor(6M / 256) rows, plus one more row when e < (6M mod 256).

| Topology | Gathered M | Classes |
| --- | --- | --- |
| EP16 (P2-32, P0; 16 local experts) | M16, M32 | (16, 16, 0) and (0, 0, 0) |
| | M6152 | (2,320, 16, 144) and (2,304, 0, 144) |
| | M12288 | (4,608, 0, 288) |
| EP32 (P2-64; 8 local experts) | M32 | (8, 8, 0) and (0, 0, 0) |
| | M64 | (8, 0, 1) and (16, 8, 1) |
| | M6168 | (1,152, 0, 144) and (1,160, 8, 144) |
| | M18440 | (3,456, 0, 432) and (3,464, 8, 432) |
| | M24576 | (4,608, 0, 576) |

- **H200.** No rows; every class is new.
- **H800.** The accepted EP16 rank-0 rows (`primary_acceptance.json`: isolated, eager, M = 16/32/6152/12288) match the rank-0 class of each M: (16, 16, 0), (2,320, 16, 144) and (4,608, 0, 288). They are training candidates only. G3 measures the other classes, and the rank-0 classes again if the candidate rows lack the typed contract the new loader requires.
- **The zero class (0, 0, 0) is measured, not assumed to cost nothing.** In a graph, the routed path still launches its kernels with zero rows.
- **Margins.**
  - EP16: L = 3, 4 (M48: (32, 16, 1) and (16, 0, 1); M64: the same two classes).
  - EP32: L = 3, 4 (M96: (24, 8, 2) and (16, 0, 2); M128: (24, 0, 3)), plus the DP4 forward with two prefill lanes, M12304. That forward is reachable but was not observed. Its classes are (2,312, 8, 288) and (2,304, 0, 288).
- **Gate.** The gate projection and top-k run on L rows: decode L ∈ {1, 2}, prefill L ∈ {1, 768}, margin {3, 4}.

### 4.4 Gaps between kernels (G4h, G4)

C1 does not change G4. The four V4 gap values come from graph replays and eager segments on the reduced engine (design §3.3), with segment classes from R1.

On H200 the G1h-02 replay records give them on CPU (design §6, implementation notes for stage 2), so no separate G4h job runs.

### 4.5 Collectives (G5h, G5)

Input bytes per rank. An EP vector lists the inputs of each DP lane in lane order; the 8 TP ranks of a lane have equal inputs. For `dsv4_ep_output_reduce_scatter`, the first element is the input and the rest are the outputs per lane.

| Collective | Mode | P2-32 (TP8, EP16) | P2-64 (TP8, EP32) |
| --- | --- | --- | --- |
| embedding all-reduce | graph | 8,192, 16,384, 32,768, 65,536, 131,072 | same |
| | eager | 8,192, 50,331,648 | same |
| SP attention all-gather | graph | 8,192, 16,384 | same |
| | eager | 8,192, 6,291,456 | same |
| SP attention reduce-scatter | graph | 65,536, 131,072 | same |
| | eager | 65,536, 50,331,648 | same |
| SP final mHC all-gather | graph | 32,768, 65,536 | same |
| | eager | 32,768, 25,165,824 | same |
| logits all-gather (eager in every mode, section 3.5) | eager | 32,320 × b, b = 1-11 (13 in the lifecycle run) | 32,320 × b, b = 1-9 |
| EP activation all-gatherv | graph | [8,192, 8,192], [16,384, 16,384] | the 4-lane equivalents |
| | eager | [6,291,456, 6,291,456], [6,291,456, 8,192], [8,192, 6,291,456] | 6 orderings of 6,291,456 and 8,192 |
| EP ids and weights all-gatherv (same bytes; one measurement serves both) | graph | [24, 24], [48, 48] | the 4-lane equivalents |
| | eager | [18,432, 18,432], [18,432, 24], [24, 18,432] | 6 orderings of 18,432 and 24 |
| EP output reduce-scatterv | graph | [131,072, 8,192, 8,192], [262,144, 16,384, 16,384] | [262,144, 8,192 × 4], [524,288, 16,384 × 4] |
| | eager | [100,663,296, 6,291,456, 6,291,456], [50,397,184, 6,291,456, 8,192], [50,397,184, 8,192, 6,291,456] | 6 orderings, for example [151,060,480, 8,192, 6,291,456, 6,291,456, 6,291,456] |

P0 (H800) uses the same points as P2-32.

Coverage today:

- **H200.** No curves exist, so every point is new.
- **H800.**
  - **Below the measured range:**
    - the embedding all-reduce at 8,192 (graph and eager);
    - the SP attention all-gather at 8,192 (graph and eager);
    - the SP attention reduce-scatter at 65,536 (graph and eager);
    - the graph EP activation all-gatherv at [8,192, 8,192];
    - the graph ids and weights all-gatherv at [24, 24] and [48, 48];
    - the graph reduce-scatterv at [131,072, 8,192, 8,192].
  - **Unequal vectors**, with no measurement at all: the eager [6,291,456, 8,192] and [8,192, 6,291,456], their ids and weights equivalents, and the two unequal reduce-scatterv vectors.
  - Every other point lies inside or exactly on the existing uniform curves, including the logits all-gather.

Grid:

- **G5h, 16 GPUs** (two nodes):
  - every TP8 point above, plus a margin for L = 3, 4 (payloads 24,576 and 32,768 for the hidden-row collectives, and the matching reduce-scatter and embedding points);
  - the logits all-gather for b = 1-24;
  - the EP16 graph vectors, plus L = 3, 4;
  - all 4 orderings of the two eager sizes for each EP16 collective.
- **G5h, 32 GPUs** (four nodes):
  - the EP32 graph vectors, plus L = 3, 4;
  - all 16 orderings of the two eager sizes for each EP32 collective. Only 6 were observed, but the other orderings are reachable, and each costs seconds.
- **G5, 16 H800 GPUs:** the below-range points and the unequal vectors above, with `vllm_forward_collectives.py`. In-range uniform points keep the existing curves.

## 5. GPU-hour estimates

These are estimates, not measurements. The basis of each is stated below, and P1h replaces them with the measured time per window.

| Job | GPUs | Content (section 4) | Estimate | Basis | Cap |
| --- | --- | --- | --- | --- | --- |
| P1h | 1 | Replay route and pilot (design §5 stage 1a, stage 1) | 2 | design cap | 2 |
| G1h | 1 | About 650 windows × 6 rows (layer variant × planner role) = about 3,900 rows | 1.5 | The H800 q1 job measured 6,144 eager coordinates, each with both methods (12,288 rows), in a 512.5 s worker phase including start-up: about 0.08 s per coordinate (`test_report_2026-10-21_v4_h800_decode_primary.md`). Graph windows need capture, node queries and replays, assumed 5-10 times an eager coordinate | G1h-G4h: 7 |
| G2h | 1 | About 20 operators × about 25 row points, eager and graph-replay forms | 1 | eager-row rate as above | |
| G3h | 1 | EP16: 7 observed + 4 margin (M, class) pairs; EP32: 9 observed + 5 margin pairs; gate L grid; one engine start per EP topology | 2 | The routed-weight set-up per class dominates. Assumed a few minutes per class | |
| G4h | 1 | Four V4 gap values | 0.5 | design cap | |
| G5h | 16, then 32 | Section 4.5 | 24 | Assumed 0.5 h per job, dominated by start-up and NCCL initialization; the points take seconds each. At 0.75 h per job, 36 | 36 |
| **H200 total** | | | **about 31** | | **45** |
| P1 | 1 | as P1h on H800 | 2 | design cap | 2 |
| G1 | 1 | as G1h | 1.5 | as G1h | 3 |
| G2 | 1 | as G2h | 1 | as G2h | 1.5 |
| G3 | 1 | EP16 classes and margin; rank-0 rows are only candidates | 1.5 | as G3h | 2 |
| G4 | 1 | as G4h | 0.5 | design cap | 0.5 |
| G5 | 16 | below-range points and unequal vectors | 8 | 0.5 h per job | 12 |
| **H800 total** | | | **about 14.5** | | **21** |

## 6. Findings for the Frontier side

1. **PDD payloads.** `_dsv4_sp_local_hidden_bytes` in `frontier/operators/families.py` admits MONOLITHIC only. Every SP payload of these cases comes from the PREFILL or unified DECODE role, so file entry 8 (PDD admission) is required before any V4 replay prices a collective.
2. **Zeroing key.** The zeroing cost depends on the total number of new block ids per forward (section 3.2). Entry 15's producer and entry 3's pricing use that count. The per-group split is not a model input.
3. **Logits gather.** It is eager in every mode (section 3.5). This agrees with design §4. Entry 3 prices it on the eager post-forward path, with the real requests' rows.
4. **H800 prefill attention.** The 6,144 prefill with no cached tokens has no H800 row, so G1 must measure it. W6 does not cover it.
5. **Prefill dummies (design §10, risk 7, unchanged).**
   - In the canonical runs, each prefill lane runs 12,920 FULL W1 dummies (P2-32, P0) or 11,931 (P2-64), plus 209 or 314 NONE W1 dummies. The profiled runs of the mode census show 8,408-9,455 FULL dummies.
   - Frontier's dummy frequency follows its existing lifecycle model. A11 reports any difference and H3o does not correct it.
6. **Different executions.** The lifecycle runs reach 13 requests per lane where the reference run reaches 11 (P2-32). Coverage keys must not assume the reference run's maximum. The grids use the union plus the W24 margin.

## 7. Limits

- C1 counts coordinates from native runs. It does not show that Frontier's replays stay inside them. A Frontier forward outside the grid raises the typed coverage error, and A11 compares the replays' mode and width census with the native one.
- Context profiles and compression mixes sample the 6,143-7,166 range. Interpolation between them is the predictor's job (file entry 1), checked by A6.
- The GPU-hour figures are estimates (section 5).
