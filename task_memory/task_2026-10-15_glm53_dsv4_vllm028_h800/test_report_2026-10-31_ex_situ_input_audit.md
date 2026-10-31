# Test report: ex-situ audit of the A10 inputs (2026-10-31)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Created: inventory of every input of the B28 A10 replays, classification by origin, findings F1-F5, and the repair proposal. |

## Scope and criterion

Decision 44 (D-EX-SITU-AUDIT, requirements.md) asks whether the A10 numbers come from a normal Frontier simulation:

- every cost is composed from profiling CSVs that one GPU can produce, with a modeled workflow;
- no input depends on data from the target large cluster, such as per-layer or per-step times measured at 32 or 64 GPUs and stored as entries, or parameters read from native E2E runs.

Audited runs: the B28 replays `stage4/a10/p2_32_warm_k19b28` and `stage4/a10/p2_64_warm_k19b28` (source `ee29d548`, receipts `receipt.json` in each run root). P2-32 runs two roles of TP8 x DP2 (EP16, 16 GPUs each); P2-64 runs two roles of TP8 x DP4 (EP32, 32 GPUs each).

## Verdict: FAIL

The compute side passes: every compute table comes from one-GPU profiling jobs, and no whole-step table is bound. Two inputs fail the criterion:

- **F1:** every collective price comes from tables measured on the target topologies (16 and 32 GPUs) at the workload's exact message sizes.
- **F2:** the run-ahead depth K = 19 was observed in the instrumented native P2-64 run.

Ingress option A stays blocked (decision 44). The current A10 numbers are in-situ calibrated results. They must not be presented as ex-situ predictions.

## 1. Input inventory

| Input | Consumer | Origin | Class | Verdict |
| --- | --- | --- | --- | --- |
| `attention_graph_replay_kernel_only.csv` (7,105 rows) | V4 attention scopes, both roles | G1h-02 and the G1h supplement: 1-GPU jobs (`"gpu": 1`), vLLM engine at TP1 with per-rank TP8 shapes from the case plan | single-GPU profiling | PASS |
| `kernel_gap.csv` (4 rows) | gaps between kernels | G1h-02, same 1-GPU replay | single-GPU profiling | PASS |
| `linear_op_kernel_only.csv` (41 rows) | linear, mHC, shared-expert operators | G2h-01: 1-GPU jobs, including the EP16 shared-expert shapes | single-GPU profiling | PASS |
| `moe_kernel_only.csv` (33 rows) | routed-expert gate, shuffle, grouped GEMM | G3h: 1-GPU job (0.0369 GPU-hours), per-rank token classes from Frontier's balanced-routing algebra (`moe_rank_plan.py`) | single-GPU profiling | PASS |
| `model_runner_block_kernel_only.csv` (197 rows), `model_runner_replay_kernel_only.csv` (2,941 rows) | per-operator runner scopes (block zeroing, block-table write, input preparation, sampling) | G2h-02/03/04: 1-GPU jobs at TP1 | single-GPU profiling; per-operator rows, not whole-step rows | PASS |
| Step-cost table (`dsv4_step_cost.json`) | none | the driver checks it is not bound | receipt `a1`: `step_cost_table_bound_by_predictor` false for PREFILL and DECODE, no step-cost file beside the inputs | PASS |
| Dummy predictor mode | none | off | receipt `a1`: `enable_dummy_mode` false for both roles | PASS |
| `data/profiling/network/h200_dgx/summary.csv` (384 rows), `measured` CC backend | every TP and EP collective of both roles | G5h-16 (TP8 DP2 EP16, 16 GPUs) and G5h-32 (TP8 DP4 EP32, 32 GPUs) jobs, plus their isolated reruns | **measured on the target cluster topology** | **FAIL (F1)** |
| `max_forwards_in_flight` K = 19 | DP host run-ahead (B26) | native P2-64 wave run `exp-1030-144848-164721`, A10 report section 3.5 observation 4 | **observed in a native 64-GPU E2E run** | **FAIL (F2)** |
| KV transfer: analytical NIXL, 126.2 Gbit/s and 0.0647 ms | P to D KV transfer | fit to an H800 two-node NIXL read microbenchmark (`data/profiling/network/h800_dgx/nixl_summary.csv`, 8 GPU pairs) | small link microbenchmark, independent of the workload, but measured on H800 | NOTE (F3) |
| `num_blocks` 109,934 (P2-32) and 112,914 (P2-64), `num_blocks_mode` explicit | decode KV admission | native startup logs (`downloads/facts.json`) | value read from native runs; does not bind | NOTE (F4) |
| Attention case grid | which coordinates G1h measures | planned to cover the window classes of the native C1 census (P2-32 43/43, P2-64 54/54) | native data used to choose profiling points, not as values | NOTE (F5) |
| Workload trace `canonical_fixed_6144_1024_req128_qps1` | arrivals and lengths | the same trace the native client sends | workload input | PASS |
| Deployment settings: attention backend `FLASHMLA_SPARSE_DSV4`, fp8 KV and indexer KV, `full_and_piecewise`, capture sizes 1, 2, 4, 8, 16-256 step 8, `max_num_seqs` 128, decode batch queue 2, balanced routing | scheduler and predictor selection | vLLM launch configuration and vLLM code rules | deployment configuration | PASS |
| Modeled mechanisms: DP lockstep and 32-step DP sync, NIXL completion at the engine step, round-robin DP placement | workflow | vLLM 0.28 source | modeled workflow | PASS |
| 3.27 ms prefill scheduling wait (A10 report section 3.1) | none (an output) | Frontier's `request_waiting_time_prefill`; no such constant in `frontier/` (`grep` finds none) | simulated result | PASS |
| V4 host time zero (decision D2 (c)) | host time | modeling simplification | no data input; a fidelity gap, not a shortcut | PASS |

## 2. F1: the collective table is measured on the target topology

Evidence:

- `summary_provenance.json` (beside the table): rows 0-245 are a byte copy of the G5h-16 `_02` run (`exp-1028-231342-951824`, TP8 DP2 EP16), rows 246-383 of the G5h-32 `_01` run (`exp-1028-231408-632060`, TP8 DP4 EP32). B28 replaced 18 EP16 rows and 90 EP32 rows with the isolated reruns (`exp-1029-012627-132543`, `exp-1030-234535-245357`). The launch files under `h3o_operator_costs/g5h/launch/` request 16 and 32 GPUs.
- The scenarios replay the V4 forward's own collective sequence with the workload's per-lane token counts: graph lanes `graph_t1`-`graph_t32`, eager lane tuples such as `eager_t6144_1_1_1`, and logits batches `logits_b1`-`logits_b24`.
- Every collective prediction of both runs is a byte-exact hit on a measured row (`ex_situ_audit/receipt_checks.txt`): P2-32 44 of 44 priced points, 6,310,644 of 6,310,644 predictions; P2-64 38 of 38, 9,002,176 of 9,002,176. No price is interpolated.
- The measured backend prices uneven EP dispatch and combine only from exact rows keyed by the ordered per-lane token vector (`frontier/cc_backend/backends/measured_cc_backend.py:3-5,58-70,181-186`). It cannot price a lane mix that was not measured on that cluster.
- B28 chose between two cluster measurements of the same topology because the isolated rows matched the native critical path. The change moved P2-32 by −12.4 ms and P2-64 by +4.6 ms per prefill forward (A10 report section 5.7).

How much of the result the table carries (`ex_situ_audit/comm_share.py`, output `comm_share.txt`): the share of the collective scopes in the composed forward time.

| Forward class | P2-32 mean ms | P2-32 collectives | P2-64 mean ms | P2-64 collectives |
| --- | --- | --- | --- | --- |
| Prefill, 6,144-token lane | 209.96 | 70.29 ms (33.5%) | 279.87 | 153.26 ms (54.8%) |
| Decode, real | 19.88 | 9.59 ms (48.2%) | 25.30 | 16.79 ms (66.4%) |
| Prefill dummy | 21.76 | 10.37 ms (47.6%) | 30.82 | 20.03 ms (65.0%) |

On P2-64 the EP dispatch AllGatherV and combine ReduceScatterV alone take 16.05 ms of the 25.30 ms decode step. The TPOT, E2E and throughput PASS results therefore rest mostly on these cluster-measured latencies.

Size of the gap an ex-situ model must close (`ex_situ_audit/price_compare.py`, output `price_compare.txt`; the analytical backend with its default parameters of 100 Gbit/s, 1 µs and 600 Gbit/s intra-node, which are not calibrated for H200):

| Call | Bytes | Measured | Default analytical |
| --- | --- | --- | --- |
| Decode EP32 AllGatherV | 8,264 | 0.2561 ms | 0.0044 ms |
| Decode EP32 ReduceScatterV | 262,144 | 0.1161 ms | 0.0044 ms |
| Decode EP16 AllGatherV | 8,264 | 0.1408 ms | 0.0027 ms |
| Prefill EP32 ReduceScatterV, 6,144-token lane | 50,528,256 | 2.1720 ms | 0.6537 ms |
| Prefill TP8 SP AllGather | 6,291,456 | 0.1616 ms | 0.5882 ms |

Inference: the small inter-node EP messages are latency-bound. A ring AllGather over 32 ranks takes 31 steps, so the measured 0.256 ms means about 8 µs per step. An ex-situ model needs a per-step latency term for the inter-node path; a bandwidth-only model with the default 1 µs latency is about 58 times too fast on the decode EP32 dispatch.

## 3. F2: K = 19 is a native observation

- Decision 43 and the B26 assignment set K from the depth measured in the native P2-64 wave run (A10 report section 3.5, observation 4: median 18.56 unfinished dummies at the switch from fast to paced host, 117 of 122 waves in 18.04-18.90).
- A10 report section 3.5 states that the cause is not identified: no call in the vLLM source sets the limit. The hypothesis is the CUDA launch queue capacity, which is not verified.
- Effect: with K unset, the P2-64 TTFT error was −43.85% (A10 report section 2.2). With K = 19 it is −10.60% (section 5.7). The TTFT result of P2-64 rests mostly on this value.

## 4. Notes F3-F5

- **F3, KV transfer.** The NIXL fit (126.2 Gbit/s, 0.0647 ms) comes from a two-node point-to-point microbenchmark. The method is ex-situ, because it does not depend on the workload or the target scale, but it was measured on H800 nodes and is used for H200. On P2-64 the KV transfer is 87.3 ms of the TTFT (A10 report section 2.2, the run before B26).
- **F4, KV block count.** The explicit values come from native startup logs. They do not bind in these runs: 0 preemptions in both runs (`system_metrics.json`), and about 21 (P2-32) and 27 (P2-64) requests are in flight over all decode lanes, against a maximum concurrency of 37.4 and 38.4 per lane (A10 report section 2.2). The metrics should not change if Frontier's memory planner sets the value. This is not verified, and it is not known whether the planner supports the five V4 KV groups.
- **F5, attention grid.** Native census classes chose which coordinates the one-GPU job measures. The values are one-GPU measurements, and B22 also closed coverage against Frontier's own census of the same trace. No change to the numbers. Future grids can be planned from Frontier's census alone.

## 5. Repair proposal

Both repairs replace inputs that YC approved earlier: the measured backend (D-W7-COMM-APPROVED item K4 and Q5 = B) and K (decision 43). They also add new modeling, which decision 25 reserves for YC. They therefore need YC's decision.

**F1, recommended: a topology-aware collective model whose inputs are only hardware link parameters.**

1. Intra-node TP8 collectives: a generic size sweep on one 8-GPU H200 node, priced by interpolation. This is Frontier's standard network profiling scope (`network/<device>/all_reduce.csv`). The sweep uses a byte grid, not the workload's sizes.
2. Inter-node EP AllGatherV and ReduceScatterV: composed from the per-step latency and bandwidth of the NVLink and NIC paths, measured by a two-node point-to-point test or taken from hardware specifications. The candidate backends are `collective_sim`, which `calibration.md` names, or `astra_sim_analytical`. Uneven lanes are priced by the largest rank's bytes, as in D-UNEVEN.
3. The G5h tables become held-out validation only, not inputs.
4. Cost:
   - code: an uneven-collective adapter in the selected backend, plus the `collective_sim` build if that backend is chosen;
   - GPUs: about 4 GPU-hours for the one-node sweep and about 2 GPU-hours for the two-node test (estimates);
   - then four A10 replays on CPU.
5. Risk: on P2-64 the decode step is 66% collective time, so the latency term of the new model decides TPOT. The current TPOT, E2E and throughput PASS may not hold.

**F2, recommended: measure the launch-blocking depth on one GPU.**

1. Replay the captured V4 dummy FULL graph back to back without synchronization on one H200, using the TP1 engine of G1h/G2h. Record how many replays are unfinished when the launch starts to block.
2. If the depth matches 19, the value is a per-GPU property and K comes from that one-GPU measurement. If it does not match, the 19 depends on the cluster run and K stays unexplained.
3. Cost: about 0.2 GPU-hours.
4. Open design point: the one-GPU graph lacks the cross-node collective kernels, which may change the queue use per replay.

**F3:** run the NIXL read test on two H200 nodes in the same job as the F1 two-node test.

**F4:** switch to Frontier's planner value, and check that the metrics are identical.

**F5:** no change.

Order: decision on F1 and F2 → {F1 code by Codex, F1/F2/F3 GPU job} → four A10 replays → audit rerun → ingress option A.

## 6. Commands and environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6; `PYTHONPATH` the repository root; `ulimit -v 31457280`.
- `comm_share.py <run roots>` (copy in `h3o_operator_costs/ex_situ_audit/`); output `comm_share.txt`, equal to the first inline run.
- `price_compare.py <run roots>`; output `price_compare.txt`.
- Receipt and collective-ledger checks: `receipt_checks.txt` (from the two `receipt.json` and `collectives.json` files).
- GPU counts of the profiling jobs: `"gpu": 1` in the G1h, G1h supplement, G2h and G3h package plans; 16 and 32 GPUs in the G5h launch files.

## Limits

- The default-parameter analytical prices in section 2 show the size of the gap. They are not a candidate model.
- The CUDA launch queue hypothesis of F2 is not verified.
- F4's no-effect statement is an inference from 0 preemptions and the in-flight count.
