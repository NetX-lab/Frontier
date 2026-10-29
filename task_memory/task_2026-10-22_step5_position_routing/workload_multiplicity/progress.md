# D29-FULL-WORKLOAD-MULTIPLICITY progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Parent resumes after API quota error, aligns clock origins and withdraws the iteration-4 slowdown claim. |
| 2026-10-23 | Record the bounded reads, the source-count probes, and the arithmetic checks. |
| 2026-10-23 | Correction pass: exact phase split, layer-60/61 identity wording, prefill 6128 + D-side 16-token replay, projection relabelling. |

## completed

- Parent correction: approximate launch1792751317.666937, first scheduler
  event1792752430.011742, active iteration4 at1792753515.053171 and execution
  receipt1792753717.842189. Pre-scheduler interval is about1112.345s; iteration4
  has202.789s remaining, not1315s. No anomalous slowdown is established.
  Offsets below start at the first scheduler event, not process launch.
  Byte-derived row equivalents do not constitute a full event census.

- Read `AGENTS.md`, `GLOSSARY.md`, and the parent records
  (`requirements.md`, `plan.md`, `progress.md`, `main08_residual_repro/`,
  `resumed_performance/`, `early_init_repro/report.md`).
- Read the real main08 argv, request input and receipts:
  `d29-parent-position-main08/offline/{command,requests.csv,execution}.json`,
  `resumed_performance/main08_execution.json`.
- Parsed the retained scheduler timeline (52 records) and established:
  - PREFILL schedules `num_tokens=6128`, `token_budget` falls 8192 -> 2064, one
    decision only, batch end at wall 204.132 s;
  - DECODE iteration 0 admits 16 tokens (`batch_num_tokens=16`), iterations 1-4
    admit 1 token each; four DECODE batches complete at wall 418.882, 640.276,
    860.437, 1085.041 s (durations 214.734, 221.392, 220.158, 224.602 s).
- Traced the call chain in source: `decode_collective.py`, `sync_entry.py`,
  `base_cluster_scheduler.py`, `ep_wave_schedule.py`, `ep_wave_metrics.py`,
  `trace_store.py`, `metrics_store.py`, `metrics_config.py`, `simulator.py`,
  `base_execution_time_predictor.py`.
- Counted the MoE layer set with the production resolver on the checked-in
  config (no trainer/constructor): `num_layers 92`, `moe_layers 88`,
  `dense [0,1,2,91]`.
- Measured two bounded 8 MiB trace windows (head and tail):
  - head: 603 rows, 602 PREFILL + 1 metadata header, layers 3-5, shapes
    `[6128, 4096]`;
  - tail: 598 rows, all DECODE, layers 58-60, 13 rows per lane per layer, two
    complete waves at 2,906,520 B and 2,905,320 B.
- Located the first DECODE layer wave in a bounded 2 MiB window at 250 MiB:
  `tensor_shape.input = [16, 4096]`, `effective_total_tokens_compute = 16`,
  `ts_start = 34.0136`, i.e. the 16-token replay chunk.
- Resolved the token bookkeeping from production code:
  `_step5_kv_transfer_prefill_target` (`attention/families.py:572-577`),
  `_set_prefill_compute_target` (`vllm_v1_kv_allocation.py:32-56`),
  `_num_recomputed_tokens` one-token commit (`entities/request.py:1387-1393`),
  and the PD-AF-only handoff-token convention (`kv_arrival.py:61` vs `:98-127`).
- Verified the arithmetic with a standalone calculation: 90,200 waves,
  18,761,600 events, 262.11 GB, closure 103,648 events, range 86.3-101.2 ks,
  zero-pricing 8.69-9.14 ks, required headroom 17.3 / 12.3 ms per wave.

## Corrections applied (this pass)

| Earlier statement | Corrected |
| --- | --- |
| "16 lanes x 5 phases x 13 operators = 208" | 13 is the per-lane operator total; phases contribute 8+1+1+1+2 = 13; 16 x 13 = 208 (144 COMPUTE + 64 COMM). |
| "layer 60 and 61 payloads equal apart from global_layer_id, so layers are emitted as 88 independent payloads" | The 208/208 equality is each layer's replay against its **own** capture; between layers 60 and 61, timings/native contexts/plans/costs match but **144 operator_cost objects differ in global_layer_id**, so proof objects are not interchangeable. |
| "1024 decode tokens x 88 = 90,112 decode plus 88 prefill waves" | Prefill target is 6,128 (6144 - block_size, rounded); the 16 residual prompt tokens run as DECODE batch 0 and commit one token. Total is 1,025 layer sets = 90,200 waves; the volume total is unchanged. |
| "prefill emits one of the 1024 tokens" (assumed) | In pd-disaggregation the handoff-token convention is not used; all 1,024 output tokens are post-prefill. |
| "hard lower bound / cannot fit" | Current-implementation estimate with explicit exclusions; the zero-pricing result is a headroom requirement (17.3 / 12.3 ms per wave), not an impossibility proof. |
| "0.838 ks startup" | Historical model-manager extrapolation; the parent separately reconstructs approximately1112.345s from main08 launch to first scheduler event. |

## in-progress

- None.

## pending

- Root decision on the acceptance route for a workload whose mandatory per-mode
  trace volume is ~262 GB and whose current-implementation wall cost is 24-28 h
  against a 2400 s cap. This task proposes no change to the trace flag, the
  workload, the cap, or any budget.

## Commands actually run (read-only except this records directory)

```bash
# layer set, real resolver, no trainer
cd /data/ycfeng/Frontier/.worktrees/step5-preview && \
timeout 60s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  /data/ycfeng/envs/frontier-py310/bin/python -B -c \
  "from frontier.config.model_config import BaseModelConfig as B; \
   c=B.create_from_name('stepfun-ai/Step-5-Preview'); \
   print(c.num_layers, len(c.get_moe_layer_ids()))"
# -> 92 88

# bounded trace windows: head 8 MiB, tail 8 MiB, one 2 MiB window at 250 MiB
# python3 read-only over op_traces.jsonl; no writes, no full-file parse
```

## Notes / decisions

- The 16-token replay batch committing one decode token is supported by
  `request.py:1387-1393` plus the retained record of 4 emitted tokens after 4
  completed batches. No per-batch token log exists at `log_level=warning`, so
  this is recorded as source-plus-count inference in the limits.
- The iteration-4 slowdown claim is withdrawn after aligning clock origins;
  neither the old1315s window nor its22.7s/wave value supports a diagnosis.
- No scratch files were created outside this records directory, and no replay
  was run: the wave count is proven from source plus retained bytes.
