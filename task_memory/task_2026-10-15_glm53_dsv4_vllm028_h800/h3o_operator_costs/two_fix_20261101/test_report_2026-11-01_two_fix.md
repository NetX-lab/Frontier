# D-WARM-TWO-FIX: NIXL wire pricing, EP dispatch count check, and two CPU replays

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Recorded the NIXL fix and unit regressions, the held-out EP evidence behind D-WARM-EP-DEFER, and the two fresh-cache CPU replays compared with B29. |

## 1. Result and authority

Authority: `requirements.md` D-WARM-TWO-FIX, with item 2 superseded by D-WARM-EP-DEFER.

1. **NIXL wire pricing — implemented, PASS.** Commit `0fb8955d`. Grouped V4 KV transfers now price wire time on selected pages × `physical_block_size_bytes` (1,002,240 B). The recorded payload keeps the structural component bytes.
2. **EP dispatch kernel count — deferred.** Held-out native traces contradict an unconditional count of three, so YC chose to defer (D-WARM-EP-DEFER). The registry keeps `kernel_count=1`.
3. **CPU checks.** Unit regressions: no new failure. Both fresh-cache CPU replays finished (exit 0) on the unchanged B29 inputs and are **bit-identical to B29**: the same forwards, collectives, KV transfer times, per-request latencies and metric values, so every verdict is unchanged.
   - P2-32: all-request gate PASS, warm reading PASS.
   - P2-64: all-request gate FAIL (repeat01 TTFT −12.254941%, the same preserved FAIL as B29), warm reading PASS.
   - Warm reading, both cases: 32/32 metrics PASS, maximum |error| 5.675933%.

   The NIXL fix is in effect (2.570035 → 2.987238 ms per transfer), but it cannot move any READ completion in these runs. A READ is found only at the target lane's next engine-step start after its wire ends, and inside every KV transfer window those step starts are at least 18.72 ms (P2-32) or 24.38 ms (P2-64) apart. No step start falls in the 0.417 ms the fix adds (Section 5.3).

Not done, as approved: host-zero changes, READ timing modeling, offsets, table promotion, new GPU runs and formal qualification. Full input qualification stays OPEN.

## 2. Environment

- Worktree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`.
- Python 3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`. CPU only; no GPU.
- Scratch root `/data/ycfeng/tmp/glm53_dsv4/two_fix_20261101/` (unit logs, junit XML, replay runs, gates, logs).
- Evidence directory (this directory): `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/two_fix_20261101/`.

## 3. NIXL wire pricing

### 3.1 Change

Native packed registration (`base_worker._register_packed_kv_cache`) registers the packed KV cache as one region with one `block_stride` descriptor per physical block, so each selected page moves a full physical block. Before the fix, Frontier priced wire time on the structural component bytes: 39,521,664 B per worker lane at 6144 tokens, versus 46,103,040 B native.

- `frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py`: new `get_grouped_nixl_wire_bytes(cache_capacity, transferred)` returns `sum(selected pages) × physical_block_size_bytes`. The grouped branch prices transfer time on it and still returns the structural payload.
- `frontier/metrics/op_trace_utils.py`: grouped KV transfer metadata records `wire_bytes` and `native_wire_bytes_known=True`. The GLM path is unchanged.
- `tests/unit/test_dsv4_structural_pd.py`: boundary prompts 191/256/257/6143/6144/6145 plus a CUDA Graph case; each checks the wire bytes, the physical block size, transfer time equal to the wire price, and the trace metadata.

No hard-coded byte count, TP multiplier, new flag or READ lifecycle change.

### 3.2 Boundary check

Command (cwd = worktree root):

```bash
(ulimit -v 31457280; PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 timeout 600s /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  "tests/unit/test_dsv4_structural_pd.py::test_real_grouped_handoff_and_all_outputs_conserve" -q -p no:cacheprovider \
  --no-header --junitxml=/data/ycfeng/tmp/glm53_dsv4/two_fix_20261101/nixl_boundaries.xml)
```

An inline Python step read the `grouped_nixl_lifecycle` test properties from the junit XML into `nixl_boundaries.json` and priced both byte counts with the harness configuration (`AnalyticalKVCacheTransferConfig(kv_connector="nixl")`: 100 Gbps, 0.1 ms latency, compression off).

Pass condition: wire bytes = selected pages × 1,002,240 and transfer time = wire price (relative tolerance 1e-12) for every case. Result: **PASS** for all seven cases.

| Prompt | Selected pages (per group) | Structural B | Wire B | Transfer ms (harness) | Wire − structural ms (harness) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 191 | 23 (1, 2, 2, 2, 16) | 16,470,144 | 23,051,520 | 1.944122 | 0.526510 |
| 256 | 23 (1, 2, 2, 2, 16) | 16,470,144 | 23,051,520 | 1.944122 | 0.526510 |
| 257 | 28 (2, 3, 3, 3, 17) | 20,609,856 | 28,062,720 | 2.345018 | 0.596229 |
| 6143 | 46 (24, 2, 2, 2, 16) | 39,521,664 | 46,103,040 | 3.788243 | 0.526510 |
| 6144 | 46 (24, 2, 2, 2, 16) | 39,521,664 | 46,103,040 | 3.788243 | 0.526510 |
| 6145 | 51 (25, 3, 3, 3, 17) | 43,661,376 | 51,114,240 | 4.189139 | 0.596229 |
| 6144, CUDA Graph | 46 (24, 2, 2, 2, 16) | 39,521,664 | 46,103,040 | 3.788243 | 0.526510 |

At the replay link (126.2 Gbps, 0.0647 ms), a 6144-token transfer gains 6,581,376 B × 8 / 126.2 Gbps = **0.417203 ms** of wire time. Because `nixl_read_completion=engine_step`, the decode side sees each READ finish at its next engine-step boundary. The observed change per transfer is therefore not 0.417 ms; Section 5.3 shows it is zero in both replays.

### 3.3 Unit regressions

Command: `two_fix_20261101/run_units.sh <log>` (retained copy). It runs every `tests/unit` file whose name matches `dsv4|nixl|kernel_gap|pd_transfer|kv_transfer|glm53_structural|comm_operator|operator_registry|operator_spec|transfer|dp_dummy`, under a 30 GiB address-space limit and a 5400 s timeout.

| Run | Source | Passed | Failed | Errors | Log SHA-256 |
| --- | --- | ---: | ---: | ---: | --- |
| Before | `66790ca0` | 1650 | 2 | 1 | `39645cc5…` (`unit_before.log`) |
| After | `0fb8955d` | 1653 | 2 | 1 | `b446aa28…` (`unit_after.log`) |

The +3 passes are the new boundary cases. The same two failures and one collection error appear in both runs and do not touch the KV transfer path:

- `test_dsv4_attention_runtime.py::test_sklearn_owner_trains_single_rows_reuses_cache_and_prices_replay`
- `test_dsv4_h200_tables.py::test_output_hashes_and_row_counts_equal_manifest[attention_graph_replay_kernel_only.csv-4945]`
- collection error: `test_dsv4_kernel_identity_check.py` (import error)

Pass condition (no new failure): **PASS**.

## 4. EP dispatch kernel count — held-out check and deferral

### 4.1 Held-out native counts

The accounting finding used DP0 TP0 prefill forwards only. `heldout_counts.py` reads TP0 of the DP1 prefill and decode traces of both P2 profile runs, which the finding did not use. For each forward it counts every collective kernel family the SP and EP paths can launch, removes the SP calls (43 attention gathers + 1 final MHC gather, 43 reduce-scatters, each lamport or NCCL by size) and the logits gather of real forwards, and attributes the rest to EP.

```bash
D=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/two_fix_20261101 && (ulimit -v 31457280; \
  timeout 1800s env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python \
  $D/heldout_counts.py $D/heldout_counts.json > $D/heldout_counts_stdout.txt 2>&1)
```

Result `heldout_counts.json`: **PASS**. All 494 forwards fit one rule, with combine = 1 kernel per layer in every forward:

| DP lanes | EP dispatch kernels per MoE layer | Forwards | Classes |
| --- | --- | ---: | --- |
| unequal | 3 `AllGatherV` (+ 1 `Reduce` combine) | 11 | eager prefill and eager dummy |
| equal | 1 fused `AllGather` (+ 1 `ReduceScatter` combine) | 483 | eager prefill, every FULL/PIECEWISE graph step, dummy |

vLLM `all_gatherv` (`cuda_communicator.py:604-658`) drops `sizes` when every rank has the same row count and issues plain `all_gather` calls inside one NCCL group, which fuse into one kernel. `per_layer_sequence_stdout.txt` (`per_layer_sequence.py`) shows the per-layer sequence: P2-64 DP1 forward 0 (unequal) runs `AllGatherV ×3, Reduce` per layer; forward 98 (equal-lane eager prefill) runs `AllGather, ReduceScatter`.

### 4.2 Frontier query mix and decision

`b29_dispatch_queries.json` tallies the EP dispatch price queries in the B29 collectives summaries (sources and SHA-256 inside the file):

| Case | Prefill equal | Prefill unequal | Prefill equal fraction | Decode equal |
| --- | ---: | ---: | ---: | ---: |
| P2-32 | 511,872 | 16,512 | 96.875% | 957,696 (100%) |
| P2-64 | 638,464 | 27,520 | 95.868% | 1,045,760 (100%) |

An unconditional count of three would add two gaps per layer that native runs do not pay to most steps: about +0.0468 ms per equal-lane eager prefill step and +0.0428 ms per FULL decode step. YC chose to defer (D-WARM-EP-DEFER). The registry keeps `kernel_count=1`, which matches native for equal-lane and graph steps. Unequal-lane eager steps undercount by two eager gaps per layer, at most 0.0468 ms per step; this is a recorded limitation. The replays below carry only the NIXL fix.

## 5. CPU replays compared with B29

### 5.1 Commands and receipts

Each case ran under one supervisor (cwd = worktree root), launched together at 14:46:39 UTC:

```bash
D=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/two_fix_20261101
nohup /data/ycfeng/envs/frontier-py310/bin/python $D/run_replay.py p2_32 /data/ycfeng/tmp/glm53_dsv4/two_fix_20261101 &
nohup /data/ycfeng/envs/frontier-py310/bin/python $D/run_replay.py p2_64 /data/ycfeng/tmp/glm53_dsv4/two_fix_20261101 &
```

`run_replay.py` refuses to start if `frontier/` or `data/` has uncommitted changes or if the run roots already exist, so each replay starts from committed source with empty caches. It runs:

1. `timeout 86400s python -u h3o_operator_costs/a10/a10_driver.py <case> <run root>` under a 64 GiB address-space limit;
2. `a10/a10_gate.py` (the all-request gate);
3. `tests/analysis/e2e_warm_reading.py` twice, with the D-WARM-TARGET repeat openers (warm reading) and with the empty opener set (a check that the reading code reproduces the all-request gate).

Environment: `PYTHONPATH=<repo>:<repo>/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=FRONTIER_TMP_ROOT=/data/ycfeng/tmp`, W&B off, one BLAS/OpenMP thread. The full records are `p2_32_command.json` and `p2_64_command.json`.

| | P2-32 | P2-64 |
| --- | --- | --- |
| Source commit | `0fb8955d`, clean | `0fb8955d`, clean |
| Driver SHA-256 | `3172d0d0…` | `3172d0d0…` |
| Supervisor wall time | 10,242 s (finished 17:37:21 UTC) | 12,442 s (finished 18:14:01 UTC) |
| Simulator `run_wall_s` (B29 → new) | 9,641.4 → 9,602.9 s | 12,043.3 → 11,811.2 s |
| Receipt SHA-256 (B29 → new) | `a6bbc64a…` → `2048e59d…` | `f9c4c493…` → `62cc1b59…` |
| Gate `reading.json` SHA-256 | `8406db84…` | `a9569318…` |
| Warm / empty reading SHA-256 | `3a766e2a…` / `6530d718…` | `fded3090…` / `2aecaae8…` |

Run roots: `/data/ycfeng/tmp/glm53_dsv4/two_fix_20261101/runs/<case>`; gates: `…/gates/<case>`; logs: `…/logs/`. Both warm readings and both empty readings report `empty_set_reproduction = PASS_EXACT`.

### 5.2 Comparison with B29

```bash
D=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/two_fix_20261101 && (ulimit -v 31457280; \
  PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python $D/compare_b29.py $D/comparison)
```

`compare_b29.py` writes `comparison/comparison.json` (SHA-256 `81666327…`) and `comparison/metrics.csv` (96 rows, SHA-256 `c4e2bf5c…`). B29 is `a69a5213` at `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/<case>_warm_k19b29`, read through the gate readings under `codex_resume_20261031/` and the warm readings under `warm_target_20261101/readings/`.

Expected: identical inputs; any difference confined to the KV transfer completion times and what follows from them. Observed:

| Check | P2-32 | P2-64 |
| --- | --- | --- |
| Input identity (input, trace and table SHA-256, dummy mode, CC backends, KV admission, added fields, overrides) | equal | equal |
| Stage-end events / stale | 23,040 / 0 (B29 the same) | 39,680 / 0 (B29 the same) |
| Collective price calls; `collectives.json` | 6,310,644; byte-identical | 9,002,176; byte-identical |
| Prefill / decode forwards (role, lane, stage, start, end, cost, requests, step, mode) | 8,192 / 14,848, identical | 15,360 / 24,320, identical |
| KV transfers: count, ledger bytes | 128, 39,521,664 B (structural payload, unchanged by design) | 128, 39,521,664 B |
| KV start / end / duration delta | 0 for 128/128 | 0 for 128/128 |
| Request TTFT, TTFT KV part, KV transfer, TPOT, E2E delta | 0 for 128/128 each | 0 for 128/128 each |
| Metric value delta (48 rows per case) | 0 for all rows | 0 for all rows |
| Status changes vs B29 | none | none |

Metrics (relative error = (Frontier − native) / native; limit ±10%). Every new value equals its B29 value, so the absolute and relative errors are those of B29:

| Case | Reading | Scope | TTFT ms: native / Frontier / error | TPOT error | E2E error | Throughput error | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| P2-32 | all-request | pooled | 364.029650 / 344.878991 / −5.260741% | +0.039260% | −0.053760% | −0.091145% | PASS |
| P2-32 | all-request | repeat01 | 361.974764 / 344.878991 / −4.722918% | +0.100175% | +0.015944% | −0.134548% | PASS |
| P2-32 | all-request | repeat02 | 364.748496 / 344.878991 / −5.447453% | +0.192044% | +0.092723% | −0.090974% | PASS |
| P2-32 | all-request | repeat03 | 365.365689 / 344.878991 / −5.607176% | −0.173713% | −0.269220% | −0.047914% | PASS |
| P2-64 | all-request | pooled | 741.216351 / 684.313827 / −7.676912% | −2.223357% | −2.371933% | +0.201993% | PASS |
| P2-64 | all-request | repeat01 | 779.888727 / 684.313827 / **−12.254941%** | −0.529032% | −0.870328% | +0.457663% | FAIL |
| P2-64 | all-request | repeat02 | 725.492280 / 684.313827 / −5.675933% | −3.656983% | −3.710082% | −0.022443% | PASS |
| P2-64 | all-request | repeat03 | 718.268047 / 684.313827 / −4.727235% | −2.433412% | −2.493894% | +0.170758% | PASS |
| P2-64 | warm | pooled | 720.639680 / 684.545783 / −5.008591% | −2.252834% | −2.325861% | +0.201993% | PASS |
| P2-64 | warm | repeat01 | 718.057862 / 685.028554 / −4.599811% | −0.552985% | −0.661659% | +0.457663% | PASS |

Verdicts per reading (B29 → new): P2-32 all-request PASS → PASS, warm PASS → PASS; P2-64 all-request FAIL → FAIL (only the repeat01 TTFT row fails), warm PASS → PASS. At P2-32 no opener is removed, so the warm reading equals the all-request reading. At P2-64 the warm repeat02/03 rows equal the all-request rows. The empty-opener reading equals the all-request reading in both cases. The full 96 rows, including the per-row B29 values and recomputed statuses (all match), are in `comparison/metrics.csv`.

### 5.3 Why the wire-price change moved nothing

```bash
D=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/two_fix_20261101 S=/data/ycfeng/tmp/glm53_dsv4 && (ulimit -v 31457280; \
  PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 timeout 1200s /data/ycfeng/envs/frontier-py310/bin/python $D/nixl_window_check.py \
  $D/nixl_window_check.json $S/h3o_b/stage4/a10/p2_32_warm_k19b29 $S/two_fix_20261101/runs/p2_32 \
  $S/h3o_b/stage4/a10/p2_64_warm_k19b29 $S/two_fix_20261101/runs/p2_64) > $D/nixl_window_check_stdout.txt
```

With `nixl_read_completion=engine_step` (`frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py`), a DECODE lane posts a READ when one of its engine steps starts, finds it at the first later step start at or after `post + transfer time`, and reports it with that step's output. In the run-ahead DP busy loop a step starts when the host forms the forward, which `forwards.jsonl.gz` records as `formed_s`. A lane that is idle with only posted READs instead wakes at the wire end, which would also put a step start there.

So the fix can move a completion only if, after some step start `s` on the target lane, another step starts in `[s + 2.570035 ms, s + 2.987238 ms]`. The two times come from the committed predictor at each run's own link settings (126.2 Gbps, 0.0647 ms) and the 6144-token byte counts in `nixl_boundaries.json`. Every request in both cases has 6144 prompt tokens.

| Run | KV transfers | Step starts inside transfer windows | Smallest gap to the next step start | Gaps ≤ 2.987 ms | Step starts in the window |
| --- | ---: | ---: | ---: | ---: | ---: |
| B29 P2-32 | 128 | 510 | 18.725 ms | 0 | 0 |
| New P2-32 | 128 | 510 | 18.725 ms | 0 | 0 |
| B29 P2-64 | 128 | 507 | 24.380 ms | 0 | 0 |
| New P2-64 | 128 | 507 | 24.380 ms | 0 | 0 |

Every READ is therefore found at the same step start before and after the fix. No step start lies at a READ's wire end either, so no idle wake decided a completion: while a READ was outstanding, its lane kept running real or dummy forwards of at least 18.7 ms. This explains the zero deltas in 5.2. The fix only changes results in runs where a decode step start or an idle wake falls inside the added wire time.

## 6. Limitations

- At P2-32 and P2-64 the NIXL fix changes no simulated time, so the TTFT under-prediction is unchanged: −5.26% and −7.68% pooled, and the preserved P2-64 repeat01 all-request FAIL of −12.25%. The fix makes the wire price match the native packed registration; it does not close the TTFT gap.
- The EP dispatch undercount on unequal-lane eager steps (at most 0.0468 ms per step) is not corrected.
- The NIXL fix changes wire time only. READ per-worker timing, host-side work and the lazy handshake cost are not modeled; no offset is applied.
- Each replay is one fresh-cache run per case; it is compared with the single B29 run, not with a run-to-run spread.
- A numerical PASS here is not formal qualification: the 48 attention workload omissions and the hardware, NIXL, payload and memory input qualifications remain OPEN.
