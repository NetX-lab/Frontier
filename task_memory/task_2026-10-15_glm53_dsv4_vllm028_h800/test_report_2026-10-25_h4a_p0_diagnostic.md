# Test report — H4a first diagnostic P0 prediction (DSV4, H800-32)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | First version: H4a-4 run and client-gate errors, H4a-5 attribution, criteria and limitations. |

**Label.** DIAGNOSTIC. This result counts toward neither the 30% promotion gate nor the 10% final gate (plan.md, H4a). Design and substitutions: [h4a_p0_diagnostic/design.md](h4a_p0_diagnostic/design.md).

## 1. Case and reference

- P0 on H800-32: P16 + D16; each role is one Replica with TP8, attention DP2, EP16, PP1. `max_model_len` 7168, `max_num_batched_tokens` 8192, `max_num_seqs` 128.
- Workload 68f891b (`data/processed_traces/canonical_fixed_6144_1024_req128_qps1`): 128 formal requests, each 6144 prompt and 1024 output tokens, deterministic arrivals at QPS 1.
- Reference: native06 R4, three repetitions (`native06_acceptance_20261023/`):

| Repeat | TTFT (ms) | TPOT (ms) | E2E (ms) | THPT (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| r1 | 403.305 | 19.0311 | 19872.12 | 6273.33 |
| r2 | 393.216 | 19.0165 | 19847.10 | 6273.13 |
| r3 | 399.946 | 18.9807 | 19817.24 | 6269.77 |

## 2. Environment and inputs

- Python 3.10.6, `/data/ycfeng/envs/frontier-py310/bin/python`; `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1`; address space capped with `ulimit -v 62914560`. CPU only, local host.
- Simulator source: commit `d6ea9661` (H4a-2), no dirty paths under `frontier/` or `data/` (run receipt). Scripts: commit `3fb2fc02` and this report's commit.
- Frontier input: native06's retained flat config `v4_p0_native_20261021/candidate_h800_p0_06/case/inputs/frontier.json`, SHA-256 `747a08e5…`. The current code adds and removes no field.
- Step-cost table: `data/profiling/compute/h800/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json`, SHA-256 `ea817ce5…`, built from the persisted profile02 copy (design section 8).
- Native side of the gate: persisted copies only. Run files from `h4a_p0_diagnostic/inputs/native06/` (checked against `inputs/manifest.json`, SHA-256 `159b07f9…`); R4 run-check result `native06_acceptance_20261023/frozen_r4_result.json` (`71bc110b…`), pinned by `primary_acceptance_r4.json` (`ad4d56d6…`). Trace binding: `request_ids.json` `318cd1ed…`, `manifest.json` `5c3acb69…`.

## 3. Substitutions and overrides

Substituted inputs (design sections 2-4):

| ID | Substitution | Source |
| --- | --- | --- |
| S1 | profile02 replaces profile04 as the step-cost source | profile04 is 13-23% slower than native06 on TPOT in its non-profiled repeats; profile02 is 4.4% slower |
| S2 | The P step is the proxy P service time, 287.354 ms | median of 256 formal requests in profile02 R2/R3; the profile04 P1 steps lie in the profiled repeat |
| S3 | KV link parameters 126.2 Gbit/s + 0.0647 ms | K6 fit to the GLM K5 NIXL READ measurement on the same H800 links |
| S4 | A k=1 step with m≥1 decoding requests costs the single m=1 sample, 30.146 ms | profile02 decode trace, 1 sample |
| S5 | A decode step costs 19.299 ms for every m in 1-128 and context 6143-7167 | profile02 decode traces, m 1-2 only (117 steps) |

Run overrides against native06's input (`receipt.json` `override_diff`):

| Field | native06 | H4a-4 | Reason |
| --- | --- | --- | --- |
| `vllm_v1_scheduler_config_num_blocks_mode` / `num_blocks` | memory_planner_profiled / 0 | explicit / 54740 | KV binding, design section 5 |
| `replica_config_attention_backend` | null | FLASHMLA_SPARSE_DSV4 | H4a-2 role inheritance needs the V4 backend |
| `replica_config_kv_cache_dtype`, `replica_config_indexer_kv_cache_dtype` | null | fp8 | native KV dtype |
| `metrics_config_store_frontier_stage_batch_ledger` | true | false | 64 MB per 3 short requests in the H4a-2 probe |
| `decode_cuda_graph_mode` | full_decode_only | none | the V4 idle-forward guard accepts only eager idle forwards; on the step-cost path the graph mode does not change any price (decode entries were measured on native graph steps) |
| four output paths | — | under `/data/ycfeng/tmp/glm53_dsv4/h4a4/run5/` | new run |

Other recorded departures:
- **Warmups.** The run replays the 128 formal rows once and no warmups. The workload contract drains each warmup segment before the formal origin, and on the step-cost path no state crosses the drain. The single formal replay is compared with each native repetition.
- **KV admission.** Frontier admits 2940 blocks per 7168-token request (18.619x); vLLM's two-decimal 18.59x corresponds to 2944-2945 blocks. Recorded as `DIFFERS`; preemption is checked below.
- **Routing.** The gate cell records routing `NOT_APPLICABLE`: whole native lane steps are priced, so Frontier's MoE routing model does not enter.

## 4. Commands

```bash
T=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic
ENV="PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1"
PY=/data/ycfeng/envs/frontier-py310/bin/python
# H4a-4 run (started 07:52 UTC)
(ulimit -v 62914560; env $ENV timeout 21600 $PY $T/h4a4_driver.py /data/ycfeng/tmp/glm53_dsv4/h4a4/run5)
# H4a-4 client gate
(ulimit -v 62914560; env $ENV $PY $T/h4a4_gate.py /data/ycfeng/tmp/glm53_dsv4/h4a4/run5 /data/ycfeng/tmp/glm53_dsv4/h4a4/gate)
# H4a-5 attribution
(ulimit -v 62914560; env $ENV $PY $T/h4a5_attribution.py /data/ycfeng/tmp/glm53_dsv4/h4a4/run5 $T/results/h4a5_attribution.json)
# H4a-5 profile04 cross-check of the handoff split (read-only probe)
(ulimit -v 31457280; $PY $T/probes/handoff_probe_profile04.py)
```

## 5. H4a criteria

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Every warmup and formal request completes at P0, without dummy mode or silent fallbacks | PASS for the 128 formal requests; warmups are not replayed (recorded departure, section 3) | `results/run5/system_metrics.json`: 128 of 128 completed; `results/run5/config.json`: `enable_dummy_mode` false, `sys_arch` pd-disaggregation, `enable_parallel_clusters` false, `simulation_mode` online; 0 preemptions; every step cost equals a table entry (P compute 287.354 ms for all 128 requests, D first step at least 21.101 ms, TPOT at least 19.299 ms). A missing table input raises a typed error (H4a-2). |
| 2 | The canonical client gate computes the metrics | PASS | `results/gate/reading.json`: gate `client`, `ttft_semantic` `request_issue_to_first_output_token`, `formal_request_count` 128, no request-id gaps, no token-count mismatches, no client-metric gaps |
| 3 | Every substitution is listed with its source | PASS | section 3 (S1-S5, overrides, departures) |
| 4 | Existing behavior is unchanged | PASS | H4a-2 regression on `d6ea9661` (progress.md, H4a-2 record): unit suite without new failures, `test_e2e_gate.py` 40 passed, dummy smoke matrix 12/12, refactor-fidelity matrix 77/77 identical. No `frontier/` file changed after `d6ea9661`. |
| 5 | The work is committed | PASS | code `d6ea9661`, scripts `3fb2fc02`, this report and its evidence in the report's commit |

## 6. Four errors (client gate)

Relative error is `abs(frontier - native) / abs(native)`. The Frontier side is the single formal replay; the pooled row is the gate's mean of the three repetitions. Gate limit 10%; verdict FAIL. This is a DIAGNOSTIC reading.

| Metric | Native reference | Native | Frontier | Frontier − native | Relative error | Gate status (10% limit) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| TTFT (ms) | r1 | 403.305 | 329.630 | -73.675 | 18.27% | FAIL |
| TTFT (ms) | r2 | 393.216 | 329.630 | -63.586 | 16.17% | FAIL |
| TTFT (ms) | r3 | 399.946 | 329.630 | -70.316 | 17.58% | FAIL |
| TTFT (ms) | mean of r1-r3 | 398.822 | 329.630 | -69.193 | 17.35% | FAIL |
| TPOT (ms) | r1 | 19.0311 | 19.4849 | +0.4538 | 2.38% | PASS |
| TPOT (ms) | r2 | 19.0165 | 19.4849 | +0.4684 | 2.46% | PASS |
| TPOT (ms) | r3 | 18.9807 | 19.4849 | +0.5042 | 2.66% | PASS |
| TPOT (ms) | mean of r1-r3 | 19.0094 | 19.4849 | +0.4755 | 2.50% | PASS |
| E2E (ms) | r1 | 19872.12 | 20262.73 | +390.61 | 1.97% | PASS |
| E2E (ms) | r2 | 19847.10 | 20262.73 | +415.63 | 2.09% | PASS |
| E2E (ms) | r3 | 19817.24 | 20262.73 | +445.49 | 2.25% | PASS |
| E2E (ms) | mean of r1-r3 | 19845.49 | 20262.73 | +417.24 | 2.10% | PASS |
| THPT (tok/s) | r1 | 6273.33 | 6238.11 | -35.22 | 0.56% | PASS |
| THPT (tok/s) | r2 | 6273.13 | 6238.11 | -35.02 | 0.56% | PASS |
| THPT (tok/s) | r3 | 6269.77 | 6238.11 | -31.66 | 0.50% | PASS |
| THPT (tok/s) | mean of r1-r3 | 6272.08 | 6238.11 | -33.97 | 0.54% | PASS |

Formal windows: Frontier 147.08 s; native 146.25-146.34 s; 917,504 tokens in each window.

## 7. Expected and observed

The expectation was written before any run (progress.md, H4a-1 record).

| Metric | Expected | Observed (mean of r1-r3) | Result |
| --- | --- | --- | --- |
| TTFT | about −18% to −22% (the native P response to D first byte takes about 120 ms) | −17.35% | matches the direction and size; just under the low end of the range |
| TPOT | about +2% to +3% | +2.50% | matches |
| E2E | within a few percent | +2.10% | matches |
| THPT | within a few percent | −0.54% | matches |

## 8. Attribution

Source: `results/h4a5_attribution.json` (`h4a5_attribution.py`), and for the profile04 cross-check `results/handoff_probe_profile04.log`. Native parts come from native06's proxy records (`proxy_requests.jsonl`, monotonic clock on the proxy host) and client records; Frontier parts from run5's `metrics_ground_truth.jsonl` and `request_metrics.csv`.

### 8.1 TTFT

Native06 TTFT = client→proxy + proxy P service (proxy arrival → prefill response) + handoff (prefill response → decode first byte) + decode first byte → client. Frontier TTFT = P queue wait + P compute + KV wait + KV transfer + D first step (KV transfer end → first output token, including the wait for a D step boundary).

| Part | Native06 mean (r1 / r2 / r3) | Native06 mean of repeats | Frontier mean | Frontier − native |
| --- | --- | ---: | ---: | ---: |
| client → proxy | 0.508 / 0.486 / 0.516 | 0.504 | not modeled | −0.50 |
| P service: P queue wait + prefill compute | 277.747 / 268.119 / 278.130 (median 264.9-267.1, min 246.0-246.7) | 274.666 | 0 + 287.354 | **+12.69** |
| handoff: KV wait + KV transfer + D first step | 124.901 / 124.437 / 121.143 (min 89.1-94.6) | 123.494 | 0 + 2.570 + 39.706 = 42.276 (min 23.67, max 51.94) | **−81.22** |
| decode first byte → client | 0.149 / 0.174 / 0.157 | 0.160 | not modeled | −0.16 |
| TTFT | 403.305 / 393.216 / 399.946 | 398.822 | 329.630 | −69.19 |

Observations:
- The handoff is the leading error source: −81.22 ms of the −69.19 ms gap. Every native handoff (minimum 89.1 ms) is longer than every Frontier handoff (maximum 51.9 ms).
- The P side runs the other way, +12.69 ms. S2 takes the P step from profile02's median P service, 287.354 ms, which is 4.6% above native06's mean and about 8% above native06's median. Inference: profile02's environment is slower than native06's, as its TPOT is (+4.4%, section 8.3).
- Frontier's D first step: 126 of 128 requests run their first D step with other requests already decoding. They wait 9.81 ms on average (0.18-19.22 ms, within one 19.3 ms decode step) for the running step to end, then pay the k=1 step, 30.146 ms (S4). The two requests that find an idle lane pay 21.101 ms after a wait of 0 and 5.36 ms. So Frontier's time from P completion to the start of the k=1 step is about 2.57 + 9.81 = 12.4 ms.

### 8.2 Handoff split from the D-side traces

The decode traces (one D lane per trace) contain the k=1 step that computes a request's last prompt token. Placing that step inside the same request's proxy handoff splits the handoff into the part before D's k=1 step and the rest. Only requests whose D lane is idle before the k=1 step give a clean split; the other traced k=1 steps in profile02 and profile04 sit inside a 10-20 s profiler pause and are excluded.

| Run, lane, request | Handoff | P response → k=1 CPU start | k=1 CPU start → GPU start | k=1 GPU step | next step start → first byte |
| --- | ---: | ---: | ---: | ---: | ---: |
| profile02 dp0 r001 | 128.62 | 74.23 | 27.76 | 21.15 | 5.48 |
| profile02 dp1 r000 | 98.60 | 39.82 | 32.19 | 21.05 | 5.55 |
| profile04 dp0 r001 | 149.48 | 78.93 | 26.61 | 33.91 | 10.03 |
| profile04 dp1 r000 | 111.28 | 50.34 | 32.61 | 21.66 | 6.68 |

All values in ms. Observations:
- Native time from P response to the start of D's k=1 GPU step: 102.0 and 72.0 ms (profile02), 105.5 and 82.9 ms (profile04), mean about 90.6 ms. Frontier's corresponding time is 2.57 ms for an idle lane and about 12.4 ms in steady state.
- After the k=1 step starts, native spends the step plus 5.5-10 ms until the first byte reaches the proxy (profile02: 26.6 ms in total). Frontier ends TTFT at the end of the k=1 step and models no output processing.
- Inference for the steady state: if a native busy-lane k=1 step costs what S4 prices (30.146 ms) and output takes 5.5 ms, the native part before the k=1 step is about 123.5 − 35.6 ≈ 87.9 ms, so about −75 ms of the −81 ms handoff gap lies before D's k=1 step and about −5.5 ms in the output processing that Frontier does not model.

What Frontier models in the handoff (from code): at KV transfer end, `handle_reserved_decode_arrival` marks the request's KV ready, adds the request to the D replica scheduler and schedules an immediate `ReplicaScheduleEvent`; the request is admitted at the next D step boundary, ahead of other waiting requests. Frontier does not model proxy → D request delivery, D input processing, vLLM's `WAITING_FOR_REMOTE_KVS` state, a NIXL read that is issued in one D step and reported at a later step boundary, or output processing up to the first byte. The vLLM-side lifecycle named here is inferred from vLLM's design and the trace timings; native06's server logs have only NIXL setup lines and second-resolution access logs, so no native measurement splits the 72-106 ms further.

### 8.3 TPOT

| Item | Value | Source |
| --- | ---: | --- |
| native06 TPOT, mean of repeats | 19.009 ms | gate |
| Frontier TPOT, mean (min / median) | 19.485 ms (19.299 / 19.501) | run5 |
| table decode step, m 1-128 (S5) | 19.299 ms | step table |
| profile02 measured decode step, m=1 / m=2 | 19.289 ms (72 steps) / 19.355 ms (45 steps) | profile02 traces |
| profile02 measured k=1 step, m=0 / m≥1 | 21.101 ms (2) / 30.146 ms (1) | profile02 traces |
| profile02 own client TPOT, R2 / R3 (non-profiled) | 19.848 / 19.826 ms | profile02 client records |
| profile02 own client TPOT, R1 (profiled) | 22.861 ms | profile02 client records |
| requests decoding at once, time-weighted mean | Frontier 17.38; native06 17.08 / 17.07 / 17.03 | run5; native06 client records |

Observations:
- Frontier's steady TPOT, 19.501 ms, is the table decode step plus the k=1 steps of later arrivals: about 20 requests arrive during one request's 1023 decode steps, and each k=1 step costs 30.146 − 19.299 = 10.85 ms more than a decode step, so 19.299 + 20 × 10.85 / 1023 ≈ 19.51 ms. The two attention-DP lanes run in lockstep, so a k=1 step on either lane delays both.
- Inference: the +2.50% TPOT error is the sum of two opposite effects. (1) profile02 runs about 4.4% slower than native06 (its non-profiled client TPOT, 19.84 ms, against native06's 19.01 ms). (2) Frontier with profile02's m 1-2 step costs is 1.8% below profile02's own client TPOT, which suggests that a decode step at about 8.5 requests per lane costs about 1.8% more than at m 1-2 (S5 underprices it). Net: +4.4% − 1.8% ≈ +2.5%.

### 8.4 E2E and THPT

- E2E: +417.24 ms = TTFT −69.19 ms + TPOT error 0.4755 ms × 1023 decode steps (+486.4 ms). E2E is dominated by the decode steps, so its error follows TPOT.
- THPT: each window is set by the arrivals (127 s from first to last formal submission) plus the last request's E2E. Frontier's window is about 0.8 s longer (147.08 s against 146.25-146.34 s), which gives −0.54%.

## 9. Recommendation: step-level costs for formal gates

Recommendation: step-level costs should not yet support the formal 30% or 10% gates. Reasons, each from the evidence above:
1. Profiler perturbation: the step-cost source runs 4.4% slower than native06 on TPOT (S1), and its P service is 4.6% above native06's (S2). An error below 10% on TPOT is then partly a cancellation (section 8.3), not a prediction.
2. Coverage: decode steps were measured at m 1-2 only; P0 decodes about 8.5 requests per lane (S5).
3. k=1 variability: the k=1 step ranges over 21-34 ms at m=0 and 30-49 ms at m≥1 across the traces, against single-sample table entries (S4).
4. S2 includes proxy overhead in the P step, and S3 uses GLM link parameters for the KV transfer.
5. The TTFT error is a modeling gap, not a cost gap: about 75 ms of the D-side handoff is not represented in Frontier at all (section 8.2). Refining step costs cannot close it.

Options for YC:

| Option | Work | Cost | What it settles |
| --- | --- | --- | --- |
| (a) | Native measurement that splits the D handoff: one P0 native run with vLLM request-lifecycle and NIXL transfer timing enabled on D | 1 GPU job on 32 H800, same case and workload as native06 | which D-side states take the 72-106 ms before the k=1 step, so (b) uses measured inputs |
| (b) | Frontier modeling of the D-side asynchronous remote-KV lifecycle (request delivery, `WAITING_FOR_REMOTE_KVS`, read completion reported at a later step boundary, output processing) | core scheduler/event change; needs (a) first to avoid a fitted constant; full regression | the −81 ms handoff gap, the leading TTFT error |
| (c) | Step-cost path speed: one per-step sync for table-priced steps instead of 6594 per-layer EP waves per step | core scheduler change; about 2.4 h per replay today, 168 of 180 profiled seconds are in the EP waves | iteration speed for every later P0/P1 replay |
| (d) | P1b scheduling (H800-64 basic native validation) | 64-GPU job(s) | P1 progress in parallel |

Recommended order: (a), then (b); (c) early, because each replay currently costs about 2.4 CPU hours; (d) in parallel at lower priority.

## 10. Limitations

- The handoff split rests on 4 idle-lane samples, all from profiled repeats. Busy-lane handoffs are not split by any trace; the steady-state −75 ms figure is an inference (section 8.2).
- Kineto times are on the D host and proxy times on the proxy host, so the split carries the clock offset between the two hosts. The next step start → first byte time is 5.5 ms on two different D hosts in profile02, which suggests the offset is small, but it is not measured.
- The vLLM-side lifecycle in section 8.2 is inferred from vLLM's design and the timings, not from native logs.
- One formal Frontier replay is compared with three native repetitions. The Frontier path uses fixed arrivals and fixed table costs; a second replay was not run to confirm that it reproduces the same numbers.
- Warmups are not replayed (section 3).
- KV admission differs by 4-5 blocks per request (0.16%); it cannot bind at about 8-9 requests per lane against a limit of 18.
- The step-cost path takes 8095 s wall for one P0 replay (`receipt.json` `run_wall_s`), which limits how many variants can be tried.
- All errors are DIAGNOSTIC: substitutions S1-S5 enter every number in this report.
