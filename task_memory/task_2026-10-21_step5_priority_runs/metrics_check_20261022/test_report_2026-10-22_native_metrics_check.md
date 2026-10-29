# D21-NATIVE-METRICS-CHECK: retained 32/64-GPU client metrics

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Recomputed qualification client p90 and token throughput from all retained request endpoints; preserved original formal-admission outcomes. |
| 2026-10-22 | Primary independently checked all four hardware/scale populations with one nearest-rank convention; added H200 results and hashed parent receipt. |

## Outcome

All four native basic workloads completed. These are **24-request qualification runs: 16 QPS1 warmups, then 8 simultaneous requests, one repetition**. The table excludes all warmups. These are not canonical 128-request/QPS1/three-repeat formal results, not admitted clean reference statistics, and not Frontier comparison errors. The detailed proxy/idle analysis below concerns H800; the primary additionally checked the H200 client populations/counts/endpoints with the same formulas.

| GPUs | Job | Client p90 TTFT (ms) | Per-request p90 TPOT (ms/token) | Client p90 E2E (ms) | Total throughput (prompt + output tok/s) | Output-only (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H80032 | `exp-1021-205523-285491` | 4791.508 | 36.164 | 40567.068 | 1413.513 | 201.930 |
| H80064 | `exp-1021-202325-579469` | 9093.689 | 43.685 | 51920.700 | 1104.436 | 157.777 |
| H20032 | `exp-1021-201544-965435` | 3881.987 | 30.952 | 34258.577 | 1673.803 | 239.115 |
| H20064 | `exp-1021-125354-923219` | 4459.317 | 38.972 | 42598.915 | 1346.119 | 192.303 |

The H80032 row is **qualification03**, not the older qualification01 whose means appear in `p0_judgment/`. Each measured block contains all 8 template IDs `formal-r000` through `formal-r007`, with transport prefix `qualification-`, suffix `-repeat01`, and actual `role=diagnostic`, `segment=concurrent`, `arrived_at=0.0`.

`parent_receipt.json` records every source path/SHA256 and the unrounded four-row nearest-rank table. H200 sources are `/data/ycfeng/tmp/step5_d21_priority_20261021/h200_32_qualification02/stage/evidence/replay/client_requests.jsonl` and `/data/ycfeng/tmp/step5_d18_native_20261021/qualification02/stage/evidence/replay/client_requests.jsonl`. Parent checks require24 unique rows,16 warmups,8 concurrent completed streams,exact6144/1024 usage/token-ID counts,ordered finite client endpoints and equality to stored per-request latency. All pass. H200 windows are34.259700812s/42.599514118s; H20064 retains `REFERENCE_UNSTABLE` and is diagnostic only. The H200 subagent's supplementary linear percentiles are not mixed into this nearest-rank table.

## Metric definitions and percentile convention

- Client TTFT = `(first_token_monotonic - dispatch_monotonic) * 1000`.
- Per-request TPOT = `(last_token_monotonic - first_token_monotonic) * 1000 / (generated_token_count - 1)`. The p90 is across 8 per-request TPOT values, not across inter-token intervals.
- Client request E2E = `(last_token_monotonic - dispatch_monotonic) * 1000`.
- Total throughput = sum of accepted prompt + generated output tokens / `(max(last_token_monotonic) - min(dispatch_monotonic))`.
- Completion for E2E/throughput is the **last client-visible output token**, consistent with retained `canonical_replay.py:74–95` and `client_metrics.py:23–31,112–129`; stream-close timing is a separate diagnostic below.
- Retained `canonical_replay.formal_metrics` uses nearest rank for p95: `sorted_values[ceil(n * .95) - 1]`. The reported p90 extends that existing convention to `.90`: `sorted_values[ceil(n * .90) - 1]`. **For n=8, nearest-rank p90 is the maximum.** It is a small-population sanity check, not a stable tail estimate.
- No worker/server interval throughput or histogram latency averages were substituted for client metrics. Pure retained endpoint and aggregate functions were loaded via Python AST without importing replay/network modules or executing a workload.

For readers using linear empirical quantiles (`h=(n-1)*.90`, interpolate the adjacent sorted values), alternative diagnostics are:

| GPUs | Linear p90 TTFT (ms) | Linear p90 TPOT (ms/token) | Linear p90 E2E (ms) |
| --- | ---: | ---: | ---: |
| H80032 | 4790.844 | 36.141 | 40566.303 |
| H80064 | 9085.809 | 43.349 | 51919.891 |

## Population, clocks, and windows

| Check / value | H80032 qualification03 | H80064 qualification02 |
| --- | ---: | ---: |
| Terminal phase | Succeeded, 4 replicas | Succeeded, 8 replicas |
| Terminal finish (UTC, 2026-10-21) | 13:09:48 | 12:38:18 |
| Client records / unique IDs / matched proxy IDs | 24 / 24 / 24 | 24 / 24 / 24 |
| Warmup / measured requests | 16 / 8 | 16 / 8 |
| Accepted prompt / generated output per request | 6144 / 1024 | 6144 / 1024 |
| Measured prompt / output / total tokens | 49152 / 8192 / 57344 | 49152 / 8192 / 57344 |
| First measured dispatch (monotonic seconds) | 14266662.563809551 | 20151005.747409420 |
| Last measured output token (monotonic seconds) | 14266703.132223584 | 20151057.668914758 |
| Token-completion window (s) | 40.568414032 | 51.921505339 |
| Actual concurrent dispatch span (ms) | 1.346 | 1.319 |
| Maximum dispatch lateness (ms) | 2.169 | 1.522 |
| Peak client in-flight | 8 | 8 |
| First dispatch to last stream-close response (s), diagnostic | 40.569358185 | 51.922314052 |
| Total throughput using stream close (tok/s), diagnostic | 1413.481 | 1104.419 |

Both summaries bind the clock to **prefill-head Linux monotonic; proxy and client on the same host**. Across all 48 requests: finite client timestamps are ordered dispatch <= first token <= last token <= stream-close response; token delta timestamps are ordered and their first/last stamps equal the endpoints; recorded TTFT/TPOT/E2E agree with recomputed values within 1e-9 ms. Every request has HTTP200, completed fixed-length stream, finish_reason=length, empty validation/transport errors, exact usage6144+1024=7168, and equal produced/accepted prompt hashes. Output counts use the sum of token-ID lengths, never SSE-delta counts. H80032 has1024 deltas per request; H80064 has1022–1024 deltas but exactly1024 token IDs in every request.

Warmup and measured template populations match the staged IDs exactly, all24 records are classified, and all24 proxy IDs join one-to-one with P/D HTTP200 and NIXL pull. No request is omitted. Every measured request overlaps all other measured requests. Warmup completion precedes the measured phase; both summaries retain verified post-warmup and post-concurrent idle observations, all required role/lane gauges zero, and each role's completion count16 then24. The missing first-start gauges remain separate diagnostics, not substituted zeros.

## Warmups, kept separate

| GPUs | Requests / schedule | p90 TTFT (ms) | p90 TPOT (ms/token) | p90 E2E (ms) | Token window (s) | Total tok/s | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H80032 | 16 / QPS1, seconds0–15 | 7726.274 | 46.254 | 54616.424 | 58.612420192 | 1956.718 | 279.531 |
| H80064 | 16 / QPS1, seconds0–15 | 14189.611 | 51.594 | 64320.126 | 71.703902457 | 1599.467 | 228.495 |

No warmup timing or tokens enter the measured table. Warmups are not independent clean repetitions and their throughput is not sustained saturation throughput.

## Formal and comparability limits

- Original summaries retain `runtime_qualification=UNVERIFIED`, `reference_admission=UNVERIFIED`, and `repeat_metric_means=null`. Retained judgments classify full basic qualification as PARTIAL and formal readiness as NO. Later32 replay evidence improves native graph observation but does not supply the missing formal population/reference admission.
- Both configurations use `load_format=dummy`. These timings are native runtime observations, not verification of trained-weight/model-quality correctness.
- Native topology is P16/D16 TP8/DP2/EP16/PP1 for32, and P32/D32 TP8/DP4/EP32/PP1 for64. **API server configuration differs:** qualification03 explicitly selects1 per role; qualification02 defaults to4 per role, directly recorded in each role-head `server.log`. Hence the observed slower64 result is not a controlled pure-GPU scaling comparison. No causal attribution was attempted.
- No canonical128-request formal repetition was recorded here. Three clean repetitions, remaining native/G3/G4 evidence and admitted profiles are still required; H80064 also requires the case-bound H80032 three-repeat four-metric30% admission. Final four10% accuracy gates are unchanged.
- Existing H80064 judgment records shutdown-associated EngineDeadError traces after completed/idle evidence. This report does not call that run error-free or promote it to a clean reference.

## Decisive artifacts and reproducibility

Common raw local root: `/data/ycfeng/tmp/step5_d21_priority_20261021/`.

For each of `h800_32_qualification03` and `h800_64_qualification02`:

- `stage/config.json`: authoritative local `evidence_dir` and native config.
- `stage/trace/replay_plan.json`: qualification mode,8 concurrent,DP2/DP4.
- `stage/trace/workload/request_ids.json` and `manifest.json`: exact templates and separate formal requirements.
- `stage/evidence/replay/client_requests.jsonl:1–16`: warmups; `:17–24`: all measured requests and token endpoints.
- `stage/evidence/replay/proxy_requests.jsonl`: complete request join and native handoff status.
- `stage/evidence/replay/canonical_summary.json`: phase counts, clock, later idle boundaries and original unverified formal outcomes.
- `stage/tests/integration/step5_pd/canonical_replay.py:49–96,205–216,240–272`: retained validation/formulas/nearest-rank convention and qualification-vs-formal distinction.
- `stage/tests/comparison/calibration/client_metrics.py:23–31`: client endpoint formulas.

Terminal artifacts: `h800_32_qualification03/parent_terminal03.json`; `h800_64_qualification02/parent_status02.json`.

API count evidence: qualification03 `stage/config.json` engine.api_server_count=1 and role-head logs use APIServer; qualification02 `stage/evidence/nodes/gpu-h800-0091.host.platform.shaipower.com/server.log:1,57` and `gpu-h800-0302.host.platform.shaipower.com/server.log:1,28` explicitly record4 API processes.

Original judgments: `../native_replay/generation03_receipt.json`, `../other_basic_judgment/test_report_2026-10-21_other_basic_judgment.md`. `../p0_judgment/` describes the older32 qualification01 and was used only to identify that historical distinction.

Exact local calculation command, Linux `/usr/bin/python3`, Python3.10.6, no conda activation, standard library only:

```bash
/usr/bin/python3 -B /data/ycfeng/tmp/step5_d21_metrics_check_20261022.py /data/ycfeng/tmp/step5_d21_metrics_check_20261022.json
```

Scratch script contains complete population, token-ID, endpoint, recorded-metric, phase-clock, proxy-status, overlap and idle checks; scratch JSON contains all per-request endpoints and unrounded results. Expected/actual:24=16+8 per case;6144/1024 per request; all48 client endpoint chains ordered; all measured IDs included; exact metric provenance; two verified later idle boundaries per case — **PASS**. Throughput numerators57344 and windows above reproduce the headline values. No GPU/platform/cloud/SSH/Git/source modification or new workload occurred.

Initial calculation attempt failed on an overly strong scratch assertion that proxy decode completion must occur after the last client-visible token. One32 warmup and one64 measured request record proxy completion0.498ms/0.431ms before client last-token delivery. Retained `pd_proxy.py:160–164` records upstream completion before downstream EOF; buffered client delivery can finish later. The corrected check preserves both independent ordered client endpoints and ordered proxy lifecycle, without using proxy completion for client latency. All client validation then passed; original evidence was unchanged.

Self-check: scope discovery and verified calculation PASS; report deliverable completed. D1–D7 CLEAR, S1–S6 CLEAR, Branch A; stop investigation/tool expansion and return the requested figures. No unfinished delegated subtask remains. Formal runs remain outside this delegation and unavailable from these populations.

Task ID: `D21-NATIVE-METRICS-CHECK`. Session ID: `ses_ea0ba9f48ffepWL8nIqleN1k5T`.
