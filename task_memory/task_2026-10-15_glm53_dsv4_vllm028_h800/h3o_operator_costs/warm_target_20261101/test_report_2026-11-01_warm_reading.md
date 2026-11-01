# WARM-READING verification and metric report

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Verified D-WARM-TARGET retained-reference readings. |

Task ID: WARM-READING. CPU only; no new native or Frontier run, GPU operation, or commit.

## Outcome

All six warm readings pass all four metrics in all three native repeats and in the pooled reading (96/96 PASS). The unchanged harness has three metric FAILs: P2-64 repeat01 TTFT, one in each generation. Six empty-exclusion checks reproduce the original pooled/per-repeat values exactly, with unchanged signed error and verdict. The all-request JSON is retained as both a parsed object and byte-exact text with SHA-256.

Authoritative outputs are under `readings/`. Root-level first-iteration reading JSONs/metrics.csv remain unchanged as history; their no-op Frontier totals used the equivalent harness representation. The final generation retains summed totals on both sides also for an empty exclusion set, while keeping the original binary-float ratio rounding (differences otherwise at about 1e-13).

## Mechanism and identification

Source: vLLM `pull_worker.py:65-77` starts the first remote-engine handshake before a KV read; `base_worker.py:686-708` logs compatibility once per TP worker during handshake; `base_worker.py:879-905` caches the remote engine and shares pending handshake futures; `base_worker.py:508-509` defaults engine TTL to 3600 s.

`identify_openers.py` reads decode request ownership from `nodes/*/serving/rank0.json` (`inventory.request_owner` + `counters.request_ids`) and the remote prefill engine from proxy KV-transfer fields. For each decode lane, the nth compatibility line of each of the eight TP workers forms one cohort. The complete cohort's one-second timestamp bins must each overlap the same unique same-lane proxy handoff (prefill response -> decode first byte), and the request must be the pair's first request. Every formal event binds uniquely; no latency threshold selects any request.

The source events log the same compatibility hash for all engines, not the remote engine ID. Therefore pair identity is supplied by the independently retained request-owner inventory and proxy record, not by the hash. All TP workers have four events per D lane for P2-64 and two for P2-32.

Clock mapping: server civil timestamps are UTC (offset 0), anchored to the corresponding server-start log and `server_ready.json.ready_at_utc`; the latest start log precedes readiness by less than 5 s. `canonical_repetitions.json` records origin_wall/origin_monotonic per phase. Proxy wall-minus-monotonic offsets agree with those origins within 1 ms in every run. Exact anchors, observed ranges, phase windows, event bins, server log paths/line numbers, and request handoff endpoints are in `openers.json`. The server log resolution is 1 s; no sub-second handshake timestamp or duration is asserted.

Warmup cohorts can overlap multiple warmup handoffs. Those candidate lists are preserved without asserting a unique request/remote-engine assignment. Their window classification is unambiguous, all warmups have drained before formal admission, and no warmup request is included in or removed from the formal population. The formal first-contact pair count independently agrees with the formal event count.

### Event and opener table

| Run | Repeat | Warmup events | Formal events | Formal opener IDs | Validation |
| --- | --- | ---: | ---: | --- | --- |
| canonical_p2_32 | 01 | 4 | 0 | none | NOT_APPLICABLE_CLEAN_NO_LIFECYCLE |
| canonical_p2_32 | 02 | 0 | 0 | none | NOT_APPLICABLE_CLEAN_NO_LIFECYCLE |
| canonical_p2_32 | 03 | 0 | 0 | none | NOT_APPLICABLE_CLEAN_NO_LIFECYCLE |
| canonical_p2_64 | 01 | 11 | 5 | r006, r017, r027, r035, r036 | NOT_APPLICABLE_CLEAN_NO_LIFECYCLE |
| canonical_p2_64 | 02 | 0 | 0 | none | NOT_APPLICABLE_CLEAN_NO_LIFECYCLE |
| canonical_p2_64 | 03 | 0 | 0 | none | NOT_APPLICABLE_CLEAN_NO_LIFECYCLE |
| lifecycle | 01 | 11 | 5 | r002, r004, r006, r009, r053 | PASS_EXACT |
| lifecycle | 02 | 0 | 0 | none | PASS_EXACT |
| lifecycle | 03 | 0 | 0 | none | PASS_EXACT |
| wave | 01 | 13 | 3 | r005, r017, r037 | PASS_EXACT |

### Unique formal bindings

| Run | Request | P lane | D lane | Log interval UTC | Proxy handoff (ms) |
| --- | --- | ---: | ---: | --- | ---: |
| canonical_p2_64 | r017-repeat01 | 2 | 0 | 2026-10-25T16:09:58+00:00 to 2026-10-25T16:09:59+00:00 | 1923.90583083 |
| canonical_p2_64 | r027-repeat01 | 0 | 0 | 2026-10-25T16:10:08+00:00 to 2026-10-25T16:10:09+00:00 | 1656.58242023 |
| canonical_p2_64 | r036-repeat01 | 0 | 1 | 2026-10-25T16:10:18+00:00 to 2026-10-25T16:10:19+00:00 | 1455.25439596 |
| canonical_p2_64 | r006-repeat01 | 1 | 2 | 2026-10-25T16:09:47+00:00 to 2026-10-25T16:09:48+00:00 | 1709.41871591 |
| canonical_p2_64 | r035-repeat01 | 3 | 2 | 2026-10-25T16:10:17+00:00 to 2026-10-25T16:10:18+00:00 | 1754.70146304 |
| lifecycle | r002-repeat01 | 3 | 0 | 2026-10-26T05:17:07+00:00 to 2026-10-26T05:17:08+00:00 | 1787.40728367 |
| lifecycle | r009-repeat01 | 2 | 0 | 2026-10-26T05:17:14+00:00 to 2026-10-26T05:17:15+00:00 | 1921.99923703 |
| lifecycle | r006-repeat01 | 3 | 1 | 2026-10-26T05:17:11+00:00 to 2026-10-26T05:17:12+00:00 | 1786.37151467 |
| lifecycle | r053-repeat01 | 1 | 2 | 2026-10-26T05:17:58+00:00 to 2026-10-26T05:17:59+00:00 | 1791.4967672 |
| lifecycle | r004-repeat01 | 1 | 3 | 2026-10-26T05:17:09+00:00 to 2026-10-26T05:17:10+00:00 | 1537.15136321 |
| wave | r005-repeat01 | 0 | 1 | 2026-10-30T07:03:57+00:00 to 2026-10-30T07:03:58+00:00 | 1515.50725102 |
| wave | r037-repeat01 | 3 | 2 | 2026-10-30T07:04:29+00:00 to 2026-10-30T07:04:30+00:00 | 1693.77084449 |
| wave | r017-repeat01 | 2 | 3 | 2026-10-30T07:04:09+00:00 to 2026-10-30T07:04:10+00:00 | 1562.06353381 |

## Commands and acceptance

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (exact version in receipts/verification.json); `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH=$PWD:$PWD/tests/analysis`; `ulimit -v 31457280` inherited by all child commands. All commands ran at the assigned repository root.

1. `python <evidence>/identify_openers.py --run LABEL=ARTIFACTS ... --output <evidence>/openers.json`: PASS; clean32=4 warmup/0 formal, clean64=11/5, lifecycle=11/5 and wave=13/3; exact lifecycle ground-truth opener reproduction on both instrumented runs.
2. Twelve final `python tests/analysis/e2e_warm_reading.py --rows ROWS --reading ORIGINAL --openers EVIDENCE --output NEW_JSON` calls through `run_analysis.py`: PASS; six originals and six empty sets exactly reproduce values; P2-32 retains 384 request-repeat rows, P2-64 retains 379 (repeat01=123, repeats02/03=128).
3. `python <evidence>/verify_outputs.py`: PASS; independent paired-row/counter checks, summed pooling totals on both sides, verbatim source bytes, SHA-256 of every source, CSV arithmetic/units/verdicts, 192 rows, 96 warm PASS / 3 all-request FAIL.

Full commands, cwd, environment, UTC start/end, exit code, git HEAD, dirty paths and output SHA-256 are in `command_receipts.json`. The initial repository already had unrelated dirty requirements.md and untracked `.cortexkit/`, `.pi/`, and `remaining_measurements/`; those were not edited.

B29 rows were found in the retained `h3o_b/stage4/codex_resume_20261031/p2_{32,64}_warm_k19b29/rows`; their reading hashes exactly match the task-retained B29 readings. No normalization or replay was needed.

## Complete metric table

CSV `readings/metrics.csv` contains exact Python float representations and both signed/absolute relative errors. Below: all-request and warm absolute values, signed relative error in %, and metric PASS/FAIL; absolute relative error is the magnitude of the signed column. Displayed values are rounded to 12 significant digits; JSON/CSV retain exact precision. No confidence intervals.

| Generation | Case | Repeat | Metric (unit) | All native | All Frontier | All signed error % | All | Warm native | Warm Frontier | Warm signed error % | Warm |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| B29 | p2_32 | pooled | ttft_ms (ms) | 364.029649531 | 344.878991339 | -5.26074132068 | PASS | 364.029649531 | 344.878991339 | -5.26074132068 | PASS |
| B29 | p2_32 | pooled | tpot_ms (ms) | 19.9189660753 | 19.9267863495 | 0.0392604421411 | PASS | 19.9189660753 | 19.9267863495 | 0.0392604421411 | PASS |
| B29 | p2_32 | pooled | request_e2e_time_ms (ms) | 20741.1319446 | 20729.9814268 | -0.0537604107897 | PASS | 20741.1319446 | 20729.9814268 | -0.0537604107897 | PASS |
| B29 | p2_32 | pooled | token_throughput_tps (tokens/s) | 6231.41549606 | 6225.73586175 | -0.0911451710221 | PASS | 6231.41549606 | 6225.73586175 | -0.0911451710221 | PASS |
| B29 | p2_32 | repeat01 | ttft_ms (ms) | 361.974763982 | 344.878991339 | -4.72291837573 | PASS | 361.974763982 | 344.878991339 | -4.72291837573 | PASS |
| B29 | p2_32 | repeat01 | tpot_ms (ms) | 19.9068446 | 19.9267863495 | 0.100175341144 | PASS | 19.9068446 | 19.9267863495 | 0.100175341144 | PASS |
| B29 | p2_32 | repeat01 | request_e2e_time_ms (ms) | 20726.6767898 | 20729.9814268 | 0.0159438829349 | PASS | 20726.6767898 | 20729.9814268 | 0.0159438829349 | PASS |
| B29 | p2_32 | repeat01 | token_throughput_tps (tokens/s) | 6234.1237622 | 6225.73586175 | -0.134548186228 | PASS | 6234.1237622 | 6225.73586175 | -0.134548186228 | PASS |
| B29 | p2_32 | repeat02 | ttft_ms (ms) | 364.74849598 | 344.878991339 | -5.44745348109 | PASS | 364.74849598 | 344.878991339 | -5.44745348109 | PASS |
| B29 | p2_32 | repeat02 | tpot_ms (ms) | 19.8885915932 | 19.9267863495 | 0.192043544505 | PASS | 19.8885915932 | 19.9267863495 | 0.192043544505 | PASS |
| B29 | p2_32 | repeat02 | request_e2e_time_ms (ms) | 20710.7776958 | 20729.9814268 | 0.0927233698411 | PASS | 20710.7776958 | 20729.9814268 | 0.0927233698411 | PASS |
| B29 | p2_32 | repeat02 | token_throughput_tps (tokens/s) | 6231.40480608 | 6225.73586175 | -0.0909737772809 | PASS | 6231.40480608 | 6225.73586175 | -0.0909737772809 | PASS |
| B29 | p2_32 | repeat03 | ttft_ms (ms) | 365.365688631 | 344.878991339 | -5.60717602382 | PASS | 365.365688631 | 344.878991339 | -5.60717602382 | PASS |
| B29 | p2_32 | repeat03 | tpot_ms (ms) | 19.9614620327 | 19.9267863495 | -0.17371314399 | PASS | 19.9614620327 | 19.9267863495 | -0.17371314399 | PASS |
| B29 | p2_32 | repeat03 | request_e2e_time_ms (ms) | 20785.9413481 | 20729.9814268 | -0.269220048079 | PASS | 20785.9413481 | 20729.9814268 | -0.269220048079 | PASS |
| B29 | p2_32 | repeat03 | token_throughput_tps (tokens/s) | 6228.72026272 | 6225.73586175 | -0.0479135495577 | PASS | 6228.72026272 | 6225.73586175 | -0.0479135495577 | PASS |
| B29 | p2_64 | pooled | ttft_ms (ms) | 741.216351415 | 684.313826604 | -7.67691170092 | PASS | 720.639680322 | 684.545782971 | -5.00859144125 | PASS |
| B29 | p2_64 | pooled | tpot_ms (ms) | 25.870530968 | 25.2953366177 | -2.22335734466 | PASS | 25.878311466 | 25.2953159424 | -2.25283448022 | PASS |
| B29 | p2_64 | pooled | request_e2e_time_ms (ms) | 27206.7695317 | 26561.4431865 | -2.37193300181 | PASS | 27194.15231 | 26561.653992 | -2.32586149695 | PASS |
| B29 | p2_64 | pooled | token_throughput_tps (tokens/s) | 5963.20631789 | 5975.25156986 | 0.201992876347 | PASS | 5885.56040229 | 5897.44881504 | 0.201992876347 | PASS |
| B29 | p2_64 | repeat01 | ttft_ms (ms) | 779.8887274 | 684.313826604 | -12.254940665 | FAIL | 718.057861836 | 685.028553945 | -4.59981147002 | PASS |
| B29 | p2_64 | repeat01 | tpot_ms (ms) | 25.4298687355 | 25.2953366177 | -0.529031900371 | PASS | 25.4359296912 | 25.2952729109 | -0.552984624745 | PASS |
| B29 | p2_64 | repeat01 | request_e2e_time_ms (ms) | 26794.6444438 | 26561.4431865 | -0.870327866558 | PASS | 26739.013936 | 26562.0927418 | -0.661659381365 | PASS |
| B29 | p2_64 | repeat01 | token_throughput_tps (tokens/s) | 5948.02962943 | 5975.25156986 | 0.457663161175 | PASS | 5715.68472203 | 5741.84330541 | 0.457663161175 | PASS |
| B29 | p2_64 | repeat02 | ttft_ms (ms) | 725.492279951 | 684.313826604 | -5.67593267149 | PASS | 725.492279951 | 684.313826604 | -5.67593267149 | PASS |
| B29 | p2_64 | repeat02 | tpot_ms (ms) | 26.2554955812 | 25.2953366177 | -3.65698282307 | PASS | 26.2554955812 | 25.2953366177 | -3.65698282307 | PASS |
| B29 | p2_64 | repeat02 | request_e2e_time_ms (ms) | 27584.8642595 | 26561.4431865 | -3.7100819616 | PASS | 27584.8642595 | 26561.4431865 | -3.7100819616 | PASS |
| B29 | p2_64 | repeat02 | token_throughput_tps (tokens/s) | 5976.59289203 | 5975.25156986 | -0.0224429236491 | PASS | 5976.59289203 | 5975.25156986 | -0.0224429236491 | PASS |
| B29 | p2_64 | repeat03 | ttft_ms (ms) | 718.268046894 | 684.313826604 | -4.72723524827 | PASS | 718.268046894 | 684.313826604 | -4.72723524827 | PASS |
| B29 | p2_64 | repeat03 | tpot_ms (ms) | 25.9262285874 | 25.2953366177 | -2.43341204703 | PASS | 25.9262285874 | 25.2953366177 | -2.43341204703 | PASS |
| B29 | p2_64 | repeat03 | request_e2e_time_ms (ms) | 27240.7998918 | 26561.4431865 | -2.49389411501 | PASS | 27240.7998918 | 26561.4431865 | -2.49389411501 | PASS |
| B29 | p2_64 | repeat03 | token_throughput_tps (tokens/s) | 5965.06571958 | 5975.25156986 | 0.170758391514 | PASS | 5965.06571958 | 5975.25156986 | 0.170758391514 | PASS |
| pass1_all | p2_32 | pooled | ttft_ms (ms) | 364.029649531 | 343.688275412 | -5.58783443757 | PASS | 364.029649531 | 343.688275412 | -5.58783443757 | PASS |
| pass1_all | p2_32 | pooled | tpot_ms (ms) | 19.9189660753 | 19.9176122972 | -0.00679642748084 | PASS | 19.9189660753 | 19.9176122972 | -0.00679642748084 | PASS |
| pass1_all | p2_32 | pooled | request_e2e_time_ms (ms) | 20741.1319446 | 20719.4056555 | -0.104749775266 | PASS | 20741.1319446 | 20719.4056555 | -0.104749775266 | PASS |
| pass1_all | p2_32 | pooled | token_throughput_tps (tokens/s) | 6231.41549606 | 6224.15766642 | -0.116471604939 | PASS | 6231.41549606 | 6224.15766642 | -0.116471604939 | PASS |
| pass1_all | p2_32 | repeat01 | ttft_ms (ms) | 361.974763982 | 343.688275412 | -5.05186835941 | PASS | 361.974763982 | 343.688275412 | -5.05186835941 | PASS |
| pass1_all | p2_32 | repeat01 | tpot_ms (ms) | 19.9068446 | 19.9176122972 | 0.0540904270367 | PASS | 19.9068446 | 19.9176122972 | 0.0540904270367 | PASS |
| pass1_all | p2_32 | repeat01 | request_e2e_time_ms (ms) | 20726.6767898 | 20719.4056555 | -0.0350810424359 | PASS | 20726.6767898 | 20719.4056555 | -0.0350810424359 | PASS |
| pass1_all | p2_32 | repeat01 | token_throughput_tps (tokens/s) | 6234.1237622 | 6224.15766642 | -0.15986361768 | PASS | 6234.1237622 | 6224.15766642 | -0.15986361768 | PASS |
| pass1_all | p2_32 | repeat02 | ttft_ms (ms) | 364.74849598 | 343.688275412 | -5.77390196276 | PASS | 364.74849598 | 343.688275412 | -5.77390196276 | PASS |
| pass1_all | p2_32 | repeat02 | tpot_ms (ms) | 19.8885915932 | 19.9176122972 | 0.145916335384 | PASS | 19.8885915932 | 19.9176122972 | 0.145916335384 | PASS |
| pass1_all | p2_32 | repeat02 | request_e2e_time_ms (ms) | 20710.7776958 | 20719.4056555 | 0.0416592740398 | PASS | 20710.7776958 | 20719.4056555 | 0.0416592740398 | PASS |
| pass1_all | p2_32 | repeat02 | token_throughput_tps (tokens/s) | 6231.40480608 | 6224.15766642 | -0.116300254645 | PASS | 6231.40480608 | 6224.15766642 | -0.116300254645 | PASS |
| pass1_all | p2_32 | repeat03 | ttft_ms (ms) | 365.365688631 | 343.688275412 | -5.93307305359 | PASS | 365.365688631 | 343.688275412 | -5.93307305359 | PASS |
| pass1_all | p2_32 | repeat03 | tpot_ms (ms) | 19.9614620327 | 19.9176122972 | -0.21967196314 | PASS | 19.9614620327 | 19.9176122972 | -0.21967196314 | PASS |
| pass1_all | p2_32 | repeat03 | request_e2e_time_ms (ms) | 20785.9413481 | 20719.4056555 | -0.320099491962 | PASS | 20785.9413481 | 20719.4056555 | -0.320099491962 | PASS |
| pass1_all | p2_32 | repeat03 | token_throughput_tps (tokens/s) | 6228.72026272 | 6224.15766642 | -0.0732509424908 | PASS | 6228.72026272 | 6224.15766642 | -0.0732509424908 | PASS |
| pass1_all | p2_64 | pooled | ttft_ms (ms) | 741.216351415 | 700.581587484 | -5.48217316757 | PASS | 720.639680322 | 701.66220356 | -2.63342101195 | PASS |
| pass1_all | p2_64 | pooled | tpot_ms (ms) | 25.870530968 | 25.2715341742 | -2.31536335523 | PASS | 25.878311466 | 25.2715301293 | -2.34474856467 | PASS |
| pass1_all | p2_64 | pooled | request_e2e_time_ms (ms) | 27206.7695317 | 26553.3610477 | -2.401639354 | PASS | 27194.15231 | 26554.4375259 | -2.35239832767 | PASS |
| pass1_all | p2_64 | pooled | token_throughput_tps (tokens/s) | 5963.20631789 | 5967.96691288 | 0.0798328070167 | PASS | 5885.56040229 | 5890.25901037 | 0.0798328070167 | PASS |
| pass1_all | p2_64 | repeat01 | ttft_ms (ms) | 779.8887274 | 700.581587484 | -10.169032726 | FAIL | 718.057861836 | 703.911290676 | -1.97011576798 | PASS |
| pass1_all | p2_64 | repeat01 | tpot_ms (ms) | 25.4298687355 | 25.2715341742 | -0.622632239822 | PASS | 25.4359296912 | 25.2715217107 | -0.646361200577 | PASS |
| pass1_all | p2_64 | repeat01 | request_e2e_time_ms (ms) | 26794.6444438 | 26553.3610477 | -0.900491128454 | PASS | 26739.013936 | 26556.6780007 | -0.681909720674 | PASS |
| pass1_all | p2_64 | repeat01 | token_throughput_tps (tokens/s) | 5948.02962943 | 5967.96691288 | 0.335191394454 | PASS | 5715.68472203 | 5734.84320535 | 0.335191394454 | PASS |
| pass1_all | p2_64 | repeat02 | ttft_ms (ms) | 725.492279951 | 700.581587484 | -3.43362612606 | PASS | 725.492279951 | 700.581587484 | -3.43362612606 | PASS |
| pass1_all | p2_64 | repeat02 | tpot_ms (ms) | 26.2554955812 | 25.2715341742 | -3.74763981861 | PASS | 26.2554955812 | 25.2715341742 | -3.74763981861 | PASS |
| pass1_all | p2_64 | repeat02 | request_e2e_time_ms (ms) | 27584.8642595 | 26553.3610477 | -3.73938114068 | PASS | 27584.8642595 | 26553.3610477 | -3.73938114068 | PASS |
| pass1_all | p2_64 | repeat02 | token_throughput_tps (tokens/s) | 5976.59289203 | 5967.96691288 | -0.144329374739 | PASS | 5976.59289203 | 5967.96691288 | -0.144329374739 | PASS |
| pass1_all | p2_64 | repeat03 | ttft_ms (ms) | 718.268046894 | 700.581587484 | -2.46237591751 | PASS | 718.268046894 | 700.581587484 | -2.46237591751 | PASS |
| pass1_all | p2_64 | repeat03 | tpot_ms (ms) | 25.9262285874 | 25.2715341742 | -2.52522040002 | PASS | 25.9262285874 | 25.2715341742 | -2.52522040002 | PASS |
| pass1_all | p2_64 | repeat03 | request_e2e_time_ms (ms) | 27240.7998918 | 26553.3610477 | -2.52356335677 | PASS | 27240.7998918 | 26553.3610477 | -2.52356335677 | PASS |
| pass1_all | p2_64 | repeat03 | token_throughput_tps (tokens/s) | 5965.06571958 | 5967.96691288 | 0.048636401335 | PASS | 5965.06571958 | 5967.96691288 | 0.048636401335 | PASS |
| pass2_all | p2_32 | pooled | ttft_ms (ms) | 364.029649531 | 343.616677887 | -5.60750248502 | PASS | 364.029649531 | 343.616677887 | -5.60750248502 | PASS |
| pass2_all | p2_32 | pooled | tpot_ms (ms) | 19.9189660753 | 19.9366675342 | 0.0888673581274 | PASS | 19.9189660753 | 19.9366675342 | 0.0888673581274 | PASS |
| pass2_all | p2_32 | pooled | request_e2e_time_ms (ms) | 20741.1319446 | 20738.8275654 | -0.0111101900192 | PASS | 20741.1319446 | 20738.8275654 | -0.0111101900192 | PASS |
| pass2_all | p2_32 | pooled | token_throughput_tps (tokens/s) | 6231.41549606 | 6224.26391513 | -0.114766555673 | PASS | 6231.41549606 | 6224.26391513 | -0.114766555673 | PASS |
| pass2_all | p2_32 | repeat01 | ttft_ms (ms) | 361.974763982 | 343.616677887 | -5.07164805991 | PASS | 361.974763982 | 343.616677887 | -5.07164805991 | PASS |
| pass2_all | p2_32 | repeat01 | tpot_ms (ms) | 19.9068446 | 19.9366675342 | 0.149812463274 | PASS | 19.9068446 | 19.9366675342 | 0.149812463274 | PASS |
| pass2_all | p2_32 | repeat01 | request_e2e_time_ms (ms) | 20726.6767898 | 20738.8275654 | 0.0586238487316 | PASS | 20726.6767898 | 20738.8275654 | 0.0586238487316 | PASS |
| pass2_all | p2_32 | repeat01 | token_throughput_tps (tokens/s) | 6234.1237622 | 6224.26391513 | -0.158159309133 | PASS | 6234.1237622 | 6224.26391513 | -0.158159309133 | PASS |
| pass2_all | p2_32 | repeat02 | ttft_ms (ms) | 364.74849598 | 343.616677887 | -5.79353124842 | PASS | 364.74849598 | 343.616677887 | -5.79353124842 | PASS |
| pass2_all | p2_32 | repeat02 | tpot_ms (ms) | 19.8885915932 | 19.9366675342 | 0.241726221732 | PASS | 19.8885915932 | 19.9366675342 | 0.241726221732 | PASS |
| pass2_all | p2_32 | repeat02 | request_e2e_time_ms (ms) | 20710.7776958 | 20738.8275654 | 0.135436099873 | PASS | 20710.7776958 | 20738.8275654 | 0.135436099873 | PASS |
| pass2_all | p2_32 | repeat02 | token_throughput_tps (tokens/s) | 6231.40480608 | 6224.26391513 | -0.114595202455 | PASS | 6231.40480608 | 6224.26391513 | -0.114595202455 | PASS |
| pass2_all | p2_32 | repeat03 | ttft_ms (ms) | 365.365688631 | 343.616677887 | -5.95266918055 | PASS | 365.365688631 | 343.616677887 | -5.95266918055 | PASS |
| pass2_all | p2_32 | repeat03 | tpot_ms (ms) | 19.9614620327 | 19.9366675342 | -0.124211836169 | PASS | 19.9614620327 | 19.9366675342 | -0.124211836169 | PASS |
| pass2_all | p2_32 | repeat03 | request_e2e_time_ms (ms) | 20785.9413481 | 20738.8275654 | -0.226661770748 | PASS | 20785.9413481 | 20738.8275654 | -0.226661770748 | PASS |
| pass2_all | p2_32 | repeat03 | token_throughput_tps (tokens/s) | 6228.72026272 | 6224.26391513 | -0.0715451554327 | PASS | 6228.72026272 | 6224.26391513 | -0.0715451554327 | PASS |
| pass2_all | p2_64 | pooled | ttft_ms (ms) | 741.216351415 | 699.733815475 | -5.59654895108 | PASS | 720.639680322 | 700.865508119 | -2.74397493542 | PASS |
| pass2_all | p2_64 | pooled | tpot_ms (ms) | 25.870530968 | 25.2930176882 | -2.23232093881 | PASS | 25.878311466 | 25.2930042573 | -2.2617673856 | PASS |
| pass2_all | p2_64 | pooled | request_e2e_time_ms (ms) | 27206.7695317 | 26574.4909106 | -2.32397536363 | PASS | 27194.15231 | 26575.6088633 | -2.2745457907 | PASS |
| pass2_all | p2_64 | pooled | token_throughput_tps (tokens/s) | 5963.20631789 | 5966.88941251 | 0.0617636624499 | PASS | 5885.56040229 | 5889.19553995 | 0.0617636624498 | PASS |
| pass2_all | p2_64 | repeat01 | ttft_ms (ms) | 779.8887274 | 699.733815475 | -10.2777369526 | FAIL | 718.057861836 | 703.22090094 | -2.06626257917 | PASS |
| pass2_all | p2_64 | repeat01 | tpot_ms (ms) | 25.4298687355 | 25.2930176882 | -0.538150820437 | PASS | 25.4359296912 | 25.2929763034 | -0.562013614366 | PASS |
| pass2_all | p2_64 | repeat01 | request_e2e_time_ms (ms) | 26794.6444438 | 26574.4909106 | -0.821632597923 | PASS | 26739.013936 | 26577.9356594 | -0.602409187588 | PASS |
| pass2_all | p2_64 | repeat01 | token_throughput_tps (tokens/s) | 5948.02962943 | 5966.88941251 | 0.317076145582 | PASS | 5715.68472203 | 5733.80779484 | 0.317076145582 | PASS |
| pass2_all | p2_64 | repeat02 | ttft_ms (ms) | 725.492279951 | 699.733815475 | -3.55048085116 | PASS | 725.492279951 | 699.733815475 | -3.55048085116 | PASS |
| pass2_all | p2_64 | repeat02 | tpot_ms (ms) | 26.2554955812 | 25.2930176882 | -3.66581499078 | PASS | 26.2554955812 | 25.2930176882 | -3.66581499078 | PASS |
| pass2_all | p2_64 | repeat02 | request_e2e_time_ms (ms) | 27584.8642595 | 26574.4909106 | -3.66278165964 | PASS | 27584.8642595 | 26574.4909106 | -3.66278165964 | PASS |
| pass2_all | p2_64 | repeat02 | token_throughput_tps (tokens/s) | 5976.59289203 | 5966.88941251 | -0.162358047427 | PASS | 5976.59289203 | 5966.88941251 | -0.162358047427 | PASS |
| pass2_all | p2_64 | repeat03 | ttft_ms (ms) | 718.268046894 | 699.733815475 | -2.58040595004 | PASS | 718.268046894 | 699.733815475 | -2.58040595004 | PASS |
| pass2_all | p2_64 | repeat03 | tpot_ms (ms) | 25.9262285874 | 25.2930176882 | -2.4423563846 | PASS | 25.9262285874 | 25.2930176882 | -2.4423563846 | PASS |
| pass2_all | p2_64 | repeat03 | request_e2e_time_ms (ms) | 27240.7998918 | 26574.4909106 | -2.44599638754 | PASS | 27240.7998918 | 26574.4909106 | -2.44599638754 | PASS |
| pass2_all | p2_64 | repeat03 | token_throughput_tps (tokens/s) | 5965.06571958 | 5966.88941251 | 0.0305728891953 | PASS | 5965.06571958 | 5966.88941251 | 0.0305728891953 | PASS |

## Limits

This is a paired request-population reading of retained runs, not a replay of the system after stronger warmup. It does not subtract handshake time from other requests, reconstruct counterfactual scheduling, assert exact handshake durations, or prove full-domain profiling/input qualification. Numerical warm PASS does not close the remaining formal admission requirements. Only three retained native repeats are reported, without CIs.
