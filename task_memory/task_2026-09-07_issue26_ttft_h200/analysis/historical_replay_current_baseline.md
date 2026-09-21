## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recomputed the existing current uniform baseline using the historical percentile helper. |

# Existing baseline before historical harness replay

This is a re-analysis of the completed current uniform pair, not a new historical-harness E2E execution. All 100 formal rows are retained. TTFT remains the D006 official-server boundary on vLLM; historical batch-derived TTFT is unavailable. No acceptance threshold is changed.

| Metric | Frontier | vLLM | Absolute error | Relative error |
| --- | ---: | ---: | ---: | ---: |
| TTFT mean (ms) | 103.510401 | 127.380512 | 23.870111 | 18.7392% |
| TTFT p90 (ms) | 133.343992 | 156.124115 | 22.780123 | 14.5910% |
| TTFT p95 (ms) | 140.329971 | 165.872025 | 25.542055 | 15.3987% |
| TPOT mean (ms) | 57.583941 | 78.336092 | 20.752150 | 26.4912% |
| TPOT p90 (ms) | 58.007590 | 78.430159 | 20.422569 | 26.0392% |
| TPOT p95 (ms) | 58.024053 | 78.435924 | 20.411871 | 26.0236% |
| request E2E mean (ms) | 59011.882459 | 80264.999897 | 21253.117438 | 26.4787% |
| request E2E p90 (ms) | 59440.230125 | 80371.275568 | 20931.045443 | 26.0429% |
| request E2E p95 (ms) | 59460.272533 | 80380.511224 | 20920.238691 | 26.0265% |
| request_throughput  (requests/s) | 0.952752 | 0.787336 | 0.165417 | 21.0097% |
| token_throughput  (tokens/s) | 4878.090773 | 4031.157800 | 846.932973 | 21.0097% |
| formal completion window  (s) | 104.959096 | 127.010657 | 22.051560 | 17.3620% |
| decode throughput  (tokens/s) | 975.618155 | 806.231560 | 169.386595 | 21.0097% |

Machine-readable provenance and values: historical_replay_current_baseline.json. Mean request E2E and formal completion window are separate metrics. Total-token throughput includes prompt plus output tokens; decode throughput includes output tokens only.
