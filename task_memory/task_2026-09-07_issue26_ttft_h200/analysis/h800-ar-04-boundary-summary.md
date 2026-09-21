## H800 `-04` bounded diagnostic boundary summary (2026-09-10)

All three H800 runs completed the client request (`request_id=warmup:pf4096_dc1024:r0:0`, `prompt_tokens=4096`, `completion_tokens_observed=1024`) and wrote 1024 batch rows for rank0. Client diagnostics reported TTFT normal `83622.728559 ms`, scalar_sync `85808.417517 ms`, skip `72944.941725 ms`; these are diagnostic values, not clean calibration metrics.

The boundary logger does not include request_id. To isolate the first request window, rows were selected from each rank at or after that run's client `dispatch_monotonic_s`, taking the first 144 rows (48 layers × 3 phases). Every rank provided all 144 rows and all three phases. The selected rows therefore form a consistent request-window candidate, subject to the logger's request identity limitation.

For each rank, `combine_return -> tp_ar_call` host gap medians/maxima (ms) and `tp_ar_call -> tp_ar_return` scope medians/maxima (ms) were:

| mode | rank0 | rank1 | rank2 | rank3 | rank4 | rank5 | rank6 | rank7 |
|---|---|---|---|---|---|---|---|---|
| normal gap | 0.694/27.390 | 0.658/105.564 | 0.664/105.738 | 0.617/102.760 | 0.624/10.782 | 0.610/106.095 | 0.578/37.375 | 0.560/71.150 |
| normal scope | 0.977/26.632 | 0.925/57.531 | 0.929/60.711 | 0.846/87.553 | 0.880/39.101 | 0.976/59.917 | 0.874/79.818 | 0.761/94.362 |
| scalar gap | 0.697/23.189 | 0.840/38.397 | 0.792/39.153 | 0.762/67.788 | 0.936/133.015 | 0.765/44.440 | 0.746/68.494 | 0.863/32.343 |
| scalar scope | 0.977/22.773 | 4.024/105.205 | 2.495/146.682 | 1.033/111.202 | 1.359/146.827 | 1.018/74.432 | 1.015/111.177 | 1.409/146.695 |
| skip gap | 0.621/31.662 | 0.651/44.626 | 0.617/55.054 | 0.649/67.833 | 0.627/52.833 | 0.645/16.171 | 0.622/42.225 | 0.563/52.930 |
| skip scope | 0.801/83.286 | 1.240/28.185 | 0.775/36.254 | 1.122/144.190 | 1.107/146.365 | 0.820/57.884 | 1.285/59.988 | 0.745/50.474 |

TP0–3 first three-layer cross-rank spreads (ms) for combine_return / tp_ar_call / tp_ar_return were:

| mode | layer 0 | layer 1 | layer 2 |
|---|---|---|---|
| normal | 2.715 / 1.757 / 38.333 | 37.494 / 37.332 / 37.411 | 37.331 / 35.969 / 53.461 |
| scalar_sync | 55.495 / 80.311 / 81.366 | 96.771 / 96.926 / 96.984 | 96.786 / 96.247 / 110.049 |
| skip | 2.984 / 35.690 / 61.083 | 66.139 / 66.283 / 83.821 | 87.940 / 87.955 / 88.009 |

The per-rank host gap medians remain sub-millisecond in normal mode while layer-level arrival spreads can reach tens of milliseconds. This supports a device/collective or upstream scheduling skew after the DP combine boundary, but the logger lacks CUDA event timestamps and explicit request_id, so it cannot yet distinguish DP collective completion from host/device queue delay. Cross-rank rows should be interpreted only within a single worker run; no H200/H800 cross-cluster comparison is valid.
