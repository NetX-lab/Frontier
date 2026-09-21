## H200 `-05` bounded diagnostic boundary summary (2026-09-10)

The first warmup request (`request_id=warmup:pf4096_dc1024:r0:0`) completed in each mode while workers continue subsequent warmup/cleanup. Client TTFT: normal `69287.300889 ms`, scalar_sync `67966.441618 ms`, skip `68724.275395 ms`. These are diagnostic metrics only.

Using each mode's client dispatch monotonic timestamp, the first 144 boundary rows per rank (48 layers × 3 phases) form a first-request window candidate. All four TP ranks had complete phase counts.

Per-rank medians/maxima in ms (`combine_return→tp_ar_call`, `tp_ar_call→tp_ar_return`):

| mode | rank0 | rank1 | rank2 | rank3 |
|---|---|---|---|---|
| normal | `3.915/36.774`, `3.965/19.230` | `3.938/31.033`, `3.963/18.066` | `3.916/35.655`, `3.991/33.603` | `3.942/64.599`, `3.959/57.415` |
| scalar_sync | `3.885/63.769`, `4.012/72.489` | `3.867/43.719`, `4.001/89.679` | `3.881/64.899`, `4.033/42.518` | `3.871/40.770`, `4.002/40.071` |
| skip | `3.922/49.647`, `4.016/100.190` | `3.910/114.202`, `4.034/103.244` | `3.894/93.998`, `3.959/68.264` | `3.938/101.106`, `4.105/99.359` |

TP0–3 cross-rank spreads for the first three layers (ms; combine/call/return):

| mode | layer 0 | layer 1 | layer 2 |
|---|---|---|---|
| normal | `19.462/19.468/19.641` | `21.401/21.370/21.394` | `21.621/32.220/32.221` |
| scalar_sync | `3.875/3.820/30.133` | `31.030/31.016/27.153` | `27.341/70.186/90.235` |
| skip | `25.624/25.584/25.541` | `50.727/62.790/58.874` | `74.127/74.271/51.479` |

H200 median host gap (~3.9 ms) is larger than H800 (~0.6–0.9 ms), while layer-level cross-rank arrival spreads remain tens of ms. The logger still lacks CUDA event completion and request_id fields, so these measurements establish boundary ordering and spread but do not classify the spread as DP collective completion, host scheduling, or queued device work.
