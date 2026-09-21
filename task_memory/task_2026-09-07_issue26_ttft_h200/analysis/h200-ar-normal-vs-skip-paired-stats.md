## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-09 | Added H200 normal/skip first-forward paired boundary and outer-span statistics. |

# H200 Normal vs Skip Paired Statistics

## Scope and selection

This is a diagnostic-only comparison of `h200-ar-normal-05` and
`h200-ar-skip-05`. The selected batch is the first 4096-prefill batch in each
run:

- `batch_id=0`
- `batch_num_prefill_tokens=4096`
- `request_ids=["cmpl-warmup:pf4096_dc1024:r0:0-0"]`
- DP0 batch files, TP0--TP3

The four batch files provide the batch identity and outer span. The boundary
files are `moe_boundary.rank0.jsonl` through `rank3.jsonl`; in this artifact
set these files contain the four TP lanes used for the first request. I used
the first 144 rows from each boundary file, corresponding to 48 layers x
(`combine_return`, `tp_ar_call`, `tp_ar_return`). The boundary logger does not
emit request_id or batch_id, so this 144-row selection is anchored to the
first-request sequence and cross-checked against the first batch identity;
it is not an independently encoded request join.

## Outer batch span

`outer span` is `max(TP0, TP1, TP2, TP3)` for the selected DP0 batch. The
rank spread is `max - min` across those four values.

| Mode | TP0 (ms) | TP1 (ms) | TP2 (ms) | TP3 (ms) | Outer max (ms) | Rank spread (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 17960.621094 | 18024.658203 | 18087.072266 | 18062.246094 | 18087.072266 | 126.451172 |
| skip | 19641.011719 | 19656.929688 | 19616.792969 | 19561.205078 | 19656.929688 | 95.724609 |

The `skip - normal` outer-span delta is **+1569.857422 ms** (`+8.679%`,
relative to normal). This is the result of these two independent diagnostic
processes; it does not establish that post-MoE AR costs 1.57 s. In particular,
`skip` changes the TP-reduced tensor value and therefore can change later
execution and scheduling.

## Boundary scope distributions

For each TP lane, `AR scope` is `tp_ar_return - tp_ar_call` and
`pre-AR gap` is `tp_ar_call - combine_return`, both in milliseconds, for each
of the 48 selected layers. `P90` uses the nearest observed sample at the 90th
percentile (`sorted(values)[int(0.9*n)-1]`, n=48). `sum` is included to expose
the aggregate rank-local scope but is not used as an outer critical path.

### normal

| TP | AR median | AR P90 | AR max | AR sum | Pre-AR median | Pre-AR P90 | Pre-AR max | Pre-AR sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP0 | 4.463778 | 23.086384 | 48.271400 | 475.479548 | 4.052426 | 7.685008 | 79.143850 | 352.502167 |
| TP1 | 5.028046 | 23.943959 | 104.041537 | 623.451842 | 4.019351 | 18.023069 | 79.586770 | 411.360952 |
| TP2 | 4.245168 | 33.933821 | 102.914111 | 660.082271 | 4.214473 | 24.360780 | 80.735915 | 505.241287 |
| TP3 | 4.321283 | 27.846760 | 40.390556 | 504.787001 | 4.231785 | 15.840320 | 80.766901 | 425.390743 |

Across the four ranks, the per-layer cross-rank spread is:

| Quantity | Median (ms) | P90 (ms) | Max (ms) |
| --- | ---: | ---: | ---: |
| AR scope spread | 19.413610 | 37.656926 | 99.960741 |
| Pre-AR gap spread | 11.653358 | 27.499937 | 76.865435 |

### skip

In `skip`, `tp_ar_call -> tp_ar_return` wraps the local clone path; it is not
a real collective duration.

| TP | Local scope median | P90 | Max | Sum | Pre-AR median | Pre-AR P90 | Pre-AR max | Pre-AR sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP0 | 4.301133 | 23.593446 | 89.100787 | 618.644659 | 4.160213 | 18.037154 | 100.355819 | 448.941325 |
| TP1 | 3.971567 | 19.214625 | 101.920742 | 538.541520 | 3.958793 | 17.945023 | 78.088008 | 425.714777 |
| TP2 | 4.218389 | 33.636830 | 89.227075 | 656.992453 | 4.144659 | 19.772727 | 100.462975 | 495.614804 |
| TP3 | 3.988934 | 33.950549 | 65.097242 | 507.207382 | 3.973406 | 14.635426 | 87.670790 | 381.511436 |

Per-layer cross-rank spread:

| Quantity | Median (ms) | P90 (ms) | Max (ms) |
| --- | ---: | ---: | ---: |
| Local-scope spread | 11.091387 | 61.305380 | 97.954032 |
| Pre-AR gap spread | 9.567780 | 49.738776 | 96.639918 |

## Interpretation and limits

The H200 pair does not show a stable reduction in outer span after removing
the TP AR: the selected `skip` run is 8.679% slower than `normal`. The
per-layer rank spread also remains present in `skip`, but its local scope is a
clone scope rather than AR. This is evidence that the observed rank skew is
not sufficient to attribute to the full-payload collective alone; it also
contains arrival/queue/runtime effects and run-to-run variation.

These two runs are not a repeated paired experiment with a shared process or
shared CUDA timeline. The boundary timestamps are host `perf_counter_ns`
values with no CUDA event completion, NCCL kernel interval, or request_id.
Consequently this artifact supports the H200 descriptive comparison only. It
does not close the causal chain

`DP combine completion -> TP AR submission -> shared TP completion -> outer span`,

and it does not authorize a Frontier operator or communication-model change.
