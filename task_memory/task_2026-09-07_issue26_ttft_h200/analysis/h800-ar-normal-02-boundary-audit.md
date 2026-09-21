## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-09 | Quantified H800 normal-02 boundary rows and recorded the initialization/OOM limitation. |

# H800 normal-02 boundary audit

## Scope

The artifact is
`analysis/h800-ar-normal-02/operators/moe_boundary.rank*.jsonl` from
`yc26-h800-ar-normal-20260909-02`. The audit checks row completeness, phase
ordering, shapes, host-side `combine_return -> tp_ar_call` gaps, and
`tp_ar_call -> tp_ar_return` scopes. It does not treat this run as a valid
first-forward measurement.

## Run validity

The environment probe passed on all eight devices (`NVIDIA H800`, CUDA compute
sanity `32.0`). The server then failed during engine initialization with:

```text
CUDA out of memory. Tried to allocate 2.37 GiB ... GPU 0 ... 477.06 MiB is free
```

The same OOM appears for both DP engine cores in `operators/server.log` at
08:35:44. There is no client log, no non-empty `server.batch.*.jsonl`, and no
non-empty `server.ops.*.jsonl`; consequently no formal request or first-forward
boundary can be identified. The boundary records were emitted before the
request path completed, while the model/engine was being initialized or
profiled. They are diagnostic initialization activity only.

## Row and schema checks

- 8 rank files × 144 rows = **1,152 rows**.
- Each rank has 48 `combine_return`, 48 `tp_ar_call`, and 48 `tp_ar_return`
  rows; phase counts are 384 each.
- Every row uses the same shape `[16384, 2048]`, `numel=33,554,432`,
  `dtype=torch.bfloat16` (64 MiB of BF16 payload).
- For all 384 rank/layer triplets,
  `combine_return <= tp_ar_call <= tp_ar_return`; there are zero phase-order
  violations.

## Host timing statistics (diagnostic only)

Statistics below aggregate the 48 layers in each rank file. Values are
milliseconds from host monotonic timestamps, not CUDA event durations.

| global rank | combine→call median / mean / max | call→return median / mean / max |
| ---: | ---: | ---: |
| 0 | 0.645 / 3.272 / 73.452 | 1.059 / 8.700 / 41.178 |
| 1 | 0.675 / 4.008 / 68.130 | 1.253 / 5.023 / 39.242 |
| 2 | 0.617 / 1.874 / 29.933 | 1.200 / 5.709 / 41.419 |
| 3 | 0.641 / 2.646 / 13.752 | 0.933 / 6.114 / 45.175 |
| 4 | 0.716 / 5.921 / 73.219 | 1.209 / 9.366 / 71.904 |
| 5 | 0.645 / 1.912 / 37.422 | 1.230 / 8.300 / 72.062 |
| 6 | 0.685 / 2.720 / 18.252 | 0.952 / 7.997 / 43.545 |
| 7 | 0.621 / 1.745 / 18.026 | 1.018 / 6.878 / 40.904 |

Across TP groups `[0,1,2,3]` and `[4,5,6,7]`, `combine_return` timestamps
are separated by **81.921–540.678 ms** and **100.951–553.860 ms** per layer,
respectively. These very large spreads are incompatible with a single
request's layer timeline and further indicate that rows belong to
initialization/profiling scheduling. They must not be interpreted as
cross-rank post-MoE AR skew for the target first-forward request.

## Conclusion and limits

**Evidence:** the artifact is structurally complete for the logger, but the
engine OOMed before serving requests; all operator and batch files are empty.

**Inference:** the rows cannot establish a normal-vs-scalar-vs-skip causal
relationship, a DP-pair completion skew, or a CUDA batch span. The observed
hundreds-of-milliseconds cross-rank timestamp spread is initialization noise,
not evidence for the target request.

**Required next evidence:** rerun normal-02 with a memory-safe configuration
that reaches the client request path, then require non-empty batch/operator
artifacts and a request identity before aligning boundaries.
