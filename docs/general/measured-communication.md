# Measured collective communication on H800

The opt-in `measured` CC backend reads vLLM collective device times. It does not
change any existing backend or the default `astra_sim_analytical` selection.

```bash
--cc_backend_config_type measured \
--measured_cc_backend_config_summary_csv_path '{profiling_data_dir}/{NETWORK_DEVICE}/summary.csv' \
--replica_config_network_device h800_dgx
```

The existing per-role type flags can also select `measured`. All roles inherit
the same `summary_csv_path` from the base measured config; there are no new
per-role path fields. Paths expand `profiling_data_dir` and `NETWORK_DEVICE`.

## Payload and lookup contract

| API operation | Communication domain | Measured operation/group |
| --- | --- | --- |
| `allreduce` | `ATTN_TP` | `all_reduce` / `tp` |
| `allgather` | `ATTN_TP` | `all_gather` / `tp` |
| `reduce_scatter` | `ATTN_TP` | `reduce_scatter` / `tp` |
| `allgather` | `EP`, `MOE_EP` | `all_gatherv` / `ep` |
| `reduce_scatter` | `EP`, `MOE_EP` | `reduce_scatterv` / `ep` |

`data_size_bytes` is the largest rank's input, not aggregate wire traffic.
For dispatch this is the largest local SP row count times the combined bytes
of bf16 hidden states, fp32 top-k weights and int32 top-k IDs. For combine the
input is the complete gathered hidden-state matrix on each rank. Callers own
these shapes and the actual collective selection.

The loader pools eager and CUDA Graph `median_us` rows at equal bytes using
their median, ignoring host launch timings. It excludes CPU `dp` rows,
`single_active` holdouts and `logits_all_gather`. Logits uses the hidden-state
curve (option B): its separate layout-copy cost is not charged. That knowingly
underprices one logits call by 9-29% in the measurements (about 0.1% of a b64
step), but avoids distorting all 74 hidden-state gathers per step.

Interpolation is piecewise linear in log(bytes) versus log(us). Exact measured
points are preserved; non-monotone segments are logged, not smoothed.
`describe_curves()` reports keys, source paths, contributing calls, point counts,
byte/time ranges and absolute relative leave-one-out errors. Sparse/singleton
curves report undefined leave-one-out statistics as `None`.

`MeasuredCCOutOfRangeError` contains `curve_key`, `requested_bytes`,
`covered_range` (`None` for an unknown key) and `reason`. Zero bytes, unmeasured
group sizes/domains/operations and requests outside the byte range fail with
this type. There is no extrapolation or analytical fallback.

## Checked-in K5 data and coverage

Both files are unchanged curated producer outputs under
`data/profiling/network/h800_dgx/`:

| File | Data rows | SHA-256 |
| --- | --- | --- |
| `summary.csv` | 971 | `3c0b5de1c853da75ff17548c160ebf42ddefe8174bbfeb4275b5a5c074fc3461` |
| `nixl_summary.csv` | 186 | `8b16ca8f34845d4319bb350b059721b1361db6eefeab7260ec262693147579e0` |

Measurements use vLLM `2cf0a6915ce544dc493a0990f2ea38d81601128a`, H800,
TP8/DP4/EP32 and the `allgather_reducescatter` backend, with three shuffled
passes and device-side start alignment. `summary.csv` includes CPU and uneven
holdouts. Curation omitted one logits row whose queue-hold check failed.

| Curve | Points | Input bytes |
| --- | --- | --- |
| TP8 all-reduce | 14 | 12,288–100,663,296 |
| TP8 hidden-state all-gather | 12 | 12,288–25,165,824 |
| TP8 reduce-scatter | 11 | 98,304–100,663,296 |
| EP32 dispatch | 11 | 12,352–12,648,448 |
| EP32 combine | 11 | 393,216–402,653,184 |

- Decode covers b1–b512 per DP rank, including R512 logits bytes. Prefill
  beyond 8,192 tokens per DP rank exceeds the dispatch range and must fail.
- EP32 has only one allocation's evidence. Cross-allocation reproducibility
  is not established. LOO errors exceed 10% at some points; loaded coverage is
  not an accuracy guarantee.
- Unequal SP row counts select different NCCL kernels. The scalar API cannot
  infer this from byte size. The future SP-MoE caller must reject uneven
  combine with the typed coverage error **before lookup**: uniform curves
  underpredict it by up to 85%. Do not pool holdouts as if they were the same
  kernel. Unequal raw tokens that still yield equal SP rows are not this case.
  Dispatch retains largest-rank bytes; its DP4 uneven holdout is within 10%
  through 512 tokens, not at every larger measured size.
- No device charge is made for async gloo DP synchronization. Same-NIC EP/NIXL
  contention and multiple queued reads remain unmodeled.

## GLM NIXL run configuration

The K5 eight-reader-pair fit uses the existing analytical KV predictor:

```bash
--analytical_kv_cache_transfer_config_kv_connector nixl \
--analytical_kv_cache_transfer_config_network_bandwidth_gbps 126.2 \
--analytical_kv_cache_transfer_config_network_latency_ms 0.0647
```

These values are applied only in the GLM P/D driver
`tests/integration/run_glm53_structural_pd.py`; global defaults stay unchanged.
The NIXL CSV is retained evidence for this fit, not a collective-backend input.
The fit replaces 129.6 Gbit/s with zero latency and agrees with K5 eight-pair
rows within approximately -3.8% to +2.3%. Small reads show up to 30% host-pair
variation in the earlier campaign. The driver still uses synthetic compute and
is a structural check, not a calibrated E2E run.

The measured backend alone does not enable GLM SP collective assembly or
measured compute. Those consumers are separate implementation steps; selecting
this backend before their integration intentionally exposes missing coverage
rather than substituting another cost model.
