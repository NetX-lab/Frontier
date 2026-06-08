## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-06-07 | Set co-location examples to default to `astra_sim_analytical` and documented `collective_sim` as optional. |
| 2026-06-07 | Added optimized co-location defaults and advanced MoE Speculative Decoding/MTP plus Prefix Caching recipes. |
| 2026-06-04 | Confirmed this directory exposes only co-location architecture scripts for `pre-release-v0.1`. |
| 2026-06-04 | Updated architecture examples for the co-location-only `pre-release-v0.1` release branch. |
| 2026-06-03 | Removed private disaggregated architecture examples from the open-source preparation branch. |

# Architecture Examples

This directory contains one-click architecture entrypoints for Frontier's release-supported runtime layout.

## Release Scope

`pre-release-v0.1` supports only `co-location`. Disaggregated architecture examples are intentionally absent from this branch because the runtime guard rejects `pd-disaggregation` and `pd-af-disaggregation`.

## Scripts

| Path | Scenario | Notes |
|------|----------|-------|
| `co-location/dense_model_basic.sh` | Dense co-location baseline | Defaults to `--cc_backend_config_type astra_sim_analytical`, dummy execution time, `decode_cuda_graph_mode=full_decode_only`, Chunked Prefill, CSV/JSON metrics |
| `co-location/moe_model_basic.sh` | MoE co-location baseline | Defaults to `--cc_backend_config_type astra_sim_analytical`, dummy execution time, `decode_cuda_graph_mode=full_decode_only`, Chunked Prefill, CSV/JSON metrics |
| `co-location/thinking_mode_basic.sh` | Thinking Mode v1 co-location | Defaults to `--cc_backend_config_type astra_sim_analytical`; one hidden round plus one final round; CSV/JSON metrics |
| `co-location/moe_spec_dec.sh` | MoE Speculative Decoding / MTP | Speculative Decoding / MTP enabled; uses `decode_cuda_graph_mode=none` to avoid the current conflict |
| `co-location/moe_prefix_caching.sh` | MoE Prefix Caching | Prefix Caching enabled with `examples/fixtures/prefix_cache_shared_session_trace.csv` |

## Thinking Mode v1

The Thinking Mode example uses:

- `--enable_thinking_mode`
- `--thinking_depth 2`
- one explicit hidden round via `--thinking_round_prefill_tokens` and `--thinking_round_decode_tokens`
- `--tool_call_latency 0.001`
- explicit `vllm_v1` scheduler settings
- `--cc_backend_config_type astra_sim_analytical` so the one-click smoke run works on a minimal single-replica layout without the optional `collective_sim` submodule
- CSV/JSON metrics enabled by default, with plots, Chrome trace, and JSON event trace disabled for lightweight artifacts


`collective_sim` is optional for these scripts. Build it only when you explicitly pass `--cc_backend_config_type collective_sim`.

## Recommended Start Order

```bash
bash examples/architecture/co-location/dense_model_basic.sh
bash examples/architecture/co-location/moe_model_basic.sh
bash examples/architecture/co-location/thinking_mode_basic.sh
bash examples/architecture/co-location/moe_spec_dec.sh
bash examples/architecture/co-location/moe_prefix_caching.sh
```

Use the baseline scripts first, then use `moe_spec_dec.sh` and `moe_prefix_caching.sh` as advanced recipes for Speculative Decoding / MTP and Prefix Caching.
