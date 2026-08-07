# Architecture Examples

## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-07 | Distinguished PDD parallel runtime support from the sequential example defaults and retained the PD-AF parallel guard. |
| 2026-07-23 | Completed the five-offline/five-online PD-AF dense/MoE/EP/CUDA Graph recipe surface and dummy matrix coverage. |
| 2026-07-22 | Added the `pd-af-disaggregation` offline/online MoE example surface and one-click smoke contract. |
| 2026-07-05 | Added profiling-independent dummy smoke matrix runner documentation. |
| 2026-06-22 | Removed legacy split-decode terminology from the public PDD surface. |
| 2026-06-14 | Added PDD pd-disaggregation script list, configuration contract, and validation criteria for local PR preparation. |

This directory contains one-click architecture entrypoints for Frontier's release-supported runtime layouts.

## Release Scope

`pre-release-v0.3` includes sequential **PDD / `pd-disaggregation`** and **PD-AF / `pd-af-disaggregation`** examples. PDD uses `PREFILL` plus unified `DECODE`; PD-AF uses `PREFILL`, `DECODE_ATTN`, and `DECODE_FFN` with KV and M2N transfers. PDD runtime supports both sequential and parallel cluster processing. The checked-in PDD examples remain sequential through `--no-enable_parallel_clusters` for reproducible one-click runs. PD-AF parallel cluster processing remains unsupported and fails fast.

`co-location` examples remain available as baseline comparison recipes and v0.1-compatible architecture references. Additional disaggregated research prototypes outside the PDD path are not exposed as release examples.

## Scripts

| Path | Scenario | Notes |
|------|----------|-------|
| `run_dummy_smoke_matrix.sh` | Profiling-independent dummy smoke matrix | Runs dense/MoE across co-location/PDD/PD-AF and offline/online; does not consume profiling CSV datasets |
| `co-location/run_all.sh` | Full co-location suite | Runs all five offline cases and all five online cases; pass extra Frontier CLI flags after `--` |
| `co-location/offline/dense_model_basic.sh` | Offline dense co-location baseline | Analytical backend by default, dummy execution time, `decode_cuda_graph_mode=full_decode_only`, Chunked Prefill, CSV/JSON metrics |
| `co-location/offline/moe_model_basic.sh` | Offline MoE co-location baseline | Analytical backend by default, dummy execution time, shared-domain MoE invariant, Chunked Prefill, CSV/JSON metrics |
| `co-location/offline/thinking_mode_basic.sh` | Offline Thinking Mode v1 co-location | Analytical backend; one hidden round plus one final round; CSV/JSON metrics |
| `co-location/offline/moe_spec_dec.sh` | Offline MoE Speculative Decoding / MTP | Speculative Decoding / MTP enabled; uses `decode_cuda_graph_mode=none` to avoid the current conflict |
| `co-location/offline/moe_prefix_caching.sh` | Offline MoE Prefix Caching | Prefix Caching enabled with `examples/fixtures/prefix_cache_shared_session_trace.csv` |
| `co-location/online/dense_model_basic_online.sh` | Online dense co-location baseline | Mirrors dense offline settings with analytical backend and `--simulation_mode online` |
| `co-location/online/moe_model_basic_online.sh` | Online MoE co-location baseline | Mirrors MoE offline settings with analytical backend and `--simulation_mode online` |
| `co-location/online/thinking_mode_basic_online.sh` | Online Thinking Mode v1 co-location | Mirrors Thinking Mode offline settings with `--simulation_mode online` |
| `co-location/online/moe_spec_dec_online.sh` | Online MoE Speculative Decoding / MTP | Mirrors Speculative Decoding offline settings with `--simulation_mode online` |
| `co-location/online/moe_prefix_caching_online.sh` | Online MoE Prefix Caching | Replays the same prefix-cache fixture with `--simulation_mode online` |
| `pdd/run_all.sh` | Full PDD suite | Runs all five offline PDD cases and all five online PDD cases; pass extra Frontier CLI flags after `--` |
| `pdd/offline/dense_model_basic.sh` | Offline dense PDD baseline | Sequential `pd-disaggregation`, analytical backend, dummy execution time, Chunked Prefill, CSV/JSON metrics |
| `pdd/offline/moe_model_basic.sh` | Offline MoE PDD baseline | Sequential `pd-disaggregation`, reference-runnable shared-domain MoE topology, Chunked Prefill, CSV/JSON metrics |
| `pdd/offline/thinking_mode_basic.sh` | Offline Thinking Mode v1 PDD | Thinking Mode with two KV transfer handoffs for the one-request smoke configuration |
| `pdd/offline/moe_spec_dec.sh` | Offline MoE PDD Speculative Decoding / MTP | Speculative Decoding enabled; Prefix Caching intentionally disabled; `DECODE_CUDA_GRAPH_MODE=none` |
| `pdd/offline/moe_prefix_caching.sh` | Offline MoE PDD Prefix Caching | Sticky scheduler with `examples/fixtures/prefix_cache_shared_session_trace.csv` |
| `pdd/online/dense_model_basic_online.sh` | Online dense PDD baseline | Mirrors dense offline settings with `--simulation_mode online` |
| `pdd/online/moe_model_basic_online.sh` | Online MoE PDD baseline | Mirrors MoE offline settings with `--simulation_mode online` |
| `pdd/online/thinking_mode_basic_online.sh` | Online Thinking Mode v1 PDD | Mirrors Thinking Mode offline settings with `--simulation_mode online` |
| `pdd/online/moe_spec_dec_online.sh` | Online MoE PDD Speculative Decoding / MTP | Mirrors Speculative Decoding offline settings with `--simulation_mode online` |
| `pdd/online/moe_prefix_caching_online.sh` | Online MoE PDD Prefix Caching | Replays the same prefix-cache fixture with `--simulation_mode online` |
| `pd-af-disagg/run_all.sh` | Full PD-AF suite | Runs five offline and five online cases; pass extra Frontier CLI flags after `--` |
| `pd-af-disagg/offline/dense_model_basic.sh` | Offline PD-AF dense baseline | Dense-safe three-role topology, analytical KV/M2N, dummy predictor by default |
| `pd-af-disagg/offline/moe_model_basic.sh` | Offline PD-AF MoE baseline | Three-role topology, analytical KV/M2N, dummy predictor by default |
| `pd-af-disagg/offline/moe_model_ep.sh` | Offline PD-AF EP baseline | Legal EP=2 topology with fail-fast domain checks |
| `pd-af-disagg/offline/dense_cuda_graph.sh` | Offline dense CUDA Graph | Global `use_cuda_graph` with capture sizes `8 16 32 64` |
| `pd-af-disagg/offline/moe_cuda_graph.sh` | Offline MoE CUDA Graph | EP=1 MoE with the same global CUDA Graph contract |
| `pd-af-disagg/online/dense_model_basic_online.sh` | Online PD-AF dense baseline | Mirrors the dense offline topology with online arrivals |
| `pd-af-disagg/online/moe_model_basic_online.sh` | Online PD-AF MoE baseline | Same topology with `--simulation_mode online` |
| `pd-af-disagg/online/moe_model_ep_online.sh` | Online PD-AF EP baseline | Same legal EP=2 contract with online arrivals |
| `pd-af-disagg/online/dense_cuda_graph_online.sh` | Online dense CUDA Graph | Dense online path with global CUDA Graph |
| `pd-af-disagg/online/moe_cuda_graph_online.sh` | Online MoE CUDA Graph | MoE online path with global CUDA Graph |

## PDD Configuration Contract

All PDD scripts use these release-supported defaults unless overridden from the shell:

- `--sys_arch pd-disaggregation`
- `--no-enable_parallel_clusters`
- explicit `PREFILL` and unified `DECODE` cluster settings
- `--cc_backend_config_type analytical`
- dummy execution-time prediction enabled by default
- CSV/JSON metrics enabled by default through `--metrics_config_write_metrics` and `--metrics_config_store_request_metrics`
- plots, Chrome trace, and JSON event trace disabled for lightweight one-click artifacts

These are example defaults, not a PDD runtime restriction. Remove `--no-enable_parallel_clusters` when intentionally running the supported parallel PDD event processors.

MoE PDD scripts also enforce that each role's attention and MoE parallel domains match before launching Frontier. This fail-fast check prevents known non-runnable MoE topology combinations from entering the simulator.

## PD-AF Configuration Contract

All PD-AF scripts use these release-supported defaults unless overridden from the shell:

- `--sys_arch pd-af-disaggregation`
- `--no-enable_parallel_clusters` (sequential role execution)
- explicit `PREFILL`, `DECODE_ATTN`, and `DECODE_FFN` cluster settings
- `--cc_backend_config_type analytical` and analytical KV/M2N transfer models
- dummy execution-time prediction enabled by default for profiling-independent smoke runs
- CSV/JSON metrics enabled; plots and trace exports disabled for lightweight artifacts

The EP recipe validates expert-domain equality and expert-count bounds before launching the simulator. A successful smoke run must emit `request_metrics.csv`, `system_metrics.json`, positive KV/M2N transfer metrics, and one completed row per generated request.

PD-AF CUDA Graph examples use `--use_cuda_graph`, not `--decode_cuda_graph_mode`. PD-AF does not support Thinking Mode, Speculative Decoding / MTP, or Prefix Caching in `pre-release-v0.3`. These scripts default to dummy prediction for one-click structural checks; dummy-mode evidence is not trained numerical parity.

## Thinking Mode v1

The Thinking Mode examples use:

- `--enable_thinking_mode`
- `--thinking_depth 2`
- one explicit hidden round via `--thinking_round_prefill_tokens` and `--thinking_round_decode_tokens`
- `--tool_call_latency 0.001`
- explicit `vllm_v1` scheduler settings
- `--cc_backend_config_type analytical` so the one-click smoke run works on a minimal single-replica layout
- CSV/JSON metrics enabled by default, with plots, Chrome trace, and JSON event trace disabled for lightweight artifacts

Under PDD, one user request can produce multiple prefill-to-decode KV handoffs. The default Thinking Mode smoke case completes one request and records two KV transfers.

## Recommended Start Order

```bash
# Profiling-independent smoke matrix: dense/MoE across co-location/PDD/PD-AF and offline/online.
# This does not consume profiling CSV datasets because it forces dummy execution-time prediction.
bash examples/architecture/run_dummy_smoke_matrix.sh

# Full PDD suite for pre-release-v0.3.
bash examples/architecture/pdd/run_all.sh

# PDD offline cases.
bash examples/architecture/pdd/offline/dense_model_basic.sh
bash examples/architecture/pdd/offline/moe_model_basic.sh
bash examples/architecture/pdd/offline/thinking_mode_basic.sh
bash examples/architecture/pdd/offline/moe_spec_dec.sh
bash examples/architecture/pdd/offline/moe_prefix_caching.sh

# PDD online cases.
bash examples/architecture/pdd/online/dense_model_basic_online.sh
bash examples/architecture/pdd/online/moe_model_basic_online.sh
bash examples/architecture/pdd/online/thinking_mode_basic_online.sh
bash examples/architecture/pdd/online/moe_spec_dec_online.sh
bash examples/architecture/pdd/online/moe_prefix_caching_online.sh

# Full PD-AF suite for pre-release-v0.3.
bash examples/architecture/pd-af-disagg/run_all.sh

# PD-AF offline cases.
bash examples/architecture/pd-af-disagg/offline/dense_model_basic.sh
bash examples/architecture/pd-af-disagg/offline/moe_model_basic.sh
bash examples/architecture/pd-af-disagg/offline/moe_model_ep.sh
bash examples/architecture/pd-af-disagg/offline/dense_cuda_graph.sh
bash examples/architecture/pd-af-disagg/offline/moe_cuda_graph.sh

# PD-AF online cases.
bash examples/architecture/pd-af-disagg/online/dense_model_basic_online.sh
bash examples/architecture/pd-af-disagg/online/moe_model_basic_online.sh
bash examples/architecture/pd-af-disagg/online/moe_model_ep_online.sh
bash examples/architecture/pd-af-disagg/online/dense_cuda_graph_online.sh
bash examples/architecture/pd-af-disagg/online/moe_cuda_graph_online.sh

# Full co-location comparison suite.
bash examples/architecture/co-location/run_all.sh

# Offline cases.
bash examples/architecture/co-location/offline/dense_model_basic.sh
bash examples/architecture/co-location/offline/moe_model_basic.sh
bash examples/architecture/co-location/offline/thinking_mode_basic.sh
bash examples/architecture/co-location/offline/moe_spec_dec.sh
bash examples/architecture/co-location/offline/moe_prefix_caching.sh

# Online cases.
bash examples/architecture/co-location/online/dense_model_basic_online.sh
bash examples/architecture/co-location/online/moe_model_basic_online.sh
bash examples/architecture/co-location/online/thinking_mode_basic_online.sh
bash examples/architecture/co-location/online/moe_spec_dec_online.sh
bash examples/architecture/co-location/online/moe_prefix_caching_online.sh
```

Use the dense baseline scripts first. For PD-AF, use the EP and CUDA Graph recipes as advanced cases. Thinking Mode, Speculative Decoding / MTP, and Prefix Caching advanced recipes apply only to their documented co-location/PDD paths.

## Cross-validation Criteria

For each offline/online pair:

1. Confirm the script exits with code `0`.
2. Confirm `request_metrics.csv` and `system_metrics.json` exist in the metrics output directory.
3. Record expected request count, actual request rows, completed request rows, total input tokens, total output tokens, mean TTFT, mean latency, and request throughput when present.
4. Confirm offline outputs include the `offline_batch` taxonomy segment and online outputs include `online_serving`.
5. Treat latency differences as expected when online mode preserves request arrival times; investigate only if counts, token totals, output files, or finite numeric metrics diverge unexpectedly.

For every PDD script, the release gate should additionally record:

1. The script exits with code `0`.
2. `request_metrics.csv` and `system_metrics.json` exist in the metrics output directory.
3. Request row count, `total_requests`, and `completed_requests` match the expected case size.
4. KV transfer count, total KV bytes, and KV transfer time are present and positive.
5. Request-level `ttft`, `tpot`, `request_e2e_time`, and `transfer_kv_cache` are finite and positive.
