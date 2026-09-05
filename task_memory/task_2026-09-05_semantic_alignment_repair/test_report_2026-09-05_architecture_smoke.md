## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Recorded final co-location, PDD, and PD-AF unit and direct smoke validation after semantic repair. |

## Test Script Information

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/semantic-alignment-repair-20260905`
- Branch: `feat/semantic-alignment-repair-20260905`
- Python: `python` from the active environment (`python -m pytest`)
- Commands:
  - `python -m pytest tests/unit/test_parallel_semantics.py tests/unit/test_canonical_parallel_names.py tests/unit/test_shared_ep_layer_protocol_guard.py tests/unit/test_moe_predictor_layer_id_semantics.py -q -p no:cacheprovider`
  - `python -m pytest tests/unit/test_pd_moe_lifecycle_reproducer.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_shared_ep_layer_protocol_guard.py -q -p no:cacheprovider`
  - `python -m pytest tests/unit/test_pdaf_config_contract.py tests/unit/test_pdaf_m2n_metrics.py tests/unit/test_decode_ffn_architecture_metadata_integrity.py tests/unit/test_decode_ep_wave_materialization.py -q -p no:cacheprovider`
  - `python -m pytest tests/unit/test_examples_dummy_run_matrix.py tests/unit/test_examples_documentation_contracts.py -q -p no:cacheprovider`
  - Direct smoke commands used `NUM_REQUESTS=1`, `PREFILL_TOKENS=8`, `DECODE_TOKENS=2`, `QPS=100`, dummy execution time `1.0ms`, and metrics under `/data/ycfeng/tmp/frontier_repair_*`.

## Validation Criteria

- Co-location and PDD smoke must exit zero, complete one request, and write `request_metrics.csv` and `system_metrics.json`.
- PD-AF smoke must reach the first scheduler cycle without an exception and complete one request.
- Unit suites must pass without modifying source during validation.

## Test Results and Evidence

| Surface | Result | Evidence |
| --- | --- | --- |
| Co-location unit contract | PASS | `45 passed in 3.23s` |
| PDD lifecycle/EP unit contract | PASS | `27 passed in 3.01s` |
| PD-AF config/M2N/EP/metrics unit contract | PASS | `86 passed in 7.18s` |
| Co-location dummy smoke | PASS | exit `0`; one request; E2E `1035.9999999999968 ms`; 10 tokens; `9.65250965250968 tokens/s`; metrics files written |
| PDD dummy smoke | PASS | exit `0`; one request; E2E `60.52097151999969 ms`; 10 tokens; `165.23198073076887 tokens/s`; metrics files written |
| Example documentation contract | FAIL (baseline) | `2 failed, 8 passed`; README lacks expected current-example text and `pd-af-disaggregation` marker |
| PD-AF dummy smoke | PASS | exit `0`; one request; E2E `631.2761049599992 ms`; prefill `20.00000000000001 ms`; decode `611.2761049599992 ms`; decode-attn `480.0 ms`; decode-ffn `128.07099050666582 ms`; 10 tokens; request/system metrics written |

The PD-AF repair closed two deterministic runtime defects observed during direct smoke. First, `_replica_load_tracker` used integer keys while `_update_load_tracker()` expected `(replica_id, dp_id)` keys. Second, the metrics store indexed DECODE_ATTN full-stage events as DP-local meters, and the scheduler attempted to read a non-existent `decode_attn_original_dp_id` field on returning batches. The final implementation restores the legacy DECODE_ATTN full-stage `None` identity, sizes metrics by the active scheduler lane contract, and completes the A↔F path.

The documentation failures are unrelated to runtime behavior and were not changed in this validation subtask.

## Follow-up Validation

Runtime materialization inspection found a remaining semantic defect in both
`frontier/cc_backend/backends/collective_sim_cc_backend.py` and
`frontier/cc_backend/backends/astra_sim_analytical_cc_backend.py`:

```text
runtime_num_replicas=3, runtime_attn_dp=2
materialized attention dims = {TP: 4, CP: 1, DP: 6, EP: 1}
materialized MoE dims       = {TP: 1, CP: 1, DP: 3, EP: 8}
```

The attention `DP=runtime_attn_dp * runtime_num_replicas` expression merges
outer serving replicas into the attention-DP collective. The requested model
has no cross-replica communication, so attention-DP must remain `2` per
Replica. The MoE path also exposes outer replicas as `DP=3`; this is only
valid when the backend invocation is explicitly cluster-wide. The current
per-Replica semantic contract therefore needs an implementation decision before
collective-sim/astra parity can be claimed.

After the scheduler tracker repair landed, all three direct dummy smokes passed:

| Surface | Exit | E2E mean | Throughput |
| --- | ---: | ---: | ---: |
| Co-location | 0 | `1036.0 ms` | `9.65250965250968 tokens/s` |
| PDD | 0 | `60.52097152 ms` | `165.23198073076887 tokens/s` |
| PD-AF | 0 | `631.27610496 ms` | `15.84092906642435 tokens/s` |

Each run completed one request and wrote both request and system metrics under
`/data/ycfeng/tmp/frontier_repair_*2`.
