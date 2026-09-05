## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-05 | Added final verification rerun evidence from fresh artifact roots. |
| 2026-09-05 | Added post-fix semantic suite count and fresh legal topology rerun evidence. |
| 2026-09-05 | Added final fresh artifact paths, legal topology trace, and numeric EP evidence. |
| 2026-09-05 | Recorded final co-location, PDD, and PD-AF unit and direct smoke validation after semantic repair. |
| 2026-09-05 | Added fresh backend-layout regression and post-repair architecture smoke evidence. |
| 2026-09-05 | Added owner-lane completion and dual-DP smoke evidence. |
| 2026-09-05 | Added stale owner-release sibling-wakeup regression evidence. |

## Test Script Information

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/semantic-alignment-repair-20260905`
- Branch: `feat/semantic-alignment-repair-20260905`
- Python: system `python` (`Python 3.12.3`); no Conda executable was available in this shell
- Commands:
  - `python -m pytest tests/unit/test_cc_backend_replica_local_layout.py tests/unit/test_parallel_semantics.py tests/unit/test_canonical_parallel_names.py -q -p no:cacheprovider`
  - The merged semantic unit command covered 22 targeted files and returned `720 passed, 19 skipped, 115 warnings`.
  - `python -m pytest tests/unit/test_parallel_semantics.py tests/unit/test_canonical_parallel_names.py tests/unit/test_shared_ep_layer_protocol_guard.py tests/unit/test_moe_predictor_layer_id_semantics.py -q -p no:cacheprovider`
  - `python -m pytest tests/unit/test_pd_moe_lifecycle_reproducer.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_shared_ep_layer_protocol_guard.py -q -p no:cacheprovider`
  - `python -m pytest tests/unit/test_pdaf_config_contract.py tests/unit/test_pdaf_m2n_metrics.py tests/unit/test_decode_ffn_architecture_metadata_integrity.py tests/unit/test_decode_ep_wave_materialization.py -q -p no:cacheprovider`
  - `python -m pytest tests/unit/test_examples_dummy_run_matrix.py tests/unit/test_examples_documentation_contracts.py -q -p no:cacheprovider`
  - `RUN_ID=semantic_repair_20260905_coloc NUM_REQUESTS=1 PREFILL_TOKENS=8 DECODE_TOKENS=2 QPS=100 METRICS_OUTPUT_DIR=/data/ycfeng/tmp/frontier_semantic_repair_coloc_20260905 DECODE_CUDA_GRAPH_MODE=none bash examples/architecture/co-location/offline/moe_model_basic.sh`
  - `RUN_ID=semantic_repair_20260905_pdd NUM_REQUESTS=1 PREFILL_TOKENS=8 DECODE_TOKENS=2 QPS=100 METRICS_OUTPUT_DIR=/data/ycfeng/tmp/frontier_semantic_repair_pdd_20260905 bash examples/architecture/pdd/offline/moe_model_basic.sh`
  - `RUN_ID=semantic_repair_20260905_pdaf NUM_REQUESTS=1 PREFILL_TOKENS=8 DECODE_TOKENS=2 QPS=100 METRICS_OUTPUT_DIR=/data/ycfeng/tmp/frontier_semantic_repair_pdaf_20260905 bash examples/architecture/pd-af-disagg/offline/moe_model_ep.sh`
  - `python -m pytest tests/unit/test_stage_execution_context.py tests/unit/test_cluster_scheduler_dp_lanes.py tests/unit/test_pdaf_prefill_model_time.py -q -p no:cacheprovider`
  - `python -m pytest tests/unit/test_cc_backend_replica_local_layout.py tests/unit/test_parallel_semantics.py tests/unit/test_canonical_parallel_names.py tests/unit/test_shared_ep_layer_protocol_guard.py tests/unit/test_moe_predictor_layer_id_semantics.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_routing_input_contract.py tests/unit/test_moe_routing_runtime.py tests/unit/test_cluster_scheduler_dp_lanes.py tests/unit/test_replica_identity_contract.py tests/unit/test_stage_execution_context.py tests/unit/test_prefill_ep_wave_materialization.py tests/unit/test_decode_ep_wave_materialization.py tests/unit/test_pd_moe_lifecycle_reproducer.py tests/unit/test_pd_decode_moe_layer_accounting.py tests/unit/test_pd_decode_moe_completion_contract.py tests/unit/test_pdaf_config_contract.py tests/unit/test_pdaf_m2n_metrics.py tests/unit/test_pdaf_ep_stage_accounting.py tests/unit/test_pdaf_cluster_scheduler_invariants.py tests/unit/test_pdaf_prefill_model_time.py tests/unit/test_transfer_metrics_contract.py -q -p no:cacheprovider`
  - Fresh direct co-location dual-DP MoE command used `ATTN_TP=4`, `ATTN_DP=2`, `MOE_TP=1`, `MOE_EP=8`, `NUM_REQUESTS=2`, `PREFILL_TOKENS=4`, `DECODE_TOKENS=2`, `QPS=1000`, dummy execution time `1 ms`, and `DECODE_CUDA_GRAPH_MODE=none`; artifacts were written under `/data/ycfeng/tmp/review_moe_dp2_legal_20260905`.
  - Final fresh co-location, PDD, and PD-AF commands were rerun with the same
    one-request dummy settings and output roots
    `/data/ycfeng/tmp/frontier_semantic_repair_final_coloc_20260905`,
    `/data/ycfeng/tmp/frontier_semantic_repair_final_pdd_20260905`, and
    `/data/ycfeng/tmp/frontier_semantic_repair_final_pdaf_20260905`.
    The reproducible invocations were:
    `RUN_ID=semantic_repair_final_coloc_20260905 NUM_REQUESTS=1 PREFILL_TOKENS=8 DECODE_TOKENS=2 QPS=100 METRICS_OUTPUT_DIR=/data/ycfeng/tmp/frontier_semantic_repair_final_coloc_20260905 DECODE_CUDA_GRAPH_MODE=none bash examples/architecture/co-location/offline/moe_model_basic.sh`,
    `RUN_ID=semantic_repair_final_pdd_20260905 NUM_REQUESTS=1 PREFILL_TOKENS=8 DECODE_TOKENS=2 QPS=100 METRICS_OUTPUT_DIR=/data/ycfeng/tmp/frontier_semantic_repair_final_pdd_20260905 bash examples/architecture/pdd/offline/moe_model_basic.sh`,
    and
    `RUN_ID=semantic_repair_final_pdaf_20260905 NUM_REQUESTS=1 PREFILL_TOKENS=8 DECODE_TOKENS=2 QPS=100 METRICS_OUTPUT_DIR=/data/ycfeng/tmp/frontier_semantic_repair_final_pdaf_20260905 bash examples/architecture/pd-af-disagg/offline/moe_model_ep.sh`.
  - Final legal topology command used `ATTN_TP=4`, `ATTN_DP=2`, `MOE_TP=1`,
    `MOE_EP=8`, `NUM_REQUESTS=2`, `PREFILL_TOKENS=4`, `DECODE_TOKENS=2`,
    `QPS=1000`, dummy execution time `1 ms`, and wrote metrics to
    `/data/ycfeng/tmp/frontier_semantic_repair_final_tp4_dp2_ep8_20260905` and
    cluster events to
    `/data/ycfeng/tmp/frontier_semantic_repair_final_tp4_dp2_ep8_20260905_trace`.
  - Post-fix legal topology rerun used the same parameters with
    `--vllm_v1_scheduler_config_enable_chunked_prefill` and wrote metrics to
    `/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905`
    and event logs to
    `/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905_trace`.
    The command exited `0` and its complete stdout/stderr was archived at
    `/data/ycfeng/tmp/frontier_semantic_repair_final4_tp4_dp2_ep8_20260905.run.log`.

## Validation Criteria

- Co-location and PDD smoke must exit zero, complete one request, and write `request_metrics.csv` and `system_metrics.json`.
- PD-AF smoke must reach the first scheduler cycle without an exception and complete one request.
- Dual-DP co-location smoke must complete both requests and leave the event queue and scheduler registry empty.
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

## Owner-Lane Regression Evidence

| Check | Result | Evidence |
| --- | --- | --- |
| Owner-lane focused units | PASS | `37 passed` across `test_stage_execution_context.py`, `test_cluster_scheduler_dp_lanes.py`, and `test_pdaf_prefill_model_time.py` |
| Merged semantic targeted units | PASS | `720 passed, 19 skipped, 115 warnings` |
| Dual-DP co-location MoE smoke | PASS | exit `0`; request 0 completed at `1.943999999999991 s`; request 1 completed at `1.4579999999999942 s`; `completed_requests=2`; E2E mean `1700.9999999999927 ms`; stage-ledger identities were `ATTN_DP_LANE`; no residual scheduler state |
| PDD fresh dummy smoke | PASS | exit `0`; `completed_requests=1`; E2E `60.52097151999969 ms`; 10 tokens |
| PD-AF fresh dummy smoke | PASS | exit `0`; `completed_requests=1`; E2E `631.2761049599992 ms`; 10 tokens |
| Source hygiene | PASS | `git diff --check` exit `0`; `python -m compileall -q frontier tests/unit` exit `0` |

The PD-AF repair closed two deterministic runtime defects observed during direct smoke. First, `_replica_load_tracker` used integer keys while `_update_load_tracker()` expected `(replica_id, dp_id)` keys. Second, the metrics store indexed DECODE_ATTN full-stage events as DP-local meters, and the scheduler attempted to read a non-existent `decode_attn_original_dp_id` field on returning batches. The final implementation restores the legacy DECODE_ATTN full-stage `None` identity, sizes metrics by the active scheduler lane contract, and completes the A↔F path.

The documentation failures are unrelated to runtime behavior and were not changed in this validation subtask.

The stale completion regression verifies the previously missing release
boundary: a stale event whose identity disagrees with the batch still releases
the batch owner, marks the shared stage idle, and returns one queued sibling
schedule event. This closes the residual-ticket failure identified in the
independent review.

The public-example review found a separate wrapper-level failure: the legacy
guard rejects `ATTN_TP=4, ATTN_DP=2, MOE_TP=1, MOE_EP=8` before Frontier starts.
This report records it as deferred scope; the direct CLI dual-DP run above
verified the runtime contract without reusing legacy H800 data.

## Follow-up Validation

The backend repair removed outer-Replica factors from both runtime backends.
For `num_replicas=3, attn_dp=2, attn_tp=4, moe_tp=1, moe_ep=8`, fresh
materialization produced:

```text
world=8
attention={TP: 4, CP: 1, DP: 2, EP: 1}
moe={TP: 1, CP: 1, DP: 1, EP: 8}
```

The focused backend regression returned `12 passed`; the merged semantic unit
command returned `232 passed, 115 warnings`. After this repair, all three
direct dummy smokes passed:

| Surface | Exit | E2E mean | Throughput |
| --- | ---: | ---: | ---: |
| Co-location | 0 | `1036.0 ms` | `9.65250965250968 tokens/s` |
| PDD | 0 | `60.52097152 ms` | `165.23198073076887 tokens/s` |
| PD-AF | 0 | `631.27610496 ms` | `15.84092906642435 tokens/s` |

Each run completed one request and wrote both request and system metrics under
`/data/ycfeng/tmp/frontier_semantic_repair_*_20260905`.

## Final Fresh Trace

The final direct topology artifact is
`/data/ycfeng/tmp/frontier_semantic_repair_final_tp4_dp2_ep8_20260905/phi_tiny_moe_instruct/offline_batch/semantic_repair_final_tp4_dp2_ep8_20260905/`.
It contains two request rows. Both rows report E2E `971.9999999999973 ms`,
prefill `486.0000000000004 ms`, decode `485.99999999999693 ms`, and six total
tokens (`4` prefill and `2` decode). The stage ledger has four rows with
`replica_local_id={0,1}`, `execution_scope={ATTN_DP_LANE}`, and request IDs
`{0,1}`. The event trace records `64` `EP-CONSERVATION` lines, `128`
`EP-BARRIER` lines, and `64` `EP-WAVE-END` lines. Every barrier reports
`expected_ep_ids=[0,1,2,3,4,5,6,7]` and the same complete arrived set.

The final three architecture artifacts are fresh and contain these measured
values:

| Architecture | Requests | Tokens | E2E mean (ms) | Throughput (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| Co-location | 1 | 10 | `1035.9999999999968` | `9.65250965250968` |
| PDD | 1 | 10 | `60.52097151999969` | `165.23198073076887` |
| PD-AF | 1 | 10 | `631.2761049599992` | `15.84092906642435` |

No vLLM groundtruth run was performed and no legacy H800 CSV was read.

## Final Verification Rerun

The completion gate was rerun from the current worktree with new artifact
roots. All three architecture commands exited `0`, completed one request,
and produced both `request_metrics.csv` and `system_metrics.json`:

| Architecture | Artifact root | E2E (ms) | Prefill (ms) | Decode (ms) | Tokens | Throughput (tokens/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Co-location | `/data/ycfeng/tmp/frontier_semantic_repair_verify_coloc_20260905/` | `1035.9999999999968` | `518.0000000000002` | `517.9999999999967` | `10` | `9.65250965250968` |
| PDD | `/data/ycfeng/tmp/frontier_semantic_repair_verify_pdd_20260905/` | `60.52097151999969` | `20.00000000000001` | `40.52097151999968` | `10` | `165.23198073076887` |
| PD-AF | `/data/ycfeng/tmp/frontier_semantic_repair_verify_pdaf_20260905/` | `631.2761049599992` | `20.00000000000001` | `611.2761049599992` | `10` | `15.84092906642435` |

The same completion gate reran the 22-file semantic command and returned
`720 passed, 19 skipped, 115 warnings` with exit code `0`; `compileall`, JSON
parsing, and `git diff --check` passed. The rerun remained dummy Frontier-only:
`groundtruth_rerun=false` and no legacy H800 CSV was reused.
