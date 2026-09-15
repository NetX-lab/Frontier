## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-15 | Added focused-suite evidence for process-global monolithic MoE routing IDs and complete manager device-event path derivation. |
| 2026-09-15 | Recorded the persistent CPU hybrid GDN Simulator E2E, generated artifacts, numerical evidence, and test-seam limits. |

# Persistent Hybrid GDN Simulator CPU E2E Test Report

## Execution

Test:

- Test file: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/unit/test_gdn_hybrid_e2e_increment14ab.py`
- Test case: `test_hybrid_gdn_real_simulator_cpu_e2e`
- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Branch: `feature-amd-sglang-gdn`
- Source plan: `/data/ycfeng/stepfun-performance-optimization/Frontier/.draft-plan/frontier_pr31_selective_integration_plan_v2_en.md` (read-only)
- Source-plan SHA-256: `17365fe6a96cecf162d63f450fac3fd83bd91caaa2abf87f04c81131a08afd66`

Environment:

- Python: 3.12.3 (`/usr/bin/python`)
- Conda environment: none; the command used the system Python executable above
- NumPy: 2.4.6
- pandas: 3.0.3
- scikit-learn: 1.9.0
- PyTorch: 2.5.1+cu124 (CUDA build visible to Python)
- `vllm`: unavailable (`ModuleNotFoundError`)
- `sglang`: unavailable (`ModuleNotFoundError`)
- `aiter`: unavailable (`ModuleNotFoundError`)

Reproducible command:

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn
PYTHONPATH=$PWD \
WANDB_DISABLED=true \
VIDUR_DISABLE_WANDB=1 \
FRONTIER_LOG_LEVEL=ERROR \
python -m pytest \
  tests/unit/test_gdn_hybrid_e2e_increment14ab.py \
  -q -p no:cacheprovider \
  --basetemp /data/ycfeng/tmp/pr31-hybrid-simulator-e2e-20260915-v2
```

The captured output is `/data/ycfeng/tmp/pr31-hybrid-simulator-e2e-20260915-v2.log`. The run completed in 3.52 seconds and reported `4 passed`.

The persistent run's simulator artifact directory is:

`/data/ycfeng/tmp/pr31-hybrid-simulator-e2e-20260915-v2/test_hybrid_gdn_real_simulator0/sim_metrics/pr31_hybrid_fixture/offline_batch/hybrid_sim/`

The directory contains `request_metrics.csv`, `system_metrics.json`, `frontier_stage_batch_ledger.jsonl`, `frontier_stage_batch_ledger_summary.json`, `monolithic_batch_metrics.csv`, `monolithic_operation_metrics.csv`, `monolithic_cpu_operation_metrics.csv`, `op_precision_metadata.csv`, `op_traces.jsonl`, and `config.json`.

## Criteria

The check targets the Increment 14A/B CPU acceptance boundary:

1. Real `GDNTrainer` training and fresh `GDNPredictor.from_directory()` artifact loading must succeed.
2. Real `ReplicaConfig`, `Simulator`, `MemoryPlanner`, scheduler, event processing, MoE EP barrier/wave handling, prefill-to-decode continuation, request completion, metrics writing, stage ledger writing, and per-layer trace writing must execute.
3. One synthetic hybrid request must commit 18 tokens: 16 prefill tokens followed by 2 decode tokens.
4. Request metrics must contain one completed request, and system metadata must report `total_requests == completed_requests == 1`.
5. The stage ledger must contain ordered prefill and decode rows with contiguous time boundaries: prefill `[16]` then decode `[0]`.
6. Per-layer traces must preserve hybrid family identity: GDN layers `{0, 1, 2, 4, 5, 6}` use `gated_delta_net`, while dense layers `{3, 7}` use `dense_attention`.
7. GDN structured traces must expose the expected operator names, including `gdn_core_decode`, `gdn_input_projections`, and `gdn_output_projection`.

## Evidence

### Test result

**PASS — `4 passed in 3.52s`.** The four tests cover the CPU GDN training/predictor contracts and the real simulator case. The simulator case completed without an exception and wrote all required artifacts listed above.

### Request and system metrics

Observed from `request_metrics.csv` and `system_metrics.json`:

```text
request_num_tokens         = 18
request_num_prefill_tokens = 16
request_num_decode_tokens  = 2
request_e2e_time_ms        = 28.460000000000015
request_execution_time_ms  = 28.460000000000015
request_model_time_ms      = 28.86
ttft_ms                    = 15.760000000000007
tpot_ms                    = 12.70000000000001
total_requests             = 1
completed_requests         = 1
```

The completion criterion passes because the sole request is present in the CSV with 18 committed tokens and the system summary reports one total and one completed request.

### Stage ledger

The ledger contains exactly two rows, both with `operation_kind = ep_ffn`:

```text
row 1: request_num_prefill_tokens = [16]
       request_num_tokens         = [16]
       stage_start_ts             = 0.0
       stage_end_ts               = 0.015760000000000007

row 2: request_num_prefill_tokens = [0]
       request_num_tokens         = [1]
       stage_start_ts             = 0.015760000000000007
       stage_end_ts               = 0.028460000000000017
```

Both intervals are non-negative and row 2 starts exactly at row 1's end. The continuation therefore reaches the decode stage and preserves ordered stage accounting.

### Per-layer operation traces

The trace file contains 78 named events, including 18 GDN events and 4 dense-attention events. Observed identities:

```text
gdn layers   = [0, 1, 2, 4, 5, 6]
dense layers = [3, 7]
gdn family   = gated_delta_net
dense family = dense_attention
gdn names    = [gdn_core_decode, gdn_input_projections, gdn_output_projection]
```

The test also asserts that the prefill and decode GDN operator maps contain the complete structured schema. Inactive phase operators are explicitly zero, so downstream trace consumers can enumerate the same schema for both phases.

### Scope and practical limit

GDN training, artifact loading, attention-family dispatch, Replica construction, MemoryPlanner setup, Simulator scheduling/events, EP barrier/wave flow, continuation, request completion, metrics, ledger, and trace persistence are production paths. The synthetic fixture does not provide complete model-specific standard attention/MoE profiling CSVs; therefore the test injects deterministic values for unrelated dense/MoE FFN and communication timings and installs the prepared predictor through the test-local `ExecutionTimePredictorRegistry.get` seam. This is evidence for control flow, phase selection, identity, state handling, and accounting. It is not production latency fidelity or benchmark parity.

`SKIP: AMD/MI355X hardware unavailable` — no AMD/MI355X host is available. Real ROCm `DEVICE_EVENT`, vLLM HIP, AITER/MXFP4, RCCL, SGLang HIP graph, AMD benchmark comparison, and groundtruth parity remain unexecuted.

## Follow-up verification — routing identity and manager path contract

### Execution

The focused regression was run from the same worktree with Python 3.12.3 and the CPU environment listed above:

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn
PYTHONPATH=$PWD \
WANDB_DISABLED=true \
VIDUR_DISABLE_WANDB=1 \
FRONTIER_LOG_LEVEL=ERROR \
python -m pytest \
  tests/unit/test_device_timer_contract.py \
  tests/unit/test_gdn_training_predictor_increment8.py \
  tests/unit/test_gdn_hybrid_e2e_increment14ab.py \
  tests/unit/test_execution_time_predictor_max_tokens_budget.py \
  tests/unit/test_attention_tp_effective_mapping.py \
  tests/unit/test_moe_ep_non_dummy_matrix.py \
  tests/unit/test_measurement_family_selector.py \
  -q -p no:cacheprovider \
  --basetemp /data/ycfeng/tmp/pr31-wiring-focused-20260915-final2
```

Targeted checks were also run independently:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR \
python -m pytest tests/unit/test_gdn_hybrid_e2e_increment14ab.py::test_hybrid_gdn_production_constructor_cpu_e2e \
  -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr31-wiring-production-20260915-routingfix

PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR \
python -m pytest tests/unit/test_moe_predictor_layer_id_semantics.py::test_monolithic_predictor_exposes_global_per_replica_layer_routing_details \
  -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr31-routing-contract-20260915

PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR \
python -m pytest tests/unit/test_measurement_family_selector.py::test_shared_manager_returns_complete_training_file_paths \
  -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr31-path-contract-20260915
```

### Criteria and evidence

| Criterion | Expected result | Observed result |
| --- | --- | --- |
| Production hybrid constructor | Real synthetic GDN and standard profile manager construction, Registry creation, and Simulator run complete | **PASS — 1 passed in 4.43s** |
| Monolithic routing identity | Shared routing map keys equal actual process-global `Cluster.replicas` keys; direct no-ID tests retain local keys | **PASS — production assertion and routing contract passed** |
| Manager training-path taxonomy | Complete path dictionary includes configured/derived compute, attention, and MoE device-event fields | **PASS — 1 passed in 2.73s** |
| Focused regression | No timer, GDN, hybrid, predictor, MoE, or measurement-family regressions | **PASS — 219 passed in 13.54s** |

The first focused rerun had failed with `ValueError: routing_details missing target_replica_id 1` because the map contained only local key `0`. The repaired path passes `Cluster.replicas.keys()` through `Simulator` and the MoE Registry, validates cardinality/uniqueness, and reruns cleanly. This is a CPU identity/control-flow result; no GPU timing or benchmark inference is made.
