## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded the Increment 14A/B early CPU hybrid dispatch checkpoint. |

# Increment 14A/B Early CPU Hybrid Dispatch Checkpoint

## Execution

Environment:

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: 3.12.3
- NumPy: 2.4.6
- pandas: 3.0.3
- scikit-learn: 1.9.0
- Measurement fixture: synthetic `DEVICE_EVENT`, `device=cpu`, `runtime_stack_signature=synthetic_cpu_v1`, `tensor_parallel_size=1`

Commands:

```bash
python -m pytest \
  tests/unit/test_gdn_hybrid_e2e_increment14ab.py \
  tests/unit/test_gdn_training_predictor_increment8.py \
  tests/unit/test_gdn_semantic_core.py \
  tests/unit/test_stage_execution_time.py \
  tests/unit/test_moe_predictor_layer_id_semantics.py \
  tests/unit/test_shared_prediction_model_manager_eager_attention_decode.py \
  -q -p no:cacheprovider
```

Result: **PASS — 76 passed in 5.33s**.

```bash
python -m pytest \
  tests/unit/test_sklearn_disaggregation_execution_time_predictor.py \
  tests/unit/test_dense_execution_time_layer_scaling.py \
  tests/unit/test_attention_predictor_correctness.py \
  tests/unit/test_mla_predictor_runtime_operator_times.py \
  tests/unit/test_moe_ep_aggregate_admission.py \
  tests/unit/test_pd_decode_moe_layer_accounting.py \
  tests/unit/test_pdaf_prefill_model_time.py \
  -q -p no:cacheprovider
```

Result: **PASS — 116 passed in 4.10s**.

```bash
FROOT=/data/ycfeng/tmp/pr31-inc14ab-fidelity-20260914-run2
PYTHONPATH="$PWD" python tests/integration/run_scheduler_refactor_fidelity.py \
  --baseline "$PWD" \
  --candidate "$PWD" \
  --output "$FROOT" \
  --case co-location_offline_dense_model_basic_short \
  --workers 1
```

Result: **PASS — 1 passed, 0 failed**. The case completed its request/metrics workflow and wrote artifacts under `$FROOT`.

```bash
python -m compileall -q frontier tests
git diff --check
```

Result: **PASS** for both checks after the source/test edits.

## Criteria and evidence

| Criterion | Expected result | Observed evidence | Status |
| --- | --- | --- | --- |
| Real GDN training and fresh load | Six phase-qualified estimators load from a temporary artifact directory without runtime fitting | `GDNTrainer` fit completed; fresh `GDNPredictor` and fresh model-manager load validated `DEVICE_EVENT` and `synthetic_cpu_v1` identity | PASS |
| GDN phase dispatch | Prefill uses `gdn_core_prefill`; ordinary decode uses `gdn_core_decode` | `0.22 ms` and `0.05 ms` matched the synthetic fixture | PASS |
| One-token continuation | A one-token continuation with incomplete prefill remains `prefill` | `GDNBatchFeatures.phase == "prefill"` | PASS |
| Diagnostic aggregate exclusion | `gdn_layer_e2e` must not be a standard predictor target | Fixture sets `gdn_layer_e2e=99.0`; returned operator map contains no `gdn_layer_e2e` key | PASS |
| Hybrid layer order | Eight IDs and family sequence remain exact | IDs `(0,1,2,3,4,5,6,7)` and families `(G,G,G,A,G,G,G,A)` | PASS |
| Stage aggregation | `StageExecutionTime` sums real layer blocks and keeps owner work once | Stage result model time equals the sum of its recorded layer block times in the test | PASS |
| Existing dense/MoE behavior | Current predictor and disaggregation contracts retain their existing focused behavior | Extended suite passed 116/116; selected dense two-worktree fidelity passed 1/1 | PASS |
| AMD/MI355X runtime | Real ROCm profiling/writer/benchmark evidence | No AMD/MI355X host is available | SKIP: AMD/MI355X hardware unavailable |

## Limits and follow-up

The hybrid test uses real GDN training, artifact loading, and attention dispatch. It supplies deterministic CPU hooks for unrelated FFN and communication operators because this worktree has no complete model-specific standard attention/MoE profiling CSVs for the eight-layer synthetic model. This proves control flow, identity, phase selection, and stage aggregation; it does not prove MI355X latency accuracy, benchmark parity, groundtruth parity, or a full production simulator request/metrics run for the hybrid fixture. That production-data run remains open for the next 14A/B iteration.
