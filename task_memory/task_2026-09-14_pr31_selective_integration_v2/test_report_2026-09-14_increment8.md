## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded Increment 8 GDN training, predictor, CLI, and artifact-loading verification. |

# Increment 8 Test Report — Standard GDN Training and Prediction

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Candidate source before this increment: `5764daf7`
- Python: `3.12.3`
- NumPy: `2.4.6`
- pandas: `3.0.3`
- scikit-learn: `1.9.0`
- Fixture: `tests/fixtures/pr31_hybrid/gdn.csv`
- Persistent temporary artifact directory: `/data/ycfeng/tmp/pr31-inc8-gdn-final-20260914`

Fresh checks:

```bash
python -m compileall -q frontier tests
git diff --check
python -m pytest \
  tests/unit/test_gdn_training_predictor_increment8.py \
  tests/unit/test_gdn_semantic_core.py \
  tests/unit/test_gdn_runtime_guards.py \
  tests/unit/test_gdn_increment6_memory.py \
  -q -p no:cacheprovider
```

The focused command completed with **33 passed in 4.85s**. Compilation and
whitespace checks passed.

The actual CLI path was exercised with:

```bash
python -m frontier.training.cli gdn \
  --dataset_path tests/fixtures/pr31_hybrid/gdn.csv \
  --output_dir /data/ycfeng/tmp/pr31-inc8-gdn-final-20260914 \
  --measurement_type DEVICE_EVENT \
  --model_architecture_profile qwen3_5_moe \
  --quant_signature none \
  --device cpu \
  --tensor_parallel_size 1 \
  --runtime_stack_signature synthetic_cpu_v1
```

`python -m frontier.training.cli gdn --help` completed successfully and
listed the direct `gdn` parser with `DEVICE_EVENT`, runtime-stack, identity,
and deterministic estimator options.

## Criteria and evidence

| Criterion | Evidence | Result |
| --- | --- | --- |
| One shared physical feature contract | `GDNBatchFeatures` is used by both `GDNTrainer` and `GDNPredictor`; features are `batch_size`, `batch_num_tokens`, `max_query_len`, `query_len_cv`, and `num_stateful_requests`. | PASS |
| Pure phase admission | Same-batch prefill plus decode raises a clear `ValueError`; decode does not use request history/context length. | PASS |
| Identity filtering | Trainer filters measurement family, device, profile, quant signature, TP, and runtime stack; runtime backend, rank aggregation, layout, dtype, and GDN dimensions must each be unique. | PASS |
| Six phase-qualified tasks | The fixture run writes six estimators: `gdn_input_projections_prefill`, `gdn_core_prefill_prefill`, `gdn_output_projection_prefill`, `gdn_input_projections_decode`, `gdn_core_decode_decode`, and `gdn_output_projection_decode`. | PASS |
| Diagnostic target separation | `gdn_layer_e2e` is present only as a possible raw diagnostic column and is not referenced by trainer or predictor fitting. | PASS |
| Artifact manifest | `/data/ycfeng/tmp/pr31-inc8-gdn-final-20260914/gdn_manifest.json` records schema version 1, full identity, task mapping, feature names, and target columns. | PASS |
| Exact-row lookup | Prefill fixture prediction returned `0.11`, `0.22`, `0.33` ms for input/core/output. Decode returned `0.04`, `0.05`, `0.06` ms. | PASS |
| Structured attention result | `AttentionTime.operator_times` contains the active phase-specific GDN core and both projections; prefill total for the fixture was `0.67` ms with `norm_time_ms=0.01`. | PASS |
| Identity mismatch rejection | Loading the same artifacts with `CUDA_EVENT` or TP=2 raises `GDN model identity mismatch`. | PASS |
| Extrapolation policy | A feature vector outside profiled bounds logs `outside profiled feature bounds` and uses the estimator without clipping or TP scaling. | PASS |
| Simulator manager seam | A fresh `ExecutionTimePredictionModelManager` load test resolved the fixture artifact and preserved `runtime_stack_signature=synthetic_cpu_v1`; no fit call is made by `GDNPredictor`. | PASS |
| Public config surface | `BaseExecutionTimePredictorConfig.gdn_input_file` resolves to `{DEVICE}/{MODEL}/gdn.csv`; no `gdn_kernel_only_input_file` was added. | PASS |

## Scope boundary

This increment verifies the CPU-safe CSV → trainer → artifact → predictor
path. The full hybrid simulator path still needs Increment 14A/B: full-attention
artifact loading, global layer-spec dispatch, one real-layer `ExecutionTime`
per layer, ordered `StageExecutionTime`, and request/metrics completion. The
current manager intentionally skips homogeneous attention training for a model
that contains GDN, so this increment does not claim a complete hybrid model
E2E run.

`SKIP: AMD/MI355X hardware unavailable`. No AMD `DEVICE_EVENT` writer,
ROCm kernel execution, benchmark parity, or groundtruth parity was run or
claimed. The synthetic `DEVICE_EVENT` fixture validates schema and identity
plumbing only.
