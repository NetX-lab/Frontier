## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Implemented and verified shared-manager runtime selection and model identity under approved D012. |

# D012 shared-manager implementation

Owner: `/root/cpu_frontier_prepare`. Owned production file: `frontier/execution_time_predictor/shared_prediction_model_manager.py`. Owned test: `tests/unit/test_moe_routing_runtime_model_sharing.py`. Parent owns config, resolver, predictor and their tests. No GPU commands or numerical case executed here. No commit before parent integration.

## Delivered behavior

Balanced expert loads can select the separately configured uniform_topk runtime through both dataset validation and shared training. FFN dedup includes the resolved runtime, so otherwise identical cluster configurations do not skip required alternate routing models. Explicit standard and the unchanged empty/default setting still share.

The existing model registries use `(model_name, selected_layer_identity, routing_runtime_path)` for typed or runtime-qualified models. Ordinary untyped models without a runtime retain their existing legacy registry. No parallel registry was introduced and no physical layer contract was changed. Routing runtime metadata is derived from the exactly selected training dataframe, retained in fresh estimator pickles, and restored/validated before cache-hit registration. Normal routing_topk and its prefill_hot pseudo-model follow the same mechanism. Conflicting metadata fails clearly.

Cluster projection resolves the requested runtime from ReplicaConfig. Generic lookup accepts an optional runtime and refuses an ambiguous unqualified lookup. Precision and measurement-family buckets remain separate. The existing disk hash already includes the selected dataframe; no additional hash was added.

## Large-module cleanup and split boundary

The manager exceeds 2,000 lines before this work. Inspected the directly affected runtime selection and registry paths first. Removed the redundant runtime resolution after validation by sharing the earlier selection with FFN dedup. Consolidated duplicate typed/legacy lookup and ambiguity branches into one candidate-filtering path, reducing that method while adding the runtime dimension. No unrelated dead-code removal or broad refactor was made.

A future functional split should move semantic model registry storage/projection (`_contract_*`, `_legacy_*`, `_store_model_precision`, `_get_family_model`, `_models_view_for_family`) into `execution_time_predictor/model_registry.py`; manager retains cluster planning and training orchestration, while existing cache_io retains persistence. That split is larger than this approved correction because method fixtures, typed-contract resolution callbacks, and public precision/cluster projections need coordinated migration. This change keeps one existing module boundary and tests it directly; it introduces no new module to evade the size rule.

## Execution and validation

Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907`. Conda environment: `dev-vidur-v03-hopper-e2e`. Exact Python: `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, Python3.13.13, sklearn1.9.0.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python -m pytest tests/unit/test_moe_routing_runtime_model_sharing.py tests/unit/test_profiling_governance_minimal_red.py -k 'runtime_variants or shared_training_separates or runtime_survives or routing_training_rejects or dataset_validation_requires_explicit or manager_typed_registry or manager_projection or manager_keeps_precision or manager_keeps_eager or train_single_model or manager_cache_hash or manager_cache_marker' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/issue26-d012-sharing-03
```

PASS: **20 passed, 43 deselected in 2.40s**. Eleven new test cases exercise native tiny RandomForest training, two same-contract clusters with distinct routing implementations, unchanged/equivalent runtime reuse, both topk contexts, exact selected runtime rows, fresh and disk-cache-hit registration, typed and untyped model storage, BF16/FP8 plus eager/kernel-only isolation, and rejection of mixed routing runtime rows. Nine existing registry/cache tests preserve prior identity and cache behavior.

Synthetic standard/uniform routing target values are10ms and110ms, deliberately distinct to expose accidental model reuse. Retrieved predictions equal the respective10/110values. These are test sentinels, not H200 measurements or calibration results.

Initial iterations:

1. First run:7passed/1failed. The synthetic prefill_hot fixture omitted required gating_runtime_context_impl, so the production admission correctly excluded it. Fixed the test by using the existing canonical get_moe_gating_runtime_context_metadata helper; production filtering remained unchanged.
2. Second run:17passed/1failed/42deselected. A broad `-k runtime` expression accidentally selected an unrelated GPU MoE wrapper test, which failed `ModuleNotFoundError: No module named 'torch'` in the CPU-only environment. Replaced the selector with explicit relevant names; no package install, skip marker, or production fallback was added. The wrapper module is outside the modified files and this validation scope.

After the passing run, formatting and type annotations only were adjusted in the new manager helper/condition. Parent integration tests remain pending. Actual CPU-master fresh uniform simulation is a separate acceptance step.
