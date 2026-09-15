## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded Increment 12 shared MoE routing-helper implementation and verification. |

# Increment 12 Verification Report

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Candidate source revision before this increment: `0d1b87a4`
- Python: `3.12.3`
- NumPy: `2.4.6`
- AMD/MI355X availability: `SKIP: AMD/MI355X hardware unavailable`
- No ROCm profiling, SGLang router observation, benchmark comparison, or groundtruth parity was attempted.

## Change under test

`generate_moe_routing_ratios()` in `frontier/moe_ep_workload.py` is now the shared pure ratio generator for the standard MoE predictor and the disaggregation predictor. It preserves the previous `default_rng(seed + layer_id)` construction, ascending expert IDs, four supported distributions, normalization, and downstream integerization in `materialize_layer_ep_workload()`.

## Criteria and evidence

| Criterion | Command or check | Result |
| --- | --- | --- |
| Helper distribution contract | `python -m pytest tests/unit/test_moe_shared_routing_helper.py -q -p no:cacheprovider` | **PASS: 9 passed** |
| Existing MoE predictor semantics | `python -m pytest tests/unit/test_moe_predictor_layer_id_semantics.py tests/unit/test_moe_ep_aggregate_admission.py -q -p no:cacheprovider` | **PASS: 62 passed** |
| Disaggregation and workload conservation regression | `python -m pytest tests/unit/test_sklearn_disaggregation_execution_time_predictor.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_ep_workload_materializer.py tests/unit/test_moe_shared_routing_helper.py -q -p no:cacheprovider` | **PASS: 94 passed** |
| Old formula preservation | One-off exact dictionary comparison over 48 cases: four distributions × three expert counts × two seeds × two layer IDs | **PASS: exact equality in all 48 cases** |
| Syntax and whitespace | `python -m compileall -q frontier tests`; `git diff --check` | **PASS** |
| Baseline replay collection behavior | Same test at candidate and detached `0d1b87a4` baseline | **FAIL at collection on both revisions with the same import-cycle error** |

The baseline replay collection failure is:

```text
ImportError: cannot import name 'get_operator_family' from partially initialized module 'frontier.operators.families'
ValueError: model architecture profile 'step3_text' references unknown operator family
```

The call chain begins in `tests/e2e/moe_ep_non_dummy_matrix.py`, imports `frontier.operators.families`, and re-enters `frontier.config` while `frontier.operators.families` is still initializing. Because the detached Increment 6 revision reproduces the same failure, this is recorded as a pre-existing collection/environment issue and is outside the routing-helper scope. No fallback or unrelated import refactor was added.

## Result

Increment 12 meets its CPU acceptance criteria. The implementation is ready to commit and the next planned stage is Increment 8 (`GDNTrainer` + `GDNPredictor`).
