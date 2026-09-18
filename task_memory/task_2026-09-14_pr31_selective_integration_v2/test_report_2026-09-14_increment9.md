# Increment 9 verification report

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: `3.12.3`; CPU environment; vLLM, ROCm, and AITER are unavailable.
- Command:

  ```bash
  python -m pytest \
    tests/unit/test_vllm_rocm_attention_wrapper_increment9.py \
    tests/unit/test_profiling_accelerator.py \
    tests/unit/test_profiling_confirmation_attention.py \
    tests/unit/test_linear_op_profiling_output_metadata.py \
    tests/unit/test_gdn_profiler_cpu_increment7.py \
    tests/unit/test_gdn_training_predictor_increment8.py \
    tests/unit/test_attention_predictor_correctness.py \
    tests/unit/test_moe_profiling_output_metadata.py -q -p no:cacheprovider
  python -m compileall -q frontier tests
  git diff --check
  ```

## Criteria

- VLLM_ROCM is an explicit backend choice and does not get selected from GPU SKU metadata.
- CPU sequence metadata maps prefill/decode queries and slots deterministically, and standard mixed execution fails before a row is emitted.
- Current vLLM context/signature compatibility is isolated in a small helper; normal imports do not require vLLM.
- Generic linear/MoE launchers use shared ROCm/CUDA visibility discovery while CUDA_EVENT behavior remains unchanged.

## Evidence

- **PASS** — focused suite: `51 passed`.
- **PASS** — compilation and diff checks.
- **PASS** — CPU import of RMSNorm/RoPE compatibility modules without vLLM.
- **FAIL (baseline-only)** — `tests/unit/test_examples_profiling_contracts.py::test_profiling_readme_documents_migration_scope_and_legacy_path`; stale release README modification-history contract already present in the baseline report and unrelated to Increment 9 source behavior.
- **SKIP: AMD/MI355X hardware unavailable** — native vLLM ROCm kernel execution, AITER path, DEVICE_EVENT writer output, and GPU mixed/fresh/continuation/decode runtime checks were not run.
