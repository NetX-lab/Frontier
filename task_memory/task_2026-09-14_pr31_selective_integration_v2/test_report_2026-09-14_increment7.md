# Increment 7 verification report

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: `3.12.3`; CPU environment; vLLM, SGLang, and AITER are not installed.
- Commands:

  ```bash
  python -m pytest tests/unit/test_gdn_profiler_cpu_increment7.py \
    tests/unit/test_gdn_semantic_core.py \
    tests/unit/test_gdn_training_predictor_increment8.py \
    tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider
  python -m compileall -q frontier tests
  git diff --check
  python -m frontier.profiling.gdn.main --help
  ```

## Criteria

- Standard input carries explicit logical phase, query/context lengths, prefill mask, physical batch size, state provenance, and reserved state page IDs.
- A one-token continuation remains `PREFILL`; mixed prefill/decode is rejected before GPU work or CSV output.
- vLLM/Triton/PyTorch imports are lazy; ROCm standard event timing accepts `DEVICE_EVENT` and rejects `CUDA_EVENT`.
- Carried state is prepared outside timing and restored before each measured iteration in the GPU wrapper.

## Evidence

- **PASS** — focused suite: `28 passed`.
- **PASS** — compileall and `git diff --check`.
- **PASS** — CLI help imported without constructing vLLM or a GPU runtime.
- **SKIP: AMD/MI355X hardware unavailable** — real vLLM Qwen GDN construction, HIP decomposition/e2e equivalence, TP rank aggregation, carried-state continuity, and DEVICE_EVENT CSV row production were not executed.
- Synthetic BF16 weights and CPU/mock contract results are functional schema evidence only; they are not AMD timing, benchmark parity, or groundtruth parity.
