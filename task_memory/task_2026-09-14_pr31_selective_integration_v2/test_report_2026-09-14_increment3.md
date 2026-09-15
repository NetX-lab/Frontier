## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded Increment 3 DEVICE_EVENT and CUDA timing compatibility verification. |

# Increment 3 Verification — DEVICE_EVENT measurement family

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: 3.12.3
- Environment: CPU master with PyTorch 2.5.1+cu124; no ROCm/vLLM/SGLang/AITER packages.
- Command: `python -m pytest tests/unit/test_device_timer_contract.py tests/unit/test_measurement_family_selector.py tests/unit/test_profiling_output_contract.py tests/unit/test_profiling_timing_stats_contract.py tests/unit/test_profiling_accelerator.py -q -p no:cacheprovider`
- Command: `python -m pytest tests/unit/test_profiling_confirmation_attention.py tests/unit/test_attention_tp_effective_mapping.py tests/unit/test_operator_parity_op_family_coverage_oracle.py tests/unit/test_model_architecture_registry.py -q -p no:cacheprovider`
- Command: `python -m compileall -q frontier tests`

## Criteria

- `device_event` maps only to `MeasurementType.DEVICE_EVENT` and is rejected for CUDA; `cuda_event` is rejected for ROCm.
- Device-event output filenames are separate from historical CUDA-event filenames.
- CPU imports do not initialize a GPU runtime; mocked event timing preserves operation order and elapsed-time materialization.
- Existing `CudaTimer` users retain KINETO, RECORD_FUNCTION, CUDA_EVENT, and PERF_COUNTER semantics through the singleton timing store.
- Shared model-manager registries separate `device_event` from `eager` and select the family from configured device metadata rather than host discovery.
- Existing attention/operator/profile behavior remains unchanged.

## Evidence

- PASS — focused measurement/timer/manager suite: **43 passed**.
- PASS — existing rename/binding/profile regression suite: **84 passed**.
- PASS — compileall exited 0.
- PASS — mocked timer test observed `record -> body -> record`; KINETO compatibility observed `profiler_enter -> body -> profiler_exit`.
- PASS — CUDA path still returns the historical `attention.csv`/`eager` naming; DEVICE_EVENT uses `attention_device_event.csv` and a separate manager registry.
- SKIP: AMD/MI355X hardware unavailable — no real ROCm DEVICE_EVENT writer or AMD elapsed timing was executed.

The CPU/mocked checks validate schema, dispatch, path identity, and compatibility behavior. They do not establish GPU timing fidelity or ROCm runtime support.
