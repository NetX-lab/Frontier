## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded Increment 2 GPU platform/SKU/discovery verification. |

# Increment 2 Verification — GPU platform + MI355X

## Execution

- Worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`
- Python: 3.12.3
- Environment: CPU master with PyTorch 2.5.1+cu124; `vllm`, `sglang`, and `aiter` unavailable.
- Command: `python -m pytest tests/unit/test_profiling_accelerator.py -q -p no:cacheprovider`
- Command: `python -m compileall -q frontier tests`

## Criteria

- Every known GPU SKU exposes `gpu_platform` in `{cuda, rocm}`.
- MI355X resolves to ROCM metadata; MI355X_UBB resolves to eight MI355X devices.
- Unknown SKU parsing fails before any discovery subprocess.
- Visibility and mocked `amd-smi`/`nvidia-smi` discovery are deterministic on CPU.
- A device/backend mismatch warns and preserves the selected backend.
- No test is allowed to imply an AMD runtime result.

## Evidence

- PASS — `tests/unit/test_profiling_accelerator.py`: **12 passed**.
- PASS — compileall exited 0.
- PASS — mocked discovery used `amd-smi` JSON first and fell back to `nvidia-smi`; no real GPU command was required by the tests.
- PASS — unknown `unknown_gpu` raised the existing `BaseDeviceSKUConfig` invalid-type error.
- PASS — mismatch warning text explicitly states that the user-selected backend is preserved.
- SKIP: AMD/MI355X hardware unavailable — no ROCm runtime, `amd-smi`, or MI355X profiling execution was claimed.

The passing results establish schema/config/discovery behavior only. They do not establish ROCm kernel execution, DEVICE_EVENT timing, or AMD numerical parity.
