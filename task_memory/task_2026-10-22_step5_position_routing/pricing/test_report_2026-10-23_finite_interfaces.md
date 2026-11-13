# Finite interfaces validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record second interface verification. |

Working directory: step5-position-pricing. Python frontier-py310 and
asyncopd-py312; PYTHONDONTWRITEBYTECODE=1, OMP/OPENBLAS/MKL_NUM_THREADS=1,
TMPDIR=/data/ycfeng/tmp, pytest no:cacheprovider.

Command: `/data/ycfeng/envs/frontier-py310/bin/python -m pytest
tests/unit/test_step5_moe_pricing.py tests/unit/test_step5_comm_transport.py
tests/unit/test_step5_comm_cost_export.py tests/unit/test_collective_benchmark_helpers.py
tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_package.py
tests/unit/test_comm_operator_families.py tests/unit/test_step5_contract_reuse.py
-q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_pricing_pytest5`.

Expected: exact plan/subset validation, no scalar fallback or CPU fixture
production admission, original finite formula and real HTSIM completion.
Actual: **PASS200, SKIP1**,23.60s. Log:
`/data/ycfeng/tmp/d29_pricing_complete_tests.log`. Skip is torch absent in py310.

Separate asyncopd-py312 AST extraction executed the unchanged native modular
_fused_experts, NoDPEP.finalize and hybrid selector methods with CPU tensors
through NativeRoutedExperts.apply_prepared/prepared_context. Expected order:
inclusive expert then weighted finalization, never full apply/source prepare.
Actual PASS; all22 local experts retained and17 gathered rows, E8M0 OR shape
selection observed. Native source input:
`FRONTIER_STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`.
Log `/data/ycfeng/tmp/d29_pricing_native_tests.log`. GPU leaf kernels are not run.

This is CPU structure/model evidence, not GPU coefficient calibration, native
installed source/image qualification or E2E error acceptance. Root owns those.
