# Pricing interface verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record initial interface tests and corrected test-field mismatch. |

Environment: /data/ycfeng/envs/frontier-py310/bin/python (Python3.10), pricing
worktree; PYTHONPATH=$PWD, PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1.

Command: `python -m pytest tests/unit/test_step5_moe_pricing.py
tests/unit/test_comm_operator_families.py tests/unit/test_step5_contract_reuse.py
-q -p no:cacheprovider`.

Expected: selected BF16/FP8 prepare targets, real ROUTED EP-agnostic binding,
FP32 gate/FP8 prepare precision, immutable plan order/size/count constraints,
unchanged scalar dispatch and cached Step binding behavior. Actual: **60 passed**
in10.81s. Raw log: /data/ycfeng/tmp/d29_pricing_interfaces_tests.log.

Initial new test:2 failed/2 passed because the test named nonexistent
ResolvedLayerContract.family_id; inspected actual field operator_family_id,
corrected assertion only. Production resolver already succeeded.

Limits: no GPU timing/calibration, native source proof, graph/full-model evidence,
or E2E qualification. No fitted coefficients fabricated.
