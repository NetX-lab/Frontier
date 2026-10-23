# CORE/SOURCE interface validation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record baseline and first interface checks. |

All commands run in `/data/ycfeng/Frontier/.worktrees/step5-position-core` with `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; pytest uses `-q -p no:cacheprovider` and persistent temporary roots under `/data/ycfeng/tmp`.

1. Baseline focused checks: `/data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_fixed_probe.py tests/unit/test_step5_moe_profiling.py --basetemp=/data/ycfeng/tmp/d29-core-baseline`: **45 PASS, 8 SKIP**, 5.27s. No baseline failures in this selected set; torch-dependent profiling tests skip because frontier-py310 has no torch.
2. First interface checks: same Python, `tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_conservation.py tests/unit/test_moe_routing_runtime.py tests/unit/test_step5_fixed_probe.py --basetemp=/data/ycfeng/tmp/d29-core-interfaces-01`: **53 PASS, 1 SKIP**, 4.47s. Expected: exact full/P-D/chunk counts, tail residue96, q1 residue224, masked physical/live conservation, actual immutable Batch extraction, ratio regressions, full-receipt gate bound-callable positives and wrong-root/spec/hash/missing-receipt/foreign-method failures, legacy identity omission and shared-only fingerprint. Observed as expected. Tensor test skips in this environment.
3. Initial asyncopd-py312 full interface collection: `/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_moe_position_routing.py tests/unit/test_step5_fixed_probe.py --basetemp=/data/ycfeng/tmp/d29-core-interfaces-py312`: **environment BLOCKED**, `ModuleNotFoundError: No module named 'fasteners'` while importing scheduler -> predictor. This environment has torch but lacks the simulator dependency; no dependency installation or behavior fallback. Move the scheduler import into its actual Batch test so independent tensor/count/source tests can execute without this unavailable prerequisite.
4. `git diff --check`: **PASS**, no whitespace diagnostics. Source Python AST and exact quota accounting are recorded before commit.
5. Independent actual torch tensors/count/materializer checks in asyncopd-py312: `-m pytest tests/unit/test_moe_position_routing.py -k 'not actual_batches' --basetemp=/data/ycfeng/tmp/d29-core-tensors-py312`: **4 PASS, 1 deselected**, 2.30s. Real CPU torch positions/int64/mask arrays match queries/context/padding. Scheduler-dependent check remains frontier-py310 verified.
6. Python AST parsed all 7 changed code/test paths. Gross actual diff including untracked new code: **713/4404**, PASS. Source/test file quotas individually remain within retained scope allowances.

CPU source fixtures use generated fake installed modules and explicitly synthetic image references; they prove verifier rejection/identity plumbing, not actual image/source/backend execution or launched OCI digest. Root owns actual installed-image and native/runtime/E2E qualification.
