# Finite review repair verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record F3/F6 RED→GREEN; F4/F5 pending. |

TaskID: D29-PRICING-FINITE-REPAIR. Branch feat/step5-position-pricing.
Parent baseline8e61c2d; isolated submodule baseline1a16426.
Workdir `/data/ycfeng/Frontier/.worktrees/step5-position-pricing`.
Environment frontier-py310 Python3.10.6, CPU-only; no new dependency installation.

All parent invocations use:
```sh
env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_step5_comm_cost_export.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f36_green --junitxml=/data/ycfeng/tmp/d29_f36_green.xml
```
GREEN53 PASS,0 FAIL/ERROR/SKIP in10.85s. Log d29_f36_green.log under tmp.
RED same command using d29_f36_red names:49 PASS/4 FAIL (exporter partial/zero
DID NOT RAISE; nan/inf reached affine checks rather than completion seam).
F6 alone before edits (same flags, tests/unit/test_collective_sim_cc_backend.py,
d29_f6_red names):4 PASS/3 FAIL; exact network/context/count ModuleNotFoundError
reproduced root cold-process report. Constructor mock suppressed normal bootstrap.

Submodule cwd `frontier/cc_backend/backends/collective-sim`:
```sh
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-position-pricing:$PWD/python PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/test_grouped_plan_input.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_f3_sub_green --junitxml=/data/ycfeng/tmp/d29_f3_sub_green.xml
```
GREEN12 PASS in1.69s; RED same d29_f3_sub_red2 names:7 PASS/5 FAIL, all truncated
partial/zero/nan/inf payloads accepted. Earlier d29_f3_sub_red omitted parent path
required by surrounding package pytest imports:12 setup errors, not bug evidence.
Shared completed_plan_network_ms rejects nonempty partial/zero/nonfinite HTSIM
before prediction or residual fitting. Valid actual HTSIM and TP-local no-network
tests pass; legacy predictor logic untouched. CPU local coefficients remain fixtures.

No GPU/platform/auth/cloud/SSH/Docker/delegation/push, peer edits or import.
Final F4/F5 counts/budget/full patch to be appended after affected validation.
