# Native interface CPU verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record baseline and actual-source interface validation. |

Environment: asyncopd-py312, Python3.12.13, Torch2.9.1+cu128. No CUDA execution.
All commands set PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
Python executable: /data/ycfeng/envs/asyncopd-py312/bin/python -B.

- Baseline: `python -B -m pytest tests/unit/test_step5_canonical_g3.py
  tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_reference.py
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-baseline-pytest`.
  Expected existing controls PASS; actual251 PASS/32.68s.
  Log: /data/ycfeng/tmp/d29-native-baseline.log.
- Interface: `STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm
  python -B -m pytest tests/unit/test_step5_native_position_cyclic.py -q
  -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29-native-interface-pytest03`.
  Actual9 PASS/2.33s. Log: /data/ycfeng/tmp/d29-native-interface03.log.
  Expected exact ID/weight/mask/dtype, P/D/chunks/q1, seeded legacy strategies,
  real outer-forward tensor identity/nested/exception cleanup/counter retention.
- Earlier test-fixture attempts failed: mutable SimpleNamespace mode default in
  dataclass, then missing Callable annotation dependency. Corrected dependency
  fixtures; no native behavior was changed to suppress these errors.
- AST parsed actual four modified native files. Source receipt records99 native
  gross lines; two context/router class stubs represent import dependencies only.
  Executed strategy/base selection/factory/outer-forward bodies are actual source.

Limits: CPU formula agreement is not selected-score CUDA arithmetic, graph replay,
all-rank module auto-backend, G3 numerical state or clean E2E qualification.
