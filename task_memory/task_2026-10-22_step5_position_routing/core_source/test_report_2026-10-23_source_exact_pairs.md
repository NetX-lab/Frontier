# Position source exact duration/count regression

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual fitted source-scope RED/GREEN and legacy regression. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Local frontier-py310 Python3.10.6; no GPU or native timing. All commands use `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp`.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q tests/unit/test_step5_measured_ffn.py::test_position_real_estimator_reaches_execution_time_without_shuffle
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py > /data/ycfeng/tmp/d29-core-source-exact-check03.log 2>&1
```

RED: real producer rows and shared sklearn fit reached H04/ExecutionTime. Removing only norm count key `(16.,)` still returned a cost (`DID NOT RAISE ValueError`,7.70s). Existing exact-pair guard applied only to supplied MoE contexts, so source norm/shared/merge used legacy interpolation. The correction applies that same guard to the already selected position runtime without changing legacy defaults. Duration/count schema, exact key sets, requested key and target names must agree; positive duration and integral actual counts remain mandatory.

First GREEN attempt exposed an incorrect empty context passed to the strict peer selector (`KeyError: model_architecture_profile`), not a native schema defect. Pass the existing `None` contract instead; no defaults/selector changes. Final affected FFN/replay suites: **27 PASS,28.63s**. Count-key deletion now fails before returning source cost. Collective cost in this test is explicitly CPU fixture evidence; this does not settle PRICING C1/C2 or actual-main/native qualification.

Owned accounting after this correction: **44/44 changed paths,+3087/-155=3242/4404 gross**,1162 remaining. Per-file quotas and AST PASS. `test_step5_measured_ffn.py` is at its108-line allowance; no further additions there without allocation.
