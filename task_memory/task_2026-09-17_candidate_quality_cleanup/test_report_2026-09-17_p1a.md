## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded construction/snapshot cleanup validation. |

# P1a validation

Environment: Python 3.12.3, isolated uv venv `/data/ycfeng/tmp/quality-review-env` (no conda); CPU. Baseline `c288a19f59bec09529ee18d782fa57218da2c781` in `../quality-baseline-c288a19f`.

Commands from the corresponding worktree roots:

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-pre-nondummy
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/tmp/quality-review-env/bin/python -m pytest tests/unit/test_stage_execution_time.py tests/unit/test_stage_finalized_contract.py tests/unit/test_execution_time_op_times.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/quality-p1a
```

Criteria: preserve predictor arguments, isolated snapshots, complete/unique IDs and eight independent ownership/numeric oracles. Compare with `tests.integration.test_pr33_nondummy_acceptance.compare_baseline`, pairing `test_nondummy_simulator_accept{0..7}/run` beneath the two basetemp directories.

Observed: baseline **8 passed in 65.34s**; candidate **78 passed in 71.00s**. Artifact counts by case: 8,12,15,8,9,13,16,9; **90/90 PASS** under existing tolerances. This establishes refactor preservation, not hardware prediction error or native parity. Logs: `/data/ycfeng/tmp/quality-pre-nondummy.log`, `/data/ycfeng/tmp/quality-p1a.log`.

Full-unit collection separately encountered missing torch/matplotlib and is not included in the PASS claim. Symmetric file-set and supplemental summary comparisons are recorded below when completed.
