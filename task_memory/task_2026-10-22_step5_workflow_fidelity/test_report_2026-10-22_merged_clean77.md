# Test report: merged Step clean77 legacy fidelity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record exact commands, strict 77/77 PASS and 902 artifact rehash checks. |

## Scope and raw request

Task `D26-D27-D28-MERGED-FIDELITY`; [requirements.md](requirements.md) records the
parent request. Tested SHA: `fa5bae2980366da1d075125d6c26248a737fc419`.
Checkout: `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2`.
Verification branch: `verify/fidelity-candidate-fa5bae2`.

This is CPU execution of the existing legacy matrix. It is not native
CPU-overhead measurement, full Step CLI acceptance or numerical calibration.
Prior reviews and the parent's 121/191/299/81 checks are not credited here.

## Baseline identity and reuse decision

Reused label: `/data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/candidate`.
Its source and harness revision both equal
`7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb`; source/driver checkout:
`/data/ycfeng/Frontier/.worktrees/fidelity-candidate-7e8bea3`.

Observed preflight checks passed: 77 unique case records, one clean revision,
all cases executed in one clean-cache run, 77 identical case definitions and
451 artifacts rehashed against recorded sizes/SHA-256. Baseline manifest lists
519 cache filenames. Existing proof at
`/data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/comparison.json` recorded
77/77 identical cases, no failures or missing evidence, and clean/full cache
comparison against revision `26db654cbbb38b361990aa1aa8ae1dad5db8e000`.
The earlier root comparison likewise recorded 77/77 identical cases against
`0cdd6cb483faae436b50f15d88f2b529bf747dc0`.

`git diff --name-only 7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb
fa5bae2980366da1d075125d6c26248a737fc419 -- tests/e2e/refactor_fidelity examples data`
was empty: runner, comparator, matrix, wrappers, fixtures and checked-in input
data are identical. Python executable/dependency versions match the baseline
manifest. The proven baseline was not rerun or modified.

Baseline manifest SHA-256:
`95ff242692a16b9dc2e1ba49f097ab5df739b93bf77c89e44ee915fe327eae19`.
Baseline results SHA-256:
`b5b12103c3d6e174a8140b83c93486a5337b8e7da8f9be678373adfeeb88c9a6`.
Prior comparison SHA-256:
`5753dc5399065bed197e879047caae448f55c51de7a9b180c7cb53e020d29488`.

## Commands and environment

Checkout creation (cwd `/data/ycfeng/Frontier`):

```bash
git worktree add -b verify/fidelity-candidate-fa5bae2 /data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2 fa5bae2980366da1d075125d6c26248a737fc419
```

Candidate capture (cwd is the verification checkout):

```bash
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/step5_workflow_fidelity_20261022 --jobs 4 --clean-cache --continue-on-failure > /data/ycfeng/tmp/step5_workflow_fidelity_20261022/candidate_run.log 2>&1
```

Strict comparison command (same cwd; no partial waiver):

```bash
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/step5_workflow_fidelity_20261022 --baseline-label /data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/candidate --candidate-label candidate > /data/ycfeng/tmp/step5_workflow_fidelity_20261022/comparison.log 2>&1
```

Post-run read-only evidence audit (same cwd):

```bash
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/fidelity-candidate-fa5bae2 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/step5_workflow_fidelity_20261022/audit_evidence.py > /data/ycfeng/tmp/step5_workflow_fidelity_20261022/evidence_audit.log 2>&1
```

Python environment: `/data/ycfeng/envs/frontier-py310`, Python 3.10.6; numpy
2.2.6, pandas 2.3.3, sklearn 1.7.2, scipy 1.15.3, plotly 7.1.0, fasteners 0.20,
ddsketch 3.0.1. No conda activation/change was used. Host platform:
`Linux-5.10.25-nvidia-gpu-x86_64-with-glibc2.35`.

The existing runner forces `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`,
the tested checkout's `PYTHONPATH`, its selected `PYTHON_BIN`, W&B disabled,
case-specific `RUN_ID` and metrics root, plus the original case overrides.
Seven trained-predictor cases execute serially before 70 other cases run at
jobs4. Default per-case timeout is 1800 seconds. `--clean-cache` acts only on
the fresh candidate checkout. Continue-on-failure makes capture exit status
insufficient; the results and strict comparator determine acceptance.

## Expected results and comparison method

The unchanged 77-case matrix covers dense/MoE, offline/online, DP lanes, PDD,
PD-AF, runtime features and trained-predictor/cache behavior. Group counts:
trained predictor6, offline dense14, offline MoE11, online6, features8, PDD15,
PD-AF10 and DP placement7. One of the DP cases also uses a trained predictor.

Pass requires all 77 cases producing retained artifacts and actually compared;
no source, definition, failure, missing-evidence, artifact or cache-name
findings; full clean-cache comparison enabled. The existing comparator uses
exact content equality after its original run-path substitutions; no metric
tolerance, expectation change or ignored artifact is added. JSON, JSONL and CSV
content plus binary artifacts and filenames are included.

## Actual result

**PASS: strict 77/77 legacy fidelity at the exact requested merged SHA.**

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Candidate output population | 77 valid cases | 77/77, rc0/error null | PASS |
| Actual strict comparison | 77 compared/identical | 77/77 compared and identical | PASS |
| Artifact/metric mismatches | 0 | 0 | PASS |
| Baseline/candidate-only failures | 0/0 | 0/0 | PASS |
| Missing cases/evidence/definitions | 0/0/0 | 0/0/0 | PASS |
| Not compared/unexplained/absent from both | 0/0/0 | 0/0/0 | PASS |
| Source/harness provenance findings | 0 | 0; uniform clean exact-SHA source | PASS |
| Clean full-run cache comparison | enabled | both true | PASS |
| Cache filename differences | 0 baseline-only/0 candidate-only | 0/0; 519 names on each side | PASS |
| Recorded artifact size/hash integrity | all intact | 451 baseline +451 candidate =902/902 | PASS |
| Actual cache versus recorded inventory | unchanged | unchanged on both sides | PASS |
| Candidate revision/status after execution | requested SHA/clean | fa5bae2980366da1d075125d6c26248a737fc419/empty | PASS |

Capture, strict comparator and evidence audit completed successfully; the
strict comparator and audit each returned 0. No expectation, source, fixture,
case definition or tolerance was changed. The comparison reports no differing
metric value or artifact. First concrete mismatch: none.

Candidate manifest SHA-256:
`6eb2d5419a5528ccd1fdee94b4aad89235e1bf8d3a6cd222a7024731cab71055`.
Candidate results SHA-256:
`da45dd4a41a81ab01521067a8a40d3d6cc82af60ec6bf35fe1e9a1c10fd8b624`.
Strict comparison SHA-256:
`e1f4f94047f24889db7109f5de02b364e8ab033df8d36421da6978b496e4d677`.

## Evidence paths and limits

Raw output root: `/data/ycfeng/tmp/step5_workflow_fidelity_20261022`.
Preflight receipt: `preflight_receipt.json` there.
Candidate artifacts/logs: `candidate/cases/<case_id>/` there.
Manifest/results: `candidate/manifest.json` and `candidate/results.jsonl` there.
Comparison: `comparison.json` and `comparison.log` there.
Post-run receipt: `postrun_receipt.json` there; exact audit script/log:
`audit_evidence.py` and `evidence_audit.log` there.
Persistent selected manifests, logs, comparison and receipts:
`task_memory/task_2026-10-22_step5_workflow_fidelity/evidence/`.

Full Step measured CLI, D26-specific workflow acceptance, native READ/state
qualification, balanced-routing generation, GPU/operator-gap alignment and
CPU-overhead measurement remain with the parent. No source, tests or fixtures
are changed by this task; parent `.cortexkit/` and `.pi/` stay untouched.
