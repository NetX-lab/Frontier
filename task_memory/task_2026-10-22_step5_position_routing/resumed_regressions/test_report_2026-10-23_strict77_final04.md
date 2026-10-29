# Frozen e41 strict77 final04

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record77/77 equality with full retained-baseline audit. |

TaskID D29-E41-FINAL-REGRESSIONS. Candidate
e41abc073500d61f224eda4fca1123e74e40544d; submodule
3dac53c544e546574cb6e8078cd14053d0916113, clean before/after. Cwd
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`.
Retained baseline source713e6a81708e87b86cf91fd89f3b1f954bdcbead,
`/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate`.

Environment: frontier-py310 Python3.10.6; numpy2.2.6, pandas2.3.3,
sklearn1.7.2, scipy1.15.3, plotly7.1.0, fasteners0.20, ddsketch3.0.1.
PYTHONPATH=repo only; PYTHONDONTWRITEBYTECODE=1;
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp; OMP/OPENBLAS/MKL threads1.
Baseline, candidate and actual `_package_versions` match exactly.

Source files and scenarios unchanged: cases.py, compare.py and run_matrix.py
byte-identical to713e6a8; examples diff empty. The original driver generates
repo/cache, so bubblewrap mounts every existing source entry read-only at the
original path while newly generated cache/outputs go to unique scratch. No
source modification, changed scenario or stale label is involved.

```bash
python3 /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/run_suite.py strict77
python3 /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/run_strict_audit.py
```

Executed payloads, with expanded mount commands retained in evidence:
```bash
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/strict77_final04 --jobs 4 --clean-cache --continue-on-failure
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/strict77_final04 --baseline-label /data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate --candidate-label /data/ycfeng/tmp/d29-e41-final-regressions-20261023-6dah8l8s/strict77_final04/candidate
```

Expected complete strict equality. Observed **77/77 identical,0 mismatches**;
run exit0/474.553s, compare exit0/2.770s, retained evidence audit exit0.
No baseline/candidate failures, missing IDs/artifacts, changed definitions,
uncompared cases, provenance errors or cache-name differences.
Both sides:77 unique clean-source/full-clean-capture records,451 artifact hashes
reverified and519 cache files checked. Total artifact hashes checked902; no
suite pass totals are summed. Cache inventory unchanged on both sides.

Comparison SHA256: `c85b00cc5d0ea382b91c789e15be08827b720fe31dc92160f101239565d62b91`.
Baseline manifest/results SHA256 retain locked e8e62786/ece79d9f values.
Candidate manifest/results:5980d9ef/301cac56 (full hashes in evidence).

Result: PASS strict77 final04 only. Scratch raw data and generated cache remain
under the unique scratch root. Retained evidence `evidence/strict77/` includes
commands, manifests, per-case source-stamped results, comparison and hash audit.
`postrun_receipt.json` preserves the original auditor's historical hard-coded
task ID unchanged; `final04_receipt.json` explicitly maps that raw artifact to
this new TaskID and e41 source. This is a fresh e41 run, not relabelled final03.
GPU/native/G3/E2E and root main06 remain separate gates.
