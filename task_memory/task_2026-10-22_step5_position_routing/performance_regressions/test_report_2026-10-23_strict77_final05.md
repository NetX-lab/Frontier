# Fresh79 strict77 final05

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record77/77 complete comparison and original evidence audit on frozen performance source. |

TaskID D29-PERFORMANCE-FINAL-REGRESSIONS. Candidate source
79f1a3fd35d42b751cd1520c764ddac717b923d9, tree
8c6db078e852d3b885a6f14bdf5e8007d9ecddb7; gitlink/submodule
3dac53c544e546574cb6e8078cd14053d0916113, clean before/after.
Cwd `/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06`.
Original baseline source713e6a81708e87b86cf91fd89f3b1f954bdcbead,
`/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate`.

## Exact execution

```bash
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/run_suite.py strict77
python3 /data/ycfeng/tmp/d29_performance_regressions_final05/audit_strict.py
```

Executed unchanged harness payloads:
```bash
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/d29_workflow_fidelity_20261023_final05 --jobs 4 --clean-cache --continue-on-failure
/data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/d29_workflow_fidelity_20261023_final05 --baseline-label /data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate --candidate-label /data/ycfeng/tmp/d29_workflow_fidelity_20261023_final05/candidate
```

Expanded read-only mounts, exact argv/environment/raw exits retained in
`evidence/strict77/execution.json`. Source entries read-only at original absolute
path; generated cache lives under unique scratch namespace. Both output and
namespace absent before launch; no destructive cleanup of prior artifacts.

Python3.10.6 frontier-py310; numpy2.2.6, pandas2.3.3, scipy1.15.3,
sklearn1.7.2, plotly7.1.0, fasteners0.20, ddsketch3.0.1. Baseline/candidate/actual
versions match exactly. PYTHONPATH=repo only, PYTHONDONTWRITEBYTECODE=1,
TMPDIR/FRONTIER_TMP_ROOT=/data/ycfeng/tmp, OMP/OPENBLAS/MKL threads1.
cases.py/compare.py/run_matrix.py bytes equal original713e6a8;examples diff empty.

## Expected versus observed

Expected unchanged strict definition with77/77 equality and complete independent
hash/cache/scenario/provenance evidence. Observed **PASS**:

- 77 unique cases per side,77 compared/identical;zero differences in every field.
- 451 artifact hashes per side independently verified;no evidence error.
- 519 cache files per side;no filename/inventory difference.
- Full clean-cache capture,complete scenario definitions and source provenance valid.
- Run exit0/463.004s;compare exit0/2.806s;original retained auditor exit0.

Comparison SHA256 `aae7a53b899703725805388a23e603a3b13b2d8bb89de860d320f0417873cf59`.
Fresh manifest SHA256 `f597643a6782680d8cfb1b452ce9646a1ae9b4fd76b9329d1dacbc03505cbc16`;
fresh results SHA256 `11247b368ce0cfacdc36bf7f847d7c68843302c44f5508a6a4a5127725dcf7b6`.
Complete compact audit: `evidence/strict77/final05_receipt.json`.

The unchanged original auditor retains its historical raw task ID in
`postrun_receipt.json`;fresh final05 receipt explicitly binds that raw output to
this TaskID/source79/tree8c6. No stale final04 label or outcome is reused.

Raw case artifacts,cache,manifest/results and run/compare/audit logs remain under
`/data/ycfeng/tmp/d29_workflow_fidelity_20261023_final05` and
`/data/ycfeng/tmp/d29_performance_regressions_final05/strict77/generated_repo_root`.
This verifies strict legacy fidelity only. Unit comparisons,main08 and runtime/
GPU/G3/E2E numerical qualification remain separate;no suite totals summed.
