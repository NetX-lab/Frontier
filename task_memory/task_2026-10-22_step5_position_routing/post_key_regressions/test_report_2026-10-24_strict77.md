# Post-key frozen strict77 final06

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record77/77 exact fidelity,complete451-hash/519-cache audits,and frozen source bindings. |

TaskID D29-POST-KEY-FROZEN-REGRESSIONS. Source
46fe83ac55876c10f6dd8ff792705e87efd1905f,tree
bbbca98802261662e02d32a59c1a4d6ab20a12e5,submodule
3dac53c544e546574cb6e8078cd14053d0916113. Cwd
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-07`;all clean
before/after. Original baseline remains713e6a81708e87b86cf91fd89f3b1f954bdcbead,
`/data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate`;not rerun.

## Exact execution

```bash
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/run_suite.py strict77
python3 -B /data/ycfeng/tmp/d29_post_key_regressions_20261024/audit_strict.py
```

Unchanged harness payloads with explicit resource timeout:
```bash
timeout 1800s /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-07 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --label candidate --output-root /data/ycfeng/tmp/d29_workflow_fidelity_20261024_final06 --jobs 4 --clean-cache --continue-on-failure
timeout 180s /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/d29_workflow_fidelity_20261024_final06 --baseline-label /data/ycfeng/tmp/d30_workflow_fidelity_20261022/candidate --candidate-label /data/ycfeng/tmp/d29_workflow_fidelity_20261024_final06/candidate
```

Expanded prefix `sudo -n systemd-run --scope --quiet -p MemoryMax=30G runuser
-u brainpp -- env ... bwrap ...`,full argv/env/packages in
`evidence/strict77/execution.json`. The audit uses the same scope/user and a60s
cap. Source entries read-only;generated clean cache belongs to fresh scratch.
Required existing3dac binary projected read-only,not built or added to source.

Python3.10.6,numpy2.2.6,pandas2.3.3,scipy1.15.3,sklearn1.7.2,plotly7.1.0,
fasteners0.20,ddsketch3.0.1. Baseline/candidate/actual package versions equal.
PYTHONPATH=repo only,PYTHONDONTWRITEBYTECODE=1,TMPDIR=/data/ycfeng/tmp,
OMP/OPENBLAS/MKL threads1. Original cases.py/compare.py/run_matrix.py bytes and
examples remain unchanged. No definition,scenario or baseline exclusion.

## Expected versus observed

Expected77/77 equality with complete unchanged evidence. Observed **PASS**:

- Run exit0,460.640s;compare exit0,15.700s;retained auditor exit0.
- 77 unique cases each side,77 compared/identical;zero mismatches or unexplained missing cases.
- 451 artifacts independently rehashed each side;zero evidence/provenance findings.
- 519 cache files each side;no filename/inventory difference,clean population complete.
- Full scenario definitions,source provenance and exact compare evidence valid.

Candidate manifest SHA256
`3dc9d9888fdef7d5912ef4acb5be2c820ada00c700857102101aca1eba24dc20`;
results `a6d2869d43ddb5f7cc0071803430125581e1bfa887353c187d6cff606f7ac847`;
comparison `22064464739130d4a5675394559353ea716cc93d5eb966c9b6f431b13074bb28`.
Fresh final06 audit receipt SHA256
`d223ffaf54a3e9ea9227697ca64add8f0518d3c9086b73c1197bd7ecec148551`.

Retained original auditor keeps its historical rawTaskID in postrun_receipt.json;
fresh final06 receipt explicitly binds this TaskID/source/tree without changing
its predicates. Small receipts retained under evidence/strict77;raw case artifacts,
logs,manifest/results and audit output remain at the requested fresh strict root.

This is legacy strict fidelity only. Exact unit IDs and the approved snapshot
test-name replacement are separate;no full CLI/runtime/native/GPU acceptance or
aggregate overlapping-suite test total is claimed.
