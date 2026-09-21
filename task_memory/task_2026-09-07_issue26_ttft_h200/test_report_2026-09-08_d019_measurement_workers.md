## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Added explicit low-density compute modes and bracketed batch controls. |

# D019 measurement worker preparation

Execution: from the active worktree, `bash -n tests/e2e/issue26_h200_diagnostics_worker.sh tests/e2e/issue26_h200_compute_suite.sh`; Python AST parsing with feature_version=(3,10) for tests/performance/issue26_attention_exact_profile.py.

Criteria: valid shell/target-Python grammar; three separate allowlists, one selected formal batch per DP, immutable diagnostic commit supplied by launcher; original400request control preserved.

Evidence: PASS syntax and source checks. Runtime and numerical validation PENDING H200. The attention helper calls the existing wrapper only at explicit prefill4096/KV0/TP4 and preserves repeated raw rows. No timing correction is established by these checks.
