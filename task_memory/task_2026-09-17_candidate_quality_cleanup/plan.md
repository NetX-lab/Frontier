## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Created dependency-ordered cleanup and verification plan. |

# Plan

Dependency: `P0 -> P1 -> P2 -> P3 -> P4 -> P5`. Independent Standards and Spec read-only reviews run alongside P0/P1; implementation stays in this worktree, in bounded verified commits.

| Phase | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| P0 | Pin revisions; inventory actual diff; existing decisions, environment and fresh candidate baselines | Every changed production module categorized; reproducible comparison inputs | completed |
| P1 | Runtime invariants and control/data flow | Focused contracts, pre/post real simulation outputs, >=50-case fidelity, paired run timing | completed; final integrated fidelity still required |
| P2 | Metrics, traces, ledgers and operator ownership | Same requested artifacts/numerics/layer and lane identities; focused reporting tests | in-progress |
| P3 | Standard/experimental profiling orchestration | Preserved platform/path/schema/cleanup contracts; CPU profiling suite and explicit native limits | pending |
| P4 | Training/model-manager/configuration/artifacts | Same path precedence, selected identities, saved/loaded predictions and supported optionals | pending |
| P5 | Full regression, inventory reconciliation, evidence archive | No new candidate failure/skip; all high-value reviewed issues resolved or explicit consequential decision | pending |

## Execution sub-steps

Within each phase: inspect changed hunks and authoritative callers -> record specific invariant and bounded change -> characterize current outputs -> apply simplification and valid fixtures -> focused regression/output comparison -> commit -> next dependency.

Initial runtime ranking: timing entities and predictor stage construction; immutable attention/model binding; scheduler GDN ownership/memory; MoE routing; simulator integration. Inspect >2,000-line modules for candidate-specific removal first and record split analysis where they remain large.

## Verification strategy

- Reuse `tests/integration/test_pr33_nondummy_acceptance.py` for eight synthetic real Simulator cases and independent numeric ownership oracles.
- Reuse the established 58-scenario fidelity driver and comparator, with frozen candidate as the principal preservation reference and pinned main as a classified historical control.
- Reuse `tests/performance/measure_pr33_paired.py` / `sim_walltime_scaling/run_case.py`; compare frozen candidate with cleaned candidate using interleaved unprofiled samples while no concurrent test load runs. Keep main comparison available.
- Run targeted tests at each logical change and corresponding broader CPU suites at phase boundaries. Run full `tests/unit` before final closure; match failure nodes and causes against fresh pre-refactor candidate/main when needed.
- Use original comparator tolerances. Retain observed failures and distinguish process success, numerical parity, synthetic predictions and native hardware verification.

## Errors and blockers

None established. A discovery command returned exit 1 because no nested AGENTS.md matched; this was a file-search result, not a test failure.
