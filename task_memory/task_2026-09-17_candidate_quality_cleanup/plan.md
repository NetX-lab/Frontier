## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Recorded PR33 remediation, explicit R01-A decision and final verification status. |
| 2026-09-17 | Created dependency-ordered cleanup and verification plan. |
| 2026-09-17 | Closed all phases against final correctness, artifact, inventory and isolated timing evidence. |

# Plan

Dependency: `P0 -> P1 -> P2 -> P3 -> P4 -> P5`. Independent Standards and Spec read-only reviews run alongside P0/P1; implementation stays in this worktree, in bounded verified commits.

| Phase | Scope | Acceptance | Status |
| --- | --- | --- | --- |
| P0 | Pin revisions; inventory actual diff; existing decisions, environment and fresh candidate baselines | Every changed production module categorized; reproducible comparison inputs | completed |
| P1 | Runtime invariants and control/data flow | Focused contracts, pre/post real simulation outputs, >=50-case fidelity, paired run timing | completed; final 58-case fidelity and 18-run timing verified |
| P2 | Metrics, traces, ledgers and operator ownership | Same requested artifacts/numerics/layer and lane identities; focused reporting tests | completed |
| P3 | Standard/experimental profiling orchestration | Preserved platform/path/schema/cleanup contracts; CPU profiling suite and explicit native limits | completed |
| P4 | Training/model-manager/configuration/artifacts | Same path precedence, selected identities, saved/loaded predictions and supported optionals | completed; 545-test integrated PASS |
| P5 | Full regression, inventory reconciliation, evidence archive | No new candidate failure/skip; all high-value reviewed issues resolved or explicit consequential decision | completed |

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

No unresolved blocker. P4 exposed and corrected a new test-global-state leak without changing production guards; exact commands and failure evidence are in progress and the config report. Previously reproduced main/frozen-candidate environment/legacy failures remain classified separately from new regressions. Discovery-only missing-path results did not change production behavior.

## Final acceptance

- Inventory: 106/106 changed production-area paths reviewed; 38 cleaned and 68 intentionally retained. Every disposition is linked in `inventory.md`; issue RCA, existing owners and retained optionality are in `refactoring_record.md`.
- Full unit: 3813 PASS / 18 existing FAIL / 25 existing SKIP. Exact failure nodes match both frozen candidate and main; exact skipped nodes also skip on frozen candidate. No new observed regression. See `test_report_2026-09-17_p5_final.md`.
- Reporting-enabled non-dummy: 8 PASS, 90 stable artifacts, 106-file symmetric inventory and 16 nonempty supplemental pairs match. Final fidelity: 58 PASS, 304 artifacts, zero difference in 902,628 finite numeric leaf pairs.
- Final timing: 18 successful isolated measurements, all nine pairs preserve events/completions. Median paired Simulator.run changes are -13.46% / -2.00% / -1.46% for small dense / longer dense / MoE. Three reporting-disabled dummy samples per side do not establish general/native performance. See `test_report_2026-09-17_p5_timing.md`.
- Implementation and consequential design decisions remaining: none. English completion archive: `summary.md`. Unrelated legacy failures and native-device verification limits remain explicitly recorded, not silently repaired or declared PASS.

## Review remediation — in progress

Baseline: d43ae93240444bd4eff9bd99f296d2e751370514, clean worktree.
Dependency: reproduce R01 -> resolve phase-policy decision -> R01 fix -> R02 -> R03 -> R04 -> R05 -> R06 -> R07 -> R08 -> R09 -> R10 -> C01/C02 assessment -> integrated verification -> local C03 handoff. Independent read-only diagnosis may proceed while a decision is pending; implementation remains sequential.

Each correction: reproduce the stated boundary using the real public path -> fix the root cause -> run focused regression -> record evidence -> commit. R01 requires real non-dummy Simulator concurrency; R03 requires overwrite interruption and reader interleaving; R04 requires ragged CSV round trip; R06 requires cross-campaign provenance; R07 must execute shell postflight. Use existing CPU environment and scratch root. Run the existing >=50-case fidelity matrix after runtime changes, classifying intentional GDN policy corrections separately. Initial state: all R01–R10 and C01–C03 pending; later checkpoints below record their disposition.

### R01-A accepted correction

User explicitly authorizes mixed-to-prefill prediction with a warning and documented co-location limitation. Implement only at GDN runtime feature selection; preserve scheduler behavior and real phase counts. Replace phase-pure integration acceptance with observed mixed dispatch, prefill estimator selection, warning visibility, final-prefill-one-token recognition and ownership release/reuse. Native profiling and training mixed rows remain rejected. Earlier policy question is closed.

### Review remediation checkpoint — 2026-09-17

R01–R10 implementation and focused verification: completed in individual commits. C01 and C02 completed with allocation/predictor-call evidence. The accepted R01-A approximation governs mixed batches; native producer/training phase contracts remain strict.

Current dependency: frozen source `35ac95eb` -> {full unit, nine non-dummy cases, 58-case fidelity} -> isolated paired wall-clock measurement -> C03 local PR description and final archive. The three correctness campaigns can run concurrently; timing starts only after all three exit. Production and tests remain unchanged while they run. Native hardware acceptance is not claimed. C03 publication is external and remains outside the present authorization.

### Review remediation final acceptance — completed

R01–R10 and C01/C02 are implemented and individually committed after focused checks. C03 is delivered locally in pr33_description.md; external publication is outside authorization. Final production/test source 35ac95eb: 3928 unit PASS, 18 exact existing failures and 25 exact existing skips; nine non-dummy PASS; eight-case reporting artifacts preserved; 58/58 fidelity PASS with zero numeric drift; 18 paired measurements preserve events/completions. One small-dense timing pair regresses 16.62%, retained alongside all other samples; no general speedup claim. Final report and durable evidence: test_report_2026-09-17_review_final.md and review_final_evidence.json.

Pending in-scope implementation: none. Newly discovered unresolved code regressions: none observed. Native campaigns, the documented mixed-batch approximation and related gate-variant limits remain explicit. No source changes occurred after 35ac95eb; archive edits are documentation only.
