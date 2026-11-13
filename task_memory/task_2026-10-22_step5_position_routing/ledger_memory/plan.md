# D29 EP Lane Ledger Memory Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Reallocate writer/performance caps to36/14; finish bounded caller verification and prepare an explicitly unadmitted handoff candidate. |
| 2026-10-25 | Own the existing performance save-state dependency and stop broad setup reruns pending primary allocation/readiness. |
| 2026-10-24 | Allocate four-path135-gross repair and define scoped implementation, correctness and freeze gates. |

## Scope and dependencies

`baseline + allocation -> RED controls -> write/read repair -> focused parity ->
clean accounting + independent review -> final frozen regressions -> integration`.
Copy/initialization diagnoses are independent; worker/fullCLI launch remains
dependent on its own source, resource, cloud-path and runtime readiness.

| Owned path | Step gross cap | Cumulative cap |
| --- | ---: | ---: |
| frontier/metrics/metrics_store.py | 36 | 36 |
| tests/integration/run_step5_measured_main.py | 20 | 67 |
| tests/integration/test_step5_measured_main.py | 65 | 325 |
| tests/performance/step5_actual06_iteration.py | 14 | 14 |

Baseline is1aee0af, accepted sourceb0277287. New paths have zero existing
cumulative delta. Amendment donors are frozen and unedited; no other source/test
path is authorized by this step. Stop for root reallocation if an exact cap cannot
be met; do not hide source changes in task records or preserve history through a
fallback/list proxy.

## Acceptance and verification

The existing owned performance caller also serializes MetricsStore in save_state.
An open TextIO writer is not serializable by its current CloudPickler. This is a
caller blocker, not permission for a new serialization/public API or list proxy.
The prior12 cap blocked completion; primary's886945f3 now authorizes14 through
the same-owner zero-sum transfer. The new worktree's empty
collective-sim path blocks four constructor tests; retain raw failures and let
primary settle checkout readiness without donor/backend edits. Do not rerun the
whole integration file merely to recheck those four known setup failures.

- Watch both real retention and actual validator materialization controls go RED
  before the implementation, then GREEN afterwards.
- Reuse typed CPU fixture/reporting seams: no new model/CSV/trace scan, pricing
  wave, required-shape call, full CLI or worker allocation in the implementation.
- Prove full successful JSONL bytes and ordering, cumulative row count, zero-row
  behavior and failure semantics; exercise missing/empty/malformed/source-less,
  null and invalid-proof rows without suppressing existing exceptions.
- Run the affected integration test file, existing EP trace/reporting suites and
  necessary local supplements with raw baseline FAIL/SKIP states preserved.
- Verify caller adaptation and exact cumulative135/owner/global/path budgets on
  clean committed source. Independent Standards/Spec review precedes integration.
- Freeze the final successor and rerun required strict77/full-unit/supplement
  gates; preserve raw baseline failures. Record observations, exact commands,
  configured versus measured resource controls and limitations in the test report.

Default CPU verification envelope:60s per focused command, configured30GiB
systemd scope, Python3.10.6, single BLAS/OpenMP threads, /data/ycfeng/tmp caches.
Retain raw failures. If the configured cap cannot start, do not silently run
uncapped commands or add retries/limits to pricing invocation ceilings.

## Current handoff work

`two-line zero-sum reallocation -> existing persistent-reference caller repair ->
bounded RED/GREEN + focused checkout verification -> code/record commits ->
complete handoff prompt`.
Final clean composed audit, independent review and frozen regression gates remain
mandatory before accepted-production integration. They may remain next-session
work at the user's handoff boundary; report their exact state and source target.
