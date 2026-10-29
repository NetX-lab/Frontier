# 26: H800 primary matrix and independent holdouts

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | RETIRED by decision D12 (calibration scope reduction); see the retirement note below. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Created the approved frozen H800 matrix and holdout ticket. |

**What to build:** Evaluate the frozen H80064 primary matrix and independent holdouts with the frozen predictor/statistical fits, delivering per-cell predictive online E2E verdicts.
Preserve independence, complete request accounting, and immutable evidence when a cell or holdout fails.

**Blocked by:** `25` — Representative revalidation and fit freeze.

**External prerequisites:** Complete 25 and frozen qualified reference/campaign/predictor/independent-holdout identities are required. Existing resource authority does not permit weakening frozen gates or reusing consumed holdouts as unseen data. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.
A failed holdout requires an evidence-based scoped correction and new reviewed, frozen revalidation generation, not silent refitting of the old held-out data.

**Status:** RETIRED 2026-10-21 (decision D12)

This phase is documentation review only. In the next execution session, routine in-scope implementation, instrumentation, CPU/GPU tests and measurements use the standing authorization in `../requirements.md`; evidence prerequisites still apply.

**Source workstreams:** S09.
**Shared owners:** C06 and the existing campaign, native driver, and predictive gate owners.

## Review disposition — 2026-10-18

NOT STARTED, and now RETIRED (2026-10-21): the frozen per-cell H80064 gates and independent holdouts below are no longer requirements. They are preserved as history only; the canonical 64-H200 case in tickets `23` and `28` carries the remaining evidence obligation. Preserve failures and consumed holdout membership; a later correction requires a new independently evaluated generation.

## Acceptance

- [ ] Every frozen primary cell and independent length/arrival/graph-edge holdout is evaluated with identified source, bundle, workload, and native reference; formal primary cells retain at least three clean repeats.
- [ ] Frontier independently forms queues, batches, routing workload, and completion times from preregistered inputs; compared-run batch composition, exact routing, timing, and future holdout observations never drive prediction.
- [ ] Each cell independently passes mean client TTFT, TPOT, E2E, output-only throughput, and applicable internal metrics at <=10%, plus eligible p95 client TTFT/ITL at <=15%; no aggregate hides a failing cell.
- [ ] Actual/predicted values, absolute errors, and relative errors are reported; actual zero remains undefined for relative error and uses only an applicable preregistered absolute budget.
- [ ] Required requests, tokens, ranks, and pairs reconcile through completion; overload, unfinished requests, insufficient samples, unstable repeats, and ineligible p95 remain explicit and cannot pass through omission.
- [ ] Predictive online E2E, trace-conditioned cost replay, controlled workflow replay, and fixture replay retain distinct claims; boundary tests or replay accuracy do not replace independent E2E holdouts.
- [ ] A failed holdout preserves its old evidence and prevents this acceptance from closing; any scoped fix creates a new generation with disclosed data membership and newly independent evaluation rather than relabeling consumed holdouts.
- [ ] H800 closure identifies each passing primary and holdout result and its limitations; it is a phase result and does not waive the required H200 work.
- [ ] Authorized StepMind authentication, intranet access, and RJob operations use the original personal local Python RJobBackend; ordinary local preparation, CPU checks, and public documentation reads may use local tools. At >=32 GPUs auxiliary validator errors are logged, marked unverified, and notified without stopping healthy work solely for harness failure. Native failures retain normal handling, and final verdicts require valid records.

Specification references: §§7–10.

## Retirement note — 2026-10-21

The H800 primary matrix and its independent length/arrival/graph-edge holdouts are removed from scope along with the W1–W6 × three-load campaign. The canonical case replaces the matrix. Do not resurrect cells, holdout families or QPS sweeps from this file; the only permitted load adjustment is the single justified one recorded in ticket `23`.
