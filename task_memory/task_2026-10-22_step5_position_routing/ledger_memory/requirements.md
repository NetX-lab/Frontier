# D29 EP Lane Ledger Memory Requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Reallocate two unused writer lines to the existing snapshot caller; close the active repair for handoff without claiming final workflow admission. |
| 2026-10-25 | Record the delegated owner's repair/cap boundary; no authority or acceptance change. |
| 2026-10-24 | Allocate bounded complete-output retention and reader repair under standing path/performance authority. |

## Execution decision and authority

Continue the approved D29 full-output workflow. Root requirements.md119–134
quotes YC's **“批准，类似问题都授权”** and permits necessary repair/test-path
adjustments within unchanged owner/global budgets. The analogous performance
authority is recorded at354–360. No new human answer is invented here.

Root allocates `repairs/ep_lane_ledger_scope_amendment.json`: four owned paths,
135 gross maximum, all CORE. Two existing previously unallocated files have
identical original/current blobs and zero cumulative gross from1aee0af to
b0277287. Effective inventory becomes116; all owner/source/global ceilings stay
4404/3956/1582,9942/10000. Donor allowance changes are zero-sum; donor code is
untouched. Accepted production remains9566source/9580global before integration.

## Required behavior

- Preserve every successful EP lane ledger value, proof, key, row order and
  existing JSONL serialization. Do not reduce, summarize, compress or drop output.
- Avoid retaining prior EP lane row objects in the real MetricsStore. Preserve
  cumulative `len` needed by current reporting; adapt the owned performance reader
  rather than retaining rows to preserve its old slice implementation.
- Read the actual-main EP ledger incrementally and keep all source, conservation,
  operator, compute-proof, collective-proof and dummy-owner checks.
- Record error timing honestly: syntax/source errors may occur at their row after
  earlier semantic checks; partial files may exist on failure, and native I/O
  failures may occur during simulation. Keep existing exception classes/messages;
  no exception suppression, fallback, retry or success inference from partial data.
- Zero-row runs must not invent an EP ledger. Final output must close/flush the
  writer without truncating previously written rows.
- No new runtime flags, numerical/proof cache, input/schema change, native backend
  change, source mount, worker, GPU or required-shape invocation.

This is a complete-output capability repair, not a measured RSS bound or a fix
for the separate2400-second CLI timeout. Stage ledger/event-trace memory, total
write volume, current cohesive timing and worker readiness remain unqualified.

## [Original Request] Delegated owner

Task ID: `D29-EP-LANE-LEDGER-IMPLEMENTATION`.

> Goal: eliminate historical EP row retention in the real MetricsStore and whole-EP-file materialization in actual-main validate(), preserving complete successful JSONL values/proofs/keys/order and every existing actual-point/row check.

> If the precise allocated path cap cannot be met, return the concrete minimum delta to primary for zero-sum reallocation, do not silently exceed it.

The owner must not delegate further, edit other code/test paths, expand the fixed
135-gross step or owner/global caps, run a fresh pricing/required-shape/fullCLI
scenario, or perform GPU/worker/StepMind/auth/mount/cloud/publishing operations.
Primary retains independent clean composed accounting, Standards/Spec review,
final-source freeze/full regressions and integration. Synthetic controls do not
admit native/E2E/runtime behavior. A source component commit does not override a
reported incomplete caller or setup prerequisite.

## Caller correction and handoff boundary

The initial component is committed as6c2f179c, with records66e50116; its complete
integration remains rawFAIL because the isolated checkout lacks collective-sim.
The open writer also raises PicklingError in the existing performance save_state.
Transfer two unused metrics allowance lines,38->36, to performance,12->14.
The four-file135 total, cumulative baseline and donor/owner/global budgets stay
unchanged. Bind the writer using the existing persistent-reference mechanism and
verify restored count, fresh output path, complete new rows and incremental digest.

YC requests code/document commits and a full continuation prompt on2026-10-25.
Commit the repaired candidate and consolidate its records for handoff. If final
frozen regressions and independent acceptance are unfinished, retain the candidate
as unadmitted rather than merging it into accepted production or relabeling old
regression evidence as fresh. No production-wave or worker invocation is added.
