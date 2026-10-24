# D29 EP Lane Ledger Memory Requirements

## Modification History
| Date | Change |
| --- | --- |
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
