# D29 final parent audit progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete raw-ID/strict and source bindings, reproduce the original namespace, and compose parent admission without gate waiver. |
| 2026-10-24 | Resolve identities, establish bounded execution, and start independent receipt verification. |

- completed: created the new records-only branch/worktree from exact
  `86b9923f8c29f1c58ba9f8df23a396c70a4d1caa`, without overwriting existing work.
- completed: read root requirements/plan, applicable instructions, owner final
  docs, post-key parent checker, short parent checker, and unchanged comparators.
- completed: verified that `sudo -n systemd-run --scope -p MemoryMax=30G runuser
  -u brainpp` is available; inspections run as UID 10250 with Python -B. No
  RLIMIT-only substitute is needed for this task.
- completed: metadata inspection finds 2755 listed source files totaling
  368850959 bytes, 14 inputs, and 12 scripts. Current owner manifest hashes all
  match. The second owner commit changes only five scoped status/manifest docs.
- completed: bindings phase PASS in 2.838707s. Streamed2755 source files,
  14 inputs and12 scripts; exact124/120 scope bindings and9565/9579 totals;
  all protected HEAD/index states unchanged.
- completed: first strict reproduction ran outside the retained bwrap namespace
  and returned FAIL because the recorded source/cache path is intentionally absent
  on the host. Exact retained physical namespace cache has519 files, no additions
  or losses. Preserved the failed diagnostic at
  `diagnostic_outside_recorded_namespace.json`; did not edit artifacts or predicates.
- completed: reused the recorded strict bwrap namespace with only this new audit
  output subtree additionally writable. Receipts phase PASS in2.255898s;
  original auditor returns0 and produces the exact owner postrun receipt bytes.
  Independently rehashed902 artifacts and checked519 cache filenames per side.
- completed: raw unit8028/279/32/281 andexit1; all8612 preceding46fe IDs/statuses
  retained plus8 genuine passing additions. Against79,8609 common IDs unchanged,
 11newPASS and the one approved old-ID loss. Raw presence and legacy gates remainFAIL.
- completed: thirteen environment-only added outcomes join actual same-source101;
  five historical Torch failures join52 separately. No raw outcome is relabelled.
- completed: acceptance phase PASS in0.067456s;3762 unique file hashes across phases;
  all phases verified MemoryMax32212254720 bytes, UID10250, Python-B, timeout60s.
- completed: a summary-only hash lookup assumed the inventory filename was
  `source_inventory.json` and raised StopIteration. The binding phase had already
  used the authoritative path in source_final_receipt correctly. This did not
  affect any acceptance check or require a test rerun.
- completed: final report/checksum inventory and16-file record consistency checks
  PASS. Parent/owner/frozen HEAD and index hashes remain unchanged. Parent receipt
  SHA25674735eb00f1aa190472b8927ed312c11cf18921b3b05639d7f8a972f0450753b.
- pending: scoped commit verification and return; no unfinished evidence check.
