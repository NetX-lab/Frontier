# D23-MRV2-LOAD-DIAG Summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Complete diagnosis-only handoff. |

## Outcome and deliverables
Diagnostic code commit: `c655305` on `feat/step5-mrv2-load`, based on ec5de15.
Task evidence is committed separately for optional parent integration.

The initialization path repeatedly validates/serializes the same attention input
despite warm caches. A seconds-scale profile reproduces two loader calls after
the second role's signature is already trained, with no cache read or estimator
fit at that point. Production source remains unchanged.

- `tests/performance/profile_step5_load.py`: reusable actual loader/role-union/
  warm-cache diagnostic with source-preserving prefixes and optional event root
  cache-read control. Its opt-in regression assertion intentionally returns RED.
- `task_memory/task_2026-10-22_step5_mrv2_load/test_report_2026-10-22_step5_load.md`:
  exact commands, source checks, counts/times, attribution limits and source proposal.
- Same task directory: `profile_receipt.json`, `final_command_receipt.json`,
  `event_cache_receipt.json`, task requirements/plan/progress/checkpoint.

## Observed validation
Actual config model JSON equals the diagnostic model. Prefix100/200 retain all
125 columns/source fields. Plain loader0.255748/0.453526s; cProfile
0.563309/1.033775s. Full KERNEL_ONLY184 covers92 layers and MRv2 rows. Warm
PREFILL4.942950s includes4.449957s loaders versus0.003785s cache reads.
Already-deduplicated DECODE4.681278s still loads twice. Final command repeats RED
and reads current71.15MB event root in0.529611s. Zero fits; original artifact
SHA256 and source size/mtime checks PASS. No full CLI or initialization retry.

## Limits and parent-owned work
The5.606GB event table contains53,767-character quant_signature per row. The
unchanged call chain implies four complete event loads plus four source SHA256
scans per initialization. Retained actual05 logs show no cache-miss/training
warning. This identifies pre-cache input work as a credible major cause but does
not allocate all998/1004s. Prefix-only approximately747s arithmetic is not a full
wall profile; unprofiled initialization work and concurrency remain unallocated.

Smallest proposal: manager-local reuse of identical validated union/hash with
all role/source/shape/evidence records and unchanged estimator cache identity;
feature-derivation reuse is secondary. Source implementation, review/integration
and unchanged2400s full offline/online acceptance remain parent-owned. Runtime
fixed-contract work remains owned by ses_e9f519775ffe3Cv0bhEzJSKNhH and was not
duplicated here.
