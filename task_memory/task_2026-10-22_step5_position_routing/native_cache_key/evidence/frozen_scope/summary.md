# D29 native-key scope audit summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive the independent committed two-path numerical/binding audit. |

## Outcome

PASS_FROZEN_NATIVE_KEY_SCOPE_DELTA, zero violations, on source376983d2,
treec8f8657f650e4645ec0fc0c86b84283766c09b3c. Exact2-path3gross/net1 change;
cumulative source increases only1. All17 overlays,114 allocated/113 changed paths
and124/122 binding closure reconcile. CORE4261/4404, PRICING3955/3956,
native1350/1582; source9566/9942, global9580/10000. Unchanged submodule333,
native patch99, docs12 and gitlink2 are separately counted.

## Deliverables

Owned directory:
`task_memory/task_2026-10-22_step5_position_routing/native_cache_key/evidence/frozen_scope/`.

- `recompute_frozen_scope.py`: bounded ledger/overlay/two-path source recomputation.
- `frozen_scope_summary.json`: compact parent receipt bound to detailed evidence/TSV.
- `frozen_scope_receipt.json`: exact guards,124 binding identities,114 rows,
  measured counts, authority/input hashes, commands and limits.
- `frozen_scope_paths.tsv`: all effective caps, actuals and row provenance.
- `test_report_2026-10-24_native_key_scope.md`: observed results and proof method.
- `artifact_sha256.json`: checksum inventory of the scoped records.

## Limits

Candidate and submodule clean/exact before and after; primary index unchanged.
No source/test/donor changes, suite/wave/benchmark reruns, large inputs, network,
GPU or native rebuild. This admits accounting/bindings only. Implementation
review, required regressions, native qualification and wholeCLI remain separate.
