# D29 native-key verification plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Complete all component steps, independent audit/reviews and primary source/binding acceptance. |
| 2026-10-24 | Allocate the minimal change and separate key-only evidence from native qualification. |

## Scope and dependencies

Source/lifetime inspection -> budget overlay -> RED -> one-line implementation
-> {focused tests and native-schema controls, bounded key-only timing,
canonical208-event replay} -> {independent review, composed scope guard,
required fidelity/full-unit regression} -> primary integration.

The snapshot-copy, initialization-local and required-shape investigations have
independent ownership and are not duplicated or blocked by this change.

## Checks

1. Verify the accepted source18cc3e9e is production-equivalent to dbe19f53; record
   dirty files and exact source/submodule bindings in the isolated worktree.
2. Add only the existing native-schema test's fixed-byte key assertion and
   reproduce its failure against the unchanged implementation.
3. Replace only the artifact component; AST comparison must show the rest of
   `predict_collective_plan` and all pre-lookup checks unchanged.
4. Run the affected existing suites and native-schema cache/freshness/error
   controls. Compare complete cost fields and retain raw baseline/candidate logs.
5. In one60s/30GiB command, measure8 distinct fixture keys and64 warm lookups per
   sample, paired in one process with GC enabled. Report raw wall/CPU samples,
   shared artifact-object counts and input identity; do not extrapolate a native
   whole-wave speedup from this isolated expression.
6. Reuse the protected wave body and cached fixture wiring without a candidate
   method monkeypatch. Reconcile full output, negatives, live calls and hashes.
7. Use existing strict77/full-unit mechanisms against the last accepted source;
   preserve raw failure/skip sets and the approved older metadata-ID replacement.
8. Reconcile all114 caps and unchanged source/native bindings, then review and
   integrate. Stop dependent acceptance on any unexplained output/check drift.

All eight steps are complete. Accepted cumulative usage is9566source/9580global,
PRICING3955/3956, with zero violations. Source376983d2 is integrated as c668a938.
The independent scope receipt and primary_integration.json bind the final result;
the whole-CLI/native acceptance route remains open in the parent plan.
