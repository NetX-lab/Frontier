# D29 Required Wave Shape Plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Record primary takeover, recovered invocation census, and three bounded independent shape checks. |

## Dependency Graph

`cancelled-session census -> source/fixture preflight -> {P6128,D16,BF16} -> receipts`

Snapshot-copy and initialization analyses run independently in their own records
worktrees; neither is a dependency of these shape checks.

## Steps

1. Recover complete tool-command history for the cancelled shape session and
   confirm whether any production-path invocation or artifact write occurred.
2. Bind the retained setup/body, exact frozen source, original CLI/config and
   cached fixture metadata. Use actual Batch/forward-binding/routing/materializer
   mechanisms rather than inventing an expert count or collective cost.
3. Prepare one driver with three explicit cases. P6128 uses logical positions
   starting0; first-DECODE16 starts6128; BF16 checks actual layers88,89,90 on a
   retained ordinary-DECODE shape. Derive geometry and graph policy from config.
4. Execute each case once within60s/30GiB, with all writes in fresh local scratch.
   Inspect actual operator costs, applicable operators, source/local/gathered
   rows, native-context labels, complete keys, event/ledger identity and proofs.
5. Archive selected evidence, commands, limits, hashes, errors and case verdicts.
   Re-read requirements before closure. No production repair is assigned.

## Verification

- Assert source HEAD/clean state and source/setup/body SHA256 bindings before
  use; verify unchanged source/fixture metadata afterward.
- Record the production functions' actual code paths and count live numerical,
  proof and collective calls without replacing their behavior.
- Compare every emitted duration/proof/key to the corresponding actual lane
  result and ledger; record complete JSONL/ledger bytes and hashes.
- FP8/BF16 operator applicability comes from actual model/context selection,
  not a forced208-event assertion. BF16 must have no standalone input-quantizer.
- A missing exact count/model/shape receipt is a retained blocker, not permission
  to synthesize data, bypass admission, load fullCSV or increase resource limits.
