# Artifact bytes repair plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Define the scoped implementation and verification sequence. |

Dependency: source/evidence binding -> {baseline wave, baseline affected tests}
-> source replacement -> old-hook instrumentation observation -> hook adaptation
-> {candidate wave, candidate affected tests} -> accounting/checkpoint -> code
commit -> records commit. No delegated work or independent source writers.

1. Bind the current HEAD/source and unchanged parent layer60 setup/body; relocate
   all parser/metrics/cache writes into fresh scratch before config construction.
2. Replace only cache representations and the existing test instrumentation.
3. Verify 80 current reads/16 EP, full 208-event exact bytes/ledger/proofs/keys;
   record cold/warm pricing, both read-method counts and JSON parser counts.
4. Run the four existing affected test files and original replacement/negative
   cases. Require unchanged assertion coverage and original error identities.
5. Measure Git step gross/net and retained counts. Respect all three file caps;
   inspect a review-ready diff and run the phase progress checkpoint.
6. Commit source/test first; archive exact commands, hashes, observations and
   limits in these records, then commit records. Leave the worktree clean.

Completion is focused component acceptance, not independent review or full CLI,
regression-union, native/GPU/G3/E2E acceptance. Those remain root-owned.
