# D29 EP Lane Ledger Write and Read Design

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Select the existing MetricsStore persistence seam and bounded validator iteration; declare failure-order changes. |

## Evidence and gain

`MetricsStore` owns the list and final writer. `record_ep_wave` appends complete
rows; it never mutates an earlier row. Complete workload multiplicity is
1,443,200 rows/mode. Historical serialized exposure projects approximately208GB
for this artifact alone; this is not an RSS measurement. The existing validator
also materializes all rows. Removing these two growth patterns enables complete
output without requiring a run-long in-memory ledger; no existing ledger
persistence mechanism already provides that capability.

Read-only design: D29-LEDGER-MEMORY-DESIGN.
Scope basis: D29-LEDGER-SCOPE-BASIS, ses_e91a0552fffeM34Rv3FXfozhYu. The latter
used timeout-only small Git reads after systemd authentication failed; its30GiB
cap was not applied. Primary separately rechecks both blobs and accepted totals
under a configured30GiB systemd scope before allocation acceptance.

## Write-side interface

Use a small private write-through object at the existing MetricsStore-owned
append seam, retaining no prior row objects. Its interface is append, cumulative
length and terminal close; resource ownership stays with MetricsStore. Open the
same output lazily, use the existing sorted-key JSON serialization and ensure
complete appended rows are readable by the existing measurement reader. Do not
add a general logging framework or a second retained list.

Replace the final EP rewrite with closure. List-backed SimpleNamespace trace
fixtures remain valid because the producer is unchanged. The real reporting
test's cumulative length and final file remain valid. The performance probe must
read persisted rows for its existing before/after slice digest instead of assuming
the real store holds a mutable history.

MetricsStore is already5803 lines. The touched cleanup removes the now-redundant
final EP rewrite, not unrelated metrics. The private persistence object remains
at the existing owner seam to avoid distributing path/lifetime state; no new
public interface is needed. The existing functional split in
docs/general/module-size-exceptions.md moves component/diagnostic ledgers together
later. That larger movement is outside this135-line repair and must preserve
the documented lifecycle and77-case fidelity gate before implementation.

## Read-side interface

A small iterator can perform JSON decoding and the existing source-census check
per row, then yield to the unchanged semantic-check loop. It must reject an empty
file and close the source stream. This avoids re-indenting every semantic check
and avoids a second full-file parsing pass. A later malformed line may no longer
win over an earlier semantic failure: record this accepted necessary streaming
error-order change, do not claim identical global failure priority.

## Feedback loops and limits

1. Real writer retention: prior row objects stay live today; weak-reference
   controls should become releasable while complete ordered bytes persist.
2. Actual validator materialization: a ledger-specific `read_text` fault guard
   fails today; streaming must pass using the same completed-artifact fixture.
3. Independent I/O exposure: successful output bytes must stay identical, so total
   write volume does not shrink. No throughput/RSS/whole-CLI benefit is inferred.

These controls are CPU artifact/reporting checks, not new required-shape pricing
invocations. The historical CLI timeout remains separate and uncorrected by a
memory-control PASS.
