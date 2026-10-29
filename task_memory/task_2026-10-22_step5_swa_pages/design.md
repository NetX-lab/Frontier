# Step transfer cap and live pages

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record the approved live-window boundary and source proof. |

The native exporter keeps a conservative window cap. Captured canonical and
boundary lists contain33 SWA entries, the first null; actual descriptors/bytes
contain32. Native end-trimming limits source entries to D's receiving list.
The Step resolver currently treats the cap as the payload count.

Keep `AttentionKVCacheGroup.transfer_page_count` as the existing conservative
cap, clarify its documentation, and trim inside the Step resolver using the
half-open retained token interval `[max(0,N-window),N)`. The number of touched
pages is `ceil(N/block)-floor(max(0,N-window)/block)`. Intersect this with the
existing native clip cap. Full-history groups and all other resolvers retain
their current behavior. This gains exact wire bytes/descriptors using existing
metadata; no other path presently distinguishes cap from live KV pages.

Source proof: main actual-main prefill_census/report.md and
swa_parent_rows02.json, locked native base_scheduler153-161/264-305 and
base_worker3481-3487. D's actual block list was not captured; descriptor/byte
totals and the independently run native CPU reservation corroborate its size.

The user approved this scope as D27; no general cache-capacity redesign is part
of it. The source width is supplied by the caller, not inferred from bytes.
