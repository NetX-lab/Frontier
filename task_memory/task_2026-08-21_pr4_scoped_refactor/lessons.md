## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-26 | Created the reusable lessons record for the typed-lane and MTP token-ledger work. |

# Lessons

## Physical versus scheduler domains

`Batch.num_tokens` and `Batch.total_num_tokens` describe the physical
pre-routing forward width after batch formation. `planned_draft_tokens` controls
admission and outcome accounting, while `router_topk` expands only the routed
assignment domain. A scheduler frontier can be a distinct one-time projection
without becoming a second compute ledger.

## Typed lane ownership

An expert-token map does not identify a physical EP lane. The topology-derived
local width and owned expert interval belong in an immutable descriptor. Maps
remain useful for logs and compatibility output, but rebuilding a descriptor
from a map at a downstream consumer loses the ownership and barrier contract.

## Zero-load participants

A zero-routed lane remains a physical barrier participant. Validate its
descriptor and topology first, then return zero for routed-dependent model
lookups. This preserves collective shape and avoids querying a positive-load
profiling row for an empty lane.
