# P Computed Prompt Target

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record P/D lifecycle seam, source geometry and module-size boundary. |

`Request.num_prefill_tokens` is the semantic prompt in both clusters. The new
`prefill_compute_target` is selected at the existing P add_request seam only for
registered measured grouped NIXL PD/PP1 without prefix caching/speculation. Step
registers its one-block replay target; DSA/dense do not. Dummy timing and legacy
structural paths keep the original full-prompt contract. Positive remote count
is required, matching the locked native truncation guard; no new short-prefix
policy is introduced.

P's scheduler token budget, peak admission, allocation and completion consume
the target. At P completion, processed positions equal the computed prefix, no
client output is credited, and P retains source leases. Transfer batches expose
the computed width. The existing source common-block API returns logical
source-backed prefix pages, not aggregate physical cache-pool blocks. Canonical
P owns383 pages per group (1532 initially allocated pool blocks); existing D
remote reservation likewise holds1532 pool blocks, then grows for the local tail.
P admission previously computed6144 and allocated1536 blocks.

Exact measured transfer coverage must agree with the selected target. On READ
completion the existing recompute cursor starts at that prefix, while semantic
processed progress is anchored to the complete client prompt. D's tail execution
is16/31/17 for6144/6143/6145 and grants the first client output; remaining outputs
use q1 forwards. Ready/source-release preflight and ordering remain unchanged.

## Module-size inspection and split analysis
Baseline `Request` is2020 lines; this change makes it2049. Inspection found no
duplicate target/completion owner or dead code in the touched seams to delete.
The29 added lines keep the same object responsible for semantic prompt length,
source completion and the existing destination recompute cursor; moving just
this scalar would require a second lifecycle object and cross-object mutation.
The scheduler module remains1801 and KV allocation1130 lines, below2000.

A future movement-only split can move thinking-round affinity, hidden/final
round timing and requeue methods together into a thinking-round mixin, retaining
Request's public method delegates and shared state. A separate request metrics
projection split can move derived timing accessors after the lifecycle boundary
is stable. Neither split is required for the approved P correction; mixing it
here would expand unrelated legacy regression scope. Parent final strict77 and
independent review remain required before integration acceptance.
