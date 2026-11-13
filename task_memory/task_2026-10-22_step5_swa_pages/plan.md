# D27 implementation and verification plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Define a small Step resolver correction using existing group metadata. |

Source/raw proof -> real public payload regression RED -> Step resolver fix ->
focused GREEN -> strict legacy fidelity -> integration with D26.

Use the existing group's transfer cap and actual window endpoint to trim pages
that carry no live KV. No new registry field or shared signature. Test aligned
and unaligned endpoints, short windows and canonical source6112/6128; verify
capacity stays unchanged. Preserve full-history, FP8 summary and open-region
behavior. Existing tests with33 for aligned payloads must be corrected from
native evidence, not simply deleted. Parent owns source and acceptance.

Change marker: D27_STEP_SWA_LIVE_PAGES. Native numerical agreement and final
latency accuracy remain separate; this correction concerns exact payloads.
