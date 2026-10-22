# D29 Frontier position-routing requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Bind the design artifact to the approved parent decision and complete workflow scope. |

## [Original Request]

YC's routing request and position-cyclic choice are recorded verbatim in
`../task_2026-10-15_step5_preview_h800/requirements.md`, D29.
The parent delegated the Frontier/profiling design for that decision.

Use `(logical_token_position * top_k + slot) % num_experts`, preserve P/D and
independent-reference routing, retain physical rows and masked assignments as
distinct quantities, and derive positions from Frontier's own scheduled work.
Reuse the native router in profiling and bind a fresh policy/source identity.
The actual-main fixture is required for the complete parent task.

Current active native scope is Step P/D, PP1, TP8/DP2/EP16, without SP/MTP.
Existing other-model behavior remains covered by the legacy regression set.
Workflow precedes GPU/operator-gap alignment; native CPU-gap work remains last.

The design-only delegation permits this analysis and its records. Numerical
implementation scope, including any required communication/pricing correction,
must be settled before code changes exceeding the existing repair limits.
