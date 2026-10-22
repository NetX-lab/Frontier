# Pricing implementation requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record delegated scope and approved model. |

TaskID: D29-PRICING-IMPLEMENTATION.

[Original Request] Implement ../scope.json pricing-only files and H04/H17,
44 paths and at most 3956 gross code/test/config lines, including five
collective-sim files. Deliver tested operator and finite-model interfaces first.
Never edit peers or duplicate their APIs; return a concrete prerequisite boundary
if peer APIs block the next necessary step. No platform operations or delegation.

[Decision] D29-SCOPE-107-10000 approves ordered tensor/root stages, exact element
stripes, joint-eight-rail HTSIM, independently measured copy/add proxies, affine
effective remainder and no-overlap sum. Full-plan measured lookup is not selected.
See ../requirements.md, ../pricing_design.md, ../profile_source_design.md and
../shared_scope.md. Unprovided coefficients/source evidence remain unprovided.

[Original Request, integration steering] Root authorizes development merges of
CORE/SOURCE02f061d and Native2c4cc9e after verified owned edits are committed.
Use source_frame_tensors and verify_step5_profile_runtime without duplicates.
These imports are not final native/source qualification. CORE owns all other
shared paths and needs the prepared native and finite model interfaces now.
