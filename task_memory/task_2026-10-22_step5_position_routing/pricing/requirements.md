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
