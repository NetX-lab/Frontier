# D29 registry binding plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Bound the four-path repair to real binding regression and exact budgets. |

Dependency: caller inspection -> persistent actual RED -> retention/selection and
physical-row enumeration -> GREEN and affected legacy/shared/independent checks
-> exact diff budgets -> commit and root handoff. No delegated parallel work.

1. Inspect all contract registry/view/bundle consumers and real markers.
2. Add tight actual Step dual-group RED before registry changes.
3. Extend existing registry keys without changing typed/cache identities; select
   explicit native bindings and retain variants in family/bundle enumeration.
4. Forward context/source/group selection and enumerate all physical-row receipts.
5. Verify acceptance with selected tests, inspect artifacts, check quotas and commit.

Root owns independent review, full actual-main/strict77 and GPU/native qualification.
